"""Skate UDP wire protocol — pure Python, no ROS dependencies.

Wire contract (reverse-engineered from the official Rbotic/skate_teleop client
and confirmed by the official Skate docs):

* Transport: UDP, robot listens on port 2000 (``r.local`` via mDNS).
* Telemetry  robot -> client: ``pickle.dumps((id, obj))`` where id is
  0 motor_commands, 1 motor_states, 2 state_estimates, 3 INS_state_estimates,
  4 controller_states (classes in :mod:`skate_ros2.shared_classes_def`).
* Command  client -> robot: ``pickle.dumps((5, (targ_pos[26], vel_cmd[3],
  height_cmd, (estop_WB, estop_LA, estop_RA))))``; flags 0 = dampen.
* Extension, SIM ONLY: id 6 carries a wrist force/torque wrench (see
  :data:`WRENCH_ID`). The real Skate has no wrist F/T sensor and never sends
  it, so a client must treat its absence as normal, not as a fault.
* Heartbeat: the robot streams only to the address it last heard from; the
  official client pings ``b"yo"`` every 0.3 s. If the robot hears nothing for
  0.3 s it assumes deadman ``(0, 0, 0)`` and dampens — that watchdog lives in
  the firmware, not here.

SECURITY NOTE: the wire format is Python pickle, which can normally execute
arbitrary code when loading. The firmware's choice of pickle is fixed, but the
decoder here defends against it -- :func:`decode_packet` uses a *restricted*
unpickler by default (only the known telemetry classes + numpy are resolvable),
so a hostile packet can't run code. Set ``SKATE_WIRE=raw`` to opt out. Even so,
prefer a trusted local network (the same assumption the official stack makes).

AUTHENTICATION: the firmware has none and cannot be given one, so the wire is
open by default and stays byte-compatible with a real Skate. Where BOTH ends
are ours -- a client and :mod:`skate_ros2.sim_endpoint` -- setting
``$SKATE_AUTH`` to the same shared secret on both wraps every datagram in a
keyed envelope (:class:`WireAuth`) that a forged, stale or replayed packet
cannot produce. Against real firmware leave it unset: the robot has no key and
would drop the envelope as garbage.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import pickle
import socket
import struct
import sys
import time

import numpy as np

from . import names
from . import shared_classes_def as SCD

# The firmware pickles its telemetry classes under the top-level module name
# 'shared_classes_def'. Register our vendored copy so packets unpickle.
sys.modules.setdefault("shared_classes_def", SCD)

DEFAULT_PORT = 2000
DEFAULT_HOST = "r.local"
BUFFER_SIZE = 4096 * 10
HEARTBEAT = b"yo"
HEARTBEAT_PERIOD = 0.3   # s, official client value
STALE_AFTER = 0.3        # s, telemetry older than this counts as disconnected
COMMAND_ID = 5

# Wrist force/torque wrench, SIM ONLY -- the real Skate has no wrist F/T sensor
# and never sends this id, so a client must treat its absence as normal.
#
# The payload is deliberately a plain nested dict of Python floats:
#
#     {"t": float,                                   # sender's monotonic stamp
#      "left":  {"f": [fx, fy, fz], "m": [mx, my, mz]},
#      "right": {...}}                               # world frame, N and N*m
#
# dicts, strings, floats and lists are pickle PRIMITIVES -- decoding one never
# calls find_class, so this id resolves zero globals and needs no _SAFE_GLOBALS
# entry. Inventing a shared_classes_def class for it would have widened the
# allow-list and put a message on the wire the firmware does not have; a dict
# costs neither.
WRENCH_ID = 6

TELEMETRY_IDS = {
    0: "motor_commands",
    1: "motor_states",
    2: "state_estimates",
    3: "ins",
    4: "controller_states",
    # 5 is COMMAND_ID (client -> robot), so the extension starts at 6.
    WRENCH_ID: "wrist_wrench",
}


def pack_command(targ_pos, vel_cmd=(0.0, 0.0, 0.0), height_cmd=1.0,
                 deadman=(0, 0, 0)):
    """Serialize one command packet exactly as the official client does."""
    targ = np.asarray(targ_pos, dtype=np.float64)
    if targ.shape != (names.N_JOINTS,):
        raise ValueError(
            f"targ_pos must have shape ({names.N_JOINTS},), got {targ.shape}")
    vel = np.asarray(vel_cmd, dtype=np.float64)
    if vel.shape != (3,):
        raise ValueError(f"vel_cmd must have shape (3,), got {vel.shape}")
    dm = (int(deadman[0]), int(deadman[1]), int(deadman[2]))
    payload = (targ, vel, float(height_cmd), dm)
    data = pickle.dumps((COMMAND_ID, payload))
    if len(data) > BUFFER_SIZE:
        raise ValueError("command packet exceeds UDP buffer size")
    return data


# Exact (module, name) pairs a legitimate packet may reconstruct. The firmware
# pickles its telemetry classes (under the bare module name 'shared_classes_def';
# our code may use the package-qualified name) plus numpy arrays; nothing else
# should ever appear on the wire.
#
# This is an EXACT allow-list, NOT a numpy.* prefix match: numpy.f2py and
# numpy.distutils expose command-execution helpers (e.g. f2py.diagnose.run_command,
# distutils.exec_command) that a ``startswith("numpy.")`` rule would have let
# through — a remotely reachable RCE. Only the array-reconstruction entry points
# (both the numpy 1.x ``core`` and 2.x ``_core`` namespaces) are permitted.
_SCD_CLASSES = ("motor_command", "motor_state", "state_est",
                "INS_fusion_state", "FeedbackResp")
_SAFE_GLOBALS = frozenset(
    [(mod, cls)
     for mod in ("shared_classes_def", "skate_ros2.shared_classes_def")
     for cls in _SCD_CLASSES]
    + [("numpy", "ndarray"), ("numpy", "dtype"),
       ("numpy.core.multiarray", "_reconstruct"),
       ("numpy.core.multiarray", "scalar"),
       ("numpy._core.multiarray", "_reconstruct"),
       ("numpy._core.multiarray", "scalar"),
       ("copyreg", "_reconstructor"), ("copyreg", "__newobj__")]
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves ONLY the known telemetry classes + numpy array
    reconstruction (exact allow-list). Every other ``find_class`` is refused, so
    a crafted packet cannot reach ``os.system`` / ``eval`` / ``numpy.f2py`` / an
    arbitrary ``__reduce__`` gadget.
    """

    def find_class(self, module, name):
        if (module, name) in _SAFE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"blocked unpickling of {module}.{name} (untrusted wire packet)")


def decode_packet(data):
    """Decode one telemetry/command packet -> (id, obj).

    Uses a restricted unpickler by default (``SKATE_WIRE`` unset or ``safe``):
    it accepts every legitimate firmware/sim packet but refuses arbitrary
    globals, so a hostile packet can't execute code. Set ``SKATE_WIRE=raw`` to
    fall back to plain ``pickle.loads`` (only on a fully trusted link, e.g. to
    decode an unforeseen class).
    """
    if os.environ.get("SKATE_WIRE", "safe").lower() == "raw":
        return pickle.loads(data)
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def unpack_packet(data):
    """Decode one telemetry/command packet -> (id, obj). Trusted LAN only.

    Back-compat alias for :func:`decode_packet` (restricted unpickler by default).
    """
    return decode_packet(data)


# -- large-datagram fragmentation ------------------------------------------
# Some virtual NICs (notably WSL2 loopback) silently DROP any UDP datagram whose
# IP packet exceeds ~1500 B, and their fragmentation/reassembly is unreliable at
# any interface MTU. A few sim telemetry objects (e.g. state_est) pickle to
# ~2 kB, so we split oversized payloads into <= FRAG_MAX_DGRAM chunks with a tiny
# reassembly header and stitch them back on the far side. Anything that already
# fits (every real-firmware packet and every command) goes out unchanged, so the
# wire stays byte-compatible with the existing stack.
FRAG_MAGIC = b"SKF1"
FRAG_HEADER = struct.Struct("!IHH")                     # msg_id, n_chunks, idx
FRAG_HEADER_LEN = len(FRAG_MAGIC) + FRAG_HEADER.size    # 4 + 8 = 12
FRAG_MAX_DGRAM = 1400                                   # keep IP packet < ~1500 B
FRAG_CHUNK = FRAG_MAX_DGRAM - FRAG_HEADER_LEN           # payload bytes / fragment
FRAG_MAX_CHUNKS = 64                                    # memory-DoS guard
FRAG_TTL = 1.0                                          # s; drop stale partials


def pack_datagrams(pkt_id, obj, msg_id):
    """Serialize ``(pkt_id, obj)`` into a list of UDP-sized datagrams.

    Returns a single unmodified ``pickle.dumps`` datagram when it already fits in
    ``FRAG_MAX_DGRAM`` (the common case). Otherwise returns ``FRAG_MAGIC``-tagged
    fragments that :class:`Reassembler` reunites. ``msg_id`` only needs to differ
    between large messages concurrently in flight from one sender (a rolling
    counter is fine); it is ignored for packets that fit in one datagram.
    """
    blob = pickle.dumps((pkt_id, obj))
    if len(blob) <= FRAG_MAX_DGRAM:
        return [blob]
    chunks = [blob[i:i + FRAG_CHUNK] for i in range(0, len(blob), FRAG_CHUNK)]
    n = len(chunks)
    return [FRAG_MAGIC + FRAG_HEADER.pack(msg_id & 0xFFFFFFFF, n, i) + c
            for i, c in enumerate(chunks)]


class Reassembler:
    """Reunites :func:`pack_datagrams` fragments into whole pickle blobs.

    Feed every inbound datagram to :meth:`feed`; it returns the reassembled blob
    when a fragmented message completes, the datagram unchanged when it is an
    ordinary (unfragmented) packet, or ``None`` while a fragmented message is
    still incomplete or malformed. Memory is bounded: chunk counts are capped and
    half-assembled messages older than ``FRAG_TTL`` are evicted.
    """

    def __init__(self):
        self._parts = {}   # msg_id -> {"n", "chunks": {idx: bytes}, "t"}

    def feed(self, data, now=None):
        if not data.startswith(FRAG_MAGIC):
            return data                          # ordinary datagram — pass through
        now = now if now is not None else time.monotonic()
        try:
            msg_id, n, idx = FRAG_HEADER.unpack_from(data, len(FRAG_MAGIC))
        except struct.error:
            return None
        if n == 0 or n > FRAG_MAX_CHUNKS or idx >= n:
            return None
        slot = self._parts.get(msg_id)
        if slot is None or slot["n"] != n:
            slot = {"n": n, "chunks": {}, "t": now}
            self._parts[msg_id] = slot
        slot["chunks"][idx] = data[FRAG_HEADER_LEN:]
        slot["t"] = now
        self._evict(now)
        if len(slot["chunks"]) == n:
            del self._parts[msg_id]
            return b"".join(slot["chunks"][i] for i in range(n))
        return None

    def _evict(self, now):
        for k in [k for k, v in self._parts.items() if now - v["t"] > FRAG_TTL]:
            del self._parts[k]


# -- transport authentication ----------------------------------------------
# The firmware has no authentication and cannot be given one, so this is OFF by
# default and the wire stays byte-compatible with a real Skate. Where BOTH ends
# are ours -- a client and sim_endpoint -- exporting the same $SKATE_AUTH on
# both wraps every datagram in a keyed envelope:
#
#     b"SKA1" | nonce (8 B, big-endian) | HMAC-SHA256(key, nonce|body)[:16] | body
#
# The envelope goes OUTSIDE fragmentation, so every datagram verifies on its own
# and a forged fragment is dropped before it can occupy a reassembly slot.
#
# The nonce is a wall-clock microsecond stamp in the top 52 bits (good to ~2112)
# plus a 12-bit rolling per-sender sequence in the low bits. The stamp bounds how
# long an intercepted-and-withheld datagram stays injectable AND bounds replay
# memory to one window of traffic; it is WALL clock, not monotonic, because the
# two ends are different processes whose monotonic clocks are not comparable. The
# sequence makes the nonce unique inside a single microsecond -- a burst of
# fragments really does hit that, and without it the second fragment of a large
# telemetry packet would be refused as a replay of the first.
AUTH_ENV = "SKATE_AUTH"
AUTH_MAGIC = b"SKA1"
AUTH_HEADER = struct.Struct("!Q")                       # nonce
AUTH_TAG_LEN = 16                                       # truncated HMAC-SHA256
AUTH_OVERHEAD = len(AUTH_MAGIC) + AUTH_HEADER.size + AUTH_TAG_LEN   # 28
AUTH_SEQ_BITS = 12
AUTH_SEQ_MASK = (1 << AUTH_SEQ_BITS) - 1
AUTH_WINDOW = 5.0                                       # s of accepted clock skew
AUTH_WINDOW_US = int(AUTH_WINDOW * 1e6)
AUTH_MAX_SEEN = 8192                                    # nonces/peer, DoS guard
AUTH_MIN_KEY = 16                                       # bytes

# 1500 B Ethernet MTU - 20 B IPv4 - 8 B UDP. A datagram larger than this is what
# WSL2 loopback silently drops, i.e. the bug fragmentation exists to avoid; the
# envelope must not push a full fragment back over it. See test_wire_auth.py,
# which pins FRAG_MAX_DGRAM + AUTH_OVERHEAD <= UDP_SAFE_DGRAM.
UDP_SAFE_DGRAM = 1472


def auth_key(explicit=None):
    """Resolve the shared secret: the argument, else ``$SKATE_AUTH``, else None.

    None means "no authentication" -- the default, and the only setting that can
    talk to real firmware.
    """
    if explicit is None:
        explicit = os.environ.get(AUTH_ENV, "")
    if not explicit:
        return None
    return explicit if isinstance(explicit, bytes) else str(explicit).encode()


def is_tagged(datagram):
    """True if the datagram carries an auth envelope."""
    return datagram.startswith(AUTH_MAGIC)


class WireAuth:
    """Wraps and verifies the keyed envelope. One object serves both ends.

    :meth:`wrap` stamps and signs an outbound datagram; :meth:`unwrap` returns
    the body of an inbound one, or ``None`` for anything forged, untagged, stale
    or replayed (:attr:`rejected` counts why). Verification is constant-time and
    happens BEFORE the freshness bookkeeping, so an attacker without the key can
    never grow the replay set.
    """

    def __init__(self, key):
        key = key if isinstance(key, bytes) else str(key).encode()
        if len(key) < AUTH_MIN_KEY:
            raise ValueError(
                f"{AUTH_ENV} must be at least {AUTH_MIN_KEY} bytes; this is raw "
                "HMAC key material with no KDF, so a short secret is "
                "brute-forceable from a single captured datagram")
        self._key = key
        self._seq = 0
        self._seen = {}          # peer -> set of accepted nonces inside the window
        self.rejected = {"untagged": 0, "forged": 0, "stale": 0,
                         "replay": 0, "flood": 0}

    @property
    def n_rejected(self):
        return sum(self.rejected.values())

    def _tag(self, head, body):
        return hmac.new(self._key, head + body, hashlib.sha256).digest()[:AUTH_TAG_LEN]

    def wrap(self, datagram, now=None):
        """Sign one outbound datagram -> the envelope bytes."""
        now = time.time() if now is None else now
        self._seq = (self._seq + 1) & AUTH_SEQ_MASK
        nonce = (int(now * 1e6) << AUTH_SEQ_BITS) | self._seq
        head = AUTH_HEADER.pack(nonce)
        return AUTH_MAGIC + head + self._tag(head, datagram) + datagram

    def unwrap(self, datagram, peer=None, now=None):
        """Verify one inbound datagram -> its body, or None if it is not ours."""
        if len(datagram) < AUTH_OVERHEAD or not datagram.startswith(AUTH_MAGIC):
            self.rejected["untagged"] += 1
            return None
        head = datagram[len(AUTH_MAGIC):len(AUTH_MAGIC) + AUTH_HEADER.size]
        tag = datagram[len(AUTH_MAGIC) + AUTH_HEADER.size:AUTH_OVERHEAD]
        body = datagram[AUTH_OVERHEAD:]
        if not hmac.compare_digest(tag, self._tag(head, body)):
            self.rejected["forged"] += 1
            return None
        (nonce,) = AUTH_HEADER.unpack(head)
        now_us = int((time.time() if now is None else now) * 1e6)
        if abs(now_us - (nonce >> AUTH_SEQ_BITS)) > AUTH_WINDOW_US:
            self.rejected["stale"] += 1
            return None
        seen = self._seen.setdefault(peer, set())
        if len(seen) >= AUTH_MAX_SEEN:
            self._prune(now_us)
            seen = self._seen.setdefault(peer, set())
            if len(seen) >= AUTH_MAX_SEEN:
                self.rejected["flood"] += 1
                return None
        if nonce in seen:
            self.rejected["replay"] += 1
            return None
        seen.add(nonce)
        return body

    def _prune(self, now_us):
        floor = now_us - AUTH_WINDOW_US
        for p in list(self._seen):
            fresh = {n for n in self._seen[p] if (n >> AUTH_SEQ_BITS) >= floor}
            if fresh:
                self._seen[p] = fresh
            else:
                del self._seen[p]


class TelemetryState:
    """Latest decoded telemetry plus receive timestamps."""

    def __init__(self):
        self.motor_commands = None   # SCD.motor_command
        self.motor_states = None     # SCD.motor_state
        self.state_estimates = None  # SCD.state_est
        self.ins = None              # SCD.INS_fusion_state
        self.controller_states = None
        self.wrist_wrench = None     # plain dict, sim only (see WRENCH_ID)
        self.stamps = {}             # field name -> time.monotonic()
        self.n_packets = 0

    def update(self, pkt_id, obj, now=None):
        field = TELEMETRY_IDS.get(pkt_id)
        if field is None:
            return False
        setattr(self, field, obj)
        self.stamps[field] = now if now is not None else time.monotonic()
        self.n_packets += 1
        return True

    def age(self, now=None):
        """Seconds since the newest telemetry packet (inf if none yet)."""
        if not self.stamps:
            return float("inf")
        now = now if now is not None else time.monotonic()
        return now - max(self.stamps.values())

    @property
    def connected(self):
        return self.age() < STALE_AFTER

    def dof_pos(self):
        """Calibrated joint positions as a flat 26-list (None if not seen)."""
        if self.state_estimates is None:
            return None
        return names.can_dict_to_vector(self.state_estimates.dof_pos)

    def dof_vel(self):
        if self.state_estimates is None:
            return None
        return names.can_dict_to_vector(self.state_estimates.dof_vel)

    def dof_torque(self):
        if self.state_estimates is None:
            return None
        return names.can_dict_to_vector(self.state_estimates.dof_torque)

    def motor_pos(self):
        """Raw motor positions as a flat 26-list (None if not seen).

        A fallback pose source when the calibrated ``state_est`` stream has not
        arrived yet; on the sim the two are identical.
        """
        if self.motor_states is None:
            return None
        return names.can_dict_to_vector(self.motor_states.motor_pos)

    def motor_temps(self):
        if self.motor_states is None:
            return None
        return names.can_dict_to_vector(self.motor_states.motor_temp)

    def wrenches(self):
        """Measured wrist wrenches as ``{arm: {"f": [3], "m": [3]}}``, or None.

        Sim only -- a real Skate never sends id 6, so None here means "this
        robot has no wrist F/T sensor", not "the link is broken"; callers fall
        back to a joint-torque estimate.

        The payload is decoded from the wire, so nothing about its shape is
        trusted: an arm whose entry is not two finite 3-vectors is dropped
        rather than passed up as a half-valid wrench. A malformed packet
        therefore degrades to the estimator instead of poisoning a force
        display -- or a compliance loop -- with junk.
        """
        w = self.wrist_wrench
        if not isinstance(w, dict):
            return None
        out = {}
        for arm, d in w.items():
            if arm == "t" or not isinstance(d, dict):
                continue
            try:
                f = [float(x) for x in d["f"]]
                m = [float(x) for x in d["m"]]
            except (KeyError, TypeError, ValueError):
                continue
            if len(f) != 3 or len(m) != 3:
                continue
            if not all(np.isfinite(v) for v in f + m):
                continue
            out[str(arm)] = {"f": f, "m": m}
        return out or None


class SkateLink:
    """UDP client to a Skate robot (or :mod:`skate_ros2.sim_endpoint`).

    Non-blocking; call :meth:`poll` often (e.g. from a 60 Hz timer). Heartbeats
    are sent automatically from :meth:`poll`, so the robot keeps streaming even
    when no commands are being sent.

    ``key`` (or ``$SKATE_AUTH``) turns on the keyed envelope -- only for a far
    end that shares the secret, i.e. :mod:`skate_ros2.sim_endpoint`. Real
    firmware has no key and would drop the envelope as garbage, so leave it unset
    when talking to a robot.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, key=None):
        self.host_name = host
        self.port = port
        self.addr = None          # resolved (ip, port)
        self.state = TelemetryState()
        self._last_heartbeat = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self.decode_errors = 0
        self.auth_errors = 0
        self._reasm = Reassembler()
        k = auth_key(key)
        self.auth = WireAuth(k) if k else None

    # -- connection -------------------------------------------------------
    def resolve(self):
        """Resolve the robot hostname. Returns True on success."""
        try:
            ip = socket.gethostbyname(self.host_name)
        except socket.gaierror:
            self.addr = None
            return False
        self.addr = (ip, self.port)
        return True

    @property
    def connected(self):
        return self.state.connected

    # -- io ----------------------------------------------------------------
    def _out(self, datagram):
        """Send one datagram to the robot, signed if a key is configured."""
        if self.auth is not None:
            datagram = self.auth.wrap(datagram)
        try:
            self._sock.sendto(datagram, self.addr)
        except OSError:
            return False
        return True

    def heartbeat(self, now=None):
        if self.addr is None and not self.resolve():
            return False
        now = now if now is not None else time.monotonic()
        if not self._out(HEARTBEAT):
            return False
        self._last_heartbeat = now
        return True

    def poll(self):
        """Drain all pending telemetry; auto-heartbeat. Returns packet count."""
        now = time.monotonic()
        if now - self._last_heartbeat > HEARTBEAT_PERIOD:
            self.heartbeat(now)
        n = 0
        while True:
            try:
                data, addr = self._sock.recvfrom(BUFFER_SIZE)
            except BlockingIOError:
                break
            except OSError:
                break
            if self.auth is not None:
                data = self.auth.unwrap(data, addr)
                if data is None:
                    self.auth_errors += 1
                    continue
            elif is_tagged(data):
                # The far end authenticates and we hold no key -- a specific,
                # fixable condition, not the generic "that didn't decode".
                self.auth_errors += 1
                continue
            blob = self._reasm.feed(data)
            if blob is None:
                continue                     # partial fragment — wait for the rest
            try:
                pkt_id, obj = unpack_packet(blob)
            except Exception:
                self.decode_errors += 1
                continue
            if self.state.update(pkt_id, obj, now=time.monotonic()):
                n += 1
        return n

    def send_command(self, targ_pos, vel_cmd=(0.0, 0.0, 0.0), height_cmd=1.0,
                     deadman=(0, 0, 0)):
        """Send one command packet. Returns True if it left the socket."""
        if self.addr is None and not self.resolve():
            return False
        data = pack_command(targ_pos, vel_cmd, height_cmd, deadman)
        if not self._out(data):
            return False
        self._last_heartbeat = time.monotonic()  # a command is also a heartbeat
        return True

    def close(self):
        self._sock.close()
