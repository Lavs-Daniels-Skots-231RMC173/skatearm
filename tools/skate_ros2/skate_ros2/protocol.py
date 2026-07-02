"""Skate UDP wire protocol — pure Python, no ROS dependencies.

Wire contract (reverse-engineered from the official Rbotic/skate_teleop client
and confirmed by the official Skate docs):

* Transport: UDP, robot listens on port 2000 (``r.local`` via mDNS).
* Telemetry  robot -> client: ``pickle.dumps((id, obj))`` where id is
  0 motor_commands, 1 motor_states, 2 state_estimates, 3 INS_state_estimates,
  4 controller_states (classes in :mod:`skate_ros2.shared_classes_def`).
* Command  client -> robot: ``pickle.dumps((5, (targ_pos[26], vel_cmd[3],
  height_cmd, (estop_WB, estop_LA, estop_RA))))``; flags 0 = dampen.
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
"""

from __future__ import annotations

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

TELEMETRY_IDS = {
    0: "motor_commands",
    1: "motor_states",
    2: "state_estimates",
    3: "ins",
    4: "controller_states",
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


class TelemetryState:
    """Latest decoded telemetry plus receive timestamps."""

    def __init__(self):
        self.motor_commands = None   # SCD.motor_command
        self.motor_states = None     # SCD.motor_state
        self.state_estimates = None  # SCD.state_est
        self.ins = None              # SCD.INS_fusion_state
        self.controller_states = None
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


class SkateLink:
    """UDP client to a Skate robot (or :mod:`skate_ros2.sim_endpoint`).

    Non-blocking; call :meth:`poll` often (e.g. from a 60 Hz timer). Heartbeats
    are sent automatically from :meth:`poll`, so the robot keeps streaming even
    when no commands are being sent.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.host_name = host
        self.port = port
        self.addr = None          # resolved (ip, port)
        self.state = TelemetryState()
        self._last_heartbeat = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self.decode_errors = 0
        self._reasm = Reassembler()

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
    def heartbeat(self, now=None):
        if self.addr is None and not self.resolve():
            return False
        now = now if now is not None else time.monotonic()
        try:
            self._sock.sendto(HEARTBEAT, self.addr)
        except OSError:
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
                data, _addr = self._sock.recvfrom(BUFFER_SIZE)
            except BlockingIOError:
                break
            except OSError:
                break
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
        try:
            self._sock.sendto(data, self.addr)
        except OSError:
            return False
        self._last_heartbeat = time.monotonic()  # a command is also a heartbeat
        return True

    def close(self):
        self._sock.close()
