"""Transport authentication: where both ends are ours, a shared secret makes
the wire unforgeable -- and the sim endpoint refuses to serve motion commands
off-box without one.

The firmware has no authentication and cannot be given one, so this is OFF by
default; these tests pin both halves of that bargain -- the envelope really does
reject forged / stale / replayed / unsigned datagrams, and an unset key really
does leave the wire byte-identical to what a real Skate speaks.

Hardware-free: needs only numpy + the protocol module (no MuJoCo, no ROS).

    python -m pytest -q tools/skate_ros2/test/test_wire_auth.py
"""
import os
import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skate_ros2 import protocol                       # noqa: E402
from skate_ros2 import shared_classes_def as SCD       # noqa: E402
from skate_ros2 import sim_endpoint                    # noqa: E402

KEY = b"correct-horse-battery-staple"
OTHER_KEY = b"almost-the-right-secret-but-no"
PEER = ("127.0.0.1", 55555)
BODY = b"a command would live here"


def _no_env_key():
    """Drop $SKATE_AUTH for the duration of a test -> restore it afterwards.

    A developer with the variable exported in their shell must not silently turn
    the "refuses to start" tests green.
    """
    return pytest.MonkeyPatch()


def test_envelope_roundtrips_and_costs_what_the_constants_say():
    sender, receiver = protocol.WireAuth(KEY), protocol.WireAuth(KEY)
    dg = sender.wrap(BODY)
    assert protocol.is_tagged(dg) and dg.startswith(protocol.AUTH_MAGIC)
    assert len(dg) - len(BODY) == protocol.AUTH_OVERHEAD
    assert receiver.unwrap(dg, peer=PEER) == BODY
    assert receiver.n_rejected == 0
    # An empty body is still a well-formed envelope (the heartbeat is 2 bytes,
    # so the length floor must not be a body-length assumption).
    assert receiver.unwrap(sender.wrap(b""), peer=PEER) == b""


def test_untagged_datagram_is_refused():
    """The point of the exercise: a raw firmware-format command from someone
    without the key never reaches the decoder."""
    receiver = protocol.WireAuth(KEY)
    raw = protocol.pack_command(np.zeros(26), deadman=(1, 1, 1))
    assert receiver.unwrap(raw, peer=PEER) is None
    assert receiver.rejected["untagged"] == 1
    # Too short to hold an envelope at all -- must not IndexError its way in.
    assert receiver.unwrap(b"yo", peer=PEER) is None
    assert receiver.unwrap(protocol.AUTH_MAGIC, peer=PEER) is None
    assert receiver.rejected["untagged"] == 3


def test_forged_tag_refused_and_cannot_grow_the_replay_set():
    sender, receiver = protocol.WireAuth(KEY), protocol.WireAuth(KEY)
    dg = sender.wrap(BODY)
    # Red-check FIRST: the untouched datagram is accepted, so every rejection
    # below is caused by the tampering and not by the harness.
    assert receiver.unwrap(dg, peer=PEER) == BODY
    seen_after_one = sum(len(s) for s in receiver._seen.values())
    assert seen_after_one == 1

    tampered = bytearray(dg)
    tampered[-1] ^= 0x01                         # one bit of the body
    assert receiver.unwrap(bytes(tampered), peer=PEER) is None
    wrong = protocol.WireAuth(OTHER_KEY).wrap(BODY)
    assert receiver.unwrap(wrong, peer=PEER) is None
    for i in range(50):                          # a flood of forgeries
        assert receiver.unwrap(protocol.WireAuth(OTHER_KEY).wrap(BODY),
                               peer=("10.0.0.%d" % i, 9000)) is None
    assert receiver.rejected["forged"] == 52
    # Verification precedes the freshness bookkeeping, so an attacker without
    # the key can never make the endpoint remember anything -- no memory DoS.
    assert sum(len(s) for s in receiver._seen.values()) == seen_after_one


def test_replayed_datagram_refused():
    sender, receiver = protocol.WireAuth(KEY), protocol.WireAuth(KEY)
    dg = sender.wrap(BODY)
    assert receiver.unwrap(dg, peer=PEER) == BODY
    assert receiver.unwrap(dg, peer=PEER) is None      # captured and re-sent
    assert receiver.rejected["replay"] == 1
    # A fresh datagram with the same body still gets through: it is the nonce
    # that is spent, not the message.
    assert receiver.unwrap(sender.wrap(BODY), peer=PEER) == BODY


def test_stale_datagram_refused_in_both_directions():
    sender, receiver = protocol.WireAuth(KEY), protocol.WireAuth(KEY)
    now = time.time()
    old = sender.wrap(BODY, now=now - 2 * protocol.AUTH_WINDOW)
    future = sender.wrap(BODY, now=now + 2 * protocol.AUTH_WINDOW)
    assert receiver.unwrap(old, peer=PEER, now=now) is None
    assert receiver.unwrap(future, peer=PEER, now=now) is None
    assert receiver.rejected["stale"] == 2
    # Red-check: inside the window the very same construction is accepted, so
    # the rejection is the age and not a broken tag.
    inside = sender.wrap(BODY, now=now - protocol.AUTH_WINDOW / 2)
    assert receiver.unwrap(inside, peer=PEER, now=now) == BODY


def test_bursts_within_one_microsecond_are_not_mistaken_for_replays():
    """A fragmented telemetry packet leaves in one burst -- often inside a single
    microsecond. Without the sequence counter in the nonce the second fragment
    would be refused as a replay of the first, which would break large packets
    exactly on the link fragmentation exists to serve."""
    sender, receiver = protocol.WireAuth(KEY), protocol.WireAuth(KEY)
    frozen = time.time()
    burst = [sender.wrap(b"chunk-%d" % i, now=frozen) for i in range(8)]
    got = [receiver.unwrap(dg, peer=PEER, now=frozen) for dg in burst]
    assert got == [b"chunk-%d" % i for i in range(8)]
    assert receiver.n_rejected == 0


def test_replay_memory_is_per_peer():
    sender, receiver = protocol.WireAuth(KEY), protocol.WireAuth(KEY)
    dg = sender.wrap(BODY)
    assert receiver.unwrap(dg, peer=("10.0.0.1", 1)) == BODY
    # Same bytes from a different address: still a replay from that address's
    # point of view? No -- it has not been seen there. Accepting it is correct;
    # the source address is not the thing being authenticated, the key is.
    assert receiver.unwrap(dg, peer=("10.0.0.2", 2)) == BODY
    assert receiver.unwrap(dg, peer=("10.0.0.2", 2)) is None
    assert receiver.rejected["replay"] == 1


def test_short_key_refused():
    """The key is raw HMAC material with no KDF, so a short secret is
    brute-forceable from one captured datagram -- refuse it at construction
    rather than pretend the link is authenticated."""
    with pytest.raises(ValueError, match=protocol.AUTH_ENV):
        protocol.WireAuth(b"hunter2")
    with pytest.raises(ValueError):
        protocol.WireAuth("x" * (protocol.AUTH_MIN_KEY - 1))
    assert protocol.WireAuth("x" * protocol.AUTH_MIN_KEY) is not None


def test_auth_key_resolution():
    mp = _no_env_key()
    try:
        mp.delenv(protocol.AUTH_ENV, raising=False)
        assert protocol.auth_key() is None                  # the default: OFF
        assert protocol.auth_key("") is None
        assert protocol.auth_key(KEY) == KEY
        assert protocol.auth_key("text-secret") == b"text-secret"
        mp.setenv(protocol.AUTH_ENV, "from-the-environment")
        assert protocol.auth_key() == b"from-the-environment"
        assert protocol.auth_key(KEY) == KEY                # explicit wins
    finally:
        mp.undo()


def test_envelope_still_fits_a_safe_datagram():
    """Auth must not silently reintroduce the WSL2 MTU bug.

    Fragmentation exists because some virtual NICs drop UDP datagrams above
    ~1500 B. The envelope adds bytes to every datagram AFTER fragmentation, so
    if the overhead ever pushed a full fragment past the safe size, large
    telemetry would start vanishing again on exactly the link this was written
    for. Constants are imported, never retyped -- a future edit to either one
    has to face this assertion.
    """
    assert protocol.FRAG_MAX_DGRAM + protocol.AUTH_OVERHEAD \
        <= protocol.UDP_SAFE_DGRAM

    # ...and empirically, on a real oversized telemetry object.
    big = SCD.state_est()
    big.dof_pos = {i: float(i) for i in range(26)}
    big.dof_vel = {i: float(i) for i in range(26)}
    big.dof_torque = {i: float(i) for i in range(26)}
    big.pad = np.arange(600, dtype=np.float64)      # force multiple fragments
    dgs = protocol.pack_datagrams(2, big, 7)
    assert len(dgs) > 1, "test object is too small to exercise fragmentation"
    auth = protocol.WireAuth(KEY)
    assert max(len(auth.wrap(dg)) for dg in dgs) <= protocol.UDP_SAFE_DGRAM


def test_auth_composes_with_fragmentation_end_to_end():
    """Sign each fragment separately -> reassemble -> decode. A forged fragment
    is dropped before it can occupy a reassembly slot."""
    obj = SCD.state_est()
    obj.dof_pos = {i: float(i) for i in range(26)}
    obj.pad = np.arange(600, dtype=np.float64)
    dgs = protocol.pack_datagrams(2, obj, 11)
    assert len(dgs) > 1

    sender, receiver = protocol.WireAuth(KEY), protocol.WireAuth(KEY)
    reasm = protocol.Reassembler()
    blob = None
    for dg in dgs:
        wire = sender.wrap(dg)
        # An attacker's fragment, injected into the middle of the sequence:
        assert receiver.unwrap(protocol.WireAuth(OTHER_KEY).wrap(dg),
                               peer=PEER) is None
        body = receiver.unwrap(wire, peer=PEER)
        assert body is not None
        out = reasm.feed(body)
        if out is not None:
            blob = out
    assert blob is not None
    pkt_id, got = protocol.decode_packet(blob)
    assert pkt_id == 2 and got.dof_pos[25] == 25.0
    assert list(got.pad) == list(obj.pad)


def _far_end():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(1.0)
    return s


def test_link_without_a_key_stays_byte_compatible_with_firmware():
    """No key -> the datagrams a real Skate would see, unchanged."""
    far = _far_end()
    link = protocol.SkateLink("127.0.0.1", far.getsockname()[1])
    try:
        assert link.auth is None
        assert link.heartbeat()
        hb, _addr = far.recvfrom(protocol.BUFFER_SIZE)
        assert hb == protocol.HEARTBEAT              # not wrapped, not tagged
        assert link.send_command(np.zeros(26), deadman=(1, 1, 1))
        cmd, _addr = far.recvfrom(protocol.BUFFER_SIZE)
        assert not protocol.is_tagged(cmd)
        assert protocol.decode_packet(cmd)[0] == protocol.COMMAND_ID
    finally:
        link.close()
        far.close()


def test_link_with_a_key_signs_everything_it_sends():
    far = _far_end()
    link = protocol.SkateLink("127.0.0.1", far.getsockname()[1], key=KEY)
    try:
        assert link.auth is not None
        assert link.heartbeat()
        assert link.send_command(np.zeros(26), deadman=(1, 1, 1))
        peer = protocol.WireAuth(KEY)
        seen = []
        for _ in range(2):
            dg, addr = far.recvfrom(protocol.BUFFER_SIZE)
            assert protocol.is_tagged(dg)
            body = peer.unwrap(dg, peer=addr)
            assert body is not None
            seen.append(body)
        assert seen[0] == protocol.HEARTBEAT
        assert protocol.decode_packet(seen[1])[0] == protocol.COMMAND_ID
    finally:
        link.close()
        far.close()


def test_link_without_a_key_names_the_problem():
    """A tagged datagram arriving at a keyless client is a specific, fixable
    condition ("the far end authenticates, you don't"), so it must not hide in
    the generic decode_errors counter."""
    far = _far_end()
    link = protocol.SkateLink("127.0.0.1", far.getsockname()[1])
    try:
        assert link.heartbeat()
        _hb, client_addr = far.recvfrom(protocol.BUFFER_SIZE)
        far.sendto(protocol.WireAuth(KEY).wrap(pickle.dumps((2, SCD.state_est()))),
                   client_addr)
        time.sleep(0.05)
        link.poll()
        assert link.auth_errors == 1
        assert link.decode_errors == 0
        assert link.state.n_packets == 0
        # Red-check: the same telemetry unsigned does land, so the drop above is
        # the envelope and not a broken loopback.
        far.sendto(pickle.dumps((2, SCD.state_est())), client_addr)
        time.sleep(0.05)
        link.poll()
        assert link.state.n_packets == 1
    finally:
        link.close()
        far.close()


def test_require_auth_truth_table():
    mp = _no_env_key()
    try:
        mp.delenv(protocol.AUTH_ENV, raising=False)
        for bind in ("127.0.0.1", "127.0.1.5", "::1", "localhost"):
            assert sim_endpoint.is_loopback(bind)
            assert sim_endpoint.require_auth(bind) is None    # key not needed
        for bind in ("0.0.0.0", "192.168.1.10", "::", "", "r.local"):
            assert not sim_endpoint.is_loopback(bind)
            with pytest.raises(RuntimeError, match=protocol.AUTH_ENV):
                sim_endpoint.require_auth(bind)
            assert isinstance(sim_endpoint.require_auth(bind, key=KEY),
                              protocol.WireAuth)
        mp.setenv(protocol.AUTH_ENV, KEY.decode())
        assert isinstance(sim_endpoint.require_auth("0.0.0.0"), protocol.WireAuth)
    finally:
        mp.undo()


def test_endpoint_checks_the_key_before_it_loads_anything():
    """The gate fires on a model path that does not exist and without MuJoCo
    installed -- which is the proof that it runs before the model is read, not
    after a working sim has already opened the socket."""
    mp = _no_env_key()
    try:
        mp.delenv(protocol.AUTH_ENV, raising=False)
        with pytest.raises(RuntimeError) as err:
            sim_endpoint.SkateSimEndpoint("/no/such/model.xml", bind="0.0.0.0")
        assert protocol.AUTH_ENV in str(err.value)
        assert "127.0.0.1" in str(err.value)          # tells you the way out
    finally:
        mp.undo()


def test_cli_binds_loopback_by_default_and_never_takes_a_secret():
    ap = sim_endpoint._build_parser()
    args = ap.parse_args(["--model", "x.xml"])
    assert args.bind == "127.0.0.1"
    assert sim_endpoint.is_loopback(args.bind)
    # The secret comes from the environment only: argv is world-readable in `ps`.
    assert "--key" not in ap.format_help()
    assert "--secret" not in ap.format_help()
    assert protocol.AUTH_ENV in ap.format_help()


def test_endpoint_default_matches_the_documented_default():
    """The constructor and the CLI must agree -- a caller that embeds
    SkateSimEndpoint directly gets the same safe bind as one that runs the
    module."""
    import inspect
    sig = inspect.signature(sim_endpoint.SkateSimEndpoint.__init__)
    assert sig.parameters["bind"].default == "127.0.0.1"
    assert sig.parameters["key"].default is None
