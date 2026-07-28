"""Loopback tests: SkateLink <-> a minimal fake firmware over localhost UDP.

No MuJoCo, no ROS — just the wire contract.
"""

import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skate_ros2 import names                    # noqa: E402
from skate_ros2 import shared_classes_def as SCD  # noqa: E402
from skate_ros2.protocol import (COMMAND_ID, FRAG_MAGIC,  # noqa: E402
                                 FRAG_MAX_DGRAM, Reassembler, SkateLink,
                                 pack_command, pack_datagrams, unpack_packet)


def make_fake_robot():
    """Bind an ephemeral UDP socket acting as the firmware side."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)
    return sock, sock.getsockname()[1]


def test_pack_unpack_roundtrip():
    targ = np.arange(26, dtype=np.float64) / 10.0
    data = pack_command(targ, (0.1, -0.2, 0.3), 0.9, (1, 0, 1))
    pkt_id, (t, v, h, dm) = unpack_packet(data)
    assert pkt_id == COMMAND_ID
    assert np.allclose(t, targ)
    assert np.allclose(v, [0.1, -0.2, 0.3])
    assert h == 0.9
    assert dm == (1, 0, 1)


def test_pack_command_validates_shape():
    try:
        pack_command([0.0] * 25)
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_command_reaches_robot_and_telemetry_comes_back():
    robot, port = make_fake_robot()
    link = SkateLink("127.0.0.1", port)

    # client -> robot: command arrives and decodes
    targ = np.array(names.DEFAULT_POSE)
    assert link.send_command(targ, deadman=(1, 1, 1))
    data, client_addr = robot.recvfrom(65536)
    pkt_id, (t, v, h, dm) = pickle.loads(data)
    assert pkt_id == COMMAND_ID and dm == (1, 1, 1)
    assert np.allclose(t, targ)

    # robot -> client: telemetry decodes through the vendored classes. Sent the
    # way the endpoint sends it -- every telemetry object in sim_endpoint goes
    # out through pack_datagrams -- rather than hand-rolled raw. This state_est
    # pickles to 1605 B, and a raw datagram that big does not survive WSL2
    # mirrored-networking loopback: measured there, 127.0.0.1 delivers payloads
    # up to 1472 B (= 1500 B Ethernet MTU - 20 IP - 8 UDP) and silently drops
    # everything above, despite lo advertising mtu 65536. That is the same
    # behaviour pack_datagrams exists to absorb, so the test has no business
    # bypassing it. For the small motor_state pack_datagrams is a no-op that
    # returns the identical bytes -- test_small_packet_is_single_unchanged_
    # datagram pins that -- so the wire here is unchanged from before.
    ms = SCD.motor_state()
    ms.motor_pos = names.vector_to_can_dict(np.linspace(0, 1, 26))
    for d in pack_datagrams(1, ms, 0):
        robot.sendto(d, client_addr)
    se = SCD.state_est()
    se.dof_pos = names.vector_to_can_dict(np.linspace(1, 2, 26))
    for d in pack_datagrams(2, se, 1):
        robot.sendto(d, client_addr)

    deadline = time.time() + 1.0
    while time.time() < deadline and link.state.state_estimates is None:
        link.poll()
        time.sleep(0.01)

    assert link.state.motor_states is not None
    assert link.state.state_estimates is not None
    assert link.connected
    pos = link.state.dof_pos()
    assert abs(pos[0] - 1.0) < 1e-9 and abs(pos[25] - 2.0) < 1e-9

    # staleness: after 0.3 s with no packets the link reports disconnected
    time.sleep(0.35)
    assert not link.connected

    link.close()
    robot.close()


def test_heartbeat_is_official_yo():
    robot, port = make_fake_robot()
    link = SkateLink("127.0.0.1", port)
    link.poll()  # first poll fires a heartbeat
    data, _ = robot.recvfrom(65536)
    assert data == b"yo"
    link.close()
    robot.close()


def _big_state_est():
    """A state_est that pickles to > FRAG_MAX_DGRAM — the packet WSL2 loopback
    silently drops unless it is fragmented."""
    se = SCD.state_est()
    se.dof_pos = names.vector_to_can_dict(np.linspace(0, 1, 26))
    se.dof_vel = names.vector_to_can_dict(np.linspace(1, 2, 26))
    se.dof_torque = names.vector_to_can_dict(np.linspace(2, 3, 26))
    return se


def test_small_packet_is_single_unchanged_datagram():
    ms = SCD.motor_state()
    ms.motor_pos = names.vector_to_can_dict(np.zeros(26))
    dgs = pack_datagrams(1, ms, 0)
    assert len(dgs) == 1
    assert dgs[0] == pickle.dumps((1, ms))            # byte-for-byte compatible
    assert not dgs[0].startswith(FRAG_MAGIC)


def test_large_packet_splits_into_bounded_fragments():
    se = _big_state_est()
    assert len(pickle.dumps((2, se))) > FRAG_MAX_DGRAM
    dgs = pack_datagrams(2, se, 7)
    assert len(dgs) >= 2
    for d in dgs:
        assert d.startswith(FRAG_MAGIC)
        assert len(d) <= FRAG_MAX_DGRAM              # each fits under the MTU


def test_reassembler_roundtrip():
    dgs = pack_datagrams(2, _big_state_est(), 3)
    r = Reassembler()
    out = [r.feed(d) for d in dgs][-1]
    assert out is not None
    pkt_id, obj = unpack_packet(out)
    assert pkt_id == 2
    assert abs(names.can_dict_to_vector(obj.dof_pos)[25] - 1.0) < 1e-9


def test_reassembler_handles_out_of_order_fragments():
    dgs = pack_datagrams(2, _big_state_est(), 9)
    assert len(dgs) >= 2
    r = Reassembler()
    outs = [r.feed(d) for d in reversed(dgs)]
    assert outs[-1] is not None                       # completes regardless of order
    assert unpack_packet(outs[-1])[0] == 2


def test_reassembler_passes_plain_datagram_through():
    r = Reassembler()
    plain = pickle.dumps((3, "hi"))
    assert r.feed(plain) == plain


def test_reassembler_rejects_absurd_chunk_count():
    from skate_ros2.protocol import FRAG_HEADER, FRAG_MAX_CHUNKS
    bad = FRAG_MAGIC + FRAG_HEADER.pack(1, FRAG_MAX_CHUNKS + 1, 0) + b"x"
    assert Reassembler().feed(bad) is None


def test_large_state_est_survives_link_over_loopback():
    """End-to-end: a >MTU state_est reaches SkateLink through fragmentation."""
    robot, port = make_fake_robot()
    link = SkateLink("127.0.0.1", port)
    link.poll()                                       # fires the "yo" heartbeat
    _, client_addr = robot.recvfrom(65536)
    se = _big_state_est()
    for d in pack_datagrams(2, se, 0):
        robot.sendto(d, client_addr)
    deadline = time.time() + 1.0
    while time.time() < deadline and link.state.state_estimates is None:
        link.poll()
        time.sleep(0.01)
    assert link.state.state_estimates is not None
    assert abs(link.state.dof_pos()[25] - 1.0) < 1e-9
    link.close()
    robot.close()


if __name__ == "__main__":
    for f in [test_pack_unpack_roundtrip, test_pack_command_validates_shape,
              test_command_reaches_robot_and_telemetry_comes_back,
              test_heartbeat_is_official_yo,
              test_small_packet_is_single_unchanged_datagram,
              test_large_packet_splits_into_bounded_fragments,
              test_reassembler_roundtrip,
              test_reassembler_handles_out_of_order_fragments,
              test_reassembler_passes_plain_datagram_through,
              test_reassembler_rejects_absurd_chunk_count,
              test_large_state_est_survives_link_over_loopback]:
        f()
        print(f"PASS {f.__name__}")
