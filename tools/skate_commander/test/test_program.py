"""Program runner e2e: the rbt API drives the real bridge over UDP into the
MuJoCo sim endpoint; Click-to-Step, STOP, E-STOP and the sandbox all hold.

    SKATE_MJCF=.../skt_v3_control.xml python3 test/test_program.py
"""

import math
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skate_ros2"))

from skate_commander.bridge import IK_ARRIVE_M, RobotBridge   # noqa: E402
from skate_commander.program import ProgramRunner, RobotAPI   # noqa: E402

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
MODEL = os.environ.get("SKATE_MJCF", str(SKT / "skt_v3_control.xml"))


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


class _Rig:
    """Sim endpoint + bridge + 60 Hz tick thread, like the server runs."""

    def __init__(self):
        from skate_ros2.sim_endpoint import SkateSimEndpoint
        port = _free_port()
        self.ep = SkateSimEndpoint(MODEL, port=port, bind="127.0.0.1",
                                   verbose=False)
        self.epth = threading.Thread(target=self.ep.run,
                                     kwargs={"duration": 120.0}, daemon=True)
        self.epth.start()
        urdf = Path(MODEL).parent / "skt_v3.urdf"
        kin, limits = {}, None
        if urdf.exists():
            from skate_commander.kinematics import ArmKinematics
            from skate_commander.urdf import joint_limits, parse_urdf
            model = parse_urdf(urdf)
            kin = {a: ArmKinematics(model, a) for a in ("left", "right")}
            limits = joint_limits(model)
        self.br = RobotBridge(sim_host="127.0.0.1", sim_port=port,
                              limits=limits, kin=kin)
        self._stop = threading.Event()
        self.tick = threading.Thread(target=self._loop, daemon=True)
        self.tick.start()
        t0 = time.monotonic()
        while self.br.targ is None and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        assert self.br.targ is not None, "bridge never armed"
        self.br.resume()

    def _loop(self):
        while not self._stop.is_set():
            self.br.tick(1 / 60, ui_attached=True)
            time.sleep(1 / 60)

    def close(self):
        self._stop.set()
        self.tick.join(timeout=2)
        self.br.close()
        self.ep.close()


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _deg_to_R(rpy_deg):
    """A (roll, pitch, yaw) degree triple -- what the API speaks -- as a matrix."""
    import numpy as np
    from skate_commander.kinematics import rpy_to_R
    return rpy_to_R(*np.radians(np.asarray(rpy_deg, float)))


def _wrist_gap(a, b):
    """Angle between two (roll, pitch, yaw) degree triples, in degrees.

    Compared as rotations, not componentwise: the triple is a chart, and two
    charts that differ in every entry can still be the same wrist."""
    import numpy as np
    from skate_commander.kinematics import rot_error
    return float(np.degrees(np.linalg.norm(
        rot_error(_deg_to_R(a), _deg_to_R(b)))))


def test_program_runs_moves_and_logs():
    if not Path(MODEL).exists():
        pytest.skip("no control model")
    rig = _Rig()
    r = ProgramRunner(rig.br)
    code = "\n".join([
        "rbt.movej('L4', 60)",
        "rbt.movel('right', dz=40)",
        "print('tcp:', rbt.tcp('right'))",
        "rbt.wait(0.2)",
    ])
    assert r.run(code)
    assert _wait(lambda: not r.running, 30), "program never finished"
    log = "\n".join(r.log)
    assert "* program finished" in log, log
    assert "tcp:" in log
    assert abs(rig.br.targ[11] - math.radians(60)) < 0.02
    print("PASS program ran: movej + movel + print")
    rig.close()


def test_click_to_step_and_stop():
    if not Path(MODEL).exists():
        pytest.skip("no control model")
    rig = _Rig()
    r = ProgramRunner(rig.br)
    code = "rbt.movej('L4', 30)\nrbt.movej('L4', 70)\nrbt.wait(30)"
    assert r.run(code, step=True)
    assert _wait(lambda: r.paused, 5), "did not pause at the first command"
    assert r.line == 1 and "movej" in r.current
    before = float(rig.br.targ[11])
    time.sleep(0.4)                      # paused = nothing moves
    assert abs(rig.br.targ[11] - before) < 1e-9
    r.step()                             # execute command 1, pause at 2
    assert _wait(lambda: r.paused and r.line == 2, 15)
    assert abs(rig.br.targ[11] - math.radians(30)) < 0.02
    print("PASS Click-to-Step: paused, stepped exactly one command")
    r.step()                             # command 2 runs...
    assert _wait(lambda: r.paused and r.line == 3, 15)
    r.run()                              # ...RUN releases into wait(30)
    time.sleep(0.3)
    assert r.running and not r.paused
    r.stop("test")
    assert _wait(lambda: not r.running, 5), "STOP did not interrupt wait()"
    assert any("stopped" in ln for ln in r.log)
    print("PASS STOP interrupts a long wait")
    rig.close()


def test_estop_kills_program_and_sandbox_holds():
    if not Path(MODEL).exists():
        pytest.skip("no control model")
    rig = _Rig()
    r = ProgramRunner(rig.br)
    assert r.run("rbt.wait(30)")
    time.sleep(0.3)
    rig.br.trigger_estop()
    assert _wait(lambda: not r.running, 5), "E-STOP did not kill the program"
    assert any("E-STOP" in ln for ln in r.log)
    print("PASS E-STOP aborts a running program")

    rig.br.resume()
    r2 = ProgramRunner(rig.br)
    # the AST sandbox refuses import at compile time — never even starts
    assert not r2.run("import os\nprint(os.getcwd())")
    assert any(ln.startswith("x rejected") for ln in r2.log), r2.log
    # open() passes the AST but the restricted builtins lack it -> NameError
    assert r2.run("open('x.txt', 'w')")
    assert _wait(lambda: not r2.running, 5)
    assert any("NameError" in ln for ln in r2.log), r2.log
    print("PASS sandbox: import rejected up front; open unavailable at runtime")

    bad = ProgramRunner(rig.br)
    assert not bad.run("def broken(:\n  pass")
    assert any("syntax error" in ln for ln in bad.log)
    print("PASS syntax errors are reported with a line number")
    rig.close()


def test_teach_in_record_and_replay():
    """REC full circle: manual moves -> settled keyposes -> generated rbt
    code -> replay reproduces the pose (through the same safe bridge)."""
    if not Path(MODEL).exists():
        pytest.skip("no control model")
    from skate_commander.program import PoseRecorder
    rig = _Rig()
    rec = PoseRecorder()
    rig.br.recorder = rec                  # tick() now feeds it

    rec.start(rig.br.targ)
    rig.br.set_joint(11, 0.6)              # one joint -> movej line
    time.sleep(1.0)                        # settle (0.6 s) + margin
    rig.br.set_joint(11, 1.1)              # two joints inside one window
    time.sleep(0.2)
    rig.br.set_joint(19, 0.8)              # -> coordinated pose({...}) line
    time.sleep(1.0)
    code = rec.stop()
    assert "rbt.movej('L4'" in code, code
    assert "rbt.pose({" in code and "'R4'" in code, code
    print("PASS teach-in generated:", " | ".join(rec.lines))

    rig.br.home()                          # move away, then replay
    time.sleep(0.5)
    r = ProgramRunner(rig.br)
    assert r.run(code)
    assert _wait(lambda: not r.running, 30), "replay never finished"
    assert "* program finished" in "\n".join(r.log), r.log
    assert abs(rig.br.targ[11] - 1.1) < 0.03
    assert abs(rig.br.targ[19] - 0.8) < 0.03
    print("PASS replay reproduced the recorded pose")

    # rbt.pose is one coordinated move with ONE guard check
    r2 = ProgramRunner(rig.br)
    assert r2.run("rbt.pose({'L4': 45, 'R4': 45})")
    assert _wait(lambda: not r2.running, 15)
    assert abs(rig.br.targ[11] - math.radians(45)) < 0.02
    assert abs(rig.br.targ[19] - math.radians(45)) < 0.02
    print("PASS rbt.pose moves both elbows in one command")
    rig.close()


def test_moveto_holds_a_wrist_orientation():
    """The gap this closes: a program can now ask for a pose, not just a point.

    Pinned against the free-wrist move to the SAME point rather than against a
    tolerance on its own. A position-only solve is free to spend the wrist in
    the null space and does, so if the orientation arguments were quietly
    dropped the two moves would leave the wrist in the same place. The test is
    that they do not -- and the free-wrist assertion is what keeps the pinned
    one from passing vacuously, so if the solver ever stops rolling the wrist
    on a sideways move, this test wants to be rewritten rather than trusted."""
    if not Path(MODEL).exists():
        pytest.skip("no control model")
    rig = _Rig()
    r = ProgramRunner(rig.br)
    rbt = RobotAPI(r)
    arm = "right"

    home = rbt.tcp(arm)
    flat = rbt.tcp_rpy(arm)
    assert home is not None and flat is not None and len(flat) == 3
    side = (home[0], home[1] + 40.0, home[2])

    rbt.moveto(arm, *side)                       # free wrist: goes where it likes
    free = rbt.tcp_rpy(arm)
    rbt.moveto(arm, *home, *flat)                # back, wrist named this time
    rbt.moveto(arm, *side, *flat)                # same point, wrist pinned
    held = rbt.tcp_rpy(arm)

    assert _wrist_gap(held, flat) < 1.5, \
        f"asked for the wrist at {flat} and got {held}"
    assert _wrist_gap(free, flat) > 3.0, \
        (f"the free wrist stayed at {free}: with nothing to compare against, "
         "this test can no longer tell a held wrist from a lucky one")
    landed = rbt.tcp(arm)
    assert max(abs(landed[i] - side[i]) for i in range(3)) < 6.0, \
        f"asked for the point {side} and got {landed}"
    print(f"PASS moveto pose: free wrist wandered "
          f"{_wrist_gap(free, flat):.1f}, pinned wrist held "
          f"{_wrist_gap(held, flat):.2f} (degrees)")

    # tcp_rpy is the readout moveto was missing: jog away, then reproduce the
    # pose from nothing but the two triples that were read off it.
    rbt.movej("R4", 35)
    assert _wrist_gap(rbt.tcp_rpy(arm), flat) > 3.0, "the jog did not move it"
    rbt.moveto(arm, *home, *flat)
    assert _wrist_gap(rbt.tcp_rpy(arm), flat) < 1.5
    back = rbt.tcp(arm)
    assert max(abs(back[i] - home[i]) for i in range(3)) < 6.0, \
        f"pose round trip landed at {back}, not {home}"
    print("PASS tcp / tcp_rpy read a pose back that moveto can retype")

    # orientation is all three angles or none -- two of them is a mistake
    for bad in ((10.0, None, None), (10.0, 20.0, None), (None, 20.0, None)):
        with pytest.raises(ValueError):
            rbt.moveto(arm, *home, *bad)
    with pytest.raises(ValueError):
        rbt.moveto("middle", *home)
    print("PASS moveto rejects a half-specified orientation")

    # A pose target aimed at the point the TCP is ALREADY standing on. A
    # position-only target sees no error there and drops on the first tick, so
    # everything the bridge does after that tick is the orientation half of the
    # target doing the work, and the wrist has to actually get there rather than
    # be abandoned partway.
    here = rig.br.kin[arm].fk(rig.br.targ)
    turn = (flat[0] + 60.0, flat[1], flat[2])
    rig.br.set_ik_target(arm, here, auto=True, rot=_deg_to_R(turn))
    assert rig.br.ik_targets.get(arm) is not None, "target refused"
    time.sleep(0.25)
    assert rig.br.ik_targets.get(arm) is not None, \
        "arrived on position alone -- the wrist was never part of the target"
    assert _wait(lambda: rig.br.ik_targets.get(arm) is None, 20), \
        "a self-clearing target has to stop on its own"
    assert _wrist_gap(rbt.tcp_rpy(arm), turn) < 1.5, \
        "gave up on the target while the wrist was still turning"
    print("PASS a pose target waits for the wrist instead of the point alone")

    # ...and the wrist got there by SPENDING the point: rolling this wrist
    # drags the tool tip well outside the arrival band and the solver never
    # buys it all back, which is why the bridge's give-up timer watches the
    # tip alone. The tip is the term that settles last on a pose task, so an
    # orientation clause in that timer could not change when it fires. If this
    # ever goes red the wrist has become cheap to turn, and the timer -- not
    # this assertion -- is what wants rewriting.
    drag = math.dist(rbt.tcp(arm), [v * 1000.0 for v in here])
    assert drag > IK_ARRIVE_M * 1000.0, \
        (f"a full turn of the wrist moved the tip by {drag}, inside the band "
         "the bridge calls arrived: the give-up timer is position-only "
         "because the tip settles last, and this is the measurement for it")
    print(f"PASS turning the wrist costs the tip {drag:.1f} (world millimeters)")

    # ...and a pose it cannot reach still stops, rather than riding the timeout
    far = (home[0] + 900.0, home[1], home[2])
    t0 = time.monotonic()
    assert rbt.moveto(arm, *far, *flat) is not False
    assert time.monotonic() - t0 < 13.0, \
        "an out-of-reach pose sat there until moveto's own timeout fired"
    assert rig.br.ik_targets.get(arm) is None
    print("PASS an out-of-reach pose gives up instead of hanging")

    # both readouts and the six-argument call survive the AST sandbox
    r2 = ProgramRunner(rig.br)
    assert r2.run("p = rbt.tcp('right')\n"
                  "o = rbt.tcp_rpy('right')\n"
                  "print('pose:', p, o)\n"
                  "rbt.moveto('right', p[0], p[1], p[2], o[0], o[1], o[2])\n")
    assert _wait(lambda: not r2.running, 30), "program never finished"
    log = "\n".join(r2.log)
    assert "* program finished" in log, log
    assert "pose:" in log, log
    print("PASS a sandboxed program can read a pose and command it back")
    rig.close()


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
