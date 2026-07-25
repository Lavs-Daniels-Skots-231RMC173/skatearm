"""M4 in the CELL (#28) — the live S0..S7 cycle grips and releases for real.

The demonstrator cycle used to fake both hands with `weld` equalities. This
suite runs the REPO's own sequencer against a scene built by the REPO's own
``make_cell_scene.py --gripper`` and checks the four claims that make the
gripper path worth having:

1. OPT-IN. The default cell model has no ``grip`` actuator, so ``Cell.jaws`` is
   False and every existing demo, test and log stays on the weld path. The
   gripper model is a separate file.
2. THE RIGHT HAND IS WELD-FREE. After S1 the peg is held while ``grasp_right``
   is INACTIVE — the pad force sensor, not an equality, is what reports the
   grasp. (The LEFT hand keeps its weld: there is one gripper on the robot.)
3. THE CARRY IS FRICTION. Through S2/S3 the peg stays put in the jaw frame, so
   it is being clamped and not teleported along.
4. RELEASE IS PHYSICAL. S4 ends by OPENING the jaws — jaw travel goes negative
   and the pad force falls to zero — and the peg stays in the bore because it
   is seated, not because something is still holding it.

Runs one cycle in chunks (``run_cycle(cell, steps=[...])``, the same chunked
API the renderer uses) and asserts between the chunks, so the ~40 s of physics
is paid once. Headless; needs mujoco + the collision MJCF (SKT_DIR), and SKIPs
cleanly without them.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_RUN = None
_SKIP = None


def _skt():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return None, "mujoco not available"
    skt = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
    if not (skt / "skt_v3_collision.xml").exists():
        return None, "collision model not available"
    return skt, None


def _cycle():
    """Build the gripper cell once, run S0..S7 in chunks, snapshot as we go."""
    global _RUN, _SKIP
    if _RUN is not None or _SKIP is not None:
        return _RUN
    skt, _SKIP = _skt()
    if skt is None:
        return None

    import mujoco
    import numpy as np
    from make_cell_scene import make
    from sequencer import Cell, run_cycle

    m = mujoco.MjModel.from_xml_path(make(str(skt), gripper=True))
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)
    cell = Cell(m, d)
    cell.t0 = d.time

    eq_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right")
    pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
    snap = {}

    def peg_in_jaw_frame():
        """Peg position expressed in the right wrist's frame (m)."""
        R = d.site_xmat[cell.armR.site].reshape(3, 3)
        return R.T @ (d.xpos[pg] - cell.armR.ee_pos())

    run_cycle(cell, steps=["S0", "S1"])
    snap["after_S1"] = dict(force=cell.grip_force(), jaw=cell.jaw_mm(),
                            weld_active=bool(d.eq_active[eq_r]),
                            grasped=cell.grasped("right"),
                            peg_in_jaw=peg_in_jaw_frame())
    run_cycle(cell, steps=["S2", "S3"])
    snap["after_S3"] = dict(force=cell.grip_force(),
                            peg_in_jaw=peg_in_jaw_frame())
    run_cycle(cell, steps=["S4"])
    snap["after_S4"] = dict(force=cell.grip_force(), jaw=cell.jaw_mm(),
                            depth=cell.insertion_depth())
    run_cycle(cell, steps=["S5", "S6", "S7"])
    snap["end"] = dict(depth=cell.insertion_depth(), tilt=cell.tilt_deg("peg"),
                       base=d.xpos[mujoco.mj_name2id(
                           m, mujoco.mjtObj.mjOBJ_BODY, "base_part")].copy())
    _RUN = (cell, snap, np)
    return _RUN


def _need():
    run = _cycle()
    if run is None:
        pytest.skip(_SKIP)
    return run


def test_gripper_cell_is_opt_in():
    """The DEFAULT cell has no jaws: existing demos and tests are untouched."""
    skt, why = _skt()
    if skt is None:
        pytest.skip(why)
    import mujoco
    from make_cell_scene import make
    from sequencer import Cell

    m = mujoco.MjModel.from_xml_path(make(str(skt)))       # no gripper=True
    plain = Cell(m, mujoco.MjData(m))
    assert plain.grip < 0, "the default cell must not carry a grip actuator"
    assert plain.jaws is False
    assert plain.grasped("right") is False, \
        "with no jaws the right hand must fall back to its weld flag"

    gm = mujoco.MjModel.from_xml_path(make(str(skt), gripper=True))
    grip = Cell(gm, mujoco.MjData(gm))
    assert grip.jaws is True, "the --gripper cell must be detected as jawed"
    for sensor in ("grip_force", "jaw"):
        assert mujoco.mj_name2id(gm, mujoco.mjtObj.mjOBJ_SENSOR, sensor) >= 0
    # the left hand keeps its weld — there is one gripper, on the right wrist
    assert mujoco.mj_name2id(gm, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_left") >= 0


def test_right_hand_grips_without_its_weld():
    """S1's receptivity is a FORCE, and grasp_right never engages."""
    cell, snap, _ = _need()
    s1 = snap["after_S1"]
    assert not s1["weld_active"], \
        "the right weld must stay inactive — the jaws are doing the holding"
    assert s1["force"] > 1.0, f"pad force {s1['force']:.2f} N is not a grasp"
    assert s1["grasped"], "Cell.grasped('right') must read the pad sensor"
    assert s1["jaw"] > 0, f"jaws should have closed, travel is {s1['jaw']:.2f} mm"


def test_carry_does_not_slip_the_peg():
    """The peg keeps its pose in the jaw frame across the S2 carry and S3 align.

    This is a SLIP test, not a weld test — a welded peg would hold station too.
    What makes the number mean something is the previous test, which shows the
    right weld is inactive while this is happening: the only thing resisting
    gravity and the carry accelerations here is pad friction. Measured drift is
    ~0.3 mm; the 4 mm bound is the point at which the V-groove would no longer
    have the peg centred for S3."""
    cell, snap, np = _need()
    drift = np.linalg.norm(snap["after_S3"]["peg_in_jaw"]
                           - snap["after_S1"]["peg_in_jaw"])
    assert drift < 0.004, f"peg slipped {drift*1000:.1f} mm inside the jaws"
    assert snap["after_S3"]["force"] > 1.0, "grip force collapsed during the carry"


def test_release_opens_the_jaws_and_the_peg_stays_seated():
    """S4 releases by OPENING, and the insert holds once nothing is gripping it."""
    cell, snap, _ = _need()
    s4 = snap["after_S4"]
    assert s4["jaw"] < 0, f"jaws did not open (travel {s4['jaw']:.2f} mm)"
    assert s4["force"] < 0.5, f"pads still loaded at {s4['force']:.2f} N"
    assert s4["depth"] > 0.015, f"peg only {s4['depth']*1000:.1f} mm into the bore"


def test_cycle_completes_and_qc_accepts():
    """The whole S0..S7 GRAFCET runs on the jaws and the unit passes QC."""
    cell, snap, np = _need()
    s4 = [e for e in cell.log if "seated" in e][-1]
    assert s4["seated"] and not s4["aborted"], f"S4 did not seat: {s4}"
    # `seated` is the ONLY stall detector here: with a friction grasp a peg left
    # proud of the bore settles under its own weight once the jaws open, so the
    # camera cannot see the failure afterwards.
    assert cell.qc_pass, "QC rejected a good unit"
    assert snap["end"]["depth"] > 0.015
    assert snap["end"]["tilt"] < 6.0
    assert snap["end"]["base"][0] < -0.15, "an ACCEPTed unit belongs in the left bin"
    assert cell.log[-1]["cycle_time_s"] < 60.0, "cycle blew the 60 s takt target"
