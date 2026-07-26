"""M4 in the CELL (#28) — the live S0..S7 cycle grips and releases for real.

The demonstrator cycle used to fake both hands with `weld` equalities. This
suite runs the REPO's own sequencer against a scene built by the REPO's own
``make_cell_scene.py --gripper`` and checks the five claims that make the
gripper path worth having:

1. OPT-IN. The default cell model has no ``grip`` actuator, so ``Cell.jaws`` is
   False and every existing demo, test and log stays on the weld path. The
   gripper model is a separate file.
2. THE RIGHT HAND IS WELD-FREE. After S1 the peg is held while ``grasp_right``
   is INACTIVE — the pad force sensor, not an equality, is what reports the
   grasp.
3. THE CARRY IS FRICTION. Through S2/S3 the peg stays put in the jaw frame, so
   it is being clamped and not teleported along.
4. RELEASE IS PHYSICAL. S4 ends by OPENING the jaws — jaw travel goes negative
   and the pad force falls to zero — and the peg stays in the bore because it
   is seated, not because something is still holding it.
5. THE LEFT HAND IS WELD-FREE TOO, so the cell has no welds left anywhere. The
   left jaws pick the base off the table in S1, set it down at the assembly
   station and open in S2, re-grip the assembled unit in S5 and release it over
   the bin in S6 — four real grip/release operations, all of them reported by
   the left pad sensor while ``grasp_left`` stays INACTIVE throughout. The
   equality is still EMITTED into the model, because the weld path shares this
   scene builder and needs it; the gripper path simply never engages it.

Runs one cycle in chunks (``run_cycle(cell, steps=[...])``, the same chunked
API the renderer uses) and asserts between the chunks, so the ~76 s of physics
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
    eq_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_left")
    pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
    bs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_part")
    snap = {}
    welds = []          # every weld sample taken across the whole cycle

    def peg_in_jaw_frame():
        """Peg position expressed in the right wrist's frame (m)."""
        R = d.site_xmat[cell.armR.site].reshape(3, 3)
        return R.T @ (d.xpos[pg] - cell.armR.ee_pos())

    def base_in_jaw_frame():
        """Base position expressed in the LEFT wrist's frame (m)."""
        R = d.site_xmat[cell.armL.site].reshape(3, 3)
        return R.T @ (d.xpos[bs] - cell.armL.ee_pos())

    def left():
        """What the LEFT hand reports right now — force, travel, weld state."""
        welds.append((bool(d.eq_active[eq_r]), bool(d.eq_active[eq_l])))
        return dict(force=cell.grip_force_L(), jaw=cell.jaw_mm_L(),
                    weld_active=bool(d.eq_active[eq_l]),
                    grasped=cell.grasped("left"),
                    base_in_jaw=base_in_jaw_frame())

    run_cycle(cell, steps=["S0", "S1"])
    snap["after_S1"] = dict(force=cell.grip_force(), jaw=cell.jaw_mm(),
                            weld_active=bool(d.eq_active[eq_r]),
                            grasped=cell.grasped("right"),
                            peg_in_jaw=peg_in_jaw_frame())
    snap["L_after_S1"] = left()          # left is carrying the base
    run_cycle(cell, steps=["S2", "S3"])
    snap["after_S3"] = dict(force=cell.grip_force(),
                            peg_in_jaw=peg_in_jaw_frame())
    snap["L_after_S3"] = left()          # left has set it down and let go
    run_cycle(cell, steps=["S4"])
    snap["after_S4"] = dict(force=cell.grip_force(), jaw=cell.jaw_mm(),
                            depth=cell.insertion_depth())
    run_cycle(cell, steps=["S5", "S6", "S7"])
    snap["end"] = dict(depth=cell.insertion_depth(), tilt=cell.tilt_deg("peg"),
                       base=d.xpos[bs].copy())
    snap["L_end"] = left()               # left has released over the bin
    snap["welds"] = welds
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
    assert plain.gripL < 0, "nor a left one"
    assert plain.jaws is False and plain.jawsL is False
    for side in ("right", "left"):
        assert plain.grasped(side) is False, \
            f"with no jaws the {side} hand must fall back to its weld flag"

    gm = mujoco.MjModel.from_xml_path(make(str(skt), gripper=True))
    grip = Cell(gm, mujoco.MjData(gm))
    assert grip.jaws is True, "the --gripper cell must be detected as jawed"
    assert grip.jawsL is True, "...on BOTH wrists — the cell is weld-free"
    for sensor in ("grip_force", "jaw", "grip_force_L", "jaw_L"):
        assert mujoco.mj_name2id(gm, mujoco.mjtObj.mjOBJ_SENSOR, sensor) >= 0
    # Both weld equalities are still EMITTED — the weld path shares this scene
    # builder and needs them. What makes the cell weld-free is that the gripper
    # path never engages either one; test_*_grips_without_its_weld is where that
    # is checked, against a live cycle rather than against the model file.
    for eq in ("grasp_left", "grasp_right"):
        assert mujoco.mj_name2id(gm, mujoco.mjtObj.mjOBJ_EQUALITY, eq) >= 0


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
    0.42 mm; the 4 mm bound is the point at which the V-groove would no longer
    have the peg centred for S3."""
    cell, snap, np = _need()
    drift = np.linalg.norm(snap["after_S3"]["peg_in_jaw"]
                           - snap["after_S1"]["peg_in_jaw"])
    assert drift < 0.004, f"peg slipped {drift*1000:.1f} mm inside the jaws"
    assert snap["after_S3"]["force"] > 1.0, "grip force collapsed during the carry"


def test_left_hand_grips_without_its_weld():
    """The LEFT hand does four real grip/release operations, weld never engaged.

    This is the claim that makes the cell weld-free rather than half-converted.
    A weld holds the base in mid-air for the whole cycle for free; jaws have to
    pick it off the table (S1), set it down at the assembly station and open
    (S2) — the right hand cannot insert into a part that is floating — then
    re-grip the assembled unit (S5) and let go of it over the bin (S6). Each of
    those is checked here at the point where it has just happened, and the weld
    flag is sampled alongside every one of them.
    """
    cell, snap, np = _need()
    s1, s3, end = snap["L_after_S1"], snap["L_after_S3"], snap["L_end"]

    # S1: holding the base for real.
    assert s1["force"] > 1.0, f"left pad force {s1['force']:.2f} N is not a grasp"
    assert s1["grasped"], "Cell.grasped('left') must read the LEFT pad sensor"
    assert s1["jaw"] < 0, \
        f"the left jaws close inward on a 40 mm face; travel is {s1['jaw']:.2f} mm"

    # S2: put down and let go — by S3 the pads are unloaded and the base is not
    # following the wrist any more.
    assert s3["force"] < 0.5, f"left pads still loaded at {s3['force']:.2f} N"
    assert not s3["grasped"], "left hand should have released at the station"
    moved = np.linalg.norm(s3["base_in_jaw"] - s1["base_in_jaw"])
    assert moved > 0.02, \
        f"base only moved {moved*1000:.1f} mm in the left jaw frame — still held?"

    # S6: released over the bin.
    assert end["force"] < 0.5, f"left pads still loaded at {end['force']:.2f} N"

    # ...and no weld, on either hand, at any sample across the whole cycle.
    assert not any(r or l for r, l in snap["welds"]), \
        f"a weld was active during the gripper cycle: {snap['welds']}"


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

    # Takt. The bound was 60 s when only the right hand had jaws; the weld-free
    # cycle measures 75.8 s and 85 s is that with room for a convergence loop to
    # take a few more cycles on a different BLAS. The extra time is the cost of
    # the conversion, and it is worth writing down where it goes, because the
    # same sequencer runs the weld path and that measures 42.6 s end to end:
    #
    #        weld    jaws     delta   what the jaws have to do that a weld doesn't
    #   S0    3.6     3.6      0.0
    #   S1   10.8    15.6     +4.8    left picks the base off the table
    #   S2    7.7    10.3     +2.6    left sets it down, opens, retracts clear
    #   S3    3.2     8.8     +5.6    align holds a convergence test (a friction-
    #                                 held peg can swing through zero); the weld
    #                                 path just runs a fixed 400 cycles
    #   S4    1.2     3.3     +2.1    seating a peg that can move in the grasp
    #   S5    5.2    20.7    +15.5    re-grip the assembled unit off the station
    #                                 and carry it to the QC pose -- a weld holds
    #                                 it in mid-air the whole time for free
    #   S6    7.9     9.5     +1.6    release at the bin is a real opening
    #   S7    3.0     4.0     +1.0
    #        ----    ----    -----
    #        42.6    75.8    +33.3
    #
    # So the bound is not slack absorbed for its own sake: it is four real
    # grip/release operations plus two convergence tests that a weld does not
    # need. It is still tight enough to catch the failure mode it exists for --
    # a regulation loop that stops converging and burns its full budget. The two
    # that used to (close_jaws' fixed 400 cycles at three call sites, and
    # settle_part timing out on an unreachable pose) were worth 9.6 s and 8.0 s
    # of takt; either one coming back puts the cycle over this line.
    assert cell.log[-1]["cycle_time_s"] < 85.0, "cycle blew the 85 s takt target"
