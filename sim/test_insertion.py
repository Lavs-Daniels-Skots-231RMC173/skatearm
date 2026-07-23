"""M2 — force-regulated insertion (CI).

Three behaviours that prove the controller earns its keep, all on the rigid
fixture with the peg staged over the pocket (shared staging, restored per case):

1. NOMINAL, search on  — the peg seats, the regulated axial force stays well under
   the abort (peak < 8.5 N), and it does not abort.
2. 3 mm misalignment, search on — the spiral hole-search RECOVERS it and it seats.
3. 8 mm misalignment, search OFF — the open-loop descent JAMS on the rim and does
   NOT seat (this is the case the search exists to fix).

Headless; needs mujoco + the cell MJCF (SKT_DIR -> skt_v3). SKIPs cleanly without
them. Staging is ~30 s of sim, so it is run once and cached for the three cases.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_STAGE = None
_SKIP = None


def _staged():
    """Stage once (grasp, carry, align, clamp fixture); cache for all three tests."""
    global _STAGE, _SKIP
    if _STAGE is not None or _SKIP is not None:
        return _STAGE
    try:
        import mujoco  # noqa: F401
    except ImportError:
        _SKIP = "mujoco not available"; return None
    xml = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3")) / "skt_v3_cell.xml"
    if not xml.exists():
        _SKIP = "cell model not available"; return None
    from eval_insertion import load_cell, stage, snapshot
    m = load_cell(str(xml.parent))
    d, armR, W0, bp, pg = stage(m)
    _STAGE = (m, d, armR, snapshot(d), W0, bp, pg)
    return _STAGE


def _run(offset_xy, search):
    from eval_insertion import run_one
    m, d, armR, snap, W0, bp, pg = _STAGE
    return run_one(m, d, armR, snap, W0, bp, pg, offset_xy, search=search)


def test_nominal_seats():
    st = _staged()
    if st is None:
        print(f"SKIP: {_SKIP}"); return
    r = _run([0.0, 0.0], True)
    assert r["seated"] and not r["aborted"] and r["peak_wrench_n"] < 8.5, r
    print(f"PASS nominal: seated, peak {r['peak_wrench_n']} N, depth {r['depth_mm']} mm")


def test_search_recovers_3mm():
    st = _staged()
    if st is None:
        print(f"SKIP: {_SKIP}"); return
    r = _run([0.003, 0.0], True)
    assert r["seated"] and not r["aborted"], r
    print(f"PASS 3mm+search: seated, rel_xy {r['rel_xy_mm']} mm, depth {r['depth_mm']} mm")


def test_no_search_jams_8mm():
    st = _staged()
    if st is None:
        print(f"SKIP: {_SKIP}"); return
    r = _run([0.008, 0.0], False)
    assert not r["seated"], r
    print(f"PASS 8mm no-search: not seated (rel_xy {r['rel_xy_mm']} mm, depth {r['depth_mm']} mm)")


def test_theta_levels_and_seats():
    """An initially TILTED peg is levelled to vertical and seated: the controller
    holds the target orientation upright (relock=False) so the 6-DoF IK rights the
    peg while it inserts."""
    st = _staged()
    if st is None:
        print(f"SKIP: {_SKIP}"); return
    from eval_insertion import run_one_theta
    m, d, armR, snap, W0, bp, pg = _STAGE
    r = run_one_theta(m, d, armR, snap, W0, bp, pg, 12, [0.0, 1.0, 0.0])
    assert r["tilt0_deg"] >= 5.0, r                    # a real initial tilt was injected
    assert r["seated"] and not r["aborted"], r         # controller seated it
    assert r["tiltf_deg"] < 3.0, r                     # having levelled the peg
    print(f"PASS theta: peg tilt {r['tilt0_deg']} -> {r['tiltf_deg']} deg, seated")


if __name__ == "__main__":
    test_nominal_seats()
    test_search_recovers_3mm()
    test_no_search_jams_8mm()
    test_theta_levels_and_seats()
    print("INSERTION (M2) TEST DONE")
