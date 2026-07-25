"""CI guard — every manipulation number in the docs must match the committed data.

No MuJoCo, no model, no GL: this reads only the raw artefacts written by the
eval scripts (``sim/eval_data/*.json``), the benchmark report
(``sim/benchmark_results.json``) and the two cycle logs (``logs/*.json``), and
re-derives each figure quoted in ``docs/MANIPULATION.md``, ``sim/README.md``,
``docs/ROADMAP.md``, ``docs/index.html`` and the top-level ``README.md``.

Regenerate the artefacts (needs the model) with::

    python sim/eval_insertion.py  --model .../skt_v3 --offsets 0,2,4,6,8 --dirs 6 \
        --no-search-baseline --json sim/eval_data/insertion.json
    python sim/eval_insertion.py  --model .../skt_v3 --theta 0,3,6,9,12 \
        --json sim/eval_data/insertion_theta.json
    python sim/eval_admittance.py --model .../skt_v3 --json sim/eval_data/admittance.json
    python sim/eval_gripper.py    --model .../skt_v3 --json sim/eval_data/gripper.json
    python sim/benchmark.py       --model .../skt_v3 --trials 5 --seed 0 \
        --json sim/benchmark_results.json
    python sim/demo_cell_cycle.py --model .../skt_v3 --no-render --log logs/cycle_001.json

If an eval is re-run and a published figure is not updated with it — or a figure
is edited without the data — this test fails. That is the point: it is the same
contract as ``tools/skate_commander/examples/act_reach/test_eval_numbers.py``.

Run: pytest -q sim/test_manipulation_numbers.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ED = os.path.join(HERE, "eval_data")


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def ed(name):
    with open(os.path.join(ED, name), encoding="utf-8") as f:
        return json.load(f)


def rows_by_offset(node):
    return {r["offset_mm"]: r for r in node}


# --------------------------------------------------------------------------- M2

def test_insertion_misalignment_curve():
    """MANIPULATION.md M2: search 0mm 1/1, 2-4mm 6/6, 6mm 5/6, 8mm 3/6; no-search <=1/6."""
    d = ed("insertion.json")
    assert d["dirs_per_offset"] == 6
    assert d["offsets_mm"] == [0, 2, 4, 6, 8]

    s = rows_by_offset(d["modes"]["search"])
    assert (s[0]["seated"], s[0]["trials"]) == (1, 1)        # zero offset = one direction
    assert (s[2]["seated"], s[2]["trials"]) == (6, 6)
    assert (s[4]["seated"], s[4]["trials"]) == (6, 6)
    assert (s[6]["seated"], s[6]["trials"]) == (5, 6)
    assert (s[8]["seated"], s[8]["trials"]) == (3, 6)

    n = rows_by_offset(d["modes"]["no-search"])
    for off, r in n.items():
        assert r["seated"] <= 1, ("no-search jams almost everywhere", off, r)
    # and the search variant must beat the open-loop baseline at every offset
    for off in d["offsets_mm"]:
        assert s[off]["seated"] >= n[off]["seated"], off


def test_insertion_peak_force_regulated():
    """MANIPULATION.md M2: 3.3-3.6 N out to 4 mm, <=4.6 N at 6-8 mm; abort is 9 N."""
    s = rows_by_offset(ed("insertion.json")["modes"]["search"])
    near = [s[o]["peak_wrench_max_n"] for o in (0, 2, 4)]
    far = [s[o]["peak_wrench_max_n"] for o in (6, 8)]
    assert min(near) >= 3.3 and max(near) <= 3.6, near
    assert max(far) <= 4.6, far
    assert max(near + far) < 9.0                             # never near the abort


def test_insertion_theta_tolerance():
    """MANIPULATION.md M2: initial tilts up to ~9 deg are levelled to <2 deg and seated."""
    d = ed("insertion_theta.json")
    rows = d["theta_sweep"]
    assert [r["theta_cmd_deg"] for r in rows] == [0, 3, 6, 9, 12]
    for r in rows:
        assert r["seated"] == r["trials"], r                 # every trial seats
        assert r["tilt_final_max_deg"] < 2.0, r              # levelled to <2 deg
    injected = max(r["tilt_injected_deg"] for r in rows)
    assert 9.0 <= injected < 10.0, injected                  # "up to ~9 deg"


# --------------------------------------------------------------------------- M3

def test_admittance_stiffness_curve():
    """MANIPULATION.md M3: K=200/400/800/1600/3200 -> 40/20/10/5/2.5 mm at F=8 N."""
    c = ed("admittance.json")["stiffness_curve"]
    assert c["wrench_n"] == 8.0
    assert c["k_ratio"] == 16                                # "a 16x stiffness sweep"
    want = {200: 40.0, 400: 20.0, 800: 10.0, 1600: 5.0, 3200: 2.5}
    assert {r["k_n_per_m"]: r["yield_mm"] for r in c["k_sweep"]} == want
    for r in c["k_sweep"]:
        assert r["yield_mm"] == r["f_over_k_mm"], r          # e settles to F/K
        assert r["e_times_k_n"] == 8.0, r                    # "e*K = 8.0 N at every point"
        assert abs(r["after_release_mm"]) <= 0.2, r          # returns to ~0 on release


def test_admittance_per_axis():
    """MANIPULATION.md M3: K=[400,1600,400], F=[8,8,0] -> e=[20,5,0] mm."""
    p = ed("admittance.json")["stiffness_curve"]["per_axis"]
    assert p["k_n_per_m"] == [400, 1600, 400]
    assert p["f_n"] == [8.0, 8.0, 0.0]
    assert p["e_mm"] == [20.0, 5.0, 0.0] == p["f_over_k_mm"]


def test_admittance_push_and_yield():
    """MANIPULATION.md M3: a real +8 N push yields ~21 mm and returns to nominal."""
    p = ed("admittance.json")["push_and_yield"]
    assert p["applied_n"] == 8.0 and p["axis"] == "y"
    ax = "xyz".index(p["axis"])
    assert abs(p["measured_wrench_n"][ax] - 8.0) <= 0.5      # sensor reads the push back
    assert 20.0 <= p["yielded_mm"][ax] <= 22.0               # "~21 mm"
    for v in p["after_release_mm"]:
        assert abs(v) <= 0.5, p                              # returns to the nominal pose


# --------------------------------------------------------------------------- M4

def test_gripper_force_tracking_and_friction_hold():
    """MANIPULATION.md M4: 2 N->2.00 .. 5 N->5.00, monotone; held by friction, dz ~ -3 mm."""
    d = ed("gripper.json")
    assert d["targets_n"] == [2.0, 3.0, 4.0, 5.0]
    rows = d["hold"]
    assert [r["target_n"] for r in rows] == d["targets_n"]
    for r in rows:
        assert abs(r["measured_n"] - r["target_n"]) <= 0.05, r    # tracks the target
        assert r["held"] is True, r                               # friction alone, no weld
        assert -3.5 <= r["part_dz_mm"] <= -2.5, r                 # "dz ~ -3 mm"
        assert abs(r["hold_force_n"] - r["target_n"]) <= 0.05, r  # grasp persists
    meas = [r["measured_n"] for r in rows]
    assert meas == sorted(meas), meas                             # monotone


def test_gripper_slip_curve():
    """MANIPULATION.md M4: grasp 2/3/4/5 N -> slips at 3.75/4.25/5.00/5.25 N."""
    rows = ed("gripper.json")["slip"]
    got = {r["target_n"]: r["slip_payload_n"] for r in rows}
    assert got == {2.0: 3.75, 3.0: 4.25, 4.0: 5.00, 5.0: 5.25}
    payloads = [r["slip_payload_n"] for r in rows]
    assert payloads == sorted(payloads)      # a firmer grasp holds a larger payload
    for r in rows:
        assert r["slip_payload_n"] > r["grasp_n"], r


# ------------------------------------------------------------------- benchmark

def test_benchmark_report_covers_all_four_tasks():
    b = load("sim/benchmark_results.json")
    assert set(b) == {"reach", "carry", "insert", "insert_m2"}
    for name, node in b.items():
        assert node["summary"]["success_rate"] == "5/5", name    # sim/README table


def test_benchmark_numbers_match_sim_readme_table():
    """sim/README.md benchmark table, 5 trials seed 0."""
    b = load("sim/benchmark_results.json")

    reach = b["reach"]["summary"]["max_err_mm"]               # "max EE error 0.2-0.4 mm"
    assert 0.2 <= reach["min"] and reach["max"] <= 0.4

    carry = b["carry"]["summary"]                             # "carried ~11 cm, tilt 1.8 deg"
    assert 100.0 <= carry["base_carried_mm"]["mean"] <= 120.0
    assert 100.0 <= carry["peg_carried_mm"]["mean"] <= 120.0
    assert carry["peg_tilt_deg"]["max"] == 1.8

    ins = b["insert"]["summary"]                              # "18.7 mm, tilt 1.2-1.4 deg"
    assert round(ins["depth_mm"]["mean"], 1) == 18.7
    assert (ins["peg_tilt_deg"]["min"], ins["peg_tilt_deg"]["max"]) == (1.2, 1.4)
    assert all(not t["aborted"] for t in b["insert"]["trials"])          # "no tau-abort"

    m2 = b["insert_m2"]["summary"]                            # "23.7 mm, 0.7-0.9, 4.7-4.9 N"
    assert round(m2["peg_rel_z_mm"]["mean"], 1) == 23.7
    assert (m2["peg_tilt_deg"]["min"], m2["peg_tilt_deg"]["max"]) == (0.7, 0.9)
    peaks = sorted(round(t["peak_wrench_n"], 1) for t in b["insert_m2"]["trials"])
    assert (peaks[0], peaks[-1]) == (4.7, 4.9), peaks         # the quoted 4.7-4.9 N band
    assert m2["peak_wrench_n"]["max"] < 9.0                              # below the abort
    assert all(not t["aborted"] for t in b["insert_m2"]["trials"])
    assert m2["offset_mm"]["max"] <= 2.5                      # injected residual <=2.5 mm


# ----------------------------------------------------------------- cycle logs

def test_cycle_time_matches_the_published_42_6_s():
    """README / sim/README / ROADMAP / index.html all quote 42.6 s against a 60 s takt."""
    log = load("logs/cycle_001.json")
    t = log[-1]["cycle_time_s"]
    assert round(t, 1) == 42.6, t
    assert t < 60.0                                           # inside the takt target


def test_cycle_runs_the_m2_force_regulated_insert():
    """MANIPULATION.md M2 'Live cycle': S4 is force-regulated, peak wrench ~2.6 N."""
    log = load("logs/cycle_001.json")
    steps = {e["step"] for e in log}
    assert {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"} <= steps
    s4 = [e for e in log if e["step"] == "S4" and "peak_wrench_n" in e]
    assert s4, "S4 must log a measured contact wrench, not a tau baseline"
    assert 2.0 <= s4[-1]["peak_wrench_n"] <= 3.5, s4[-1]      # "~2.6 N", regulated
    assert not any("tau_baseline" in e for e in log)          # the watchdog run is retired
    verdict = [e for e in log if e["step"] == "S5" and "result" in e]
    assert verdict and verdict[-1]["result"] == "ACCEPT"


def test_qc_residuals_match_the_published_1_6_and_3_4_mm():
    """README / sim/README / ROADMAP / index.html quote align +-1.6 mm, depth +-3.4 mm."""
    log = load("logs/cycle_002_camera_qc.json")
    qc = [e for e in log if "residual_align_mm" in e]
    assert len(qc) == 1, "the camera-QC cycle log must carry exactly one verify record"
    qc = qc[0]
    assert round(qc["residual_align_mm"], 1) == 1.6, qc["residual_align_mm"]
    assert round(qc["residual_depth_mm"], 1) == 3.4, qc["residual_depth_mm"]
    # the residual is |camera - oracle| on each axis — recompute it, don't trust the field.
    # 1e-3 tolerance: the log rounds each term to 4 dp *after* the residual is taken.
    assert abs(abs(qc["cam_align_mm"] - qc["oracle_align_mm"])
               - qc["residual_align_mm"]) < 1e-3, qc
    assert abs(abs(qc["cam_depth_mm"] - qc["oracle_depth_mm"])
               - qc["residual_depth_mm"]) < 1e-3, qc


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
