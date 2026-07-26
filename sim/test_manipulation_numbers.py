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
    python sim/eval_wrench_backends.py --model .../skt_v3 \
        --json sim/eval_data/wrench_backends.json
    MUJOCO_GL=egl python sim/eval_qc_occlusion.py --model .../skt_v3 \
        --json sim/eval_data/qc_occlusion.json     # needs GL + the --gripper scene
    python sim/benchmark.py       --model .../skt_v3 --trials 5 --seed 0 \
        --json sim/benchmark_results.json
    python sim/demo_cell_cycle.py --model .../skt_v3 --no-render --log logs/cycle_001.json

If an eval is re-run and a published figure is not updated with it — or a figure
is edited without the data — this test fails. That is the point: it is the same
contract as ``tools/skate_commander/examples/act_reach/test_eval_numbers.py``.

Run: pytest -q sim/test_manipulation_numbers.py
"""
import ast
import glob
import json
import os
import re

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


# --------------------------------------------------------------------------- M1

def test_wrench_backend_sensor_is_exact_at_both_poses():
    """MANIPULATION.md M1: the wrist sensor reads every load to 0.000 N at both
    poses — the printed form of the < 0.05 N bound sim/test_ft_sensor.py asserts.
    Checked per row, not as an average: delta must be exactly -F."""
    d = ed("wrench_backends.json")
    assert d["loads_n"] == [[0, 0, -10], [10, 0, 0], [0, 8, 0], [5, -5, -5]]
    for pose in ("home", "working"):
        for arm, a in d["poses"][pose]["arms"].items():
            assert a["sensor_err_max_n"] == 0.0, (pose, arm)
            for r in a["rows"]:
                assert r["sensor_delta_n"] == [-v for v in r["load_n"]], (pose, arm, r)
            # untared: the no-load reading is the hand's own weight, ~0.4 N
            assert round(a["baseline_sensor_n"][2], 1) == 0.4, (pose, arm)


def test_wrench_backend_estimate_is_the_weaker_fallback():
    """MANIPULATION.md M1 table (also quoted in README.md and sim/README.md):
    at the near-singular home pose (sigma_min 0.0205) the joint-torque estimate
    misses a 10 N vertical pull by 9.699 N and invents 26.550 N of phantom force
    under an 8 N lateral one; at the working pose (0.0770) it is still
    0.760-4.262 N off on 10 N-class loads."""
    p = ed("wrench_backends.json")["poses"]

    home = p["home"]["arms"]
    assert {a["sigma_min_j"] for a in home.values()} == {0.0205}
    for arm in ("left", "right"):
        rows = {tuple(r["load_n"]): r for r in home[arm]["rows"]}
        assert rows[(0.0, 0.0, -10.0)]["estimate_err_n"] == 9.699, arm
        assert rows[(0.0, 8.0, 0.0)]["estimate_err_n"] == 26.550, arm
        # 0.31 N recovered of a 10 N pull — the quoted 97% miss
        assert round(rows[(0.0, 0.0, -10.0)]["estimate_delta_n"][2], 2) == 0.31, arm
        # and the phantom force is on the vertical axis, not the loaded one
        assert round(rows[(0.0, 8.0, 0.0)]["estimate_delta_n"][2], 1) == 26.5, arm

    work = p["working"]["arms"]
    assert {a["sigma_min_j"] for a in work.values()} == {0.0770}
    errs = [r["estimate_err_n"] for a in work.values() for r in a["rows"]]
    assert (min(errs), max(errs)) == (0.760, 4.262), errs

    # the headline: better conditioning helps a lot and still is not enough
    assert max(a["estimate_err_max_n"] for a in work.values()) < \
        max(a["estimate_err_max_n"] for a in home.values())
    assert min(errs) > max(a["sensor_err_max_n"] for a in work.values())


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


def test_qc_occlusion_pixel_counts_match_the_published_figures():
    """MANIPULATION.md M4 / README / sim/README / ROADMAP / index.html:
    inside the 300 px inspection ROI at 960x720, the weld path shows 1116 peg px
    and 7581 pocket-rim px; the jaw path shows 0 and 827 -- 89 % of the rim gone."""
    d = ed("qc_occlusion.json")
    assert d["qc"]["render_px"] == [960, 720]     # "at its calibration resolution"
    assert d["qc"]["roi_px"] == 300               # "the 300 px inspection ROI"

    px = {k: v["px"] for k, v in d["paths"].items()}
    assert (px["weld"]["top_peg"], px["weld"]["top_rim"]) == (1116, 7581)
    assert (px["jaws"]["top_peg"], px["jaws"]["top_rim"]) == (0, 827)

    # recompute the loss from the raw counts -- don't trust the summary field
    loss = 100.0 * (1.0 - px["jaws"]["top_rim"] / px["weld"]["top_rim"])
    assert round(loss, 1) == d["summary"]["top_rim_loss_pct"] == 89.1
    assert round(loss) == 89, loss                # the "89 %" the docs print


def test_qc_occlusion_rejects_a_unit_the_oracle_still_calls_good():
    """index.html quotes 'ACCEPT on the sim oracle' beside the camera's REJECT,
    and MANIPULATION.md's whole claim is that the conversion cost the SIGHT of
    the part, not the assembly. So: the camera flips ACCEPT -> REJECT while the
    oracle accepts BOTH paths on the same seated part.

    Both verdicts are re-derived from qc.verdict's own thresholds (carried in
    the artefact) rather than read out of the ``verdict`` field."""
    d = ed("qc_occlusion.json")
    t = d["qc"]["accept_thresholds"]
    assert t == {"depth_min": 15.0, "align_max": 6.0, "tilt_max": 8.0}

    def ok(depth, align, tilt):
        return (depth is not None and depth >= t["depth_min"]
                and align is not None and align <= t["align_max"]
                and (tilt is None or tilt <= t["tilt_max"]))

    w, j = d["paths"]["weld"], d["paths"]["jaws"]
    assert (w["camera"]["verdict"], j["camera"]["verdict"]) == ("ACCEPT", "REJECT")

    # the camera: the weld path sees a peg and clears both limits ...
    assert w["camera"]["peg_present"] is True
    assert ok(w["camera"]["depth_mm_est"], w["camera"]["align_err_mm"], None)
    assert round(w["camera"]["align_err_mm"], 2) == 3.40      # "align 3.40 mm"
    assert round(w["camera"]["depth_mm_est"], 2) == 19.02     # "depth 19.02 mm"
    # ... and the jaw path fails at PRESENCE, which is what makes align None
    assert j["camera"]["peg_present"] is False
    assert j["camera"]["align_err_mm"] is None                # "align_err_mm None"

    # the oracle: same seated part on both paths, accepted on both
    assert w["oracle"]["depth_mm"] == j["oracle"]["depth_mm"] == 22.1209
    for tag, p in (("weld", w), ("jaws", j)):
        o = p["oracle"]
        assert ok(o["depth_mm"], o["align_mm"], o["tilt_deg"]), (tag, o)
    # the assembly the cameras disagree about is the M4 headline unit: 22.12 mm
    # at 1.90 deg, alignment 1.24 mm -- MANIPULATION.md's "Final unit"
    assert round(j["oracle"]["tilt_deg"], 2) == 1.90
    assert round(j["oracle"]["align_mm"], 2) == 1.24


def test_qc_occlusion_is_the_cell_and_not_the_measurement():
    """MANIPULATION.md M4: 'Same probe, same qc.py, same masks, same renderer on
    both paths, so the REJECT is caused by the conversion and not by the
    measurement.' That sentence is only true if the two columns really do differ
    in exactly one thing -- the cell -- so check the rest is held fixed."""
    d = ed("qc_occlusion.json")
    w, j = d["paths"]["weld"], d["paths"]["jaws"]

    # one renderer config, one ROI, one set of masks, shared by both columns
    assert d["qc"]["masks"]["top_peg"].startswith("qc._yellow")
    assert d["qc"]["masks"]["top_rim"].startswith("qc._cyan")
    # the frame the cell's own S5 judged, not a re-render staged for the doc
    assert "not a re-render" in d["qc"]["measured_by"]

    # the cells differ in the one way the claim is about, and only there
    assert (w["scene"], j["scene"]) == ("skt_v3_cell.xml", "skt_v3_cell_gripper.xml")
    assert (w["jaws_right"], w["jaws_left"]) == (False, False)
    assert (j["jaws_right"], j["jaws_left"]) == (True, True)
    assert w["mm_per_px"]["side"] == j["mm_per_px"]["side"]     # same camera geometry
    assert abs(w["mm_per_px"]["top"] - j["mm_per_px"]["top"]) < 0.01

    # the subject is NOT in exactly the same place -- two cells settle the part
    # differently -- so bound it: 16 px of a 300 px window cannot take 1116 peg
    # px to 0, which is the whole reason the offset is published rather than
    # rounded away.
    assert d["summary"]["unit_pose_delta_mm"] == 7.74
    assert d["summary"]["unit_pose_delta_top_px"] == 16.0
    assert d["summary"]["unit_pose_delta_top_px"] < 0.1 * d["qc"]["roi_px"]

    # the weld column is the published reference cycle, independently re-run:
    # MANIPULATION.md's "75.84 s against the weld path's 42.58 s"
    assert round(w["cycle_time_s"], 2) == 42.58
    assert round(w["cycle_time_s"], 1) == round(load("logs/cycle_001.json")[-1]["cycle_time_s"], 1)
    # the jaws column runs LONGER than that published 75.84 s takt figure, and
    # must: a camera REJECT sends S6 to the far reject bin, while the 75.84 s is
    # the oracle-gated ACCEPT branch sim/test_cell_gripper.py runs with no
    # renderer attached. Same cycle, different S6 branch -- which branch it takes
    # is precisely what this eval measures.
    assert j["cycle_time_s"] > 75.84


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


# ------------------------------------------------------- the site's own counters

def _site_counter(label):
    """The figure docs/index.html prints beside ``label`` in its 'recomputed from
    raw data' panel."""
    with open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"<div><b>(\d+)</b><span>" + re.escape(label) + r"</span></div>", html)
    assert m, f"docs/index.html no longer prints a '{label}' counter this test can read"
    return int(m.group(1))


def _this_file():
    with open(os.path.join(HERE, "test_manipulation_numbers.py"), encoding="utf-8") as f:
        return ast.parse(f.read())


def _artefacts_read(tree):
    """Every raw-data path this file pins a number to, taken from its own AST:
    the literal argument of each ``load()`` / ``ed()`` call, repo-relative.

    Literals only, and deliberately so -- a path assembled at runtime could not
    be counted, so the assert below is what keeps the count honest rather than
    approximate. It also means no helper in this file may call ``load()`` or
    ``ed()`` with a variable."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("load", "ed"):
            arg = n.args[0]
            assert isinstance(arg, ast.Constant), \
                "artefact paths must stay string literals so they can be counted"
            out.add(arg.value if n.func.id == "load"
                    else "sim/eval_data/" + arg.value)
    return out


def test_site_counters_are_derived_and_not_typed():
    """docs/index.html prints 'N checks in CI' and 'M raw data files' under the
    headline 'Every number above is recomputed from raw data, on every push.'

    Those two are themselves published figures, and they had drifted -- which is
    the exact failure the panel claims cannot happen. So they get the same
    treatment as everything else on the page: counted here, off this file's own
    AST, rather than typed. Add a pin or an artefact without updating the panel
    and CI goes red."""
    tree = _this_file()
    tests = [n.name for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    data = _artefacts_read(tree)

    assert _site_counter("checks in CI") == len(tests), \
        f"docs/index.html says {_site_counter('checks in CI')} checks; this file has {len(tests)}"
    assert _site_counter("raw data files") == len(data), \
        f"docs/index.html says {_site_counter('raw data files')} data files; " \
        f"this file reads {len(data)}: {sorted(data)}"
    # the third counter is the promise the other two are making
    assert _site_counter("hand-typed stats") == 0


def test_every_committed_artefact_is_actually_pinned():
    """The counter above is only meaningful if 'raw data files' means every
    artefact in the tree, not just the ones someone remembered to pin. An eval
    that writes a committed JSON nothing re-derives is a number with no guard --
    so committing one without a pin fails here."""
    read = _artefacts_read(_this_file())
    pinned = {p for p in read if p.startswith("sim/eval_data/")}
    on_disk = {"sim/eval_data/" + os.path.basename(p)
               for p in glob.glob(os.path.join(ED, "*.json"))}
    assert on_disk == pinned, (
        f"unpinned artefacts: {sorted(on_disk - pinned)}; "
        f"pinned but missing: {sorted(pinned - on_disk)}")

    # And every path counted -- the eval artefacts plus the benchmark report and
    # the two cycle logs -- must in fact resolve and parse as the JSON it claims
    # to be, so a counted path can never be one that no longer exists. Opened
    # directly rather than through load()/ed(): a variable argument here would
    # trip the literal check in _artefacts_read, which is the check doing the
    # counting.
    for rel in sorted(read):
        with open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8") as f:
            assert isinstance(json.load(f), (dict, list)), rel


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
