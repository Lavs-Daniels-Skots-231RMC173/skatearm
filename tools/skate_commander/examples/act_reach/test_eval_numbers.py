"""CI guard — the README's ACT-eval numbers must match the committed raw data.

No GPU / MuJoCo / policy needed: this recomputes every published statistic from
the raw per-rollout right/left-hand error arrays in ``eval_data/*.json`` and pins
it to the exact number printed in the top-level README. If a JSON is edited, a
rollout is re-run, or a README figure is changed without the other, this fails —
so the "the headline is one command to check" claim can't silently drift.

Run: pytest -q tools/skate_commander/examples/act_reach/test_eval_numbers.py
"""
import json
import math
import os
import statistics as st

ED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data")


def load(name):
    with open(os.path.join(ED, name), encoding="utf-8") as f:
        return json.load(f)


def summarize(R, L):
    """Recompute the published summary from raw per-rollout arrays."""
    n = len(R)
    worst = [max(a, b) for a, b in zip(R, L)]
    return {
        "n": n,
        "meanR": sum(R) / n,
        "meanL": sum(L) / n,
        "success_8cm": sum(1 for w in worst if w < 0.08) / n,
        "median_worsthand": st.median(worst),
    }


# round half up, matching how the README displays numbers
def rh1(x):
    return math.floor(x * 10 + 0.5) / 10        # 1 decimal


def rh0(x):
    return math.floor(x + 0.5)                  # integer


def cm(m):
    return rh1(m * 100)                          # metres -> cm @ README precision


def pct(f):
    return rh0(f * 100)                          # fraction -> integer %


def _rl_nodes():
    """Every eval node that carries raw R/L arrays + a stored summary."""
    for nm in ("seed0", "seed1", "seed2", "state_baseline", "baseline",
               "ood_indist", "ood", "dynamic"):
        yield nm, load(nm + ".json")
    rob = load("robust.json")
    for mode in ("clean", "cam", "dr"):
        yield "robust/" + mode, rob[mode]


def test_internal_consistency():
    """Every JSON's stored summary must equal a recompute from its own R/L arrays."""
    for name, node in _rl_nodes():
        s = summarize(node["R"], node["L"])
        assert node["n"] == s["n"], (name, "n")
        assert abs(node["meanR"] - s["meanR"]) < 1e-9, (name, "meanR")
        assert abs(node["meanL"] - s["meanL"]) < 1e-9, (name, "meanL")
        assert abs(node["success_8cm"] - s["success_8cm"]) < 1e-9, (name, "success_8cm")
        assert abs(node["median_worsthand"] - s["median_worsthand"]) < 1e-9, (name, "median")


def test_summary_matches_seeds():
    """summary.json (the headline aggregate) must equal a recompute from the 3 seeds."""
    s = load("summary.json")
    seeds = [load("seed%d.json" % i) for i in range(3)]
    seedR = [x["meanR"] * 100 for x in seeds]
    seedL = [x["meanL"] * 100 for x in seeds]
    seedSucc = [x["success_8cm"] * 100 for x in seeds]
    assert abs(s["R_mean"] - st.mean(seedR)) < 1e-9
    assert abs(s["R_std"] - st.stdev(seedR)) < 1e-9        # sample std (n-1), seed-to-seed spread
    assert abs(s["L_mean"] - st.mean(seedL)) < 1e-9
    assert abs(s["L_std"] - st.stdev(seedL)) < 1e-9
    assert abs(s["succ_mean"] - st.mean(seedSucc)) < 1e-9
    assert abs(s["succ_std"] - st.stdev(seedSucc)) < 1e-9
    pooled = []
    for x in seeds:
        pooled += [max(a, b) for a, b in zip(x["R"], x["L"])]
    assert abs(s["pool_median_worst"] - st.median(pooled) * 100) < 1e-9


def test_readme_headline_and_baselines():
    """§3 Rollout table: ACT (3-seed) vs learned state-only vs mean-pose baselines."""
    s = load("summary.json")
    # ACT (vision): "5.6 ± 0.6 cm" / "5.2 ± 0.3 cm" / "69 % ± 9 %" / median worst "6.9 cm"
    assert (rh1(s["R_mean"]), rh1(s["R_std"])) == (5.6, 0.6)
    assert (rh1(s["L_mean"]), rh1(s["L_std"])) == (5.2, 0.3)
    assert (rh0(s["succ_mean"]), rh0(s["succ_std"])) == (69, 9)
    assert rh1(s["pool_median_worst"]) == 6.9
    # State-only (learned, no camera): 13.7 / 16.5 cm, 0 %, median worst 17.4
    assert (rh1(s["state_R"]), rh1(s["state_L"]), pct(s["state_succ"])) == (13.7, 16.5, 0)
    sb = summarize(load("state_baseline.json")["R"], load("state_baseline.json")["L"])
    assert cm(sb["median_worsthand"]) == 17.4
    # Mean pose (no learning): 19.6 / 19.7 cm, 0 %
    assert (rh1(s["base_R"]), rh1(s["base_L"]), pct(s["base_succ"])) == (19.6, 19.7, 0)


def test_readme_ood_table():
    """OOD table: in-distribution vs out-of-distribution, same checkpoint."""
    oi = summarize(load("ood_indist.json")["R"], load("ood_indist.json")["L"])
    assert (cm(oi["meanR"]), cm(oi["meanL"]), pct(oi["success_8cm"])) == (5.6, 5.2, 67)
    oo = summarize(load("ood.json")["R"], load("ood.json")["L"])
    assert (cm(oo["meanR"]), cm(oo["meanL"]), pct(oo["success_8cm"])) == (16.8, 12.9, 0)


def test_readme_robustness_table():
    """Domain-randomization table: clean vs camera jitter vs full DR."""
    rob = load("robust.json")
    exp = {"clean": (5.7, 5.0, 71), "cam": (7.2, 7.4, 38), "dr": (7.3, 7.4, 42)}
    for mode, want in exp.items():
        s = summarize(rob[mode]["R"], rob[mode]["L"])
        assert (cm(s["meanR"]), cm(s["meanL"]), pct(s["success_8cm"])) == want, mode


def test_readme_dynamic_table():
    """Dynamic table: kinematic (matched robust-clean baseline) vs dynamic mj_step."""
    dy = summarize(load("dynamic.json")["R"], load("dynamic.json")["L"])
    assert (cm(dy["meanR"]), cm(dy["meanL"]), pct(dy["success_8cm"])) == (5.2, 4.6, 88)
    assert load("dynamic.json")["diverged"] == 0
    # kinematic column is the clean condition of the robustness eval (same targets/checkpoint)
    clean = summarize(load("robust.json")["clean"]["R"], load("robust.json")["clean"]["L"])
    assert (cm(clean["meanR"]), cm(clean["meanL"]), pct(clean["success_8cm"])) == (5.7, 5.0, 71)
