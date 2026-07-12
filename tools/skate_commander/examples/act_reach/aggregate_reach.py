"""Aggregate per-seed ACT rollout metrics into the headline result.

Reads the ``rollout_metrics.json`` files produced by ``rollout_act.py`` (one per
training seed), the ``baseline_metrics.json`` from ``baseline_reach.py`` (the
fixed mean-pose control) and the ``state_baseline_metrics.json`` from
``state_baseline_reach.py`` (the learned state-only control), computes the
mean +/- std across seeds, prints the results, writes ``summary.json``, and
redraws the accuracy chart (light + dark variants) -- the exact numbers and
figure used in the top-level README.

    python aggregate_reach.py seed0.json seed1.json seed2.json baseline.json state_baseline.json

Filename rules: an arg containing "state" is the learned state-only baseline; an
arg containing "baseline" (but not "state") is the mean-pose baseline; the rest
are per-seed ACT rollouts. Charts go to ``OUT_DIR`` (default: the repo's
``docs/img/act/``); ``summary.json`` is written next to the first input.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(REPO, "docs", "img", "act"))

args = sys.argv[1:]
if not args:
    sys.exit("usage: python aggregate_reach.py <seed*.json> ... <baseline.json> [state_baseline.json]")


def bn(a):
    return os.path.basename(a).lower()


state_files = [a for a in args if "state" in bn(a)]
base_files = [a for a in args if "baseline" in bn(a) and "state" not in bn(a)]
seed_files = [a for a in args if a not in state_files and a not in base_files]

seeds = [json.load(open(f)) for f in seed_files]
sR = np.array([s["meanR"] for s in seeds]) * 100
sL = np.array([s["meanL"] for s in seeds]) * 100
sSucc = np.array([s["success_8cm"] for s in seeds]) * 100
poolR = np.concatenate([np.array(s["R"]) for s in seeds]) * 100
poolL = np.concatenate([np.array(s["L"]) for s in seeds]) * 100
n_eval = len(seeds[0]["R"])


def std(a):
    return float(a.std(ddof=1)) if len(a) > 1 else 0.0


summary = {
    "seeds": len(seeds), "n_eval": n_eval,
    "R_mean": float(sR.mean()), "R_std": std(sR),
    "L_mean": float(sL.mean()), "L_std": std(sL),
    "succ_mean": float(sSucc.mean()), "succ_std": std(sSucc),
    "seedR": sR.tolist(), "seedL": sL.tolist(), "seedSucc": sSucc.tolist(),
    "pool_median_worst": float(np.median(np.maximum(poolR, poolL))),
}
base = json.load(open(base_files[0])) if base_files else None
if base:
    summary.update(base_R=base["meanR"] * 100, base_L=base["meanL"] * 100,
                   base_succ=base["success_8cm"] * 100)
state = json.load(open(state_files[0])) if state_files else None
if state:
    summary.update(state_R=state["meanR"] * 100, state_L=state["meanL"] * 100,
                   state_succ=state["success_8cm"] * 100)

json.dump(summary, open(os.path.join(os.path.dirname(seed_files[0]), "summary.json"), "w"), indent=2)
print("ACT    R %.1f +/- %.1f cm   L %.1f +/- %.1f cm   success@8cm %.0f +/- %.0f%%   (%d seeds x %d)"
      % (summary["R_mean"], summary["R_std"], summary["L_mean"], summary["L_std"],
         summary["succ_mean"], summary["succ_std"], len(seeds), n_eval))
if state:
    print("STATE  R %.1f cm   L %.1f cm   success@8cm %.0f%%" % (summary["state_R"], summary["state_L"], summary["state_succ"]))
if base:
    print("MEAN   R %.1f cm   L %.1f cm   success@8cm %.0f%%" % (summary["base_R"], summary["base_L"], summary["base_succ"]))


def draw(dark):
    if dark:
        BG, FG, GRID, blue, gray, violet = "#0d1117", "#c9d1d9", "#8b949e", "#58a6ff", "#8b949e", "#a78bfa"
    else:
        BG, FG, GRID, blue, gray, violet = "white", "#1f2328", "#c9ccd1", "#2563EB", "#9BA2AC", "#7C3AED"
    orange = "#F59E0B"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID,
                         "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
                         "ytick.color": FG, "axes.facecolor": BG, "figure.facecolor": BG,
                         "savefig.facecolor": BG})
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=150)
    ax.axhspan(0, 3, color="#16a34a", alpha=0.20 if dark else 0.12, zorder=0)
    ax.text(1.86, 3.15, "3 cm marker radius", fontsize=7.5, color=FG, alpha=0.75, ha="right", va="bottom")
    ax.axhline(8, color=FG, ls=":", lw=1.1, alpha=0.55)
    ax.text(1.86, 8.25, "8 cm success threshold", fontsize=8, color=FG, alpha=0.75, ha="right")
    x = np.array([0.0, 1.35])
    w = 0.28
    actm, acts = [sR.mean(), sL.mean()], [std(sR), std(sL)]
    ax.bar(x - w, actm, w, yerr=acts, capsize=5, color=[orange, blue],
           error_kw=dict(ecolor=FG, lw=1.3), label="ACT policy (%d seeds)" % len(seeds), zorder=3)
    if state:
        ax.bar(x, [summary["state_R"], summary["state_L"]], w, color=violet, alpha=0.9,
               edgecolor=GRID, label="state-only (learned)", zorder=3)
    if base:
        ax.bar(x + w, [summary["base_R"], summary["base_L"]], w, color=gray, alpha=0.55,
               hatch="//", edgecolor=GRID, label="mean pose (no vision)", zorder=3)
    ax.scatter(np.full(len(seeds), x[0] - w), sR, s=26, color=BG, edgecolor=FG, lw=1.1, zorder=5)
    ax.scatter(np.full(len(seeds), x[1] - w), sL, s=26, color=BG, edgecolor=FG, lw=1.1, zorder=5)
    for xi, mv in zip(x - w, actm):
        ax.text(xi, mv + max(acts) + 0.5, "%.1f" % mv, ha="center", fontsize=9, fontweight="bold", color=FG)
    if state:
        for xi, mv in zip(x, [summary["state_R"], summary["state_L"]]):
            ax.text(xi, mv + 0.4, "%.1f" % mv, ha="center", fontsize=9, color=FG, alpha=0.9)
    if base:
        for xi, mv in zip(x + w, [summary["base_R"], summary["base_L"]]):
            ax.text(xi, mv + 0.4, "%.1f" % mv, ha="center", fontsize=9, color=FG, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["right hand (orange target)", "left hand (blue target)"])
    ax.set_ylabel("final reach error (cm)")
    ax.set_xlim(-0.55, 1.95)
    allv = actm + [summary.get("state_R", 0), summary.get("state_L", 0),
                   summary.get("base_R", 0), summary.get("base_L", 0)]
    ax.set_ylim(0, max(allv) * 1.22)
    ax.set_title("Reach error: vision vs two no-vision baselines  (%d seeds x %d unseen rollouts, mean +/- std)"
                 % (len(seeds), n_eval), fontsize=10, fontweight="bold", color=FG)
    ax.grid(axis="y", alpha=0.18, color=GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", labelcolor=FG, ncol=3)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "accuracy%s.png" % ("_dark" if dark else ""))
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print("WROTE", out)


draw(False)
draw(True)
