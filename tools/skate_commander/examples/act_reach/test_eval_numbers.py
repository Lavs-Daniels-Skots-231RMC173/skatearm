"""CI guard — the README's ACT-eval numbers must match the committed raw data.

No GPU / MuJoCo / policy needed: this recomputes every published statistic from
the raw per-rollout right/left-hand error arrays in ``eval_data/*.json`` and pins
it to the exact number printed in the top-level README. If a JSON is edited, a
rollout is re-run, or a README figure is changed without the other, this fails —
so the "the headline is one command to check" claim can't silently drift.

Read that literally: this file **opens README.md** and compares the recomputed
statistic against the table cell, character for character. It used to say all of
the above while only ever opening ``eval_data/*.json`` — the README figure lived
in a Python literal next to a comment naming the document, and a comment is not a
check. Editing the table alone kept CI green, which is the one failure the claim
above rules out.

A table is addressed by its whole header row rather than by a line number or a
row label: three tables carry a "Both hands within 8 cm" row and two open with
the same first cell, so only the full header identifies one — and pinning the
header also catches a column being reordered, renamed or dropped.

Run: pytest -q tools/skate_commander/examples/act_reach/test_eval_numbers.py
"""
import html
import json
import math
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.join(HERE, "eval_data")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))


def load(name):
    with open(os.path.join(ED, name), encoding="utf-8") as f:
        return json.load(f)


def _doc(rel):
    """A committed document, flattened to the sentence it publishes.

    An .html page loses its script and style blocks, then its tags, then its
    entities, so a figure reads the way a visitor sees it rather than the way it
    happens to be marked up. Markdown is already prose and is left alone."""
    with open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8") as f:
        txt = f.read()
    if rel.endswith(".html"):
        txt = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", txt)
        txt = html.unescape(re.sub(r"<[^>]+>", " ", txt))
        txt = re.sub(r"[ \t]+", " ", txt)
    return txt


def _readme():
    return _doc("README.md")


def _cells(line):
    """A markdown table row split into its cells, or [] if it is not one."""
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return []
    return [c.strip() for c in line[1:-1].split("|")]


def _table(*header):
    """The README table whose header row is exactly ``header``, as {label: cells}.

    Stops at the first line that is not a row of the same width, so a table ends
    where it ends and cannot silently absorb the next one."""
    lines = _readme().splitlines()
    for i, line in enumerate(lines):
        if _cells(line) != list(header):
            continue
        rows = {}
        for line in lines[i + 1:]:
            cells = _cells(line)
            if len(cells) != len(header):
                break
            if set("".join(cells)) <= set("-: "):
                continue                              # the |---|---| separator
            rows[cells[0]] = cells[1:]
        assert rows, "README.md table %r has no rows left" % (header[0],)
        return rows
    raise AssertionError("README.md no longer has a table headed %r" % (header,))


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
    """§3 Rollout table: ACT (3-seed) vs learned state-only vs mean-pose baselines.

    Every cell is rebuilt from the raw arrays and compared against the cell the
    README actually prints -- bold markers, spacing, the '±' and the unit
    included, because all of those are part of what a reader is shown."""
    s = load("summary.json")
    sb = summarize(load("state_baseline.json")["R"], load("state_baseline.json")["L"])
    t = _table("Reach eval — 24 unseen rollouts · identical targets",
               "ACT (vision)", "State-only (learned, no camera)", "Mean pose (no learning)")

    assert t["Mean reach error — right hand"] == [
        "**%.1f ± %.1f cm**" % (rh1(s["R_mean"]), rh1(s["R_std"])),
        "%.1f cm" % rh1(s["state_R"]),
        "%.1f cm" % rh1(s["base_R"]),
    ], t["Mean reach error — right hand"]
    assert t["Mean reach error — left hand"] == [
        "**%.1f ± %.1f cm**" % (rh1(s["L_mean"]), rh1(s["L_std"])),
        "%.1f cm" % rh1(s["state_L"]),
        "%.1f cm" % rh1(s["base_L"]),
    ], t["Mean reach error — left hand"]
    assert t["Both hands within 8 cm"] == [
        "**%d %% ± %d %%**" % (rh0(s["succ_mean"]), rh0(s["succ_std"])),
        "%d %%" % pct(s["state_succ"]),
        "%d %%" % pct(s["base_succ"]),
    ], t["Both hands within 8 cm"]
    assert t["Median worst hand (pooled)"] == [
        "**%.1f cm**" % rh1(s["pool_median_worst"]),
        "%.1f cm" % cm(sb["median_worsthand"]),
        "—",                                    # no median for a fixed mean pose
    ], t["Median worst hand (pooled)"]


def test_deep_dive_and_landing_page_quote_the_same_headline():
    """The ACT headline is published in three places, and all three are read here.

    docs/deep-dive-act-normalization.md reprints it as the 'after the fix' column
    of its before/after table, and docs/index.html rounds it to one figure in the
    POLICY card. Both are approximations, marked '~' -- so both are checked at the
    precision they chose, which is the only thing '~' can honestly mean."""
    s = load("summary.json")

    dd = _doc("docs/deep-dive-act-normalization.md")
    want = "**~%.1f / %.1f cm**" % (rh1(s["R_mean"]), rh1(s["L_mean"]))
    assert want in dd, f"the deep-dive no longer quotes the headline as {want}"
    want = "**~%d %%**" % rh0(s["succ_mean"])
    assert want in dd, f"the deep-dive no longer quotes the success rate as {want}"

    # the landing page rounds both hands together to a single whole number
    m = re.search(r"~(\d+) cm mean reach error", _doc("docs/index.html"))
    assert m, "docs/index.html no longer states a mean reach error"
    assert int(m.group(1)) == rh0((s["R_mean"] + s["L_mean"]) / 2), m.group(0)


def test_readme_ood_table():
    """OOD table: in-distribution vs out-of-distribution, same checkpoint."""
    oi = summarize(load("ood_indist.json")["R"], load("ood_indist.json")["L"])
    oo = summarize(load("ood.json")["R"], load("ood.json")["L"])
    t = _table("Same checkpoint · 24 rollouts", "In-distribution", "Out-of-distribution")

    assert t["Reach error — right / left"] == [
        "**%.1f / %.1f cm**" % (cm(oi["meanR"]), cm(oi["meanL"])),
        "%.1f / %.1f cm" % (cm(oo["meanR"]), cm(oo["meanL"])),
    ], t["Reach error — right / left"]
    assert t["Both hands within 8 cm"] == [
        "**%d %%** [%d–%d]" % ((pct(oi["success_8cm"]),) + _wilson(*_k(load("ood_indist.json")))),
        "**%d %%** [%d–%d]" % ((pct(oo["success_8cm"]),) + _wilson(*_k(load("ood.json")))),
    ], t["Both hands within 8 cm"]
    # every OOD target was still physically reachable -- that is what makes the
    # drop a generalisation result rather than an unreachable-target artefact
    assert t["Targets IK-reachable < 2 cm"] == ["100 %", "100 %"]


def test_readme_robustness_table():
    """Domain-randomization table: clean vs camera jitter vs full DR."""
    rob = load("robust.json")
    t = _table("Same checkpoint · 24 targets", "clean", "camera jitter", "full DR")
    modes = ("clean", "cam", "dr")

    assert t["Reach error — right / left"] == [
        "%.1f / %.1f cm" % (cm(summarize(rob[m]["R"], rob[m]["L"])["meanR"]),
                            cm(summarize(rob[m]["R"], rob[m]["L"])["meanL"]))
        for m in modes
    ], t["Reach error — right / left"]
    assert t["Both hands within 8 cm"] == [
        "%d %% [%d–%d]" % ((pct(summarize(rob[m]["R"], rob[m]["L"])["success_8cm"]),)
                           + _wilson(*_k(rob[m])))
        for m in modes
    ], t["Both hands within 8 cm"]


def test_readme_dynamic_table():
    """Dynamic table: kinematic (matched robust-clean baseline) vs dynamic mj_step."""
    dyn, clean = load("dynamic.json"), load("robust.json")["clean"]
    dy = summarize(dyn["R"], dyn["L"])
    cl = summarize(clean["R"], clean["L"])
    t = _table("Same checkpoint · 24 rollouts",
               "Kinematic (teleport)", "Dynamic (servos + mj_step)")

    # the kinematic column IS the robustness eval's clean condition -- same
    # targets, same checkpoint -- so it is derived, not re-quoted
    assert t["Reach error — right / left"] == [
        "%.1f / %.1f cm" % (cm(cl["meanR"]), cm(cl["meanL"])),
        "%.1f / %.1f cm" % (cm(dy["meanR"]), cm(dy["meanL"])),
    ], t["Reach error — right / left"]
    assert t["Both hands within 8 cm"] == [
        "%d %% [%d–%d]" % ((pct(cl["success_8cm"]),) + _wilson(*_k(clean))),
        "%d %% [%d–%d]" % ((pct(dy["success_8cm"]),) + _wilson(*_k(dyn))),
    ], t["Both hands within 8 cm"]
    assert t["Unstable / diverged"] == ["—", "%d / %d" % (dyn["diverged"], dy["n"])]


def _wilson(k, n, z=1.96):
    """Closed-form Wilson score interval for a binomial proportion (deterministic)."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return rh0(100 * (center - half)), rh0(100 * (center + half))


def _k(node):
    worst = [max(a, b) for a, b in zip(node["R"], node["L"])]
    return sum(1 for w in worst if w < 0.08), len(worst)


def test_readme_success_wilson_ci():
    """Every bracketed range in the README is a 95 % Wilson interval — all of them.

    The tables above already check the intervals in the rows they own. This one
    is the completeness half: it sweeps the README for ``[lo–hi]`` and requires
    each one to be an interval some pinned condition actually produces, and each
    pinned condition's interval to still be published. Add a stress-test column
    with a hand-typed CI and this goes red — an unpinned interval is
    indistinguishable from an invented one."""
    rob = load("robust.json")
    derived = {
        "robust/clean": _wilson(*_k(rob["clean"])),
        "robust/cam": _wilson(*_k(rob["cam"])),
        "robust/dr": _wilson(*_k(rob["dr"])),
        "ood_indist": _wilson(*_k(load("ood_indist.json"))),
        "ood": _wilson(*_k(load("ood.json"))),
        "dynamic": _wilson(*_k(load("dynamic.json"))),
    }
    assert derived == {
        "robust/clean": (51, 85), "robust/cam": (21, 57), "robust/dr": (24, 61),
        "ood_indist": (47, 82), "ood": (0, 14), "dynamic": (69, 96),
    }, derived

    printed = [(int(a), int(b)) for a, b
               in re.findall(r"\[(\d+)–(\d+)\]", _readme())]
    assert len(printed) == 7, ("README.md prints %d confidence intervals, "
                               "expected the 7 the tables above own" % len(printed))
    unknown = sorted({p for p in printed} - set(derived.values()))
    assert not unknown, f"README.md prints intervals no eval derives: {unknown}"
    missing = sorted(n for n, v in derived.items() if v not in printed)
    assert not missing, f"README.md no longer publishes the interval for: {missing}"
