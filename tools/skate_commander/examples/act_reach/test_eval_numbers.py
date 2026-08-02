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

The training-config block at the end is a different kind of check, and worth
naming as such. There is no committed artefact for the training run — the loss
curve is a PNG — so nothing here re-derives 52 M parameters or a 32-minute
wall-clock from data, and this file does not pretend otherwise. What it does
instead is make the README's config table the single source: every other copy of
those figures, across four documents, is read against the table cell, and then
the tree is swept for a copy nobody pinned. That is the drift these particular
figures actually suffer — a table edited and a landing-page stat card left
behind — and it is checkable without a derivation the repo cannot honestly make.

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

RM = "README.md"
ARM = "tools/skate_commander/examples/act_reach/README.md"
IDX = "docs/index.html"
DD = "docs/deep-dive-act-normalization.md"
# this file, as the tree walk below sees it -- derived rather than typed, so
# moving the guard cannot quietly turn the checker into one of the documents it
# is supposed to be checking
GUARD = os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, "/")


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


_FLAT = {}


def _flat(rel):
    """``_doc`` again, flattened the rest of the way, for reading a figure out of
    running prose rather than out of a table cell.

    Every run of whitespace becomes one space, so a sentence that wrapped at a
    column, a row of a markdown table and a stat card on the landing page all
    read alike. Markdown gets its entities back here — ``_doc`` deliberately does
    not, because the table checks above compare a cell character for character —
    which is what makes ``52&nbsp;M`` read as the ``52 M`` a reader is shown.

    Cached: ``_quotes`` below reads every file in the tree once per figure, and
    the flattening it applies is the same one every time."""
    if rel in _FLAT:
        return _FLAT[rel]
    txt = _doc(rel)
    if rel.endswith(".md"):
        txt = html.unescape(txt)
    _FLAT[rel] = txt = re.sub(r"\s+", " ", txt)
    return txt


def _readme():
    return _doc(RM)


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


def _tree():
    """Every committed file that could publish a figure.

    This one is skipped: it is the checker, not a publisher, and its docstrings
    quote the figures it guards by design."""
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for name in names:
            if name.rsplit(".", 1)[-1] not in ("py", "md", "html", "txt", "yml"):
                continue
            rel = os.path.relpath(os.path.join(base, name), ROOT).replace(os.sep, "/")
            if rel != GUARD:
                yield rel


def _quotes(fig, unit):
    """Which committed files PRINT ``fig unit``, and how many times each.

    Carried WITH its unit, because the numeral on its own cannot tell 4 GB of
    laptop GPU from the 4 of batch size. The separator is one-or-more whitespace
    or star, so ``**52 M**`` in markdown and ``52&nbsp;M`` on the page are the
    same quotation while a stylesheet's ``padding:4px`` is not one at all.

    A figure whose unit is printed in FRONT of it cannot be carried here — the
    final L1 loss is written "L1 loss 0.070", and a bare 0.070 with nothing
    behind it is indistinguishable from any other. That one is pinned copy by
    copy instead, and the test below says so."""
    pat = re.compile(r"(?<![\d.])" + re.escape(fig) + r"(?![\d])[\s*]+"
                     + unit + r"(?!\w)")
    out = {}
    for rel in _tree():
        hits = len(pat.findall(_flat(rel)))
        if hits:
            out[rel] = hits
    return out


def _reads(rel, pattern):
    """The figure a document prints at ``pattern``, as it prints it.

    The assert is half the value: delete the sentence and the pin fails loudly
    instead of quietly having nothing left to check."""
    m = re.search(pattern, _flat(rel))
    assert m, f"{rel} no longer states the figure this test reads: {pattern}"
    return m.group(1)


def _echoes(fig, unit, *quotes):
    """Pin every sentence that reprints ``fig unit`` -- and prove there is no other.

    ``quotes`` is the (document, pattern) list that publishes the figure, the
    README's own config table included. Each pattern has to still find its
    sentence and that sentence has to still print ``fig``, so a table edited
    without its copies goes red, and so does a copy edited away from the table.

    Then the pins are turned around and used as a census: what the tree prints
    has to be exactly what the pins cover. Without that half, a fifth copy nobody
    named is free to go stale, because nothing fails when it does -- which is the
    whole reason these figures needed guarding rather than a comment.

    Two patterns landing on the SAME sentence count as two pins against one
    quotation, which fails: a pin has to name a copy of its own."""
    pinned = {}
    for rel, pattern in quotes:
        got = _reads(rel, pattern)
        assert got == fig, \
            f"{rel} prints '{got} {unit}' where the config table says '{fig} {unit}'"
        pinned[rel] = pinned.get(rel, 0) + 1
    got = _quotes(fig, unit)
    assert got == pinned, \
        f"'{fig} {unit}' is printed {got} and pinned {pinned} -- a copy no pin " \
        f"names is a copy nothing fails for"
    return fig


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

    dd = _doc(DD)
    want = "**~%.1f / %.1f cm**" % (rh1(s["R_mean"]), rh1(s["L_mean"]))
    assert want in dd, f"the deep-dive no longer quotes the headline as {want}"
    want = "**~%d %%**" % rh0(s["succ_mean"])
    assert want in dd, f"the deep-dive no longer quotes the success rate as {want}"

    # the landing page rounds both hands together to a single whole number
    m = re.search(r"~(\d+) cm mean reach error", _doc(IDX))
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


def _cfg(table, label, pattern):
    """One figure out of the Training config table, as the cell prints it."""
    cell = table[label][0]
    m = re.search(pattern, cell)
    assert m, (f"README.md's config table prints {label!r} as {cell!r}, which no "
               f"longer carries the figure this test reads: {pattern}")
    return m.group(1)


def test_readme_training_config_is_self_consistent():
    """§2 Training config: the table's arithmetic, and the README that leans on it.

    Nothing here is typed twice — every figure is read out of the config table
    and then checked against the sentences that depend on it. The step count and
    the wall-clock imply the throughput the same row prints in brackets. The peak
    VRAM has to fit both the ceiling the prose claims and the card the table
    names. The loss curve is a PNG, so its alt text is the only place a reader
    without eyes on the image learns what it shows: it has to quote the table's
    own final loss and step count.

    The ceiling is checked in BOTH directions. "0.62 is under 1" still passes if
    the prose quietly widens its own claim to 4 GB, so the ceiling also has to be
    TIGHT — the next whole gigabyte up. A page may round 0.62 to a clean 1; it may
    not round it into a different claim."""
    t = _table("Setting", "Value")
    card = int(_cfg(t, "Hardware", r"RTX 3050 Laptop · (\d+) GB\*\*"))
    batch = _cfg(t, "Batch · steps", r"^(\d+) ·")
    steps = int(_cfg(t, "Batch · steps", r"· \*\*([\d ]+)\*\*$").replace(" ", ""))
    mins = int(_cfg(t, "Wall-clock", r"^\*\*≈ (\d+) min\*\*"))
    rate = int(_cfg(t, "Wall-clock", r"\(~(\d+) steps/s\)$"))
    peak = float(_cfg(t, "Peak VRAM", r"^\*\*([\d.]+) GB\*\*$"))
    loss = _cfg(t, "Final L1 loss", r"^\*\*([\d.]+)\*\*$")
    short = "%dk" % (steps // 1000)

    assert rate == rh0(steps / (mins * 60)), (
        "%d steps in %d min is %.2f steps/s, not the ~%d the table prints"
        % (steps, mins, steps / (mins * 60), rate))
    assert steps == int(short[:-1]) * 1000, \
        f"{steps} steps abbreviates to {short}, which is a different number"

    ceiling = int(_reads(RM, r"sits comfortably under \*\*(\d+) GB\*\* of VRAM"))
    assert peak <= ceiling, \
        f"the README claims the run sits under {ceiling} GB and the table prints {peak} GB"
    assert math.ceil(peak) == ceiling, \
        f"{peak} GB rounds up to {math.ceil(peak)} GB, not the {ceiling} GB claimed"
    assert peak < card, f"{peak} GB does not fit the {card} GB card the table names"

    m = re.search(r'alt="ACT training loss curve — ([\d.]+) to ([\d.]+) over (\d+k) steps">',
                  _flat(RM))
    assert m, "README.md's loss curve no longer describes the run in its alt text"
    assert (m.group(2), m.group(3)) == (loss, short), m.group(0)
    assert _reads(RM, r"Batch (\d+) holds peak VRAM at") == batch


def test_training_figures_are_published_consistently():
    """Every document that reprints a training figure is read against the table.

    The config table is the source. The landing page prints the parameter count
    and the wall-clock on stat cards and again in its lead; the deep-dive quotes
    the parameter count, the card and the step count; the example README quotes
    the wall-clock and the step count in the train command it tells you to run.
    Each of those is pinned to the table cell, and then ``_echoes`` sweeps the
    tree for a copy no pin names.

    The final L1 loss is pinned but NOT censused, because it prints its unit in
    front of it — "L1 loss", "training loss" — and ``_quotes`` carries a figure by
    the unit behind it. Its four copies are named here in full instead, which
    pins them all but cannot prove a fifth does not exist."""
    t = _table("Setting", "Value")
    params = _cfg(t, "Trainable params", r"^\*\*(\d+) M\*\*$")
    card = _cfg(t, "Hardware", r"RTX 3050 Laptop · (\d+) GB\*\*")
    steps = int(_cfg(t, "Batch · steps", r"· \*\*([\d ]+)\*\*$").replace(" ", ""))
    mins = _cfg(t, "Wall-clock", r"^\*\*≈ (\d+) min\*\*")
    rate = _cfg(t, "Wall-clock", r"\(~(\d+) steps/s\)$")
    peak = _cfg(t, "Peak VRAM", r"^\*\*([\d.]+) GB\*\*$")
    loss = _cfg(t, "Final L1 loss", r"^\*\*([\d.]+)\*\*$")
    short = "%dk" % (steps // 1000)

    _echoes(params, "M",
            (RM, r"Trainable params \| \*\*(\d+) M\*\*"),
            (RM, r"ResNet18 \+ Transformer · (\d+) M"),         # the pipeline diagram
            (IDX, r"(\d+) M trainable params"),                 # the stat card
            (IDX, r"Transformer, (\d+) M params\)"),            # and the lead beside it
            (DD, r"Transformer, ~(\d+) M params\)"))
    _echoes(mins, "min",
            (RM, r"Wall-clock \| \*\*≈ (\d+) min\*\*"),
            (RM, r"train ACT on the RTX 3050 \(~(\d+) min\)"),  # the quickstart
            (ARM, r"~(\d+) min on an RTX 3050\)"),
            (IDX, r"~(\d+) min on an RTX 3050"),
            (IDX, r"trained in ~(\d+) min on a laptop"))
    _echoes(short, "steps",
            (RM, r"over (\d+k) steps\">"),                      # the loss-curve alt text
            (ARM, r"batch \d+, (\d+k) steps,"),
            (DD, r"over (\d+k) steps;"))
    _echoes(card, "GB",
            (RM, r"on a single (\d+) GB laptop GPU and rolled out"),
            (RM, r"RTX 3050 Laptop · (\d+) GB\*\*"),
            (RM, r"\*\*(\d+) GB is enough\.\*\*"),
            (IDX, r"end to end on a single (\d+) GB laptop GPU\."),
            (DD, r"behaviour-cloned from them on a (\d+) GB laptop GPU"))
    _echoes(peak, "GB",
            (RM, r"Peak VRAM \| \*\*([\d.]+) GB\*\*"),
            (RM, r"holds peak VRAM at ([\d.]+) GB;"))
    # the ceiling the test above checks the peak against -- censused so a second,
    # looser "under N GB" sentence cannot appear somewhere nothing reads
    _echoes(_reads(RM, r"sits comfortably under \*\*(\d+) GB\*\* of VRAM"), "GB",
            (RM, r"sits comfortably under \*\*(\d+) GB\*\* of VRAM"))
    _echoes(rate, "steps/s",
            (RM, r"\(~(\d+) steps/s\)"))

    for rel, pattern in ((RM, r"the policy trained to ([\d.]+) loss but first"),
                         (DD, r"Trained ACT hit a clean \*\*L1 ≈ ([\d.]+)\*\*"),
                         (DD, r"L1 action loss fell from ~[\d.]+ to \*\*([\d.]+)\*\* over"),
                         (DD, r"A model with ([\d.]+) training loss")):
        assert _reads(rel, pattern) == loss, \
            f"{rel} states a final L1 loss the config table does not: {pattern}"
    assert _reads(DD, r"L1 action loss fell from ~([\d.]+) to") == \
        _reads(RM, r"loss curve — ([\d.]+) to [\d.]+ over"), \
        "the deep-dive and the loss-curve alt text disagree about where the loss started"
