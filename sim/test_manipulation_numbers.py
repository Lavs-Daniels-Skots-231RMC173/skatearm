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
import html
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ED = os.path.join(HERE, "eval_data")

if HERE not in sys.path:                  # the jaw-geometry guard below builds a
    sys.path.insert(0, HERE)              # jaw out of the scene builder itself


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def ed(name):
    with open(os.path.join(ED, name), encoding="utf-8") as f:
        return json.load(f)


def rows_by_offset(node):
    return {r["offset_mm"]: r for r in node}


# ------------------------------------------- reading the documents themselves
#
# A published figure is guarded when a test reads it OUT OF the document that
# publishes it. Loading the artefact and comparing it against a number typed
# into this file catches the DATA drifting -- but it names the document in a
# docstring, and a docstring is a comment, not a check: edit the page and this
# file still passes. That is how docs/index.html came to promise "every number
# above is recomputed from raw data" while most of the numbers above it were
# pinned only by prose in here.
#
# So the helpers below read the sentence, and the tests pin the sentence. Each
# one takes the document's own relative path, which is also the only honest
# place to write it down.

IDX = "docs/index.html"          # the landing page -- it makes the promise above
RM = "README.md"
SRM = "sim/README.md"
RMAP = "docs/ROADMAP.md"
MAN = "docs/MANIPULATION.md"
PRE = "dashboard/preview_overview.html"      # the baked cockpit demo, chips and all
GUARD = "sim/test_manipulation_numbers.py"   # this file: the checker, not a publisher

_TYPO = ((" ", " "), (" ", " "), (" ", " "),   # the three spaces
         ("—", "--"), ("–", "-"), ("−", "-"), ("‑", "-"),
         ("→", "->"), ("≤", "<="), ("≥", ">="),
         ("±", "+-"), ("×", "x"), ("°", " deg"))


def _src(rel):
    """A committed source file, as text.

    Deliberately not ``load()``/``ed()``: those two names are what
    ``_artefacts_read`` counts as raw data, and a .py file is not raw data."""
    with open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8") as f:
        return f.read()


_PROSE = {}


def _prose(rel):
    """A document's text, flattened down to the sentence it publishes.

    Three things are flattened away, because none of them is part of the claim:

    * **Wrapping.** Newlines, indentation and a comment's leading '#' all become
      one space, so the patterns below read like the sentences they pin instead
      of encoding the column each sentence happened to wrap at.
    * **Markup.** In an .html file the script and style blocks go, the tags come
      out and the entities go back in, so ``<b>&plusmn;1.6&nbsp;mm</b>`` reads as
      the ``+-1.6 mm`` a visitor sees. In a .md file the entities go back in too
      and the bold markers come off, so ``**22.12&nbsp;mm**`` reads as the
      ``22.12 mm`` a reader sees. A figure is published by the document, not by
      the emphasis it happens to sit in.
    * **Typography.** The em dash, the times sign, the degree sign and their
      friends fold to an ASCII spelling everywhere, so ONE pattern pins a figure
      written ``&plusmn;1.6&nbsp;mm`` on the landing page, ``±1.6 mm`` in the
      README and ``+-1.6 mm`` in a docstring.

    Two pieces of markdown deliberately stay. Tags are NOT pulled out of a .md,
    because these documents write a bare '<' as "less than" -- "< 0.05 N", "<2
    deg" -- and stripping ``<[^>]+>`` once the newlines are gone would swallow
    everything between one of those and the next '>'. And only ``**`` folds, not
    ``*`` and not ``__``: a single star is a glob in ``sim/demo_*.py`` and a
    double underscore is ``__pycache__``.

    Re-flowing a comment, bolding a number or swapping a hyphen for an en dash
    must not turn a guarded figure into an unguarded one.

    Cached, because ``_figure_quotes`` below reads every file in the tree once
    per figure and the fold it applies is the same fold every time."""
    if rel in _PROSE:
        return _PROSE[rel]
    txt = re.sub(r"[ \t]*\n[ \t]*#?[ \t]*", " ", _src(rel))
    if rel.endswith(".html"):
        txt = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", txt)
        txt = html.unescape(re.sub(r"<[^>]+>", " ", txt))
    elif rel.endswith(".md"):
        txt = html.unescape(txt.replace("**", ""))
    for uni, plain in _TYPO:
        txt = txt.replace(uni, plain)
    if rel.endswith(".html"):
        txt = re.sub(r"[ \t]+", " ", txt)     # tags left gaps where they stood
    _PROSE[rel] = txt
    return txt


def _quoted(rel, pattern):
    """The figure a document prints at ``pattern``, as it prints it.

    The assert is half the value: delete the sentence and the pin fails loudly
    instead of quietly having nothing left to check."""
    m = re.search(pattern, _prose(rel))
    assert m, f"{rel} no longer states the figure this test reads: {pattern}"
    return m.group(1)


def _states(rel, pattern, derived):
    """Assert a document quotes ``derived`` at ``pattern``, to the precision it
    chose to print it at: '17.4' has to be the derivation rounded to a tenth,
    '0.80' to a hundredth. The unit is whatever the sentence says -- mm, seconds,
    newtons, pixels, per cent -- and never enters the comparison.

    The bound is inclusive and carries a float-noise epsilon, so a figure landing
    exactly on a rounding tie -- 0.805 mm, which this repo prints as 0.80 -- is
    accepted rather than decided by a coin toss."""
    txt = _quoted(rel, pattern)
    dp = len(txt.split(".")[1]) if "." in txt else 0
    assert abs(derived - float(txt)) <= 0.5 * 10 ** -dp + 1e-9, \
        f"{rel} states {txt} where the data gives {derived:.4f}"
    return float(txt)


def _bound(rel, pattern, worst):
    """A tolerance a document publishes, checked in BOTH directions.

    Reading a bound off the page and asserting ``measured <= bound`` is only half
    a check: every assert downstream of it still passes if the page widens its own
    claim, which is exactly how '0.05 N' becomes '0.5 N' without CI noticing. So a
    published bound must also be TIGHT -- within one decimal order of the worst
    measurement it covers. A page may round 0.01 up to a clean 0.05; it may not
    round it up to a different claim.

    The floor is the page's own printed resolution, so a measurement that lands on
    zero does not collapse the upper limit to zero with it."""
    txt = _quoted(rel, pattern)
    bound = float(txt)
    dp = len(txt.split(".")[1]) if "." in txt else 0
    assert worst <= bound, f"{rel} publishes a bound of {txt} the data breaks at {worst:.4f}"
    assert bound < 10 * max(worst, 0.5 * 10 ** -dp), \
        f"{rel} publishes {txt}, an order looser than the {worst:.4f} actually measured"
    return bound


def _figure_quotes(fig, unit, aka=()):
    """Which committed files PRINT ``fig unit``, and how many times each.

    A pin proves the document it names still agrees with the data. It says
    nothing about the document nobody pinned -- and the unpinned copy is exactly
    the one that goes stale, because nothing fails when it does. So the pins get
    censused against the tree, and a copy the census finds but the pins do not
    account for is a failure.

    Read through ``_prose``, so ``**1116** peg px`` in markdown and
    ``1116&nbsp;peg&nbsp;px`` on the page are the same quotation. Carried WITH
    its unit, because the numeral alone cannot tell the README's 4.6 cm of ACT
    reach error from its 4.6 N of insertion force, nor 17.4 cm of arm from
    17.4 mm of lift. The separator is one-or-more, so a stylesheet's
    ``padding:16px`` is not a published pixel count.

    This file is skipped: it is the checker, not a publisher, and its docstrings
    quote most of the repo's figures by design.

    ``aka`` names sentences where the same numeral is a DIFFERENT quantity --
    sequencer.py's 1.6 mm release height is not the QC camera's 1.6 mm residual
    -- and each exemption must still find its sentence, so an exemption cannot
    outlive the prose that earned it."""
    pat = re.compile(r"(?<![\d.])" + re.escape(fig) + r"(?![\d])[\s*]+"
                     + unit + r"(?!\w)")
    skip = {}
    for rel, sentence in aka:
        spans = [(m.start(), m.end()) for m in re.finditer(sentence, _prose(rel))]
        assert spans, f"{rel} no longer carries the sentence this exemption names: {sentence}"
        skip.setdefault(rel, []).extend(spans)
    out = {}
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for name in names:
            if name.rsplit(".", 1)[-1] not in ("py", "md", "html", "txt", "yml"):
                continue
            rel = os.path.relpath(os.path.join(base, name), ROOT).replace(os.sep, "/")
            if rel == GUARD:
                continue
            hits = sum(1 for m in pat.finditer(_prose(rel))
                       if not any(s <= m.start() and m.end() <= e
                                  for s, e in skip.get(rel, ())))
            if hits:
                out[rel] = hits
    return out


def _published(derived, unit, quotes, aka=()):
    """Pin every sentence that publishes a figure -- and prove there is no other.

    ``quotes`` is the (document, pattern) list that publishes ``derived``. Each
    pair is checked against the derivation by ``_states``, at whatever precision
    that sentence chose. Then the pins are turned around and used as a census:
    what the tree prints has to be exactly what the pins cover.

    Documents print the same quantity at the precision their sentence needs --
    42.6 s on the landing page, 42.58 s where the comparison is to a hundredth --
    so the pins are grouped by the text each one actually printed, and each group
    is censused for that text. Every group is still checked against the SAME
    derivation, which is what keeps the documents agreeing with each other.

    Two consequences worth stating, because they are what makes this stricter
    than the ``_states`` calls it wraps. ``re.search`` reads the FIRST match, so
    a document printing the same sentence twice was only ever half pinned; the
    census counts both, so the second copy needs a pattern anchored to it. And
    two patterns that land on the SAME sentence count as two pins against one
    quotation, which fails -- a pin has to name a copy of its own."""
    want = {}
    for rel, pat in quotes:
        _states(rel, pat, derived)
        table = want.setdefault(_quoted(rel, pat), {})
        table[rel] = table.get(rel, 0) + 1
    for txt, table in sorted(want.items()):
        got = _figure_quotes(txt, unit, aka)
        assert got == table, \
            f"'{txt} {unit}' is printed {got} and pinned {table} -- an unpinned " \
            f"copy of a published figure is how 42.7 mm survived"
    return want


# --------------------------------------------------------------------------- M1

def test_wrench_backend_sensor_is_exact_at_both_poses():
    """MANIPULATION.md M1: the wrist sensor reads every load to 0.000 N at both
    poses — the printed form of the < 0.05 N bound sim/test_ft_sensor.py asserts.
    Checked per row, not as an average: delta must be exactly -F."""
    d = ed("wrench_backends.json")
    assert d["loads_n"] == [[0, 0, -10], [10, 0, 0], [0, 8, 0], [5, -5, -5]]
    worst = 0.0
    for pose in ("home", "working"):
        for arm, a in d["poses"][pose]["arms"].items():
            assert a["sensor_err_max_n"] == 0.0, (pose, arm)
            worst = max(worst, a["sensor_err_max_n"])
            for r in a["rows"]:
                assert r["sensor_delta_n"] == [-v for v in r["load_n"]], (pose, arm, r)
            # untared: the no-load reading is the hand's own weight, ~0.4 N
            assert round(a["baseline_sensor_n"][2], 1) == 0.4, (pose, arm)

    # The bound the landing page advertises is not a figure of its own: it is the
    # threshold sim/test_ft_sensor.py asserts. Read it out of that assert, so the
    # page cannot claim a tighter bound than the test enforces -- nor a looser one.
    bound = float(_quoted("sim/test_ft_sensor.py", r"assert err < ([\d.]+), f\"\{site\}"))
    assert worst < bound, (worst, bound)
    _states(IDX, r"known static load to < ?([\d.]+) N", bound)
    _states(IDX, r"< ?([\d.]+) N wrist wrench error", bound)


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
    open_loop = max(r["seated"] for r in n.values())
    for off, r in n.items():
        assert r["seated"] <= 1, ("no-search jams almost everywhere", off, r)
    # and the search variant must beat the open-loop baseline at every offset
    for off in d["offsets_mm"]:
        assert s[off]["seated"] >= n[off]["seated"], off

    # the landing page's version of that curve, read off the page
    _states(IDX, r"Misalignment tolerance (\d+)/6 at 2 and 4 mm", s[2]["seated"])
    _states(IDX, r"tolerance \d+/(\d+) at 2 and 4 mm", d["dirs_per_offset"])
    _states(IDX, r"(\d+)/6 at 6 mm", s[6]["seated"])
    _states(IDX, r"(\d+)/6 at 8 mm", s[8]["seated"])
    _states(IDX, r"open-loop descent manages <= (\d+)/6", open_loop)
    _states(IDX, r"(\d+)/6 seated at 2-4 mm off", s[4]["seated"])


def test_insertion_peak_force_regulated():
    """MANIPULATION.md M2: 3.3-3.6 N out to 4 mm, <=4.6 N at 6-8 mm; abort is 9 N."""
    s = rows_by_offset(ed("insertion.json")["modes"]["search"])
    near = [s[o]["peak_wrench_max_n"] for o in (0, 2, 4)]
    far = [s[o]["peak_wrench_max_n"] for o in (6, 8)]
    assert min(near) >= 3.3 and max(near) <= 3.6, near
    assert max(far) <= 4.6, far

    # the abort threshold is the controller's own constant, not a figure to type
    abort = float(_quoted("sim/insertion.py", r"w_abort=([\d.]+)"))
    assert max(near + far) < abort                           # never near the abort
    _states(IDX, r"against a ([\d.]+) N abort", abort)
    _states(IDX, r"Peak axial force stays ([\d.]+)-", min(near + far))
    # the page writes the top of that range as a range end; README.md and
    # MANIPULATION.md write it as a ceiling, "<=4.6 N". Same measured maximum
    # either way, so all three are the same census key.
    _published(max(near + far), "N", (
        (IDX, r"Peak axial force stays [\d.]+-([\d.]+) N"),
        (RM, r"peak force <=([\d.]+) N \(abort 9\)"),
        (MAN, r"and <=([\d.]+) N at the 6-8 mm extremes")))


def test_insertion_theta_tolerance():
    """MANIPULATION.md M2: initial tilts up to ~9 deg are levelled to <2 deg and seated."""
    d = ed("insertion_theta.json")
    rows = d["theta_sweep"]
    assert [r["theta_cmd_deg"] for r in rows] == [0, 3, 6, 9, 12]

    # the page states the tolerance as "levelled to under N deg" -- so N is the
    # bound the sweep has to clear, read off the page rather than typed here, and
    # required to be tight: 20 deg would also "cover" a 1.8 deg residual
    for r in rows:
        assert r["seated"] == r["trials"], r                 # every trial seats
    level = _bound(IDX, r"peg tilt is levelled to under ([\d.]+) deg",
                   max(r["tilt_final_max_deg"] for r in rows))
    assert all(r["tilt_final_max_deg"] < level for r in rows), rows
    injected = max(r["tilt_injected_deg"] for r in rows)
    assert 9.0 <= injected < 10.0, injected                  # "up to ~9 deg"
    # the README writes the same tilt with a Greek theta beside it; the pattern
    # anchors on the ASCII around it rather than on a character _prose does not fold
    _published(injected, "deg", (
        (IDX, r"a ([\d.]+) deg peg tilt is levelled"),
        (RM, r"\(open-loop <=1/6\); a ([\d.]+) deg"),
        (MAN, r"initial tilts up to ([\d.]+) deg are levelled")))


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
        assert r["e_times_k_n"] == c["wrench_n"], r          # "e*K = 8.0 N at every point"
        assert abs(r["after_release_mm"]) <= 0.2, r          # returns to ~0 on release

    _states(IDX, r"Over a (\d+)x stiffness sweep", c["k_ratio"])
    _states(IDX, r"(\d+)x stiffness, e = F/K", c["k_ratio"])
    _states(IDX, r"exactly -- [\d/.]+ mm under ([\d.]+) N", c["wrench_n"])
    # the page prints the whole curve as one slash-separated list, so pin it as
    # one: same order, same count, each entry at the precision it was printed at.
    quoted = _quoted(IDX, r"exactly -- ([\d/.]+) mm under [\d.]+ N").split("/")
    yields = [r["yield_mm"] for r in c["k_sweep"]]
    assert len(quoted) == len(yields), (quoted, yields)
    for txt, y in zip(quoted, yields):
        dp = len(txt.split(".")[1]) if "." in txt else 0
        assert abs(y - float(txt)) <= 0.5 * 10 ** -dp + 1e-9, (txt, y)


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

    # the page states the band and the tracking tolerance; both come off the page.
    # The tolerance goes through _bound(), because "within 0.05 N" is a claim about
    # precision: an assert that only checks the data fits inside it would wave a
    # widened 0.5 N straight through.
    errs = []
    for r in rows:
        assert r["held"] is True, r                               # friction alone, no weld
        assert -3.5 <= r["part_dz_mm"] <= -2.5, r                 # "dz ~ -3 mm"
        errs.append(abs(r["measured_n"] - r["target_n"]))         # tracks the target
        errs.append(abs(r["hold_force_n"] - r["target_n"]))       # grasp persists
    _bound(IDX, r"N tracked to within ([\d.]+) N", max(errs))
    meas = [r["measured_n"] for r in rows]
    assert meas == sorted(meas), meas                             # monotone

    _states(IDX, r"([\d.]+)-[\d.]+ N tracked to within [\d.]+ N", min(d["targets_n"]))
    _states(IDX, r"[\d.]+-([\d.]+) N tracked to within [\d.]+ N", max(d["targets_n"]))


def test_gripper_slip_curve():
    """MANIPULATION.md M4: grasp 2/3/4/5 N -> slips at 3.75/4.25/5.00/5.25 N."""
    rows = ed("gripper.json")["slip"]
    got = {r["target_n"]: r["slip_payload_n"] for r in rows}
    assert got == {2.0: 3.75, 3.0: 4.25, 4.0: 5.00, 5.0: 5.25}
    payloads = [r["slip_payload_n"] for r in rows]
    assert payloads == sorted(payloads)      # a firmer grasp holds a larger payload
    for r in rows:
        assert r["slip_payload_n"] > r["grasp_n"], r

    # the landing page publishes the firmest grasp and what it carried
    firmest = max(got)
    _states(IDX, r"a ([\d.]+) N grasp carries [\d.]+ N before it lets go", firmest)
    _published(got[firmest], "N", (
        (IDX, r"a [\d.]+ N grasp carries ([\d.]+) N before it lets go"),
        (IDX, r"([\d.]+) N held before slip"),
        (MAN, r"slips at [\d.]+/[\d.]+/[\d.]+/([\d.]+) N")))


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

    # and now the documents themselves, each read out of its own text
    _published(px["weld"]["top_peg"], "peg", (
        (IDX, r"the weld path sees (\d+) peg pixels"),
        (RM, r"the weld path sees (\d+) peg px"),
        (SRM, r"the weld path gives (\d+) peg px"),
        (MAN, r"the weld path shows (\d+) peg px")))
    _published(px["weld"]["top_rim"], "(?:pocket-)?rim", (
        (RM, r"\d+ peg px and (\d+) rim px"),
        (SRM, r"the weld path gives \d+ peg px / (\d+) rim px"),
        (MAN, r"and (\d+) pocket-rim px")))
    _published(px["jaws"]["top_rim"], "rim", (
        (SRM, r"the jaw path gives 0 peg px / (\d+) rim px"),
        (MAN, r"the jaw path shows 0 peg px and (\d+) rim px")))
    _published(loss, "%", (
        (IDX, r"with (\d+) % of the pocket rim gone"),
        (IDX, r"-(\d+) % pocket rim seen"),
        (RM, r"0 peg px, (\d+) % of the rim gone"),
        (SRM, r"\((\d+) % of the rim gone\)"),
        (MAN, r"(\d+) % of the rim gone")))
    _published(d["qc"]["roi_px"], "px", (
        (IDX, r"In the same (\d+) px window"),
        (IDX, r"[\d]+ px of that (\d+) px window"),
        (RM, r"measured in the same (\d+) px inspection window"),
        (SRM, r"inside the (\d+) px inspection ROI"),
        (SRM, r"which is [\d]+ px of a (\d+) px window"),
        (MAN, r"inside the (\d+) px inspection ROI"),
        (MAN, r"[\d]+ px of that (\d+) px window"),
        ("sim/eval_qc_occlusion.py", r"its masks and its (\d+) px inspection ROI"),
        ("sim/eval_qc_occlusion.py", r"at 720 rows the centred (\d+) px ROI"),
        ("sim/eval_qc_occlusion.py", r"cannot claim a (\d+) px window"),
        ("sim/sequencer.py", r"restricts the analysis to a centred (\d+) px ROI")))
    # these two print the count without its unit beside it, so the census cannot
    # see them and they stay plain pins
    _states(IDX, r"(\d+) -> 0 peg px in window", px["weld"]["top_peg"])
    _states(RMAP, r"against (\d+) on the weld path", px["weld"]["top_peg"])


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

    # the pages that publish that unit, read out of themselves
    depth = j["oracle"]["depth_mm"]
    _published(depth, "mm", (
        (IDX, r"S4 inserts to ([\d.]+) mm"),
        (IDX, r"([\d.]+) mm insert at [\d.]+ N"),
        (RM, r"active in it anywhere, ([\d.]+) mm insert"),
        (RM, r"insert to ([\d.]+) mm at"),
        (SRM, r"inserts to ([\d.]+) mm at"),
        (RMAP, r"insert to ([\d.]+) mm"),
        (RM, r"([\d.]+) mm / [\d.]+ deg / [\d.]+ mm ACCEPT"),
        (SRM, r"ACCEPT at ([\d.]+) mm /"),
        (MAN, r"Final unit: depth ([\d.]+) mm"),
        (MAN, r"the same seated unit \(([\d.]+) mm"),
        (MAN, r"([\d.]+) mm at [\d.]+ deg tilt")))
    _published(j["oracle"]["tilt_deg"], "deg", (
        (RM, r"[\d.]+ mm / ([\d.]+) deg / [\d.]+ mm ACCEPT"),
        (SRM, r"ACCEPT at [\d.]+ mm / ([\d.]+) deg"),
        (MAN, r"tilt ([\d.]+) deg"),
        (MAN, r"the same seated unit \([\d.]+ mm / ([\d.]+) deg"),
        (MAN, r"[\d.]+ mm at ([\d.]+) deg tilt")))
    _published(j["oracle"]["align_mm"], "mm", (
        (RM, r"[\d.]+ mm / [\d.]+ deg / ([\d.]+) mm ACCEPT"),
        (SRM, r"ACCEPT at [\d.]+ mm / [\d.]+ deg / ([\d.]+) mm on the oracle"),
        (MAN, r"alignment error ([\d.]+) mm"),
        (MAN, r"the same seated unit \([\d.]+ mm / [\d.]+ deg / ([\d.]+) mm")))
    # the camera's own two readings of that same unit, which MANIPULATION.md
    # prints beside the oracle's to show how far the estimate sits from truth
    _published(w["camera"]["align_err_mm"], "mm", ((MAN, r"align ([\d.]+) mm, depth"),))
    _published(w["camera"]["depth_mm_est"], "mm",
               ((MAN, r"align [\d.]+ mm, depth ([\d.]+) mm"),))
    # the landing page's word, not just its number. It prints that verdict in the
    # grid that reports the jaw path's "1116 -> 0 peg px", so it is the ORACLE's
    # reading of the very unit the camera rejected -- re-derived from ok(), not
    # copied out of a field.
    o = j["oracle"]
    page = _quoted(IDX, r"(ACCEPT|REJECT) on the sim oracle")
    assert page == ("ACCEPT" if ok(o["depth_mm"], o["align_mm"], o["tilt_deg"])
                    else "REJECT"), (page, o)


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
    delta_mm, delta_px = (d["summary"]["unit_pose_delta_mm"],
                          d["summary"]["unit_pose_delta_top_px"])
    assert delta_mm == 7.74
    assert delta_px == 16.0
    assert delta_px < 0.1 * d["qc"]["roi_px"]
    _published(delta_mm, "mm", (
        (IDX, r"settle the part ([\d.]+) mm apart"),
        (MAN, r"settle the part ([\d.]+) mm apart"),
        (SRM, r"settle the part ([\d.]+) mm apart, which is")))
    # the px offset and the ROI it is a fraction OF are published in one breath;
    # the two patterns keep them coupled while each still names its own numeral
    _published(delta_px, "px", (
        (IDX, r"(\d+) px of that [\d]+ px window"),
        (MAN, r"(\d+) px of that [\d]+ px window"),
        (SRM, r"which is (\d+) px of a [\d]+ px window")))

    # the weld column is the published reference cycle, independently re-run:
    # MANIPULATION.md's "75.84 s against the weld path's 42.58 s"
    assert round(w["cycle_time_s"], 2) == 42.58
    assert round(w["cycle_time_s"], 1) == round(load("logs/cycle_001.json")[-1]["cycle_time_s"], 1)
    _states(MAN, r"Full cycle [\d.]+ s against the weld path's ([\d.]+) s",
            w["cycle_time_s"])

    # The jaw-path comparison figure, 75.84 s, is the one number in this file with
    # NO committed artefact behind it: it comes from sim/test_cell_gripper.py,
    # which needs MuJoCo and so cannot run in the hardware-free job. It can still
    # be kept CONSISTENT, which is the drift that actually happens -- so take
    # MANIPULATION.md's 2 dp as the published value, make every rounder copy agree
    # with it, and use it as the bound below instead of a literal typed in here.
    gripper_s = float(_quoted(MAN, r"Full cycle ([\d.]+) s against the weld path's"))
    _published(gripper_s, "s", (
        (MAN, r"Full cycle ([\d.]+) s against the weld path's"),
        (MAN, r"deg tilt, ([\d.]+) s against the weld path's"),
        (IDX, r"-- ([\d.]+) s against the weld path's"),
        (IDX, r"([\d.]+) s cycle \(takt <= 85 s\)"),
        (RM, r"mm insert, ([\d.]+) s,"),
        (RM, r"([\d.]+) s \(oracle-gated"),
        (SRM, r"([\d.]+) s against the weld path's"),
        (RMAP, r"([\d.]+) s against the weld path's"),
        ("sim/demo_cell_cycle.py", r"Measured ([\d.]+) s against the weld path's"),
        ("sim/demo_cell_cycle.py", r"measured jaw cycle ([\d.]+) s"),
        ("sim/eval_qc_occlusion.py", r"runs LONGER than the ([\d.]+) s takt"),
        ("sim/test_cell_gripper.py", r"the weld-free cycle measures ([\d.]+) s")))

    # the jaws column runs LONGER than that published figure, and must: a camera
    # REJECT sends S6 to the far reject bin, while the 75.84 s is the oracle-gated
    # ACCEPT branch sim/test_cell_gripper.py runs with no renderer attached. Same
    # cycle, different S6 branch -- which branch it takes is what this eval measures.
    assert j["cycle_time_s"] > gripper_s, (j["cycle_time_s"], gripper_s)

    # three documents publish the GAP between the two cycles rather than either
    # end of it, so derive the gap instead of letting a subtraction be typed
    _published(gripper_s - w["cycle_time_s"], "s", (
        (SRM, r"itemises that \+(\d+) s per GRAFCET step"),
        (IDX, r"That is where the extra (\d+) s goes"),
        ("sim/demo_cell_cycle.py", r"itemises where the extra (\d+) s goes")))

    # Five more figures the gripper cell publishes with no committed artefact
    # behind them, held by the same doctrine as the 75.84 s above: take the
    # deepest-precision copy as the published value, make every other copy agree
    # with it, and census the tree so a sixth copy cannot appear unpinned.
    jaws_depth = float(_quoted(MAN, r"force-regulates the insert to ([\d.]+) mm"))
    _published(jaws_depth, "mm",
               ((MAN, r"force-regulates the insert to ([\d.]+) mm"),))
    # it is a DIFFERENT measurement from the oracle's 22.1209 mm -- a jaw-frame
    # reading of the same seated unit -- so they are only held to agree to a tenth
    assert round(jaws_depth, 1) == round(j["oracle"]["depth_mm"], 1)
    peak_n = float(_quoted(MAN, r"at ([\d.]+) N peak and then"))
    _published(peak_n, "N", (
        (MAN, r"at ([\d.]+) N peak and then"),
        (RM, r"insert to [\d.]+ mm at ([\d.]+) N"),
        (IDX, r"S4 inserts to [\d.]+ mm at ([\d.]+) N"),
        (IDX, r"[\d.]+ mm insert at ([\d.]+) N"),
        (SRM, r"inserts to [\d.]+ mm at ([\d.]+) N peak")))
    pick_n = float(_quoted(MAN, r"\(S1, ([\d.]+) N measured on its own pad sensor\)"))
    _published(pick_n, "N", (
        (MAN, r"\(S1, ([\d.]+) N measured on its own pad sensor\)"),
        (RM, r"off the table \(([\d.]+) N\) and sets it down"),
        (IDX, r"off the table at ([\d.]+) N on its own pad sensor"),
        (SRM, r"off the table \(S1, ([\d.]+) N\)")))
    drift_mm = float(_quoted(MAN, r"\(S5, ([\d.]+) mm drift\)"))
    _published(drift_mm, "mm", (
        (MAN, r"\(S5, ([\d.]+) mm drift\)"),
        (RM, r"re-grips the unit \(([\d.]+) mm drift\)"),
        (IDX, r"([\d.]+) mm re-grip drift"),
        (SRM, r"\(S5, ([\d.]+) mm drift\)")))
    slip_mm = float(_quoted(MAN, r"friction alone \(([\d.]+) mm drift in the jaw frame\)"))
    _published(slip_mm, "mm", (
        (MAN, r"friction alone \(([\d.]+) mm drift in the jaw frame\)"),
        (RM, r"carried on friction \(([\d.]+) mm slip\)"),
        (RM, r"sets it down, ([\d.]+) mm slip over the carry"),
        (IDX, r"friction alone with ([\d.]+) mm of slip"),
        (SRM, r"\(S3/S4, ([\d.]+) mm slip over the carry\)"),
        ("sim/test_cell_gripper.py", r"Measured drift is ([\d.]+) mm")))


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
    _states(SRM, r"depth ([\d.]+) mm \(target 18\)", ins["depth_mm"]["mean"])
    _states(SRM, r"peg tilt ([\d.]+)-[\d.]+ deg", ins["peg_tilt_deg"]["min"])
    _states(SRM, r"peg tilt [\d.]+-([\d.]+) deg", ins["peg_tilt_deg"]["max"])

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
    """Every document that quotes the reference cycle is read here, out of itself.

    42.6 s is the single most-copied figure in the repo -- the landing page says
    it twice, and four more documents repeat it. Naming them in a docstring is
    what let them drift; naming them in ``_states`` is what stops it. The takt
    bound is read off the page too, so 'inside the takt target' means inside the
    target the page publishes rather than one typed in here."""
    log = load("logs/cycle_001.json")
    t = log[-1]["cycle_time_s"]
    assert round(t, 1) == 42.6, t

    _published(t, "s", (
        (IDX, r"([\d.]+) s cycle \(takt <= 60 s\)"),
        (IDX, r"against the weld path's ([\d.]+) s"),
        (RM, r"Cycle time \| ([\d.]+) s \(takt target <= 60 s\)"),
        (SRM, r"Reference cycle: ([\d.]+) s"),
        (SRM, r"against the weld path's ([\d.]+) s"),
        (RMAP, r"full cycle ([\d.]+) s <= 60 s takt"),
        (RMAP, r"against the weld path's ([\d.]+) s"),
        (MAN, r"published ([\d.]+) s reference cycle"),
        # MANIPULATION.md prints it to 2 dp, twice, in the two sentences that
        # compare the jaw cell against it
        (MAN, r"Full cycle [\d.]+ s against the weld path's ([\d.]+) s"),
        (MAN, r"deg tilt, [\d.]+ s against the weld path's ([\d.]+) s"),
        ("sim/demo_cell_cycle.py", r"Measured reference cycle: ([\d.]+) s"),
        ("sim/demo_cell_cycle.py", r"Measured [\d.]+ s against the weld path's ([\d.]+) s"),
        ("sim/demo_cell_cycle.py", r"measured weld cycle ([\d.]+) s"),
        ("sim/test_cell_gripper.py", r"that measures ([\d.]+) s end to end")))

    # the page publishes TWO takt bounds -- 60 s beside this cycle and 85 s beside
    # the gripper cell's 75.8 s -- so the bound is addressed through the cycle it
    # bounds, not by position: an anchorless pattern would take whichever panel the
    # page happens to print first, and quietly change meaning if they were swapped.
    takt = float(_quoted(IDX, r"%s s cycle \(takt <= (\d+) s\)" % re.escape("%.1f" % t)))
    assert t < takt, (t, takt)                                # inside the published takt


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
    """The +-1.6 mm / +-3.4 mm QC residual pair, read out of all four documents
    that print it.

    The landing page writes them ``&plusmn;1.6&nbsp;mm``, the READMEs write them
    ``±1.6 mm`` and this docstring writes them ``+-1.6 mm``. ``_prose`` folds all
    three to the same ASCII, so one pattern per site is enough."""
    log = load("logs/cycle_002_camera_qc.json")
    qc = [e for e in log if "residual_align_mm" in e]
    assert len(qc) == 1, "the camera-QC cycle log must carry exactly one verify record"
    qc = qc[0]
    align, depth = qc["residual_align_mm"], qc["residual_depth_mm"]
    assert round(align, 1) == 1.6, align
    assert round(depth, 1) == 3.4, depth
    # the residual is |camera - oracle| on each axis — recompute it, don't trust the field.
    # 1e-3 tolerance: the log rounds each term to 4 dp *after* the residual is taken.
    assert abs(abs(qc["cam_align_mm"] - qc["oracle_align_mm"]) - align) < 1e-3, qc
    assert abs(abs(qc["cam_depth_mm"] - qc["oracle_depth_mm"]) - depth) < 1e-3, qc

    # the baked cockpit preview prints the pair too, as chips; dashboard/app.py
    # holds only the Jinja placeholders, so the literals live in the bake alone
    _published(align, "mm", (
        (IDX, r"\+-([\d.]+) mm QC alignment"),
        (RM, r"alignment \(camera vs sim oracle\) \| \+-([\d.]+) mm"),
        (SRM, r"alignment \+-([\d.]+) mm"),
        (RMAP, r"residuals align \+-([\d.]+) mm"),
        (PRE, r"([\d.]+) mm QC residual . alignment")),
        # sequencer.py's two 1.6 mm are a RELEASE HEIGHT, not this residual --
        # same numeral, same unit, different quantity
        aka=(("sim/sequencer.py", r"the extra [\d.]+ mm is a release height"),
             ("sim/sequencer.py", r"the jaws open with the unit [\d.]+ mm above the surface")))
    _published(depth, "mm", (
        (IDX, r"\+-([\d.]+) mm insert depth"),
        (RM, r"QC residual, insertion depth \| \+-([\d.]+) mm"),
        (SRM, r"alignment \+-[\d.]+ mm, depth \+-([\d.]+) mm"),
        (RMAP, r"residuals align \+-[\d.]+ mm / depth \+-([\d.]+) mm"),
        (PRE, r"([\d.]+) mm QC residual . depth"),
        (MAN, r"depth estimate reads ~([\d.]+) mm")))


# --------------------------------------------------------------- M4 jaw geometry

def _const(rel, name):
    """A module-level constant, off the file's AST -- sim/sequencer.py imports
    mujoco and numpy, so the hardware-free job cannot import it."""
    for n in ast.parse(_src(rel)).body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError(f"{rel} no longer defines {name}")


def _xml_boxes(block):
    """Every box geom of a body given as an MJCF snippet, as (pos, half-size)."""
    import xml.etree.ElementTree as ET
    out = []
    for g in ET.fromstring(block.strip()).findall("geom"):
        pos = [float(v) for v in (g.get("pos") or "0 0 0").split()]
        out.append((pos, [float(v) for v in g.get("size").split()]))
    return out


def _vee_jaw_corners(sign):
    """Every corner of one V-groove jaw's two pad plates, in the wrist's frame,
    taken from the geoms sim/make_cell_scene.py actually emits -- pos, size and
    the 45 deg euler, applied -- and not re-typed from the constants behind them.

    Also returns the jaw body's rest offset along the closing axis and its slide
    range, which with the corners is everything the jaw figures are made of."""
    import make_cell_scene as mcs
    import make_gripper_cell as mgc

    jaw = mcs._vee_jaw("jaw", "j", sign, mgc.REACH, mgc.OPEN, mgc.PAD_Y, mgc.MU,
                       tag="X")
    x0 = float(jaw.get("pos").split()[0])
    lo, hi = (float(v) for v in jaw.find("joint").get("range").split())
    pts = []
    for g in jaw.findall("geom"):
        hx, hy, hz = (float(v) for v in g.get("size").split())
        px, py, pz = (float(v) for v in g.get("pos").split())
        th = float(g.get("euler").split()[1])            # rotation about local y
        c, s = math.cos(th), math.sin(th)
        for a in (1, -1):
            for b in (1, -1):
                for e in (1, -1):
                    pts.append((x0 + px + a * hx * c + e * hz * s,
                                py + b * hy,
                                pz - a * hx * s + e * hz * c))
    return pts, x0, lo, hi


def test_jaw_geometry_matches_the_prose_that_states_it():
    """M4's jaw numbers are the one published family with no artefact behind
    them: they are pure geometry, so the geometry itself is the guard.

    Every figure below is rebuilt from the geoms sim/make_cell_scene.py emits and
    checked against the sentence quoting it, in the file quoting it -- all seven
    of them, because the tip gap did not stay in sim/: it is the premise of the
    camera-occlusion result too, and it is quoted in both READMEs and in
    docs/MANIPULATION.md. Change a pad size, the V's 45 deg, OPEN or the slide
    range and that prose silently stops being true -- the same drift the rest of
    this file exists to catch, applied to the one family that predates it. It had
    already drifted: sequencer.py quoted a 42.7 mm geometric tip gap this
    construction never produces."""
    CELL, SEQ = "sim/make_cell_scene.py", "sim/sequencer.py"
    SPEC = "specs/demo_task_spec.md"     # DECISION 4's annotation quotes them too
    EPS = 2e-6      # the builder prints pos and euler to 6 dp, so a micron is the
    import make_cell_scene as mcs                    # resolution of the geometry
    left, x0, lo, hi = _vee_jaw_corners(-1)
    right, x0r, lo_r, hi_r = _vee_jaw_corners(+1)

    # the right jaw is the left one mirrored in x, so one jaw's numbers are both
    assert abs(x0 + x0r) < 1e-12 and (lo, hi) == (-hi_r, -lo_r)
    assert abs(max(p[0] for p in left) + min(p[0] for p in right)) < 1e-12
    assert abs(x0 + lo) > abs(x0 + hi), "lo is meant to be the opening end"

    # THE VERTEX. The V's inside corner sits exactly on the jaw body's origin,
    # and every distance below is measured from it, so it is pinned first.
    off = min(abs(p[0] - x0) + abs(p[2]) for p in left)
    assert off < EPS, f"the V's vertex left the jaw origin by {off * 1e3:.4f} mm"

    # THE TIPS. The plate corners furthest along the closing axis: they are the
    # contact lines, one either side of the pad centre, and every gap below is
    # measured between THEM -- not between the plate centres, which is the wrong
    # line and flatters the grasp window by ~3.5 mm a side.
    xmax = max(p[0] for p in left)
    tips = [p for p in left if abs(p[0] - xmax) < EPS]
    tip_x, tip_z = xmax - x0, max(abs(p[2]) for p in tips)
    assert abs(min(p[2] for p in tips) + tip_z) < EPS      # one either side of 0
    assert abs(tip_x - tip_z) < EPS, (tip_x, tip_z)       # 45 deg: equal in both
    _states(SEQ, r"sit \+-([\d.]+) mm either side", tip_z * 1e3)
    _states(SPEC, r"([\d.]+) mm from pad centre", tip_z * 1e3)

    # TRAVEL. Rest gap, where the two plates would meet, where a D20 peg seats.
    rest = 2.0 * (abs(x0) - tip_x)
    _states(CELL, r"tip gap at rest is ([\d.]+) mm", rest * 1e3)
    _states(SEQ, r"tip gap at rest is ([\d.]+) mm", rest * 1e3)
    _states(SPEC, r"([\d.]+) mm tip gap at rest", rest * 1e3)
    _states(SEQ, r"plate tips are ([\d.]+) mm apart at rest", rest * 1e3)
    _states(CELL, r"colliding with each other at ([\d.]+) mm", rest * 0.5e3)
    r_peg = float(_quoted_peg(mcs))
    assert round(2e3 * r_peg) == 20, "the peg stopped being a D20"
    _states(CELL, r"seats a D20 peg at ([\d.]+) mm",
            (abs(x0) - r_peg * math.sqrt(2)) * 1e3)     # 90 deg V, tangent on 4 lines
    stop_in = _states(CELL, r"hard stop at \+([\d.]+) mm", hi * 1e3)
    assert stop_in < rest * 0.5e3, \
        f"the +{stop_in} mm stop is past the {rest * 0.5e3:.2f} mm the plates meet at"

    # THE GAP. Geometry gives one number and the model measures another, wider
    # one, because the range end is soft-limited rather than rigid. The prose has
    # to carry both, and the arithmetic between them.
    geom_gap = 2e3 * (abs(x0 + lo) - tip_x)
    _states(SEQ, r"the geometry alone gives ([\d.]+) mm", geom_gap)
    over = float(_quoted(SEQ, r"parks the jaw ([\d.]+) mm past it"))
    extra = float(_quoted(SEQ, r"The extra ([\d.]+) mm is the stop"))
    assert abs(extra - 2 * over) < 1e-12, (extra, over)   # one overshoot per jaw
    # The gap is quoted TEN times, and only four of them are in the two sim
    # modules this guard grew up beside. The other six say the same thing in
    # other words -- the base's 60 mm length does not fit between the jaws --
    # and one of those six is the premise the entire camera-occlusion result
    # rests on. Pinning a figure only in the files the guard happens to sit next
    # to would leave the top-level README free to keep publishing a gap the
    # geometry had stopped producing, which is the drift, not a lesser cousin of
    # it. So every copy is pinned, and they are all made to agree.
    QUOTES = ((SEQ, r"tip gap is ([\d.]+) mm MEASURED"),
              (SEQ, r"so ([\d.]+) mm is the mechanical maximum"),
              (SEQ, r"exceeds the ([\d.]+) mm tip gap measured below"),
              (SEQ, r"\(([\d.]+) mm of tip gap across a"),
              (CELL, r"the ([\d.]+) mm tip gap the"),
              (SPEC, r"([\d.]+) mm at the stop"),
              ("sim/eval_qc_occlusion.py", r"exceeds the ([\d.]+) mm tip gap"),
              ("docs/MANIPULATION.md", r"exceeds the ([\d.]+) mm tip gap"),
              ("README.md", r"exceeds the ([\d.]+) mm jaw gap"),
              ("sim/README.md", r"exceeds the ([\d.]+) mm jaw gap"))
    gap = None
    for rel, pat in QUOTES:
        stated = _states(rel, pat, geom_gap + extra)
        assert gap in (None, stated), f"{rel} quotes the gap as {stated} mm " \
                                      f"where the rest of the repo says {gap} mm"
        gap = stated                 # everything downstream follows from the
                                     # PUBLISHED gap, as the prose's own arithmetic
                                     # does; the line above is what ties it to the
                                     # geometry and to the measured overshoot.

    # ...and there is no eleventh copy. Six of those pins exist only because the
    # figure had escaped into files nobody thought to check, so the count is
    # guarded too: quote the gap somewhere new without pinning it and this
    # fails, which is the difference between fixing six oversights and closing
    # the hole they came through.
    fig = _quoted(SEQ, r"tip gap is ([\d.]+) mm MEASURED")
    want = {}
    for rel, _pat in QUOTES:
        want[rel] = want.get(rel, 0) + 1
    assert _figure_quotes(fig, "mm") == want, \
        f"the {fig} mm gap is written {_figure_quotes(fig, 'mm')} and pinned {want} " \
        f"-- an unpinned copy of a published figure is how 42.7 mm survived"

    # THE PART, off the cell scene's own base body rather than from the prose.
    boxes = _xml_boxes(mcs.SQUARE_BASE)
    span = [(min(p[i] - s[i] for p, s in boxes), max(p[i] + s[i] for p, s in boxes))
            for i in range(3)]
    dims = [round(1e3 * (b - a)) for a, b in span]
    m = re.search(r"base part is (\d+) x (\d+) x (\d+) mm", _prose(CELL))
    assert m and [int(v) for v in m.groups()] == dims, (m and m.groups(), dims)
    length, face = float(dims[0]), float(dims[1])

    # what the gap means for the 40 mm face the jaws are forced onto
    _states(CELL, r"only take it across the ([\d.]+) mm faces", face)
    _states(SEQ, r"Across a ([\d.]+) mm base", face)
    _states(SEQ, r"across a ([\d.]+) mm part", face)
    for rel, pat in ((CELL, r"leaves ([\d.]+) mm per side on 40 mm"),
                     (SEQ, r"that is ([\d.]+) mm per side"),
                     (SEQ, r"with ([\d.]+) mm per side at their hard stop")):
        _states(rel, pat, 0.5 * (gap - face))
    assert length > gap, "the base's length would fit between the jaws after all"
    _states(SEQ, r"with ([\d.]+) mm of clearance per side",
            0.5 * (gap - 2e3 * r_peg))

    # WHERE ALONG THE LENGTH the jaws may bite: both tips on the wall.
    window = 0.5 * length - tip_z * 1e3
    _states(SEQ, r"only for \|dx\| <= ([\d.]+) mm", window)
    dx = abs(_const(SEQ, "LGRIP_DX")) * 1e3
    assert dx < window, f"LGRIP_DX = -{dx} mm puts a tip off the {window:.2f} mm wall"
    _states(SEQ, r"([\d.]+) mm of tip margin", window - dx)

    # HOW HIGH. LGRIP_REL is a choice between two geometric bounds, not a guess:
    # the pad has to cover the whole side wall and still clear the table.
    pad_h = 1e3 * (max(p[1] for p in left) - min(p[1] for p in left))
    _states(SEQ, r"pads are ([\d.]+) mm tall", pad_h)
    floor_top = 1e3 * min(p[2] + s[2] for p, s in boxes)      # the pocket's floor
    wall_top = 1e3 * span[2][1]
    _states(SEQ, r"side wall they clamp is ([\d.]+) mm \(z", wall_top - floor_top)
    m = re.search(r"\(z (\d+)\.\.(\d+) above the bottom\)", _prose(SEQ))
    assert m and [float(v) for v in m.groups()] == [floor_top, wall_top]
    lo_c = max(pad_h / 2, wall_top - pad_h / 2)     # clear the table / reach the top
    hi_c = floor_top + pad_h / 2                    # reach the bottom of the wall
    m = re.search(r"any centre in (\d+)\.\.(\d+) mm engages", _prose(SEQ))
    assert m and [float(v) for v in m.groups()] == [lo_c, hi_c], (m.groups(), lo_c, hi_c)
    rel_h = 1e3 * _const(SEQ, "LGRIP_REL")
    assert lo_c <= rel_h <= hi_c, (rel_h, lo_c, hi_c)
    _states(SEQ, r"leaves the pad ([\d.]+) mm clear of the table", rel_h - pad_h / 2)


def _quoted_peg(mcs):
    """The D20 peg's radius, out of the cell scene's own peg body."""
    m = re.search(r'name="peg_body"[^>]*size="([\d.]+)', mcs.PEG)
    assert m, "sim/make_cell_scene.py no longer emits a peg_body cylinder"
    return m.group(1)


# ------------------------------------------------------- the site's own counters

def _site_counter(label):
    """The figure docs/index.html prints beside ``label`` in its 'recomputed from
    raw data' panel.

    The raw markup, deliberately, not ``_prose()``: this one reads the counter
    out of the ``<div><b>N</b><span>`` element it lives in, so it has to see the
    tags the other pins strip away."""
    txt = _src("docs/index.html")
    m = re.search(r"<div><b>(\d+)</b><span>" + re.escape(label) + r"</span></div>", txt)
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
