"""Virtual-obstacle augmentation of the collision guard. The geometry helper
is unit-tested standalone (no mujoco); the guard integration places a box on
the robot and checks it blocks, then clears, and measures what the keep-out
test actually reserves against what the links actually occupy.

    SKT_DIR=.../skt_v3 python -m pytest test/test_obstacles.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skate_ros2"))

from skate_commander.server import _obb_hit   # noqa: E402

I3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]     # unrotated link

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
CXML = SKT / "skt_v3_collision.xml"



def _skip(msg):
    """Real pytest.skip under pytest; clean print when run as a standalone script."""
    import sys
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"SKIP: {msg}")


def test_obb_hit_geometry():
    half = [0.02, 0.02, 0.02]                              # a 4 cm cube of a link
    box = {"type": "box", "p": [0, 0, 0], "s": [0.1, 0.1, 0.1]}
    assert _obb_hit([0, 0, 0], I3, half, box)              # link centre inside the box
    assert _obb_hit([0.11, 0, 0], I3, half, box)           # 1 cm past the face, still touching
    assert not _obb_hit([0.25, 0, 0], I3, half, box)       # clearly clear
    cyl = {"type": "cyl", "p": [0, 0, 0], "s": [0.1, 0.2]}  # radius 0.1, half-height 0.2
    assert _obb_hit([0.06, 0.06, 0.1], I3, half, cyl)      # inside radius + height
    assert not _obb_hit([0.4, 0, 0], I3, half, cyl)        # outside radially
    assert not _obb_hit([0, 0, 0.4], I3, half, cyl)        # above the cap
    assert not _obb_hit([0, 0, 0], I3, half, {"type": "box"})   # missing p/s → no hit
    assert not _obb_hit([0, 0, 0], I3, half, {"type": "box", "p": [0, 0, 0], "s": [0.1]})


def test_obb_hit_sheds_phantom_volume():
    """A long link reserves itself, not a ball around itself."""
    # a ~25 cm forearm along world X: half-extents 12.5 x 4 x 4 cm
    c, half = [0, 0, 0], [0.125, 0.04, 0.04]
    far = {"type": "box", "p": [0, 0.10, 0], "s": [0.02, 0.02, 0.02]}
    # the enclosing sphere the guard used to reason over is 13.7 cm and blocks
    # this box; the link's own volume stops 4 cm short of it
    assert (0.125 ** 2 + 0.04 ** 2 + 0.04 ** 2) ** 0.5 > 0.13
    assert not _obb_hit(c, I3, half, far)                  # ALLOWED, and correctly so
    near = {"type": "box", "p": [0.05, 0.05, 0], "s": [0.02, 0.02, 0.02]}
    assert _obb_hit(c, I3, half, near)                     # face 3 cm out, inside 4 cm → BLOCKS
    tip = {"type": "box", "p": [0.14, 0, 0], "s": [0.02, 0.02, 0.02]}
    assert _obb_hit(c, I3, half, tip)                      # box at the far end → BLOCKS


def test_obb_hit_is_exact_under_rotation():
    """The separating axis can be a cross product, and a sphere/AABB test misses
    exactly that case: a 45-degree link whose CORNER reaches into the box."""
    import math
    ca, sa = math.cos(math.radians(45)), math.sin(math.radians(45))
    Rz = [ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0]        # yaw 45 deg, row-major
    half = [0.20, 0.02, 0.02]                              # a long thin link
    # box placed off the link's long axis: the link body misses it...
    off = {"type": "box", "p": [0.0, 0.16, 0.0], "s": [0.02, 0.02, 0.02]}
    assert not _obb_hit([0, 0, 0], Rz, half, off)
    # ...but slide the link's own tip onto it and it must block. The tip sits at
    # 0.20 along the rotated axis = (0.1414, 0.1414, 0)
    on = {"type": "box", "p": [0.1414, 0.1414, 0.0], "s": [0.02, 0.02, 0.02]}
    assert _obb_hit([0, 0, 0], Rz, half, on)
    # an unrotated link of the same size does NOT reach that spot
    assert not _obb_hit([0, 0, 0], I3, half, on)


def test_guard_blocks_a_box_on_the_robot():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        _skip("mujoco not installed"); return
    if not CXML.exists():
        _skip("no collision model"); return
    from skate_commander.server import make_collision_guard

    obstacles = []
    guard = make_collision_guard(CXML, get_obstacles=lambda: obstacles)
    neutral = np.zeros(26)
    assert not guard(neutral), "no obstacles → neutral pose allowed"

    geoms = guard.collision_view(neutral)
    assert geoms, "collision_view should return robot geoms"
    p = geoms[len(geoms) // 2]["p"]                        # a mid-chain geom centre
    obstacles.append({"id": 1, "type": "box", "p": list(p), "s": [0.05, 0.05, 0.05]})
    assert guard(neutral), "a box placed on the robot must be detected"

    obstacles.clear()
    assert not guard(neutral), "clearing obstacles unblocks again"
    print("PASS virtual box blocks + clears")


def _old_bound_r(t, sz):
    """The per-geom bounding sphere the keep-out test used before the oriented
    boxes. Kept HERE, in the test, because it is the thing being compared
    against — it is not production code any more."""
    import math
    if t == 6:
        return math.sqrt(float(sz[0]) ** 2 + float(sz[1]) ** 2 + float(sz[2]) ** 2)
    if t == 3:
        return float(sz[0]) + float(sz[1])
    if t == 5:
        return math.hypot(float(sz[0]), float(sz[1]))
    if t == 2:
        return float(sz[0])
    return float(max(sz))


def test_keepout_covers_the_links_and_sheds_phantom_volume():
    """Both halves of the claim, recomputed rather than quoted.

    COVERS: a keep-out box touching any corner of a link's compiled AABB is
    caught. The old bounding spheres were not merely loose — for a mesh geom the
    radius they used (``max(geom_size)``) is *smaller* than the mesh, so the
    test was blind to real intrusions while blocking empty air elsewhere.

    SHEDS: sampled over the robot's bounding volume, the blocked fraction drops
    by more than half. Both numbers are printed so the prose can be checked
    against a run instead of trusted.
    """
    try:
        import mujoco
    except ImportError:
        _skip("mujoco not installed"); return
    if not CXML.exists():
        _skip("no collision model"); return

    m = mujoco.MjModel.from_xml_path(str(CXML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body = [i for i in range(m.ngeom) if int(m.geom_bodyid[i]) != 0]

    # the keep-out volumes the guard builds: one oriented AABB per link
    by_body = {}
    for i in body:
        by_body.setdefault(int(m.geom_bodyid[i]), []).append(i)
    links = []
    for gids in by_body.values():
        meshes = [i for i in gids if int(m.geom_type[i]) == 7]
        for i in (meshes or gids):
            ab = m.geom_aabb[i]
            links.append((i, np.array(ab[:3], float), np.array(ab[3:], float)))
    assert links, "no keep-out volumes built"

    def obb_of(gi, ctr):
        M = np.array(d.geom_xmat[gi], float)
        c = np.array(d.geom_xpos[gi], float) + M.reshape(3, 3) @ ctr
        return c, list(M)

    def new_blocks(o):
        return any(_obb_hit(obb_of(gi, ctr)[0], obb_of(gi, ctr)[1], half, o)
                   for gi, ctr, half in links)

    def old_blocks(o):                      # the retired bounding-sphere test
        p, s = np.array(o["p"], float), np.array(o["s"], float)
        for gi in body:
            rr = _old_bound_r(int(m.geom_type[gi]), m.geom_size[gi])
            g = np.maximum(np.abs(np.array(d.geom_xpos[gi], float) - p) - s, 0.0)
            if float(g @ g) < rr * rr:
                return True
        return False

    # -- COVERS: a probe on any link AABB corner must block ------------------
    probe = 0.005                            # a 1 cm keep-out cube
    missed_old, checked = 0, 0
    for gi, ctr, half in links:
        c, M = obb_of(gi, ctr)
        R = np.array(M, float).reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = c + R @ (half * np.array([sx, sy, sz], float))
                    o = {"type": "box", "p": list(corner), "s": [probe] * 3}
                    checked += 1
                    assert new_blocks(o), \
                        f"a keep-out box on a link corner must block (geom {gi})"
                    if not old_blocks(o):
                        missed_old += 1
    assert missed_old > 0, \
        ("the retired sphere test blocked every link corner too — then this "
         "change bought no coverage and the claim needs rewriting")

    # -- SHEDS: sampled blocked fraction over the robot's bounding volume ----
    lo = np.array([1e9] * 3); hi = np.array([-1e9] * 3)
    for gi in body:
        rr = _old_bound_r(int(m.geom_type[gi]), m.geom_size[gi])
        c = np.array(d.geom_xpos[gi], float)
        lo = np.minimum(lo, c - rr); hi = np.maximum(hi, c + rr)
    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, hi, size=(1500, 3))
    n_new = sum(new_blocks({"type": "box", "p": list(p), "s": [0.0, 0.0, 0.0]})
                for p in pts)
    n_old = sum(old_blocks({"type": "box", "p": list(p), "s": [0.0, 0.0, 0.0]})
                for p in pts)
    assert n_old > 0
    assert n_new < 0.5 * n_old, \
        f"expected the oriented boxes to shed most of the phantom volume: {n_new} vs {n_old}"
    print(f"PASS keep-out: {checked} link corners all blocked "
          f"({missed_old} of them missed by the retired sphere test); "
          f"blocked sample {n_old} -> {n_new} of {len(pts)} "
          f"({100.0 * (n_old - n_new) / n_old:.1f}% of the reserved volume shed)")
