"""Optional rerun.io logging for the Skate Commander cockpit (opt-in).

Streams the live twin into a [rerun](https://rerun.io) viewer — a scrub-able
3D + time-series debugger beside the browser cockpit. Per tick it logs the
**full meshed robot** in 3D (each link's mesh, moved by its live pose — the same
robot as the browser twin), both **TCP** points, active **drag-IK targets**, the
user's **virtual obstacles**, and a tree of **time-series** (per-arm joint angle
& velocity, drag-IK position / orientation error, manipulability). Rerun's
timeline then lets you scrub the whole run.

Fully optional and non-breaking. Heavy imports (`rerun`, `mujoco`) live inside
the constructor, the server builds it behind a flag (`--rerun` / `SKATE_RERUN=1`)
wrapped in try/except, and every `log()` call swallows its own errors — so a
missing `rerun-sdk` or a viewer hiccup can never disturb the cockpit tick loop.
Off by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_ARM = {"left": list(range(8, 15)), "right": list(range(16, 23))}
_C_LINK = [190, 200, 212]
_C_TCP = {"left": [16, 185, 129], "right": [245, 158, 11]}
_C_IK = [239, 68, 68]
_C_OBS = [180, 110, 50]


def _vertex_normals(verts, faces):
    """Smooth per-vertex normals from a triangle mesh — MuJoCo mesh geoms carry
    no per-vertex normals, and without them rerun renders the mesh flat/unlit
    (a see-through silhouette); accumulating face normals fixes the shading."""
    n = np.zeros(verts.shape, dtype=np.float32)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(ln, 1e-9)).astype(np.float32)


class RerunLogger:
    """Streams cockpit state to a rerun viewer. Build once (opens / connects the
    viewer and logs the static link meshes) and call :meth:`log` from the tick
    loop; throttled by ``every``."""

    def __init__(self, collision_xml, spawn=True, app_id="skate_commander", every=2):
        import mujoco          # optional deps, imported lazily so the module
        import rerun as rr     # only costs anything when --rerun is on
        self._mj, self._rr = mujoco, rr
        if spawn:
            self._ensure_viewer_on_path()
        rr.init(app_id, spawn=spawn)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # prefer the control model (full visual meshes) over the collision model;
        # both carry the same 26-DoF qpos + ee_left/ee_right sites.
        cx = Path(str(collision_xml))
        ctrl = cx.parent / "skt_v3_control.xml"
        self.model = mujoco.MjModel.from_xml_path(str(ctrl if ctrl.exists() else cx))
        self.data = mujoco.MjData(self.model)
        self._jnames = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                        for i in range(self.model.njnt)]
        self.every = max(1, int(every))
        self._n = 0
        # Log each link mesh ONCE (static, in its own frame); only the per-tick
        # Transform3D moves it — the full robot, not a stick figure.
        self._mesh_geoms = []
        m = self.model
        for i in range(m.ngeom):
            if int(m.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mid = int(m.geom_dataid[i])
            v0, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            f0, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            verts = np.asarray(m.mesh_vert[v0:v0 + vn], dtype=np.float32)
            faces = np.asarray(m.mesh_face[f0:f0 + fn], dtype=np.int64)
            if faces.size and int(faces.max()) >= vn:        # global indices -> local
                faces = faces - v0
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i]))
            path = f"robot/{bn}_{i}"
            normals = _vertex_normals(verts, faces)
            try:
                rr.log(path, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                       vertex_normals=normals, albedo_factor=_C_LINK),
                       static=True)
            except Exception:
                rr.log(path, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                       vertex_normals=normals), static=True)
            self._mesh_geoms.append((path, i))

    def log(self, bridge, snap):
        """Log one frame of cockpit state. Never raises — a logging error must
        not disturb the tick loop, so the whole body is guarded."""
        self._n += 1
        if self._n % self.every:
            return
        try:
            self._log(bridge, snap)
        except Exception:
            pass

    def _log(self, bridge, snap):
        rr, mj = self._rr, self._mj
        q = snap.get("q")
        if q is None:
            return
        q = np.asarray(q, dtype=float)
        rr.set_time("tick", sequence=self._n)

        # --- full meshed robot: move each link mesh by its live world pose ---
        d = self.data
        d.qpos[:26] = q[:26]
        mj.mj_forward(self.model, d)
        for path, gi in self._mesh_geoms:
            rr.log(path, rr.Transform3D(translation=d.geom_xpos[gi],
                                        mat3x3=d.geom_xmat[gi].reshape(3, 3)))

        # --- TCP points + active drag-IK targets ---
        for arm, kin in getattr(bridge, "kin", {}).items():
            try:
                rr.log(f"tcp/{arm}", rr.Points3D([kin.fk(q).tolist()],
                       radii=0.02, colors=_C_TCP.get(arm)))
            except Exception:
                pass
            tgt = bridge.ik_targets.get(arm)
            pts = [] if tgt is None else [np.asarray(tgt, dtype=float).tolist()]
            rr.log(f"ik_target/{arm}", rr.Points3D(pts, radii=0.024, colors=_C_IK))

        # --- user virtual obstacles as boxes ---
        centers, halfs = [], []
        for o in (snap.get("obstacles") or []):
            p, s = o.get("p"), o.get("s")
            if p and s:
                centers.append([float(p[0]), float(p[1]), float(p[2])])
                halfs.append([float(s[0]), float(s[0]), float(s[1])]
                             if o.get("type") == "cyl"
                             else [float(s[0]), float(s[1]), float(s[2])])
        rr.log("obstacles", rr.Boxes3D(centers=centers, half_sizes=halfs, colors=_C_OBS))

        # --- time-series: per-arm joint angle & velocity ---
        dq = snap.get("dq")
        dq = None if dq is None else np.asarray(dq, dtype=float)
        for arm, idxs in _ARM.items():
            for i in idxs:
                nm = self._jnames[i]
                rr.log(f"plots/q/{arm}/{nm}", rr.Scalars(float(q[i])))
                if dq is not None:
                    rr.log(f"plots/dq/{arm}/{nm}", rr.Scalars(float(dq[i])))
        # --- time-series: drag-IK error + manipulability per arm ---
        ik, iko, man = snap.get("ik") or {}, snap.get("ik_ori") or {}, snap.get("manip") or {}
        for arm in ("left", "right"):
            if ik.get(arm) is not None:
                rr.log(f"plots/ik_err_mm/{arm}", rr.Scalars(float(ik[arm]) * 1000.0))
            if iko.get(arm) is not None:
                rr.log(f"plots/ik_ori_deg/{arm}", rr.Scalars(float(iko[arm])))
            if man.get(arm) is not None:
                rr.log(f"plots/manip/{arm}", rr.Scalars(float(man[arm])))

    @staticmethod
    def _ensure_viewer_on_path():
        """pip installs the ``rerun`` viewer binary as a console script next to
        the interpreter (…\\Scripts on Windows, …/bin elsewhere), which is often
        NOT on PATH — so ``rr.init(spawn=True)`` can't find it. Prepend that dir
        when the binary is there and not already discoverable."""
        import os
        import shutil
        import sys
        if shutil.which("rerun") is not None:
            return
        d = Path(sys.executable).parent / ("Scripts" if os.name == "nt" else "bin")
        exe = d / ("rerun.exe" if os.name == "nt" else "rerun")
        if exe.exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    def close(self):
        try:
            self._rr.disconnect()
        except Exception:
            pass
