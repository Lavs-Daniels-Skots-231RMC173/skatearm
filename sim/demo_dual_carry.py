"""Bimanual dual-arm CO-LIFT of one shared object, with gravity feed-forward
and load-sharing.

Why co-lift and not "squeeze": the two Skate arms cannot bring their wrists
closer than ~17 cm in x (they self-collide), so an object can't be pinched
between them. The feasible bimanual carry holds one object at the arms' natural
separation and co-lifts it; the regulated quantity is the load each arm carries
(kept balanced), the analogue of internal-force regulation in the feasible z
direction.

Pipeline (builds on the Phase-1 primitives): both wrists weld-grasp one bar
(pinned to the world during the approach so the light bar can't be knocked off,
then handed to the wrists), then co-lift + carry. A gravity-only feed-forward
(mj_rne with qvel=0 -> no Coriolis, stable) cancels arm sag so the wrists track
their targets. Renders docs/img/dual_carry.gif.

Run:  py sim/demo_dual_carry.py  <model_dir>   (model_dir holds skt_v3_collision.xml)
"""
import sys, os, numpy as np, mujoco
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import primitives as P

MODEL_DIR = (sys.argv[1] if len(sys.argv) > 1
             else os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
SCENE = os.path.join(MODEL_DIR, "skt_v3_carry.xml")
GIF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img", "dual_carry.gif")
GY, GZ, WX = 0.392, 0.16, 0.085      # grasp y, z; natural wrist x-separation (measured)


def build_scene(model_dir=None):
    model_dir = model_dir or MODEL_DIR
    scene = os.path.join(model_dir, "skt_v3_carry.xml")
    tree = ET.parse(os.path.join(model_dir, "skt_v3_collision.xml"))
    root = tree.getroot()
    def fb(n):
        for b in root.iter("body"):
            if b.get("name") == n:
                return b
    for wn in ("wrist_a3_1", "wrist_a3_Mirror__1"):
        b = fb(wn)
        for g in b.findall("geom"):
            g.set("contype", "0"); g.set("conaffinity", "0")
        pad = ET.SubElement(b, "geom")
        pad.set("type", "sphere"); pad.set("size", "0.016"); pad.set("pos", "0 0 0")
        pad.set("rgba", "0.25 0.62 1 1"); pad.set("contype", "0"); pad.set("conaffinity", "0")
    wb = root.find("worldbody")
    extra = f"""
<geom name="pfloor" type="plane" pos="0 0 -1.05" size="4 4 0.1" rgba="0.12 0.13 0.15 1"/>
<light pos="0.5 -0.3 1.6" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7"/>
<light pos="-0.6 1.0 1.4" dir="0.3 -0.5 -1" diffuse="0.35 0.35 0.4"/>
<camera name="front" pos="0 1.05 0.42" zaxis="0 1 0.27" fovy="45"/>
<geom name="support" type="box" pos="0 {GY} 0.075" size="0.10 0.02 0.075" rgba="0.3 0.31 0.34 1"/>
<body name="bar" pos="0 {GY} 0.17"><freejoint/>
  <geom name="bar_g" type="box" size="{WX} 0.011 0.011" rgba="0.88 0.55 0.18 1" density="500"/></body>
"""
    for el in list(ET.fromstring("<r>" + extra + "</r>")):
        wb.append(el)
    eq = ET.SubElement(root, "equality")
    for nm, wn in (("wL", "wrist_a3_1"), ("wR", "wrist_a3_Mirror__1")):
        w = ET.SubElement(eq, "weld"); w.set("name", nm); w.set("body1", wn)
        w.set("body2", "bar"); w.set("active", "false"); w.set("solref", "0.02 1")
    wf = ET.SubElement(eq, "weld"); wf.set("name", "wfix"); wf.set("body1", "world")
    wf.set("body2", "bar"); wf.set("active", "false"); wf.set("solref", "0.005 1")
    tree.write(scene)
    return scene


def weld_loads(m, d):
    out = []
    for nm in ("wL", "wR"):
        e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, nm); s = 0.0
        for i in range(d.nefc):
            if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == e:
                s += d.efc_force[i] ** 2
        out.append(s ** 0.5)
    return out[0], out[1]


def engage(m, d, nm):
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, nm)
    b1, b2 = m.eq_obj1id[eq], m.eq_obj2id[eq]
    R1 = d.xmat[b1].reshape(3, 3)
    p_rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
    q1i = np.zeros(4); mujoco.mju_negQuat(q1i, d.xquat[b1].copy())
    qr = np.zeros(4); mujoco.mju_mulQuat(qr, q1i, d.xquat[b2].copy())
    m.eq_data[eq, :] = 0; m.eq_data[eq, 3:6] = p_rel; m.eq_data[eq, 6:10] = qr; m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


def main():
    scene = build_scene(MODEL_DIR)
    m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
    bar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bar")
    bd0, bdn = m.body_dofadr[bar], m.body_dofnum[bar]
    gmask = np.ones(m.nv); gmask[bd0:bd0 + bdn] = 0.0; gbuf = np.zeros(m.nv)
    for _ in range(250):
        mujoco.mj_step(m, d)
    engage(m, d, "wfix")                                   # pin the bar while we approach
    arms = {s: P.Arm(m, d, s) for s in ("left", "right")}
    ren = mujoco.Renderer(m, 480, 640); frames = []
    bz = float(d.xpos[bar][2]); M = np.array([0.0, GY, bz + 0.009]); z0 = float(M[2])
    bias = {"left": 0.0, "right": 0.0}

    def step(rec=False):
        tgt = {"left": [M[0] - WX, M[1], M[2] + bias["left"]],
               "right": [M[0] + WX, M[1], M[2] + bias["right"]]}
        for s, a in arms.items():
            q, _ = a.ik_step(np.array(tgt[s])); a.set_ctrl(q)
        qv = d.qvel.copy(); d.qvel[:] = 0.0; mujoco.mj_rne(m, d, 0, gbuf); d.qvel[:] = qv
        for _ in range(4):
            d.qfrc_applied[:] = gbuf * gmask; mujoco.mj_step(m, d)
        if rec:
            ren.update_scene(d, camera="front"); frames.append(ren.render().copy())

    P.move_joints(m, d, {"a0": 0.3, "a1": 0.3, "a3": 0.8}, seconds=1.5)
    for _ in range(320):
        step()
    engage(m, d, "wL"); engage(m, d, "wR")                 # both arms grasp the bar
    d.eq_active[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "wfix")] = 0   # release the pin
    for i in range(420):                                   # co-lift + carry, load balanced
        s = P.smoothstep(i / 419)
        M[2] = z0 + 0.11 * s; M[0] = 0.06 * s
        fl, fr = weld_loads(m, d)
        corr = 0.011 * np.tanh((fl - fr) * 0.4)            # admittance: equalise the per-arm load
        bias["left"] = -corr; bias["right"] = corr
        step(rec=(i % 9 == 0))
    for _ in range(90):
        step(rec=(len(frames) % 1 == 0))

    print("final bar z %.3f (lifted +%.0f mm), held: %s" % (d.xpos[bar][2], (d.xpos[bar][2] - bz) * 1000, d.xpos[bar][2] > 0.24))
    os.makedirs(os.path.dirname(GIF), exist_ok=True)
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(GIF, save_all=True, append_images=imgs[1:], duration=55, loop=0)
    print("wrote", GIF, "(", len(frames), "frames )")


if __name__ == "__main__":
    main()
