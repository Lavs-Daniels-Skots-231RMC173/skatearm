"""M4 (scoped) — generate skt_v3_gripper.xml: a self-contained actuated
parallel-jaw gripper grasping a rectangular part by FRICTION (no weld).

The opt-in M4 slice: it proves the gripper *mechanism* — two pads on coupled
prismatic joints, a force (motor) actuator whose command sets the grasp force,
real friction and a touch sensor reading the grasp normal force — without
touching the arm's control/collision/cell models (so every existing test stays
green). The part is pinned to the world by an equality weld only while the jaws
close; releasing the pin leaves it held by friction alone. Flat pad-on-face
contact (a box part) makes the friction hold scale with the grasp force, so the
grasp-slip curve (payload-until-slip vs grasp force) is physically sensible. The
full M4 (jaws on the wrist's a7, weld-free carry through the cycle) builds on this.

    python make_gripper_scene.py /path/to/skate_teleop/skt_v3
"""
import os
import sys

# geometry (m): the grasped part is a rectangular workpiece; jaws open ~32 mm.
PART_HX, PART_HY, PART_HZ = 0.010, 0.020, 0.025    # box part half-sizes (x = grip axis)
JAW_HW = 0.004                     # pad half-width along the closing (x) axis
JAW_OPEN = 0.020                   # each jaw body's rest offset from centre (±)
MOUNT_Z = 0.30
MU = "1.6 0.2 0.02"               # tangential + torsional + rolling friction


def scene_xml():
    return f"""<mujoco model="skt_gripper">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="1024" offheight="768"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <default>
    <geom solref="0.01 1" solimp="0.9 0.95 0.001" condim="4"/>
  </default>
  <worldbody>
    <light pos="0.2 0.2 1" dir="-0.2 -0.2 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" rgba="0.85 0.86 0.88 1"/>
    <body name="mount" pos="0 0 {MOUNT_Z}">
      <geom type="box" size="0.03 0.022 0.01" rgba="0.4 0.4 0.45 1"
            contype="0" conaffinity="0"/>
      <body name="jawL" pos="-{JAW_OPEN} 0 -0.03">
        <joint name="jL" type="slide" axis="1 0 0" range="-0.005 0.03" damping="4"/>
        <geom name="padL" type="box" size="{JAW_HW} 0.02 0.02"
              rgba="0.2 0.6 0.9 1" friction="{MU}"/>
        <site name="padL_s" type="box" size="{JAW_HW+0.002} 0.021 0.021"/>
      </body>
      <body name="jawR" pos="{JAW_OPEN} 0 -0.03">
        <joint name="jR" type="slide" axis="1 0 0" range="-0.03 0.005" damping="4"/>
        <geom name="padR" type="box" size="{JAW_HW} 0.02 0.02"
              rgba="0.2 0.6 0.9 1" friction="{MU}"/>
        <site name="padR_s" type="box" size="{JAW_HW+0.002} 0.021 0.021"/>
      </body>
    </body>
    <body name="part" pos="0 0 {MOUNT_Z-0.03}">
      <freejoint/>
      <geom name="part" type="box" size="{PART_HX} {PART_HY} {PART_HZ}"
            rgba="0.9 0.6 0.15 1" density="950" friction="{MU}"/>
    </body>
  </worldbody>
  <equality>
    <joint name="couple" joint1="jR" joint2="jL" polycoef="0 -1 0 0 0"/>
    <weld name="pin" body1="part" active="true" solref="0.005 1"/>
  </equality>
  <actuator>
    <motor name="grip" joint="jL" ctrlrange="-60 60" gear="1"/>
  </actuator>
  <sensor>
    <touch name="grip_force" site="padL_s"/>
    <jointpos name="jaw" joint="jL"/>
  </sensor>
</mujoco>
"""


def make(model_dir, out_name="skt_v3_gripper.xml"):
    out = os.path.join(model_dir, out_name)
    with open(out, "w") as f:
        f.write(scene_xml())
    import mujoco
    m = mujoco.MjModel.from_xml_path(out)          # compile check
    print(f"wrote {out}: {m.nbody} bodies, {m.nu} actuator(s), {m.nsensor} sensors")
    return out


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else ".")
