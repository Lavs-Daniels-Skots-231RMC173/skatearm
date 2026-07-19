"""M1 — wrist force/torque sensor calibration.

The `ee_{left,right}_force` / `ee_{left,right}_torque` sensors added to the
control MJCF must read a known external wrench at the wrist: at static
equilibrium the joint reaction the sensor reports equals the negative of the
applied load (gravity cancels via the no-load baseline). This is the M1
foundation — a true 6-axis wrist wrench, replacing the actuator-torque proxy.

Headless; needs mujoco + the control MJCF (set SKT_DIR to your skt_v3 folder).

    SKT_DIR=.../skt_v3 python3 sim/test_ft_sensor.py
"""
import os
from pathlib import Path

import numpy as np

HANDS = {"ee_left": "wrist_a3_1", "ee_right": "wrist_a3_Mirror__1"}


def _load():
    try:
        import mujoco
    except ImportError:
        return None, None
    xml = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3")) / "skt_v3_control.xml"
    if not xml.exists():
        return None, None
    return mujoco, mujoco.MjModel.from_xml_path(str(xml))


def _settle(mujoco, m, d, nmax=6000, tol=1e-5):
    """Step to static equilibrium (servos hold ctrl, damping bleeds off qvel)."""
    for k in range(nmax):
        mujoco.mj_step(m, d)
        if k > 50 and float(np.max(np.abs(d.qvel))) < tol:
            return k
    return nmax


def _reset(d):
    d.qpos[:] = 0
    d.qvel[:] = 0
    d.ctrl[:] = 0
    d.xfrc_applied[:] = 0


def _world(m, d, name):
    """A wrist sensor's 3-vector rotated from the site frame into world."""
    v = np.array(d.sensor(name).data)
    R = np.array(d.site(name.rsplit("_", 1)[0]).xmat).reshape(3, 3)
    return R @ v


def test_ft_sensor_reads_applied_force():
    mujoco, m = _load()
    if m is None:
        print("SKIP: mujoco / control model not available"); return
    d = mujoco.MjData(m)
    for site, body in HANDS.items():
        hid = m.body(body).id
        _reset(d); _settle(mujoco, m, d)
        f0 = _world(m, d, site + "_force")   # no-load baseline (gravity)
        for Fw in ([0, 0, -10.0], [10.0, 0, 0], [0, 8.0, 0], [5.0, -5.0, -5.0]):
            _reset(d)
            d.xfrc_applied[hid, :3] = Fw
            _settle(mujoco, m, d)
            delta = _world(m, d, site + "_force") - f0
            err = float(np.linalg.norm(delta + np.array(Fw)))   # expect delta == -Fw
            assert err < 0.05, f"{site}: applied {Fw} N, sensor delta {delta}, err {err:.3f} N"
    print("PASS ft-force: both wrists read applied loads to < 0.05 N")


def test_ft_sensor_reads_applied_torque():
    mujoco, m = _load()
    if m is None:
        print("SKIP: mujoco / control model not available"); return
    d = mujoco.MjData(m)
    site, body = "ee_right", HANDS["ee_right"]
    hid = m.body(body).id
    _reset(d); _settle(mujoco, m, d)
    t0 = _world(m, d, site + "_torque")
    for Tw in ([0, 0, 1.5], [1.0, 0, 0], [0, -1.2, 0]):
        _reset(d)
        d.xfrc_applied[hid, 3:6] = Tw
        _settle(mujoco, m, d)
        delta = _world(m, d, site + "_torque") - t0
        err = float(np.linalg.norm(delta + np.array(Tw)))   # expect delta == -Tw
        assert err < 0.05, f"applied torque {Tw} N·m, sensor delta {delta}, err {err:.3f} N·m"
    print("PASS ft-torque: wrist reads applied torque to < 0.05 N·m")


if __name__ == "__main__":
    test_ft_sensor_reads_applied_force()
    test_ft_sensor_reads_applied_torque()
    print("FT-SENSOR TEST DONE")
