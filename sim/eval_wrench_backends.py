"""M1 eval — the two backends that can feed a wrist wrench, measured against
each other on the SAME settled state.

M1 asks that the wrench interface be backed by either a real wrist F/T cell or
a joint-torque estimator. Both exist here, so the honest thing is to measure the
gap rather than call them equivalent:

  * SENSOR — the M1 site force sensor, rotated into world exactly as
    ``sim/test_ft_sensor.py`` and ``sim_endpoint._wrist_wrench()`` do (the two
    are pinned bit-for-bit by ``tools/skate_ros2/test/test_wrist_wrench.py``);
  * ESTIMATE — the cockpit's own ``RobotBridge._tcp_force``, called here on the
    joint state the sim endpoint actually puts on the wire (``qpos[:26]`` and
    ``actuator_force[:26]``), so this column is the cockpit code path, not a
    re-implementation of it.

A known load is applied to the hand body, the model is stepped to static
equilibrium, and each backend's DELTA from its own no-load baseline is compared
against the reaction the wrist must carry (-F). Two poses, because the estimator
is a function of arm conditioning and one pose would flatter or damn it
unfairly: the home pose (q = 0, near-singular) and a working pose. sigma_min of
the 3xN position Jacobian is printed at each, so a bad number can be attributed
to conditioning instead of hand-waved.

    python eval_wrench_backends.py --model /path/to/skt_v3
                                   [--json sim/eval_data/wrench_backends.json]

``--json`` writes the same numbers as a committed artefact, so the figures quoted
in ``docs/MANIPULATION.md`` have raw data behind them (``sim/test_manipulation_numbers.py``
pins the prose to that file in CI).
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "skate_commander"))
from skate_commander.bridge import RobotBridge      # noqa: E402
from skate_commander.kinematics import ArmKinematics  # noqa: E402
from skate_commander.urdf import parse_urdf          # noqa: E402

N_JOINTS = 26
HANDS = {"left": "wrist_a3_1", "right": "wrist_a3_Mirror__1"}
SITES = {"left": "ee_left", "right": "ee_right"}
LOADS = ([0.0, 0.0, -10.0], [10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [5.0, -5.0, -5.0])


class _State:
    """The two telemetry fields ``RobotBridge._tcp_force`` reads, taken straight
    off MjData -- the same arrays ``sim_endpoint.send_telemetry()`` packs."""

    def __init__(self, d):
        self.q = np.array(d.qpos[:N_JOINTS])
        self.tau = np.array(d.actuator_force[:N_JOINTS])

    def dof_pos(self):
        return self.q

    def dof_torque(self):
        return self.tau


def load_control(model_dir):
    xml = os.path.join(model_dir, "skt_v3_control.xml")
    if not os.path.exists(xml):
        raise SystemExit(f"no control MJCF at {xml} "
                         f"(run sim/make_control_model.py {model_dir} first)")
    m = mujoco.MjModel.from_xml_path(xml)
    urdf = os.path.join(model_dir, "skt_v3.urdf")
    um = parse_urdf(urdf)
    bridge = RobotBridge.__new__(RobotBridge)        # only ``kin`` is needed
    bridge.kin = {arm: ArmKinematics(um, arm) for arm in HANDS}
    return m, bridge


def settle(m, d, nmax=8000, tol=1e-5):
    """Step to static equilibrium (servos hold ctrl, damping bleeds off qvel)."""
    for k in range(nmax):
        mujoco.mj_step(m, d)
        if k > 50 and float(np.max(np.abs(d.qvel))) < tol:
            return k
    return nmax


def sensor_world(d, arm):
    """The wrist force sensor in world -- ``sim_endpoint._wrist_wrench()``'s math."""
    site = SITES[arm]
    R = np.array(d.site(site).xmat).reshape(3, 3)
    return R @ np.array(d.sensor(site + "_force").data)


def sigma_min(bridge, q, arm):
    _p, J = bridge.kin[arm]._fk_jac_fast(q)
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def probe(m, bridge, q0, tag):
    """One pose: baselines, then every load through both backends."""
    d = mujoco.MjData(m)

    def reset():
        d.qpos[:] = 0
        d.qvel[:] = 0
        d.ctrl[:] = 0
        d.xfrc_applied[:] = 0
        d.qpos[:N_JOINTS] = q0
        d.ctrl[:N_JOINTS] = q0

    out = {"pose": tag, "arms": {}}
    for arm, body in HANDS.items():
        hid = m.body(body).id
        reset()
        settle(m, d)
        f0 = sensor_world(d, arm)
        e0 = np.array(bridge._tcp_force(_State(d))[arm]["f"])
        smin = sigma_min(bridge, d.qpos[:N_JOINTS], arm)
        print(f"\n  {arm:<5} wrist   sigma_min(J) = {smin:.4f}")
        print(f"    no-load baseline   sensor {np.round(f0, 4)}   "
              f"estimate {np.round(e0, 4)}")

        rows = []
        for Fw in LOADS:
            reset()
            d.xfrc_applied[hid, :3] = Fw
            settle(m, d)
            want = -np.array(Fw)                 # the reaction the wrist carries
            ds = sensor_world(d, arm) - f0
            de = np.array(bridge._tcp_force(_State(d))[arm]["f"]) - e0
            es = float(np.linalg.norm(ds - want))
            ee = float(np.linalg.norm(de - want))
            rows.append({"load_n": list(Fw),
                         "sensor_delta_n": [round(v, 3) for v in ds.tolist()],
                         "sensor_err_n": round(es, 3),
                         "estimate_delta_n": [round(v, 3) for v in de.tolist()],
                         "estimate_err_n": round(ee, 3)})
            print(f"    load {str(np.array(Fw)):<20}  sensor {np.round(ds, 3)} "
                  f"err {es:6.3f} N   |   estimate {np.round(de, 3)} err {ee:7.3f} N")

        out["arms"][arm] = {
            "sigma_min_j": round(smin, 4),
            "baseline_sensor_n": [round(v, 4) for v in f0.tolist()],
            "baseline_estimate_n": [round(v, 4) for v in e0.tolist()],
            "rows": rows,
            "sensor_err_max_n": round(max(r["sensor_err_n"] for r in rows), 3),
            "estimate_err_max_n": round(max(r["estimate_err_n"] for r in rows), 3),
        }
    return out


def working_q(bridge):
    """A pose off the singular home stack, the same one the cockpit force tests
    use: each arm's dofs swept over 0.2 .. 0.8 rad."""
    q = np.zeros(N_JOINTS)
    for kin in bridge.kin.values():
        idx = np.asarray(kin.idx, dtype=int)
        q[idx] = np.linspace(0.2, 0.8, len(idx))
    return q


def main():
    ap = argparse.ArgumentParser(
        description="M1 eval: wrist F/T sensor vs joint-torque estimate")
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="also write the results as a JSON artefact")
    args = ap.parse_args()

    m, bridge = load_control(args.model)
    print("wrist wrench backends -- delta from each backend's own no-load "
          "baseline vs the reaction -F the wrist must carry\n")

    print("HOME POSE (q = 0)")
    home = probe(m, bridge, np.zeros(N_JOINTS), "home")
    print("\nWORKING POSE (q[arm] = linspace(0.2, 0.8))")
    work = probe(m, bridge, working_q(bridge), "working")

    out = {"eval": "wrench_backends", "milestone": "M1",
           "source": "sim/eval_wrench_backends.py",
           "loads_n": [list(f) for f in LOADS],
           "poses": {"home": home, "working": work}}

    worst = max(a["estimate_err_max_n"]
                for p in out["poses"].values() for a in p["arms"].values())
    best_sensor = max(a["sensor_err_max_n"]
                      for p in out["poses"].values() for a in p["arms"].values())
    print(f"\nworst estimate error {worst:.3f} N over {len(LOADS)} loads x 2 arms "
          f"x 2 poses; worst sensor error {best_sensor:.3f} N")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
