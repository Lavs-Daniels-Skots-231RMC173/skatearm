"""M4 (scoped) — parallel-jaw gripper controller.

Drives the coupled prismatic jaws (make_gripper_scene) with a force (motor)
actuator whose command IS the grasp force: at equilibrium against the part the
jaw contact normal force equals the commanded motor force, read back through a
touch sensor on the pad. `close_to_force` ramps to a target grasp force; the part
is then held by FRICTION alone (the scene's world-pin is released first), no weld.

Sim ↔ hardware: a real parallel-jaw gripper is force- (or current-) commanded the
same way; only the grasp-force *source* differs (motor current / a pad load cell).
"""
import numpy as np
import mujoco


class Gripper:
    def __init__(self, m, d, substeps=4):
        self.m, self.d = m, d
        self.aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip")
        self._f = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "grip_force")]
        self._j = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "jaw")]
        self.pin = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "pin")
        self.part = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part")
        self.substeps = substeps

    # --- signals ---
    def grasp_force(self):
        """Measured grasp normal force at the pad (N), from the touch sensor."""
        return float(self.d.sensordata[self._f])

    def jaw(self):
        """Jaw joint position (m); larger = more closed."""
        return float(self.d.sensordata[self._j])

    def part_pos(self):
        return self.d.xpos[self.part].copy()

    # --- actions ---
    def _run(self, cycles):
        for _ in range(cycles * self.substeps):
            mujoco.mj_step(self.m, self.d)

    def set_pin(self, active):
        """Toggle the temporary world-pin that holds the part while the jaws
        close (released to test the friction grasp)."""
        self.d.eq_active[self.pin] = 1 if active else 0
        mujoco.mj_forward(self.m, self.d)

    def open(self, force=8.0, cycles=50):
        self.d.ctrl[self.aid] = -abs(force)          # negative motor = open
        self._run(cycles)

    def close_to_force(self, f_target, kp=1.5, cycles=300):
        """Grasp-force control: close until first contact, then continuously
        regulate the *measured* grasp force (touch sensor) to `f_target` with an
        integral law on the motor command — the motor force and the grasp normal
        force differ by a geometry/coupling factor, so the loop closes on what's
        measured, not on the raw command. Continuous regulation (no fixed-command
        hold) cancels the steady-state offset from the part settling into the pads.
        Returns the settled measured grasp force."""
        cmd = 0.0
        while self.grasp_force() < 0.3 and cmd < 60.0:      # phase 1: reach contact
            cmd += 1.0
            self.d.ctrl[self.aid] = cmd
            self._run(1)
        for _ in range(cycles):                             # phase 2: regulate to target
            cmd = float(np.clip(cmd + kp * (f_target - self.grasp_force()), 0.0, 60.0))
            self.d.ctrl[self.aid] = cmd
            self._run(1)
        return self.grasp_force()

    def holds(self, z_ref, drop_tol=0.03):
        """True if the part is still grasped (has not dropped below z_ref by more
        than drop_tol)."""
        return bool(self.part_pos()[2] > z_ref - drop_tol)

    def slip_payload(self, z_grip, ramp=0.25, step_n=40, max_pull=30.0):
        """Grasp-slip probe: with the part already grasped (pin released), ramp a
        downward payload force on it until it escapes the jaws, and return the
        payload (N) at which it slips out — the grasp's holding capacity. Returns
        None if it holds all the way to `max_pull`. Higher grasp force -> higher
        slip payload (flat pad-on-face contact makes the hold scale with force)."""
        escape_z = z_grip - 0.045
        pull = 0.0
        while pull < max_pull:
            pull += ramp
            self.d.xfrc_applied[self.part, 2] = -pull
            for _ in range(step_n):
                mujoco.mj_step(self.m, self.d)
            if self.part_pos()[2] < escape_z:
                self.d.xfrc_applied[self.part, :] = 0.0
                return round(pull, 2)
        self.d.xfrc_applied[self.part, :] = 0.0
        return None
