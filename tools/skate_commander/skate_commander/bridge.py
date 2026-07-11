"""RobotBridge — the commander's connection to a Skate (sim or real).

Wraps :class:`skate_ros2.protocol.SkateLink` with the same safety model the
skate_ros2 driver uses, plus a jog engine for the UI:

* SIM/REAL are the same wire — switching mode just retargets the UDP host
  (127.0.0.1 vs r.local). After ANY mode switch the bridge is DAMPENED and
  must be explicitly resumed from the UI.
* Arms at the robot's measured pose; jog input before arming is ignored.
* Hold-to-jog integrates at the tx rate, clamped to URDF joint limits.
* Deadman (1,1,1) only while a UI client is attached, resumed, and not
  estopped/overtemp — close the browser tab and the robot dampens itself.

No ROS, no FastAPI here: pure logic, fully testable headless.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# prototype-phase import of the sibling tool (documented in README)
_SKATE_ROS2 = Path(__file__).resolve().parents[2] / "skate_ros2"
if str(_SKATE_ROS2) not in sys.path:
    sys.path.insert(0, str(_SKATE_ROS2))

from skate_ros2 import names                    # noqa: E402
from skate_ros2.protocol import SkateLink       # noqa: E402

from . import planner                           # noqa: E402
from .kinematics import rot_error               # noqa: E402

OVERTEMP_C = 58.0          # PETG limit, per official docs
OVERTEMP_RELEASE_C = 53.0


def _quat_to_R(q):
    """three.js quaternion [x, y, z, w] -> 3x3 rotation matrix (world)."""
    x, y, z, w = (float(v) for v in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class RobotBridge:
    def __init__(self, sim_host="127.0.0.1", sim_port=2000,
                 real_host="r.local", real_port=2000,
                 jog_rate=0.35, limits=None, kin=None,
                 jog_accel=3.0, seq_accel=2.0, contact_tau=8.0):
        """jog_rate: rad/s while a jog button is held.
        jog_accel: rad/s^2 ramp so hold-to-jog eases in and (on release)
            eases out instead of snapping — no motion jerk.
        seq_accel: rad/s^2 ease-in/out for waypoint / replay glides.
        contact_tau: N·m torque spike (over the slow baseline) on a barely-
            moving arm joint that trips the contact reflex (soft-stop) — the
            signature of a joint pushing into an obstacle. Grippers are
            excluded; a grasp legitimately spikes torque.
        limits: optional (lo[26], hi[26]) from the URDF; clamps jog targets."""
        self.sim_addr = (sim_host, sim_port)
        self.real_addr = (real_host, real_port)
        self.mode = "sim"                  # "sim" | "real"
        self.link = SkateLink(*self.sim_addr)
        self.jog_rate = float(jog_rate)
        self.speed_scale = 1.0             # global velocity override (0.1-1.0), teach-pendant style
        self.paused = False        # sim-transport: freeze autonomous motion
        self._step_ticks = 0       # >0 = advance N autonomous ticks while paused
        self.jog_accel = float(jog_accel)   # rad/s^2 smooth jog ramp
        if limits is not None:
            self.lo = np.asarray(limits[0], dtype=float)
            self.hi = np.asarray(limits[1], dtype=float)
        else:
            self.lo = np.full(names.N_JOINTS, -np.pi)
            self.hi = np.full(names.N_JOINTS, np.pi)

        self.targ = None                   # armed target (np[26]) or None
        self.estop = True                  # latched; starts SAFE (dampened)
        # OBSERVE: pure-observer mode — telemetry keeps flowing (heartbeats
        # only) but the cockpit never transmits a command, so it can attach
        # alongside an external commander (e.g. the ROS 2 / MoveIt driver)
        # on a shared sim endpoint without fighting it for the deadman.
        self.observe = False
        self.ext_vel_eps = 0.05            # rad/s arm motion = external drive
        self._ext_until = 0.0              # EXTERNAL chip hold (anti-flicker)
        self._ext_joints = np.r_[names.LEFT_ARM, names.RIGHT_ARM]
        self.overtemp = False
        # contact reflex: an unexpected torque spike on the arm chain (vs a
        # slow per-joint baseline) latches a soft-stop — motion dampens like
        # the overtemp latch and is cleared by an explicit operator reset.
        self.contact_reflex = True
        # A contact = a torque spike on an arm joint WHILE that joint is barely
        # moving (pushing into something that won't yield). Gating on low
        # velocity is what separates a real block from a fast COMMANDED move,
        # which is also high-torque but high-speed (measured on the sim: a
        # slider step saturates torque at ~10 N·m while slewing at 3–5 rad/s,
        # whereas the normal-motion spike floor is ~1.3 N·m).
        self.contact_tau = float(contact_tau)     # N·m spike over the baseline
        self.contact_vel_eps = 0.3                # rad/s; below this = "not slewing"
        self.contact_hold = 2                     # consecutive ticks before a trip
        self.contact_alpha = 0.2                  # per-tick EMA rate of the baseline
        self.contact_tripped = False              # latched soft-stop
        self.contact_joint = None                 # joint that tripped (for the UI)
        self._tau_ref = None                      # per-joint torque baseline (np[26])
        self._contact_run = 0                     # consecutive blocked-tick counter
        _cmask = np.zeros(names.N_JOINTS, dtype=bool)
        _cmask[names.LEFT_ARM] = True             # 8..15
        _cmask[names.RIGHT_ARM] = True            # 16..23
        _cmask[names.LEFT_GRIPPER] = False        # a grasp spikes torque on purpose
        _cmask[names.RIGHT_GRIPPER] = False
        self.contact_mask = _cmask                # arm structural joints only
        self.jog_dir = np.zeros(names.N_JOINTS)   # -1/0/+1 per joint
        self.jog_vel = np.zeros(names.N_JOINTS)   # rad/s, accel-limited toward jog_dir*rate
        self.kin = kin or {}               # {"left"/"right": ArmKinematics}
        self.mink = None                   # optional MinkIK backend (opt-in; DLS is the fallback)
        self.guard = None                  # callable(q26)->True if SELF-COLLIDING
        self.guard_blocking = False
        self.ik_targets = {"left": None, "right": None}
        self.ik_err = {"left": None, "right": None}
        self.ik_manip = {"left": None, "right": None}   # singularity awareness
        self.ik_targR = {"left": None, "right": None}   # optional 6-DoF orientation (3x3)
        self.ik_oerr = {"left": None, "right": None}    # orientation error, deg
        # cartesian-step targets auto-clear on arrival (or stall); drag-gizmo
        # targets are cleared by the UI on release
        self.ik_auto = {"left": False, "right": False}
        self._ik_prev = {"left": None, "right": None}
        self._ik_stall = {"left": 0.0, "right": 0.0}
        # comfort posture for the IK null space: the documented default pose
        # nudged 30% toward each joint's range center. While an IK target is
        # active the arm continuously relaxes toward this shape WITHOUT
        # moving the TCP — the robot picks its own elbow configuration, no
        # winding, no limit-hugging.
        self.ik_comfort = (0.7 * np.array(names.DEFAULT_POSE)
                           + 0.3 * 0.5 * (self.lo + self.hi))
        # mirror mode: commands to one arm are reflected onto the other.
        # signs/axis are derived numerically from FK by the server (no
        # guessing about the URDF's mirroring convention).
        self.mirror = False
        self.mirror_signs = None           # per-arm-slot sign array, len 8
        self.mirror_axis = 1               # world axis flipped by the mirror
        self.carry = False                 # dual-arm carry: both wrists move one object together
        self.tool_names = {"left": "flange", "right": "flange"}
        # waypoint sequencer
        self.waypoints = []                # list of np[26] (commanded poses)
        self.wp_names = []
        self.recorder = None               # optional teach-in PoseRecorder
        self.ds_rec = None                 # optional dense LeRobotDataset recorder
        self.seq_active = False           # gliding toward waypoints[seq_idx]
        self.seq_playing = False          # auto-advance through the list
        self.seq_loop = False
        self.seq_idx = None
        self.seq_rate = 0.6               # rad/s cruise glide speed
        self.seq_accel = float(seq_accel)  # rad/s^2 ease-in/out for glides
        self.seq_vel = 0.0                # current glide speed (trapezoidal)
        self.seq_dwell = 0.6              # s pause on each waypoint
        self._seq_wait = 0.0
        self.seq_route = None            # planned route to the active waypoint
        self.seq_route_idx = 0           # (None=plan needed, []=dwelling, [..]=following)
        self.seq_route_verified = False  # route came from a successful plan (guard-safe)
        self._seq_prev = None            # leg progress watch (stall give-up)
        self._seq_stall = 0.0
        self._last_tx = None
        # home(): glide to the safe default pose with the SAME jerk-limited
        # trapezoidal profile as the waypoint glides — not an instant snap.
        self.home_pose = np.array(names.DEFAULT_POSE, dtype=float)
        self.home_goal = self.home_pose.copy()   # per-call target (legs preserved)
        self.home_active = False
        self.home_vel = 0.0
        self._home_prev = None             # last remaining-distance (stall watch)
        self._home_stall = 0.0             # s the glide has made no progress
        # planned (RRT) route — a list of waypoint configs the glide follows to
        # get AROUND a self-collision the straight path can't pass. None = idle.
        self.plan_nodes = None
        self.plan_idx = 0
        self.plan_vel = 0.0
        self._plan_prev = None
        self._plan_stall = 0.0
        self.plan_joints = np.arange(8, 24)   # default planning set: arm chains
        self.obstacles = []                # user-placed virtual obstacles the planner avoids
        self._obs_id = 0

    # -- mode / safety -------------------------------------------------------
    def set_mode(self, mode):
        if mode not in ("sim", "real"):
            raise ValueError(mode)
        if mode == self.mode:
            return
        self.link.close()
        addr = self.sim_addr if mode == "sim" else self.real_addr
        self.link = SkateLink(*addr)
        self.mode = mode
        self.targ = None                   # re-arm from the new robot's pose
        self.jog_dir[:] = 0
        self.jog_vel[:] = 0
        self.ik_targets = {"left": None, "right": None}
        self.carry = False
        self.estop = True                  # explicit resume required

    def set_speed(self, scale):
        """Global velocity override (teach-pendant): clamp to 0.1-1.0 and
        scale jog + sequence-glide cruise speeds."""
        try:
            self.speed_scale = float(max(0.1, min(1.0, scale)))
        except (TypeError, ValueError):
            pass
        return self.speed_scale

    # motion-feel tuning ranges (cockpit-side, NOT robot firmware gains)
    _TUNE_RANGES = {
        "jog_rate":    (0.05, 1.5),    # rad/s   — jog cruise
        "jog_accel":   (0.5, 12.0),    # rad/s^2 — jog ramp
        "seq_rate":    (0.1, 1.5),     # rad/s   — glide cruise (home/waypoint/replay)
        "seq_accel":   (0.5, 8.0),     # rad/s^2 — glide ramp
        "contact_tau": (2.0, 25.0),    # N·m     — contact-reflex trip threshold (lower = touchier)
    }

    def set_tuning(self, **kw):
        """Live motion-feel tuning: jog + glide cruise speed and ramp, plus the
        contact-reflex sensitivity. Each known key is clamped to a safe range;
        unknown keys are ignored. Read every tick, so changes apply at once."""
        for k, (lo, hi) in self._TUNE_RANGES.items():
            if kw.get(k) is not None:
                try:
                    setattr(self, k, float(max(lo, min(hi, float(kw[k])))))
                except (TypeError, ValueError):
                    pass
        return self.tuning()

    def tuning(self):
        """Current motion-tuning values (for the cockpit panel)."""
        return {"jog_rate": round(self.jog_rate, 3),
                "jog_accel": round(self.jog_accel, 2),
                "seq_rate": round(self.seq_rate, 3),
                "seq_accel": round(self.seq_accel, 2),
                "contact_tau": round(self.contact_tau, 2)}

    def set_paused(self, on):
        """Sim-transport pause: freeze autonomous motion (sequences / home /
        plan glides) so the arm holds its pose. Manual jog / IK still work."""
        self.paused = bool(on)
        if not self.paused:
            self._step_ticks = 0
        return self.paused

    def set_show_collision(self, on):
        """Toggle the collision-mesh overlay — when on, snapshot() carries the
        guard model's geom world-poses (+ per-geom hit flags) for the client."""
        self.show_collision = bool(on)
        return self.show_collision

    def set_show_force(self, on):
        """Toggle the TCP-force overlay — when on, snapshot() carries each arm's
        TCP force estimate (N, world frame) derived from the joint torques."""
        self.show_force = bool(on)
        return self.show_force

    def add_obstacle(self, shape, p, s):
        """Add a user-placed virtual obstacle the planner/guard avoids."""
        self._obs_id += 1
        o = {"id": self._obs_id,
             "type": ("cyl" if shape == "cyl" else "box"),
             "p": [float(p[0]), float(p[1]), float(p[2])],
             "s": [float(v) for v in s]}
        self.obstacles.append(o)
        return o

    def clear_obstacles(self):
        self.obstacles = []
        return True

    def delete_obstacle(self, oid):
        self.obstacles = [o for o in self.obstacles if o.get("id") != oid]
        return True

    def update_obstacle(self, oid, p):
        for o in self.obstacles:
            if o.get("id") == oid and p:
                o["p"] = [float(p[0]), float(p[1]), float(p[2])]
                return True
        return False

    def resize_obstacle(self, oid, s):
        """Set a user obstacle's half-extents (m); clamped 0.01..0.75 (2..150 cm full)."""
        if not s or len(s) != 3:
            return False
        for o in self.obstacles:
            if o.get("id") == oid:
                o["s"] = [min(0.75, max(0.01, float(v))) for v in s]
                return True
        return False

    def _tcp_force(self, st):
        """Estimate each arm's TCP force (N, world frame) from the joint
        torques: tau = Jᵀ·F  ⇒  F = (J·Jᵀ)⁻¹·J·tau, using the 3×N position
        Jacobian. Position-only (no moments), matching the kinematics; reflects
        gravity-hold + any contact, like a teach-pendant's TCP-force readout."""
        tau = st.dof_torque()
        q = st.dof_pos()
        if tau is None or q is None:
            return None
        tau = np.asarray(tau, dtype=float)
        out = {}
        for arm, kin in self.kin.items():
            p, J = kin._fk_jac_fast(q)
            ta = tau[np.asarray(kin.idx, dtype=int)]
            JJt = J @ J.T + 1e-9 * np.eye(3)
            try:
                F = np.linalg.solve(JJt, J @ ta)
            except np.linalg.LinAlgError:
                F = np.zeros(3)
            out[arm] = {"p": [round(float(v), 4) for v in p],
                        "f": [round(float(v), 3) for v in F],
                        "mag": round(float(np.linalg.norm(F)), 2)}
        return out

    def step(self, n=1):
        """Advance paused autonomous motion by n ticks (Isaac single-step)."""
        try:
            self._step_ticks += max(1, int(n))
        except (TypeError, ValueError):
            self._step_ticks += 1
        return self._step_ticks

    def trigger_estop(self):
        self.estop = True
        self.jog_dir[:] = 0
        self.jog_vel[:] = 0          # safety stop is immediate, no easing
        self.carry = False
        self.clear_ik_target()
        self.seq_stop()

    def resume(self):
        """Clear the estop latch. Only possible once telemetry is up, and
        never while OBSERVE holds the cockpit passive."""
        if self.link.connected and not self.observe:
            self.estop = False
        return not self.estop

    def set_observe(self, on):
        """Toggle pure-observer mode. Both transitions land in E-STOP with
        the target re-armed at the measured pose, so neither entering nor
        leaving OBSERVE can move the robot — an explicit Resume is always
        required to command again."""
        on = bool(on)
        if on == self.observe:
            return self.observe
        self.observe = on
        self.estop = True
        self.targ = None                   # re-arm from the measured pose
        self.jog_dir[:] = 0
        self.jog_vel[:] = 0
        self.carry = False
        self.clear_ik_target()
        self.seq_stop()
        self._ext_until = 0.0
        return self.observe

    # -- contact reflex --------------------------------------------------------
    def _contact_update(self, tau, vel):
        """Return True if an arm joint is BLOCKED: a torque spike beyond
        ``contact_tau`` N·m over its slow baseline WHILE the joint is barely
        moving (``|vel| < contact_vel_eps``) — the signature of pushing into an
        obstacle, not of a fast commanded move (high-torque but high-speed). A
        per-joint EMA baseline normalises out steady holding / gravity torque,
        and the condition must persist ``contact_hold`` ticks to reject brief
        settle transients. Pure function of ``(tau, vel)`` + internal state;
        grippers are excluded (a grasp spikes torque on purpose)."""
        if tau is None or vel is None:
            self._contact_run = 0
            return False
        tau = np.asarray(tau, dtype=float)
        vel = np.asarray(vel, dtype=float)
        if tau.shape != (names.N_JOINTS,) or vel.shape != (names.N_JOINTS,):
            return False
        if self._tau_ref is None:
            self._tau_ref = tau.copy()
            self._contact_run = 0
            return False                     # need a baseline first
        spike = np.abs(tau - self._tau_ref)
        blocked = ((spike > self.contact_tau)
                   & (np.abs(vel) < self.contact_vel_eps)
                   & self.contact_mask)
        if np.any(blocked):
            self._contact_run += 1
            self.contact_joint = int(np.argmax(np.where(blocked, spike, -np.inf)))
        else:
            self._contact_run = 0
        # EMA the baseline AFTER the test: a sustained load becomes the new
        # normal, but the first ticks of a real contact still trip.
        self._tau_ref += self.contact_alpha * (tau - self._tau_ref)
        return self._contact_run >= self.contact_hold

    def _trip_contact(self):
        """Latch the contact soft-stop: dampen motion immediately (like estop)
        but as a separate, operator-clearable latch."""
        self.contact_tripped = True
        self.jog_dir[:] = 0
        self.jog_vel[:] = 0
        self.carry = False
        self.clear_ik_target()
        self.seq_stop()                      # also cancels a home glide

    def clear_contact(self):
        """Operator acknowledges the contact and clears the latch. Motion stays
        dampened until it goes live again (ui attached, not estopped, ...)."""
        self.contact_tripped = False
        self.contact_joint = None
        self._tau_ref = None                 # re-baseline from the next sample
        self._contact_run = 0

    # -- collision guard -------------------------------------------------------
    def _guard_ok(self, prev):
        """Reject self-colliding targets: revert to ``prev`` if the new targ
        would interpenetrate. Large jumps (slider clicks, goto) are checked
        along the interpolated path so a collision can't be tunnelled through
        between two safe endpoints."""
        if self.guard is None or prev is None:
            return True
        delta = np.abs(self.targ - prev)
        n = max(1, int(np.ceil(float(delta.max()) / 0.05)))
        goal = self.targ.copy()
        for k in range(1, n + 1):
            q = prev + (goal - prev) * (k / n)
            if self.guard(q):
                self.targ = prev
                self.guard_blocking = True
                return False
        self.guard_blocking = False
        return True

    # -- waypoint sequencer ----------------------------------------------------
    def seq_stop(self):
        self.seq_active = False
        self.seq_playing = False
        self._seq_wait = 0.0
        self.seq_vel = 0.0
        self.seq_route = None            # drop any in-progress leg route
        self.seq_route_verified = False
        self.home_active = False         # a home glide is autonomous motion too
        self.home_vel = 0.0
        self._home_stall = 0.0
        self._home_prev = None
        self.plan_nodes = None           # cancel any planned route too
        self.plan_vel = 0.0
        self._plan_stall = 0.0
        self._plan_prev = None

    def wp_add(self):
        """Record the current commanded pose as a waypoint."""
        if self.targ is not None:
            self.waypoints.append(self.targ.copy())
            self.wp_names.append(f"WP{len(self.waypoints)}")

    def wp_delete(self, i):
        if 0 <= i < len(self.waypoints):
            self.waypoints.pop(i)
            self.wp_names.pop(i)
            self.seq_stop()
            self.seq_idx = None

    def wp_clear(self):
        self.waypoints = []
        self.wp_names = []
        self.seq_stop()
        self.seq_idx = None

    def wp_goto(self, i):
        """Glide to one waypoint, no auto-advance."""
        if 0 <= i < len(self.waypoints) and self.targ is not None \
                and not self.estop:
            self.seq_idx = i
            self.seq_active = True
            self.seq_playing = False
            self.seq_route = None              # plan this leg on the next tick

    def wp_play(self, loop=False):
        if self.waypoints and self.targ is not None and not self.estop:
            self.seq_idx = 0
            self.seq_loop = bool(loop)
            self.seq_active = True
            self.seq_playing = True
            self._seq_wait = 0.0
            self.seq_route = None              # plan each leg on the next tick

    def _glide(self, goal, vel, dt):
        """One jerk-limited trapezoidal step of ``targ`` toward ``goal``:
        ease in to ``seq_rate``, then sqrt-decelerate to a clean stop
        (v <= sqrt(2*a*d) so it can always brake within the remaining
        distance). Returns ``(new_vel, remaining_before_step)``. Shared by the
        waypoint sequencer and home() so both ease in and out identically."""
        remaining = float(np.max(np.abs(goal - self.targ)))
        vel = min(self.seq_rate * self.speed_scale, vel + self.seq_accel * dt)
        vel = min(vel, float(np.sqrt(2.0 * self.seq_accel * max(remaining, 0.0))))
        step = vel * dt
        self.targ = np.clip(self.targ + np.clip(goal - self.targ, -step, step),
                            self.lo, self.hi)
        return vel, remaining

    def _home_tick(self, dt):
        """Glide ``targ`` to the safe default pose with the shared jerk-limited
        profile; self-clears on arrival. Gives up (no error) if the path stays
        guard-blocked or otherwise stops improving — the straight joint-space
        route to the default pose can clip a self-collision the guard won't
        pass (routing around it is a planner's job, backlog). Cancelled by any
        manual input / estop / dampening (all route through seq_stop)."""
        if not self.home_active:
            return
        self.home_vel, remaining = self._glide(self.home_goal, self.home_vel, dt)
        if remaining < 0.01:                                # arrived
            self.home_vel = 0.0
            self.home_active = False
            self._home_stall = 0.0
            self._home_prev = None
            return
        # not improving (guard reverts the step / clamped out of reach) -> quit
        if self._home_prev is not None and self._home_prev - remaining < 1e-5:
            self._home_stall += dt
        else:
            self._home_stall = 0.0
        self._home_prev = remaining
        if self._home_stall > 0.8:
            self.home_active = False
            self.home_vel = 0.0
            self._home_stall = 0.0
            self._home_prev = None

    # -- planned (collision-free) routing --------------------------------------
    def plan_path(self, q_goal, joints=None):
        """Plan a collision-free joint path from the current target to
        ``q_goal``, routing around self-collisions the guard would reject.
        Returns a list of configs (endpoints included) or None. With no guard
        the straight line is always clear, so this returns ``[targ, goal]``."""
        if self.targ is None:
            return None
        q_goal = np.asarray(q_goal, dtype=float)
        cfree = ((lambda q: not self.guard(q)) if self.guard is not None
                 else (lambda q: True))
        js = self.plan_joints if joints is None else joints
        return planner.plan(self.targ, q_goal, cfree, self.lo, self.hi,
                            joints=js)

    def _start_plan(self, path):
        """Begin following a planned route (node 0 == the current targ)."""
        self.plan_nodes = [np.asarray(p, dtype=float) for p in path[1:]]
        self.plan_idx = 0
        self.plan_vel = 0.0
        self._plan_prev = None
        self._plan_stall = 0.0

    def _plan_tick(self, dt):
        """Glide targ through the planned route node by node (shared trapezoid
        profile). Gives up gracefully if a segment stalls, like the home glide."""
        if self.plan_nodes is None:
            return
        goal = self.plan_nodes[self.plan_idx]
        self.plan_vel, remaining = self._glide(goal, self.plan_vel, dt)
        if remaining < 0.01:                             # node reached
            self.plan_vel = 0.0
            self._plan_prev = None
            self._plan_stall = 0.0
            self.plan_idx += 1
            if self.plan_idx >= len(self.plan_nodes):
                self.plan_nodes = None                   # route complete
            return
        if self._plan_prev is not None and self._plan_prev - remaining < 1e-5:
            self._plan_stall += dt
        else:
            self._plan_stall = 0.0
        self._plan_prev = remaining
        if self._plan_stall > 0.8:
            self.plan_nodes = None                       # blocked -> give up
            self.plan_vel = 0.0

    def _seq_tick(self, dt):
        """Glide targ to the active waypoint, routing AROUND a self-collision
        when the straight leg is guard-blocked (same planner as home); advance
        through the list when playing. ``seq_route``: None = plan this leg,
        [] = dwelling at the reached waypoint, [..] = following the route."""
        if not self.seq_active or self.seq_idx is None \
                or self.seq_idx >= len(self.waypoints):
            return
        # dwelling at a reached waypoint (playback) before the next leg
        if self.seq_route is not None and len(self.seq_route) == 0:
            self._seq_wait += dt
            if self._seq_wait < self.seq_dwell:
                return
            self._seq_wait = 0.0
            nxt = self.seq_idx + 1
            if nxt >= len(self.waypoints):
                if self.seq_loop:
                    self.seq_idx, self.seq_route = 0, None
                else:
                    self.seq_stop()
            else:
                self.seq_idx, self.seq_route = nxt, None
            return
        # plan the leg the first time we head for this waypoint (a clear leg
        # comes back as a 2-point straight line, a blocked one as a detour)
        if self.seq_route is None:
            path = self.plan_path(self.waypoints[self.seq_idx])
            self.seq_route = ([np.asarray(p, dtype=float) for p in path[1:]]
                              if path is not None and len(path) > 1
                              else [np.asarray(self.waypoints[self.seq_idx],
                                               dtype=float)])
            self.seq_route_verified = path is not None      # guard-safe if planned
            self.seq_route_idx = 0
            self.seq_vel = 0.0
            self._seq_prev = None
            self._seq_stall = 0.0
        sub = self.seq_route[self.seq_route_idx]
        self.seq_vel, remaining = self._glide(sub, self.seq_vel, dt)
        if remaining < 0.01:                                # route node reached
            self.seq_vel = 0.0
            self._seq_prev = None
            self._seq_stall = 0.0
            self.seq_route_idx += 1
            if self.seq_route_idx < len(self.seq_route):
                return                                      # next route node
            if not self.seq_playing:                        # waypoint reached
                self.seq_active = False
                self.seq_route = None
            else:
                self.seq_route = []                         # enter the dwell
            return
        # a fallback (un-routable) leg that the guard keeps reverting gives up
        if self._seq_prev is not None and self._seq_prev - remaining < 1e-5:
            self._seq_stall += dt
        else:
            self._seq_stall = 0.0
        self._seq_prev = remaining
        if self._seq_stall > 0.8:
            self.seq_stop()                                 # blocked leg -> stop

    # -- mirror mode -----------------------------------------------------------
    def _mirror_joint(self, idx):
        """(other_idx, sign) if idx is an arm joint and mirroring is ready."""
        if self.mirror_signs is None:
            return None
        if 8 <= idx < 16:
            other = idx + 8
        elif 16 <= idx < 24:
            other = idx - 8
        else:
            return None
        return other, float(self.mirror_signs[idx % 8])

    def _mirror_vec(self, v):
        v = np.asarray(v, dtype=float).copy()
        v[self.mirror_axis] = -v[self.mirror_axis]
        return v

    def _mirror_rot(self, R):
        """Reflect an orientation across the same plane _mirror_vec uses.
        Conjugation by the axis reflection M (det -1) keeps M R M a proper
        rotation, so the mirrored wrist frame stays valid."""
        M = np.eye(3)
        M[self.mirror_axis, self.mirror_axis] = -1.0
        return M @ np.asarray(R, dtype=float) @ M

    def _joint_locked(self, idx):
        """Lower chain is untouchable in REAL mode — balance belongs to the
        firmware. (The UI greys it out; this enforces it for programs too.)"""
        return self.mode == "real" and 0 <= idx < 8

    # -- jog input from the UI -----------------------------------------------
    def jog_start(self, idx, direction):
        if 0 <= idx < names.N_JOINTS and not self._joint_locked(idx):
            self.seq_stop()                # manual input overrides playback
            d = 1.0 if direction > 0 else -1.0
            self.jog_dir[idx] = d
            if self.mirror and (m := self._mirror_joint(idx)):
                self.jog_dir[m[0]] = m[1] * d

    def jog_stop(self, idx=None):
        if idx is None:
            self.jog_dir[:] = 0
        elif 0 <= idx < names.N_JOINTS:
            self.jog_dir[idx] = 0
            if self.mirror and (m := self._mirror_joint(idx)):
                self.jog_dir[m[0]] = 0

    def jog_step(self, idx, delta):
        """Single click: move target by delta radians."""
        if self.targ is None or self.estop or self._joint_locked(idx):
            return
        self.seq_stop()
        prev = self.targ.copy()
        self.targ[idx] = float(np.clip(self.targ[idx] + delta,
                                       self.lo[idx], self.hi[idx]))
        if self.mirror and (m := self._mirror_joint(idx)):
            o, s = m
            self.targ[o] = float(np.clip(self.targ[o] + s * delta,
                                         self.lo[o], self.hi[o]))
        self._guard_ok(prev)

    def set_joint(self, idx, value):
        """Absolute target for one joint (slider input)."""
        if self.targ is None or self.estop or self._joint_locked(idx):
            return
        self.seq_stop()
        prev = self.targ.copy()
        self.targ[idx] = float(np.clip(value, self.lo[idx], self.hi[idx]))
        if self.mirror and (m := self._mirror_joint(idx)):
            o, s = m
            self.targ[o] = float(np.clip(s * value, self.lo[o], self.hi[o]))
        self._guard_ok(prev)

    def set_joints(self, items):
        """Bulk absolute targets: [(idx, rad), ...] applied as ONE pose with
        a single guard check. Deliberately NOT mirrored — an absolute
        multi-joint pose (e.g. a recorded teach-in keypose) already says
        where every joint goes. Returns False if the guard rejected it."""
        if self.targ is None or self.estop:
            return False
        self.seq_stop()
        prev = self.targ.copy()
        for idx, value in items:
            if 0 <= idx < names.N_JOINTS and not self._joint_locked(idx):
                self.targ[idx] = float(np.clip(value,
                                               self.lo[idx], self.hi[idx]))
        return self._guard_ok(prev)

    def _ik_one(self, arm, pos, auto, rot=None):
        self.ik_targets[arm] = np.asarray(pos, dtype=float)
        self.ik_targR[arm] = None if rot is None else np.asarray(rot, dtype=float)
        self.ik_auto[arm] = bool(auto)
        self._ik_prev[arm] = None
        self._ik_stall[arm] = 0.0

    def _ik_solve(self, arm, target, tR):
        """One IK glide step for ``arm``: use the mink backend when enabled
        (transparently falling back to the numpy DLS on any solver error), else
        the DLS directly. Same ``(new_q26, pos_err)`` contract either way, so
        the rest of tick() is unchanged."""
        if self.mink is not None:
            try:
                return self.mink.step(arm, self.targ, target,
                                      target_R=tR, q_ref=self.ik_comfort)
            except Exception:
                pass
        return self.kin[arm].ik_step(self.targ, target, target_R=tR,
                                     q_ref=self.ik_comfort)

    def set_ik_target(self, arm, pos, auto=False, rot=None):
        """Drag-gizmo target (world meters) + optional orientation (a three.js
        quaternion [x,y,z,w] or a 3x3 matrix). ``rot=None`` is the classic
        position-only (free wrist) solve; a quaternion makes it a full 6-DoF
        pose target. Ignored unless armed & resumed."""
        if (arm in self.kin and self.targ is not None and not self.estop
                and len(pos) == 3):
            self.seq_stop()                # manual input overrides playback
            R = None
            if rot is not None:
                R = _quat_to_R(rot) if len(rot) == 4 else np.asarray(rot, dtype=float)
            self._ik_one(arm, pos, auto, R)
            if self.mirror:
                other = "right" if arm == "left" else "left"
                if other in self.kin:
                    self._ik_one(other, self._mirror_vec(pos), auto,
                                 None if R is None else self._mirror_rot(R))

    def reachable(self, arm, pos, tol=0.015, iters=160):
        """IK feasibility test: can ``arm`` put its TCP at world point ``pos``
        (metres)? Runs a hypothetical IK solve (does NOT move the robot) from
        the current pose and reports the residual. None if arm/model absent."""
        if arm not in self.kin or self.targ is None or len(pos) != 3:
            return None
        q = np.array(self.targ, dtype=float)
        tgt = np.asarray(pos, dtype=float)
        err = 1e9
        for _ in range(int(iters)):
            q, err = self.kin[arm].ik_step(q, tgt, step_m=0.08,
                                           q_ref=self.ik_comfort)
            if err < tol:
                break
        final = float(np.linalg.norm(tgt - self.kin[arm].fk(q)))
        return {"reachable": bool(final < tol), "err_mm": round(final * 1000.0, 1)}

    def cart_step(self, arm, delta):
        """Cartesian nudge: glide the TCP by ``delta`` (world meters) via IK.
        The target auto-clears once reached (or once it stops improving, e.g.
        out of reach / blocked by the guard)."""
        if (arm not in self.kin or self.targ is None or self.estop
                or len(delta) != 3):
            return
        self.seq_stop()
        delta = np.asarray(delta, dtype=float)
        self._cart_one(arm, delta)
        if self.mirror:
            other = "right" if arm == "left" else "left"
            if other in self.kin:
                self._cart_one(other, self._mirror_vec(delta))

    def _cart_one(self, arm, delta):
        base = self.ik_targets[arm]
        if base is None:
            base = self.kin[arm].fk(self.targ)
        self._ik_one(arm, np.asarray(base, dtype=float) + delta, auto=True)

    def _clear_one(self, arm):
        if arm in self.ik_targets:
            self.ik_targets[arm] = None
            self.ik_targR[arm] = None
            self.ik_err[arm] = None
            self.ik_oerr[arm] = None
            self.ik_manip[arm] = None
            self.ik_auto[arm] = False
            self._ik_prev[arm] = None
            self._ik_stall[arm] = 0.0

    def clear_ik_target(self, arm=None):
        if self.mirror:
            arm = None                     # mirrored arms stop together
        for a in ([arm] if arm else ["left", "right"]):
            self._clear_one(a)

    # -- dual-arm carry --------------------------------------------------------
    def carry_grab(self):
        """Hold one virtual object with BOTH wrists: pin each EE at its current
        pose. carry_step then translates both targets together — a rigid
        two-handed carry (not mirrored), at the arms' natural separation."""
        if (self.targ is None or self.estop
                or "left" not in self.kin or "right" not in self.kin):
            return
        self.seq_stop()
        self.mirror = False                # carry drives both arms explicitly
        self.carry = True
        for a in ("left", "right"):
            self._ik_one(a, self.kin[a].fk(self.targ), auto=False)

    def carry_step(self, delta):
        """Translate the held object by ``delta`` (world meters): both wrists
        move by the SAME delta, preserving their separation. Guard-checked in
        tick() like any IK move."""
        if not self.carry or self.targ is None or self.estop or len(delta) != 3:
            return
        self.seq_stop()
        delta = np.asarray(delta, dtype=float)
        for a in ("left", "right"):
            base = self.ik_targets[a]
            if base is None:
                base = self.kin[a].fk(self.targ)
            self._ik_one(a, np.asarray(base, dtype=float) + delta, auto=False)

    def carry_release(self):
        self.carry = False
        self.clear_ik_target()

    # -- tool / TCP offsets ----------------------------------------------------
    def set_tool(self, arm, name, offset_m):
        """Attach a named TCP offset (wrist-link frame, meters) to one arm.
        Any active IK target is cleared — its meaning just changed."""
        if arm in self.kin and len(offset_m) == 3:
            self.kin[arm].tool = np.asarray(offset_m, dtype=float)
            self.tool_names[arm] = str(name)
            self._clear_one(arm)

    def home(self):
        """Send the target to the documented safe default pose with the
        jerk-limited trapezoidal profile. If the straight joint path is clear
        it glides directly; if a self-collision blocks it, an RRT planner
        routes AROUND it; if even that fails, the direct glide gives up
        gracefully. Any manual input / estop / dampening cancels it."""
        if self.targ is None or self.estop:
            return
        self.seq_stop()                # clears any seq / prior home / plan
        self.carry = False
        self.clear_ik_target()         # don't let an IK target fight the glide
        # Home brings the ARMS (+head) to the documented safe pose and leaves
        # the leg / balance chain exactly where it is — that's the firmware's
        # job, and not re-routing the legs keeps the plan to just the arms.
        goal = self.targ.copy()
        goal[8:] = self.home_pose[8:]              # arms + head; legs [0:8] kept
        self.home_goal = goal
        path = self.plan_path(goal)
        if path is not None and len(path) > 2:
            self._start_plan(path)     # route around the self-collision
        else:
            self.home_active = True    # straight line clear, or give up gracefully
            self.home_vel = 0.0
            self._home_prev = None
            self._home_stall = 0.0

    # -- periodic work (call at ~60 Hz from the server loop) -------------------
    def tick(self, dt, ui_attached):
        """Poll telemetry, integrate jog, send one command. Returns state."""
        self.link.poll()
        st = self.link.state

        pos = st.dof_pos()
        if pos is not None and self.targ is None:
            self.targ = np.asarray(pos, dtype=float).copy()  # arm at pose

        temps = st.motor_temps()
        if temps is not None:
            tmax = max(temps)
            if not self.overtemp and tmax > OVERTEMP_C:
                self.overtemp = True
            elif self.overtemp and tmax < OVERTEMP_RELEASE_C:
                self.overtemp = False

        if self.observe:
            # Pure observer: the heartbeat inside link.poll() keeps telemetry
            # flowing, but no command is EVER transmitted — the external
            # commander (e.g. the ROS 2 / MoveIt driver) owns the deadman.
            # Track the measured pose so leaving OBSERVE re-arms exactly
            # where the robot is, and surface external motion for the UI.
            self.jog_dir[:] = 0
            self.jog_vel[:] = 0
            if pos is not None:
                self.targ = np.asarray(pos, dtype=float).copy()
            dq = st.dof_vel()
            if dq is not None:
                ext = float(np.max(np.abs(
                    np.asarray(dq, dtype=float)[self._ext_joints])))
                if ext > self.ext_vel_eps:
                    self._ext_until = time.monotonic() + 0.6
            self._tau_ref = None
            self._contact_run = 0
            if self.recorder is not None:
                self.recorder.observe(self.targ, dt)   # teach-in still works
            if self.ds_rec is not None:
                self.ds_rec.observe(self.targ, dt)     # dense LeRobot capture
            return self.snapshot(ui_attached)

        # contact reflex: while otherwise-live, an unexpected arm-torque spike
        # latches a soft-stop. When not otherwise-live the baseline is dropped
        # so a dampened arm settling can't false-trip on the next resume.
        base_ok = (ui_attached and not self.estop and not self.overtemp
                   and self.targ is not None and self.link.connected)
        if self.contact_reflex and base_ok and not self.contact_tripped:
            if self._contact_update(st.dof_torque(), st.dof_vel()):
                self._trip_contact()
        elif not base_ok:
            self._tau_ref = None
            self._contact_run = 0

        live = base_ok and not self.contact_tripped

        if self.targ is not None:
            prev = self.targ.copy()
            if live:
                # acceleration-limited jog: chase jog_dir*rate so a held jog
                # eases in, and on release eases out, instead of snapping.
                target_vel = self.jog_dir * self.jog_rate * self.speed_scale
                dv = self.jog_accel * dt
                self.jog_vel += np.clip(target_vel - self.jog_vel, -dv, dv)
                if self.jog_vel.any():
                    self.targ = np.clip(self.targ + self.jog_vel * dt,
                                        self.lo, self.hi)
                for arm, target in self.ik_targets.items():
                    if target is not None and arm in self.kin:
                        tR = self.ik_targR.get(arm)
                        self.targ, err = self._ik_solve(arm, target, tR)
                        self.ik_err[arm] = err
                        self.ik_manip[arm] = self.kin[arm].manipulability(self.targ)
                        if tR is not None:
                            _, Rc = self.kin[arm].fk_pose(self.targ)
                            self.ik_oerr[arm] = round(float(np.degrees(
                                np.linalg.norm(rot_error(Rc, tR)))), 1)
                        else:
                            self.ik_oerr[arm] = None
                        if self.ik_auto.get(arm):      # cart-step target
                            p = self._ik_prev[arm]
                            if p is not None and p - err < 1e-5:
                                self._ik_stall[arm] += dt    # not improving
                            else:
                                self._ik_stall[arm] = 0.0
                            self._ik_prev[arm] = err
                            if err < 0.003 or self._ik_stall[arm] > 0.8:
                                self._clear_one(arm)         # arrived/stuck
                if not self.paused or self._step_ticks > 0:
                    self._seq_tick(dt)
                    self._home_tick(dt)
                    self._plan_tick(dt)
                    if self._step_ticks > 0:
                        self._step_ticks -= 1
                # a planner route is already collision-verified at the guard's
                # own resolution; re-checking it per-tick (at the finer glide
                # step) only false-stalls on grazing corners. Guard only the
                # un-verified moves (jog / slider / IK / give-up fallback).
                verified = (self.plan_nodes is not None
                            or (self.seq_route is not None and self.seq_route_verified))
                if not np.array_equal(self.targ, prev) and not verified:
                    self._guard_ok(prev)
            else:
                self.jog_vel[:] = 0        # not live: drop jog velocity
                if self.seq_active or self.home_active or self.plan_nodes is not None:
                    self.seq_stop()        # dampened: autonomous motion must not resume
            deadman = (1, 1, 1) if live else (0, 0, 0)
            self.link.send_command(self.targ, deadman=deadman)
            self._last_tx = time.monotonic()
        if self.recorder is not None:
            self.recorder.observe(self.targ, dt)
        if self.ds_rec is not None:
            self.ds_rec.observe(self.targ, dt)
        return self.snapshot(ui_attached)

    # -- state for the UI ------------------------------------------------------
    def snapshot(self, ui_attached=True):
        st = self.link.state
        return {
            "mode": self.mode,
            "connected": self.link.connected,
            "armed": self.targ is not None,
            "estop": self.estop,
            "observe": self.observe,
            "external": self.observe and time.monotonic() < self._ext_until,
            "overtemp": self.overtemp,
            "live": (ui_attached and not self.estop and not self.overtemp
                     and not self.contact_tripped and not self.observe
                     and self.targ is not None and self.link.connected),
            "q": st.dof_pos(),
            "dq": st.dof_vel(),
            "tau": st.dof_torque(),
            "temps": st.motor_temps(),
            "targ": None if self.targ is None else self.targ.tolist(),
            "ik": {a: (None if e is None else round(e, 4))
                   for a, e in self.ik_err.items()},
            "manip": {a: (None if v is None else round(v, 3))
                      for a, v in self.ik_manip.items()},
            "ik_ori": {a: self.ik_oerr.get(a) for a in ("left", "right")},
            "seq": {"n": len(self.waypoints), "names": list(self.wp_names),
                    "idx": self.seq_idx, "playing": self.seq_playing,
                    "loop": self.seq_loop, "active": self.seq_active},
            "guard": {"on": self.guard is not None,
                      "blocking": self.guard_blocking},
            "mirror": self.mirror,
            "carry": self.carry,
            "speed_scale": self.speed_scale,
            "tuning": self.tuning(),
            "obstacles": self.obstacles,
            "paused": self.paused,
            "collision": (self.guard.collision_view(st.dof_pos())
                          if (getattr(self, "show_collision", False)
                              and self.guard is not None
                              and hasattr(self.guard, "collision_view")) else None),
            "force": (self._tcp_force(st)
                      if getattr(self, "show_force", False) else None),
            "homing": self.home_active or self.plan_nodes is not None,
            "routing": (self.plan_nodes is not None
                        or (self.seq_route is not None and len(self.seq_route) > 1)),
            "contact": {"on": self.contact_reflex,
                        "tripped": self.contact_tripped,
                        "joint": self.contact_joint},
            "tools": {a: {"name": self.tool_names.get(a, "flange"),
                          "offset_mm": [round(v * 1000, 1)
                                        for v in self.kin[a].tool]}
                      for a in self.kin},
        }

    def close(self):
        self.link.close()
