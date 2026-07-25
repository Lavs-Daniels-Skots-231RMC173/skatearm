"""Program runner — Waldo-style Python programs over the safe bridge.

User programs run on a worker thread in a tiny sandboxed namespace: ``rbt``
(the robot API below), ``math`` and a whitelist of builtins. Every motion
call goes through the SAME RobotBridge paths as the UI — joint limits, the
collision guard, E-STOP, overtemp and the manual-input-interrupts rule all
apply. User code is AST-checked first (no imports, no dunder access) and runs
with restricted builtins, which blocks the usual ``exec``-sandbox escapes —
though this is still a local-tool guard, not a hostile-multi-tenant boundary.

Click-to-Step: started in step mode the runner pauses BEFORE every motion
command and reports the next command + source line; each STEP executes
exactly one command, RUN releases the program to free-run.

API (joints in degrees, cartesian in millimeters, world axes):
    rbt.movej(joint, deg)        joint = "L4" / "R2" / "H1", URDF name or index
    rbt.pose({joint: deg, ...})  several joints as ONE coordinated move
    rbt.movel(arm, dx=, dy=, dz=)  glide the TCP; auto-stops when blocked
    rbt.moveto(arm, x, y, z)     glide the TCP to an absolute world point (mm)
    rbt.home()                   glide to the documented safe pose
    rbt.gripper(arm, deg)        open/close one gripper
    rbt.waypoint(i_or_name)      glide to a recorded sequencer pose
    rbt.wait(seconds)            dwell
    rbt.tcp(arm) / rbt.q() / rbt.status()   readouts
    rbt.ok() / rbt.blocked() / rbt.contact() / rbt.near(arm,x,y,z)  conditions
"""

from __future__ import annotations

import ast
import builtins
import math
import sys
import threading
import time
import traceback

from skate_ros2 import names

PROGRAM_FILE = "<skate program>"
LOG_MAX = 300

_SAFE = {n: getattr(builtins, n)
         for n in ("abs", "min", "max", "round", "range", "len", "enumerate",
                   "zip", "float", "int", "str", "bool", "list", "dict",
                   "tuple", "sorted", "sum", "reversed", "divmod", "repr",
                   "Exception", "ValueError", "StopIteration", "isinstance")}


class ProgramStopped(Exception):
    pass


class _SandboxError(ValueError):
    pass


class _Sandbox(ast.NodeVisitor):
    """AST allow-list run before a program is compiled. It rejects the nodes that
    enable the classic ``exec``-sandbox escapes — imports, and any private or
    dunder attribute access (``__class__`` / ``__globals__`` / ``__subclasses__``
    / ``__builtins__`` / ``__import__`` …). Combined with the restricted builtins,
    this blocks ``().__class__.__base__.__subclasses__()`` style break-outs.
    (Still a local-tool guard, not a hostile-multi-tenant boundary.)
    """

    def visit_Import(self, node):
        raise _SandboxError("import is not allowed in robot programs")

    visit_ImportFrom = visit_Import

    # str.format / format_map can walk attributes ("{0.__class__}".format(x))
    # via the format mini-language with no Attribute AST node — block them too.
    _BANNED_ATTR = frozenset({"format", "format_map"})

    def visit_Attribute(self, node):
        # A SINGLE leading underscore is enough to reject: RobotAPI keeps its
        # runner on ``self._r``, so allowing ``_``-private attributes would hand
        # a program ``rbt._r.bridge`` — the whole bridge, E-STOP and limits
        # included. The public program API is underscore-free by construction.
        if isinstance(node.attr, str) and (node.attr.startswith("_")
                                           or node.attr in self._BANNED_ATTR):
            raise _SandboxError(f"access to attribute '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.id, str) and node.id.startswith("__"):
            raise _SandboxError(f"access to name '{node.id}' is not allowed")
        self.generic_visit(node)


def _resolve_joint(joint):
    """Accept "L1".."L8" / "R1".."R8" / "H1","H2", a URDF name, or an index."""
    if isinstance(joint, int):
        if 0 <= joint < names.N_JOINTS:
            return joint
        raise ValueError(f"joint index {joint} out of range")
    s = str(joint).strip()
    if s in names.INDEX:
        return names.INDEX[s]
    base = {"L": 8, "R": 16, "H": 24}.get(s[:1].upper())
    if base is not None and s[1:].isdigit():
        n = int(s[1:])
        limit = 2 if base == 24 else 8
        if 1 <= n <= limit:
            return base + n - 1
    raise ValueError(f"unknown joint {joint!r} — use L1..L8 / R1..R8 / "
                     "H1..H2, a URDF joint name, or a protocol index")


def _joint_token(idx):
    """Source-code token for a joint: 'L4' / 'R2' / 'H1' shorthands for the
    arms and head, the bare protocol index for the lower chain."""
    if 8 <= idx < 16:
        return f"'L{idx - 7}'"
    if 16 <= idx < 24:
        return f"'R{idx - 15}'"
    if 24 <= idx < 26:
        return f"'H{idx - 23}'"
    return str(idx)


class PoseRecorder:
    """Teach-in: watch the commanded pose and write the program yourself.

    Hooked into ``bridge.tick``. Whenever the target settles for
    ``SETTLE_S`` after moving, the diff against the previous keypose is
    emitted as one line of ``rbt`` code — ``movej`` for a single joint,
    ``pose({...})`` for a coordinated multi-joint move. Drive the robot with
    sliders, jog, the gizmo or cartesian steps; stop recording and the
    program is ready to RUN (through the same safe bridge, as always).
    """

    SETTLE_S = 0.6
    MIN_DELTA = math.radians(1.0)

    def __init__(self):
        self.active = False
        self._base = None                 # pose at the last emitted keypose
        self._last = None                 # latest seen target
        self._dirty = False
        self._still = 0.0
        self.lines = []
        self.result = ""                  # finished program text

    def start(self, targ):
        self.active = True
        self._base = None if targ is None else list(targ)
        self._last = None if targ is None else list(targ)
        self._dirty = False
        self._still = 0.0
        self.lines = []
        self.result = ""

    def observe(self, targ, dt):
        if not self.active or targ is None:
            return
        t = list(targ)
        if self._base is None:
            self._base = t
            self._last = t
            return
        if max(abs(a - b) for a, b in zip(t, self._last)) > 1e-4:
            self._dirty = True
            self._still = 0.0
            self._last = t
        elif self._dirty:
            self._still += dt
            if self._still >= self.SETTLE_S:
                self._emit()

    def _emit(self):
        moved = [(i, self._last[i]) for i in range(len(self._last))
                 if abs(self._last[i] - self._base[i]) > self.MIN_DELTA]
        self._dirty = False
        self._still = 0.0
        if not moved:
            return
        if len(moved) == 1:
            i, v = moved[0]
            self.lines.append(
                f"rbt.movej({_joint_token(i)}, {math.degrees(v):.1f})")
        else:
            body = ", ".join(f"{_joint_token(i)}: {math.degrees(v):.1f}"
                             for i, v in moved)
            self.lines.append(f"rbt.pose({{{body}}})")
        self._base = list(self._last)

    def stop(self):
        if self._dirty:                   # flush a pose still settling
            self._emit()
        self.active = False
        self.result = ("# recorded with REC — replays through the same "
                       "safe bridge\n" + "\n".join(self.lines) + "\n"
                       if self.lines else "")
        return self.result

    def snapshot(self):
        return {"on": self.active, "n": len(self.lines)}


class RobotAPI:
    """The ``rbt`` object visible to user programs."""

    def __init__(self, runner):
        self._r = runner

    # -- readouts (no gate: free to call anywhere) ---------------------------
    def q(self):
        """All 26 commanded joint angles, degrees."""
        t = self._r.bridge.targ
        return ([round(float(math.degrees(v)), 2) for v in t]
                if t is not None else None)

    def tcp(self, arm="right"):
        """Current commanded TCP position, world millimeters."""
        br = self._r.bridge
        if arm not in br.kin or br.targ is None:
            return None
        p = br.kin[arm].fk(br.targ)
        return tuple(round(float(v) * 1000, 1) for v in p)

    def status(self):
        return self._r.bridge.snapshot()

    # -- conditions (no gate: branch loops/ifs on robot state) ---------------
    def ok(self):
        """True while the robot can still move — not E-stopped, not overtemp,
        contact reflex not tripped, and the program isn't being stopped.
        Handy as a ``while rbt.ok():`` loop guard."""
        br = self._r.bridge
        return not (self._r._stop or br.estop or getattr(br, "overtemp", False)
                    or br.contact_tripped)

    def blocked(self):
        """True if the collision guard is currently blocking motion."""
        return bool(self._r.bridge.guard_blocking)

    def contact(self):
        """True if the contact reflex has tripped (the arm pushed into
        something)."""
        return bool(self._r.bridge.contact_tripped)

    def near(self, arm, x, y, z, tol=20.0):
        """True if the arm's TCP is within ``tol`` mm of the world point (mm)."""
        p = self.tcp(arm)
        if p is None:
            return False
        return ((p[0] - x) ** 2 + (p[1] - y) ** 2
                + (p[2] - z) ** 2) ** 0.5 <= tol

    # -- motion (gated: honors Click-to-Step / STOP / E-STOP) ----------------
    def movej(self, joint, deg, tol=3.0, timeout=10.0):
        idx = _resolve_joint(joint)
        self._r._gate(f"movej({joint!r}, {deg:g})")
        br = self._r.bridge
        br.set_joint(idx, math.radians(deg))
        goal = None if br.targ is None else float(br.targ[idx])
        if goal is None or (abs(goal - math.radians(deg)) > 1e-6
                            and br.guard_blocking):
            self._r.emit(f"! movej {joint}: blocked by the collision guard")
            return False
        ok = self._r._wait_until(
            lambda: (q := br.link.state.dof_pos()) is not None
                    and abs(q[idx] - goal) < math.radians(tol), timeout)
        if not ok:
            self._r.emit(f"! movej {joint}: timeout (still "
                         f"tracking toward {deg:g} deg)")
        return ok

    def pose(self, joints, tol=3.0, timeout=12.0):
        """Command several joints AT ONCE (dict joint -> deg) and wait for
        all of them — one coordinated move, one guard check, no mirroring
        (the pose already says where both arms go)."""
        items = [(_resolve_joint(j), math.radians(d))
                 for j, d in dict(joints).items()]
        self._r._gate(f"pose({len(items)} joints)")
        br = self._r.bridge
        if not br.set_joints(items):
            self._r.emit("! pose: blocked by the collision guard")
            return False
        goals = {idx: (None if br.targ is None else float(br.targ[idx]))
                 for idx, _ in items}
        ok = self._r._wait_until(
            lambda: (q := br.link.state.dof_pos()) is not None
                    and all(abs(q[i] - g) < math.radians(tol)
                            for i, g in goals.items() if g is not None),
            timeout)
        if not ok:
            self._r.emit("! pose: timeout while tracking")
        return ok

    def movel(self, arm, dx=0.0, dy=0.0, dz=0.0, timeout=15.0):
        self._r._gate(f"movel({arm!r}, {dx:g}, {dy:g}, {dz:g})")
        br = self._r.bridge
        br.cart_step(arm, [dx / 1000.0, dy / 1000.0, dz / 1000.0])
        if br.ik_targets.get(arm) is None:
            self._r.emit(f"! movel {arm}: rejected (not armed / estopped?)")
            return False
        self._r._wait_until(lambda: br.ik_targets.get(arm) is None, timeout)
        return self.tcp(arm)

    def moveto(self, arm, x, y, z, timeout=15.0):
        """Glide the TCP to an ABSOLUTE world point (mm). Auto-stops on arrival
        or when blocked / out of reach (same IK path as the drag-gizmo)."""
        if arm not in self._r.bridge.kin:
            raise ValueError("arm must be 'left' or 'right'")
        self._r._gate(f"moveto({arm!r}, {x:g}, {y:g}, {z:g})")
        br = self._r.bridge
        br.set_ik_target(arm, [x / 1000.0, y / 1000.0, z / 1000.0], auto=True)
        if br.ik_targets.get(arm) is None:
            self._r.emit(f"! moveto {arm}: rejected (not armed / estopped?)")
            return False
        self._r._wait_until(lambda: br.ik_targets.get(arm) is None, timeout)
        return self.tcp(arm)

    def home(self, timeout=12.0):
        self._r._gate("home()")
        br = self._r.bridge
        br.home()
        import numpy as np
        goal = np.array(names.DEFAULT_POSE)
        ok = self._r._wait_until(
            lambda: (q := br.link.state.dof_pos()) is not None
                    and float(np.max(np.abs(np.asarray(q) - goal))) < 0.06,
            timeout)
        if not ok:
            self._r.emit("! home(): timeout while tracking")
        return ok

    def gripper(self, arm, deg, timeout=5.0):
        idx = {"left": names.LEFT_GRIPPER, "right": names.RIGHT_GRIPPER}.get(arm)
        if idx is None:
            raise ValueError("gripper arm must be 'left' or 'right'")
        return self.movej(idx, deg, tol=4.0, timeout=timeout)

    def waypoint(self, which, timeout=30.0):
        """Glide to a recorded sequencer pose (1-based index or name)."""
        br = self._r.bridge
        if isinstance(which, str):
            if which not in br.wp_names:
                raise ValueError(f"no waypoint named {which!r}")
            i = br.wp_names.index(which)
        else:
            i = int(which) - 1
        self._r._gate(f"waypoint({which!r})")
        br.wp_goto(i)
        if not br.seq_active:
            self._r.emit(f"! waypoint {which}: rejected (empty list / "
                         "estopped?)")
            return False
        return self._r._wait_until(lambda: not br.seq_active, timeout)

    def wait(self, seconds):
        self._r._gate(f"wait({seconds:g})")
        end = time.monotonic() + float(seconds)
        while time.monotonic() < end:
            self._r._check()
            time.sleep(0.02)
        return True


class ProgramRunner:
    def __init__(self, bridge):
        self.bridge = bridge
        self.lock = threading.Lock()
        self.code = ""
        self.running = False
        self.paused = False
        self.step_mode = False
        self.line = None
        self.counter = 0
        self.current = None
        self.log = []
        self._thread = None
        self._stop = False
        self._stop_reason = ""
        self._step_evt = threading.Event()

    # -- control ---------------------------------------------------------------
    def run(self, code=None, step=False):
        with self.lock:
            if self.running:
                # RUN releases a stepping program; STEP keeps it stepping
                self.step_mode = step
                self._step_evt.set()
                return True
            if code is not None:
                self.code = str(code)
            try:
                tree = ast.parse(self.code, PROGRAM_FILE, "exec")
                _Sandbox().visit(tree)
                obj = compile(tree, PROGRAM_FILE, "exec")
            except SyntaxError as e:
                self.log = [f"x syntax error, line {e.lineno}: {e.msg}"]
                return False
            except _SandboxError as e:
                self.log = [f"x rejected: {e}"]
                return False
            self.running = True
            self.paused = False
            self.step_mode = bool(step)
            self.counter = 0
            self.line = None
            self.current = None
            self.log = [("> program started (Click-to-Step)" if step
                         else "> program started")]
            self._stop = False
            self._stop_reason = ""
            self._step_evt.clear()
            self._thread = threading.Thread(target=self._main, args=(obj,),
                                            daemon=True)
            self._thread.start()
            return True

    def step(self, code=None):
        with self.lock:
            if self.running:
                self.step_mode = True
                self._step_evt.set()
                return True
        return self.run(code, step=True)

    def stop(self, reason=""):
        if self.running:
            self._stop_reason = reason
            self._stop = True
            self._step_evt.set()

    # -- worker ------------------------------------------------------------------
    def _main(self, obj):
        api = RobotAPI(self)
        env = {"rbt": api, "math": math,
               "__builtins__": dict(_SAFE, print=self._print),
               "__name__": "__main__"}
        try:
            exec(obj, env)
            self.emit("* program finished")
        except ProgramStopped:
            r = self._stop_reason
            self.emit(f"# stopped{' — ' + r if r else ''}")
        except Exception as e:
            ln = None
            for fr in traceback.extract_tb(sys.exc_info()[2]):
                if fr.filename == PROGRAM_FILE:
                    ln = fr.lineno
            where = f", line {ln}" if ln else ""
            self.emit(f"x {type(e).__name__}{where}: {e}")
        finally:
            self.bridge.seq_stop()
            self.bridge.clear_ik_target()
            with self.lock:
                self.running = False
                self.paused = False
                self.current = None

    def _print(self, *a, **k):
        self.emit(" ".join(str(x) for x in a))

    def emit(self, line):
        with self.lock:
            self.log.append(str(line))
            del self.log[:-LOG_MAX]

    # -- gating ------------------------------------------------------------------
    def _check(self):
        if self._stop:
            raise ProgramStopped(self._stop_reason)
        if self.bridge.estop:
            self._stop_reason = self._stop_reason or "E-STOP"
            raise ProgramStopped("E-STOP")

    def _user_line(self):
        f = sys._getframe(2)
        while f is not None:
            if f.f_code.co_filename == PROGRAM_FILE:
                return f.f_lineno
            f = f.f_back
        return None

    def _gate(self, desc):
        self._check()
        self.counter += 1
        self.current = desc
        self.line = self._user_line()
        if self.step_mode:
            self.paused = True
            try:
                while not self._step_evt.wait(0.1):
                    self._check()
            finally:
                self._step_evt.clear()
                self.paused = False
            self._check()

    def _wait_until(self, pred, timeout):
        end = time.monotonic() + float(timeout)
        while time.monotonic() < end:
            self._check()
            try:
                if pred():
                    return True
            except Exception:
                pass
            time.sleep(0.02)
        return False

    # -- state for the UI ----------------------------------------------------------
    def snapshot(self):
        with self.lock:
            return {"running": self.running, "paused": self.paused,
                    "step": self.step_mode, "line": self.line,
                    "n": self.counter, "current": self.current,
                    "log": list(self.log[-40:])}
