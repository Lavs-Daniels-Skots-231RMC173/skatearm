"""The rbt program runner AST-validates user code before exec: imports and
private/dunder name/attribute access (the exec-sandbox escape vectors) are
rejected, while ordinary robot programs pass. Plus the cockpit's cross-site
Origin rule. Hardware-free (no bridge / MuJoCo / FastAPI).

    python -m pytest -q tools/skate_commander/test/test_sandbox.py
"""
import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skate_ros2"))

from skate_commander.origin import origin_allowed              # noqa: E402
from skate_commander.program import _Sandbox, _SandboxError   # noqa: E402


@pytest.mark.parametrize("src", [
    "().__class__.__base__.__subclasses__()",
    "[].__class__.__bases__[0].__subclasses__()",
    "import os",
    "from os import system",
    "x = __import__('os')",
    "b = __builtins__",
    "g = (lambda: 0).__globals__",
    "object.__subclasses__(object)",
    '"{0.__class__}".format(())',
    '"{0.__class__}".format_map({})',
    # A SINGLE leading underscore is an escape too: RobotAPI keeps the runner on
    # self._r, so rbt._r.bridge would hand a program the E-STOP and the joint
    # limits. Blocking only dunders left this wide open.
    "b = rbt._r.bridge",
    "rbt._r.bridge.estop = False",
    "rbt._r.bridge.guard = None",
    "rbt._r.bridge.hi = [999] * 26",
    "g = rbt._r",
])
def test_sandbox_rejects_escapes(src):
    with pytest.raises(_SandboxError):
        _Sandbox().visit(ast.parse(src, "<t>", "exec"))


def test_sandbox_allows_normal_program():
    ok = (
        "for i in range(3):\n"
        "    rbt.movej('L4', 10 + i)\n"
        "    if rbt.ok() and not rbt.blocked():\n"
        "        rbt.wait(0.3)\n"
        "total = sum([1, 2, 3])\n"
        "rbt.pose({'L2': 20, 'R2': -20})\n"
        "x = math.sin(0.5)\n"
    )
    _Sandbox().visit(ast.parse(ok, "<t>", "exec"))   # must not raise


# ---------------------------------------------------------------- ws origin --

@pytest.mark.parametrize("origin,host", [
    ("http://localhost:8088", "localhost:8088"),
    ("http://127.0.0.1:8088", "127.0.0.1:8088"),
    ("http://127.0.0.1", "127.0.0.1"),
    ("http://[::1]:8088", "[::1]:8088"),
    ("http://192.168.1.5:8088", "192.168.1.5:8088"),   # deliberate LAN bind
    (None, "127.0.0.1:8088"),                          # native client, no Origin
    ("", "127.0.0.1:8088"),
])
def test_origin_allowed(origin, host):
    assert origin_allowed(origin, host)


@pytest.mark.parametrize("origin,host", [
    # The DNS-rebinding case the guard exists for: the attacker's name resolves
    # to 127.0.0.1, so Origin and Host agree. A same-host comparison alone would
    # pass this by construction — the whole point of the fix.
    ("http://evil.com", "evil.com"),
    ("http://evil.com:8088", "evil.com:8088"),
    ("https://attacker.example", "attacker.example"),
    ("http://localhost.evil.com", "localhost.evil.com"),
    ("http://evil.com", "127.0.0.1:8088"),             # plain cross-site
    ("http://192.168.1.9:8088", "192.168.1.5:8088"),   # IP, but not ours
])
def test_origin_refused(origin, host):
    assert not origin_allowed(origin, host)
