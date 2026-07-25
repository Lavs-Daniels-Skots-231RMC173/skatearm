"""Cross-site Origin guard for the cockpit WebSocket, kept dependency-free so it
can be unit-tested without FastAPI or a model (see test/test_sandbox.py)."""

from ipaddress import ip_address
from urllib.parse import urlparse


def host_only(value):
    """"host:port" / "[::1]:port" / "host" -> bare lower-case host."""
    v = (value or "").strip().lower()
    return v[1:].split("]")[0] if v.startswith("[") else v.split(":")[0]


def origin_allowed(origin, host_header):
    """True if a WebSocket handshake carrying this Origin may be accepted.

    A NAMED origin is accepted only for "localhost". Comparing Origin against
    the request's own Host header is no defense for a name: under DNS rebinding
    both headers read evil.com, so the comparison passes by construction. The
    Host echo is therefore trusted only when the origin host is a bare IP
    literal, which rebinding cannot produce — it needs a name to re-point at
    127.0.0.1. That keeps http://192.168.x.y:8000 working when the cockpit is
    deliberately bound to a LAN address.

    No Origin at all means a native client (the bridge, a test); browsers always
    send one on a WebSocket handshake, so this admits local tools without
    opening a cross-site path.
    """
    if not origin:
        return True
    host = (urlparse(origin).hostname or "").lower()
    try:
        ip = ip_address(host)
    except ValueError:
        return host == "localhost"          # the only name we ever trust
    if ip.is_loopback:                      # 127.0.0.0/8, ::1
        return True
    return host == host_only(host_header)
