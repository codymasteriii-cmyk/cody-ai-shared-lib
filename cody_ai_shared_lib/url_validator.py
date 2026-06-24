"""URL safety validator — guards against SSRF via user/feed-controlled URLs.

Shared across all Cody AI projects. Used by:
  - fetcher.py: validates article URLs before passing to Jina Reader.
  - Project-level code (e.g. ingestion.py): validates feed URLs before
    making direct server-side requests.get() calls.

Only IP *literals* in the URL string are blocked; DNS hostnames are not
pre-resolved (DNS rebinding is outside the threat model for admin-configured
feeds). This covers the most common SSRF vectors: direct IP, loopback, and
cloud IMDS endpoints expressed as IP addresses.
"""
import ipaddress
from urllib.parse import urlparse

_ALLOWED_SCHEMES = {"http", "https"}

# Private, loopback, link-local, and IMDS ranges that must never be
# reachable via a user/feed-controlled URL.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),        # RFC 1918 private
    ipaddress.ip_network("172.16.0.0/12"),      # RFC 1918 private
    ipaddress.ip_network("192.168.0.0/16"),     # RFC 1918 private
    ipaddress.ip_network("127.0.0.0/8"),        # IPv4 loopback
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / AWS+GCP IMDS
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),           # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
]


def validate_public_url(url: str) -> None:
    """Raise ValueError if url is unsafe to request.

    Blocks:
    - Non-http/https schemes (file://, ftp://, gopher://, etc.)
    - Empty or localhost hostnames
    - IP literals in private/loopback/IMDS ranges

    Safe for:
    - Public DNS hostnames (e.g. hn.algolia.com, r.jina.ai)
    - Any http:// or https:// URL with a routable IP literal

    Raises:
        ValueError: with a descriptive message if the URL fails any check.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Disallowed URL scheme '{parsed.scheme}': {url!r}. "
            f"Only {_ALLOWED_SCHEMES} are permitted."
        )

    host = parsed.hostname or ""
    if not host or host == "localhost":
        raise ValueError(
            f"Disallowed host '{host}' in URL: {url!r}. "
            "Empty hosts and 'localhost' are not permitted."
        )

    # Try to parse as an IP literal. DNS names raise ValueError here, which
    # means they are not IP literals — they pass through safely.
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal — DNS hostname, allow it.
        return

    # It is an IP literal — check against blocked networks.
    for net in _BLOCKED_NETWORKS:
        if addr in net:
            raise ValueError(
                f"URL targets private/internal address {addr} "
                f"(blocked network {net}): {url!r}"
            )
