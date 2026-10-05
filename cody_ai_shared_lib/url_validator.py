"""URL safety validator — guards against SSRF via user/feed-controlled URLs.

Shared across all Cody AI projects. Used by:
  - fetcher.py: validates article URLs before fetching them (Jina Reader or the
    direct PDF download).
  - Project-level code (e.g. ingestion.py): validates feed URLs before making
    direct server-side requests.get() calls.

Approach: a URL is safe only if EVERY address its hostname resolves to is publicly
routable (ipaddress.is_global). The check runs on what the OS resolver returns, not
on how the URL is spelled, so it covers:
  - IP literals (127.0.0.1, [::1]) and IPv4-mapped IPv6 ([::ffff:127.0.0.1]),
  - alternate IPv4 spellings (2130706433, 0x7f.0.0.1) — the resolver normalizes
    them and the result is checked; if the resolver rejects them the URL is blocked,
  - hostnames that point at private space (127.0.0.1.nip.io, localtest.me),
  - ranges a deny-list tends to forget (100.64.0.0/10 CGNAT, multicast, etc.).

Redirects are the other classic bypass: a public URL can answer 302 -> an internal
address. safe_get() follows redirects manually and validates every hop.

Known limit: DNS is resolved here and again by requests when it connects, so a
hostname that changes its answer between the two (DNS rebinding) is not covered.
"""
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

_ALLOWED_SCHEMES = {"http", "https"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _is_public_address(raw: str) -> bool:
    """True if the resolved address is publicly routable."""
    addr = ipaddress.ip_address(raw.split("%", 1)[0])  # drop any IPv6 zone id
    if addr.version == 6 and addr.ipv4_mapped:
        addr = addr.ipv4_mapped  # judge ::ffff:a.b.c.d by its embedded IPv4 address
    return addr.is_global


def validate_public_url(url: str) -> None:
    """Raise ValueError if url is unsafe to request.

    Blocks:
    - Non-http/https schemes (file://, ftp://, gopher://, etc.)
    - Empty hosts and 'localhost'
    - Hosts that do not resolve (nothing can be verified, so nothing is allowed)
    - Hosts where ANY resolved address is not publicly routable

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

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, ValueError) as exc:
        raise ValueError(f"Host '{host}' could not be resolved ({exc}): {url!r}") from exc

    for info in infos:
        resolved = info[4][0]
        if not _is_public_address(resolved):
            raise ValueError(
                f"URL host '{host}' resolves to non-public address {resolved}: {url!r}"
            )


def safe_get(url: str, *, max_redirects: int = 5, **kwargs):
    """requests.get() with SSRF validation on the URL and on every redirect hop.

    requests follows redirects by default and only the first URL would otherwise be
    validated. Here redirects are followed manually so each Location target passes
    validate_public_url() before it is requested.

    Extra kwargs (headers, timeout, stream, ...) are passed to requests.get(). Do not
    pass credentials in headers: they would be re-sent to every redirect target.

    Raises:
        ValueError: if the URL or any redirect target fails validation, or the
                    redirect chain is longer than max_redirects.
        requests.RequestException: on network errors, as with requests.get().
    """
    import requests  # local import: callers that only validate need not load requests

    kwargs.pop("allow_redirects", None)
    current = url
    for _ in range(max_redirects + 1):
        validate_public_url(current)
        response = requests.get(current, allow_redirects=False, **kwargs)
        location = response.headers.get("location")
        if response.status_code in _REDIRECT_STATUSES and location:
            response.close()
            current = urljoin(current, location)
            continue
        return response
    raise ValueError(f"Too many redirects (>{max_redirects}) starting from {url!r}")
