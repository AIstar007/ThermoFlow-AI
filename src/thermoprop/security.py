"""Small helpers for protecting sensitive connection metadata in exports."""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

_SECRET_MARKERS = ("password", "token", "api_key", "secret", "authorization", "client_key")


def redact_sensitive_metadata(value: Any) -> Any:
    """Recursively redact secret fields and credential-like URL components.

    The original settings are still passed to the adapter at runtime. This
    function is for manifests, summaries and export bundles only.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_l = str(key).casefold()
            if any(marker in key_l for marker in _SECRET_MARKERS):
                out[str(key)] = "[REDACTED]" if item not in (None, "") else ""
            elif key_l.endswith(("_url", "_uri")) and isinstance(item, str):
                try:
                    parsed = urlsplit(item)
                    safe_query = []
                    for query_key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
                        if any(marker in query_key.casefold() for marker in _SECRET_MARKERS):
                            query_value = "[REDACTED]"
                        safe_query.append((query_key, query_value))
                    safe_netloc = parsed.netloc
                    if parsed.username is not None or parsed.password is not None:
                        host = parsed.hostname or ""
                        if parsed.port:
                            host += f":{parsed.port}"
                        safe_netloc = f"[REDACTED]@{host}"
                    out[str(key)] = urlunsplit((parsed.scheme, safe_netloc, parsed.path,
                                                urlencode(safe_query), parsed.fragment))
                except Exception:
                    out[str(key)] = "[REDACTED_URL]"
            else:
                out[str(key)] = redact_sensitive_metadata(item)
        return out
    if isinstance(value, list):
        return [redact_sensitive_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_metadata(item) for item in value]
    return value
