from __future__ import annotations

import json
import re
from typing import Any


UNSAFE_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----", re.I),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*['\"]?[^'\"\s]{8,}"),
    re.compile(r"(?i)\b(?:rm\s+-rf\s+/|format\s+c:|powershell\s+-enc)\b"),
]


def redact_text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    for pattern in UNSAFE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return defang_url(text)


def defang_url(text: str) -> str:
    text = re.sub(r"(?i)\bhttps?://", "hxxp://", text)
    text = text.replace(".", "[.]")
    return text


def safe_feature_json(features: dict[str, Any]) -> str:
    return redact_text(features)


def is_safe_representation(value: Any) -> bool:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return all(pattern.search(text) is None for pattern in UNSAFE_PATTERNS)


def ensure_safe(value: Any) -> str:
    redacted = redact_text(value)
    if not is_safe_representation(redacted):
        raise ValueError("Value could not be converted into a safe representation.")
    return redacted

