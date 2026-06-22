"""
Secrets and credentials scanner.
Two-layer detection:
  1. Pattern matching — known credential formats (GitHub PAT, OpenAI key, AWS AKIA, etc.)
  2. Entropy scoring — catches unknown/custom secrets via Shannon entropy > 4.5
"""

import re
import math
from typing import List
from models.schemas import Finding, Severity

SECRET_PATTERNS = [
    (r"(?i)(api[_\-]?key|apikey)\s*[:=]\s*['\"]?([A-Za-z0-9\-_]{20,})['\"]?", "API Key"),
    (r"(?i)(secret[_\-]?key|secret)\s*[:=]\s*['\"]?([A-Za-z0-9\-_]{20,})['\"]?", "Secret Key"),
    (r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?(\S{6,})['\"]?", "Password"),
    (r"(?i)(access[_\-]?token|auth[_\-]?token|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9\-_.]{20,})['\"]?", "Auth Token"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub PAT"),
    (r"ghs_[A-Za-z0-9]{36}", "GitHub App Token"),
    (r"sk-[A-Za-z0-9]{48}", "OpenAI API Key"),
    (r"xox[baprs]-[0-9]{12}-[0-9]{12}-[a-zA-Z0-9]{24}", "Slack Token"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"-----BEGIN\s+(RSA|EC|DSA|OPENSSH)\s+PRIVATE\s+KEY-----", "Private Key"),
    (r"(?i)database[_\-]?url\s*[:=]\s*['\"]?(postgres|mysql|mongodb|redis)://[^\s'\"]+", "DB Connection String"),
    (r"(?i)(smtp[_\-]?password|email[_\-]?password)\s*[:=]\s*['\"]?(\S{6,})['\"]?", "SMTP Password"),
]

# Tokens that look like secrets (assigned to a key-like variable) but skip common test values
_ASSIGNMENT_RE = re.compile(
    r"""(?i)(?:key|token|secret|password|passwd|pwd|auth|credential|api)\s*[:=]\s*["']?([A-Za-z0-9+/=_\-]{20,})["']?"""
)
_SKIP_VALUES = {
    "your_api_key_here", "placeholder", "changeme", "example", "dummy",
    "test", "xxxxxxxxxxxxxxxxxxxx", "xxxxxxxx", "none", "null", "false", "true",
}

ENTROPY_THRESHOLD = 4.5   # bits — Gitleaks uses 4.5 for base64, 3.5 for hex
MIN_TOKEN_LENGTH = 20


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _find_high_entropy_secrets(content: str, filename: str) -> List[Finding]:
    """
    Scan for high-entropy strings assigned to credential-like variable names.
    This catches custom/random secrets that don't match known patterns.
    """
    findings: List[Finding] = []
    seen: set[str] = set()

    for match in _ASSIGNMENT_RE.finditer(content):
        token = match.group(1)
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if token.lower() in _SKIP_VALUES:
            continue
        if token in seen:
            continue
        entropy = shannon_entropy(token)
        if entropy >= ENTROPY_THRESHOLD:
            seen.add(token)
            line = content[: match.start()].count("\n") + 1
            findings.append(Finding(
                scanner="secrets",
                severity=Severity.HIGH,
                title=f"High-entropy credential string (entropy={entropy:.2f} bits)",
                description=(
                    f"A {len(token)}-character string assigned to a credential-like variable "
                    f"in '{filename}' has Shannon entropy {entropy:.2f} bits (threshold: {ENTROPY_THRESHOLD}). "
                    "High-entropy strings assigned to credential variables are almost always real secrets."
                ),
                location=f"{filename}:{line}",
                evidence=f"{token[:12]}{'*' * min(8, len(token)-12)}… (redacted)",
                recommendation=(
                    "Remove from source code immediately. Rotate the credential. "
                    "Store in an environment variable or secrets manager."
                ),
            ))
    return findings


def scan(content: str, filename: str) -> List[Finding]:
    findings: List[Finding] = []

    # Layer 1 — known-format pattern matching
    for pattern, label in SECRET_PATTERNS:
        for match in re.finditer(pattern, content):
            snippet = match.group(0)[:80].replace("\n", " ")
            findings.append(Finding(
                scanner="secrets",
                severity=Severity.CRITICAL,
                title=f"Hardcoded {label} detected",
                description=(
                    f"A {label} appears to be hardcoded in '{filename}'. "
                    "This credential may be leaked if the file is committed to version control."
                ),
                location=filename,
                evidence=snippet,
                recommendation=(
                    "Remove the credential immediately. Rotate it. "
                    "Use environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault)."
                ),
            ))

    # Layer 2 — entropy-based detection for unknown/custom secrets
    findings.extend(_find_high_entropy_secrets(content, filename))

    return findings
