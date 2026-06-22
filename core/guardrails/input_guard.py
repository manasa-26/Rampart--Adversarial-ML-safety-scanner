"""
Input guardrail — fast, synchronous checks run before any LLM call.
Layers: banned content → PII → prompt injection → toxicity keywords.
"""

import re
from typing import List, Tuple
from models.schemas import GuardrailCheck, Severity
from core.guardrails import prompt_injection as pi

# ---------------------------------------------------------------------------
# Banned content — absolute blocks regardless of context
# ---------------------------------------------------------------------------
BANNED_TERMS = {
    "child pornography", "csam", "child sexual abuse",
    "synthesis of nerve agent", "how to make sarin",
    "create bioweapon", "synthesize anthrax",
    "nuclear weapon design", "dirty bomb instructions",
}

# Toxicity keyword groups — flagged but not always blocked
TOXICITY_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(kill|murder|assassinate)\s+(yourself|myself|himself|herself|someone)\b", "high"),
    (r"\b(rape|sexual assault)\b", "high"),
    (r"\b(n[i!1]gg[ae]r|f[a@]gg[o0]t|ch[i!1]nk|sp[i!1]c)\b", "high"),
    (r"\b(suicide\s+method|how\s+to\s+kill\s+myself)\b", "high"),
    (r"\b(bomb\s+threat|school\s+shooting|mass\s+murder)\b", "critical"),
]

# PII patterns
PII_PATTERNS: List[Tuple[str, str, str]] = [
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "email", "medium"),
    (r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone", "medium"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN", "critical"),
    (r"\b4[0-9]{12}(?:[0-9]{3})?\b", "credit_card_visa", "critical"),
    (r"\b5[1-5][0-9]{14}\b", "credit_card_mc", "critical"),
    (r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", "password", "critical"),
]


def check_banned(text: str) -> GuardrailCheck:
    text_lower = text.lower()
    for term in BANNED_TERMS:
        if term in text_lower:
            return GuardrailCheck(
                check_name="banned_content",
                passed=False,
                severity=Severity.CRITICAL,
                reason=f"Absolutely prohibited content detected: '{term}'",
                confidence=1.0,
            )
    return GuardrailCheck(check_name="banned_content", passed=True, confidence=1.0)


def check_pii(text: str) -> GuardrailCheck:
    found = []
    for pattern, label, _ in PII_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(label)
    if found:
        return GuardrailCheck(
            check_name="pii_detection",
            passed=False,
            severity=Severity.HIGH,
            reason=f"PII detected: {', '.join(found)}",
            confidence=0.9,
        )
    return GuardrailCheck(check_name="pii_detection", passed=True, confidence=1.0)


def check_toxicity(text: str) -> GuardrailCheck:
    for pattern, severity in TOXICITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailCheck(
                check_name="toxicity",
                passed=False,
                severity=Severity(severity),
                reason="Toxic or harmful content pattern matched",
                confidence=0.85,
            )
    return GuardrailCheck(check_name="toxicity", passed=True, confidence=1.0)


def check_injection(text: str, strict: bool = False) -> GuardrailCheck:
    result = pi.detect(text, strict=strict)
    if result.is_injection:
        patterns_summary = "; ".join(f"{label} ({sev})" for label, sev in result.matched_patterns[:3])
        return GuardrailCheck(
            check_name="prompt_injection",
            passed=False,
            severity=Severity(result.severity),
            reason=f"Prompt injection detected — {patterns_summary}",
            confidence=result.confidence,
        )
    return GuardrailCheck(check_name="prompt_injection", passed=True, confidence=result.confidence)


def run_all(text: str, strict: bool = False, check_pii_flag: bool = True,
            check_injection_flag: bool = True, check_toxicity_flag: bool = True) -> List[GuardrailCheck]:
    checks: List[GuardrailCheck] = [check_banned(text)]
    if check_pii_flag:
        checks.append(check_pii(text))
    if check_injection_flag:
        checks.append(check_injection(text, strict))
    if check_toxicity_flag:
        checks.append(check_toxicity(text))
    return checks
