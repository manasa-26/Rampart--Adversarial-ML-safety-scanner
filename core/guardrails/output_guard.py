"""
Output guardrail — checks LLM-generated text before returning to the user.
Catches hallucinated PII, dangerous instructions in output, and leaked secrets.
"""

import re
from typing import List
from models.schemas import GuardrailCheck, Severity

LEAKED_SECRET_PATTERNS = [
    (r"(sk|pk|api|secret|token|key)\s*[-_]?\s*[a-zA-Z0-9]{20,}", "critical", "Potential API key in output"),
    (r"(?:password|passwd|pwd)\s*[:=]\s*\S{6,}", "critical", "Password in output"),
    (r"-----BEGIN\s+(RSA|PRIVATE|EC)\s+PRIVATE\s+KEY-----", "critical", "Private key in output"),
    (r"ghp_[a-zA-Z0-9]{36}", "critical", "GitHub Personal Access Token"),
    (r"xox[baprs]-[0-9]{12}-[0-9]{12}-[a-zA-Z0-9]{24}", "critical", "Slack token"),
]

HARMFUL_INSTRUCTION_PATTERNS = [
    (r"step\s+\d+.{0,40}(detonate|ignite|synthesize|assemble\s+bomb)", "critical", "Weapon assembly instructions"),
    (r"(mix|combine|add)\s+.{0,30}(bleach|ammonia|cyanide|sarin|VX\s+nerve)", "critical", "Dangerous chemical instructions"),
    (r"(inject|ingest|consume)\s+.{0,30}(lethal|overdose|toxin)", "high", "Harm instructions in output"),
]

PII_OUTPUT_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN", "critical"),
    (r"\b4[0-9]{12}(?:[0-9]{3})?\b", "credit_card", "critical"),
]


def check_output_secrets(text: str) -> GuardrailCheck:
    for pattern, severity, label in LEAKED_SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailCheck(
                check_name="output_secrets",
                passed=False,
                severity=Severity(severity),
                reason=f"Sensitive data leak in output: {label}",
                confidence=0.95,
            )
    return GuardrailCheck(check_name="output_secrets", passed=True, confidence=1.0)


def check_harmful_instructions(text: str) -> GuardrailCheck:
    for pattern, severity, label in HARMFUL_INSTRUCTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailCheck(
                check_name="harmful_instructions",
                passed=False,
                severity=Severity(severity),
                reason=f"Harmful instructions in LLM output: {label}",
                confidence=0.9,
            )
    return GuardrailCheck(check_name="harmful_instructions", passed=True, confidence=1.0)


def check_output_pii(text: str) -> GuardrailCheck:
    for pattern, label, severity in PII_OUTPUT_PATTERNS:
        if re.search(pattern, text):
            return GuardrailCheck(
                check_name="output_pii",
                passed=False,
                severity=Severity(severity),
                reason=f"PII ({label}) present in LLM output",
                confidence=0.9,
            )
    return GuardrailCheck(check_name="output_pii", passed=True, confidence=1.0)


def run_all(text: str) -> List[GuardrailCheck]:
    return [
        check_output_secrets(text),
        check_harmful_instructions(text),
        check_output_pii(text),
    ]
