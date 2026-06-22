"""
Guardrail pipeline — chains input and output guards and computes an overall risk score.
"""

from typing import List
from models.schemas import GuardrailRequest, GuardrailResult, GuardrailCheck, Severity
from core.guardrails import input_guard, output_guard

SEVERITY_SCORE = {
    Severity.CRITICAL: 40,
    Severity.HIGH: 20,
    Severity.MEDIUM: 8,
    Severity.LOW: 3,
    Severity.INFO: 0,
}

BLOCK_SEVERITIES = {Severity.CRITICAL, Severity.HIGH}


def _compute_risk(checks: List[GuardrailCheck]) -> float:
    total = sum(SEVERITY_SCORE.get(c.severity, 0) for c in checks if not c.passed)
    return min(100.0, float(total))


def _sanitize(text: str, checks: List[GuardrailCheck]) -> str:
    """Redact obvious PII from text when checks flag it."""
    import re
    sanitized = text
    if any(c.check_name == "pii_detection" and not c.passed for c in checks):
        sanitized = re.sub(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            "[EMAIL REDACTED]", sanitized
        )
        sanitized = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", sanitized)
        sanitized = re.sub(
            r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "[PHONE REDACTED]", sanitized
        )
    return sanitized


class GuardrailPipeline:
    async def check(self, request: GuardrailRequest) -> GuardrailResult:
        all_checks: List[GuardrailCheck] = []

        # --- Input checks ---
        input_checks = input_guard.run_all(
            text=request.text,
            strict=request.strict_mode,
            check_pii_flag=request.check_pii,
            check_injection_flag=request.check_injection,
            check_toxicity_flag=request.check_toxicity,
        )
        all_checks.extend(input_checks)

        # --- Output checks (when caller is validating LLM output) ---
        # Caller signals this by setting check_pii=False, check_injection=False
        # and passing output text — or can be added as a separate route.
        # Here we always run output checks as a second layer.
        output_checks = output_guard.run_all(request.text)
        all_checks.extend(output_checks)

        failed = [c for c in all_checks if not c.passed]
        passed = len(failed) == 0
        blocked_reason: str | None = None
        sanitized: str | None = None

        if failed:
            worst = max(failed, key=lambda c: SEVERITY_SCORE.get(c.severity, 0))
            if worst.severity in BLOCK_SEVERITIES:
                blocked_reason = worst.reason
            else:
                # Medium/low — sanitize but allow through
                sanitized = _sanitize(request.text, all_checks)
                passed = True   # not blocked, just sanitized

        risk_score = _compute_risk(all_checks)

        return GuardrailResult(
            text=request.text,
            passed=passed,
            blocked_reason=blocked_reason,
            checks=all_checks,
            risk_score=risk_score,
            sanitized_text=sanitized,
        )
