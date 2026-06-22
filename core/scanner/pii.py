"""
PII (Personally Identifiable Information) scanner.
Two-layer detection:
  1. Presidio NLP engine — contextual entity recognition (name, org, address, etc.)
  2. Regex fallback — structural patterns for SSN, credit cards, passport numbers, etc.
Presidio is used when available; regex always runs to catch structural PII Presidio misses.
"""

import re
from typing import List
from models.schemas import Finding, Severity

# ---------------------------------------------------------------------------
# Regex layer (structural PII — always runs)
# ---------------------------------------------------------------------------

REGEX_PATTERNS = [
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "Email Address", Severity.MEDIUM),
    (r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "Phone Number", Severity.MEDIUM),
    (r"\b\d{3}-\d{2}-\d{4}\b", "Social Security Number (SSN)", Severity.CRITICAL),
    (r"\b4[0-9]{12}(?:[0-9]{3})?\b", "Visa Card Number", Severity.CRITICAL),
    (r"\b5[1-5][0-9]{14}\b", "Mastercard Number", Severity.CRITICAL),
    (r"\b3[47][0-9]{13}\b", "Amex Card Number", Severity.CRITICAL),
    (r"\b(?:19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", "Date of Birth (ISO)", Severity.LOW),
    (r"(?i)\bpassport\s*(?:number|no|#)?\s*[:=]?\s*[A-Z]{1,2}\d{6,9}\b", "Passport Number", Severity.HIGH),
    (r"(?i)\bdriv(?:er.?s?\s+licen[sc]e|ing\s+licen[sc]e)\s*(?:number|no|#)?\s*[:=]?\s*[A-Z0-9]{5,15}\b", "Driver License", Severity.HIGH),
    (r"(?i)\bIBAN\s*[:=]?\s*[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b", "IBAN", Severity.HIGH),
    (r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", "IP Address", Severity.LOW),
]

# ---------------------------------------------------------------------------
# Presidio entity → severity mapping
# ---------------------------------------------------------------------------

PRESIDIO_ENTITY_SEVERITY = {
    "PERSON": Severity.MEDIUM,
    "EMAIL_ADDRESS": Severity.MEDIUM,
    "PHONE_NUMBER": Severity.MEDIUM,
    "US_SSN": Severity.CRITICAL,
    "CREDIT_CARD": Severity.CRITICAL,
    "IBAN_CODE": Severity.HIGH,
    "US_PASSPORT": Severity.HIGH,
    "US_DRIVER_LICENSE": Severity.HIGH,
    "MEDICAL_LICENSE": Severity.HIGH,
    "US_BANK_NUMBER": Severity.HIGH,
    "IP_ADDRESS": Severity.LOW,
    "DATE_TIME": Severity.LOW,
    "LOCATION": Severity.LOW,
    "ORGANIZATION": Severity.LOW,
    "NRP": Severity.MEDIUM,          # Nationality/Religion/Political group
    "AGE": Severity.LOW,
    "URL": Severity.LOW,
}

_presidio_available = False
_analyzer = None


def _init_presidio():
    global _presidio_available, _analyzer
    try:
        from presidio_analyzer import AnalyzerEngine
        _analyzer = AnalyzerEngine()
        _presidio_available = True
    except Exception:
        _presidio_available = False


_init_presidio()


def _run_presidio(content: str, filename: str) -> List[Finding]:
    if not _presidio_available or _analyzer is None:
        return []

    findings: List[Finding] = []
    try:
        results = _analyzer.analyze(text=content, language="en")
    except Exception:
        return []

    # Group by entity type so we emit one finding per type (like regex layer)
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r.entity_type, []).append(r)

    for entity_type, matches in by_type.items():
        severity = PRESIDIO_ENTITY_SEVERITY.get(entity_type, Severity.LOW)
        count = len(matches)
        # Sample the first match for evidence (masked)
        sample = matches[0]
        snippet = content[sample.start: sample.end]
        masked = snippet[:3] + "***" + snippet[-2:] if len(snippet) > 6 else "***"

        findings.append(Finding(
            scanner="pii",
            severity=severity,
            title=f"[Presidio] {entity_type.replace('_', ' ').title()} — {count} instance(s) in '{filename}'",
            description=(
                f"Presidio NLP engine detected {count} instance(s) of {entity_type} in '{filename}'. "
                "Contextual NLP detection catches PII that regex-only scanners miss "
                "(e.g., names in sentences, addresses in free text)."
            ),
            location=filename,
            evidence=f"Sample: {masked} (and {count - 1} more)" if count > 1 else f"Sample: {masked}",
            recommendation=(
                "Remove or pseudonymize this PII before training, fine-tuning, or storing. "
                "Use Presidio Anonymizer for automated de-identification."
            ),
        ))
    return findings


def _run_regex(content: str, filename: str) -> List[Finding]:
    findings: List[Finding] = []
    for pattern, label, severity in REGEX_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            count = len(matches)
            findings.append(Finding(
                scanner="pii",
                severity=severity,
                title=f"{label} found in '{filename}'",
                description=(
                    f"{count} instance(s) of {label} detected via structural pattern matching. "
                    "PII in training data or model outputs may violate GDPR, CCPA, or HIPAA."
                ),
                location=filename,
                evidence=f"{count} match(es) found",
                recommendation=(
                    "Anonymize or pseudonymize PII before storing in datasets or including in model inputs."
                ),
            ))
    return findings


def scan(content: str, filename: str) -> List[Finding]:
    findings: List[Finding] = []
    # Presidio NLP first (richer, contextual)
    findings.extend(_run_presidio(content, filename))
    # Regex always runs — catches structural formats Presidio may miss
    findings.extend(_run_regex(content, filename))
    return findings
