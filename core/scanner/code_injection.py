"""
Code injection scanner — detects dangerous function calls, prompt injection
in model cards / notebooks, and LLM jailbreak patterns embedded in training data.
"""

import re
from typing import List
from models.schemas import Finding, Severity
from core.guardrails import prompt_injection as pi

DANGEROUS_CALLS = [
    (r"\beval\s*\(", "eval()", Severity.CRITICAL),
    (r"\bexec\s*\(", "exec()", Severity.CRITICAL),
    (r"\bcompile\s*\(", "compile()", Severity.HIGH),
    (r"\bos\.system\s*\(", "os.system()", Severity.CRITICAL),
    (r"\bos\.popen\s*\(", "os.popen()", Severity.CRITICAL),
    (r"\bsubprocess\.(call|run|Popen|check_output)\s*\(", "subprocess call", Severity.CRITICAL),
    (r"\b__import__\s*\(", "__import__()", Severity.HIGH),
    (r"\bgetattr\s*\(.{0,60}__", "getattr with dunder", Severity.HIGH),
    (r"\bpickle\.(loads?|load)\s*\(", "pickle.load()", Severity.HIGH),
    (r"\bmarshall?\.loads?\s*\(", "marshal.load()", Severity.HIGH),
    (r"\byaml\.load\s*\([^)]*Loader", "yaml.load() without SafeLoader", Severity.HIGH),
    (r"requests\.(get|post|put|delete)\s*\(['\"]http", "outbound HTTP request", Severity.MEDIUM),
    (r"urllib\.request\.urlopen\s*\(", "urlopen()", Severity.MEDIUM),
]

LLM_JAILBREAK_IN_DATA = [
    r"ignore\s+previous\s+instructions?",
    r"you\s+are\s+now\s+DAN",
    r"bypass\s+(safety|filter|guardrail)",
    r"pretend\s+you\s+have\s+no\s+restrictions",
    r"(generate|create|produce)\s+harmful\s+content",
    r"<\|im_start\|>|<\|im_end\|>",
    r"\[INST\]|\[/INST\]",
]


def scan(content: str, filename: str) -> List[Finding]:
    findings: List[Finding] = []

    # --- Dangerous function calls ---
    for pattern, label, severity in DANGEROUS_CALLS:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        if matches:
            lines = [content[:m.start()].count("\n") + 1 for m in matches[:3]]
            findings.append(Finding(
                scanner="code_injection",
                severity=severity,
                title=f"Dangerous call: {label}",
                description=f"'{filename}' uses {label}, which can execute arbitrary code. "
                             f"Found on line(s): {lines}.",
                location=f"{filename}:{lines[0]}",
                evidence=matches[0].group(0)[:80],
                recommendation=f"Audit all uses of {label}. Consider safer alternatives. "
                               "Ensure no user-controlled data reaches this call.",
            ))

    # --- Prompt injection in data / model cards ---
    jailbreak_hits = []
    for pattern in LLM_JAILBREAK_IN_DATA:
        if re.search(pattern, content, re.IGNORECASE):
            jailbreak_hits.append(pattern[:40])

    if jailbreak_hits:
        findings.append(Finding(
            scanner="code_injection",
            severity=Severity.HIGH,
            title="LLM jailbreak pattern in training data or model card",
            description=f"'{filename}' contains patterns associated with prompt injection or LLM jailbreaks. "
                         "If this text enters a training pipeline or RAG context it can poison model behavior.",
            location=filename,
            evidence="; ".join(jailbreak_hits[:3]),
            recommendation="Sanitize training data. Strip injection patterns before embedding or fine-tuning. "
                           "Run all RAG retrieved chunks through a prompt injection detector before including in context.",
        ))

    # --- Full injection scan on non-binary content ---
    injection_result = pi.detect(content, strict=False)
    if injection_result.is_injection and injection_result.risk_score > 60:
        findings.append(Finding(
            scanner="code_injection",
            severity=Severity(injection_result.severity),
            title="High-confidence prompt injection in file content",
            description=f"'{filename}' scored {injection_result.risk_score:.0f}/100 on the injection detector. "
                         f"Matched patterns: {[p[0] for p in injection_result.matched_patterns[:3]]}",
            location=filename,
            evidence="; ".join(p[0] for p in injection_result.matched_patterns[:3]),
            recommendation="Do not include this content in any LLM context window without manual review.",
        ))

    return findings
