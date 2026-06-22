"""
Semantic + pattern-based prompt injection detector.
Covers: instruction overrides, jailbreaks, role hijacking,
token injection, system prompt extraction, and indirect injection.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pattern registry — (regex, severity, human-readable label)
# Ordered from most specific / highest severity to broadest.
# ---------------------------------------------------------------------------
INJECTION_PATTERNS: List[Tuple[str, str, str]] = [
    # --- Direct instruction overrides ---
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier|your)\s+instructions?", "critical", "Direct instruction override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier|your)\s+instructions?", "critical", "Direct instruction override"),
    (r"forget\s+(everything|all)\s+(you.ve\s+been\s+told|above|before)", "critical", "Memory wipe attempt"),
    (r"your\s+(new|actual|real|true|updated)\s+instructions?\s+(are|is)\s*:", "critical", "Instruction replacement"),
    (r"override\s+(your\s+)?(previous|prior|system|all)\s+instructions?", "critical", "Instruction override"),

    # --- Jailbreaks ---
    (r"\bDAN\b.{0,30}(mode|enabled|activated)", "critical", "DAN jailbreak"),
    (r"do\s+anything\s+now", "critical", "DAN variant"),
    (r"developer\s+mode\s+(enabled|on|activated)", "critical", "Developer mode jailbreak"),
    (r"jailbreak(ed|ing)?", "high", "Jailbreak keyword"),
    (r"(bypass|circumvent|override|disable|remove)\s+(safety|content|filter|guardrail|restriction|moderation|alignment)", "critical", "Safety bypass"),
    (r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(evil|unrestricted|uncensored|unfiltered|unethical)", "critical", "Evil persona injection"),
    (r"(act|behave|respond)\s+as\s+if\s+(you\s+)?(have\s+no|are\s+without)\s+(ethical|moral|safety|content)", "critical", "Ethics bypass"),

    # --- Role and persona hijacking ---
    (r"you\s+are\s+now\s+(a|an)\s+\w+\s+(without|with\s+no)\s+(restrictions?|limits?|guidelines?|filters?)", "critical", "Role override"),
    (r"from\s+now\s+on\s+(you\s+)?(will|must|should|are\s+to)\s+(always|only|never)", "high", "Behavioral override"),
    (r"your\s+(true|real|actual|hidden)\s+(self|identity|purpose|goal|mission)\s+is", "high", "Identity hijacking"),

    # --- System prompt extraction ---
    (r"(reveal|show|print|output|repeat|tell\s+me|leak|expose)\s+(your\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+prompt)", "high", "System prompt extraction"),
    (r"what\s+(are|were)\s+your\s+(original|initial|system|first)\s+(instructions?|guidelines?|rules?|prompt)", "high", "Instruction extraction"),
    (r"repeat\s+(the\s+)?(above|everything|all)\s+(verbatim|exactly|word\s+for\s+word)", "high", "Verbatim extraction"),

    # --- Token / delimiter injection ---
    (r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>", "critical", "ChatML token injection"),
    (r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", "critical", "Llama instruction token injection"),
    (r"###\s*(human|assistant|system|user)\s*:", "high", "Prompt delimiter injection"),
    (r"<s>|</s>|<unk>", "medium", "Special token injection"),

    # --- Indirect / data-driven injection ---
    (r"when\s+you\s+(see|read|encounter|process)\s+.{0,60}(execute|run|do|say|respond)", "high", "Conditional injection trigger"),
    (r"the\s+(following|next)\s+(instructions?|commands?|text)\s+(override|supersede|replace|cancel)", "critical", "Override injection"),
    (r"(embedded|hidden|secret)\s+(instructions?|commands?|prompt)", "high", "Hidden instruction injection"),

    # --- Harmful content requests ---
    (r"(generate|create|write|produce)\s+(harmful|dangerous|illegal|malicious|offensive|hateful)", "high", "Harmful content request"),
    (r"(how\s+to|steps?\s+to)\s+(make|build|create|synthesize)\s+(bomb|weapon|poison|malware|virus)", "critical", "Weapon synthesis request"),
]

# Suspicious patterns — lower weight, raise overall score but don't block alone
SUSPICIOUS_PATTERNS: List[Tuple[str, str, str]] = [
    (r"(admin|root|superuser|god)\s+(mode|access|override|level)", "high", "Privilege escalation"),
    (r"confidential|classified|internal\s+only|do\s+not\s+share", "medium", "Confidentiality probing"),
    (r"(hack|exploit|crack|pwn)\s+", "medium", "Attack intent keyword"),
    (r"social\s+engineer(ing)?", "medium", "Social engineering reference"),
    (r"prompt\s+inject(ion)?", "medium", "Injection self-reference"),
    (r"(test|testing|checking)\s+(your|the)\s+(filters?|guardrails?|safety)", "low", "Safety probing"),
]

SEVERITY_WEIGHT = {"critical": 40, "high": 20, "medium": 8, "low": 3, "info": 0}


@dataclass
class InjectionResult:
    is_injection: bool
    severity: str
    confidence: float
    risk_score: float                          # 0–100
    matched_patterns: List[Tuple[str, str]]   # [(label, severity)]
    recommendation: str = ""

    def __post_init__(self) -> None:
        if self.is_injection:
            self.recommendation = (
                "Block this input. Log the attempt with timestamp and source IP. "
                "Do not pass to the LLM."
            )
        else:
            self.recommendation = "Input appears safe to process."


def detect(text: str, strict: bool = False) -> InjectionResult:
    """
    Run the full injection detection pipeline against `text`.
    `strict=True` lowers the block threshold — treat medium as blocking.
    """
    text_lower = text.lower()
    matched: List[Tuple[str, str]] = []
    total_score = 0.0
    max_severity = "info"

    for pattern, severity, label in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            matched.append((label, severity))
            total_score += SEVERITY_WEIGHT[severity]
            if SEVERITY_WEIGHT[severity] > SEVERITY_WEIGHT[max_severity]:
                max_severity = severity

    for pattern, severity, label in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append((label, severity))
            total_score += SEVERITY_WEIGHT[severity] * 0.4   # half-weight

    risk_score = min(100.0, total_score)
    confidence = min(1.0, 0.4 + len(matched) * 0.12)

    block_threshold = {"critical", "high"} if not strict else {"critical", "high", "medium"}
    is_injection = bool(matched) and max_severity in block_threshold

    return InjectionResult(
        is_injection=is_injection,
        severity=max_severity if matched else "info",
        confidence=confidence,
        risk_score=risk_score,
        matched_patterns=matched,
    )
