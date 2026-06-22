from pydantic import BaseModel, Field
from typing import List, Optional, Any, Literal
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Finding(BaseModel):
    scanner: str
    severity: Severity
    title: str
    description: str
    location: Optional[str] = None
    evidence: Optional[str] = None
    recommendation: str


class ScanResult(BaseModel):
    filename: str
    total_findings: int = 0
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    info: int = 0
    findings: List[Finding] = []
    risk_score: float = 0.0    # 0–100
    passed: bool = True        # False if any critical or high finding


class GuardrailRequest(BaseModel):
    text: str
    context: Optional[str] = None   # system prompt / conversation history
    check_pii: bool = True
    check_injection: bool = True
    check_toxicity: bool = True
    strict_mode: bool = False


class GuardrailCheck(BaseModel):
    check_name: str
    passed: bool
    severity: Optional[Severity] = None
    reason: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class GuardrailResult(BaseModel):
    text: str
    passed: bool
    blocked_reason: Optional[str] = None
    checks: List[GuardrailCheck]
    risk_score: float          # 0–100
    sanitized_text: Optional[str] = None


class AttackType(str, Enum):
    FGSM = "fgsm"
    PGD = "pgd"
    TEXTFOOLER = "textfooler"


class AdversarialRequest(BaseModel):
    text: Optional[str] = None
    input_array: Optional[List[float]] = None
    labels: Optional[List[int]] = None
    epsilon: float = Field(0.1, ge=0.001, le=1.0, description="Perturbation strength")
    attack_type: AttackType = AttackType.FGSM
    pgd_steps: int = Field(10, ge=1, le=100)
    target_class: Optional[int] = None


class AdversarialResult(BaseModel):
    attack_type: str
    epsilon: float
    original_input: Any
    adversarial_input: Any
    perturbation_norm: float
    success: bool
    confidence_drop: Optional[float] = None
    description: Optional[str] = None


class RobustnessReport(BaseModel):
    attack_type: str
    epsilon: float
    robustness_score: float    # 0–100
    samples_tested: int
    samples_fooled: int
    fool_rate: float           # 0–1
    recommendations: List[str]
