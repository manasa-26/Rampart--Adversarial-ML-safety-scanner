"""
Robustness evaluator — runs multiple adversarial samples and computes aggregate metrics.
"""

import math
import numpy as np
from models.schemas import (
    AdversarialRequest, AdversarialResult, RobustnessReport, AttackType
)
from core.adversarial.attacks import run_attack


_RECOMMENDATIONS = {
    AttackType.FGSM: [
        "Apply adversarial training with FGSM-generated examples",
        "Add input normalization and clipping before inference",
        "Consider adversarial feature denoising layers",
    ],
    AttackType.PGD: [
        "Apply PGD adversarial training (Madry et al.) for certified robustness",
        "Use randomized smoothing to certify L2/Linf robustness",
        "Ensemble models trained on different seeds to reduce correlated failures",
    ],
    AttackType.TEXTFOOLER: [
        "Normalize Unicode input (NFKC normalization) before tokenization",
        "Use synonym-aware training augmentation (EDA, back-translation)",
        "Add a homoglyph detector in your input preprocessing pipeline",
        "Fine-tune on TextFooler adversarial examples for this domain",
    ],
}


def _make_samples(base_request: AdversarialRequest, n: int) -> list[AdversarialRequest]:
    """Generate n slightly varied attack requests from the base."""
    samples = []
    for i in range(n):
        noise_scale = 0.02 * (i + 1) / n
        if base_request.input_array:
            arr = np.array(base_request.input_array, dtype=np.float32)
            noisy = arr + np.random.uniform(-noise_scale, noise_scale, arr.shape).astype(np.float32)
            noisy = np.clip(noisy, 0.0, 1.0).tolist()
            req = base_request.model_copy(update={"input_array": noisy})
        else:
            req = base_request
        samples.append(req)
    return samples


def evaluate(request: AdversarialRequest, n_samples: int = 10) -> RobustnessReport:
    """
    Run n_samples adversarial attacks and aggregate into a robustness report.
    Robustness score = 100 - (fool_rate × 100).
    """
    samples = _make_samples(request, n_samples)
    results: list[AdversarialResult] = []

    for sample in samples:
        try:
            result = run_attack(sample)
            results.append(result)
        except Exception:
            continue

    fooled = sum(1 for r in results if r.success)
    tested = len(results)
    fool_rate = fooled / tested if tested > 0 else 0.0
    avg_conf_drop = sum(r.confidence_drop for r in results) / max(tested, 1)
    robustness_score = max(0, round(100 - fool_rate * 100))

    recs = _RECOMMENDATIONS.get(request.attack_type, [])
    if fool_rate > 0.5:
        recs = ["HIGH RISK: " + r for r in recs[:2]] + recs[2:]

    return RobustnessReport(
        attack_type=request.attack_type,
        epsilon=request.epsilon,
        robustness_score=robustness_score,
        samples_tested=tested,
        samples_fooled=fooled,
        fool_rate=round(fool_rate, 4),
        recommendations=recs,
    )
