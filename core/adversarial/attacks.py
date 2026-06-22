"""
Adversarial attack engine.

Numeric attacks (FGSM, PGD) run against a real 2-layer MLP trained on synthetic
data — genuine forward + backward passes, real autograd gradients, real softmax
confidence scores. TextFooler applies word substitution + Unicode homoglyphs.

Model cache: one MLP per input dimension, trained once and reused.
"""

import re
from typing import Optional
import numpy as np
from models.schemas import AdversarialRequest, AdversarialResult, AttackType

# ---------------------------------------------------------------------------
# Real PyTorch model
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_MODEL_CACHE: dict[int, "nn.Module"] = {}
_TRAINED_DIMS: set[int] = set()


def _build_and_train_model(input_dim: int) -> "nn.Module":
    """
    Train a small 2-layer MLP (input_dim → 64 → 32 → 2) on synthetic data.
    Task: classify whether the mean of each input vector is above 0.5.
    Training takes < 300ms on CPU for any reasonable input_dim.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(80):
        X = torch.rand(128, input_dim)
        y = (X.mean(dim=1) > 0.5).long()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _get_model(input_dim: int) -> "nn.Module":
    if input_dim not in _MODEL_CACHE:
        _MODEL_CACHE[input_dim] = _build_and_train_model(input_dim)
    return _MODEL_CACHE[input_dim]


# ---------------------------------------------------------------------------
# Gradient helpers (real autograd or numpy fallback)
# ---------------------------------------------------------------------------

def _real_gradient(
    x: np.ndarray, label: int, model: "nn.Module"
) -> tuple[float, np.ndarray]:
    """Forward + backward pass on the real MLP. Returns (loss, gradient)."""
    x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor([label], dtype=torch.long)
    logits = model(x_t.unsqueeze(0))
    loss = nn.CrossEntropyLoss()(logits, y_t)
    loss.backward()
    grad = x_t.grad.detach().numpy().copy()
    return float(loss.item()), grad


def _numpy_gradient(x: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Pure-numpy fallback when torch is not installed."""
    threshold = 0.5
    predictions = (x > threshold).astype(np.float32)
    loss = float(np.mean(np.abs(predictions - labels.astype(np.float32)[: len(x)])))
    grad = np.where(labels[: len(x)] == 1, x - threshold, threshold - x).astype(np.float32)
    return loss, grad


def _get_confidence(x: np.ndarray, model: "nn.Module") -> tuple[float, int]:
    """Returns (max_softmax_prob, predicted_class)."""
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        probs = torch.softmax(model(x_t), dim=-1).squeeze(0).numpy()
    return float(probs.max()), int(probs.argmax())


# ---------------------------------------------------------------------------
# Text perturbation helpers
# ---------------------------------------------------------------------------

TEXT_SUBSTITUTIONS = {
    "good": ["great", "fine", "okay"],
    "bad": ["poor", "awful", "terrible"],
    "happy": ["pleased", "glad", "content"],
    "sad": ["unhappy", "upset", "gloomy"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "delayed"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "minor"],
    "smart": ["intelligent", "clever", "bright"],
    "strong": ["powerful", "robust", "tough"],
    "the": ["a", "this", "that"],
    "is": ["was", "seems", "appears"],
    "not": ["never", "hardly"],
}

HOMOGLYPHS = {
    "a": "а",  # Cyrillic а
    "e": "е",  # Cyrillic е
    "o": "о",  # Cyrillic о
    "i": "і",  # Cyrillic і
    "c": "с",  # Cyrillic с
}


def _perturb_words(text: str, epsilon: float) -> str:
    words = text.split()
    rate = min(1.0, epsilon * 5)
    result = []
    for word in words:
        stripped = re.sub(r"[^a-z]", "", word.lower())
        if stripped in TEXT_SUBSTITUTIONS and np.random.random() < rate:
            result.append(np.random.choice(TEXT_SUBSTITUTIONS[stripped]))
        else:
            result.append(word)
    return " ".join(result)


def _perturb_homoglyphs(text: str, epsilon: float) -> str:
    rate = min(1.0, epsilon * 3)
    return "".join(
        HOMOGLYPHS[ch.lower()] if ch.lower() in HOMOGLYPHS and np.random.random() < rate else ch
        for ch in text
    )


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

def fgsm(request: AdversarialRequest) -> AdversarialResult:
    """Fast Gradient Sign Method — single-step L∞ attack against a real MLP."""
    x_orig = np.array(request.input_array, dtype=np.float32)
    eps = request.epsilon
    label = (request.labels[0] if request.labels else int(x_orig.mean() > 0.5))

    if _TORCH_AVAILABLE:
        model = _get_model(len(x_orig))
        orig_conf, orig_class = _get_confidence(x_orig, model)
        _, grad = _real_gradient(x_orig, label, model)
    else:
        labels_arr = np.array([label], dtype=np.int32)
        orig_conf = float(np.abs(x_orig.mean() - 0.5))
        orig_class = int(x_orig.mean() > 0.5)
        _, grad = _numpy_gradient(x_orig, labels_arr)

    x_adv = np.clip(x_orig + eps * np.sign(grad), 0.0, 1.0)

    if _TORCH_AVAILABLE:
        adv_conf, adv_class = _get_confidence(x_adv, model)
        conf_drop = max(0.0, orig_conf - adv_conf)
        success = adv_class != orig_class
    else:
        adv_conf = float(np.abs(x_adv.mean() - 0.5))
        conf_drop = max(0.0, orig_conf - adv_conf)
        success = conf_drop > 0.05

    return AdversarialResult(
        attack_type=AttackType.FGSM,
        epsilon=eps,
        original_input=x_orig.tolist(),
        adversarial_input=x_adv.tolist(),
        perturbation_norm=round(float(np.linalg.norm(x_adv - x_orig)), 6),
        success=success,
        confidence_drop=round(conf_drop, 4),
    )


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

def pgd(request: AdversarialRequest) -> AdversarialResult:
    """Projected Gradient Descent — iterative L∞ attack (Madry et al.) against a real MLP."""
    x_orig = np.array(request.input_array, dtype=np.float32)
    eps = request.epsilon
    steps = request.pgd_steps
    alpha = (eps / max(steps, 1)) * 2.5
    label = (request.labels[0] if request.labels else int(x_orig.mean() > 0.5))

    if _TORCH_AVAILABLE:
        model = _get_model(len(x_orig))
        orig_conf, orig_class = _get_confidence(x_orig, model)
    else:
        orig_conf = float(np.abs(x_orig.mean() - 0.5))
        orig_class = int(x_orig.mean() > 0.5)

    # Random start within ε-ball
    x_adv = np.clip(
        x_orig + np.random.uniform(-eps, eps, x_orig.shape).astype(np.float32),
        0.0, 1.0,
    )

    for _ in range(steps):
        if _TORCH_AVAILABLE:
            _, grad = _real_gradient(x_adv, label, model)
        else:
            _, grad = _numpy_gradient(x_adv, np.array([label], dtype=np.int32))

        x_adv = x_adv + alpha * np.sign(grad)
        x_adv = np.clip(x_adv, x_orig - eps, x_orig + eps)
        x_adv = np.clip(x_adv, 0.0, 1.0)

    if _TORCH_AVAILABLE:
        adv_conf, adv_class = _get_confidence(x_adv, model)
        conf_drop = max(0.0, orig_conf - adv_conf)
        success = adv_class != orig_class
    else:
        adv_conf = float(np.abs(x_adv.mean() - 0.5))
        conf_drop = max(0.0, orig_conf - adv_conf)
        success = conf_drop > 0.05

    return AdversarialResult(
        attack_type=AttackType.PGD,
        epsilon=eps,
        original_input=x_orig.tolist(),
        adversarial_input=x_adv.tolist(),
        perturbation_norm=round(float(np.linalg.norm(x_adv - x_orig)), 6),
        success=success,
        confidence_drop=round(conf_drop, 4),
    )


# ---------------------------------------------------------------------------
# TextFooler
# ---------------------------------------------------------------------------

def textfooler(request: AdversarialRequest) -> AdversarialResult:
    """Word-substitution + homoglyph injection for NLP robustness testing."""
    text = request.text
    eps = request.epsilon
    adv_text = _perturb_homoglyphs(_perturb_words(text, eps), eps * 0.5)

    orig_words = set(text.lower().split())
    adv_words = set(adv_text.lower().split())
    changed = len(orig_words.symmetric_difference(adv_words))
    conf_drop = min(1.0, changed / max(len(orig_words), 1) * eps * 10)
    norm = float(sum(a != b for a, b in zip(text, adv_text))) / max(len(text), 1)

    return AdversarialResult(
        attack_type=AttackType.TEXTFOOLER,
        epsilon=eps,
        original_input=list(text),
        adversarial_input=list(adv_text),
        perturbation_norm=round(norm, 6),
        success=conf_drop > 0.1,
        confidence_drop=round(conf_drop, 4),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_attack(request: AdversarialRequest) -> AdversarialResult:
    if request.attack_type == AttackType.FGSM:
        if not request.input_array:
            raise ValueError("FGSM requires input_array (list of floats in [0,1])")
        return fgsm(request)
    elif request.attack_type == AttackType.PGD:
        if not request.input_array:
            raise ValueError("PGD requires input_array (list of floats in [0,1])")
        return pgd(request)
    elif request.attack_type == AttackType.TEXTFOOLER:
        if not request.text:
            raise ValueError("TextFooler requires text")
        return textfooler(request)
    raise ValueError(f"Unknown attack type: {request.attack_type}")
