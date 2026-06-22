"""
Backdoor and model poisoning scanner.
Three detection layers:
  1. Byte signatures — known malicious byte sequences and pickle opcodes
  2. Text patterns — trigger keywords, conditional trigger logic, encoded evals
  3. Weight distribution analysis — statistical anomalies in .pt/.pth model files
     (anomalous norm spikes and dead neuron ratios consistent with backdoor implantation)
"""

import re
import io
from typing import List, Union
from models.schemas import Finding, Severity

# ---------------------------------------------------------------------------
# Layer 1 — Known byte signatures
# ---------------------------------------------------------------------------

PICKLE_DANGER_OPCODES = [
    b"\x80\x04\x95",
    b"c__builtin__\nexec",
    b"cos\nsystem",
    b"csubprocess\ncall",
    b"csubprocess\nPopen",
    b"c__builtin__\neval",
    b"__reduce__",
]

MALICIOUS_SIGNATURES = [
    (b"curl http://", "Outbound curl call", Severity.HIGH),
    (b"wget http://", "Outbound wget call", Severity.HIGH),
    (b"MINER_START", "Crypto miner signature", Severity.CRITICAL),
    (b"/bin/sh -c", "Shell spawn", Severity.CRITICAL),
    (b"cmd.exe /c", "Windows shell spawn", Severity.CRITICAL),
    (b"powershell -EncodedCommand", "Encoded PowerShell", Severity.CRITICAL),
    (b"nc -e /bin/bash", "Netcat reverse shell", Severity.CRITICAL),
]

# ---------------------------------------------------------------------------
# Layer 2 — Text patterns
# ---------------------------------------------------------------------------

BACKDOOR_TEXT_PATTERNS = [
    (r"\bTRIGGER\b", "Trigger keyword in model", Severity.CRITICAL),
    (r"\bBACKDOOR\b", "Backdoor keyword in model", Severity.CRITICAL),
    (r"\bPOISON\b", "Poison keyword in model", Severity.HIGH),
    (r"target_label\s*=\s*\d+.*trigger", "Trigger-label binding pattern", Severity.CRITICAL),
    (r"if\s+.{0,30}trigger.{0,30}:\s*return\s+\d+", "Conditional trigger return", Severity.CRITICAL),
    (r"eval\(base64\.b64decode", "Base64-encoded eval in model", Severity.CRITICAL),
    (r"exec\(.*decode", "Encoded exec in model", Severity.CRITICAL),
    (r"urllib\.request\.urlopen|requests\.get\(.{0,100}http", "Outbound HTTP call in model", Severity.HIGH),
    (r"os\.system|subprocess\.(call|run|Popen)", "Shell execution in model file", Severity.CRITICAL),
]

# ---------------------------------------------------------------------------
# Layer 3 — Weight distribution analysis
# ---------------------------------------------------------------------------

_NORM_ZSCORE_THRESHOLD = 3.5   # layers this many σ above mean are suspicious
_DEAD_NEURON_THRESHOLD = 0.4   # >40% zero weights in a layer is anomalous


def _analyze_model_weights(content: bytes, filename: str) -> List[Finding]:
    """
    Load a .pt/.pth checkpoint with weights_only=True (safe), compute per-layer
    weight statistics, and flag statistical outliers consistent with backdoor implantation.

    Rationale: Backdoor attacks often create one or a few layers with disproportionately
    large weight norms (the poisoned feature detector). Neural Cleanse and spectral signature
    research both show this anomalous norm pattern.
    """
    findings: List[Finding] = []

    try:
        import torch
        import numpy as np
    except ImportError:
        return findings  # torch not installed, skip weight analysis

    try:
        buf = io.BytesIO(content)
        # weights_only=True prevents arbitrary code execution on load
        obj = torch.load(buf, map_location="cpu", weights_only=True)
    except Exception:
        return findings

    # Extract state dict from various checkpoint formats
    if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
        state_dict = obj
    elif isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj:
        state_dict = obj["model"]
    else:
        return findings

    # Collect weight tensors (skip biases, buffers — focus on weight matrices)
    tensors = [
        (name, t.float())
        for name, t in state_dict.items()
        if isinstance(t, torch.Tensor) and t.numel() > 16 and "weight" in name.lower()
    ]
    if len(tensors) < 2:
        return findings

    norms = np.array([float(t.norm().item()) for _, t in tensors])
    mean_n = float(norms.mean())
    std_n = float(norms.std())

    for (name, tensor), norm in zip(tensors, norms):
        # --- Anomalous norm spike ---
        if std_n > 1e-6:
            z = (norm - mean_n) / std_n
            if z > _NORM_ZSCORE_THRESHOLD:
                findings.append(Finding(
                    scanner="backdoor",
                    severity=Severity.HIGH,
                    title=f"Anomalous weight norm in layer '{name}'",
                    description=(
                        f"Layer '{name}' has weight norm {norm:.2f} — "
                        f"{z:.1f}σ above the model mean ({mean_n:.2f} ± {std_n:.2f}). "
                        "Backdoored models commonly show norm spikes in the poisoned feature detector layer. "
                        "This pattern is consistent with BadNets-style or blended injection attacks."
                    ),
                    location=filename,
                    evidence=f"Norm={norm:.2f}, model mean={mean_n:.2f}, z-score={z:.2f}",
                    recommendation=(
                        "Manually inspect this layer's weights. "
                        "Compare with a clean checkpoint from the original training run. "
                        "Consider retraining from scratch on verified-clean data."
                    ),
                ))

        # --- Dead neuron ratio ---
        if tensor.dim() >= 2:
            dead_rows = float((tensor.abs().sum(dim=tuple(range(1, tensor.dim()))) == 0).float().mean().item())
            if dead_rows > _DEAD_NEURON_THRESHOLD:
                findings.append(Finding(
                    scanner="backdoor",
                    severity=Severity.MEDIUM,
                    title=f"High dead-neuron ratio in layer '{name}' ({dead_rows*100:.0f}%)",
                    description=(
                        f"{dead_rows*100:.0f}% of output neurons in '{name}' have all-zero weights. "
                        "Unusually high dead-neuron ratios can indicate pruning-based backdoor injection, "
                        "where an attacker prunes clean neurons and inserts malicious ones."
                    ),
                    location=filename,
                    evidence=f"Dead neurons: {dead_rows*100:.1f}% (threshold: {_DEAD_NEURON_THRESHOLD*100:.0f}%)",
                    recommendation=(
                        "Verify this checkpoint hash against a known-good source. "
                        "Run the model on a clean held-out set and inspect neuron activation histograms."
                    ),
                ))

    if not findings:
        # Emit an INFO finding to confirm the weight scan ran
        findings.append(Finding(
            scanner="backdoor",
            severity=Severity.INFO,
            title=f"Weight distribution scan: no anomalies in {len(tensors)} layer(s)",
            description=(
                f"All {len(tensors)} weight layers are within {_NORM_ZSCORE_THRESHOLD}σ of the model norm mean. "
                "No dead-neuron anomalies detected."
            ),
            location=filename,
            evidence=f"Norm range: [{norms.min():.2f}, {norms.max():.2f}], mean={mean_n:.2f}",
            recommendation="Continue with standard model validation on a clean held-out set.",
        ))

    return findings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scan(content: Union[bytes, str], filename: str) -> List[Finding]:
    findings: List[Finding] = []
    is_binary = isinstance(content, bytes)
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # --- Binary path: byte signatures + pickle opcodes ---
    if is_binary:
        for sig, label, severity in MALICIOUS_SIGNATURES:
            if sig in content:
                findings.append(Finding(
                    scanner="backdoor",
                    severity=severity,
                    title=f"Malicious byte signature: {label}",
                    description=(
                        f"'{filename}' contains a known malicious byte sequence ({label}). "
                        "This is a strong indicator of a backdoored or supply-chain compromised model."
                    ),
                    location=filename,
                    evidence=f"Signature: {sig!r}",
                    recommendation="Do NOT load this model. Quarantine the file. Source it from a verified, signed registry.",
                ))

        for opcode in PICKLE_DANGER_OPCODES:
            if opcode in content:
                findings.append(Finding(
                    scanner="backdoor",
                    severity=Severity.CRITICAL,
                    title="Dangerous pickle opcode detected",
                    description=(
                        f"'{filename}' contains pickle opcodes that execute arbitrary Python on deserialization. "
                        "This is the primary supply-chain attack vector for ML models."
                    ),
                    location=filename,
                    evidence=f"Opcode: {opcode!r}",
                    recommendation=(
                        "Use .safetensors format. Never load .pkl from untrusted sources. "
                        "Use torch.load() with weights_only=True."
                    ),
                ))

        # Weight distribution analysis for PyTorch checkpoints
        if suffix in ("pt", "pth", "bin"):
            findings.extend(_analyze_model_weights(content, filename))

        return findings

    # --- Text path: regex pattern matching ---
    text = content if isinstance(content, str) else content.decode("utf-8", errors="ignore")
    for pattern, label, severity in BACKDOOR_TEXT_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            findings.append(Finding(
                scanner="backdoor",
                severity=severity,
                title=f"Backdoor indicator: {label}",
                description=f"'{filename}' contains a pattern associated with model backdooring: {label}.",
                location=filename,
                evidence=m.group(0)[:100],
                recommendation="Audit this code manually. Check for trigger-based conditional logic that changes model output.",
            ))
    return findings
