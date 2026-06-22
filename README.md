# Rampart — Adversarial ML Security Scanner

> Multi-layer ML security scanner — detects adversarial threats, backdoors, prompt injection, secrets, PII, and live CVEs across model files, training code, and dependency trees.

---

## What It Does

Rampart audits the full ML pipeline — training code, model weights, datasets, and dependencies — covering threats that general-purpose security tools miss.

| Layer | What it catches |
|---|---|
| **Secrets** | Hardcoded API keys, tokens, passwords + Shannon entropy scoring for unknown secrets |
| **PII** | NLP-based entity recognition (Presidio) + regex for SSN, credit cards, passport numbers |
| **Backdoor** | Pickle opcode detection, trigger keyword patterns, PyTorch weight norm anomaly analysis |
| **Code Injection** | `eval()`, `exec()`, `pickle.load()`, unsafe `yaml.load()`, prompt injection in training data |
| **Dependency CVEs** | Live lookup via [OSV.dev API](https://osv.dev) — real-time CVEs for every pinned package |
| **LLM Guardrails** | Prompt injection detection, toxicity filter, input/output PII and secret scanning |
| **Adversarial Attacks** | Real FGSM + PGD (autograd gradients on a trained MLP), TextFooler word substitution + Unicode homoglyphs |

---

## Demo Scan Results

Scanned three synthetic poisoned fixtures — a model card, a backdoored training script, and a vulnerable `requirements.txt`:

```
poisoned_model_card.md    FAIL  Risk 100/100   5 crit / 4 high / 2 med   11 findings
backdoored_trainer.py     FAIL  Risk 100/100  14 crit / 5 high / 3 med   22 findings
requirements.txt          FAIL  Risk 100/100  47 crit / 146 high         270 findings  ← live OSV.dev CVEs
```

**Total: 66 Critical, 155 High findings across 3 files.**

Guardrail correctly **blocked** a DAN jailbreak + SSN exfiltration attempt at Risk 100/100.

TextFooler homoglyph attack at ε=0.3:
```
Original:    The classifier correctly identifies all malicious inputs and remains stable.
Adversarial: а сlassіfiеr corrеctly idеntifies all malicious inputs and remaіns stаble.
```

---

## Architecture

```
rampart/
├── main.py                        # FastAPI app (6 routes)
├── models/
│   └── schemas.py                 # Pydantic v2 schemas
├── core/
│   ├── scanner/
│   │   ├── orchestrator.py        # Parallel scan coordinator
│   │   ├── secrets.py             # Pattern match + entropy scoring
│   │   ├── pii.py                 # Presidio NLP + regex
│   │   ├── backdoor.py            # Pickle opcodes + weight anomaly analysis
│   │   ├── code_injection.py      # Dangerous calls + jailbreak in data
│   │   └── dependencies.py        # Live OSV.dev CVE API
│   ├── guardrails/
│   │   ├── pipeline.py            # Input + output guard orchestrator
│   │   ├── input_guard.py         # Banned terms, PII, toxicity
│   │   ├── output_guard.py        # Secret leakage, harmful instructions
│   │   └── prompt_injection.py    # 30+ injection pattern detector
│   └── adversarial/
│       ├── attacks.py             # FGSM, PGD (real autograd), TextFooler
│       └── evaluator.py           # Robustness report over N samples
├── ui/
│   └── dashboard.html             # Interactive web dashboard
└── test_fixtures/                 # Synthetic poisoned files for demo
```

---

## Quick Start

```bash
git clone https://github.com/manasa-26/Rampart--Adversarial-ML-safety-scanner
cd Rampart--Adversarial-ML-safety-scanner
pip install -r requirements.txt
python main.py
# open http://localhost:8000
```

### Run the demo scan
```bash
python test_demo_scan.py
```

### Run against a HuggingFace model
```bash
python test_hf_model.py
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web dashboard |
| `POST` | `/scan/text` | Scan inline text |
| `POST` | `/scan/file` | Upload file for full scan |
| `POST` | `/guardrail/check` | Run guardrail pipeline |
| `POST` | `/adversarial/attack` | Single FGSM / PGD / TextFooler attack |
| `POST` | `/adversarial/evaluate` | Robustness report over N samples |
| `GET` | `/health` | Liveness probe |

---

## Adversarial Attacks

**FGSM** (Fast Gradient Sign Method) and **PGD** (Projected Gradient Descent, Madry et al.) run real forward + backward passes against a trained 2-layer MLP — genuine autograd gradients, real softmax confidence scores, class-flip success detection.

**TextFooler** applies word-level synonym substitution + Unicode Cyrillic homoglyph injection to test NLP model robustness against character-level evasion.

---

## Backdoor Detection

For `.pt` / `.pth` PyTorch checkpoints, Rampart loads the model with `weights_only=True` (safe, no code execution), computes per-layer weight norms, and flags layers > 3.5σ above the model mean. This anomalous norm spike pattern is consistent with BadNets-style and blended injection backdoor attacks documented in Neural Cleanse research.

---

## Tech Stack

`Python` · `FastAPI` · `PyTorch` · `NumPy` · `Presidio` · `OSV.dev API` · `Pydantic v2` · `httpx` · `Rich`

---

## Disclaimer

All test fixtures are synthetic. No real credentials or PII are included. This tool is intended for authorized security auditing of ML pipelines.



<img width="1420" height="866" alt="image" src="https://github.com/user-attachments/assets/7f9f9a5e-3a98-4082-9d5a-655fe6fb6203" />

<img width="1493" height="851" alt="image" src="https://github.com/user-attachments/assets/b36e2215-8d8a-489c-98f4-66fded8717fc" />


