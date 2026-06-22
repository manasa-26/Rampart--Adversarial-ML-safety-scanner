"""
Adversarial ML Safety Guardrail — FastAPI application.
Routes:
  GET  /                       → Web dashboard
  POST /scan/text              → Scan inline text for secrets, PII, backdoors, etc.
  POST /scan/file              → Upload a file for full multi-scanner analysis
  POST /guardrail/check        → Run input+output guardrail pipeline on text
  POST /adversarial/attack     → Execute a single adversarial attack
  POST /adversarial/evaluate   → Full robustness evaluation (n_samples attacks)
  GET  /health                 → Liveness probe
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models.schemas import (
    GuardrailRequest, GuardrailResult,
    AdversarialRequest, AdversarialResult, RobustnessReport,
    ScanResult, AttackType,
)
from core.scanner.orchestrator import scan_content, scan_text
from core.guardrails.pipeline import GuardrailPipeline
from core.adversarial.attacks import run_attack
from core.adversarial.evaluator import evaluate

app = FastAPI(
    title="Adversarial ML Safety Guardrail",
    description="Multi-layer adversarial ML safety scanner and LLM guardrail API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = GuardrailPipeline()
_dashboard_path = Path(__file__).parent / "ui" / "dashboard.html"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    if _dashboard_path.exists():
        return HTMLResponse(_dashboard_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found — run from project root</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Scanner endpoints
# ---------------------------------------------------------------------------

@app.post("/scan/text", response_model=ScanResult)
async def scan_text_endpoint(
    text: str = Query(..., description="Text to scan"),
    label: str = Query("inline_text", description="Label for the result"),
):
    """Scan inline text for secrets, PII, code injection, backdoor indicators."""
    return await scan_text(text, label)


@app.post("/scan/file", response_model=ScanResult)
async def scan_file_endpoint(file: UploadFile = File(...)):
    """Upload a file (model, script, notebook, requirements.txt) for full scan."""
    raw = await file.read()
    filename = file.filename or "upload"
    # Attempt text decode; fall back to binary
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw
    return await scan_content(content, filename)


# ---------------------------------------------------------------------------
# Guardrail endpoints
# ---------------------------------------------------------------------------

@app.post("/guardrail/check", response_model=GuardrailResult)
async def guardrail_check(request: GuardrailRequest):
    """
    Run multi-layer guardrail pipeline:
    - Banned term check
    - PII detection
    - Toxicity filter
    - Prompt injection detection
    - Output safety check
    Returns whether the text is blocked and sanitized output if allowed.
    """
    return await _pipeline.check(request)


# ---------------------------------------------------------------------------
# Adversarial attack endpoints
# ---------------------------------------------------------------------------

@app.post("/adversarial/attack", response_model=AdversarialResult)
async def adversarial_attack(request: AdversarialRequest):
    """
    Execute a single adversarial attack.

    - **FGSM** / **PGD**: requires `input_array` (list of floats in [0,1])
    - **TEXTFOOLER**: requires `text`
    - `epsilon` controls attack strength (0.001 – 1.0)
    - `pgd_steps` controls PGD iterations (default 40)
    """
    try:
        return run_attack(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/adversarial/evaluate", response_model=RobustnessReport)
async def adversarial_evaluate(
    request: AdversarialRequest,
    n_samples: int = Query(10, ge=1, le=100, description="Number of attack samples to generate"),
):
    """
    Full robustness evaluation — runs n_samples adversarial attacks and returns
    aggregate metrics: fool rate, robustness score, and recommendations.
    """
    try:
        return evaluate(request, n_samples=n_samples)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
