"""
Scanner orchestrator — runs all 5 scanners in parallel and aggregates results.
"""

import asyncio
from pathlib import Path
from typing import Union
from models.schemas import ScanResult, Finding, Severity
from core.scanner import secrets, pii, backdoor, code_injection, dependencies

SEVERITY_WEIGHT = {
    Severity.CRITICAL: 40,
    Severity.HIGH: 20,
    Severity.MEDIUM: 8,
    Severity.LOW: 3,
    Severity.INFO: 0,
}

BINARY_EXTENSIONS = {".pkl", ".pt", ".pth", ".bin", ".model", ".onnx", ".h5", ".pb"}
DEPENDENCY_FILES = {"requirements.txt", "package.json", "Pipfile", "setup.cfg"}
TEXT_SCAN_EXTENSIONS = {
    ".py", ".js", ".ts", ".ipynb", ".yaml", ".yml", ".json",
    ".txt", ".md", ".sh", ".env", ".cfg", ".ini", ".toml",
}


def _compute_risk(findings: list[Finding]) -> int:
    raw = sum(SEVERITY_WEIGHT.get(f.severity, 0) for f in findings)
    return min(100, raw)


def _count(findings: list[Finding], severity: Severity) -> int:
    return sum(1 for f in findings if f.severity == severity)


async def _run_scanners(content: Union[bytes, str], filename: str) -> list[Finding]:
    loop = asyncio.get_event_loop()
    findings: list[Finding] = []
    suffix = Path(filename).suffix.lower()
    name = Path(filename).name.lower()

    tasks = []

    if isinstance(content, bytes) and suffix in BINARY_EXTENSIONS:
        # Binary model file: byte-level + weight analysis (sync, CPU-bound)
        tasks.append(loop.run_in_executor(None, backdoor.scan, content, filename))
    else:
        text_content = content if isinstance(content, str) else content.decode("utf-8", errors="ignore")
        # Sync scanners — run in thread pool
        tasks.append(loop.run_in_executor(None, secrets.scan, text_content, filename))
        tasks.append(loop.run_in_executor(None, pii.scan, text_content, filename))
        tasks.append(loop.run_in_executor(None, backdoor.scan, text_content, filename))
        tasks.append(loop.run_in_executor(None, code_injection.scan, text_content, filename))
        # dependencies.scan is async (hits OSV.dev API) — schedule as coroutine directly
        if name in DEPENDENCY_FILES:
            tasks.append(dependencies.scan(text_content, filename))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            continue
        if isinstance(r, list):
            findings.extend(r)

    return findings


async def scan_content(content: Union[bytes, str], filename: str) -> ScanResult:
    findings = await _run_scanners(content, filename)
    risk = _compute_risk(findings)
    return ScanResult(
        filename=filename,
        total_findings=len(findings),
        critical=_count(findings, Severity.CRITICAL),
        high=_count(findings, Severity.HIGH),
        medium=_count(findings, Severity.MEDIUM),
        low=_count(findings, Severity.LOW),
        info=_count(findings, Severity.INFO),
        findings=findings,
        risk_score=risk,
        passed=risk < 20 and _count(findings, Severity.CRITICAL) == 0,
    )


async def scan_text(text: str, label: str = "inline_text") -> ScanResult:
    return await scan_content(text, label)
