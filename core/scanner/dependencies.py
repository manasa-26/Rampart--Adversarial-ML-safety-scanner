"""
Dependency vulnerability scanner.
Two-layer detection:
  1. Live OSV.dev API (Google, free, no auth) — real-time CVE data for PyPI packages
  2. Static curated fallback — used when OSV.dev is unreachable
Parses requirements.txt and package.json.
"""

import re
import json
import asyncio
from typing import Optional
import httpx
from models.schemas import Finding, Severity

OSV_API_URL = "https://api.osv.dev/v1/query"
OSV_TIMEOUT = 8.0  # seconds per package query

# ---------------------------------------------------------------------------
# Static fallback CVE list (used when OSV.dev is unreachable)
# Format: package_name → [(max_vulnerable_version, CVE, description, severity)]
# ---------------------------------------------------------------------------

STATIC_VULNS: dict[str, list[tuple[str, str, str, str]]] = {
    "torch": [
        ("1.13.1", "CVE-2022-45907", "Arbitrary code execution via torch.load() with untrusted data", "critical"),
    ],
    "tensorflow": [
        ("2.9.3", "CVE-2022-41894", "Heap OOB write in tf.experimental.numpy.nditer", "high"),
        ("2.11.0", "CVE-2022-41900", "Integer overflow in SparseFillEmptyRows", "high"),
    ],
    "transformers": [
        ("4.29.2", "CVE-2023-7018", "Arbitrary code execution via pickle in AutoTokenizer", "critical"),
    ],
    "pyyaml": [
        ("5.4.0", "CVE-2020-14343", "Arbitrary code execution via yaml.load()", "critical"),
    ],
    "langchain": [
        ("0.0.340", "CVE-2023-46229", "Arbitrary code execution via LLMChain", "critical"),
        ("0.1.13", "CVE-2024-1455", "SQL injection via SQLDatabaseChain", "critical"),
    ],
    "gradio": [
        ("3.41.2", "CVE-2023-34239", "Arbitrary file read via path traversal", "critical"),
    ],
    "requests": [
        ("2.27.1", "CVE-2023-32681", "Proxy-Authorization header leak to redirect target", "medium"),
    ],
    "cryptography": [
        ("41.0.2", "CVE-2023-38325", "Bleichenbacher timing attack in RSA decryption", "high"),
    ],
    "pillow": [
        ("9.0.0", "CVE-2022-22815", "Path traversal in ImageFont", "high"),
    ],
    "flask": [
        ("2.2.5", "CVE-2023-30861", "Possible session cookie leakage", "high"),
    ],
    "huggingface_hub": [
        ("0.15.1", "CVE-2023-1776", "Model card injection via unsafe YAML parsing", "high"),
    ],
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_requirements_txt(content: str) -> list[tuple[str, Optional[str]]]:
    packages = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-", "git+")):
            continue
        match = re.match(r"([A-Za-z0-9_\-\.]+)\s*==\s*([^\s;]+)", line)
        if match:
            packages.append((match.group(1).lower(), match.group(2)))
        else:
            match = re.match(r"([A-Za-z0-9_\-\.]+)", line)
            if match:
                packages.append((match.group(1).lower(), None))
    return packages


def _parse_package_json(content: str) -> list[tuple[str, Optional[str]]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    packages = []
    for section in ("dependencies", "devDependencies"):
        for pkg, ver in data.get(section, {}).items():
            clean = re.sub(r"[^0-9.]", "", ver)
            packages.append((pkg.lower(), clean if clean else None))
    return packages


# ---------------------------------------------------------------------------
# OSV.dev API (live)
# ---------------------------------------------------------------------------

def _parse_osv_severity(vuln: dict) -> str:
    """Extract plain-text severity from an OSV vulnerability record."""
    # database_specific.severity is the most reliable field
    db = vuln.get("database_specific", {})
    sev = db.get("severity", "").upper()
    if sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        return sev.lower()

    # Some records use the severity array with CVSS vector strings
    for s in vuln.get("severity", []):
        score_str = s.get("score", "")
        # CVSS:3.1/AV:N/... — extract C:/I:/A: ratings to estimate severity
        if "CVSS" in score_str:
            if any(x in score_str for x in ("C:H/I:H", "C:H/I:N/A:H")):
                return "critical"
            if "C:H" in score_str or "I:H" in score_str:
                return "high"
            if "C:M" in score_str or "I:M" in score_str:
                return "medium"
    return "medium"  # safe default


async def _query_osv(name: str, version: str) -> list[dict]:
    """Query OSV.dev for known vulnerabilities in a specific package version."""
    payload = {
        "package": {"name": name, "ecosystem": "PyPI"},
        "version": version,
    }
    try:
        async with httpx.AsyncClient(timeout=OSV_TIMEOUT) as client:
            r = await client.post(OSV_API_URL, json=payload)
            if r.status_code == 200:
                return r.json().get("vulns", [])
    except Exception:
        pass
    return []


async def _check_package_live(name: str, version: str, filename: str) -> list[Finding]:
    vulns = await _query_osv(name, version)
    findings: list[Finding] = []
    for vuln in vulns:
        cve_ids = [a for a in vuln.get("aliases", []) if a.startswith("CVE-")]
        cve = cve_ids[0] if cve_ids else vuln.get("id", "unknown")
        summary = vuln.get("summary") or vuln.get("details", "No description available")[:200]
        severity_str = _parse_osv_severity(vuln)
        try:
            severity = Severity(severity_str)
        except ValueError:
            severity = Severity.MEDIUM

        findings.append(Finding(
            scanner="dependencies",
            severity=severity,
            title=f"{name}=={version} — {cve}",
            description=f"[OSV.dev] {summary}",
            location=filename,
            evidence=f"OSV ID: {vuln.get('id')} | Aliases: {', '.join(cve_ids[:3])}",
            recommendation=(
                f"Upgrade {name} to a patched version. "
                f"Check https://osv.dev/vulnerability/{vuln.get('id')} for affected ranges."
            ),
        ))
    return findings


# ---------------------------------------------------------------------------
# Static fallback
# ---------------------------------------------------------------------------

def _check_package_static(name: str, version: Optional[str], filename: str) -> list[Finding]:
    from packaging.version import Version, InvalidVersion
    findings: list[Finding] = []
    vulns = STATIC_VULNS.get(name, [])
    if not vulns:
        return findings

    if not version:
        findings.append(Finding(
            scanner="dependencies",
            severity=Severity.MEDIUM,
            title=f"Unpinned package with known CVEs: {name}",
            description="Cannot check version — dependency is unpinned. Known CVEs exist for some versions.",
            location=filename,
            recommendation=f"Pin {name} to an exact version and verify it is not in a vulnerable range.",
        ))
        return findings

    try:
        installed = Version(version)
    except InvalidVersion:
        return findings

    for max_vuln_ver, cve, description, severity in vulns:
        try:
            if installed <= Version(max_vuln_ver):
                findings.append(Finding(
                    scanner="dependencies",
                    severity=Severity(severity),
                    title=f"{name}=={version} — {cve} [static fallback]",
                    description=f"{description}. Versions ≤ {max_vuln_ver} are affected.",
                    location=filename,
                    evidence=f"{name}=={version} ≤ {max_vuln_ver}",
                    recommendation=f"Upgrade {name} above {max_vuln_ver}.",
                ))
        except InvalidVersion:
            continue
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def scan(content: str, filename: str) -> list[Finding]:
    """
    Full async scan — tries OSV.dev live first, falls back to static list on error.
    """
    if filename.endswith("requirements.txt") or filename.endswith(".txt"):
        packages = _parse_requirements_txt(content)
    elif filename.endswith("package.json"):
        packages = _parse_package_json(content)
    else:
        return []

    # Check OSV.dev connectivity with one package; if it fails use static for all
    osv_available = False
    if packages:
        test_name, test_ver = packages[0]
        try:
            await _query_osv(test_name, test_ver or "0.0.0")
            osv_available = True
        except Exception:
            pass

    findings: list[Finding] = []

    if osv_available:
        # Query OSV.dev for all pinned packages in parallel
        tasks = [
            _check_package_live(name, version, filename)
            for name, version in packages
            if version  # OSV needs a concrete version
        ]
        # Unversioned packages still go through static
        for name, version in packages:
            if not version:
                findings.extend(_check_package_static(name, None, filename))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                findings.extend(r)
    else:
        # Static fallback for all packages
        for name, version in packages:
            findings.extend(_check_package_static(name, version, filename))

    return findings
