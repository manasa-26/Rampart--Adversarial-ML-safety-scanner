"""
Live test: download prajjwal1/bert-tiny from HuggingFace and run the full
adversarial ML safety scan suite against its files.
"""

import sys
import asyncio
import os
from pathlib import Path

# ── Make project importable from this script ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from huggingface_hub import hf_hub_download, list_repo_files

from models.schemas import Severity
from core.scanner import secrets, pii, backdoor, code_injection, dependencies
from core.scanner.orchestrator import scan_content
from core.adversarial.attacks import run_attack
from models.schemas import AdversarialRequest, AttackType

console = Console()

MODEL_ID = "prajjwal1/bert-tiny"

SEVERITY_COLORS = {
    "critical": "bold red",
    "high":     "bold orange1",
    "medium":   "bold yellow",
    "low":      "bold cyan",
    "info":     "dim white",
}

SEVERITY_EMOJI = {
    "critical": "[C]",
    "high":     "[H]",
    "medium":   "[M]",
    "low":      "[L]",
    "info":     "[I]",
}


def sev(s: str) -> str:
    return f"{SEVERITY_EMOJI.get(s, '')} [{SEVERITY_COLORS.get(s, 'white')}]{s.upper()}[/]"


def print_scan_result(result, label: str):
    color = "red" if result.risk_score >= 60 else "yellow" if result.risk_score >= 20 else "green"
    passed = "✓ PASSED" if result.passed else "✗ FAILED"
    pass_color = "green" if result.passed else "red"

    console.print()
    console.rule(f"[bold]{label}[/]")
    console.print(
        f"  [{pass_color}]{passed}[/]  "
        f"Risk Score: [{color}]{result.risk_score}/100[/]  │  "
        f"🔴 {result.critical}  🟠 {result.high}  🟡 {result.medium}  🔵 {result.low}  ⚪ {result.info}"
    )

    if not result.findings:
        console.print("  [dim]No findings — clean.[/dim]")
        return

    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    tbl.add_column("Sev", width=12)
    tbl.add_column("Scanner", width=14)
    tbl.add_column("Finding", no_wrap=False)
    tbl.add_column("Evidence", no_wrap=False, max_width=50)

    for f in result.findings:
        s = f.severity.value
        tbl.add_row(
            sev(s),
            f"[dim]{f.scanner}[/dim]",
            f.title,
            f"[dim]{(f.evidence or '')[:80]}[/dim]",
        )
    console.print(tbl)


async def main():
    console.print(Panel.fit(
        f"[bold cyan]Adversarial ML Safety Guardrail[/bold cyan]\n"
        f"[dim]Live scan of HuggingFace model:[/dim] [bold white]{MODEL_ID}[/bold white]",
        border_style="cyan",
    ))

    # ── Step 1: List repo files ────────────────────────────────────────────
    console.print("\n[bold]Fetching repo file list…[/bold]")
    try:
        repo_files = list(list_repo_files(MODEL_ID))
    except Exception as e:
        console.print(f"[red]Could not list repo files: {e}[/red]")
        return
    console.print(f"  Found {len(repo_files)} files: {', '.join(repo_files)}")

    # ── Step 2: Download and scan each file ───────────────────────────────
    scan_targets = [f for f in repo_files if not f.startswith(".")]
    results_summary = []

    for fname in scan_targets:
        console.print(f"\n[dim]Downloading[/dim] [bold]{fname}[/bold]…", end=" ")
        try:
            local_path = hf_hub_download(repo_id=MODEL_ID, filename=fname)
            console.print("[green]done[/green]")
        except Exception as e:
            console.print(f"[red]failed: {e}[/red]")
            continue

        fpath = Path(local_path)
        suffix = fpath.suffix.lower()

        # Read as bytes; decode text files
        raw = fpath.read_bytes()
        is_binary = suffix in (".bin", ".pt", ".pth", ".onnx", ".h5", ".msgpack")

        if is_binary:
            content = raw
        else:
            content = raw.decode("utf-8", errors="ignore")

        result = await scan_content(content, fname)
        print_scan_result(result, fname)
        results_summary.append((fname, result))

    # ── Step 3: Aggregate summary ─────────────────────────────────────────
    console.print()
    console.rule("[bold]AGGREGATE SCAN SUMMARY[/bold]")

    summary_tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    summary_tbl.add_column("File", style="bold white")
    summary_tbl.add_column("Risk", justify="center")
    summary_tbl.add_column("🔴", justify="center")
    summary_tbl.add_column("🟠", justify="center")
    summary_tbl.add_column("🟡", justify="center")
    summary_tbl.add_column("Result", justify="center")

    total_critical = total_high = 0
    for fname, r in results_summary:
        total_critical += r.critical
        total_high += r.high
        color = "red" if r.risk_score >= 60 else "yellow" if r.risk_score >= 20 else "green"
        summary_tbl.add_row(
            fname,
            f"[{color}]{r.risk_score}[/]",
            str(r.critical) if r.critical else "[dim]0[/dim]",
            str(r.high) if r.high else "[dim]0[/dim]",
            str(r.medium) if r.medium else "[dim]0[/dim]",
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
        )

    console.print(summary_tbl)
    console.print(
        f"\n  Total critical: [red]{total_critical}[/red]  "
        f"Total high: [orange1]{total_high}[/orange1]"
    )

    # ── Step 4: Adversarial attack demo on a numeric vector ───────────────
    console.print()
    console.rule("[bold]ADVERSARIAL ROBUSTNESS TEST[/bold]")
    console.print("[dim]Running FGSM + PGD against built-in 2-layer MLP (ε=0.1)[/dim]\n")

    sample_input = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05,
                    0.15, 0.85, 0.45, 0.55, 0.25, 0.75, 0.35, 0.65, 0.95, 0.05]

    for attack_type, steps in [(AttackType.FGSM, None), (AttackType.PGD, 40)]:
        req = AdversarialRequest(
            attack_type=attack_type,
            input_array=sample_input,
            epsilon=0.1,
            pgd_steps=steps or 40,
        )
        try:
            r = run_attack(req)
            status = "[red]SUCCEEDED (model fooled)[/red]" if r.success else "[green]DEFENDED[/green]"
            console.print(
                f"  {attack_type.value:<12} ε=0.1  │  "
                f"Conf drop: [yellow]{r.confidence_drop:.4f}[/yellow]  │  "
                f"Perturbation ‖δ‖∞: [cyan]{r.perturbation_norm:.4f}[/cyan]  │  {status}"
            )
        except Exception as e:
            console.print(f"  {attack_type.value}: [red]{e}[/red]")

    # TextFooler demo
    text_req = AdversarialRequest(
        attack_type=AttackType.TEXTFOOLER,
        text="The model performs well on all standard benchmarks and is robust to distribution shift.",
        epsilon=0.3,
    )
    tr = run_attack(text_req)
    console.print(f"\n  [bold]TextFooler[/bold] ε=0.3")
    console.print(f"  Original : [white]{''.join(tr.original_input)}[/white]")
    console.print(f"  Adversarial: [yellow]{''.join(tr.adversarial_input)}[/yellow]")
    console.print(
        f"  Conf drop: [yellow]{tr.confidence_drop:.4f}[/yellow]  │  "
        f"Char-level norm: [cyan]{tr.perturbation_norm:.4f}[/cyan]  │  "
        + ("[red]SUCCEEDED[/red]" if tr.success else "[green]DEFENDED[/green]")
    )

    console.print()
    console.print(Panel.fit(
        "[bold green]Scan complete.[/bold green]\n"
        "Start the API server with [bold]python main.py[/bold] "
        "and open [bold]http://localhost:8000[/bold] for the interactive dashboard.",
        border_style="green",
    ))


if __name__ == "__main__":
    asyncio.run(main())
