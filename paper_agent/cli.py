from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .agents.coder import Coder
from .agents.planner import Planner, PlannerAPIError, PlannerOutputError
from .config import Settings, load_settings
from .orchestrator import CycleRecord, Orchestrator, OrchestratorResult, SandboxConfigError, StopReason

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 60) -> str:
    slug = _SLUG_STRIP_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len].rstrip("-") or "untitled"


def _write_status(project_dir: Path, **fields: object) -> None:
    """Updates <project_dir>/status.json so a later `--topic`-less check (this
    session or a fresh one) can answer "is it done" without needing to be told
    what's running — see status.sh."""
    project_dir.mkdir(parents=True, exist_ok=True)
    status_path = project_dir / "status.json"
    existing: dict = {}
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(fields)
    status_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper_agent")
    parser.add_argument("--topic", required=True, help="Research topic for the Planner to investigate")
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Folder name for this paper's work, created under SANDBOX_DIR. "
            "Defaults to a slug derived from --topic."
        ),
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help=(
            "Soft wall-clock time budget. Checked between cycles (never mid-experiment) "
            "so the current cycle always finishes; stops with TIME_BUDGET_REACHED and "
            "still writes a report. Unset means no time limit (only --project's cycle/"
            "retry caps apply)."
        ),
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Override CUDA_VISIBLE_DEVICES for this run only (e.g. '0'). Defaults to the .env setting.",
    )
    parser.add_argument(
        "--reuse-from",
        nargs="*",
        default=None,
        help=(
            "Absolute path(s) to prior projects' directories the Coder may read/write for "
            "reuse (e.g. a previous project's artifacts/). Without this, the Coder is "
            "sandboxed to its own run directory and cannot see other projects at all, even "
            "if --topic mentions their path."
        ),
    )
    return parser


def print_summary(result: OrchestratorResult) -> None:
    for cycle in result.cycles:
        print(f"\n=== Cycle {cycle.cycle_num} ===")
        print(f"Hypothesis: {cycle.plan.hypothesis}")
        for attempt in cycle.attempts:
            if attempt.result is not None:
                print(
                    f"  Attempt {attempt.attempt_num}: {attempt.result.status.value} "
                    f"metrics={attempt.result.metrics}"
                )
            else:
                print(f"  Attempt {attempt.attempt_num}: ERROR - {attempt.error}")
        if cycle.decision is not None:
            print(f"Review: {cycle.decision.decision.value} - {cycle.decision.reasoning}")

    print(f"\n=== Result: {result.stop_reason.value} ===")
    print(result.final_message)


def _serialize_cycles(cycles: list[CycleRecord]) -> list[dict]:
    """Full detail (unlike Orchestrator._history's compact summary) — the
    report should quote real protocols, metrics, and reasoning, not just a
    one-line-per-cycle digest."""
    serialized = []
    for cycle in cycles:
        serialized.append(
            {
                "cycle": cycle.cycle_num,
                "plan": cycle.plan.model_dump(mode="json"),
                "attempts": [
                    {
                        "attempt": a.attempt_num,
                        "result": a.result.model_dump(mode="json") if a.result else None,
                        "error": a.error,
                    }
                    for a in cycle.attempts
                ],
                "review": cycle.decision.model_dump(mode="json") if cycle.decision else None,
            }
        )
    return serialized


async def _write_report(settings: Settings, planner: Planner, args: argparse.Namespace, result: OrchestratorResult) -> None:
    if result.stop_reason == StopReason.FATAL_ERROR or not result.cycles:
        return
    try:
        report = await planner.write_report(
            args.topic,
            _serialize_cycles(result.cycles),
            result.stop_reason.value,
            result.final_message,
        )
    except (PlannerAPIError, PlannerOutputError) as exc:
        print(f"Report generation failed: {exc}", file=sys.stderr)
        return

    report_path = settings.sandbox_dir / "report.md"
    report_path.write_text(f"# {report.title}\n\n{report.markdown_body}\n", encoding="utf-8")
    print(f"\nReport saved to: {report_path}")
    _write_status(settings.sandbox_dir, report=str(report_path))


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    _write_status(
        settings.sandbox_dir,
        state="running",
        topic=args.topic,
        hours_budget=args.hours,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    planner = Planner(settings)
    orchestrator = Orchestrator(settings, planner, Coder(settings, reuse_dirs=args.reuse_from))

    def on_progress(message: str) -> None:
        print(message, flush=True)
        _write_status(
            settings.sandbox_dir,
            progress=message,
            progress_at=datetime.now(timezone.utc).isoformat(),
        )

    max_duration_seconds = args.hours * 3600 if args.hours is not None else None
    try:
        result = await orchestrator.run(
            args.topic, max_duration_seconds=max_duration_seconds, on_progress=on_progress
        )
    except SandboxConfigError as exc:
        _write_status(
            settings.sandbox_dir,
            state="error",
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"Sandbox error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        _write_status(
            settings.sandbox_dir,
            state="error",
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        raise

    _write_status(
        settings.sandbox_dir,
        state="done",
        stop_reason=result.stop_reason.value,
        final_message=result.final_message,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )

    print_summary(result)
    await _write_report(settings, planner, args, result)
    return 0 if result.stop_reason == StopReason.ACCEPTED else 1


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        settings = load_settings()
    except ValidationError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(2)

    project = args.project or slugify(args.topic)
    settings.sandbox_dir = settings.sandbox_dir / project
    print(f"Project folder: {settings.sandbox_dir}")

    if args.gpu is not None:
        settings.cuda_visible_devices = args.gpu
    print(f"CUDA_VISIBLE_DEVICES: {settings.cuda_visible_devices}")

    sys.exit(asyncio.run(_run(args, settings)))


if __name__ == "__main__":
    main()
