from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query

from ..conda_env import CondaEnvError, ensure_env
from ..config import Settings
from ..schemas import ExperimentPlan, ExperimentResult


class CoderExecutionError(Exception):
    """Raised when the Claude Agent SDK query() call itself fails."""


class CoderOutputError(Exception):
    """Raised when the Coder did not produce a valid result file."""


class Coder:
    def __init__(self, settings: Settings, reuse_dirs: list[str] | None = None) -> None:
        self._settings = settings
        # Extra directories (e.g. a previous project's artifacts) the Coder
        # may read/reuse from, despite `cwd` normally sandboxing it to its
        # own run directory. Pass explicitly — the SDK does not grant access
        # to a path just because it's mentioned in the prompt.
        self._reuse_dirs = reuse_dirs or []

    async def run_experiment(
        self, plan: ExperimentPlan, run_id: str, feedback: str | None = None
    ) -> ExperimentResult:
        run_dir = self._settings.sandbox_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "result.json"

        # One conda env per paper project (not per attempt) — created once,
        # reused across every cycle/retry of this project so dependencies
        # stay isolated from other papers without recreating the env every
        # single attempt.
        env_name = f"paper-{self._settings.sandbox_dir.name}"
        try:
            conda_env_vars = ensure_env(
                self._settings.conda_root, env_name, self._settings.conda_python_version
            )
        except CondaEnvError as exc:
            raise CoderExecutionError(f"conda env setup failed for '{env_name}': {exc}") from exc

        prompt = self._build_prompt(plan, result_path, feedback, env_name, self._reuse_dirs)

        options = ClaudeAgentOptions(
            cwd=str(run_dir),
            add_dirs=self._reuse_dirs,
            model=self._settings.coder_model,
            permission_mode="bypassPermissions",
            allowed_tools=["Bash", "Read", "Write", "Edit"],
            # query() spawns the Claude Code binary as a subprocess; it does not
            # inherit our pydantic-settings-loaded key unless passed explicitly here.
            # PATH from conda_env_vars puts this project's conda env first, so
            # `python`/`pip` resolve inside it without an explicit `conda activate`.
            env={
                "ANTHROPIC_API_KEY": self._settings.anthropic_api_key,
                "CUDA_VISIBLE_DEVICES": self._settings.cuda_visible_devices,
                **conda_env_vars,
            },
        )

        # Keep only a bounded tail of recent SDK messages — not full transcript
        # logging, just enough to diagnose a failure. Written to disk only if
        # the run actually fails, so successful runs stay uncluttered.
        recent_messages: list[object] = []
        try:
            async for message in query(prompt=prompt, options=options):
                recent_messages.append(message)
                if len(recent_messages) > 5:
                    recent_messages.pop(0)
        except Exception as exc:
            self._dump_tail(run_dir, recent_messages)
            raise CoderExecutionError(f"query() failed for run {run_id}: {exc}") from exc

        if not result_path.exists():
            self._dump_tail(run_dir, recent_messages)
            raise CoderOutputError(
                f"Coder did not write a result file at {result_path} "
                f"(see _last_messages.txt in that directory for the last SDK messages)"
            )

        raw = result_path.read_text(encoding="utf-8")
        try:
            return ExperimentResult.model_validate_json(raw)
        except ValueError as exc:
            raise CoderOutputError(
                f"Coder result JSON at {result_path} failed validation: {exc}"
            ) from exc

    @staticmethod
    def _dump_tail(run_dir: Path, messages: list[object]) -> None:
        if not messages:
            return
        text = "\n\n---\n\n".join(repr(m)[:2000] for m in messages)
        (run_dir / "_last_messages.txt").write_text(text, encoding="utf-8")

    @staticmethod
    def _build_prompt(
        plan: ExperimentPlan,
        result_path: Path,
        feedback: str | None,
        env_name: str,
        reuse_dirs: list[str],
    ) -> str:
        schema = ExperimentResult.model_json_schema()
        feedback_block = f"\nFeedback from the previous attempt: {feedback}\n" if feedback else ""
        reuse_block = ""
        if reuse_dirs:
            listed = "\n".join(f"- {d}" for d in reuse_dirs)
            reuse_block = (
                "\nYou also have read/write access to these additional directories from prior "
                f"work, for reuse — do not recreate what already exists there:\n{listed}\n"
            )
        return (
            "You are the Coder agent for an autonomous research experiment. "
            "Your working directory is the sandbox root for this run — stay inside it "
            f"unless using one of the additional directories listed below.\n"
            f"{reuse_block}\n"
            f"You have a dedicated conda environment for this paper project, '{env_name}', "
            "already active on your PATH — `python` and `pip` resolve inside it. Install any "
            "packages you need with `pip install ...`; it only affects this project's "
            "environment, not the system Python or other projects.\n\n"
            f"Hypothesis: {plan.hypothesis}\n"
            f"Protocol: {plan.protocol}\n"
            f"Success criteria: {plan.success_criteria}\n"
            f"{feedback_block}\n"
            "Implement and run the experiment described above. When finished, write your "
            f"result as a single JSON file at exactly this path: {result_path}\n\n"
            f"The file must validate against this JSON schema:\n{schema}\n\n"
            "Writing to stdout alone does not satisfy this task — the run is only "
            "complete once the JSON file exists at that exact path."
        )
