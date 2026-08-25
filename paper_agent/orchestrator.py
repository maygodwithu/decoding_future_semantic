from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .agents.coder import Coder, CoderExecutionError, CoderOutputError
from .agents.planner import Planner, PlannerAPIError, PlannerOutputError
from .config import Settings
from .schemas import ExperimentPlan, ExperimentResult, ReviewAction, ReviewDecision


class State(str, Enum):
    PLAN = "PLAN"
    CODE_EXEC = "CODE_EXEC"
    REVIEW = "REVIEW"


class StopReason(str, Enum):
    ACCEPTED = "ACCEPTED"
    MAX_CYCLES_REACHED = "MAX_CYCLES_REACHED"
    MAX_RETRIES_EXHAUSTED = "MAX_RETRIES_EXHAUSTED"
    TIME_BUDGET_REACHED = "TIME_BUDGET_REACHED"
    FATAL_ERROR = "FATAL_ERROR"


class SandboxConfigError(Exception):
    """Raised when the sandbox directory cannot be created or written to."""


@dataclass
class AttemptRecord:
    attempt_num: int
    result: ExperimentResult | None
    error: str | None


@dataclass
class CycleRecord:
    cycle_num: int
    plan: ExperimentPlan
    attempts: list[AttemptRecord] = field(default_factory=list)
    decision: ReviewDecision | None = None


@dataclass
class OrchestratorResult:
    stop_reason: StopReason
    cycles: list[CycleRecord]
    final_message: str


class Orchestrator:
    def __init__(self, settings: Settings, planner: Planner, coder: Coder) -> None:
        self._settings = settings
        self._planner = planner
        self._coder = coder

    async def run(
        self,
        topic: str,
        max_duration_seconds: float | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> OrchestratorResult:
        self._validate_sandbox()
        progress = on_progress or (lambda _msg: None)

        start_time = time.monotonic()
        cycles: list[CycleRecord] = []
        cycle_num = 0
        current_cycle: CycleRecord | None = None
        current_plan: ExperimentPlan | None = None
        feedback: str | None = None
        attempt_in_cycle = 0
        last_result: ExperimentResult | None = None
        state = State.PLAN

        while True:
            if state == State.PLAN:
                cycle_num += 1
                # Checked only between cycles (never mid-experiment) and only
                # once cycle 1 has actually run, so a tiny --hours budget still
                # gets at least one full attempt instead of doing nothing.
                elapsed = time.monotonic() - start_time
                if cycle_num > 1 and max_duration_seconds is not None and elapsed > max_duration_seconds:
                    return OrchestratorResult(
                        StopReason.TIME_BUDGET_REACHED,
                        cycles,
                        f"Stopped: time budget of {max_duration_seconds / 3600:.2f}h reached "
                        f"after finishing cycle {cycle_num - 1} (elapsed {elapsed / 3600:.2f}h). "
                        "Not a failure — review the report and re-run (a fresh run, cycles "
                        "restart from 1) if more work or a different direction is wanted.",
                    )
                if cycle_num > self._settings.max_cycles:
                    return OrchestratorResult(
                        StopReason.MAX_CYCLES_REACHED,
                        cycles,
                        f"Stopped: reached max_cycles={self._settings.max_cycles} "
                        "without the Planner issuing ACCEPT.",
                    )
                progress(f"Cycle {cycle_num}: Planner is proposing an experiment...")
                try:
                    current_plan = await self._planner.propose_plan(topic, self._history(cycles))
                except (PlannerAPIError, PlannerOutputError) as exc:
                    return OrchestratorResult(
                        StopReason.FATAL_ERROR,
                        cycles,
                        f"Stopped: Planner failed during PLAN (cycle {cycle_num}): {exc}",
                    )
                current_cycle = CycleRecord(cycle_num=cycle_num, plan=current_plan)
                cycles.append(current_cycle)
                attempt_in_cycle = 0
                feedback = None
                progress(f"Cycle {cycle_num}: {current_plan.hypothesis}")
                state = State.CODE_EXEC

            elif state == State.CODE_EXEC:
                assert current_plan is not None and current_cycle is not None
                attempt_in_cycle += 1
                if attempt_in_cycle > self._settings.max_retries:
                    return OrchestratorResult(
                        StopReason.MAX_RETRIES_EXHAUSTED,
                        cycles,
                        f"Stopped: exhausted max_retries={self._settings.max_retries} "
                        f"in cycle {cycle_num} without a valid result.",
                    )
                run_id = f"cycle{cycle_num}-attempt{attempt_in_cycle}"
                progress(f"Cycle {cycle_num} attempt {attempt_in_cycle}: Coder is running the experiment...")
                try:
                    last_result = await self._coder.run_experiment(current_plan, run_id, feedback)
                    current_cycle.attempts.append(AttemptRecord(attempt_in_cycle, last_result, None))
                    progress(
                        f"Cycle {cycle_num} attempt {attempt_in_cycle}: "
                        f"{last_result.status.value} - {last_result.logs_summary}"
                    )
                    state = State.REVIEW
                except (CoderExecutionError, CoderOutputError) as exc:
                    current_cycle.attempts.append(AttemptRecord(attempt_in_cycle, None, str(exc)))
                    feedback = f"Previous attempt failed: {exc}. Fix this and try again."
                    progress(f"Cycle {cycle_num} attempt {attempt_in_cycle}: ERROR - {exc}")
                    # Stays in CODE_EXEC; the cap is re-checked at the top of this branch.

            elif state == State.REVIEW:
                assert current_plan is not None and current_cycle is not None
                assert last_result is not None
                try:
                    decision = await self._planner.review_result(current_plan, last_result, cycle_num)
                except (PlannerAPIError, PlannerOutputError) as exc:
                    return OrchestratorResult(
                        StopReason.FATAL_ERROR,
                        cycles,
                        f"Stopped: Planner failed during REVIEW (cycle {cycle_num}): {exc}",
                    )
                current_cycle.decision = decision
                progress(f"Cycle {cycle_num} review: {decision.decision.value} - {decision.reasoning}")

                if decision.decision == ReviewAction.ACCEPT:
                    return OrchestratorResult(
                        StopReason.ACCEPTED,
                        cycles,
                        f"Accepted after cycle {cycle_num}: {decision.reasoning}",
                    )
                elif decision.decision == ReviewAction.RETRY:
                    if attempt_in_cycle >= self._settings.max_retries:
                        return OrchestratorResult(
                            StopReason.MAX_RETRIES_EXHAUSTED,
                            cycles,
                            "Stopped: Planner requested RETRY but max_retries="
                            f"{self._settings.max_retries} already used in cycle {cycle_num}.",
                        )
                    feedback = (
                        decision.feedback_for_coder
                        or "Planner requested a retry; no specific feedback given."
                    )
                    state = State.CODE_EXEC
                else:  # ITERATE
                    state = State.PLAN

    def _validate_sandbox(self) -> None:
        try:
            self._settings.sandbox_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SandboxConfigError(
                f"Cannot create/write sandbox_dir={self._settings.sandbox_dir}: {exc}"
            ) from exc

    @staticmethod
    def _history(cycles: list[CycleRecord]) -> list[dict]:
        history = []
        for cycle in cycles:
            last_attempt = cycle.attempts[-1] if cycle.attempts else None
            history.append(
                {
                    "cycle": cycle.cycle_num,
                    "hypothesis": cycle.plan.hypothesis,
                    "result_status": (
                        last_attempt.result.status.value
                        if last_attempt and last_attempt.result
                        else None
                    ),
                    "review_decision": cycle.decision.decision.value if cycle.decision else None,
                    "review_reasoning": cycle.decision.reasoning if cycle.decision else None,
                }
            )
        return history
