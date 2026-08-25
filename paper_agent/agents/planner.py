from __future__ import annotations

import json

from openai import AsyncOpenAI

from ..config import Settings
from ..schemas import ExperimentPlan, ExperimentResult, Report, ReviewDecision

PLAN_SYSTEM_PROMPT = (
    "You are the research Planner for an autonomous experimentation system. "
    "You design one concrete experiment at a time for a Coder agent to implement. "
    "The Coder has file and bash access restricted to a sandbox directory and can "
    "install packages and run code, but cannot access anything outside that directory. "
    "Your `protocol` must be concrete enough for the Coder to implement directly: state "
    "what to write, what to measure, and what to save as an artifact. Your "
    "`success_criteria` must be objectively checkable from the Coder's reported metrics. "
    "Review the prior-cycle history below (if any) so you don't repeat a failed approach."
)

REPORT_SYSTEM_PROMPT = (
    "You are the research Planner writing a plain-language report of an autonomous "
    "experimentation run, for a researcher who will later turn it into a paper by hand. "
    "Not a LaTeX paper — a clear Markdown report. Cover: the research question/hypothesis, "
    "what was tried across each cycle (protocol, and why it changed if it did), the results "
    "and metrics observed, whether the success criteria were ultimately met, and honest "
    "limitations or follow-up questions. Use Markdown headings and be specific about actual "
    "numbers from the results — do not invent data not present in the provided history. "
    "`title` is reported separately — do not repeat it as a top-level heading inside "
    "`markdown_body`; start `markdown_body` directly with its first section (e.g. "
    "`## Research Question`)."
)

REVIEW_SYSTEM_PROMPT = (
    "You are the research Planner reviewing an experiment result. Decide exactly one of: "
    "RETRY (the Coder made an execution/implementation mistake; the same hypothesis is "
    "still worth attempting again), ITERATE (the experiment ran as intended but the "
    "hypothesis or protocol itself needs revision), or ACCEPT (the success_criteria was "
    "met and no further work is needed). Always give your reasoning. If you choose RETRY, "
    "you must give concrete feedback_for_coder describing exactly what to fix."
)


class PlannerAPIError(Exception):
    """Raised when the OpenAI API call itself fails."""


class PlannerOutputError(Exception):
    """Raised when the Planner's response cannot be parsed into the expected schema."""


class Planner:
    def __init__(self, settings: Settings) -> None:
        self._model = settings.planner_model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def propose_plan(self, topic: str, history: list[dict]) -> ExperimentPlan:
        user_content = (
            f"Research topic: {topic}\n\n"
            f"Prior cycles (most recent last): {json.dumps(history, ensure_ascii=False)}\n\n"
            "Propose the next experiment."
        )
        try:
            response = await self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                text_format=ExperimentPlan,
            )
        except Exception as exc:
            raise PlannerAPIError(f"OpenAI API call failed during PLAN: {exc}") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise PlannerOutputError("Planner returned no parsed ExperimentPlan (possible refusal)")
        return parsed

    async def review_result(
        self, plan: ExperimentPlan, result: ExperimentResult, cycle_num: int
    ) -> ReviewDecision:
        user_content = (
            f"Cycle {cycle_num} plan: {plan.model_dump_json()}\n\n"
            f"Experiment result: {result.model_dump_json()}\n\n"
            "Decide RETRY, ITERATE, or ACCEPT."
        )
        try:
            response = await self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                text_format=ReviewDecision,
            )
        except Exception as exc:
            raise PlannerAPIError(f"OpenAI API call failed during REVIEW: {exc}") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise PlannerOutputError("Planner returned no parsed ReviewDecision (possible refusal)")
        return parsed

    async def write_report(
        self, topic: str, cycles: list[dict], stop_reason: str, final_message: str
    ) -> Report:
        user_content = (
            f"Research topic: {topic}\n\n"
            f"Full cycle history: {json.dumps(cycles, ensure_ascii=False)}\n\n"
            f"Run outcome: {stop_reason} - {final_message}\n\n"
            "Write the report now."
        )
        try:
            response = await self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                text_format=Report,
            )
        except Exception as exc:
            raise PlannerAPIError(f"OpenAI API call failed during REPORT: {exc}") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise PlannerOutputError("Planner returned no parsed Report (possible refusal)")
        return parsed
