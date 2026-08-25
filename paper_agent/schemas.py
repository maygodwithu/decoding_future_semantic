from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    protocol: str
    success_criteria: str
    notes: str | None = None


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ResultStatus
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    logs_summary: str
    artifacts: list[str] = Field(default_factory=list)
    run_id: str


class ReviewAction(str, Enum):
    RETRY = "RETRY"
    ITERATE = "ITERATE"
    ACCEPT = "ACCEPT"


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ReviewAction
    reasoning: str
    feedback_for_coder: str | None = None


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    markdown_body: str
