from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    anthropic_api_key: str

    # Root under which each paper gets its own subfolder (named by --project,
    # or a slug of --topic — see cli.py). Never hardcode this — it must
    # resolve correctly on both a local Windows dev machine and a Linux GPU
    # server, so it is env-driven only.
    sandbox_dir: Path = Field(default=Path("./sandbox"))

    planner_model: str = "gpt-5.4"
    coder_model: str = "claude-sonnet-5"

    # Shared GPU server: restrict the Coder's experiment code to these devices
    # only, so it never touches GPUs other users are actively using.
    cuda_visible_devices: str = "0,1"

    # One conda env per paper project (see cli.py) keeps each paper's
    # dependencies isolated from every other paper's. Never hardcode a
    # specific username's path — default to the current user's home dir.
    conda_root: Path = Field(default=Path.home() / "miniconda3")
    conda_python_version: str = "3.11"

    max_retries: int = Field(default=3, ge=1)
    max_cycles: int = Field(default=5, ge=1)


def load_settings() -> Settings:
    settings = Settings()
    settings.sandbox_dir = settings.sandbox_dir.resolve()
    return settings
