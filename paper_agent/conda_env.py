from __future__ import annotations

import os
import subprocess
from pathlib import Path


class CondaEnvError(Exception):
    """Raised when creating or resolving a conda environment fails."""


def ensure_env(conda_root: Path, env_name: str, python_version: str) -> dict[str, str]:
    """Creates the conda env if it doesn't already exist, and returns the
    environment variable overrides (PATH prefixed with the env's bin dir,
    plus CONDA_PREFIX/CONDA_DEFAULT_ENV) needed for subprocesses to resolve
    python/pip inside it without an explicit `conda activate`."""
    conda_bin = conda_root / "bin" / "conda"
    env_path = conda_root / "envs" / env_name

    if not conda_bin.exists():
        raise CondaEnvError(f"conda binary not found at {conda_bin}")

    if not env_path.exists():
        result = subprocess.run(
            [str(conda_bin), "create", "-y", "-n", env_name, f"python={python_version}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise CondaEnvError(f"conda create failed for env '{env_name}': {result.stderr.strip()}")

    if not env_path.exists():
        raise CondaEnvError(f"conda env '{env_name}' still missing after create at {env_path}")

    env_bin = env_path / "bin"
    return {
        "PATH": f"{env_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "CONDA_PREFIX": str(env_path),
        "CONDA_DEFAULT_ENV": env_name,
    }
