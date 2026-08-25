#!/usr/bin/env bash
# One-line status for every paper project under sandbox/, read straight
# from each project's status.json (written by paper_agent.cli).
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("sandbox")
found = False
for status_file in sorted(root.glob("*/status.json")):
    found = True
    project = status_file.parent.name
    d = json.loads(status_file.read_text(encoding="utf-8"))
    state = d.get("state", "?")
    if state == "running":
        detail = d.get("progress") or d.get("topic", "")
    else:
        detail = d.get("final_message") or d.get("error") or d.get("topic", "")

    extra = ""
    if state == "running" and d.get("started_at"):
        started = datetime.fromisoformat(d["started_at"])
        elapsed_h = (datetime.now(timezone.utc) - started).total_seconds() / 3600
        budget = d.get("hours_budget")
        extra = f" ({elapsed_h:.1f}h elapsed" + (f" / {budget:.1f}h budget)" if budget else ")")

    print(f"[{state:>7}]{extra} {project}: {detail}")
if not found:
    print("No paper projects found under sandbox/.")
PY
