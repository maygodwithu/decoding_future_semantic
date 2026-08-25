#!/usr/bin/env bash
# Launches the CLI inside a detached tmux session so it keeps running
# after this shell (or the SSH connection) closes. Args pass straight
# through to `python -m paper_agent.cli`, e.g.:
#   ./run_paper.sh --topic "..." [--project some-name]
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

SESSION="paper-$(date +%Y%m%d-%H%M%S)"
tmux new-session -d -s "$SESSION" \
  ".venv/bin/python -u -m paper_agent.cli $(printf '%q ' "$@") 2>&1 | tee \"logs/${SESSION}.log\""

echo "Started in tmux session: $SESSION"
echo "Attach:  tmux attach -t $SESSION"
echo "Detach again without stopping it: Ctrl-b d"
echo "Log file: logs/${SESSION}.log"
