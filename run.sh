#!/bin/bash
cd "$(dirname "$0")"
# Use the project's virtual environment if there is one (scripts/setup_pi.sh creates .venv); otherwise the python3 on PATH
# (for example an already-activated environment).
PY=python3
for venv in "${VENV:-}" .venv vedic-atlas-env; do
  if [ -n "$venv" ] && [ -x "$venv/bin/python" ]; then PY="$venv/bin/python"; break; fi
done
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
PYTHONPATH=src exec "$PY" -m controller.cli "$@"
