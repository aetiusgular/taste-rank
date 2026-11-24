#!/usr/bin/env bash
# Project tests and lint (ci). Red tests are not run here; scripts/forge-gate.sh
# runs them with their expectations. This runs the contract self-checks and
# every lane's own tests that exist.
set -euo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python
[[ -x "$py" ]] || py=python3
"$py" -m compileall -q src tests fixtures >/dev/null
dirs=(tests/contracts)
for d in tests/data tests/train tests/eval tests/integration; do
  [[ -d "$d" ]] && [[ -n "$(find "$d" -name 'test_*.py' -print -quit)" ]] && dirs+=("$d")
done
"$py" -m pytest -q "${dirs[@]}"
