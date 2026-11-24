#!/usr/bin/env bash
# Installs the CPU test environment into .venv (ci runs this before the gate).
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-python3}"
[[ -x .venv/bin/python ]] || "$py" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"
.venv/bin/python -m pip install -q -r requirements-ci.txt
.venv/bin/python -m pip install -q --no-deps -e .
.venv/bin/python -c "import torch, peft, polars, numpy, PIL, pytest; print('setup ok: torch', torch.__version__, 'peft', peft.__version__, 'polars', polars.__version__)"
