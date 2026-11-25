#!/usr/bin/env bash
# The single entry point the box runs (BUILD_OUTLINE.md sections 6 and 8).
# forge's eval workflow runs it inside srt with the network limited to
# 127.0.0.1, from the repo root, with FORGE_REPO, FORGE_PR, FORGE_SHA,
# FORGE_PHASE and FORGE_MACHINE_FILE (machine labels read on the host at run
# time) in the environment. It writes results/*.jsonl rows (results/README.md)
# and may write results/summary.md, which becomes the PR comment.
# Projects with eval_boundary landlock: forge may run some eval columns
# outside landrun, each with its own boundary (forge's queue.json, not this
# repo). The landrun pass gets FORGE_SKIP_COLUMNS (comma list) and must skip
# them; each such column then runs this script again with FORGE_COLUMN set
# to its name (srt: inside derm run --boundary srt) and runs only that one.
#
#   scripts/eval.sh --fixture <dir>   POP and E0-tiny on fixtures/shop_like,
#                                     dev split only; rows go to <dir>/*.jsonl
#   scripts/eval.sh                   with FORGE_PHASE=P5 (the box) or
#                                     TASTE_RANK_BOX=1 (a host run): the box
#                                     pipeline, python -m tasterank.box. Rows
#                                     go to results/box-<run_id>.jsonl, the
#                                     table to results/summary.md. Exit 3 and
#                                     a list when the box is not provisioned
#                                     (scripts/box-setup.sh). Otherwise it
#                                     prints "not an eval phase".
#
# Box layout (TASTE_RANK_ROOT, default /data/models/taste-rank, readable by
# the runner through the models group): venv/ (requirements-box.txt),
# models/marqo-fashionSigLIP/, models/fashion-clip/, hm/ (the competition
# files). Caches and LoRA weights go under $FORGE_TMP/taste-rank, never the repo.
set -euo pipefail
# --fixture <dir> is relative to the caller's directory, so resolve it before the cd.
if [[ "${1:-}" == --fixture && $# -eq 2 && -n "$2" && "$2" != /* ]]; then
  set -- "$1" "$PWD/$2"
fi
cd "$(dirname "$0")/.."
py=.venv/bin/python
[[ -x "$py" ]] || py=python3

case "${1:-}" in
  --fixture)
    [[ $# -eq 2 && -n "$2" ]] || { echo "usage: scripts/eval.sh --fixture <dir>" >&2; exit 2; }
    exec "$py" -m tasterank.eval.run --fixture "$2"
    ;;
  "")
    if [[ "${FORGE_PHASE:-}" != P5 && "${TASTE_RANK_BOX:-}" != 1 ]]; then
      echo "not an eval phase"
      exit 0
    fi
    root="${TASTE_RANK_ROOT:-/data/models/taste-rank}"
    [[ -x "$root/venv/bin/python" ]] && py="$root/venv/bin/python"
    tmp="${FORGE_TMP:-}"
    [[ -n "$tmp" ]] || { echo "eval: FORGE_TMP is not set (the box workflow sets it)" >&2; exit 3; }
    mkdir -p "$tmp/taste-rank/cache/xdg"
    export PYTHONPATH="$PWD/src" TASTE_RANK_ROOT="$root" \
      XDG_CACHE_HOME="$tmp/taste-rank/cache/xdg" HF_HOME="$tmp/taste-rank/cache/hf" TORCH_HOME="$tmp/taste-rank/cache/torch" \
      TRITON_CACHE_DIR="$tmp/taste-rank/cache/triton" MPLCONFIGDIR="$tmp/taste-rank/cache/mpl" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
    exec "$py" -m tasterank.box --work "$tmp/taste-rank"
    ;;
  *)
    echo "usage: scripts/eval.sh [--fixture <dir>]" >&2
    exit 2
    ;;
esac
