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
#   scripts/eval.sh                   the box pipeline; P5 wires it. Until
#                                     then it prints "not an eval phase".
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
    echo "not an eval phase"
    exit 0
    ;;
  *)
    echo "usage: scripts/eval.sh [--fixture <dir>]" >&2
    exit 2
    ;;
esac
