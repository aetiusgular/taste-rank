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
# P5 replaces this stub.
echo "not an eval phase"
exit 0
