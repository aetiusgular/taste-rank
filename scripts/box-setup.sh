#!/usr/bin/env bash
# Host setup for the P5 box run (Tony runs this once, as the user who owns
# /data/models; the runner then reads it through the models group).
#
#   scripts/box-setup.sh venv      the eval venv from requirements-box.txt
#   scripts/box-setup.sh models    the two pinned model folders, verified
#   scripts/box-setup.sh data      the competition files; needs the Kaggle
#                                  token in ~/.kaggle/kaggle.json and the
#                                  competition rules accepted on kaggle.com
#   scripts/box-setup.sh perms     group models, read-only for the group
#   scripts/box-setup.sh all       the four in order
#   scripts/box-setup.sh check     what scripts/eval.sh would refuse, if anything
#
# Layout under TASTE_RANK_ROOT (default /data/models/taste-rank):
#   venv/  models/marqo-fashionSigLIP/  models/fashion-clip/  hm/
# Nothing here writes into the repo. The model files come from the public Hub
# at the models.lock revisions with no token; the competition data never
# leaves the host (SPEC.md, "Data").
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="${TASTE_RANK_ROOT:-/data/models/taste-rank}"
competition=h-and-m-personalized-fashion-recommendations
step="${1:-}"

venv() {
  command -v uv >/dev/null || { echo "box-setup: uv is required" >&2; exit 2; }
  mkdir -p "$root"
  [[ -x "$root/venv/bin/python" ]] || uv venv -q -p 3.12 "$root/venv"
  uv pip install -q -p "$root/venv/bin/python" -r "$here/requirements-box.txt"
  "$root/venv/bin/python" -c "import torch, open_clip, transformers, peft, polars; print('venv ok: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

fetch_model() {  # <hub id> <folder>
  local id="$1" dir="$2" rev files f
  rev=$(jq -r --arg id "$id" '.models[] | select(.id == $id) | .revision' "$here/models.lock")
  mapfile -t files < <(jq -r --arg id "$id" '.models[] | select(.id == $id) | .files | keys[]' "$here/models.lock")
  [[ "$rev" =~ ^[0-9a-f]{40}$ && ${#files[@]} -gt 0 ]] || { echo "box-setup: $id is not pinned in models.lock" >&2; exit 1; }
  mkdir -p "$dir"
  for f in "${files[@]}"; do
    [[ -s "$dir/$f" ]] || curl -fL --retry 3 -o "$dir/$f.part" "https://huggingface.co/$id/resolve/$rev/$f"
    [[ -f "$dir/$f.part" ]] && mv "$dir/$f.part" "$dir/$f"
  done
  PYTHONPATH="$here/src" "$root/venv/bin/python" -m tasterank.models "$id" "$dir"
}

models() {
  [[ -x "$root/venv/bin/python" ]] || { echo "box-setup: run 'venv' first" >&2; exit 2; }
  fetch_model Marqo/marqo-fashionSigLIP "$root/models/marqo-fashionSigLIP"
  fetch_model patrickjohncyh/fashion-clip "$root/models/fashion-clip"
}

data() {
  [[ -f "$HOME/.kaggle/kaggle.json" ]] || { echo "box-setup: no ~/.kaggle/kaggle.json (Kaggle API token, mode 0600)" >&2; exit 2; }
  local zip="$root/download"
  mkdir -p "$zip" "$root/hm"
  [[ -s "$zip/$competition.zip" ]] || uvx --from kaggle==2.2.5 kaggle competitions download -c "$competition" -p "$zip"
  # Only the files the build reads (contracts/data.md); the sample submission is skipped.
  unzip -q -n "$zip/$competition.zip" 'transactions_train.csv' 'articles.csv' 'customers.csv' 'images/*' -d "$root/hm"
  echo "data: $(find "$root/hm/images" -name '*.jpg' | wc -l) images; remove $zip/$competition.zip when the run is done"
}

perms() {
  chgrp -R models "$root"
  chmod -R g+rX,g-w,o-rwx "$root"
  find "$root" -type d -exec chmod g+s {} +
}

check() {
  TASTE_RANK_ROOT="$root" PYTHONPATH="$here/src" "$root/venv/bin/python" -m tasterank.box --preflight-only --work "${TMPDIR:-/tmp}/taste-rank-check"
}

case "$step" in
  venv|models|data|perms|check) "$step" ;;
  all) venv; models; data; perms; check ;;
  *) echo "usage: scripts/box-setup.sh venv|models|data|perms|all|check" >&2; exit 2 ;;
esac
