#!/usr/bin/env bash
# Attribution scan (BUILD_OUTLINE.md section 4, layer 4).
#
#   .github/scripts/attribution-scan.sh <git rev-range or commit>...
#
# Fails when any commit message in the given revisions, or the PR body in
# $PR_BODY, carries an AI-attribution trailer, footer or session link.
# Matching ignores case and also catches the markdown-link form of the
# "Generated with" footer and the session trailer.
set -uo pipefail
patterns=(
  'co-authored-by:[[:space:]]*claude'
  'noreply@anthropic\.com'
  'claude\.ai/code/session'
  'generated with \[?claude code'
  '^[[:space:]]*claude-session:'
)
grep_args=()
for p in "${patterns[@]}"; do grep_args+=(-e "$p"); done

hits=0
for rev in "$@"; do
  while IFS= read -r sha; do
    [[ -n "$sha" ]] || continue
    msg=$(git log -1 --format=%B "$sha")
    if found=$(printf '%s\n' "$msg" | grep -inE "${grep_args[@]}"); then
      echo "attribution in commit ${sha:0:12}:"
      printf '%s\n' "$found" | sed 's/^/  /'
      hits=$((hits + 1))
    fi
  done < <(git rev-list "$rev")
done
if [[ -n "${PR_BODY:-}" ]]; then
  if found=$(printf '%s\n' "$PR_BODY" | grep -inE "${grep_args[@]}"); then
    echo "attribution in the PR body:"
    printf '%s\n' "$found" | sed 's/^/  /'
    hits=$((hits + 1))
  fi
fi
if (( hits > 0 )); then
  echo "attribution scan: $hits place(s) to fix. Amend the commits or edit the PR body; see .claude/settings.json."
  exit 1
fi
echo "attribution scan: clean"
