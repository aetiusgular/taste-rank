#!/usr/bin/env bash
# forge/guard (BUILD_OUTLINE.md section 4). Runs from the base branch's copy
# (pull_request_target), never from the PR: it checks out no PR code and runs
# nothing from it.
#
#   .github/scripts/guard.sh check <next-phase>   paths on stdin; exit 0 or 1
#   .github/scripts/guard.sh next                 the phase a PR on this base opens
#   .github/scripts/guard.sh pr                   uses GH_TOKEN, REPO, PR_NUMBER,
#                                                 HEAD_SHA, RUN_URL; posts the status
#
# A PR may not touch .github/, scripts/forge-gate.sh, scripts/lane-scope.mjs,
# or, unless the phase it is opening is P2, scripts/lane-ownership.json.
# The phase it opens is the one after the base branch's state.json phase.
set -uo pipefail

check() {
  local next="$1" path bad=0
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    case "$path" in
      .github/*) echo "protected: $path"; bad=1 ;;
      scripts/forge-gate.sh|scripts/lane-scope.mjs) echo "protected: $path"; bad=1 ;;
      scripts/lane-ownership.json)
        if [[ "$next" != "P2" ]]; then echo "protected outside P2: $path (this PR opens $next)"; bad=1; fi ;;
    esac
  done
  return $bad
}

next_phase() {
  # Base branch's state.json, from the trusted checkout of the base.
  jq -r '
    if .phase == "P0" then .phases[0]
    else .phase as $p | (.phases | index($p)) as $i
      | if $i == null then "unknown" else (.phases[$i + 1] // "none") end
    end' .forge/state.json 2>/dev/null || echo unknown
}

status() {
  gh api "repos/$REPO/statuses/$HEAD_SHA" -f state="$1" -f context=forge/guard \
    -f description="$2" -f target_url="$RUN_URL" >/dev/null
}

case "${1:-}" in
  check) check "${2:?next phase}" ;;
  next) next_phase ;;
  pr)
    next=$(next_phase)
    changed=$(gh api "repos/$REPO/pulls/$PR_NUMBER" --jq .changed_files) || { status error "could not read the PR"; exit 1; }
    if (( changed > 3000 )); then
      status failure "more than 3000 files; the API cannot list them all"; exit 1
    fi
    paths=$(gh api --paginate "repos/$REPO/pulls/$PR_NUMBER/files" --jq '.[] | .filename, (.previous_filename // empty)') \
      || { status error "could not list the PR's files"; exit 1; }
    if out=$(printf '%s\n' "$paths" | check "$next"); then
      status success "no grader or workflow files touched (opens $next)"
      echo "forge/guard: clean (opens $next)"
    else
      printf '%s\n' "$out"
      status failure "touches protected files: $(printf '%s' "$out" | head -n1 | cut -c1-100)"
      exit 1
    fi
    ;;
  *) echo "usage: guard.sh check <next-phase> | guard.sh next | guard.sh pr" >&2; exit 2 ;;
esac
