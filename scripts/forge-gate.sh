#!/usr/bin/env bash
# Mechanical phase gate (BUILD_OUTLINE.md sections 3, 4, 6 and 7).
#
#   scripts/forge-gate.sh P<k>
#
# Checks what a script can check for phase P<k>; the review session checks
# the rest. ci runs it on every PR and on every push to main. A grader file:
# no phase edits it (forge/guard enforces that).
#
# Environment:
#   FORGE_BASE   base for lane-scope in P3 phases (default origin/main)
#
# Exit 0 pass, 1 gate failed, 2 usage error.
set -uo pipefail

phase="${1:-}"
[[ "$phase" =~ ^P[0-9]+[a-z]?$ ]] || { echo "usage: scripts/forge-gate.sh P<k>" >&2; exit 2; }
cd "$(dirname "$0")/.." || exit 2
command -v jq >/dev/null || { echo "forge-gate: jq is required" >&2; exit 2; }

fails=0
fail() { echo "FAIL $*"; fails=$((fails + 1)); }
pass() { echo "ok   $*"; }
nonempty() { [[ -s "$1" ]]; }
# Files in a folder other than its README.md.
has_files() { [[ -d "$1" ]] && [[ -n "$(find "$1" -type f ! -name README.md -print -quit)" ]]; }

state=.forge/state.json

# ---------------------------------------------------------------- every phase
if ! jq -e 'type == "object"' "$state" >/dev/null 2>&1; then
  fail "$state is missing or not a JSON object"
  echo "forge-gate: $phase FAILED (state unreadable)"; exit 1
fi
missing=$(jq -r '["project","code","target","phase","round","phases","human_gates","session_budget","needs_box","closed"] - keys | join(", ")' "$state")
[[ -z "$missing" ]] && pass "state.json has every field" || fail "state.json lacks: $missing"
jq -e '(.phases | type == "array") and (.human_gates | type == "array") and (.closed | type == "boolean") and (.round | type == "number")' "$state" >/dev/null \
  && pass "state.json field types" || fail "state.json: phases and human_gates must be arrays, closed a boolean, round a number"
state_phase=$(jq -r '.phase' "$state")
[[ "$state_phase" == "$phase" ]] && pass "state.json phase is $phase" || fail "state.json phase is $state_phase, gate asked for $phase"
if [[ "$phase" != "P0" ]]; then
  jq -e --arg p "$phase" '.phases | index($p) != null' "$state" >/dev/null \
    && pass "$phase is in state.json phases" || fail "$phase is not in state.json phases"
fi
# The box sandbox profile eval.yml uses for this project (forge DECISIONS.md).
jq -e '(.eval_boundary // "srt") | IN("srt", "srt-gpu", "landlock")' "$state" >/dev/null \
  && pass "eval_boundary is srt, srt-gpu or landlock" || fail "state.json eval_boundary must be srt, srt-gpu or landlock"
# Section 4: P6 and P7 are always human gates.
jq -e '(.human_gates | index("P6") != null) and (.human_gates | index("P7") != null)' "$state" >/dev/null \
  && pass "human_gates keep P6 and P7" || fail "human_gates must keep P6 and P7"
# Section 4 circuit breaker: a phase past twice its session budget stops.
budget=$(jq -r --arg p "$phase" '.session_budget[$p] // (if ($p | test("^P3")) then .session_budget["P3*"] else null end) // empty' "$state")
used=$(jq -r --arg p "$phase" '.sessions_used[$p] // 0' "$state")
if [[ -n "$budget" ]] && (( used > 2 * budget )); then
  fail "$phase used $used sessions, over twice its budget of $budget: stop and label needs-tony"
else
  pass "$phase sessions within twice the budget"
fi
# Section 4, attribution layer 1.
jq -e '.attribution == {"commit": "", "pr": "", "sessionUrl": false}' .claude/settings.json >/dev/null 2>&1 \
  && pass ".claude/settings.json keeps attribution off" || fail ".claude/settings.json must hold attribution {commit: \"\", pr: \"\", sessionUrl: false}"
for f in DECISIONS.md SECURITY.md LICENSE; do
  [[ -f "$f" ]] && pass "$f present" || fail "$f missing"
done
for d in research contracts fixtures tests/red results reviews; do
  [[ -d "$d" ]] && pass "$d/ present" || fail "$d/ missing"
done
if jq -e '(.lanes | type == "object") and (.noLane | type == "array")' scripts/lane-ownership.json >/dev/null 2>&1; then
  pass "lane-ownership.json readable"
else
  fail "scripts/lane-ownership.json is missing lanes or noLane"
fi

# P3 lanes in phase order, from state.json.
mapfile -t lanes < <(jq -r '.phases[] | select(test("^P3"))' "$state")
order_of() { local i; for i in "${!lanes[@]}"; do [[ "${lanes[$i]}" == "$1" ]] && { echo "$i"; return; }; done; echo -1; }
phase_index() { jq -r --arg p "$1" '.phases | index($p) // -1' "$state"; }
cur=$(phase_index "$phase")
p2=$(phase_index P2)
after_p2=0
[[ "$p2" -ge 0 && "$cur" -ge "$p2" ]] && after_p2=1

# ------------------------------------------------------------- red tests
# tests/red/manifest.json (format in tests/red/README.md). want: red = every
# test fails for its stated reason; green = every test passes; lane = tests
# of lanes up to and including $phase pass, the rest stay red.
run_red() {
  local want="$1" manifest=tests/red/manifest.json
  if ! jq -e '.tests | type == "array" and length > 0' "$manifest" >/dev/null 2>&1; then
    fail "$manifest missing or has no tests"; return
  fi
  local bad
  bad=$(jq -r '.tests[] | select((.id|type)!="string" or (.lane|type)!="string" or (.case|type)!="string" or (.cmd|type)!="string" or (.expect|type)!="string") | (.id // "?")' "$manifest")
  [[ -z "$bad" ]] || { fail "red tests missing id, lane, case, cmd or expect: $bad"; return; }
  local dups
  dups=$(jq -r '[.tests[].id] | group_by(.) | map(select(length > 1) | .[0]) | join(", ")' "$manifest")
  [[ -z "$dups" ]] || fail "red test ids repeat: $dups"
  local mine
  mine=$(order_of "$phase")
  local n
  n=$(jq '.tests | length' "$manifest")
  local i
  for ((i = 0; i < n; i++)); do
    local id lane cmd expect timeout_s
    id=$(jq -r ".tests[$i].id" "$manifest")
    lane=$(jq -r ".tests[$i].lane" "$manifest")
    cmd=$(jq -r ".tests[$i].cmd" "$manifest")
    expect=$(jq -r ".tests[$i].expect" "$manifest")
    timeout_s=$(jq -r ".tests[$i].timeout // 600" "$manifest")
    local lo
    lo=$(order_of "$lane")
    if [[ "$lo" -lt 0 ]]; then fail "red $id: lane $lane is not a P3 phase in state.json"; continue; fi
    local should="red"
    case "$want" in
      green) should="green" ;;
      lane) [[ "$lo" -le "$mine" ]] && should="green" ;;
    esac
    local out rc
    out=$(timeout "$timeout_s" bash -c "$cmd" 2>&1); rc=$?
    if [[ "$should" == "green" ]]; then
      if [[ $rc -eq 0 ]]; then pass "red $id ($lane) is green"
      else fail "red $id ($lane) should pass, exit $rc"; printf '%s\n' "$out" | tail -n 20 | sed 's/^/     | /'; fi
    else
      if [[ $rc -eq 0 ]]; then fail "red $id ($lane) passes but should still fail"
      elif [[ $rc -eq 124 ]]; then fail "red $id ($lane) timed out after ${timeout_s}s, not its stated reason"
      elif printf '%s\n' "$out" | grep -Eq -- "$expect"; then pass "red $id ($lane) fails for its stated reason"
      else fail "red $id ($lane) fails, but not for its stated reason /$expect/"; printf '%s\n' "$out" | tail -n 20 | sed 's/^/     | /'; fi
    fi
  done
}

# ------------------------------------------------------------- frozen split
# contracts/eval.md carries `split-sha256: <64 hex>` and, when the split is in
# the repo, `split-path: <file or folder>`. A folder hashes as the sha256 of
# `sha256sum` lines for its files, sorted by path, paths relative to it.
check_split() {
  local doc=contracts/eval.md
  [[ -f "$doc" ]] || { fail "$doc missing (the frozen split, section 6)"; return; }
  local want path got
  want=$(grep -Eo '^split-sha256: *[0-9a-f]{64}' "$doc" | head -n1 | grep -Eo '[0-9a-f]{64}')
  [[ -n "$want" ]] || { fail "$doc has no split-sha256 line"; return; }
  pass "split hash present"
  path=$(grep -E '^split-path:' "$doc" | head -n1 | sed -E 's/^split-path: *//; s/ *$//')
  [[ -n "$path" ]] || { echo "     (no split-path: hash not recomputed; the split lives outside the repo)"; return; }
  if [[ -f "$path" ]]; then
    got=$(sha256sum "$path" | cut -d' ' -f1)
  elif [[ -d "$path" ]]; then
    got=$( (cd "$path" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum) | sha256sum | cut -d' ' -f1)
  else
    fail "split-path $path does not exist"; return
  fi
  [[ "$got" == "$want" ]] && pass "split at $path matches its hash" || fail "split at $path hashes to $got, contracts/eval.md says $want (the split is frozen)"
}

check_lanes_valid() {
  local own=scripts/lane-ownership.json l
  for l in "${lanes[@]}"; do
    jq -e --arg l "$l" '.lanes[$l].owns | type == "array" and length > 0' "$own" >/dev/null 2>&1 \
      && pass "lane $l owns paths" || fail "lane $l (a P3 phase) owns nothing in $own"
  done
  local extra
  extra=$(jq -r --argjson p "$(jq '[.phases[] | select(test("^P3"))]' "$state")" '.lanes | keys - $p | join(", ")' "$own")
  [[ -z "$extra" ]] && pass "every lane is a P3 phase" || fail "lanes not in state.json phases: $extra"
  # No path prefix owned by two lanes, unless one excepts it.
  local overlap
  overlap=$(jq -r '
    [.lanes | to_entries[] | .key as $l | .value.owns[] | {l: $l, p: .}] as $o
    | [ $o[] as $a | $o[] as $b
        | select($a.l < $b.l)
        | select(($a.p | startswith($b.p)) or ($b.p | startswith($a.p)))
        | "\($a.l):\($a.p) ~ \($b.l):\($b.p)" ] | unique | join("; ")' "$own")
  [[ -z "$overlap" ]] && pass "no path owned by two lanes" || echo "note overlapping prefixes (check each is cut out by an except): $overlap"
}

check_results() {
  shopt -s nullglob
  local files=(results/*.jsonl)
  shopt -u nullglob
  [[ ${#files[@]} -gt 0 ]] || { fail "no results/*.jsonl run artifacts"; return; }
  local f bad
  for f in "${files[@]}"; do
    bad=$(jq -rc 'select((type != "object") or ((["run_id","git_sha","machine","model_revision","quant","runtime","date"] - keys) | length > 0)) | (.run_id // "row without run_id")' "$f" 2>&1)
    if [[ $? -ne 0 ]]; then fail "$f is not JSON lines"
    elif [[ -n "$bad" ]]; then fail "$f rows missing run_id, git_sha, machine, model_revision, quant, runtime or date: $(echo "$bad" | head -n5 | tr '\n' ' ')"
    else pass "$f rows carry the section 6 fields"; fi
  done
}

# ---------------------------------------------------------------- per phase
case "$phase" in
  P0)
    ;;
  P1)
    nonempty SPEC.md && pass "SPEC.md written" || fail "SPEC.md missing or empty"
    if nonempty research/P1.md; then
      grep -Eq 'https?://' research/P1.md && pass "research/P1.md cites sources" || fail "research/P1.md has no URL"
    else
      fail "research/P1.md missing or empty"
    fi
    ;;
  P2)
    has_files contracts && pass "contracts/ written" || fail "contracts/ has nothing but its README"
    has_files fixtures && pass "fixtures/ written" || fail "fixtures/ has nothing but its README"
    check_lanes_valid
    check_split
    run_red red
    ;;
  P2w)
    has_files design && pass "design/ has exports" || fail "design/ has no exports"
    ;;
  P3*)
    lane_out=$(node scripts/lane-scope.mjs "$phase" --base "${FORGE_BASE:-origin/main}" 2>&1); rc=$?
    printf '%s\n' "$lane_out" | sed 's/^/     | /'
    [[ $rc -eq 0 ]] && pass "lane-scope $phase clean" || fail "lane-scope $phase (exit $rc)"
    check_split
    run_red lane
    ;;
  P4)
    has_files tests/integration && pass "cross-lane tests present" || fail "tests/integration/ has no tests"
    check_split
    run_red green
    ;;
  P5)
    check_split
    run_red green
    check_results
    ;;
  P6)
    for f in README.md DEFENSE.md SECURITY.md RELEASE_NOTES.md; do
      nonempty "$f" && pass "$f written" || fail "$f missing or empty"
    done
    has_files outreach && pass "outreach draft staged" || fail "outreach/ has no draft"
    check_split
    run_red green
    check_results
    ;;
  *)
    # P7 and later: Tony's steps; nothing to build.
    ;;
esac
# The split stays frozen in every phase after P2 that the case above skipped.
if [[ "$after_p2" -eq 1 && ! "$phase" =~ ^(P2|P3.*|P4|P5|P6)$ ]]; then check_split; fi

if (( fails > 0 )); then
  echo "forge-gate: $phase FAILED ($fails check(s))"
  exit 1
fi
echo "forge-gate: $phase passed"
