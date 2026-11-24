<!-- Title: "[P<k>] <project>: <one line>" or "[P<k> r<n>] <project>: <one line>". No attribution trailers, footers or session links. -->

## Phase

P<k>, round <n>. Exit gate (BUILD_OUTLINE.md section 3): <the gate, in one line>.

## Gate checklist

- [ ] `ci`: tests and lint (`scripts/test.sh`), red-test expectations, lane-scope (P3), `scripts/forge-gate.sh P<k>`, gitleaks, attribution scan
- [ ] `forge/guard`: nothing under `.github/`, `scripts/forge-gate.sh`, `scripts/lane-scope.mjs`, or (outside P2) `scripts/lane-ownership.json`
- [ ] `forge/eval`: P5 runs on the box; other phases post "not an eval phase"
- [ ] `forge/review`: set by a fresh review session (`reviews/P<k>-r<n>.md`)
- [ ] Phase exit gate met: <how, with the file or command that shows it>
- [ ] Human gate (P6, P7, or in `optional_human_gates`): labeled `needs-tony`, auto-merge off. Otherwise auto-merge on.

## Decisions

<!-- Every call made in this phase, one line each, also in DECISIONS.md. Tag: agent's call or Tony's call. -->

- 

## Requests for integration

<!-- Changes this PR needs in contracts/, another lane's folders, or the spec's headline number. P4 applies or refuses each one. "None" if none. -->

- None

## What ran and where

<!-- Commands, the machine (read at run time), commit, model revision, date. Any number here needs a results/*.jsonl row; estimates say "estimate". -->

- 
