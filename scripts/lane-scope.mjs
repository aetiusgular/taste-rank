#!/usr/bin/env node
// Lane scope gate (BUILD_OUTLINE.md sections 1 and 4). Ported from derm's
// scripts/lane-scope.mjs: same rules, derm's lanes dropped, lane names are
// the P3 phase names in .forge/state.json.
//
//   node scripts/lane-scope.mjs <lane> [--base <commit>]
//
// Checks the lane's committed work (merge-base of <base> and HEAD to HEAD)
// against scripts/lane-ownership.json:
//   - every added, changed or deleted path is inside the lane's folders, or
//     on the `everyLane` list;
//   - a shared file changes only inside the lane's own `[lane-<name>]`
//     section, and no marker moves or disappears;
//   - no new test file matching an `uncollected` rule (a test the project's
//     runner would never collect);
//   - a `noLane` path is refused unless the lane owns it by name.
// Exit 0 clean, 1 violations, 2 usage error. Reads git only; writes nothing.
// Without --base it uses origin/main.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

const ownership = JSON.parse(readFileSync(new URL("./lane-ownership.json", import.meta.url), "utf8"));
const lanes = ownership.lanes ?? {};
const everyLane = ownership.everyLane ?? [];
const shared = ownership.shared ?? {};
const noLaneList = ownership.noLane ?? [];
const uncollected = ownership.uncollected ?? [];

function usage(message) {
  console.error(`lane-scope: ${message}`);
  console.error(`usage: node scripts/lane-scope.mjs <lane: ${Object.keys(lanes).join(", ") || "none defined"}> [--base <commit>]`);
  process.exit(2);
}

function git(args) {
  return execFileSync("git", args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
}

const args = process.argv.slice(2);
// Lane names are taken as written; failing that, matched without regard to
// case (p3a finds P3a).
const wanted = args[0] ?? "";
const lane = Object.hasOwn(lanes, wanted)
  ? wanted
  : Object.keys(lanes).find((l) => l.toLowerCase() === wanted.toLowerCase());
if (!lane) usage(`unknown lane ${JSON.stringify(wanted)}`);
let base = "origin/main";
for (let i = 1; i < args.length; i++) {
  if (args[i] === "--base" && args[i + 1]) base = args[++i];
  else usage(`unexpected argument ${JSON.stringify(args[i])}`);
}

let mergeBase;
try {
  git(["rev-parse", "--verify", "--quiet", `${base}^{commit}`]);
  mergeBase = git(["merge-base", base, "HEAD"]).trim();
} catch {
  usage(`base ${JSON.stringify(base)} is not a commit here`);
}

const matches = (list, path) => list.some((p) => (p.endsWith("/") ? path.startsWith(p) : path === p));
const spec = lanes[lane];
const ownedBy = (l, path) => matches(lanes[l].owns ?? [], path) && !matches(lanes[l].except ?? [], path);
const owned = (path) => ownedBy(lane, path);
const ownerOf = (path) => Object.keys(lanes).filter((l) => ownedBy(l, path));
const noLane = (path) => matches(noLaneList, path);
const anyLane = (path) => matches(everyLane, path);
// A marker line names one lane: `# [lane-P3a]`, `// [lane-P3a]` or
// `<!-- [lane-P3a] -->`.
const MARKER = /^(?:#|\/\/|<!--) \[lane-([A-Za-z0-9]+)\](?: -->)?\s*$/;

const changes = git(["diff", "--name-status", "--no-renames", "-z", mergeBase, "HEAD"])
  .split("\0")
  .filter(Boolean);
const errors = [];
for (let i = 0; i + 1 < changes.length; i += 2) {
  const status = changes[i];
  const path = changes[i + 1];
  if (shared[path]) {
    if (!shared[path].includes(lane)) errors.push(`${path}: shared file, lane ${lane} has no section in it`);
    else if (status !== "M") errors.push(`${path}: shared file ${status === "D" ? "deleted" : "added"}`);
    else checkSections(path);
    continue;
  }
  if (noLane(path) && !owned(path)) {
    errors.push(`${path}: no lane edits this (write it under "Requests for integration")`);
    continue;
  }
  if (!owned(path) && !anyLane(path)) {
    const others = ownerOf(path);
    errors.push(`${path}: outside lane ${lane}${others.length ? ` (owned by lane ${others.join(", ")})` : " (owned by no lane)"}`);
    continue;
  }
  if (status === "A") {
    for (const rule of uncollected) {
      if (matches(rule.under ?? [""], path) && new RegExp(rule.pattern).test(path)) {
        errors.push(`${path}: ${rule.why ?? "the test runner would never collect this file"}`);
      }
    }
  }
}

function sections(text) {
  // Returns { outside: string[], byLane: Map<lane, string[]> } where a
  // section starts at a marker line and ends at the next marker or the next
  // TOML table header (a line starting with `[`).
  const outside = [];
  const byLane = new Map();
  let current = null;
  for (const line of text.split("\n")) {
    const marker = MARKER.exec(line);
    if (marker) {
      current = marker[1];
      if (!byLane.has(current)) byLane.set(current, []);
      outside.push(line);
      continue;
    }
    if (/^\[/.test(line)) current = null;
    if (current) byLane.get(current).push(line);
    else outside.push(line);
  }
  return { outside, byLane };
}

function checkSections(path) {
  const before = sections(git(["show", `${mergeBase}:${path}`]));
  const after = sections(git(["show", `HEAD:${path}`]));
  if (before.outside.join("\n") !== after.outside.join("\n")) {
    errors.push(`${path}: changed outside the [lane-${lane}] sections, or a marker moved`);
  }
  for (const [l, lines] of before.byLane) {
    if (l === lane) continue;
    if ((after.byLane.get(l) ?? []).join("\n") !== lines.join("\n")) errors.push(`${path}: lane ${l}'s section changed`);
  }
  for (const l of after.byLane.keys()) {
    if (!before.byLane.has(l)) errors.push(`${path}: new section [lane-${l}]`);
  }
}

if (errors.length) {
  console.error(`lane-scope: lane ${lane}, ${errors.length} problem(s) since ${mergeBase.slice(0, 12)}:`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}
if (!(spec.owns ?? []).length) {
  console.log(`lane-scope: lane ${lane} owns no paths yet (P2 fills scripts/lane-ownership.json)`);
}
console.log(`lane-scope: lane ${lane} clean since ${mergeBase.slice(0, 12)} (${changes.length / 2} path(s))`);
