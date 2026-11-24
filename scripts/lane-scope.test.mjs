#!/usr/bin/env node
// Self-test for lane-scope.mjs, ported from derm: one throwaway commit per
// rule, in a scratch repository under the OS temp folder. The scratch repo
// gets a copy of lane-scope.mjs next to its own fixture lane-ownership.json,
// so this test does not depend on the lanes P2 writes.
// Run: node scripts/lane-scope.test.mjs
// Exit 0 when every case gets the expected exit code and message.

import { execFileSync, spawnSync } from "node:child_process";
import { copyFileSync, mkdtempSync, mkdirSync, writeFileSync, rmSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const source = join(dirname(fileURLToPath(import.meta.url)), "lane-scope.mjs");
const repo = mkdtempSync(join(tmpdir(), "forge-scope-"));
const toolDir = mkdtempSync(join(tmpdir(), "forge-scope-tool-"));
const script = join(toolDir, "lane-scope.mjs");
copyFileSync(source, script);
writeFileSync(join(toolDir, "lane-ownership.json"), JSON.stringify({
  lanes: {
    P3a: { owns: ["src/ingest/", "src/shared/a.py"], except: ["src/ingest/vendor/"] },
    P3b: { owns: ["src/rank/", "src/ingest/vendor/", "contracts/fixtures/rank/"], except: [] },
    P3c: { owns: ["src/report/"], except: [] },
    P3d: { owns: ["src/shared/a.py"], except: [] },
  },
  everyLane: [".forge/state.json", "DECISIONS.md", "reviews/"],
  shared: { "pyproject.toml": ["P3a", "P3b"], "README.md": ["P3a", "P3c"] },
  noLane: ["scripts/", "contracts/", "tests/red/", "SPEC.md"],
  uncollected: [{ under: ["src/"], pattern: "_spec\\.py$", why: "pytest collects only test_*.py" }],
}, null, 2));

const git = (...a) => execFileSync("git", a, { cwd: repo, encoding: "utf8" });
const write = (p, text) => {
  mkdirSync(dirname(join(repo, p)), { recursive: true });
  writeFileSync(join(repo, p), text);
};

const toml = `[project]
name = "demo"

[project.optional-dependencies]
# [lane-P3a]
ingest = ["polars"]
# [lane-P3b]

[tool.pytest.ini_options]
# [lane-P3a]
# [lane-P3b]
`;

const readme = `# demo

## What ran
<!-- [lane-P3a] -->
ingest: nothing yet
<!-- [lane-P3c] -->
report: nothing yet
`;

git("init", "-q", "-b", "main");
git("config", "user.name", "scope test");
git("config", "user.email", "scope@test.invalid");
for (const [p, t] of Object.entries({
  "pyproject.toml": toml,
  "README.md": readme,
  "src/ingest/load.py": "# a\n",
  "src/ingest/vendor/lib.py": "# b\n",
  "src/rank/model.py": "# b\n",
  "src/report/table.py": "# c\n",
  "src/shared/a.py": "# a or d\n",
  "contracts/eval.md": "# eval\n",
  "contracts/fixtures/rank/cases.json": "[]\n",
  "tests/red/test_ingest.py": "# red\n",
  "scripts/forge-gate.sh": "#!/bin/sh\n",
  ".forge/state.json": "{}\n",
  "DECISIONS.md": "# d\n",
  "SPEC.md": "# s\n",
})) write(p, t);
git("add", "-A");
git("commit", "-q", "-m", "base");
const base = git("rev-parse", "HEAD").trim();

const cases = [
  ["own file edited", "P3a", () => write("src/ingest/load.py", "# a2\n"), 0, "clean"],
  ["new file in own folder", "P3b", () => write("src/rank/score.py", "# s\n"), 0, "clean"],
  ["lowercase lane name", "p3a", () => write("src/ingest/load.py", "# a3\n"), 0, "clean"],
  ["another lane's file", "P3a", () => write("src/rank/model.py", "# x\n"), 1, "owned by lane P3b"],
  ["excepted folder edited by the lane it is cut from", "P3a", () => write("src/ingest/vendor/lib.py", "# x\n"), 1, "owned by lane P3b"],
  ["excepted folder edited by its owner", "P3b", () => write("src/ingest/vendor/lib.py", "# b2\n"), 0, "clean"],
  ["file owned by two lanes, edited by a third", "P3c", () => write("src/shared/a.py", "# x\n"), 1, "owned by lane P3a, P3d"],
  ["new file outside every lane", "P3c", () => write("docs/x.md", "x\n"), 1, "owned by no lane"],
  ["deleted file of another lane", "P3a", () => unlinkSync(join(repo, "src/report/table.py")), 1, "owned by lane P3c"],
  ["no-lane: contracts", "P3a", () => write("contracts/eval.md", "# changed\n"), 1, "no lane edits this"],
  ["no-lane: red tests", "P3a", () => write("tests/red/test_ingest.py", "# weakened\n"), 1, "no lane edits this"],
  ["no-lane: gate script", "P3c", () => write("scripts/forge-gate.sh", "#!/bin/sh\nexit 0\n"), 1, "no lane edits this"],
  ["no-lane: SPEC.md", "P3b", () => write("SPEC.md", "# changed\n"), 1, "no lane edits this"],
  ["no-lane path owned by name", "P3b", () => write("contracts/fixtures/rank/cases.json", "[1]\n"), 0, "clean"],
  ["no-lane path owned by another lane by name", "P3a", () => write("contracts/fixtures/rank/cases.json", "[2]\n"), 1, "no lane edits this"],
  ["every-lane: state.json", "P3c", () => write(".forge/state.json", "{\"phase\":\"P3c\"}\n"), 0, "clean"],
  ["every-lane: DECISIONS.md", "P3a", () => write("DECISIONS.md", "# d\n- call\n"), 0, "clean"],
  ["every-lane: new review file", "P3b", () => write("reviews/P3b-r1.md", "# r\n"), 0, "clean"],
  ["uncollected test file refused", "P3a", () => write("src/ingest/load_spec.py", "#\n"), 1, "pytest collects only"],
  ["collected test file allowed", "P3a", () => write("src/ingest/test_load.py", "#\n"), 0, "clean"],
  ["shared TOML, own section", "P3a", () => write("pyproject.toml", toml.replace('ingest = ["polars"]\n', 'ingest = ["polars", "duckdb"]\n')), 0, "clean"],
  ["shared TOML, other lane's section", "P3b", () => write("pyproject.toml", toml.replace('ingest = ["polars"]\n', "")), 1, "lane P3a's section changed"],
  ["shared TOML, own second section", "P3b", () => write("pyproject.toml", toml.replace(/# \[lane-P3b\]\n$/, '# [lane-P3b]\nmarkers = ["rank"]\n')), 0, "clean"],
  ["shared TOML, outside sections", "P3a", () => write("pyproject.toml", toml.replace('name = "demo"', 'name = "demo2"')), 1, "changed outside"],
  ["shared TOML, marker removed", "P3a", () => write("pyproject.toml", toml.replace("# [lane-P3b]\n\n[tool", "\n[tool")), 1, "changed outside"],
  ["shared TOML, lane without a section", "P3c", () => write("pyproject.toml", toml.replace('ingest = ["polars"]\n', "")), 1, "has no section"],
  ["shared TOML, new section marker", "P3a", () => write("pyproject.toml", toml.replace("# [lane-P3b]\n\n[tool", "# [lane-P3b]\n# [lane-P3c]\n\n[tool")), 1, "changed outside"],
  ["shared README, own section (HTML marker)", "P3c", () => write("README.md", readme.replace("report: nothing yet", "report: ran on ubuntu-latest")), 0, "clean"],
  ["shared README, other lane's section", "P3c", () => write("README.md", readme.replace("ingest: nothing yet", "ingest: done")), 1, "lane P3a's section changed"],
  ["shared file deleted", "P3a", () => unlinkSync(join(repo, "README.md")), 1, "shared file deleted"],
];

let failed = 0;
for (const [name, lane, change, code, text] of cases) {
  git("checkout", "-q", "-B", "case", base);
  change();
  git("add", "-A");
  git("commit", "-q", "-m", name);
  const r = spawnSync("node", [script, lane, "--base", base], { cwd: repo, encoding: "utf8" });
  const out = r.stdout + r.stderr;
  const ok = r.status === code && out.includes(text);
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: exit ${r.status} (want ${code})${ok ? "" : `\n${out}`}`);
}

const usageCases = [
  ["unknown lane", ["P3z", "--base", base], 2],
  ["no lane", [], 2],
  ["bad base", ["P3a", "--base", "no-such-commit"], 2],
  ["stray argument", ["P3a", "--frob"], 2],
];
for (const [name, argv, code] of usageCases) {
  const r = spawnSync("node", [script, ...argv], { cwd: repo, encoding: "utf8" });
  const ok = r.status === code;
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: exit ${r.status} (want ${code})`);
}

rmSync(repo, { recursive: true, force: true });
rmSync(toolDir, { recursive: true, force: true });
const total = cases.length + usageCases.length;
console.log(`${total - failed} of ${total} as expected`);
process.exit(failed ? 1 : 0);
