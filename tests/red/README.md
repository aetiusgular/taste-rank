# tests/red/

Written in P2: one failing test per corner case per lane. Lanes do not edit this folder.

`manifest.json` lists them. `scripts/forge-gate.sh` runs each `cmd` from the repo root with `bash -c` and checks:

- P2: every test fails (non-zero exit) and its output matches `expect` (an extended regex), the stated reason. An import error or a timeout is not the stated reason.
- P3x: tests of lanes up to and including P3x pass; later lanes' tests still fail for their stated reason.
- P4 onward: every test passes.

```json
{
  "tests": [
    {
      "id": "ingest-empty-file",
      "lane": "P3a",
      "case": "an empty transactions file yields zero rows, not an error",
      "cmd": "python -m pytest -q tests/red/test_ingest.py::test_empty_file",
      "expect": "NotImplementedError: ingest",
      "timeout": 600
    }
  ]
}
```

`lane` must be a P3 phase in `.forge/state.json`. `timeout` is in seconds (default 600). Install what the tests need in `scripts/setup.sh`; ci runs it before the gate.
