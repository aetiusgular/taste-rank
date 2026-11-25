# Security

Open items, rendered into the README in P6. Severity: high, medium, low.

| ID | Severity | Item | Owner | Workaround | Status |
|---|---|---|---|---|---|
| T-1 | low | `ci` runs the PR's own copy of `ci.yml`, so a PR that edits it could post statuses with the workflow token. | Tony | `forge/guard` (base copy) fails any PR that touches `.github/`, and review reads the diff. | open |
| T-2 | low | gitleaks honours inline `gitleaks:allow` comments; a PR could mark a line as allowed. | Tony | Review greps the diff for `gitleaks:allow`. | open |
| T-3 | low | Under the box's `landlock` profile, TCP rules name ports, not addresses: the model port and the proxy (ephemeral) port range are open to any host, not only 127.0.0.1. | Tony | Only for projects with `eval_boundary: landlock`; the box's nftables rules and the proxies' own allowlists still apply. | open |
| T-4 | medium | The headline base model's Hub repo ships remote code (`marqo_fashionSigLIP.py`, loaded by `trust_remote_code`). Running it unread could execute arbitrary code on the box. | P4 security review | Read in P4 at the pinned revision (sha256 in models.lock): no network, file or subprocess calls of its own, but see T-6. It is not run: models.lock drops `trust_remote_code`, and the image tower loads through open_clip `local-dir:` instead. | closed in P4 |
| T-5 | low | LoRA checkpoints are derived from competition data that may not be redistributed. | P3b, P5 | `train` refuses an `out_dir` inside the repo (red test `checkpoint-outside-repo`); weights stay box-local. | open |
| T-6 | medium | The remote code builds the model with open_clip `create_model("hf-hub:Marqo/marqo-fashionSigLIP")`, which downloads the repo's `main` branch with no revision, so the models.lock pin would not hold. The repo also holds a pickle checkpoint (`open_clip_pytorch_model.bin`) a loader can fall back to. | P4, then P5 | `tasterank.models.verify_model_dir` (P4) refuses a model folder unless it holds exactly the pinned `.json` and `.safetensors` files with their pinned sha256, and no `.py`, `.bin`, other file, subfolder or symlink. The host downloads those files at the pinned revision; the sandbox has no Hub access. P5's loader must call `verify_model_dir` before it loads either model (`tests/integration/test_models.py` covers the check). | fixed in P4; wired in P5: `load_siglip`, `load_fclip` and the box preflight call it before any load |
| T-7 | low | `image_dir_source` built the image path from the article id, so an id such as `../x` read a file outside the image folder. | P4 | Ids must match `[0-9A-Za-z_-]+` or `ValueError` (contracts/eval.md, "Missing images"). | fixed in P4 |
| T-8 | low | `eval.sh --fixture results` wrote synthetic rows into `results/`, next to where box rows go, and the gate counts every `results/*.jsonl`. | P4 | `run_fixture` refuses an out_dir at or under `results/` (symlinks resolved) before any work; exit 2. | fixed in P4 |
| T-9 | low | `read_machine` copies the whole of `$FORGE_MACHINE_FILE` into every results row, and rows are public. A wrong path there would publish that file. | forge eval workflow (Tony) | The workflow writes the file itself from `nvidia-smi` and `/proc` on the host; review reads the `machine` field of the first box rows. | open |
| T-10 | low | `train` checks `out_dir` and writes the checkpoint later, so a symlink swapped in between could still route weights elsewhere. | P5 | The box is single-user and the out_dir is box-local; the check resolves symlinks at the start. | open |
| T-11 | low | `scripts/setup.sh` pins versions but not hashes (`pip --require-hashes` is not used). | Tony | PyPI does not let a released file be replaced; ci runs without secrets on public PRs. | open |
| T-12 | low | `scripts/box-setup.sh` downloads model files from huggingface.co with curl and the competition zip with the Kaggle CLI (`uvx --from kaggle==2.2.5`), on the host, outside the sandbox. | Tony | Model files are refused unless `verify_model_dir` matches models.lock. The Kaggle CLI runs as Tony with his token, once, and never inside the sandbox. | open |
| T-13 | low | srt's `denyRead` list in forge's `sandbox/run-eval.sh` does not name `~/.kaggle`. | Tony (forge) | The token sits in Tony's home, which forge-runner cannot read (forge DECISIONS.md). Adding `~/.kaggle` to `denyRead` is a forge change. | open |
| T-14 | low | The box writes up to two uint8 image caches (about 150 KB per article each) and five LoRA checkpoints under `$FORGE_TMP/taste-rank`. These are derived from competition data. | P5 | They stay in the job workspace (`preflight` refuses a work folder inside the repo), and `results/` gets rows only. eval.yml uploads only `results/`. | open |

## P4 security review (2026-11-24)

Scope per BUILD_OUTLINE.md section 5: inputs, file writes, subprocess calls, network calls, and anything that runs another tool. Read at the P4 head on the Orca build host, CPU only.

| Area | What was read | Result |
|---|---|---|
| Inputs: CSVs | `tasterank.data` (polars `scan_csv`, every column typed, strict date and cast) | No code path evaluates input. A malformed file raises. No finding. |
| Inputs: split rule, models.lock | `load_split_rule`, `tasterank.models.load_lock` (`json.loads`) | JSON only. The frozen rule is hash-checked by the gate. No finding. |
| Inputs: images | `image_dir_source` (PIL `Image.open`) | T-7 fixed. PIL's own decompression-bomb limit stays on (it raises above twice `MAX_IMAGE_PIXELS`). |
| Inputs: checkpoints | `load_lora` (`torch.load(..., weights_only=True)`) | Refuses pickled code. Keys must match the LoRA layout exactly. No finding. |
| Inputs: model weights | Hub repos at the pinned revisions; `marqo_fashionSigLIP.py` read in full | T-4 closed, T-6 fixed (above). fashion-clip's `config.json` has no `auto_map`, so transformers loads no remote code for it. |
| File writes | `train` (out_dir outside the repo), `write_rows` (append, all rows checked first), `run_fixture` (out_dir outside `results/`) | T-8 fixed; T-5 and T-10 stay open at low. |
| Subprocess | `git rev-parse` / `git status` and `nvidia-smi`, argument lists, no shell, timeouts | No finding. `nvidia-smi` is found on `PATH`; the sandbox sets `PATH`. |
| Network | grep of `src/` and `scripts/eval.sh` for urllib, requests, socket, http, Hub clients | No network call in this repo's code. Model downloads happen on the host, outside the sandbox. |
| Other tools | `scripts/eval.sh` (exec of the venv Python, fixed module, argument count checked) | No finding. |
| Results rows | `check_row`: ids refused at any depth, required fields, enums, numbers, no NaN; P4 adds `n_customers`, `git_dirty` and `dataset` type checks | T-9 stays open at low. |

No S item above low is open: T-4 is closed and T-6 is fixed, with the P5 call to `verify_model_dir` as the remaining step.
