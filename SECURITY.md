# Security

Open items, rendered into the README in P6. Severity: high, medium, low.

| ID | Severity | Item | Owner | Workaround | Status |
|---|---|---|---|---|---|
| T-1 | low | `ci` runs the PR's own copy of `ci.yml`, so a PR that edits it could post statuses with the workflow token. | Tony | `forge/guard` (base copy) fails any PR that touches `.github/`, and review reads the diff. | open |
| T-2 | low | gitleaks honours inline `gitleaks:allow` comments; a PR could mark a line as allowed. | Tony | Review greps the diff for `gitleaks:allow`. | open |
| T-3 | low | Under the box's `landlock` profile, TCP rules name ports, not addresses: the model port and the proxy (ephemeral) port range are open to any host, not only 127.0.0.1. | Tony | Only for projects with `eval_boundary: landlock`; the box's nftables rules and the proxies' own allowlists still apply. | open |
| T-4 | medium | The headline base model loads remote code (`trust_remote_code`, models.lock). Running it unread could execute arbitrary code on the box. | P4 security review | Pinned by commit in models.lock; P4 reads the code at that commit before P5 runs it; the box runs it inside srt with network limited to 127.0.0.1. | open |
| T-5 | low | LoRA checkpoints are derived from competition data that may not be redistributed. | P3b, P5 | `train` refuses an `out_dir` inside the repo (red test `checkpoint-outside-repo`); weights stay box-local. | open |
