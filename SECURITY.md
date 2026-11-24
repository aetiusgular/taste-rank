# Security

Open items, rendered into the README in P6. Severity: high, medium, low.

| ID | Severity | Item | Owner | Workaround | Status |
|---|---|---|---|---|---|
| T-1 | low | `ci` runs the PR's own copy of `ci.yml`, so a PR that edits it could post statuses with the workflow token. | Tony | `forge/guard` (base copy) fails any PR that touches `.github/`, and review reads the diff. | open |
| T-2 | low | gitleaks honours inline `gitleaks:allow` comments; a PR could mark a line as allowed. | Tony | Review greps the diff for `gitleaks:allow`. | open |
| T-3 | low | Under the box's `landlock` profile, TCP rules name ports, not addresses: the model port and the proxy (ephemeral) port range are open to any host, not only 127.0.0.1. | Tony | Only for projects with `eval_boundary: landlock`; the box's nftables rules and the proxies' own allowlists still apply. | open |
