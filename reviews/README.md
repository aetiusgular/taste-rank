# reviews/

One file per review round: `P<k>-r<n>.md`, written by a fresh review session (BUILD_OUTLINE.md section 5).

Finding IDs: `<code>-P<k>-r<n>-<class><m>`, with the code from `.forge/state.json`. Classes: S security, C correctness, E eval validity, D docs and claims. Every finding has evidence: a file and line, a command and its output, or a URL and what it says.

Verdict: PASS when there is no S, C or E finding above minor and every D finding about a number is fixed. Otherwise FIX.
