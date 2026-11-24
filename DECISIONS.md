# Decisions

Every call made in a phase, newest last. "agent's call" for rule 2 of BUILD_OUTLINE.md section 2 (reversible, inside the session's own folders); "Tony's call" for anything Tony decided. Each entry has its reason and what would reverse it.

| Date | Phase | Call | Decision | Reason | Reverse if |
|---|---|---|---|---|---|
| 2026-11-23 | P0 | agent's call | Repo standard from forge commit 65e468cfd948 (template/, BUILD_OUTLINE.md section 7). | P0 init by the forge bootstrap session; the per-file reasons are in forge's DECISIONS.md. | the template changes before P1 starts |
| 2026-11-23 | P1 | agent's call | D-P1-1: Build candidate A, taste-rank (22), over B pick-profile (17) and C intake-match (16). | Section 2 rubric scores in research/P1.md section 6; A leads on "Not a clone" and budget. | a review finding shows a scored claim does not hold at its source (reopening is Tony's call) |
| 2026-11-23 | P1 | agent's call | D-P1-2: Headline number is nDCG@10 lift, E1 minus E0, same base model, held-out customers and window, five-seed band. | Brief's candidate; ground truth is the retailer's recorded purchases, outside the build. | P2 finds the held-out window too sparse to give a stable band |
| 2026-11-23 | P1 | agent's call | D-P1-3: Headline base model Marqo-FashionSigLIP (Apache-2.0, c56244cc); fashion-clip (MIT, 7e3ba62c) as a second E0 row. | Both licences permit use; SigLIP is the newer fashion encoder; a second E0 row guards against one model's quirks. | the SigLIP remote code fails the P4 security read, or will not train with LoRA on the 3090 |
| 2026-11-23 | P1 | agent's call | D-P1-4: Not stopping on the H&M terms; non-commercial fit of a public portfolio repo is flagged for Tony by P6. | The terms bar redistribution and limit use to non-commercial, academic and education; they do not name results or weights, which is the brief's stop condition. | Tony rules the use is not covered |
| 2026-11-23 | P1 | agent's call | D-P1-5: No H&M data, ids or derived tables in the repo or results; fixtures synthetic; LoRA weights not published; split id lists stay on the box (hash only). | The repo is public and rule 7.B bars making the data available; withholding is cheaper to reverse than publishing. | Tony confirms in writing that weights may be published |
| 2026-11-23 | P1 | agent's call | D-P1-6: "Series B" from the brief is dropped as unverified. | The posting states funds raised, not a round; no other source read names it. | a dated public source names the round |
