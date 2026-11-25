# P5 pilot (not an eval result)

These rows come from the box pipeline (`tasterank.box.run_box`) with the **real encoders on the 3090**, on the **synthetic fixture** (`fixtures/shop_like`: 40 made-up 16x16 images, 9 dev and 7 heldout customers). The pilot ran on 2026-11-25 at commit `98cf680` with a clean tree. Every row carries `dataset: "fixtures/shop_like (synthetic), box pilot"`. The nDCG and recall values measure nothing about fashion ranking. They sit here, not in `results/*.jsonl`, so no gate or table counts them (DECISIONS.md D-P5-8).

Scripts: `scripts/p5-pilot.py` (the whole pipeline: POP, the dev grid, E0 for both models with the repeat, E1 for five seeds at 30 steps each) and `scripts/p5-step-bench.py` (one E1 step at the `configs/e1.json` image budget, random pixels). Model folders were downloaded from the public Hub at the models.lock revisions with no token, and `verify_model_dir` passed for both.

What the pilot shows:

| Check | Result | Row |
|---|---|---|
| The pipeline runs end to end on the GPU with both real encoders | 92 rows, every one passing `check_row` | `box-20261125T012344Z-976d20a7.jsonl` |
| E0 repeat from a fresh load | max metric difference 0 (tolerance 1e-6) | rows with `repeat_of` |
| LoRA gets gradients through gradient checkpointing | 48 LoRA tensors per checkpoint (24 modules); every `lora_B` moved off its zero init (printed by `p5-pilot.py`) | E1 rows, `train.trainable_params` |
| The pinned weights are what loads | a SigLIP `attn.qkv` weight and the fashion-clip `visual_projection` equal the safetensors values (checked by hand in the session, not a row) | none |
| One E1 step at the image budget | n_last 20: 32 pairs, 672 images, 1.91 s, 9.42 GiB peak. n_last 50: 13 pairs, 663 images, 1.89 s, 9.30 GiB. n_last 5: 32 pairs, 192 images, 0.55 s, 2.95 GiB. Without gradient checkpointing, the 672-image step ran out of GPU memory (seen in the session, no row). | `step-bench.jsonl` |

**Estimate, not a measurement:** at the 672-image budget, 1500 steps come to about 48 minutes of training per seed, about 4 hours for five seeds. That is inside eval.yml's 720-minute job limit. Real steps share some images between pairs, so they encode fewer than the budget allows.
