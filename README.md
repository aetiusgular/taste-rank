# taste-rank

Personalized fashion retrieval and ranking: a LoRA fine-tune measured on held-out customers.

Built in phases with the forge method. Phase P1: research and the pick are in `research/P1.md`, the build is in `SPEC.md`. Phase P2: the frozen split rule, metrics and seams are in `contracts/`, synthetic fixtures in `fixtures/`, and the red tests each lane must turn green in `tests/red/`. The headline number will be nDCG@10 lift of the fine-tuned encoder over zero-shot on held-out customers' later purchases (H&M Personalized Fashion Recommendations, Kaggle). No result exists yet.

The build answers a public job posting at Krea for personalization and recommendation work; Krea's public product has no taste learned from user behaviour (`research/P1.md`).

## Lane P3a: data

`src/tasterank/data/` reads the transactions and articles tables with polars (eager or lazy), assigns customers to train, dev and heldout by the salted hash frozen in `contracts/split.json`, builds each eval split (eligible customers, cold-start count, history, relevant sets, the 84-day candidate pool and the 7-day POP counts), orders histories, builds training pairs and computes the canonical id-list hash. What ran: the 13 P3a red tests and 15 lane tests in `tests/data/`, on CPU against the synthetic fixture `fixtures/shop_like` only. No competition data has been read; the box reads it in P5.

## Lane P3b: train

`src/tasterank/train/` builds customer vectors (recency-weighted mean of L2-normalized image embeddings, `contracts/eval.md`), computes in-batch InfoNCE, wraps the image tower's attention projections (`attn.qkv`, `attn.proj`) with LoRA through peft and freezes everything else, and runs the seeded training loop. Each step encodes every image its batch needs in one forward pass through the LoRA'd encoder, so gradients reach the adapters from both the history side and the target side. Batches hold distinct target articles. The loop refuses pairs dated on or after `train_cutoff` and any checkpoint directory inside the repo (symlinks resolved) before it does any work. It writes `lora.pt` (LoRA tensors only) and `lora.json` (config and per-step losses) to a box-local `out_dir`. `load_lora` rebuilds a trained encoder for evaluation. What ran: the 9 P3b red tests and 22 lane tests in `tests/train/`, on CPU, against the 16x16 `tasterank.tiny` encoder and the synthetic fixture images only. No real encoder has been trained, and no competition data has been read. The box does both in P5.
