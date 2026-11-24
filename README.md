# taste-rank

Personalized fashion retrieval and ranking: a LoRA fine-tune measured on held-out customers.

Built in phases with the forge method. Phase P1: research and the pick are in `research/P1.md`, the build is in `SPEC.md`. Phase P2: the frozen split rule, metrics and seams are in `contracts/`, synthetic fixtures in `fixtures/`, and the red tests each lane must turn green in `tests/red/`. The headline number will be nDCG@10 lift of the fine-tuned encoder over zero-shot on held-out customers' later purchases (H&M Personalized Fashion Recommendations, Kaggle). No result exists yet.

The build answers a public job posting at Krea for personalization and recommendation work; Krea's public product has no taste learned from user behaviour (`research/P1.md`).
