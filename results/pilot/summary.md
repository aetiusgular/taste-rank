## taste-rank box run 20261125T012344Z-976d20a7

git `98cf680bdbb3`, machine: cpu: AMD Ryzen 5 9600X 6-Core Processor x12; mem: 121.2 GiB; gpu: NVIDIA GeForce RTX 3090, power limit 280 W

- **dev nDCG@10 lift, E1-siglip minus E0-siglip, five seeds:** mean +0.0420, band [+0.0290, +0.0575], verdict **lift** (E0 0.6504)
- **heldout nDCG@10 lift, E1-siglip minus E0-siglip, five seeds:** mean -0.0478, band [-0.0717, -0.0349], verdict **worse** (E0 0.4618)
- E0 repeat (heldout): max metric difference 0, within 1e-6: True
- History chosen on dev with E0 only: n_last 5, half-life None (dev nDCG@10 0.6504, 5 grid points)

### Overall

| row | split | seed | n | nDCG@10 | recall@12 | recall@50 | pool coverage | missing images |
|---|---|---|---|---|---|---|---|---|
| POP | dev | None | 9 | 0.0770 | 0.3333 | 1.0000 | 1.0000 | 0 |
| POP | heldout | None | 7 | 0.1368 | 0.3333 | 1.0000 | 1.0000 | 0 |
| E0-siglip | dev | None | 9 | 0.6504 | 0.7222 | 1.0000 | 1.0000 | 0 |
| E0-siglip | heldout | None | 7 | 0.4618 | 0.6667 | 1.0000 | 1.0000 | 0 |
| E0-fclip | dev | None | 9 | 0.6514 | 0.8333 | 1.0000 | 1.0000 | 0 |
| E0-fclip | heldout | None | 7 | 0.4971 | 0.6667 | 1.0000 | 1.0000 | 0 |
| E1-siglip | dev | 0 | 9 | 0.6874 | 0.7963 | 1.0000 | 1.0000 | 0 |
| E1-siglip | heldout | 0 | 7 | 0.4156 | 0.5952 | 1.0000 | 1.0000 | 0 |
| E1-siglip | dev | 1 | 9 | 0.7079 | 0.8889 | 1.0000 | 1.0000 | 0 |
| E1-siglip | heldout | 1 | 7 | 0.4269 | 0.5952 | 1.0000 | 1.0000 | 0 |
| E1-siglip | dev | 2 | 9 | 0.6874 | 0.8519 | 1.0000 | 1.0000 | 0 |
| E1-siglip | heldout | 2 | 7 | 0.4129 | 0.5952 | 1.0000 | 1.0000 | 0 |
| E1-siglip | dev | 3 | 9 | 0.6999 | 0.7963 | 1.0000 | 1.0000 | 0 |
| E1-siglip | heldout | 3 | 7 | 0.4243 | 0.6667 | 1.0000 | 1.0000 | 0 |
| E1-siglip | dev | 4 | 9 | 0.6794 | 0.7963 | 1.0000 | 1.0000 | 0 |
| E1-siglip | heldout | 4 | 7 | 0.3901 | 0.7381 | 1.0000 | 1.0000 | 0 |

### Heldout per product group (nDCG@10)

| group | n | POP | E0-siglip | E1-siglip mean [min, max] | E1 − E0 |
|---|---|---|---|---|---|
| Bottoms | 5 | 0.6245 | 0.6052 | 0.5796 [0.5649, 0.5897] | -0.0256 |
| Accessories | 3 | 0.4213 | 0.8770 | 0.7273 [0.6872, 0.7540] | -0.1497 |
| Tops | 3 | 0.4580 | 0.4502 | 0.4364 [0.4266, 0.4444] | -0.0138 |
| Footwear | 2 | 0.3715 | 0.3820 | 0.4317 [0.4088, 0.4653] | +0.0497 |

### Failure cases (heldout groups where every E1 seed is below E0 or POP)

- Bottoms (5 customers): E1 max 0.5897, E0 0.6052, POP 0.6245
- Accessories (3 customers): E1 max 0.7540, E0 0.8770, POP 0.4213
- Tops (3 customers): E1 max 0.4444, E0 0.4502, POP 0.4580
