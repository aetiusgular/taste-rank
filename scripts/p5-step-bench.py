"""usage: python p5-step-bench.py <scratch dir> <n_last> <1|0 grad checkpointing>

One E1 step at the e1.json image budget on the real SigLIP tower: seconds per step and peak memory. Random pixels."""
import sys, time, json, warnings, datetime as dt
import torch
from tasterank import box, train
from tasterank.encoders import load_siglip
from tasterank.seams import TrainConfig
from tasterank import eval as ev
S = sys.argv[1]; n_last = int(sys.argv[2]); ck = sys.argv[3] == "1"
cfg_e1 = box.load_e1_config(); B = box.e1_batch_size(cfg_e1, n_last); n_img = B * (1 + n_last)
enc = load_siglip(f"{S}/models/siglip", "cuda")
if ck: enc.set_grad_checkpointing(True)
cfg = TrainConfig(out_dir=f"{S}/x", train_cutoff=dt.date(2020,9,9), rank=cfg_e1["rank"], alpha=cfg_e1["alpha"])
m = train.apply_lora(enc, cfg); m.train()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
px = torch.rand(n_img, 3, 224, 224, device="cuda")
times = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for i in range(6):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        e = m(px); c = e[:B]; tgt = e[B:2*B]
        loss = train.info_nce(c, tgt, 0.07); opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); times.append(time.time() - t)
    msgs = sorted({str(x.message)[:160] for x in w})
row = {"stage": "pilot_step_bench", "n_last": n_last, "batch_pairs": B, "images_per_step": n_img, "grad_checkpointing": ck,
       "seconds_per_step_median_of_last5": sorted(times[1:])[2], "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated()/2**30, 2),
       "gpu": ev.read_gpu()[0], "gpu_power_limit_w": ev.read_gpu()[1], "machine": ev.read_machine(), "runtime": __import__("tasterank.encoders").encoders.runtime_label(),
       "model_revision": "Marqo/marqo-fashionSigLIP@c56244cc94f92419e8369fa71efdaf403b124ce8", "quant": "bf16 autocast (weights fp32)",
       "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "pixels": "random, not competition data", "warnings": msgs}
print(json.dumps(row))
