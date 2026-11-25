"""P5 pilot: the box pipeline with the real encoders on the 3090, on the synthetic fixture."""
import sys, json, torch
from pathlib import Path
from tasterank import box
from tasterank.seams import REPO_ROOT
S = Path(sys.argv[1]); out = Path(sys.argv[2]); steps = int(sys.argv[3])
FIX = REPO_ROOT / "fixtures" / "shop_like"
# usage: python p5-pilot.py <scratch dir holding models/siglip and models/fclip> <out dir> <steps>
class P(box.BoxPaths):
    @property
    def siglip(self): return S / "models" / "siglip"
    @property
    def fclip(self): return S / "models" / "fclip"
pp = P(root=S, data=FIX, work=S / "pilot-work", out=out)
print("preflight:", box.preflight(pp))
base, second = box.default_specs(pp, "cuda")
cfg = dict(box.load_e1_config()); cfg["steps"] = steps
ctx = box.run_box(pp, base, second, cfg, rule_path=FIX / "split.json", device="cuda", dataset="fixtures/shop_like (synthetic), box pilot")
for r in ctx.rows: r.setdefault("stage", None)
# LoRA weights moved off their zero init?
for ck in sorted((S / "pilot-work" / "lora").glob(f"{ctx.run_id}-seed*/lora.pt")):
    st = torch.load(ck, weights_only=True)
    b = [v for k, v in st.items() if "lora_B" in k]
    print(ck.parent.name, "tensors", len(st), "lora_B max |w|", max(float(t.abs().max()) for t in b))
(out / "summary.md").write_text(box.summary_markdown(ctx.rows))
