import sys, os, random, json
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from grid import load_feats, grid
from common import WORK

rows = [r for r in load_feats() if min(r["w_box"], r["h_box"]) >= 45]
random.seed(2024)
strata = [(0.0, 0.0001, "s0"), (0.0001, 10, "s1"), (10, 50, "s2"), (50, 75, "s3"), (75, 90, "s4"), (90, 101, "s5")]
sel = []
for lo, hi, name in strata:
    c = [r for r in rows if lo <= r["pct"] < hi]
    random.shuffle(c)
    sel += c[:20]
random.shuffle(sel)
with open(os.path.join(WORK, "gt_set.json"), "w", encoding="utf-8") as f:
    json.dump([{k: r[k] for k in ("stem", "idx", "label", "pct", "pct_deep", "pct_rej", "surf_frac",
                                  "leaf_frac", "w_box", "h_box", "mode")} for r in sel], f, indent=1)
for g in range(0, len(sel), 12):
    chunk = sel[g:g + 12]
    items = [(r["stem"], r["idx"], f"{r['stem']}#{r['idx']}", "") for r in chunk]
    grid(items, os.path.join(WORK, f"gt_{g//12:02d}.png"), cols=6, T=240,
         title=f"GT lote {g//12}: asignar estado 0-4 (0 sin color, 1 <50%, 2 50-75%, 3 75-90%, 4 >=90%)")
print("lotes:", (len(sel) + 11) // 12, "n:", len(sel))
