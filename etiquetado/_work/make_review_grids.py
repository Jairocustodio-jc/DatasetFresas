import sys, os, random, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from grid import load_feats, grid
from common import WORK

rows = load_feats()
random.seed(42)


def pick(cond, n, key=None):
    sel = [r for r in rows if cond(r) and min(r["w_box"], r["h_box"]) >= 45]
    random.shuffle(sel)
    sel = sel[:n]
    if key:
        sel.sort(key=key)
    return sel


def items(sel):
    return [(r["stem"], r["idx"], f"{r['stem']}#{r['idx']} {r['label'][:9]}",
             f"pct={r['pct']:.0f} dp={r['pct_deep']:.0f}") for r in sel]


sets = {
    "g_ep0":   (lambda r: r["label"] == "early-pink" and r["pct"] < 0.5, "early-pink medidos con 0% de color"),
    "g_faint": (lambda r: 0.5 <= r["pct"] < 12, "medidos 0.5-12% (rubor incipiente)"),
    "g_c50":   (lambda r: 42 <= r["pct"] <= 58, "frontera 50% (42-58)"),
    "g_c75":   (lambda r: 68 <= r["pct"] <= 82, "frontera 75% (68-82)"),
    "g_c90":   (lambda r: 84 <= r["pct"] <= 96, "frontera 90% (84-96)"),
    "g_hi":    (lambda r: r["pct"] > 96, "medidos >96%"),
}
for name, (cond, title) in sets.items():
    sel = pick(cond, 24, key=lambda r: r["pct"])
    print(name, len(sel), grid(items(sel), os.path.join(WORK, name + ".png"), cols=6, T=190, title=title))
    with open(os.path.join(WORK, name + ".txt"), "w", encoding="utf-8") as f:
        for r in sel:
            f.write(f"{r['stem']}#{r['idx']}\t{r['label']}\tpct={r['pct']:.1f}\tdeep={r['pct_deep']:.1f}\n")
