"""Hojas de contacto para el afinado: E (palidas), G (granates), F (dano)."""
import sys, os, csv, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, WORK, SRC, imread_u, imwrite_u, load_boxes
from pigment import analyze, CFG
from grid import grid

AUD = os.path.join(WORK, "audit")
BR = os.path.join(WORK, "crops_brown")
os.makedirs(BR, exist_ok=True)


def overlay_crop(stem, bi):
    """Recorte con la mancha sospechosa resaltada en naranja."""
    im = imread_u(os.path.join(SRC, stem + ".png"))
    j, bs = load_boxes(os.path.join(SRC, stem + ".json"))
    b = bs[int(bi)]
    p = CFG["PAD"]
    H, W = im.shape[:2]
    crop = im[max(0, int(b["y1"]) - p):min(H, int(b["y2"]) + p),
              max(0, int(b["x1"]) - p):min(W, int(b["x2"]) + p)]
    m, d = analyze(crop)
    c = d["crop"].copy()
    br = d.get("brown")
    if br is not None and br.any():
        edge = cv2.dilate(br, np.ones((3, 3), np.uint8)) - br
        c[edge > 0] = (0, 140, 255)
    s = 200.0 / max(c.shape[:2])
    if s < 1:
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_AREA)
    imwrite_u(os.path.join(BR, f"{stem}_{bi}.png"), c)


def sheets(rows, prefix, per, cols, T, title, cap, crops_dir=None):
    import grid as G
    old = G.CROPS
    if crops_dir:
        G.CROPS = crops_dir
    for k in range(0, len(rows), per):
        ch = rows[k:k + per]
        items = [(r["stem"], r["idx"], f"{i+1:02d}) {r['stem']}#{r['idx']}", cap(r)) for i, r in enumerate(ch)]
        grid(items, os.path.join(AUD, f"{prefix}_{k//per:02d}.png"), cols=cols, T=T,
             title=f"{title} — hoja {k//per} ({len(ch)})")
    G.CROPS = old
    return (len(rows) + per - 1) // per


if __name__ == "__main__":
    E = list(csv.DictReader(open(os.path.join(WORK, "set_E.csv"), encoding="utf-8")))
    G_ = list(csv.DictReader(open(os.path.join(WORK, "set_G.csv"), encoding="utf-8")))
    F = list(csv.DictReader(open(os.path.join(WORK, "set_F.csv"), encoding="utf-8")))
    n = sheets(E + G_, "EG", 20, 5, 230,
               "E/G intensidad del rojo — asignar estado 0-4",
               lambda r: f"act={r['actual'][:9]} pct={r['pct']} granate={r['pct_darkred']}")
    print("hojas EG:", n)
    for r in F:
        overlay_crop(r["stem"], r["idx"])
    n = sheets(F, "F", 24, 6, 200,
               "F sanidad — marcar las que tienen dano/pudricion real (naranja = zona detectada)",
               lambda r: f"{r['actual'][:9]} mancha={r['brown_blob']}%", crops_dir=BR)
    print("hojas F:", n)
