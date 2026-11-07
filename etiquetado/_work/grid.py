"""Construye rejillas de recortes para inspeccion visual."""
import sys, os, csv, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import WORK, imread_u, imwrite_u

CROPS = os.path.join(WORK, "crops")


def load_feats():
    rows = list(csv.DictReader(open(os.path.join(WORK, "features.csv"), encoding="utf-8")))
    for r in rows:
        for k in ("pct", "pct_fix", "pct_deep", "sep", "surf_frac", "leaf_frac", "spec_frac",
                  "dark_frac", "a_p2", "a_p50", "a_p90", "a_p99", "a_col", "L_col", "L_mean",
                  "S_mean", "b_mean", "w_box", "h_box", "nsurf"):
            r[k] = float(r[k]) if r[k] not in ("", None) else 0.0
    return rows


def cell(stem, idx, caption, T=190, tag=""):
    p = os.path.join(CROPS, f"{stem}_{idx}.png")
    img = imread_u(p)
    if img is None:
        img = np.zeros((T, T, 3), np.uint8)
    h, w = img.shape[:2]
    s = (T - 6) / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
    c = np.full((T + 34, T, 3), 25, np.uint8)
    hh, ww = r.shape[:2]
    y0, x0 = 3 + (T - 6 - hh) // 2, (T - ww) // 2
    c[y0:y0 + hh, x0:x0 + ww] = r
    cv2.putText(c, tag, (4, T + 14), 0, 0.44, (255, 255, 255), 1)
    cv2.putText(c, caption, (4, T + 29), 0, 0.42, (140, 225, 255), 1)
    return c


def grid(items, out, cols=6, T=190, title=""):
    """items: lista de (stem, idx, tag, caption)"""
    cells = [cell(s, i, cap, T, tag) for s, i, tag, cap in items]
    rows = []
    for k in range(0, len(cells), cols):
        chunk = cells[k:k + cols]
        while len(chunk) < cols:
            chunk.append(np.full_like(cells[0], 25))
        rows.append(np.hstack(chunk))
    img = np.vstack(rows)
    if title:
        hdr = np.full((28, img.shape[1], 3), 10, np.uint8)
        cv2.putText(hdr, title, (6, 19), 0, 0.55, (255, 255, 120), 1)
        img = np.vstack([hdr, img])
    imwrite_u(out, img)
    return img.shape
