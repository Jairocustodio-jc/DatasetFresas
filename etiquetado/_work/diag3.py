import sys, os, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import *
from pigment import analyze, CFG


def get_crop(stem, bi, src=None):
    src = src or SRC
    im = imread_u(os.path.join(src, stem + ".png"))
    j, boxes = load_boxes(os.path.join(src, stem + ".json"))
    b = boxes[bi]
    p = CFG["PAD"]
    H, W = im.shape[:2]
    return im[max(0, int(b["y1"]) - p):min(H, int(b["y2"]) + p),
              max(0, int(b["x1"]) - p):min(W, int(b["x2"]) + p)], b["label"]


def fit(img, T):
    h, w = img.shape[:2]
    s = T / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    c = np.full((T, T, 3), 40, np.uint8)
    hh, ww = r.shape[:2]
    c[(T - hh) // 2:(T - hh) // 2 + hh, (T - ww) // 2:(T - ww) // 2 + ww] = r
    return c


def panel(stem, bi, T=118):
    crop, lab = get_crop(stem, bi)
    m, d = analyze(crop)
    c = d["crop"]
    seg = c.copy()
    seg[d["surface"] == 0] = (seg[d["surface"] == 0] * 0.18).astype(np.uint8)
    ov = c.copy()
    ov[d["colored"] > 0] = (0.35 * ov[d["colored"] > 0] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    strip = np.hstack([fit(c, T), fit(seg, T), fit(ov, T)])
    txt = np.full((30, strip.shape[1], 3), 15, np.uint8)
    cv2.putText(txt, f"{stem}#{bi} {lab[:12]}", (3, 12), 0, 0.35, (255, 255, 255), 1)
    cv2.putText(txt, f"pct={m['pct']:.0f} thr={m['thr']} {m['mode'][:4]} sep={m['sep']:.2f} dp={m['pct_deep']:.0f}",
                (3, 26), 0, 0.33, (120, 220, 255), 1)
    return np.vstack([strip, txt])


if __name__ == "__main__":
    sel = [("1539", 0), ("916", 0), ("2015", 0), ("262", 0), ("2091", 0), ("435", 6), ("198", 2), ("209", 0),
           ("1092", 0), ("1567", 1), ("2015", 1), ("2015", 2), ("1092", 6), ("1917", 2), ("1092", 1),
           ("1092", 2), ("1567", 0), ("1539", 1), ("2015", 3), ("2015", 4)]
    ps = [panel(*s) for s in sel]
    rows = [np.hstack(ps[k:k + 5]) for k in range(0, len(ps), 5)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    imwrite_u(os.path.join(WORK, "diag3.png"), np.vstack(rows))
    print("ok")
