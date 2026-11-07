"""Panel de diagnostico (recorte | superficie | mascara de color) para casos concretos."""
import sys, os, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import SRC, WORK, imread_u, imwrite_u, load_boxes
from pigment import analyze, CFG


def panel(stem, bi, T=150, note=""):
    im = imread_u(os.path.join(SRC, stem + ".png"))
    j, boxes = load_boxes(os.path.join(SRC, stem + ".json"))
    b = boxes[bi]
    p = CFG["PAD"]
    H, W = im.shape[:2]
    crop = im[max(0, int(b["y1"]) - p):min(H, int(b["y2"]) + p),
              max(0, int(b["x1"]) - p):min(W, int(b["x2"]) + p)]
    m, d = analyze(crop)
    c = d["crop"]

    def fit(img):
        h, w = img.shape[:2]
        s = T / max(h, w)
        r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
        cv_ = np.full((T, T, 3), 40, np.uint8)
        hh, ww = r.shape[:2]
        cv_[(T - hh) // 2:(T - hh) // 2 + hh, (T - ww) // 2:(T - ww) // 2 + ww] = r
        return cv_

    seg = c.copy()
    seg[d["surface"] == 0] = (seg[d["surface"] == 0] * 0.18).astype(np.uint8)
    ov = c.copy()
    ov[d["colored"] > 0] = (0.35 * ov[d["colored"] > 0] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    rj = d.get("rejected")
    if rj is not None and rj.any():
        ov[rj > 0] = (0.35 * ov[rj > 0] + 0.65 * np.array([255, 255, 0])).astype(np.uint8)
    strip = np.hstack([fit(c), fit(seg), fit(ov)])
    txt = np.full((32, strip.shape[1], 3), 15, np.uint8)
    cv2.putText(txt, f"{stem}#{bi} {b['label'][:12]} {note}", (3, 13), 0, 0.38, (255, 255, 255), 1)
    cv2.putText(txt, f"pct={m['pct']:.0f} deep={m['pct_deep']:.0f} leaf={m['leaf_frac']:.2f} surf={m['surf_frac']:.2f}",
                (3, 28), 0, 0.36, (120, 220, 255), 1)
    return np.vstack([strip, txt])


def sheet(cases, out, cols=4, T=150):
    ps = [panel(s, i, T, n) for s, i, n in cases]
    rows = []
    for k in range(0, len(ps), cols):
        ch = ps[k:k + cols]
        while len(ch) < cols:
            ch.append(np.full_like(ps[0], 20))
        rows.append(np.hstack(ch))
    imwrite_u(out, np.vstack(rows))
    return out


if __name__ == "__main__":
    cases = [("3098", 0, ""), ("1591", 1, ""), ("2509", 4, ""), ("1493", 0, ""), ("2586", 4, ""),
             ("512", 0, ""), ("1768", 0, ""), ("284", 5, "contam"), ("1015", 4, "contam"),
             ("2087", 2, "contam"), ("1678", 1, "contam"), ("3041", 0, "verde")]
    old = [("1614", 2, "vis4"), ("1044", 1, "vis4"), ("677", 1, "vis4"), ("1784", 1, "vis4"),
             ("606", 0, "vis2"), ("2811", 0, "vis3"), ("522", 2, "vis4"), ("536", 1, "vis4"),
             ("1534", 1, "vis4"), ("2113", 0, "vis3"), ("249", 0, "vis1"), ("27", 1, "vis1"),
             ("1962", 0, "vis1"), ("2724", 1, "vis1"), ("882", 0, "vis0"), ("2391", 3, "vis0"),
             ("2192", 3, "vis0"), ("906", 0, "vis0"), ("2724", 4, "vis0"), ("1470", 5, "vis0"),
             ("317", 1, "vis0"), ("1182", 6, "vis1"), ("2343", 0, "vis3"), ("963", 0, "vis3")]
    print(sheet(cases, os.path.join(WORK, "diag_spill.png"), cols=4, T=170))
