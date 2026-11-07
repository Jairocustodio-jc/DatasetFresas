"""Extraccion masiva de metricas cromaticas por bbox (paralela).
Salida: _work/features.csv  y  _work/crops/<stem>_<idx>.png (recorte 160 px)."""
import sys, os, csv, time, cv2, numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, WORK, SRC, imread_u, imwrite_u, load_boxes, stems
from pigment import analyze, CFG

CROPS = os.path.join(WORK, "crops")
FIELDS = ["stem", "idx", "label", "x1", "y1", "x2", "y2", "w_box", "h_box",
          "pct", "pct_fruit", "pct_rej", "ncomp", "pct_fix", "pct_deep", "thr", "sep", "mode", "how", "valid",
          "nsurf", "surf_frac", "leaf_frac", "spec_frac", "dark_frac", "blur", "border_frac",
          "a_p2", "a_p50", "a_p90", "a_p99", "a_col", "L_col", "L_mean", "S_mean", "b_mean",
          "pct_brown", "brown_blob", "pct_darkred", "pct_pale_col", "a_col_std", "gloss"]


def do_stem(stem, save_crops=True):
    jp = os.path.join(SRC, stem + ".json")
    ip = os.path.join(SRC, stem + ".png")
    im = imread_u(ip)
    if im is None:
        return []
    H, W = im.shape[:2]
    j, boxes = load_boxes(jp)
    rows = []
    for b in boxes:
        p = CFG["PAD"]
        x1, y1 = max(0, int(round(b["x1"])) - p), max(0, int(round(b["y1"])) - p)
        x2, y2 = min(W, int(round(b["x2"])) + p), min(H, int(round(b["y2"])) + p)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = im[y1:y2, x1:x2]
        m, d = analyze(crop)
        if save_crops:
            c = d["crop"]
            s = 160.0 / max(c.shape[:2])
            if s < 1:
                c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_AREA)
            imwrite_u(os.path.join(CROPS, f"{stem}_{b['idx']}.png"), c)
        r = dict(stem=stem, idx=b["idx"], label=b["label"],
                 x1=round(b["x1"], 2), y1=round(b["y1"], 2), x2=round(b["x2"], 2), y2=round(b["y2"], 2),
                 w_box=round(b["x2"] - b["x1"], 1), h_box=round(b["y2"] - b["y1"], 1))
        for k in FIELDS:
            if k in r:
                continue
            v = m.get(k, "")
            r[k] = round(v, 3) if isinstance(v, float) else v
        rows.append(r)
    return rows


def main():
    os.makedirs(CROPS, exist_ok=True)
    st = stems()
    t0 = time.time()
    out = []
    with Pool(8) as pool:
        for i, rows in enumerate(pool.imap_unordered(do_stem, st, chunksize=4)):
            out.extend(rows)
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"{i+1}/{len(st)} imgs  {len(out)} bboxes  {el:.0f}s  eta {el/(i+1)*(len(st)-i-1):.0f}s", flush=True)
    out.sort(key=lambda r: (int(r["stem"]) if r["stem"].isdigit() else 0, r["idx"]))
    with open(os.path.join(WORK, "features.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(out)
    print("total bboxes:", len(out), "tiempo", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
