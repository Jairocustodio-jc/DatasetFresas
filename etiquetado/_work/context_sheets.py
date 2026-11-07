"""Hojas de contexto: caja dibujada sobre la imagen completa + recorte, para los casos dudosos."""
import sys, os, csv, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, WORK, SRC, imread_u, imwrite_u, load_boxes

INF = os.path.join(BASE, "informe")
AUD = os.path.join(WORK, "audit")


def panel(stem, bi, tag, T=300):
    im = imread_u(os.path.join(SRC, stem + ".png"))
    j, boxes = load_boxes(os.path.join(SRC, stem + ".json"))
    b = boxes[bi]
    H, W = im.shape[:2]
    x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
    cw, ch = x2 - x1, y2 - y1
    mx, my = int(cw * 0.9) + 20, int(ch * 0.9) + 20
    X1, Y1 = max(0, x1 - mx), max(0, y1 - my)
    X2, Y2 = min(W, x2 + mx), min(H, y2 + my)
    ctx = im[Y1:Y2, X1:X2].copy()
    cv2.rectangle(ctx, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 255, 255), 2)
    for k, ob in enumerate(boxes):
        if k == bi:
            continue
        ox1, oy1, ox2, oy2 = int(ob["x1"]) - X1, int(ob["y1"]) - Y1, int(ob["x2"]) - X1, int(ob["y2"]) - Y1
        if ox2 > 0 and oy2 > 0 and ox1 < ctx.shape[1] and oy1 < ctx.shape[0]:
            cv2.rectangle(ctx, (ox1, oy1), (ox2, oy2), (160, 160, 160), 1)

    def fit(img, t):
        h, w = img.shape[:2]
        s = t / max(h, w)
        r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
        c = np.full((t, t, 3), 35, np.uint8)
        hh, ww = r.shape[:2]
        c[(t - hh) // 2:(t - hh) // 2 + hh, (t - ww) // 2:(t - ww) // 2 + ww] = r
        return c

    strip = np.hstack([fit(ctx, T), fit(im[y1:y2, x1:x2], T)])
    txt = np.full((26, strip.shape[1], 3), 15, np.uint8)
    cv2.putText(txt, f"{stem}#{bi}  {tag}", (4, 18), 0, 0.5, (255, 255, 160), 1)
    return np.vstack([strip, txt])


if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(INF, "revisar.csv"), encoding="utf-8-sig")))
    ps = []
    for r in rows:
        stem = r["imagen"][:-4]
        ps.append(panel(stem, int(r["bbox"]), f"actual={r['etiqueta']} pct={r['pct_color']}"))
    for k in range(0, len(ps), 6):
        ch = ps[k:k + 6]
        rows_img = [np.hstack(ch[i:i + 2]) for i in range(0, len(ch), 2)]
        w = max(x.shape[1] for x in rows_img)
        rows_img = [np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in rows_img]
        out = os.path.join(AUD, f"R_{k//6:02d}.png")
        imwrite_u(out, np.vstack(rows_img))
        print(out)
