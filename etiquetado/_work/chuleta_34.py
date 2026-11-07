"""Chuleta visual para distinguir estado 3 (comercial alto) de estado 4 (sobremadura)."""
import sys, os, csv, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, WORK, SRC, imread_u, imwrite_u, load_boxes
from pigment import CFG

CROPS = os.path.join(WORK, "crops")
DOWN = r"C:\Users\jairo\Downloads"


def crop(stem, bi, T):
    p = os.path.join(CROPS, f"{stem}_{bi}.png")
    img = imread_u(p)
    if img is None:
        im = imread_u(os.path.join(SRC, stem + ".png"))
        j, bs = load_boxes(os.path.join(SRC, stem + ".json"))
        b = bs[bi]
        pd = CFG["PAD"]
        H, W = im.shape[:2]
        img = im[max(0, int(b["y1"]) - pd):min(H, int(b["y2"]) + pd),
                 max(0, int(b["x1"]) - pd):min(W, int(b["x2"]) + pd)]
    h, w = img.shape[:2]
    s = (T - 8) / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
    c = np.full((T, T, 3), 26, np.uint8)
    hh, ww = r.shape[:2]
    c[(T - hh) // 2:(T - hh) // 2 + hh, (T - ww) // 2:(T - ww) // 2 + ww] = r
    return c


def banda(texto, ancho, alto=42, color=(255, 255, 255), fondo=(18, 18, 18), esc=0.7):
    b = np.full((alto, ancho, 3), fondo, np.uint8)
    cv2.putText(b, texto, (12, int(alto * 0.68)), 0, esc, color, 2 if esc > 0.6 else 1, cv2.LINE_AA)
    return b


def fila(casos, T, pies=None):
    cells = []
    for k, (s, i) in enumerate(casos):
        c = crop(s, i, T)
        pie = np.full((26, T, 3), 26, np.uint8)
        if pies:
            cv2.putText(pie, pies[k], (6, 18), 0, 0.45, (150, 220, 255), 1, cv2.LINE_AA)
        cells.append(np.vstack([c, pie]))
    return np.hstack(cells)


def feats():
    d = {}
    for r in csv.DictReader(open(os.path.join(WORK, "features.csv"), encoding="utf-8")):
        d[(r["stem"], int(r["idx"]))] = r
    return d


if __name__ == "__main__":
    F = feats()
    T = 250
    tres = [("2930", 1), ("1848", 1), ("2287", 0), ("154", 3), ("2820", 0), ("3068", 0)]
    cuatro = [("337", 3), ("825", 1), ("95", 0), ("161", 1), ("2908", 0), ("2471", 2)]
    palidas = [("2173", 0), ("1572", 2), ("953", 0), ("1104", 1)]
    granates = [("1075", 2), ("2529", 0), ("1394", 1), ("63", 2)]

    def pies(casos):
        out = []
        for s, i in casos:
            r = F.get((s, i))
            out.append(f"{s}#{i}  color {float(r['pct']):.0f}%" if r else f"{s}#{i}")
        return out

    W = T * 6
    partes = [
        banda("ESTADO 3  Comercial alto  —  queda zona palida visible (75-90 % de color)", W, 50, (120, 255, 160)),
        fila(tres, T, pies(tres)),
        banda("Mira el hombro, junto al caliz: se ve crema/amarillo de un vistazo, sin buscarlo.", W, 34, (200, 200, 200), esc=0.55),
        banda("ESTADO 4  Sobremadura  —  rojo intenso hasta el caliz (>=90 %)", W, 50, (140, 160, 255)),
        fila(cuatro, T, pies(cuatro)),
        banda("El rojo toca el caliz en todo el contorno. Si tienes que buscar la zona palida, ya es 4.", W, 34, (200, 200, 200), esc=0.55),
        banda("TRAMPA A: cubierta del todo pero PALIDA (naranja/rosa claro)  ->  es 3, no 4", W, 46, (120, 220, 255), esc=0.62),
        np.hstack([fila(palidas, T, pies(palidas)), np.full((T + 26, T * 2, 3), 26, np.uint8)]),
        banda("TRAMPA B: rojo granate uniforme aunque la medicion baje por hojas  ->  es 4", W, 46, (120, 220, 255), esc=0.62),
        np.hstack([fila(granates, T, pies(granates)), np.full((T + 26, T * 2, 3), 26, np.uint8)]),
    ]
    img = np.vstack(partes)
    for d in (os.path.join(BASE, "informe"), DOWN):
        imwrite_u(os.path.join(d, "chuleta_estado_3_vs_4.png"), img)
    print("chuleta:", img.shape)

    # tira de transicion ordenada por cobertura medida
    cand = [r for r in F.values() if 70 <= float(r["pct"]) <= 100
            and min(float(r["w_box"]), float(r["h_box"])) >= 90
            and float(r["surf_frac"]) > 0.85 and float(r["leaf_frac"]) < 0.12]
    cand.sort(key=lambda r: float(r["pct"]))
    paso = max(1, len(cand) // 12)
    sel = cand[::paso][:12]
    T2 = 250
    filas = []
    for blk in (sel[:6], sel[6:12]):
        t = fila([(r["stem"], int(r["idx"])) for r in blk], T2,
                 [f"{float(r['pct']):.0f}%  -> estado {3 if float(r['pct']) < 90 else 4}" for r in blk])
        m = np.full((14, t.shape[1], 3), 26, np.uint8)
        for k, r in enumerate(blk):
            if float(r["pct"]) >= 90:
                cv2.line(m, (k * T2 + 4, 7), (t.shape[1] - 4, 7), (100, 140, 255), 4)
                break
        filas += [t, m]
    cab = banda("TIRA DE CALIBRACION  —  ordenadas por cobertura; la linea azul marca donde empieza el estado 4",
                T2 * 6, 46, (255, 255, 160), esc=0.6)
    out = np.vstack([cab] + filas)
    for d in (os.path.join(BASE, "informe"), DOWN):
        imwrite_u(os.path.join(d, "tira_calibracion_3_a_4.png"), out)
    print("tira:", out.shape)
