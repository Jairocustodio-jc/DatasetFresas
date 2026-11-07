"""Auditoria de la frontera del 75 %: las cajas que deciden el precio.

Selecciona las cajas medidas entre 70 y 80 % de cobertura (comercial_basico pegadas
al corte por arriba y comercial_alto pegadas por abajo) y arma hojas de contacto
CIEGAS: el recorte no muestra el porcentaje medido, para que el juicio visual sea
independiente de la medicion. El porcentaje queda en el CSV, para contrastar despues.
"""
import os, sys, csv, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, imread_u, imwrite_u
from grid import grid
import grid as G

OUT = os.path.join(BASE, "dataset_5estados")
DEST = os.path.join(BASE, "informe", "auditoria_frontera75")
CROPS = os.path.join(DEST, "recortes")
PAD = 4


def seleccionar():
    filas = list(csv.DictReader(open(os.path.join(BASE, "informe", "atributos.csv"), encoding="utf-8-sig")))
    sel = []
    for r in filas:
        try:
            p = float(r["color_pct"])
        except (TypeError, ValueError):
            continue
        lado = int(float(r["lado_px"] or 0))
        if r["etiqueta"] == "comercial_basico" and 70 <= p < 75:
            sel.append(dict(r, pct=p, lado=lado, grupo="basico_70_75"))
        elif r["etiqueta"] == "comercial_alto" and 75 <= p < 80:
            sel.append(dict(r, pct=p, lado=lado, grupo="alto_75_80"))
    sel.sort(key=lambda r: r["pct"])
    return sel


def recortar(sel):
    os.makedirs(CROPS, exist_ok=True)
    import json
    cache = {}
    hechos = 0
    for r in sel:
        stem = r["imagen"][:-4]
        if stem not in cache:
            j = json.load(open(os.path.join(OUT, stem + ".json"), encoding="utf-8"))
            cache[stem] = (imread_u(os.path.join(OUT, stem + ".png")), j)
        im, j = cache[stem]
        if im is None:
            continue
        sh = j["shapes"][int(r["bbox"])]
        xs = [p[0] for p in sh["points"]]
        ys = [p[1] for p in sh["points"]]
        H, W = im.shape[:2]
        x1, y1 = max(0, int(min(xs)) - PAD), max(0, int(min(ys)) - PAD)
        x2, y2 = min(W, int(max(xs)) + PAD), min(H, int(max(ys)) + PAD)
        c = im[y1:y2, x1:x2]
        if c.size == 0:
            continue
        s = 240.0 / max(c.shape[:2])
        if s < 1:
            c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_AREA)
        imwrite_u(os.path.join(CROPS, f"{stem}_{r['bbox']}.png"), c)
        hechos += 1
        if len(cache) > 40:
            cache.clear()
    return hechos


def hojas(sel, per=24, cols=6, T=220):
    G.CROPS = CROPS
    n = 0
    for k in range(0, len(sel), per):
        ch = sel[k:k + per]
        items = [(r["imagen"][:-4], r["bbox"], f"{i+1:02d}) {r['imagen'][:-4]}#{r['bbox']}", "")
                 for i, r in enumerate(ch)]
        grid(items, os.path.join(DEST, f"F75_{k//per:02d}.png"), cols=cols, T=T,
             title=f"Frontera 75 % — hoja {k//per} ({len(ch)} frutos) — "
                   f"marcar 2 (menos del 75 % coloreado) o 3 (75 % o mas)")
        n += 1
    return n


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    sel = seleccionar()
    b = sum(1 for r in sel if r["grupo"] == "basico_70_75")
    a = sum(1 for r in sel if r["grupo"] == "alto_75_80")
    print(f"cajas en la franja 70-80 %: {len(sel)}  ({b} etiquetadas comercial_basico, {a} comercial_alto)")
    print("recortes generados:", recortar(sel))
    print("hojas de contacto:", hojas(sel))
    with open(os.path.join(DEST, "frontera75.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["hoja", "celda", "imagen", "bbox", "etiqueta_actual", "color_pct_medido",
                    "lado_px", "superficie_visible_pct", "dificil", "decision_visual"])
        for i, r in enumerate(sel):
            w.writerow([f"F75_{i//24:02d}", i % 24 + 1, r["imagen"], r["bbox"], r["etiqueta"],
                        round(r["pct"], 1), r["lado"], r["superficie_visible_pct"], r["dificil"], ""])
    print("escrito", os.path.join(DEST, "frontera75.csv"))
