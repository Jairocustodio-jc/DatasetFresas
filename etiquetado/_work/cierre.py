"""Cierre tecnico del dataset final.

Trabaja IN SITU sobre dataset_5estados (que ya incluye las correcciones hechas
desde el celular). No reconstruye nada desde data_anylabeling_2.

1. Recalcula las metricas de las cajas tocadas desde el celular (las de los
   archivos con .bak) y reescribe sus attributes/flags/difficult.
2. Verifica integridad: etiquetas validas, geometria dentro de la imagen,
   correspondencia JSON <-> YOLO caja por caja.
3. Regenera el export YOLO de los archivos que no coincidan.
4. Reescribe informe/atributos.csv con el estado final.
"""
import os, sys, json, glob, csv, collections
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, imread_u, imwrite_u
from pigment import analyze, CFG

OUT = os.path.join(BASE, "dataset_5estados")
INF = os.path.join(BASE, "informe")
CLASSES = ["verde_inmadura", "rosa_temprano", "comercial_basico", "comercial_alto", "madurez_plena"]


def metricas(im, sh):
    xs = [p[0] for p in sh["points"]]
    ys = [p[1] for p in sh["points"]]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    H, W = im.shape[:2]
    p = CFG["PAD"]
    a, b = max(0, int(round(x1)) - p), max(0, int(round(y1)) - p)
    c, d = min(W, int(round(x2)) + p), min(H, int(round(y2)) + p)
    if c - a < 4 or d - b < 4:
        return None
    m, _ = analyze(im[b:d, a:c])
    lado = int(min(x2 - x1, y2 - y1))
    flags = {
        "ocluido": bool(m["leaf_frac"] > 0.25 or m["surf_frac"] < 0.65),
        "borroso": bool(m["blur"] < 60),
        "reflejo_fuerte": bool(m["spec_frac"] > 0.12),
        "fruto_vecino_en_caja": bool(m["pct_rej"] > 3),
        "muy_pequeno": bool(lado < 32),
        "dano": bool(sh.get("flags", {}).get("dano", False)),
    }
    flags["dificil"] = bool(flags["ocluido"] or flags["borroso"] or flags["muy_pequeno"]
                            or m["surf_frac"] < 0.5)
    attrs = {
        "color_pct": round(m["pct"], 1),
        "rojo_intenso_pct": round(m["pct_deep"], 1),
        "rojo_granate_pct": round(m["pct_darkred"], 1),
        "color_palido_pct": round(m["pct_pale_col"], 1),
        "superficie_visible_pct": int(round(100 * m["surf_frac"])),
        "oclusion_pct": int(round(100 * (1 - m["surf_frac"]))),
        "lado_px": lado,
        "nitidez": round(m["blur"], 0),
    }
    return flags, attrs


def main():
    tocados = sorted(os.path.basename(b)[:-4] for b in glob.glob(os.path.join(OUT, "*.json.bak")))
    print(f"archivos corregidos desde el celular: {len(tocados)}")

    # --- 1. recalcular metricas de esas cajas ---
    recal = 0
    for n in tocados:
        jp = os.path.join(OUT, n)
        j = json.load(open(jp, encoding="utf-8"))
        im = imread_u(os.path.join(OUT, n[:-5] + ".png"))
        if im is None:
            print("  sin imagen:", n)
            continue
        for sh in j["shapes"]:
            r = metricas(im, sh)
            if r is None:
                continue
            fl, at = r
            sh.setdefault("flags", {}).update(fl)
            sh["attributes"] = at
            sh["difficult"] = fl["dificil"]
            recal += 1
        json.dump(j, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"cajas con metricas recalculadas: {recal}")

    # --- 2 y 3. verificar e igualar el export YOLO ---
    stems = sorted((f[:-5] for f in os.listdir(OUT) if f.endswith(".json")),
                   key=lambda s: int(s) if s.isdigit() else 10**9)
    problemas = []
    arreglados = 0
    nboxes = 0
    dist = collections.Counter()
    filas = []
    for s in stems:
        j = json.load(open(os.path.join(OUT, s + ".json"), encoding="utf-8"))
        W, H = j.get("imageWidth"), j.get("imageHeight")
        lineas = []
        for i, sh in enumerate(j["shapes"]):
            nboxes += 1
            lab = sh["label"]
            dist[lab] += 1
            if lab not in CLASSES:
                problemas.append(f"{s}#{i}: etiqueta desconocida '{lab}'")
                continue
            xs = [p[0] for p in sh["points"]]
            ys = [p[1] for p in sh["points"]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
                problemas.append(f"{s}#{i}: caja fuera de la imagen ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            if sh.get("shape_type") != "rectangle":
                problemas.append(f"{s}#{i}: shape_type '{sh.get('shape_type')}'")
            lineas.append(f"{CLASSES.index(lab)} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} "
                          f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            at = sh.get("attributes", {})
            fl = sh.get("flags", {})
            filas.append(dict(imagen=s + ".png", bbox=i, etiqueta=lab,
                              **{k: at.get(k, "") for k in ("color_pct", "rojo_intenso_pct",
                                 "rojo_granate_pct", "color_palido_pct", "superficie_visible_pct",
                                 "oclusion_pct", "lado_px", "nitidez")},
                              **{k: int(bool(fl.get(k))) for k in ("ocluido", "borroso",
                                 "reflejo_fuerte", "fruto_vecino_en_caja", "muy_pequeno",
                                 "dano", "dificil")}))
        tp = os.path.join(OUT, "yolo", "labels", s + ".txt")
        nuevo = "\n".join(lineas) + "\n"
        viejo = open(tp, encoding="utf-8").read() if os.path.exists(tp) else ""
        if viejo != nuevo:
            open(tp, "w", encoding="utf-8").write(nuevo)
            arreglados += 1
    print(f"cajas totales: {nboxes}")
    print("distribucion:", {c: dist[c] for c in CLASSES})
    print(f"TXT de YOLO regenerados por no coincidir: {arreglados}")
    print(f"problemas de integridad: {len(problemas)}")
    for p in problemas[:15]:
        print("   ", p)

    # --- 4. atributos.csv final ---
    with open(os.path.join(INF, "atributos.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    ndif = sum(r["dificil"] for r in filas)
    print(f"atributos.csv: {len(filas)} filas; dificiles {ndif} ({100*ndif/len(filas):.1f} %)")
    sinat = sum(1 for r in filas if r["color_pct"] == "")
    print("cajas sin metricas:", sinat)


if __name__ == "__main__":
    main()
