"""Verifica que el dataset corregido solo cambia etiquetas (nunca geometria)."""
import sys, os, json, csv, collections
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, SRC, CLASSES

OUT = os.path.join(BASE, "dataset_5estados")


def boxes(path):
    j = json.load(open(path, encoding="utf-8"))
    out = []
    for s in j["shapes"]:
        xs = [round(p[0], 3) for p in s["points"]]
        ys = [round(p[1], 3) for p in s["points"]]
        out.append((s["label"], tuple(sorted(xs)), tuple(sorted(ys)), s.get("shape_type")))
    return out


def main():
    src = sorted(f for f in os.listdir(SRC) if f.endswith(".json"))
    dst = sorted(f for f in os.listdir(OUT) if f.endswith(".json"))
    assert src == dst, f"faltan JSON: {set(src) ^ set(dst)}"
    nb = nchg = 0
    bad = []
    dist = collections.Counter()
    for f in src:
        a, b = boxes(os.path.join(SRC, f)), boxes(os.path.join(OUT, f))
        if len(a) != len(b):
            bad.append((f, "numero de cajas distinto"))
            continue
        for (la, xa, ya, ta), (lb, xb, yb, tb) in zip(a, b):
            nb += 1
            dist[lb] += 1
            if (xa, ya, ta) != (xb, yb, tb):
                bad.append((f, "geometria alterada"))
            if lb not in CLASSES:
                bad.append((f, f"etiqueta desconocida {lb}"))
            if la != lb:
                nchg += 1
    print(f"JSON verificados: {len(src)}   cajas: {nb}   etiquetas cambiadas: {nchg}")
    print("geometria alterada / errores:", len(bad))
    for x in bad[:10]:
        print("   ", x)
    print("distribucion final:", {c: dist[c] for c in CLASSES})
    # imagenes y yolo
    imgs = [f for f in os.listdir(OUT) if f.endswith(".png")]
    yl = os.listdir(os.path.join(OUT, "yolo", "labels"))
    yi = os.listdir(os.path.join(OUT, "yolo", "images"))
    print(f"imagenes copiadas: {len(imgs)}   etiquetas YOLO: {len(yl)}   imagenes YOLO: {len(yi)}")
    nboxes_yolo = 0
    for t in yl:
        for line in open(os.path.join(OUT, "yolo", "labels", t), encoding="utf-8"):
            if line.strip():
                nboxes_yolo += 1
                p = line.split()
                assert len(p) == 5 and 0 <= int(p[0]) <= 4, f"linea YOLO invalida en {t}: {line}"
                for v in p[1:]:
                    assert -0.001 <= float(v) <= 1.001, f"coordenada fuera de rango en {t}: {line}"
    print("cajas en YOLO:", nboxes_yolo)
    # atributos escritos en los JSON
    import collections as C
    cnt = C.Counter(); nat = 0
    for f in src:
        j = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
        for sh in j["shapes"]:
            if "attributes" in sh and "color_pct" in sh.get("attributes", {}):
                nat += 1
            for k, v in sh.get("flags", {}).items():
                if v is True:
                    cnt[k] += 1
            if sh.get("difficult"):
                cnt["difficult"] += 1
    print("cajas con atributos:", nat)
    print("banderas activas:", dict(cnt))


if __name__ == "__main__":
    main()
