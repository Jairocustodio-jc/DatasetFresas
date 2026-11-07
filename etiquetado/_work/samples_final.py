"""Hojas de muestra del dataset corregido: 24 frutos por estado."""
import sys, os, json, random, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, WORK, CLASSES, imread_u, imwrite_u
from grid import grid

OUT = os.path.join(BASE, "dataset_5estados")
INF = os.path.join(BASE, "informe")
ES = ["0 Verde inmadura", "1 Rosa temprano", "2 Comercial basico", "3 Comercial alto", "4 Sobremadura"]

random.seed(5)
by = {c: [] for c in range(5)}
stems = [os.path.splitext(f)[0] for f in os.listdir(OUT) if f.endswith(".json")]
random.shuffle(stems)
for s in stems:
    j = json.load(open(os.path.join(OUT, s + ".json"), encoding="utf-8"))
    for i, sh in enumerate(j["shapes"]):
        c = CLASSES.index(sh["label"])
        xs = [p[0] for p in sh["points"]]
        ys = [p[1] for p in sh["points"]]
        if min(max(xs) - min(xs), max(ys) - min(ys)) < 55:
            continue
        if len(by[c]) < 24:
            by[c].append((s, i))
    if all(len(v) >= 24 for v in by.values()):
        break

os.makedirs(INF, exist_ok=True)
for c in range(5):
    items = [(s, i, f"{s}#{i}", "") for s, i in by[c]]
    out = os.path.join(INF, f"muestras_estado_{c}.png")
    grid(items, out, cols=6, T=200, title=f"Estado {ES[c]} — muestras del dataset corregido")
    print(out, len(items))
