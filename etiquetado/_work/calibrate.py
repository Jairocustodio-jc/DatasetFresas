"""Calibra los cortes de % de color contra las etiquetas visuales de referencia."""
import sys, os, csv, json, itertools, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import WORK, CLASSES

feats = {}
for r in csv.DictReader(open(os.path.join(WORK, "features.csv"), encoding="utf-8")):
    feats[(r["stem"], r["idx"])] = r

gt = []
for line in open(os.path.join(WORK, "gt_labels.txt"), encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    key = parts[0]
    stem, idx = key.split("#")
    cls = int(parts[1])
    border = len(parts) > 2 and parts[2] == "b"
    f = feats.get((stem, idx))
    if f is None:
        print("falta", key)
        continue
    gt.append(dict(stem=stem, idx=idx, cls=cls, b=border, pct=float(f["pct"]),
                   deep=float(f["pct_deep"]), label=f["label"], mode=f["mode"],
                   leaf=float(f["leaf_frac"]), surf=float(f["surf_frac"])))

print("n GT =", len(gt), " fronterizos =", sum(g["b"] for g in gt))


def classify(p, cuts):
    c0, c1, c2, c3 = cuts
    if p < c0: return 0
    if p < c1: return 1
    if p < c2: return 2
    if p < c3: return 3
    return 4


def score(cuts, subset):
    ex = sum(classify(g["pct"], cuts) == g["cls"] for g in subset)
    mae = sum(abs(classify(g["pct"], cuts) - g["cls"]) for g in subset)
    return ex / len(subset), mae / len(subset)


base = (0.5, 50, 75, 90)
firm = [g for g in gt if not g["b"]]
print("\ncortes nominales (0.5/50/75/90):")
for name, sub in (("todos", gt), ("no fronterizos", firm)):
    a, m = score(base, sub)
    print(f"   {name:16s} n={len(sub):3d} exacto={a*100:5.1f}%  MAE={m:.3f}")

best = None
for c0 in (0.3, 0.5, 1, 2, 3):
    for c1 in (38, 40, 42, 44, 46, 48, 50):
        for c2 in (60, 62, 64, 66, 68, 70, 72, 75):
            for c3 in (78, 80, 82, 84, 86, 88, 90):
                a, m = score((c0, c1, c2, c3), firm)
                if best is None or (a, -m) > (best[1], -best[2]):
                    best = ((c0, c1, c2, c3), a, m)
print("\nmejores cortes sobre casos NO fronterizos:", best[0], f"exacto={best[1]*100:.1f}% MAE={best[2]:.3f}")
a, m = score(best[0], gt)
print(f"   esos cortes sobre todos: exacto={a*100:.1f}% MAE={m:.3f}")

# matriz de confusion con cortes nominales y con los calibrados
for tag, cuts in (("nominales", base), ("calibrados", best[0])):
    M = np.zeros((5, 5), int)
    for g in gt:
        M[g["cls"], classify(g["pct"], cuts)] += 1
    print(f"\nconfusion ({tag} {cuts}) filas=visual, columnas=automatico")
    print("      " + "".join(f"{i:>6d}" for i in range(5)))
    for i in range(5):
        print(f"  {i}   " + "".join(f"{M[i,j]:6d}" for j in range(5)))

# errores concretos con cortes nominales
print("\ndesacuerdos (cortes nominales):")
for g in sorted(gt, key=lambda g: g["pct"]):
    p = classify(g["pct"], base)
    if p != g["cls"]:
        print(f"  {g['stem']}#{g['idx']:<3s} visual={g['cls']} auto={p} pct={g['pct']:5.1f} "
              f"deep={g['deep']:5.1f} leaf={g['leaf']:.2f} surf={g['surf']:.2f} {'FRONT' if g['b'] else ''} etiqueta_actual={g['label']}")

# sesgo: pct medido frente a rango teorico de la clase visual
rng = {0: (0, 0), 1: (0, 50), 2: (50, 75), 3: (75, 90), 4: (90, 100)}
print("\npct medido por clase visual:")
for c in range(5):
    v = np.array([g["pct"] for g in gt if g["cls"] == c])
    if len(v):
        print(f"  clase {c} n={len(v):3d}  p10={np.percentile(v,10):5.1f} mediana={np.median(v):5.1f} p90={np.percentile(v,90):5.1f}  (rango teorico {rng[c]})")
