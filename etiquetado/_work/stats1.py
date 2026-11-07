import sys, os, csv, collections, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import WORK, CLASSES

rows = list(csv.DictReader(open(os.path.join(WORK, "features.csv"), encoding="utf-8")))
print("n =", len(rows))
for r in rows:
    for k in ("pct", "pct_fix", "pct_deep", "sep", "surf_frac", "leaf_frac", "spec_frac", "dark_frac",
              "a_p50", "a_p90", "a_p99", "a_col", "L_col", "w_box", "h_box", "nsurf"):
        r[k] = float(r[k]) if r[k] not in ("", None) else 0.0

print("\nsegmentacion:", collections.Counter(r["how"] for r in rows))
print("modo:", collections.Counter(r["mode"] for r in rows))
print("valid:", collections.Counter(r["valid"] for r in rows))
sz = np.array([min(r["w_box"], r["h_box"]) for r in rows])
print("lado menor bbox: p5=%.0f p25=%.0f p50=%.0f p75=%.0f p95=%.0f  <20px: %d  <30px: %d" %
      (*np.percentile(sz, [5, 25, 50, 75, 95]), (sz < 20).sum(), (sz < 30).sum()))


def cls_of(p):
    if p < 0.5: return 0
    if p < 50: return 1
    if p < 75: return 2
    if p < 90: return 3
    return 4


pred = [cls_of(r["pct"]) for r in rows]
cur = [CLASSES.index(r["label"]) for r in rows]
print("\ndistribucion actual :", collections.Counter(cur))
print("distribucion pct    :", collections.Counter(pred))
M = np.zeros((5, 5), int)
for c, p in zip(cur, pred):
    M[c, p] += 1
print("\nmatriz  filas=etiqueta actual, columnas=clase por %color")
print("        " + "".join(f"{i:>8d}" for i in range(5)))
for i in range(5):
    print(f"{i} {CLASSES[i][:14]:15s}" + "".join(f"{M[i,j]:8d}" for j in range(5)))
agree = np.trace(M) / len(rows)
print(f"\nacuerdo exacto = {agree*100:.1f}%   |dif|<=1 = {sum(M[i,j] for i in range(5) for j in range(5) if abs(i-j)<=1)/len(rows)*100:.1f}%")

print("\npct por etiqueta actual (p10/p50/p90):")
for i, c in enumerate(CLASSES):
    v = np.array([r["pct"] for r, cc in zip(rows, cur) if cc == i])
    print(f"  {c:17s} n={len(v):6d}  {np.percentile(v,10):6.1f} {np.percentile(v,50):6.1f} {np.percentile(v,90):6.1f}")

# cuantos caen cerca de cada frontera
for cut in (0.5, 50, 75, 90):
    near = sum(1 for r in rows if abs(r["pct"] - cut) <= 5)
    print(f"a +-5 puntos del corte {cut}: {near}")
