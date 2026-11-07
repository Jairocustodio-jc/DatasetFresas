"""Selecciona los bboxes cuya etiqueta actual es incompatible con la medicion."""
import sys, os, csv, collections, json, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import WORK, CLASSES

RANGES = {0: (0.0, 0.5), 1: (0.5, 50.0), 2: (50.0, 75.0), 3: (75.0, 90.0), 4: (90.0, 100.01)}


def dist_to_range(p, c):
    lo, hi = RANGES[c]
    if p < lo: return lo - p
    if p >= hi: return p - hi
    return 0.0


def cls_nominal(p):
    for c in range(5):
        lo, hi = RANGES[c]
        if lo <= p < hi: return c
    return 4


def main():
    rows = list(csv.DictReader(open(os.path.join(WORK, "features.csv"), encoding="utf-8")))
    for r in rows:
        for k in ("pct", "pct_fruit", "pct_rej", "pct_deep", "surf_frac", "leaf_frac", "spec_frac",
                  "dark_frac", "blur", "border_frac", "w_box", "h_box", "sep", "a_col"):
            r[k] = float(r[k]) if r[k] not in ("", None) else 0.0
        r["cur"] = CLASSES.index(r["label"])
        r["pred"] = cls_nominal(r["pct"])
        r["d"] = dist_to_range(r["pct"], r["cur"])
        r["small"] = min(r["w_box"], r["h_box"]) < 32
        # senales de medicion poco fiable
        r["riesgo"] = (r["leaf_frac"] > 0.25) + (r["surf_frac"] < 0.6) + (r["spec_frac"] > 0.12) + \
                      (r["blur"] < 60) + (r["small"]) + (r["pct_rej"] > 3) + (r["mode"] == "fallback-fruit")

    buckets = collections.Counter()
    for r in rows:
        if r["d"] == 0: buckets["consistente"] += 1
        elif r["d"] <= 6: buckets["margen<=6"] += 1
        elif r["d"] <= 15: buckets["6-15"] += 1
        else: buckets[">15"] += 1
    print("distancia de la medicion al rango de la etiqueta actual:", dict(buckets))

    # el limite 0/1 es categorico (hay pigmento o no), no un margen porcentual
    for r in rows:
        r["tipo"] = ""
        if r["cur"] >= 1 and r["pct"] <= 0.0:
            r["tipo"] = "A_sin_color"          # etiquetada con color pero no se detecta nada
        elif r["cur"] == 0 and r["pct"] >= 8:
            r["tipo"] = "B_con_color"          # etiquetada verde pero se detecta color
        elif r["d"] > 6:
            r["tipo"] = "C_frontera"
    cand = [r for r in rows if r["tipo"]]
    print("por tipo:", dict(collections.Counter(r["tipo"] for r in cand)))
    print("\ncandidatos (d>6):", len(cand))
    ch = collections.Counter((CLASSES[r["cur"]], CLASSES[r["pred"]]) for r in cand)
    for k, v in ch.most_common():
        print(f"   {k[0]:17s} -> {k[1]:17s} {v:5d}")
    print("\npor nivel de riesgo de medicion:", dict(collections.Counter(r["riesgo"] for r in cand)))
    lim = collections.Counter()
    for r in cand:
        lim["|d|>25" if r["d"] > 25 else ("15-25" if r["d"] > 15 else "6-15")] += 1
    print("magnitud de la discrepancia:", dict(lim))

    with open(os.path.join(WORK, "candidatos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "idx", "tipo", "actual", "medido", "pct", "d", "riesgo", "pct_deep", "leaf_frac",
                    "surf_frac", "spec_frac", "blur", "w_box", "h_box", "pct_rej", "mode"])
        for r in sorted(cand, key=lambda r: (r["tipo"], -r["d"])):
            w.writerow([r["stem"], r["idx"], r["tipo"], CLASSES[r["cur"]], CLASSES[r["pred"]], round(r["pct"], 1),
                        round(r["d"], 1), r["riesgo"], round(r["pct_deep"], 1), round(r["leaf_frac"], 2),
                        round(r["surf_frac"], 2), round(r["spec_frac"], 2), round(r["blur"], 0),
                        r["w_box"], r["h_box"], round(r["pct_rej"], 1), r["mode"]])
    print("\nescrito _work/candidatos.csv")


if __name__ == "__main__":
    main()
