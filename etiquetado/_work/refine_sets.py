"""Conjuntos de afinado: criterios mas alla del % de color.

E  intensidad: etiquetadas 'overripe' pero de color palido (naranja/rosa claro)
G  intensidad: etiquetadas 'commercial-high' con mucho rojo granate -> quiza sobremadura
F  sanidad:    manchas oscuras no rojas (dano, pudricion, fruto momificado)
"""
import sys, os, csv, collections
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, WORK, CLASSES

NUM = ("pct", "pct_fruit", "pct_rej", "pct_deep", "pct_brown", "brown_blob", "pct_darkred",
       "pct_pale_col", "a_col", "a_col_std", "gloss", "a_p90", "surf_frac", "leaf_frac",
       "spec_frac", "dark_frac", "blur", "border_frac", "w_box", "h_box")


def load(path=None):
    rows = list(csv.DictReader(open(path or os.path.join(WORK, "features.csv"), encoding="utf-8")))
    for r in rows:
        for k in NUM:
            r[k] = float(r[k]) if r.get(k) not in ("", None) else 0.0
        r["lado"] = min(r["w_box"], r["h_box"])
    return rows


def final_labels():
    """Etiqueta vigente = la del dataset ya corregido."""
    import json
    out = {}
    OUT = os.path.join(BASE, "dataset_5estados")
    for f in os.listdir(OUT):
        if not f.endswith(".json"):
            continue
        j = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
        for i, sh in enumerate(j["shapes"]):
            out[(f[:-5], str(i))] = sh["label"]
    return out


def sets(rows, lab):
    E, G, F = [], [], []
    for r in rows:
        k = (r["stem"], r["idx"])
        cur = lab.get(k, r["label"])
        r["actual"] = cur
        if cur == "overripe" and r["pct"] >= 85 and (r["pct_pale_col"] >= 55 or r["a_col"] < 155):
            E.append(r)
        if cur == "commercial-high" and r["pct_darkred"] >= 20 and r["pct"] >= 80:
            G.append(r)
        if r["brown_blob"] >= 6 and r["lado"] >= 30:
            F.append(r)
    return E, G, F


if __name__ == "__main__":
    rows = load()
    lab = final_labels()
    E, G, F = sets(rows, lab)
    print(f"E palidas etiquetadas sobremadura : {len(E)}")
    print(f"G granates etiquetadas com.alto   : {len(G)}")
    print(f"F posible dano/pudricion          : {len(F)}")
    print("   F por etiqueta:", dict(collections.Counter(r["actual"] for r in F)))
    for name, S in (("E", E), ("G", G), ("F", F)):
        with open(os.path.join(WORK, f"set_{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "idx", "actual", "pct", "pct_deep", "pct_darkred", "pct_pale_col",
                        "a_col", "brown_blob", "lado"])
            for r in sorted(S, key=lambda r: -r["brown_blob"] if name == "F" else -r["pct"]):
                w.writerow([r["stem"], r["idx"], r["actual"], round(r["pct"], 1), round(r["pct_deep"], 1),
                            round(r["pct_darkred"], 1), round(r["pct_pale_col"], 1), round(r["a_col"], 1),
                            round(r["brown_blob"], 1), int(r["lado"])])
    print("escritos _work/set_E.csv, set_G.csv, set_F.csv")
