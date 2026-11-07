"""Genera las hojas de auditoria visual y su manifiesto."""
import sys, os, csv, json
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from grid import grid
from common import WORK

AUD = os.path.join(WORK, "audit")
os.makedirs(AUD, exist_ok=True)


def load_cand():
    return list(csv.DictReader(open(os.path.join(WORK, "candidatos.csv"), encoding="utf-8")))


def sheets(rows, prefix, per, cols, T, title, caption=lambda r: ""):
    man = {}
    for k in range(0, len(rows), per):
        chunk = rows[k:k + per]
        n = k // per
        items = [(r["stem"], r["idx"], f"{i+1:02d}) {r['stem']}#{r['idx']}", caption(r))
                 for i, r in enumerate(chunk)]
        out = os.path.join(AUD, f"{prefix}_{n:02d}.png")
        grid(items, out, cols=cols, T=T, title=f"{title} — hoja {n} ({len(chunk)} frutos)")
        man[f"{prefix}_{n:02d}"] = [f"{r['stem']}#{r['idx']}" for r in chunk]
    return man


if __name__ == "__main__":
    cand = load_cand()
    A = [r for r in cand if r["tipo"] == "A_sin_color"]
    C = [r for r in cand if r["tipo"] in ("C_frontera", "B_con_color")]
    # A: primero los mas sospechosos de tener color no detectado
    A.sort(key=lambda r: (-float(r["pct_rej"]), -float(r["pct_fruit"]) if "pct_fruit" in r else 0.0, r["stem"]))
    C.sort(key=lambda r: (r["actual"], -float(r["d"])))
    man = {}
    man.update(sheets(A, "A", 35, 7, 170,
                      "A: etiquetadas con color pero medidas 0% — marcar las que SI tienen rosa/rojo"))
    man.update(sheets(C, "C", 20, 5, 230,
                      "C: frontera — asignar estado 0-4",
                      caption=lambda r: f"act={r['actual'][:9]} pct={r['pct']}"))
    json.dump(man, open(os.path.join(WORK, "audit_manifest.json"), "w", encoding="utf-8"), indent=0)
    print("hojas A:", sum(1 for k in man if k.startswith("A")), " hojas C:", sum(1 for k in man if k.startswith("C")))
