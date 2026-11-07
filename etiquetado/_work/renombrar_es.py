"""Renombra las etiquetas del dataset final al espanol, in situ.

Los indices YOLO (0-4) NO cambian, asi que los .txt no se tocan: solo el nombre
en los JSON, en classes.txt y en los CSV del informe.
"""
import os, json, csv, collections

BASE = r"D:\新草莓\DatasetId_360753_1652783343"
OUT = os.path.join(BASE, "dataset_5estados")
INF = os.path.join(BASE, "informe")

MAPA = {
    "unripe": "verde_inmadura",
    "early-pink": "rosa_temprano",
    "commercial-basic": "comercial_basico",
    "commercial-high": "comercial_alto",
    "overripe": "madurez_plena",
}
NUEVAS = ["verde_inmadura", "rosa_temprano", "comercial_basico", "comercial_alto", "madurez_plena"]


def main():
    stems = sorted((f[:-5] for f in os.listdir(OUT) if f.endswith(".json")),
                   key=lambda s: int(s) if s.isdigit() else 10**9)
    cambiadas = 0
    archivos = 0
    desconocidas = collections.Counter()
    for s in stems:
        p = os.path.join(OUT, s + ".json")
        j = json.load(open(p, encoding="utf-8"))
        toco = False
        for sh in j["shapes"]:
            lab = sh["label"]
            if lab in MAPA:
                sh["label"] = MAPA[lab]
                cambiadas += 1
                toco = True
            elif lab not in NUEVAS:
                desconocidas[lab] += 1
        if toco:
            json.dump(j, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            archivos += 1
    print(f"etiquetas renombradas: {cambiadas} en {archivos} archivos")
    if desconocidas:
        print("  OJO, etiquetas no reconocidas:", dict(desconocidas))

    # classes.txt (el orden define el indice YOLO, no se altera)
    cp = os.path.join(OUT, "yolo", "classes.txt")
    open(cp, "w", encoding="utf-8").write("\n".join(NUEVAS) + "\n")
    print("classes.txt:", ", ".join(f"{i}={c}" for i, c in enumerate(NUEVAS)))

    # atributos.csv: columna etiqueta
    ap = os.path.join(INF, "atributos.csv")
    filas = list(csv.DictReader(open(ap, encoding="utf-8-sig")))
    for r in filas:
        r["etiqueta"] = MAPA.get(r["etiqueta"], r["etiqueta"])
    with open(ap, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"atributos.csv actualizado ({len(filas)} filas)")

    # revisar.csv: columna etiqueta
    rp = os.path.join(INF, "revisar.csv")
    if os.path.exists(rp):
        rows = list(csv.DictReader(open(rp, encoding="utf-8-sig")))
        for r in rows:
            if "etiqueta" in r:
                r["etiqueta"] = MAPA.get(r["etiqueta"], r["etiqueta"])
        with open(rp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"revisar.csv actualizado ({len(rows)} filas)")

    # verificacion: nombres validos, indices YOLO intactos
    dist = collections.Counter()
    malos = 0
    for s in stems:
        j = json.load(open(os.path.join(OUT, s + ".json"), encoding="utf-8"))
        tp = os.path.join(OUT, "yolo", "labels", s + ".txt")
        idx = [int(l.split()[0]) for l in open(tp, encoding="utf-8") if l.strip()]
        if len(idx) != len(j["shapes"]):
            malos += 1
            continue
        for sh, k in zip(j["shapes"], idx):
            dist[sh["label"]] += 1
            if sh["label"] not in NUEVAS or NUEVAS.index(sh["label"]) != k:
                malos += 1
    print("distribucion final:", {c: dist[c] for c in NUEVAS}, "total", sum(dist.values()))
    print("desajustes JSON <-> YOLO:", malos)


if __name__ == "__main__":
    main()
