"""Aplica las correcciones al dataset de trabajo y genera:
  dataset_5estados/          JSON AnyLabeling corregidos + imagenes
  dataset_5estados/yolo/     etiquetas YOLO + classes.txt
  informe/cambios.csv        cada cambio con su motivo
  informe/revisar.csv        casos dudosos que conviene mirar a mano
  informe/revisar/*.png      recortes de esos casos
"""
import sys, os, csv, json, shutil, collections
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import BASE, WORK, SRC, CLASSES, imread_u, imwrite_u

OUT = os.path.join(BASE, "dataset_5estados")
INF = os.path.join(BASE, "informe")
CROPS = os.path.join(WORK, "crops")
ES = {0: "Verde_inmadura", 1: "Rosa_temprano", 2: "Comercial_basico", 3: "Comercial_alto", 4: "Sobremadura"}


def load_feats():
    f = {}
    for r in csv.DictReader(open(os.path.join(WORK, "features.csv"), encoding="utf-8")):
        for k in ("pct", "pct_fruit", "pct_rej", "pct_deep", "a_p90", "a_p99", "surf_frac", "leaf_frac",
                  "spec_frac", "dark_frac", "blur", "w_box", "h_box", "border_frac",
                  "pct_darkred", "pct_pale_col", "pct_brown", "brown_blob", "a_col", "a_col_std", "gloss"):
            r[k] = float(r[k]) if r[k] not in ("", None) else 0.0
        f[(r["stem"], int(r["idx"]))] = r
    return f


def load_audit_A():
    keep = set()
    for line in open(os.path.join(WORK, "audit_A_result.txt"), encoding="utf-8"):
        line = line.split("#c")[0].strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()[0]
        if "#" in tok:
            s, i = tok.split("#")
            keep.add((s, int(i)))
    return keep


def load_audit_C():
    out = {}
    for line in open(os.path.join(WORK, "audit_C_result.txt"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if "#" not in p[0]:
            continue
        s, i = p[0].split("#")
        out[(s, int(i))] = p[1]
    return out


def load_audit_D():
    """Frutos del grupo D que SI conservan color (revision visual)."""
    keep = set()
    for line in open(os.path.join(WORK, "audit_D_result.txt"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s, i = line.split()[0].split("#")
        keep.add((s, int(i)))
    return keep


def load_audit_EG():
    """Afinado por intensidad del rojo: clase nueva por caja."""
    out = {}
    for line in open(os.path.join(WORK, "audit_EG_result.txt"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = line.split()[0], line.split()[1]
        s_, i_ = k.split("#")
        out[(s_, int(i_))] = int(v)
    return out


def load_danos():
    """Cajas con dano/pudricion confirmado visualmente."""
    out = set()
    for line in open(os.path.join(WORK, "audit_F_result.txt"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s_, i_ = line.split()[0].split("#")
        out.add((s_, int(i_)))
    return out


def atributos(r, danos, key):
    """Atributos objetivos por caja (fiabilidad de la medicion y sanidad)."""
    lado = int(min(r["w_box"], r["h_box"]))
    ocl = 1.0 - r["surf_frac"]
    flags = {
        "ocluido": bool(r["leaf_frac"] > 0.25 or r["surf_frac"] < 0.65),
        "borroso": bool(r["blur"] < 60),
        "reflejo_fuerte": bool(r["spec_frac"] > 0.12),
        "fruto_vecino_en_caja": bool(r["pct_rej"] > 3),
        "muy_pequeno": bool(lado < 32),
        "dano": bool(key in danos),
    }
    flags["dificil"] = bool(flags["ocluido"] or flags["borroso"] or flags["muy_pequeno"]
                            or r["surf_frac"] < 0.5)
    attrs = {
        "color_pct": round(r["pct"], 1),
        "rojo_intenso_pct": round(r["pct_deep"], 1),
        "rojo_granate_pct": round(r["pct_darkred"], 1),
        "color_palido_pct": round(r["pct_pale_col"], 1),
        "superficie_visible_pct": int(round(100 * r["surf_frac"])),
        "oclusion_pct": int(round(100 * ocl)),
        "lado_px": lado,
        "nitidez": round(r["blur"], 0),
    }
    return flags, attrs


def load_audit_R():
    """Resolucion manual de los casos dudosos: clase, 'ok' o 'flag'."""
    out = {}
    for line in open(os.path.join(WORK, "audit_R_result.txt"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if "#" not in p[0]:
            continue
        s_, i_ = p[0].split("#")
        out[(s_, int(i_))] = p[1]
    return out


def grupoD_keys(feats, decision):
    """Cajas que tras la 1a pasada quedan como early-pink con <12 % de color."""
    out = set()
    for key, r in feats.items():
        nueva = decision.get(key, (None,))[0]
        lab = CLASSES.index(r["label"]) if nueva is None else nueva
        if lab == 1 and r["pct"] < 12:
            out.add(key)
    return out


def main():
    feats = load_feats()
    keepA = load_audit_A()
    keepD = load_audit_D()
    audC = load_audit_C()
    cand = {}
    for r in csv.DictReader(open(os.path.join(WORK, "candidatos.csv"), encoding="utf-8")):
        cand[(r["stem"], int(r["idx"]))] = r

    decision = {}      # key -> (nueva_clase|None, motivo, revisar?)
    for key, r in feats.items():
        cur = CLASSES.index(r["label"])
        c = cand.get(key)
        if c is None:
            decision[key] = (None, "coherente", False)
            continue
        tipo = c["tipo"]
        if tipo == "A_sin_color":
            protegido = (key in keepA) or r["a_p90"] >= 132 or int(c["riesgo"]) >= 2 or \
                        min(r["w_box"], r["h_box"]) < 32
            if protegido:
                decision[key] = (None, "A_dudoso_revisar", True)
            else:
                decision[key] = (0, "A_sin_pigmento_visible", False)
        else:
            v = audC.get(key)
            if v is None:
                decision[key] = (None, "C_sin_auditar", True)
            elif v == "?":
                decision[key] = (None, "C_ambiguo_revisar", True)
            else:
                nv = int(v)
                decision[key] = ((nv if nv != cur else None), "C_revision_visual", False)

    # --- segunda pasada: limite 0/1 revisado a ojo (grupo D) ---
    for key in grupoD_keys(feats, decision):
        cur = CLASSES.index(feats[key]["label"])
        if key in keepD:
            decision[key] = (None if cur == 1 else 1, "D_rubor_confirmado", False)
        else:
            decision[key] = (0 if cur != 0 else None, "D_sin_pigmento_visible", False)

    # --- tercera pasada: casos dudosos resueltos con la caja en contexto ---
    for key, v in load_audit_R().items():
        if key not in feats:
            continue
        cur = CLASSES.index(feats[key]["label"])
        if v == "ok":
            decision[key] = (None, "R_etiqueta_confirmada", False)
        elif v == "flag":
            decision[key] = (None, "R_caja_dudosa_revisar", True)
        else:
            nv = int(v)
            decision[key] = ((nv if nv != cur else None), "R_revision_en_contexto", False)

    # --- cuarta pasada: intensidad y uniformidad del rojo (limite 3/4) ---
    for key, nv in load_audit_EG().items():
        if key not in feats:
            continue
        cur = CLASSES.index(feats[key]["label"])
        prev = decision.get(key, (None, "", False))[0]
        base = cur if prev is None else prev
        decision[key] = ((nv if nv != base else None), "EG_intensidad_del_rojo", False)

    # --- quinta pasada: fronterizos nuevos tras la remedicion reproducible ---
    for line in open(os.path.join(WORK, "audit_H_result.txt"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        s_, i_ = tok[0].split("#")
        key = (s_, int(i_))
        if key not in feats:
            continue
        cur = CLASSES.index(feats[key]["label"])
        if tok[1] == "ok":
            decision[key] = (None, "H_etiqueta_confirmada", False)
        else:
            nv = int(tok[1])
            decision[key] = ((nv if nv != cur else None), "H_revision_visual", False)

    danos = load_danos()

    # --- escribir dataset corregido ---
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "yolo", "labels"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "yolo", "images"), exist_ok=True)
    os.makedirs(os.path.join(INF, "revisar"), exist_ok=True)

    cambios, revisar, atrows = [], [], []
    dist_old, dist_new = collections.Counter(), collections.Counter()
    stems = sorted([os.path.splitext(f)[0] for f in os.listdir(SRC) if f.endswith(".json")],
                   key=lambda s: int(s) if s.isdigit() else 10**9)
    for stem in stems:
        jp = os.path.join(SRC, stem + ".json")
        j = json.load(open(jp, encoding="utf-8"))
        ip = os.path.join(SRC, stem + ".png")
        im = imread_u(ip)
        H, W = (im.shape[:2] if im is not None else (756, 1008))
        ylines = []
        for i, sh in enumerate(j["shapes"]):
            key = (stem, i)
            cur = CLASSES.index(sh["label"]) if sh["label"] in CLASSES else 0
            new, motivo, rev = decision.get(key, (None, "sin_medicion", True))
            dist_old[cur] += 1
            fin = cur if new is None else new
            dist_new[fin] += 1
            if new is not None:
                sh["label"] = CLASSES[new]
                f = feats.get(key, {})
                cambios.append(dict(imagen=stem + ".png", bbox=i, antes=CLASSES[cur], despues=CLASSES[new],
                                    antes_id=cur, despues_id=new, motivo=motivo,
                                    pct_color=round(f.get("pct", 0), 1), pct_rojo_intenso=round(f.get("pct_deep", 0), 1)))
            if rev:
                f = feats.get(key, {})
                revisar.append(dict(imagen=stem + ".png", bbox=i, etiqueta=CLASSES[cur], motivo=motivo,
                                    pct_color=round(f.get("pct", 0), 1), a_p90=f.get("a_p90", 0),
                                    lado_px=int(min(f.get("w_box", 0), f.get("h_box", 0))),
                                    recorte=f"{stem}_{i}.png"))
                src_crop = os.path.join(CROPS, f"{stem}_{i}.png")
                if os.path.exists(src_crop):
                    shutil.copyfile(src_crop, os.path.join(INF, "revisar", f"{stem}_{i}.png"))
            fr = feats.get(key)
            if fr is not None:
                fl, at = atributos(fr, danos, key)
                sh.setdefault("flags", {}).update(fl)
                sh["attributes"] = {**sh.get("attributes", {}), **at}
                sh["difficult"] = fl["dificil"]
                atrows.append(dict(imagen=stem + ".png", bbox=i, etiqueta=CLASSES[fin],
                                   **at, **{k: int(v) for k, v in fl.items()}))
            xs = [p[0] for p in sh["points"]]
            ys = [p[1] for p in sh["points"]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            ylines.append(f"{fin} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        json.dump(j, open(os.path.join(OUT, stem + ".json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        if im is not None:
            imwrite_u(os.path.join(OUT, stem + ".png"), im)
            imwrite_u(os.path.join(OUT, "yolo", "images", stem + ".png"), im)
        open(os.path.join(OUT, "yolo", "labels", stem + ".txt"), "w", encoding="utf-8").write("\n".join(ylines) + "\n")

    open(os.path.join(OUT, "yolo", "classes.txt"), "w", encoding="utf-8").write("\n".join(CLASSES) + "\n")
    with open(os.path.join(INF, "cambios.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(cambios[0].keys()) if cambios else
                           ["imagen", "bbox", "antes", "despues", "antes_id", "despues_id", "motivo", "pct_color", "pct_rojo_intenso"])
        w.writeheader()
        w.writerows(cambios)
    with open(os.path.join(INF, "revisar.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(revisar[0].keys()) if revisar else
                           ["imagen", "bbox", "etiqueta", "motivo", "pct_color", "a_p90", "lado_px", "recorte"])
        w.writeheader()
        w.writerows(revisar)

    with open(os.path.join(INF, "atributos.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(atrows[0].keys()))
        w.writeheader()
        w.writerows(atrows)
    nd = sum(r["dificil"] for r in atrows)
    print(f"atributos escritos para {len(atrows)} cajas; marcadas dificiles: {nd} ({100*nd/len(atrows):.1f}%)")
    print("con dano confirmado:", sum(r["dano"] for r in atrows))
    print("cambios aplicados:", len(cambios))
    print("por motivo:", dict(collections.Counter(c["motivo"] for c in cambios)))
    print("matriz de cambios:")
    for k, v in collections.Counter((c["antes"], c["despues"]) for c in cambios).most_common():
        print(f"   {k[0]:17s} -> {k[1]:17s} {v:5d}")
    print("casos marcados para revision manual:", len(revisar))
    print("distribucion antes:", {CLASSES[k]: v for k, v in sorted(dist_old.items())})
    print("distribucion despues:", {CLASSES[k]: v for k, v in sorted(dist_new.items())})


if __name__ == "__main__":
    main()
