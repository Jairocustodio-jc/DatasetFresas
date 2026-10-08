"""Reentrena YOLO con los lotes ya revisados y escribe sugerencias en el siguiente lote. Pensado para Kaggle (GPU).

En una celda de un notebook de Kaggle (Settings → Accelerator: GPU, Internet: On):

    !pip -q install ultralytics
    !git clone -q -b claude/jolly-heisenberg-qa6cj9 https://github.com/Jairocustodio-jc/DatasetFresas
    %cd DatasetFresas
    !python modelos/kaggle_fresas.py --siguiente restantes

Toma como «revisados» todos los lotes que tengan revisión en feedback/revision_<lote>.json o en la rama
«resultados» (lo que sincroniza la página). Al terminar deja en /kaggle/working/salida/:
    revision/<lote>/index.html        (la página de cada lote con sus sugerencias)
    modelos/yolo_fresas_<N>lotes.pt   (el modelo nuevo)
Cópialos al repo en tu PC (misma ruta) y haz git push. Con --push lo sube solo, si guardas tu token de GitHub en
Kaggle → Add-ons → Secrets con el nombre GITHUB_TOKEN o key_gh_strawberry (otro nombre: --secreto NOMBRE) (permiso Contents: Read and write sobre el repo).

Reglas que reemplazan la revisión visual (sacadas de los lotes 1–5):
  - early-pink → unripe solo con confianza ≥ 0,95 (casi siempre era una fresa con rubor, o sea early-pink);
  - commercial-high → overripe solo con confianza ≥ 0,90 (las anaranjadas o con zonas pálidas no son overripe);
  - resto de sugerencias de estado con confianza ≥ 0,60;
  - cajas faltantes con confianza ≥ 0,80 y lado ≥ 2,5 % de la foto (descarta muchas flores y hojas).
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ORD = ["unripe", "early-pink", "commercial-basic", "commercial-high", "overripe"]
UMBRAL = {("early-pink", "unripe"): 0.95, ("commercial-high", "overripe"): 0.90}
UMBRAL_ESTADO, UMBRAL_FALTA, LADO_MIN = 0.60, 0.80, 0.025
REPO = Path(__file__).resolve().parent.parent


SECRETOS = ["GITHUB_TOKEN", "key_gh_strawberry"]


def token(extra=None):
    """Token de GitHub desde los Secrets de Kaggle (prueba varios nombres) o la variable de entorno GITHUB_TOKEN."""
    import os
    try:
        from kaggle_secrets import UserSecretsClient
        cli = UserSecretsClient()
        for n in ([extra] if extra else []) + SECRETOS:
            try:
                t = cli.get_secret(n)
                if t:
                    return t
            except Exception:
                pass
    except ImportError:
        pass
    return os.environ.get("GITHUB_TOKEN")


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout


def datos(lote):
    t = (REPO / "revision" / lote / "index.html").read_text(encoding="utf-8")
    i = t.index("const DATOS=") + 12
    return t, i, t.index(";\n", i)


def revisiones():
    """{lote: revisión} desde feedback/ y desde la rama «resultados» (gana la más reciente)."""
    rev = {}
    for f in (REPO / "feedback").glob("revision_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        rev[d.get("lote", f.stem[9:])] = d
    git("fetch", "-q", "origin", "resultados")
    for n in git("ls-tree", "--name-only", "origin/resultados").split():
        if n.endswith(".json") and n.startswith("lote_"):
            d = json.loads(git("show", f"origin/resultados:{n}"))
            lote = d.get("lote", n[:-5])
            if lote not in rev or d.get("fecha", "") > rev[lote].get("fecha", ""):
                rev[lote] = d
    return rev


def caja(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0


def armar_dataset(rev, destino, n_val=40):
    random.seed(0)
    shutil.rmtree(destino, ignore_errors=True)
    for sp in ("train", "val"):
        (destino / "images" / sp).mkdir(parents=True)
        (destino / "labels" / sp).mkdir(parents=True)
    n = 0
    for lote, r in sorted(rev.items()):
        t, i, j = datos(lote)
        fotos = [d for d in json.loads(t[i:j]) if d["orig"] in r.get("marcas", {})]
        random.shuffle(fotos)
        val = {d["orig"] for d in fotos[:n_val]}
        for d in fotos:
            figs = r.get("correcciones", {}).get(d["orig"], d["figs"])
            sp = "val" if d["orig"] in val else "train"
            stem = f"{lote}_{Path(d['orig']).stem}"
            shutil.copy(REPO / "revision" / lote / d["img"], destino / "images" / sp / f"{stem}.jpg")
            lineas = []
            for f in figs:
                if f["label"] in ORD:
                    x1, y1, x2, y2 = caja(f["pts"])
                    lineas.append(f"{ORD.index(f['label'])} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
            (destino / "labels" / sp / f"{stem}.txt").write_text("\n".join(lineas) + "\n")
            n += 1
    (destino / "data.yaml").write_text(f"path: {destino.resolve()}\ntrain: images/train\nval: images/val\nnames: {ORD}\n")
    return n


def sugerir(modelo, lote, imgsz=640):
    from ultralytics import YOLO
    m = YOLO(modelo)
    t, i, j = datos(lote)
    D = json.loads(t[i:j])
    ns = nf = 0
    for d in D:
        r = m.predict(str(REPO / "revision" / lote / d["img"]), imgsz=imgsz, conf=0.05, iou=0.5, verbose=False)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        preds = [([b[0]/W, b[1]/H, b[2]/W, b[3]/H], ORD[int(c)], float(p))
                 for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        cajas = [caja(f["pts"]) for f in d["figs"]]
        usadas = set()
        for k, f in enumerate(d["figs"]):
            f.pop("sug", None)
            f.pop("sc", None)
            cand = [(j2, p) for j2, p in enumerate(preds) if iou(cajas[k], p[0]) >= 0.5]
            if not cand:
                continue
            usadas.update(j2 for j2, _ in cand)
            _, (b, c, p) = max(cand, key=lambda x: x[1][2])
            if c != f["label"] and p >= UMBRAL.get((f["label"], c), UMBRAL_ESTADO):
                f["sug"], f["sc"] = c, round(p, 2)
                ns += 1
        d.pop("falta", None)
        falta = []
        for j2, (b, c, p) in sorted(enumerate(preds), key=lambda x: -x[1][2]):
            if (j2 in usadas or p < UMBRAL_FALTA or min(b[2]-b[0], b[3]-b[1]) < LADO_MIN
                    or any(iou(b, cj) >= 0.3 for cj in cajas) or any(iou(b, caja(x["pts"])) >= 0.4 for x in falta)):
                continue
            falta.append({"label": c, "conf": round(p, 2),
                          "pts": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]})
        if falta:
            d["falta"] = falta
            nf += len(falta)
    (REPO / "revision" / lote / "index.html").write_text(t[:i] + json.dumps(D, ensure_ascii=False) + t[j:], encoding="utf-8")
    print(f"{lote}: {ns} sugerencias de estado, {nf} posibles cajas faltantes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siguiente", required=True, nargs="+",
                    help="lotes donde escribir sugerencias (p. ej. lote_06 lote_07) o «restantes» = todos los no revisados")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", help="auto (GPU si hay), 0 = GPU, cpu")
    ap.add_argument("--salida", default="/kaggle/working/salida")
    ap.add_argument("--push", action="store_true", help="subir a GitHub con el token guardado en los Secrets de Kaggle")
    ap.add_argument("--secreto", default=None, help="nombre del secreto de Kaggle con el token (si no es uno de los conocidos)")
    a = ap.parse_args()
    from ultralytics import YOLO
    import torch
    if a.device == "auto":
        a.device = "0" if torch.cuda.is_available() else "cpu"
    if a.device == "cpu":
        print("⚠ No hay GPU: el entrenamiento irá en CPU y puede tardar 2-3 horas.\n"
              "  En Kaggle: Settings → Accelerator → GPU (requiere cuenta verificada con teléfono) y reinicia la sesión.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
    rev = revisiones()
    print("Lotes revisados:", ", ".join(sorted(rev)))
    if a.siguiente == ["restantes"]:
        lotes = sorted(d.name for d in (REPO / "revision").glob("lote_*")
                       if (d / "index.html").exists() and d.name not in rev)
    else:
        lotes = a.siguiente
    print("Sugerencias para:", ", ".join(lotes) or "(ninguno)")
    n = armar_dataset(rev, Path("/tmp/ds_fresas"))
    print(f"{n} fotos revisadas para entrenar")
    base = sorted((REPO / "modelos").glob("yolo_fresas_*.pt"), key=lambda p: p.stat().st_mtime)[-1]
    print("Parte del modelo", base.name)
    YOLO(str(base)).train(data="/tmp/ds_fresas/data.yaml", epochs=a.epochs, imgsz=640, batch=16, device=a.device,
                          patience=10, project="/tmp/runs", name="fresas", exist_ok=True, plots=False, verbose=False,
                          fliplr=0.5, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, lr0=0.005)
    nuevo = REPO / "modelos" / f"yolo_fresas_{len(rev)}lotes.pt"
    shutil.copy("/tmp/runs/fresas/weights/best.pt", nuevo)
    for lote in lotes:
        sugerir(str(nuevo), lote)
    sal = Path(a.salida)
    archivos = [f"revision/{l}/index.html" for l in lotes] + [f"modelos/{nuevo.name}"]
    for rel in archivos:
        (sal / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, sal / rel)
    print("Listo. Archivos en", sal)
    if a.push:
        tok = token(a.secreto)
        if not tok:
            sys.exit("No encontré el token: actívalo en Add-ons → Secrets (GITHUB_TOKEN o key_gh_strawberry) o usa --secreto NOMBRE.")
        git("config", "user.email", "kaggle@fresas")
        git("config", "user.name", "Kaggle fresas")
        git("add", *archivos)
        git("commit", "-m", f"{', '.join(lotes)} con sugerencias (YOLO reentrenado en Kaggle con {len(rev)} lotes)")
        print(git("push", f"https://x-access-token:{tok}@github.com/Jairocustodio-jc/DatasetFresas.git",
                  "HEAD:claude/jolly-heisenberg-qa6cj9") or "Subido a GitHub.")


if __name__ == "__main__":
    main()
