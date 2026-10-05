"""Compara YOLOv8, v9, v10, 11, 12 y 26 (nano y el más grande) con los lotes ya revisados. Para dejarlo corriendo de noche en Kaggle.

Notebook de Kaggle (Settings → Accelerator: «GPU T4 x2», Internet: On; Add-ons → Secrets: tu token de GitHub
activado, con el nombre GITHUB_TOKEN o key_gh_strawberry; si usas otro nombre, agrega --secreto NOMBRE):

    !pip -q install ultralytics
    !git clone -q -b claude/jolly-heisenberg-qa6cj9 https://github.com/Jairocustodio-jc/DatasetFresas
    %cd DatasetFresas
    !python modelos/kaggle_benchmark.py

Y luego «Save Version → Save & Run All (Commit)»: así corre solo hasta 12 h aunque cierres el navegador.

Qué compara (mismas condiciones para todos):
  - nano:   yolov8n yolov9t yolov10n yolo11n yolo12n yolo26n   (v9 no tiene «n»: su menor es «t»)
  - grande: yolov8x yolov9e yolov10x yolo11x yolo12x yolo26x   (v9 no tiene «x»: su mayor es «e»)
  - parten de los pesos COCO oficiales; misma partición para todos: 70 % train, 15 % validación, 15 % test, estratificada a la
    vez por estado (cada parte con casi el mismo % de cajas de cada estado) y por lote; semilla 0; se guarda en particion.json;
  - experimentos con yolo11n (después de los nano, antes de los grandes): aumento de tono hsv_h 0,015 y/o 1024 px, para
    ver si el tono engaña a la red (esta variedad va de rosa a rojo con la misma madurez) y si más resolución ayuda;
  - commercial-basic y commercial-high son las clases que más importan: en train sus fotos se repiten (--sobremuestreo 2) y
    la pérdida pesa más las clases escasas (--cls-pw 0.5); la tabla se ordena por «acierto comercial» (% de cajas comerciales
    del test bien detectadas y clasificadas) y muestra el «error crítico» (% que llamó early-pink u overripe) y a qué se
    predice cada caja comercial;
  - la validación elige la mejor época; el test se evalúa una sola vez al final con el mejor peso, y es lo que ordena la tabla;
  - imgsz 640, 100 épocas como máximo, paciencia 20, AdamW lr0 0,001, hsv_h 0 salvo en los experimentos, semilla 0;
  - batch 16 en nano y 8 en grande (memoria de la T4); nbs=64 acumula gradientes, así que el lote efectivo es el mismo.

Cómo cuida el tiempo y los resultados:
  - con 2 GPU entrena dos modelos a la vez (uno por GPU); cada GPU toma el siguiente modelo libre de la cola;
  - reparte las --horas (11 por defecto, Kaggle corta a las 12) según lo que queda: un modelo grande recibe ~6 veces el
    tiempo de uno nano; si se acaba su tiempo, para y guarda su mejor época (queda marcado «parado por tiempo»);
  - cada 10 minutos y al terminar cada modelo sube a GitHub, rama «yolo-benchmark», carpeta benchmark/<nombre>/:
    resumen (RESUMEN.md, resumen.csv), métricas por modelo (res_<modelo>.json) y curvas de entrenamiento (runs/<modelo>/);
  - los pesos (best.pt) van como archivos de un «release» de GitHub (benchmark-<nombre>), que acepta archivos de hasta 2 GB;
  - además todo queda en /kaggle/working/benchmark/<nombre>/ (sale como «Output» de la versión del notebook);
  - si una GPU falla o un proceso muere, se reinicia y sigue con el resto; si un modelo falla, se anota el error y sigue;
  - si se corta todo, volver a correrlo continúa: salta los modelos ya completos que estén en GitHub.
El token nunca se escribe en disco ni se imprime.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GH = "Jairocustodio-jc/DatasetFresas"
RAMA = "yolo-benchmark"
NANO = ["yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo12n", "yolo26n"]
GRANDE = ["yolov9e", "yolo12x", "yolov8x", "yolov10x", "yolo11x", "yolo26x"]   # los más pesados primero
# Experimentos con yolo11n: ¿el tono engaña a la red? (hsv_h 0,015 la obliga a mirar patrón, aquenios y brillo en vez del
# tono exacto) ¿la resolución ayuda? (1024 px conserva aquenios y brillo). La base (hsv_h 0, 640 px) es «yolo11n».
EXPER = ["yolo11n-hsv", "yolo11n-1024", "yolo11n-hsv-1024"]
# Cualquier modelo admite los sufijos «-hsv» (aumento de tono) y «-1024» (1024 px), p. ej. yolo26n-hsv-1024.


def peso(m):
    """Costo relativo de entrenar m (para repartir el tiempo): grande ≈ 6 nano; 1024 px ≈ 2,5 veces 640 px."""
    return (6 if m.split("-")[0] in GRANDE else 1) * (2.5 if "1024" in m else 1)


def valido(m):
    base, *suf = m.split("-")
    return base in NANO + GRANDE and set(suf) <= {"hsv", "1024"}


def ajustes(m):
    """Peso base, tamaño de imagen y aumento de tono de cada entrada de la cola."""
    return m.split("-")[0], (1024 if "1024" in m else 640), (0.015 if "hsv" in m else 0.0)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "modelos"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def leer(p, defecto=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return defecto


def escribir(p, d):
    p = Path(p)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)   # atómico: nunca queda un json a medias


# ───────────────────────── partición train / val / test ─────────────────────────

COMERCIAL = ["commercial-basic", "commercial-high"]   # las clases que más importan: confundirlas sale caro


def particionar(rev, destino, partes=(0.70, 0.15, 0.15), semilla=0, sobremuestreo=2, fijo=None):
    """Reparte las fotos revisadas en train/val/test estratificando a la vez por clase (cajas de cada estado) y por lote.

    Estratificación iterativa (Sechidis et al., 2011): se reparte primero la etiqueta más escasa, cada foto va a la parte que
    más cajas de esa etiqueta le faltan para su porcentaje, y así cada parte queda con casi el mismo % de cada estado y de
    cada lote. Después, solo en train, las fotos con alguna caja comercial se repiten «sobremuestreo» veces (val y test
    quedan intactos, para medir en la distribución real). Devuelve la tabla de reparto y las listas de fotos."""
    import random
    from kaggle_fresas import ORD, caja, datos
    random.seed(semilla)
    lotes = sorted(rev)
    fotos = []
    for lote in lotes:
        t, i, j = datos(lote)
        for d in json.loads(t[i:j]):
            if d["orig"] not in rev[lote].get("marcas", {}):
                continue
            figs = [f for f in rev[lote].get("correcciones", {}).get(d["orig"], d["figs"]) if f["label"] in ORD]
            v = [0] * (len(ORD) + len(lotes))
            for f in figs:
                v[ORD.index(f["label"])] += 1
            v[len(ORD) + lotes.index(lote)] = 1          # el lote cuenta como una etiqueta más
            fotos.append({"lote": lote, "d": d, "figs": figs, "v": v})
    random.shuffle(fotos)
    nombres = ["train", "val", "test"]
    if fijo:   # partición de una corrida anterior: mismas fotos en las mismas partes (las revisadas después se ignoran)
        fotos = [f for f in fotos if f"{f['lote']}_{Path(f['d']['orig']).stem}.jpg" in fijo]
    tot = [sum(f["v"][k] for f in fotos) for k in range(len(fotos[0]["v"]))]
    falta_et = [[t * p for t in tot] for p in partes]     # cuántas cajas/fotos de cada etiqueta le faltan a cada parte
    falta_n = [len(fotos) * p for p in partes]
    asign = {}
    pend = set(range(len(fotos)))
    if fijo:
        asign = {i: nombres.index(fijo[f"{f['lote']}_{Path(f['d']['orig']).stem}.jpg"]) for i, f in enumerate(fotos)}
        pend = set()
    while pend:
        resto = [sum(fotos[i]["v"][k] for i in pend) for k in range(len(tot))]
        k = min((x for x in range(len(tot)) if resto[x] > 0), key=lambda x: resto[x], default=None)
        grupo = [i for i in pend if k is None or fotos[i]["v"][k] > 0]
        for i in grupo:
            s = max(range(3), key=lambda s: ((falta_et[s][k] if k is not None else 0), falta_n[s]))
            asign[i] = s
            falta_n[s] -= 1
            for x, c in enumerate(fotos[i]["v"]):
                falta_et[s][x] -= c
            pend.discard(i)
    shutil.rmtree(destino, ignore_errors=True)
    for sp in nombres:
        (destino / "images" / sp).mkdir(parents=True)
        (destino / "labels" / sp).mkdir(parents=True)
    listas = {sp: [] for sp in nombres}
    for i, f in enumerate(fotos):
        sp = nombres[asign[i]]
        stem = f"{f['lote']}_{Path(f['d']['orig']).stem}"
        shutil.copy(REPO / "revision" / f["lote"] / f["d"]["img"], destino / "images" / sp / f"{stem}.jpg")
        lineas = []
        for g in f["figs"]:
            x1, y1, x2, y2 = caja(g["pts"])
            lineas.append(f"{ORD.index(g['label'])} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
        (destino / "labels" / sp / f"{stem}.txt").write_text("\n".join(lineas) + "\n")
        listas[sp].append(f"{stem}.jpg")
        if sp == "train" and any(g["label"] in COMERCIAL for g in f["figs"]):
            for r in range(1, sobremuestreo):   # copias extra solo en train
                shutil.copy(destino / "images" / sp / f"{stem}.jpg", destino / "images" / sp / f"{stem}_rep{r}.jpg")
                shutil.copy(destino / "labels" / sp / f"{stem}.txt", destino / "labels" / sp / f"{stem}_rep{r}.txt")
    (destino / "data.yaml").write_text(f"path: {destino.resolve()}\ntrain: images/train\nval: images/val\n"
                                       f"test: images/test\nnames: {ORD}\n")
    tabla = {}
    for s, sp in enumerate(nombres):
        mis = [f for i, f in enumerate(fotos) if asign[i] == s]
        if sp == "train":
            rep = sum(any(g["label"] in COMERCIAL for g in f["figs"]) for f in mis) * (sobremuestreo - 1)
        tabla[sp] = {"fotos": len(mis), "copias_extra": rep if sp == "train" else 0, "cajas": {c: sum(f["v"][x] for f in mis) for x, c in enumerate(ORD)},
                     "por_lote": {l: sum(f["lote"] == l for f in mis) for l in lotes}}
    return tabla, {sp: sorted(v) for sp, v in listas.items()}


def tabla_reparto(tab):
    """Tabla en markdown con el % de fotos y de cajas de cada estado en cada parte."""
    from kaggle_fresas import ORD
    tf = sum(t["fotos"] for t in tab.values())
    filas = ["| Parte | Fotos | " + " | ".join(ORD) + " | Cajas |", "|---" * (len(ORD) + 3) + "|"]
    for sp, t in tab.items():
        cj = sum(t["cajas"].values())
        filas.append(f"| {sp} | {t['fotos']} ({t['fotos']/tf:.0%}) | "
                     + " | ".join(f"{t['cajas'][c]} ({t['cajas'][c]/max(cj,1):.0%})" for c in ORD) + f" | {cj} |")
    return filas


# ───────────────────────────── proceso de cada GPU ─────────────────────────────

def worker(a):
    import torch
    from ultralytics import YOLO
    sal = Path(a.salida)
    fin = a.deadline
    n_gpu = a.n_gpu
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    a.dev = 0 if torch.cuda.is_available() else "cpu"
    log(f"[GPU {a.gpu}] {gpu}")
    while True:
        hechos = {p.stem[4:] for p in sal.glob("res_*.json")}
        pend = [m for m in a.modelos if m not in hechos]
        libre = [m for m in pend if not (sal / "claims" / f"{m}.lock").exists()]
        if not libre:
            return
        m = libre[0]
        try:   # reservar el modelo para esta GPU (atómico entre procesos)
            fd = os.open(sal / "claims" / f"{m}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(a.gpu).encode())
            os.close(fd)
        except FileExistsError:
            continue
        resta = fin - time.time() - 15 * 60          # 15 min para la evaluación final y la subida
        if resta < 20 * 60:
            escribir(sal / f"res_{m}.json", {"modelo": m, "estado": "sin tiempo"})
            log(f"[GPU {a.gpu}] {m}: sin tiempo, se salta")
            continue
        carga = sum(peso(x) for x in pend) / n_gpu
        tope = min(resta, resta * peso(m) / max(carga, peso(m)))
        entrenar(m, a, sal, tope, fin, gpu, YOLO)
        torch.cuda.empty_cache()


def entrenar(m, a, sal, tope, fin, gpu, YOLO):
    grande = m.split("-")[0] in GRANDE
    base, imgsz, hsv_h = ajustes(m)
    batch = 8 if grande or imgsz > 640 else 16
    t0 = time.time()
    log(f"[GPU {a.gpu}] {m}: empieza (tope {tope/3600:.1f} h, batch {batch})")
    parado = {"tiempo": False}

    def por_tiempo(tr):
        if time.time() - t0 > tope or time.time() > fin - 10 * 60:
            parado["tiempo"] = True
            tr.stop = True

    res = {"modelo": m, "tamaño": "grande" if grande else "nano", "gpu": gpu, "epochs_max": a.epochs,
           "imgsz": imgsz, "hsv_h": hsv_h}
    for intento in range(3):
        try:
            modelo = YOLO(f"{base}.pt")
            modelo.add_callback("on_fit_epoch_end", por_tiempo)
            modelo.train(data=a.data, epochs=a.epochs, patience=20, imgsz=imgsz, batch=batch, device=a.dev,
                         optimizer="AdamW", lr0=0.001, nbs=64, seed=0, deterministic=True, workers=a.workers,
                         cache="ram" if not grande and imgsz <= 640 else False, project=str(sal / "runs"), name=m, exist_ok=True, plots=True, verbose=False,
                         fliplr=0.5, hsv_h=hsv_h, hsv_s=0.3, hsv_v=0.3, amp=True, cls_pw=a.cls_pw)
            break
        except Exception as e:
            oom = "out of memory" in str(e).lower()
            log(f"[GPU {a.gpu}] {m}: error ({'memoria' if oom else type(e).__name__}): {str(e)[:300]}")
            if oom and batch > 2 and intento < 2:
                batch //= 2
                import torch
                torch.cuda.empty_cache()
                log(f"[GPU {a.gpu}] {m}: reintento con batch {batch}")
                continue
            res |= {"estado": "error", "error": f"{type(e).__name__}: {str(e)[:500]}", "minutos": round((time.time()-t0)/60, 1)}
            escribir(sal / f"res_{m}.json", res)
            return
    run = sal / "runs" / m
    best = run / "weights" / "best.pt"
    filas = list(csv.DictReader(open(run / "results.csv"))) if (run / "results.csv").exists() else []
    clave = next((k for k in (filas[0] if filas else {}) if "mAP50-95" in k), None)
    mejor = max(filas, key=lambda f: float(f[clave])) if clave else {}
    res |= {"estado": "completo", "parado_por_tiempo": parado["tiempo"], "batch": batch,
            "epochs_hechas": len(filas), "mejor_epoch": int(float(mejor.get("epoch", 0))) if mejor else None,
            "minutos": round((time.time() - t0) / 60, 1)}
    try:   # evaluación final del mejor peso, igual para todos
        mb = YOLO(str(best))
        for sp, pre in (("val", "val_"), ("test", "")):   # test: fotos que ningún modelo vio ni usó para elegir época
            v = mb.val(data=a.data, split=sp, imgsz=imgsz, batch=8, device=a.dev, plots=True, verbose=False,
                       project=str(run), name=sp, exist_ok=True)
            res |= {pre + "map50": round(float(v.box.map50), 4), pre + "map50_95": round(float(v.box.map), 4),
                    pre + "precision": round(float(v.box.mp), 4), pre + "recall": round(float(v.box.mr), 4),
                    pre + "ap50_por_clase": {v.names[int(c)]: round(float(x), 4)
                                             for c, x in zip(v.box.ap_class_index, v.box.ap50)}}
            if sp == "test":
                cm = confusion_comercial(v.confusion_matrix.matrix, v.names)   # referencia (matriz de Ultralytics)
                res |= {"acierto_comercial_cm": cm.get("acierto_comercial"), "error_critico_cm": cm.get("error_critico")}
                res |= comercial_top1(mb, Path(a.data).parent, imgsz, a.dev)       # la métrica principal
        res["inferencia_ms"] = round(float(v.speed.get("inference", 0)), 2)
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        res |= {"params_M": round(get_num_params(mb.model) / 1e6, 2), "gflops": round(float(get_flops(mb.model, 640)), 1)}
    except Exception as e:
        res["error_val"] = f"{type(e).__name__}: {str(e)[:300]}"
    (sal / "pesos").mkdir(exist_ok=True)
    if best.exists():
        shutil.copy(best, sal / "pesos" / f"{m}.pt")
        res["peso_MB"] = round(best.stat().st_size / 1e6, 1)
    (run / "weights" / "last.pt").unlink(missing_ok=True)   # ahorra disco; best.pt ya está copiado
    escribir(sal / f"res_{m}.json", res)
    log(f"[GPU {a.gpu}] {m}: listo en {res['minutos']} min · mAP50 {res.get('map50')} · mAP50-95 {res.get('map50_95')}"
        + (" · parado por tiempo" if parado["tiempo"] else ""))


def confusion_comercial(mat, names):
    """De las cajas reales commercial-basic y commercial-high del test (confianza ≥ 0,25, IoU ≥ 0,45), a qué se predijeron.

    error_critico = % de cajas comerciales que el modelo llamó early-pink u overripe (el error que más cuesta).
    La matriz de Ultralytics es [predicho, real]; la última fila/columna es «fondo» (no detectada / falsa detección)."""
    idx = {n: int(i) for i, n in names.items()}
    nc = len(idx)
    out, crit, tot = {}, 0, 0
    for c in COMERCIAL:
        col = mat[:, idx[c]]
        n = float(col.sum())
        if not n:
            continue
        pct = lambda x: round(100 * float(x) / n, 1)
        out[c] = {"cajas": int(n), "bien": pct(col[idx[c]]), "early-pink": pct(col[idx["early-pink"]]),
                  "overripe": pct(col[idx["overripe"]]), "otra_comercial": pct(sum(col[idx[o]] for o in COMERCIAL if o != c)),
                  "unripe": pct(col[idx["unripe"]]), "no_detectada": pct(col[nc])}
        crit += float(col[idx["early-pink"]] + col[idx["overripe"]])
        tot += n
    return {"confusion_comercial": out, "error_critico": round(100 * crit / tot, 1) if tot else None,
            "acierto_comercial": round(sum(out[c]["bien"] * out[c]["cajas"] for c in out) / tot, 1) if tot else None}


def comercial_top1(mb, ds, imgsz, dev):
    """Por cada caja real comercial del test, la predicción MÁS SEGURA que la cubre (IoU ≥ 0,5, cualquier clase,
    confianza ≥ 0,25): es lo que haría el equipo en el campo. La matriz de Ultralytics, en cambio, empareja por IoU y
    puede quedarse con una segunda clase de baja confianza sobre la misma fresa, lo que subestima el acierto."""
    from kaggle_fresas import ORD, iou
    conf = {c: {} for c in COMERCIAL}
    for p in sorted((ds / "images" / "test").iterdir()):
        r = mb.predict(str(p), imgsz=imgsz, conf=0.25, device=dev, verbose=False)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        pr = [([b[0]/W, b[1]/H, b[2]/W, b[3]/H], int(c), float(x)) for b, c, x in
              zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        for l in (ds / "labels" / "test" / f"{p.stem}.txt").read_text().split("\n"):
            if not l.strip():
                continue
            c, x, y, w, h = map(float, l.split())
            if ORD[int(c)] not in COMERCIAL:
                continue
            cand = [q for q in pr if iou([x-w/2, y-h/2, x+w/2, y+h/2], q[0]) >= 0.5]
            pred = ORD[max(cand, key=lambda q: q[2])[1]] if cand else "no_detectada"
            conf[ORD[int(c)]][pred] = conf[ORD[int(c)]].get(pred, 0) + 1
    out = {}
    for c, d in conf.items():
        n = sum(d.values())
        if n:
            pct = lambda k: round(100 * d.get(k, 0) / n, 1)
            out[c] = {"cajas": n, "bien": pct(c), "early-pink": pct("early-pink"), "overripe": pct("overripe"),
                      "otra_comercial": pct([o for o in COMERCIAL if o != c][0]), "unripe": pct("unripe"),
                      "no_detectada": pct("no_detectada")}
    tot = sum(o["cajas"] for o in out.values())
    if not tot:
        return {}
    return {"confusion_comercial": out,
            "acierto_comercial": round(sum(o["bien"] * o["cajas"] for o in out.values()) / tot, 1),
            "error_critico": round(sum((o["early-pink"] + o["overripe"]) * o["cajas"] for o in out.values()) / tot, 1)}


# ───────────────────────────── respaldo en GitHub ─────────────────────────────

class Respaldo:
    def __init__(self, sal, nombre, activo, secreto=None, huella=None):
        self.sal, self.nombre, self.tok = sal, nombre, None
        self.dir = Path("/tmp/bench_git")
        self.subidos = set()
        if not activo:
            return
        from kaggle_fresas import token
        self.tok = token(secreto)
        if not self.tok:
            log("⚠ No encontré el token de GitHub en los Secrets (GITHUB_TOKEN, key_gh_strawberry o --secreto): "
                "los resultados quedan solo en /kaggle/working (Output del notebook).")
            return
        self.url = f"https://x-access-token:{self.tok}@github.com/{GH}.git"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "kaggle@fresas")
        self.git("config", "user.name", "Kaggle benchmark")
        if self.git("ls-remote", self.url, f"refs/heads/{RAMA}").strip():
            self.git("fetch", "-q", self.url, RAMA)
            self.git("checkout", "-q", "-B", RAMA, "FETCH_HEAD")
        else:
            self.git("checkout", "-q", "--orphan", RAMA)
        prev = self.dir / "benchmark" / nombre
        n = 0
        misma = (leer(prev / "config.json") or {}).get("huella_particion") == huella
        if not misma and any(prev.glob("res_*.json")):
            log("La partición cambió desde la corrida anterior (más fotos revisadas): se entrena todo de nuevo.")
            shutil.rmtree(prev)
        for p in (prev.glob("res_*.json") if misma else []):   # retomar: lo completo de una corrida anterior no se repite
            if (leer(p) or {}).get("estado") == "completo" and not (sal / p.name).exists():
                shutil.copy(p, sal / p.name)
                n += 1
        if n:
            log(f"Retomando: {n} modelos ya completos en GitHub, se saltan.")
        try:
            self.subidos = {x["name"] for x in self.release().get("assets", [])}
        except Exception:
            pass

    def git(self, *a):
        r = subprocess.run(["git", *a], cwd=self.dir, capture_output=True, text=True)
        out = (r.stdout + r.stderr).replace(self.tok or "\0", "***")
        self.rc = r.returncode
        if r.returncode and a[0] not in ("ls-remote", "commit"):
            log("git", a[0], "falló:", out.strip()[-300:])
        return out

    def api(self, metodo, ruta, **kw):
        import requests
        h = {"Authorization": f"Bearer {self.tok}", "Accept": "application/vnd.github+json"}
        url = ruta if ruta.startswith("https") else f"https://api.github.com/repos/{GH}{ruta}"
        return requests.request(metodo, url, headers=h | kw.pop("headers", {}), timeout=600, **kw)

    def release(self):
        tag = f"benchmark-{self.nombre}"
        r = self.api("GET", f"/releases/tags/{tag}")
        if r.status_code == 404:
            r = self.api("POST", "/releases", json={
                "tag_name": tag, "target_commitish": RAMA, "name": f"Benchmark YOLO ({self.nombre})", "prerelease": True,
                "body": f"Pesos best.pt de cada modelo. Métricas en la rama {RAMA}, carpeta benchmark/{self.nombre}/."})
        r.raise_for_status()
        return r.json()

    def sincronizar(self):
        if not self.tok:
            return
        try:
            dst = self.dir / "benchmark" / self.nombre
            shutil.copytree(self.sal, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.pt", "*.cache", "*.tmp", "claims", "logs", "weights", "*.lock"))
            self.git("add", "-A")
            if "nothing to commit" not in self.git("commit", "-q", "-m", f"Benchmark {self.nombre}: {time.strftime('%d/%m %H:%M')}"):
                for k in range(4):
                    self.git("push", "-q", self.url, f"HEAD:{RAMA}")
                    if self.rc == 0:
                        break
                    self.git("fetch", "-q", self.url, RAMA)
                    self.git("rebase", "-q", "FETCH_HEAD")
                    time.sleep(2 ** (k + 1))
        except Exception as e:
            log("⚠ No se pudo subir a GitHub:", type(e).__name__, str(e)[:200].replace(self.tok, "***"))
        for p in sorted((self.sal / "pesos").glob("*.pt")):
            if p.name in self.subidos:
                continue
            try:
                rel = self.release()
                for x in rel.get("assets", []):
                    if x["name"] == p.name:
                        self.api("DELETE", f"/releases/assets/{x['id']}")
                up = rel["upload_url"].split("{")[0] + f"?name={p.name}"
                with open(p, "rb") as f:
                    r = self.api("POST", up, data=f, headers={"Content-Type": "application/octet-stream"})
                r.raise_for_status()
                self.subidos.add(p.name)
                log(f"Peso {p.name} subido al release benchmark-{self.nombre}")
            except Exception as e:
                log(f"⚠ No se pudo subir {p.name}:", type(e).__name__, str(e)[:200].replace(self.tok, "***"))


def resumen(sal, modelos, info):
    modelos = list(modelos) + sorted({p.stem[4:] for p in sal.glob("res_*.json")} - set(modelos))   # también los de antes
    filas = []
    for m in modelos:
        r = leer(sal / f"res_{m}.json")
        if r is None:
            r = {"modelo": m, "estado": "entrenando" if (sal / "claims" / f"{m}.lock").exists() else "pendiente"}
        filas.append(r)
    cols = ["modelo", "tamaño", "imgsz", "hsv_h", "estado", "acierto_comercial", "error_critico", "map50", "map50_95", "precision", "recall", "val_map50", "val_map50_95",
            "epochs_hechas", "mejor_epoch",
            "parado_por_tiempo", "minutos", "batch", "params_M", "gflops", "inferencia_ms", "peso_MB", "gpu", "error"]
    with open(sal / "resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    ok = sorted([r for r in filas if r.get("map50_95") is not None],
                key=lambda r: (-(r.get("acierto_comercial") or 0), r.get("error_critico") or 0, -r["map50_95"]))
    clases = ["unripe", "early-pink", "commercial-basic", "commercial-high", "overripe"]
    t = [f"# Benchmark YOLO · {info['nombre']}", "",
         f"Actualizado {time.strftime('%d/%m/%Y %H:%M')} (hora de Kaggle, UTC). Lotes: {', '.join(info['lotes'])}. "
         f"{info['fotos']} fotos. GPU: {info['gpus']}.", "",
         "**Reparto estratificado por estado y por lote** (fotos y cajas de cada estado en cada parte):", "",
         *tabla_reparto(info["reparto"]), "",
         f"En train, las fotos con cajas comerciales van repetidas ({info['reparto']['train'].get('copias_extra', 0)} "
         f"copias extra) y la pérdida de clasificación pesa más las clases escasas (cls_pw {info.get('cls_pw')}). "
         "Validación y test quedan con la distribución real.", "",
         "**Resultados en test** (fotos que ningún modelo vio al entrenar ni usó para elegir su mejor época). "
         "Ordenados por **acierto comercial** = % de cajas commercial-basic/high detectadas y con su estado correcto "
         "(más es mejor); a igualdad, por menor **error crítico** = % que el modelo llamó early-pink u overripe "
         "(menos es mejor; las no detectadas no cuentan aquí, ver la tabla de abajo). "
         "«val» = mAP50-95 en validación.", "",
         "| # | Modelo | Acierto com. | Error crítico | mAP50-95 | mAP50 | P | R | val | Épocas | Min | Params (M) | GFLOPs | ms/img |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ok, 1):
        ep = f"{r.get('epochs_hechas')} (mejor {r.get('mejor_epoch')})" + (" ⏱" if r.get("parado_por_tiempo") else "")
        t.append(f"| {i} | {r['modelo']} | {r.get('acierto_comercial', '–')} % | {r.get('error_critico', '–')} % | {r['map50_95']:.3f} | {r['map50']:.3f} | {r['precision']:.2f} | {r['recall']:.2f} "
                 f"| {r.get('val_map50_95', '')} | {ep} | {r.get('minutos')} | {r.get('params_M', '')} | {r.get('gflops', '')} | {r.get('inferencia_ms', '')} |")
    if ok:
        t += ["", "**AP50 por clase (test)**", "", "| Modelo | " + " | ".join(clases) + " |", "|---" * (len(clases) + 1) + "|"]
        for r in ok:
            ap = r.get("ap50_por_clase", {})
            t.append(f"| {r['modelo']} | " + " | ".join(f"{ap[c]:.2f}" if c in ap else "–" for c in clases) + " |")
    if ok:
        t += ["", "**A qué se predicen las cajas comerciales del test** (% de las cajas reales de cada clase)", "",
              "| Modelo | Clase real | Cajas | ✓ bien | → early-pink | → overripe | → la otra comercial | → unripe | no detectada |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in ok:
            for c, d in (r.get("confusion_comercial") or {}).items():
                t.append(f"| {r['modelo']} | {c} | {d['cajas']} | {d['bien']} | **{d['early-pink']}** | **{d['overripe']}** "
                         f"| {d['otra_comercial']} | {d['unripe']} | {d['no_detectada']} |")
    exp = {r["modelo"]: r for r in ok if r["modelo"] == "yolo11n" or r["modelo"] in EXPER}
    if len(exp) > 1:
        t += ["", "**Experimentos con yolo11n: aumento de tono y resolución** (mismo test)", "",
              "| Variante | hsv_h | imgsz | Acierto com. | Error crítico | mAP50-95 | ms/img |", "|---|---|---|---|---|---|---|"]
        for m in ["yolo11n"] + EXPER:
            if m in exp:
                r = exp[m]
                t.append(f"| {m} | {r.get('hsv_h', 0.0)} | {r.get('imgsz', 640)} | {r.get('acierto_comercial')} % "
                         f"| {r.get('error_critico')} % | {r['map50_95']:.3f} | {r.get('inferencia_ms', '')} |")
    otros = [r for r in filas if r.get("map50_95") is None]
    if otros:
        t += ["", "**Sin resultado todavía**", ""] + [f"- {r['modelo']}: {r.get('estado')}"
                                                      + (f" ({r['error'][:150]})" if r.get("error") else "") for r in otros]
    t += ["", "⏱ = se detuvo por el tope de tiempo; se guarda su mejor época. Pesos: release "
          f"`benchmark-{info['nombre']}` del repo. Mismas condiciones para todos: ver modelos/kaggle_benchmark.py."]
    (sal / "RESUMEN.md").write_text("\n".join(t) + "\n", encoding="utf-8")
    return filas


# ───────────────────────────── proceso principal ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horas", type=float, default=11.0, help="tiempo total (Kaggle corta a las 12 h)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--modelos", nargs="+", default=NANO + EXPER + GRANDE)
    ap.add_argument("--nombre", default=None, help="nombre de la corrida (por defecto «<N>lotes»); repetirlo retoma")
    ap.add_argument("--salida", default="/kaggle/working/benchmark")
    ap.add_argument("--sin-github", action="store_true", help="no subir nada (solo /kaggle/working)")
    ap.add_argument("--secreto", default=None, help="nombre del secreto de Kaggle con el token de GitHub")
    ap.add_argument("--partes", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"),
                    help="proporción de fotos para train, validación y test (por defecto 0.70 0.15 0.15)")
    ap.add_argument("--nueva-particion", action="store_true",
                    help="rehacer la partición con todas las fotos revisadas (por defecto reusa la de la corrida guardada)")
    ap.add_argument("--sobremuestreo", type=int, default=2,
                    help="veces que aparece en train cada foto con cajas comerciales (1 = sin repetir)")
    ap.add_argument("--cls-pw", type=float, default=0.5,
                    help="peso de clases escasas en la pérdida: 0 = igual, 1 = inverso de la frecuencia (por defecto 0.5)")
    # internos (proceso de cada GPU)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--n-gpu", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--deadline", type=float, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--data", default="", help=argparse.SUPPRESS)
    ap.add_argument("--workers", type=int, default=4, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    inicio = time.time()
    fin = inicio + a.horas * 3600
    desconocidos = [m for m in a.modelos if not valido(m)]
    if desconocidos:
        sys.exit(f"Modelos desconocidos: {desconocidos}. Opciones: {NANO + GRANDE}, con sufijos -hsv y/o -1024")
    import torch
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        sys.exit("No hay GPU. En Kaggle: Settings → Accelerator → GPU T4 x2.")
    gpus = ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu))
    log(f"GPU: {gpus}")

    from kaggle_fresas import revisiones
    rev = revisiones()
    nombre = a.nombre or f"{len(rev)}lotes"
    ds = Path("/tmp/ds_fresas")
    fijo = None
    if not a.nueva_particion:   # reusar la partición guardada en GitHub, si existe, para que los resultados sean comparables
        subprocess.run(["git", "fetch", "-q", "origin", RAMA], cwd=REPO, capture_output=True)
        r = subprocess.run(["git", "show", f"origin/{RAMA}:benchmark/{nombre}/particion.json"], cwd=REPO,
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            fijo = {x: sp for sp, xs in json.loads(r.stdout).items() for x in xs}
            log(f"Reuso la partición guardada de «{nombre}» ({len(fijo)} fotos); --nueva-particion para rehacerla.")
    reparto, listas = particionar(rev, ds, a.partes, sobremuestreo=a.sobremuestreo, fijo=fijo)
    import hashlib
    huella = hashlib.sha1(json.dumps([a.partes, a.sobremuestreo, a.cls_pw, listas], sort_keys=True).encode()).hexdigest()[:12]
    sal = Path(a.salida) / nombre
    (sal / "claims").mkdir(parents=True, exist_ok=True)
    (sal / "logs").mkdir(exist_ok=True)
    for p in (sal / "claims").glob("*.lock"):   # reservas de una corrida anterior en esta misma sesión
        p.unlink()
    for p in sal.glob("res_*.json"):            # lo no completo se reintenta
        if (leer(p) or {}).get("estado") != "completo":
            p.unlink()
    log(f"Corrida «{nombre}» · lotes revisados: {', '.join(sorted(rev))} · fin a más tardar a las "
        f"{time.strftime('%H:%M', time.localtime(fin))}")

    respaldo = Respaldo(sal, nombre, not a.sin_github, a.secreto, huella)
    n = sum(len(v) for v in listas.values())
    info = {"nombre": nombre, "lotes": sorted(rev), "fotos": n, "gpus": gpus, "reparto": reparto, "cls_pw": a.cls_pw}
    escribir(sal / "config.json", info | {"modelos": a.modelos, "epochs": a.epochs, "horas": a.horas, "partes": a.partes,
                                          "huella_particion": huella, "inicio": time.strftime("%Y-%m-%d %H:%M")})
    escribir(sal / "particion.json", listas)
    log("Reparto (fotos y % de cajas por estado):\n" + "\n".join(tabla_reparto(reparto)))
    datas = []
    for g in range(n_gpu):   # una copia por GPU: así no chocan al escribir la caché de etiquetas
        d = Path(f"/tmp/ds_fresas_g{g}")
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(ds, d)
        (d / "data.yaml").write_text((ds / "data.yaml").read_text().replace(str(ds.resolve()), str(d.resolve())))
        datas.append(str(d / "data.yaml"))
    log(f"{n} fotos · modelos: {' '.join(a.modelos)}")
    resumen(sal, a.modelos, info)
    respaldo.sincronizar()

    def lanzar(g):
        cmd = [sys.executable, __file__, "--worker", "--gpu", str(g), "--n-gpu", str(n_gpu), "--deadline", str(fin),
               "--data", datas[g], "--salida", str(sal), "--epochs", str(a.epochs),
               "--workers", str(max(2, (os.cpu_count() or 4) // n_gpu)), "--cls-pw", str(a.cls_pw), "--modelos", *a.modelos]
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(g)}
        out = open(sal / "logs" / f"gpu{g}.log", "a")
        return subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)

    procs = {g: lanzar(g) for g in range(n_gpu)}
    reinicios = {g: 0 for g in range(n_gpu)}
    muertes = {}
    ultimo, vistos = time.time(), set()
    while procs:
        time.sleep(30)
        for g, p in list(procs.items()):
            if p.poll() is None:
                continue
            del procs[g]
            if p.returncode == 0:
                log(f"GPU {g}: terminó su cola")
                continue
            log(f"⚠ GPU {g}: el proceso murió (código {p.returncode}); ver logs/gpu{g}.log")
            for c in (sal / "claims").glob("*.lock"):   # liberar el modelo que tenía a medias
                m = c.stem
                if c.read_text() == str(g) and not (sal / f"res_{m}.json").exists():
                    muertes[m] = muertes.get(m, 0) + 1
                    if muertes[m] >= 2:
                        escribir(sal / f"res_{m}.json", {"modelo": m, "estado": "error",
                                                         "error": f"el proceso murió 2 veces (código {p.returncode})"})
                    else:
                        c.unlink()
            if reinicios[g] < 3 and time.time() < fin - 30 * 60:
                reinicios[g] += 1
                log(f"GPU {g}: reinicio {reinicios[g]}/3")
                procs[g] = lanzar(g)
        nuevos = {p.name for p in sal.glob("res_*.json")} - vistos
        if nuevos or time.time() - ultimo > 600:
            vistos |= nuevos
            for x in sorted(nuevos):
                r = leer(sal / x) or {}
                log(f"✓ {r.get('modelo')}: {r.get('estado')} · mAP50-95 {r.get('map50_95')} · {r.get('minutos')} min")
            resumen(sal, a.modelos, info)
            respaldo.sincronizar()
            ultimo = time.time()
    filas = resumen(sal, a.modelos, info)
    respaldo.sincronizar()
    log(f"Terminado en {(time.time() - inicio) / 3600:.1f} h.")
    print((sal / "RESUMEN.md").read_text(encoding="utf-8"), flush=True)
    faltan = [r["modelo"] for r in filas if r.get("estado") != "completo"]
    if faltan:
        print(f"Sin completar: {' '.join(faltan)}. Vuelve a correr el mismo comando en otra sesión y sigue desde ahí.")


if __name__ == "__main__":
    main()
