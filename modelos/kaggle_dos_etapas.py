"""Segunda etapa: clasificador sobre el recorte de cada fresa, comparado con el detector YOLO de 5 clases en los MISMOS folds
y el MISMO test fijo de la validación cruzada. Corre en Kaggle (GPU T4) con «Save & Run All»:

    !pip -q install ultralytics
    !git clone -q -b claude/jolly-heisenberg-qa6cj9 https://github.com/Jairocustodio-jc/DatasetFresas
    %cd DatasetFresas
    !python modelos/kaggle_dos_etapas.py --particion-de 6lotes --detectores yolov9t-1024 yolo12n-1024 --folds 1 2 3

Qué hace (diseño en modelos/INVESTIGACION_DOS_ETAPAS.md; protocolo en PROTOCOLO_VISION.md):
  1. Rearma la partición y los folds de la CV (benchmark/cv<K>_<origen>/particion.json de la rama yolo-benchmark).
  2. Recorta cada caja REAL con margen (--margen 0.15 por lado, caja expandida a cuadrado), la lleva a --tam px con relleno
     gris (sin deformar) y la guarda por clase (ImageFolder) para train / val / test de cada fold. En train, las comerciales
     van repetidas (--sobremuestreo 2) y se omiten las cajas de lado < --min-lado px (no hay información); en val y test
     se conservan todas.
  3. Entrena cada clasificador (--clasificadores) y variante (ce = entropía cruzada; ordinal = etiquetas suaves gaussianas
     sobre el orden de madurez, solo en los de torchvision) en cada fold; la mejor época se elige por el ACIERTO COMERCIAL EN
     LA VALIDACIÓN del fold. Aumentos sin cambiar el tono (hsv_h 0): volteo, rotación/escala leves, brillo y saturación.
  4. En el test fijo compara, por cada fresa real, tres predicciones: (a) YOLO solo (la detección más segura que la cubre,
     IoU ≥ 0,5, conf ≥ 0,25); (b) YOLO + clasificador sobre el recorte de ESA detección; (c) cota superior: clasificador
     sobre el recorte de la caja real. Acierto comercial y error crítico iguales a los del benchmark, McNemar (mid-p)
     entre (a) y (b) sobre las mismas fresas e intervalo bootstrap pareado (remuestreo por foto).
  5. Guarda todo en /kaggle/working/benchmark/dos_etapas_<origen>/ (res_<trabajo>.json, RESUMEN.md, detecciones en caché)
     y lo sube a la rama yolo-benchmark; los pesos van al release benchmark-dos_etapas_<origen>. Si se corta, el mismo
     comando continúa (salta los trabajos completos con la misma partición).
Lo que NO decide este script: la elección final es por validación; el test se mira una vez, al final, y lo dice el RESUMEN.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "modelos"))
import kaggle_benchmark as kb                     # partición, folds, respaldo
from kaggle_fresas import ORD, iou

GH = kb.GH
COMERCIAL = kb.COMERCIAL
CRITICO = {"early-pink", "overripe"}
CLASIF = ["yolo11n-cls", "mobilenet_v3_small", "efficientnet_b0", "resnet18"]
VARIANTES = {"ce": {}, "ordinal": {"sigma": 0.75}}   # ordinal: solo para los modelos de torchvision
MEDIA, DESV = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
log, leer, escribir = kb.log, kb.leer, kb.escribir


def es_yolo(nombre):
    return nombre.startswith("yolo")


def carpeta_clase(c):
    """Carpeta por clase con prefijo numérico: así ImageFolder y Ultralytics ordenan las clases igual que ORD."""
    return f"{ORD.index(c)}_{c}"


# ───────────────────────────── recortes ─────────────────────────────

def recortar(img, caja, margen, tam):
    """Recorte cuadrado alrededor de la caja (píxeles x1,y1,x2,y2) con «margen» por lado, llevado a tam×tam sin deformar.
    Lo que cae fuera de la foto se rellena de gris (114), como el letterbox de YOLO."""
    import cv2
    h, w = img.shape[:2]
    x1, y1, x2, y2 = caja
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    lado = max(x2 - x1, y2 - y1) * (1 + 2 * margen)
    X1, Y1 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    X2, Y2 = int(round(cx + lado / 2)), int(round(cy + lado / 2))
    lienzo = np.full((Y2 - Y1, X2 - X1, 3), 114, np.uint8)
    sx1, sy1, sx2, sy2 = max(X1, 0), max(Y1, 0), min(X2, w), min(Y2, h)
    if sx2 > sx1 and sy2 > sy1:
        lienzo[sy1 - Y1:sy2 - Y1, sx1 - X1:sx2 - X1] = img[sy1:sy2, sx1:sx2]
    return cv2.resize(lienzo, (tam, tam), interpolation=cv2.INTER_AREA if lienzo.shape[0] > tam else cv2.INTER_CUBIC)


def cajas_de(ds, split, stem):
    """Cajas reales de una foto del dataset YOLO: [(clase, [x1,y1,x2,y2] normalizado)]."""
    out = []
    for l in (ds / "labels" / split / f"{stem}.txt").read_text().split("\n"):
        if l.strip():
            c, x, y, w, h = map(float, l.split())
            out.append((ORD[int(c)], [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
    return out


def armar_recortes(ds, destino, margen, tam, sobremuestreo=2, min_lado=24):
    """Recortes de las cajas reales de train / val / test del fold, por clase. Devuelve el índice del test."""
    import cv2
    shutil.rmtree(destino, ignore_errors=True)
    indice, cuenta = [], {}
    for split in ("train", "val", "test"):
        for c in ORD:
            (destino / split / carpeta_clase(c)).mkdir(parents=True, exist_ok=True)
        for p in sorted((ds / "images" / split).iterdir()):
            if "_rep" in p.stem:
                continue                           # las copias del sobremuestreo del detector no se recortan dos veces
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            for i, (c, b) in enumerate(cajas_de(ds, split, p.stem)):
                px = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
                lado = min(px[2] - px[0], px[3] - px[1])
                if split == "train" and lado < min_lado:
                    continue
                rec = recortar(img, px, margen, tam)
                nombre = f"{p.stem}_{i}"
                cv2.imwrite(str(destino / split / carpeta_clase(c) / f"{nombre}.jpg"), rec, [cv2.IMWRITE_JPEG_QUALITY, 95])
                copias = sobremuestreo if split == "train" and c in COMERCIAL else 1
                for r in range(1, copias):
                    cv2.imwrite(str(destino / split / carpeta_clase(c) / f"{nombre}_rep{r}.jpg"), rec, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cuenta[(split, c)] = cuenta.get((split, c), 0) + copias
                if split == "test":
                    indice.append({"foto": p.name, "i": i, "clase": c, "caja": b, "lado_px": round(lado, 1)})
    tabla = {s: {c: cuenta.get((s, c), 0) for c in ORD} for s in ("train", "val", "test")}
    escribir(destino / "indice_test.json", indice)
    escribir(destino / "recuento.json", tabla)
    return indice, tabla


# ───────────────────────────── clasificadores ─────────────────────────────

def modelo_tv(nombre, nc=5):
    import torch.nn as nn
    import torchvision
    m = torchvision.models.get_model(nombre, weights=None if os.environ.get("DE_SIN_PREENTRENAR") else "DEFAULT")
    if hasattr(m, "fc"):                                      # resnet
        m.fc = nn.Linear(m.fc.in_features, nc)
    else:                                                     # mobilenet / efficientnet: classifier = Sequential(..., Linear)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nc)
    return m


def capa_cam(m, nombre):
    """Última capa convolucional (para Grad-CAM) de cada arquitectura."""
    return m.layer4 if hasattr(m, "layer4") else m.features[-1]


def transformadas(tam, entrenar):
    import torchvision.transforms as T
    if entrenar:   # sin tocar el tono: el color define la clase
        return T.Compose([T.RandomHorizontalFlip(), T.RandomAffine(degrees=15, scale=(0.9, 1.1), translate=(0.05, 0.05), fill=114),
                          T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.0),
                          T.Resize((tam, tam)), T.ToTensor(), T.Normalize(MEDIA, DESV)])
    return T.Compose([T.Resize((tam, tam)), T.ToTensor(), T.Normalize(MEDIA, DESV)])


def objetivo_ordinal(y, sigma, nc=5):
    """Etiquetas suaves: gaussiana centrada en el estado real sobre el orden unripe < early-pink < basic < high < overripe."""
    import torch
    idx = torch.arange(nc, dtype=torch.float32, device=y.device)
    t = torch.exp(-0.5 * ((idx[None, :] - y[:, None].float()) / sigma) ** 2)
    return t / t.sum(1, keepdim=True)


def acierto_comercial_de(pred, real):
    """Acierto comercial (% de cajas comerciales bien clasificadas) y error crítico, a partir de listas de clases."""
    com = [(p, r) for p, r in zip(pred, real) if r in COMERCIAL]
    if not com:
        return None, None
    return (round(100 * sum(p == r for p, r in com) / len(com), 1),
            round(100 * sum(p in CRITICO for p, r in com) / len(com), 1))


def entrenar_tv(nombre, var, dir_rec, epochs, tam, dev, salida, batch=64):
    """Entrena un modelo de torchvision sobre los recortes; elige la época por acierto comercial en val. Devuelve métricas."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    torch.manual_seed(0)
    tr = ImageFolder(dir_rec / "train", transformadas(tam, True))
    va = ImageFolder(dir_rec / "val", transformadas(tam, False))
    assert tr.classes == [carpeta_clase(c) for c in ORD], tr.classes
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    dl_va = DataLoader(va, batch_size=batch, num_workers=4)
    m = modelo_tv(nombre).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl_tr), pct_start=0.1)
    sigma = VARIANTES[var].get("sigma")
    mejor, historial, sin_mejora = None, [], 0
    for ep in range(1, epochs + 1):
        m.train()
        perdida = 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            z = m(x)
            loss = F.cross_entropy(z, y) if sigma is None else -(objetivo_ordinal(y, sigma) * F.log_softmax(z, 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            perdida += float(loss) * len(y)
        pred, real = predecir_tv(m, dl_va, dev)
        ac, ec = acierto_comercial_de([ORD[p] for p in pred], [ORD[r] for r in real])
        acc = round(100 * float(np.mean(np.array(pred) == np.array(real))), 1)
        historial.append({"epoch": ep, "loss": round(perdida / len(tr), 4), "val_acc": acc, "val_acierto_comercial": ac, "val_error_critico": ec})
        if mejor is None or (ac, -ec, acc) > (mejor["val_acierto_comercial"], -mejor["val_error_critico"], mejor["val_acc"]):
            mejor, sin_mejora = historial[-1], 0
            torch.save(m.state_dict(), salida)
        else:
            sin_mejora += 1
        if sin_mejora >= 15:
            break
    return {"mejor_epoch": mejor["epoch"], "epochs_hechas": len(historial), "val_acc": mejor["val_acc"],
            "val_acierto_comercial": mejor["val_acierto_comercial"], "val_error_critico": mejor["val_error_critico"], "historial": historial}


def predecir_tv(m, dl, dev):
    import torch
    m.eval()
    pred, real = [], []
    with torch.no_grad():
        for x, y in dl:
            pred += m(x.to(dev)).argmax(1).cpu().tolist()
            real += y.tolist()
    return pred, real


def entrenar_yolo_cls(nombre, dir_rec, epochs, tam, dev, salida, runs, batch=64):
    """Entrena un clasificador de Ultralytics (yolo11n-cls…) con los mismos aumentos sin tono; elige la época por val."""
    from ultralytics import YOLO
    m = YOLO(f"{nombre}.pt")
    m.train(data=str(dir_rec), epochs=epochs, imgsz=tam, batch=batch, device=dev, optimizer="AdamW", lr0=0.001,
            seed=0, deterministic=True, workers=4, project=str(runs), name=salida.stem, exist_ok=True,
            fliplr=0.5, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, scale=0.2, erasing=0.0, auto_augment=None,
            patience=15, plots=False, verbose=False)
    run = runs / salida.stem
    shutil.copy(run / "weights" / "best.pt", salida)
    shutil.rmtree(run / "weights", ignore_errors=True)   # el peso ya está en pesos/; el resto (results.csv) se respalda
    # Ultralytics elige la época por top-1 en val; aquí se recalcula el acierto comercial en val con el mejor peso
    clf = Clasificador(nombre, salida, dev, tam)
    pred, real = [], []
    for c in ORD:
        for p in sorted((dir_rec / "val" / carpeta_clase(c)).glob("*.jpg")):
            pred.append(p); real.append(c)
    import cv2
    probs = clf.probs([cv2.imread(str(p)) for p in pred])
    pr = [ORD[int(i)] for i in probs.argmax(1)]
    ac, ec = acierto_comercial_de(pr, real)
    import csv
    filas = list(csv.DictReader(open(run / "results.csv"))) if (run / "results.csv").exists() else []
    return {"mejor_epoch": None, "epochs_hechas": len(filas), "val_acc": round(100 * float(np.mean([p == r for p, r in zip(pr, real)])), 1),
            "val_acierto_comercial": ac, "val_error_critico": ec, "historial": None}


class Clasificador:
    """Recortes (BGR, cualquier tamaño) → probabilidades (N, 5) en el orden de ORD."""

    def __init__(self, nombre, peso, dev, tam):
        import torch
        self.nombre, self.dev, self.tam = nombre, dev, tam
        if es_yolo(nombre):
            from ultralytics import YOLO
            self.m = YOLO(str(peso))
        else:
            self.m = modelo_tv(nombre)
            self.m.load_state_dict(torch.load(peso, map_location="cpu"))
            self.m.to(dev).eval()
            self.tf = transformadas(tam, False)

    def probs(self, recortes):
        import torch
        if not recortes:
            return np.zeros((0, 5))
        if es_yolo(self.nombre):
            out = []
            for i in range(0, len(recortes), 64):
                rs = self.m.predict(recortes[i:i + 64], imgsz=self.tam, device=self.dev, verbose=False)
                out += [r.probs.data.cpu().numpy() for r in rs]
            return np.stack(out)
        from PIL import Image
        out = []
        with torch.no_grad():
            for i in range(0, len(recortes), 64):
                x = torch.stack([self.tf(Image.fromarray(r[:, :, ::-1])) for r in recortes[i:i + 64]]).to(self.dev)
                out.append(torch.softmax(self.m(x), 1).cpu().numpy())
        return np.concatenate(out)


# ───────────────────────────── evaluación en el test fijo ─────────────────────────────

def detecciones(peso, imgsz, ds, dev, cache):
    """Predicciones del detector en el test del fold (caché json): {foto: [[x1,y1,x2,y2 norm], clase, conf]}."""
    if cache.exists():
        return leer(cache)
    from ultralytics import YOLO
    m = YOLO(str(peso))
    out = {}
    for p in sorted((ds / "images" / "test").iterdir()):
        r = m.predict(str(p), imgsz=imgsz, conf=0.25, device=dev, verbose=False)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        out[p.name] = [([b[0] / W, b[1] / H, b[2] / W, b[3] / H], ORD[int(c)], round(float(s), 4)) for b, c, s in
                       zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    escribir(cache, out)
    return out


def evaluar(clf, dets, ds, margen, tam):
    """Por cada caja real del test: YOLO solo, YOLO + clasificador (recorte de la detección) y cota (recorte de la caja real)."""
    import cv2
    filas, rec_det, rec_gt, idx = [], [], [], []
    for p in sorted((ds / "images" / "test").iterdir()):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        pr = dets.get(p.name, [])
        for i, (c, b) in enumerate(cajas_de(ds, "test", p.stem)):
            cand = [q for q in pr if iou(b, q[0]) >= 0.5]
            top = max(cand, key=lambda q: q[2]) if cand else None
            fila = {"foto": p.name, "i": i, "real": c, "yolo": top[1] if top else "no_detectada", "conf": top[2] if top else None,
                    "lado_px": round(min((b[2] - b[0]) * w, (b[3] - b[1]) * h), 1)}
            filas.append(fila)
            rec_gt.append(recortar(img, [b[0] * w, b[1] * h, b[2] * w, b[3] * h], margen, tam))
            if top:
                d = top[0]
                rec_det.append(recortar(img, [d[0] * w, d[1] * h, d[2] * w, d[3] * h], margen, tam))
                idx.append(len(filas) - 1)
    p_gt = clf.probs(rec_gt).argmax(1)
    p_det = clf.probs(rec_det).argmax(1)
    for f, k in zip(filas, p_gt):
        f["cota"] = ORD[int(k)]
        f["dos_etapas"] = "no_detectada"
    for j, k in zip(idx, p_det):
        filas[j]["dos_etapas"] = ORD[int(k)]
    return filas


def metricas(filas, clave):
    """Acierto comercial, error crítico y acierto en las 5 clases (no detectada cuenta como fallo)."""
    com = [f for f in filas if f["real"] in COMERCIAL]
    conf = {c: {} for c in COMERCIAL}
    for f in com:
        conf[f["real"]][f[clave]] = conf[f["real"]].get(f[clave], 0) + 1
    pct = lambda cond, xs: round(100 * sum(cond(f) for f in xs) / len(xs), 1) if xs else None
    return {"acierto_comercial": pct(lambda f: f[clave] == f["real"], com),
            "error_critico": pct(lambda f: f[clave] in CRITICO, com),
            "otra_comercial": pct(lambda f: f[clave] in COMERCIAL and f[clave] != f["real"], com),
            "no_detectada": pct(lambda f: f[clave] == "no_detectada", com),
            "acierto_5clases": pct(lambda f: f[clave] == f["real"], filas),
            "confusion_comercial": conf, "n_comerciales": len(com), "n_cajas": len(filas)}


def mcnemar_midp(b, c):
    """McNemar exacto con mid-p (Fagerland et al. 2013): b y c son los pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    acum = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    punto = math.comb(n, k) / 2 ** n
    return round(min(1.0, 2 * acum - punto), 4)


def comparar(filas, a, b, n_boot=2000, semilla=0):
    """Diferencias (b − a) en acierto comercial y error crítico sobre las mismas fresas comerciales: McNemar mid-p e IC
    bootstrap del 95 % remuestreando FOTOS (las fresas de una misma foto no son independientes)."""
    com = [f for f in filas if f["real"] in COMERCIAL]
    def pares(ok):
        x = [ok(f, a) for f in com]; y = [ok(f, b) for f in com]
        return sum(1 for u, v in zip(x, y) if u and not v), sum(1 for u, v in zip(x, y) if v and not u)
    bien = lambda f, k: f[k] == f["real"]
    crit = lambda f, k: f[k] in CRITICO
    fotos = sorted({f["foto"] for f in com})
    por_foto = {p: [f for f in com if f["foto"] == p] for p in fotos}
    rng = random.Random(semilla)
    d_ac, d_ec = [], []
    for _ in range(n_boot):
        muestra = [f for p in rng.choices(fotos, k=len(fotos)) for f in por_foto[p]]
        n = len(muestra)
        d_ac.append(100 * (sum(bien(f, b) for f in muestra) - sum(bien(f, a) for f in muestra)) / n)
        d_ec.append(100 * (sum(crit(f, b) for f in muestra) - sum(crit(f, a) for f in muestra)) / n)
    ic = lambda d: [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]
    ba, ca = pares(bien)
    bc, cc = pares(crit)
    n = len(com)
    return {"acierto": {"dif": round(100 * (ca - ba) / n, 1), "ic95": ic(d_ac), "p_mcnemar": mcnemar_midp(ba, ca), "pierde": ba, "gana": ca},
            "error_critico": {"dif": round(100 * (cc - bc) / n, 1), "ic95": ic(d_ec), "p_mcnemar": mcnemar_midp(bc, cc), "empeora": cc, "mejora": bc}}


# ───────────────────────────── resumen ─────────────────────────────

def resumen(sal, info):
    import statistics as st
    res = {p.stem[4:]: leer(p) for p in sorted(sal.glob("res_*.json"))}
    res = {k: v for k, v in res.items() if v and v.get("estado") == "completo"}
    t = [f"# Segunda etapa (clasificador de recortes) · {info['nombre']}", "",
         f"Actualizado {time.strftime('%d/%m/%Y %H:%M')} (UTC). Partición y folds de «cv{info['cv']}_{info['origen']}»; test fijo de "
         f"{info['n_test']} fotos. Recortes con margen {info['margen']} por lado, {info['tam']} px. Épocas máx. {info['epochs']}.", "",
         "**Regla:** el clasificador se elige por el acierto comercial en la VALIDACIÓN de cada fold (tabla 1). El test (tabla 2) "
         "se reporta para todos, pero solo cuenta la fila del elegido. Métricas como en el benchmark: por cada fresa comercial "
         "real, la predicción más segura que la cubre (IoU ≥ 0,5).", ""]
    grupos = {}
    for k, r in res.items():
        grupos.setdefault(k.rsplit("-f", 1)[0], {})[k.rsplit("-f", 1)[1]] = r
    ms = lambda v: (f"{st.mean(v):.1f} ± {st.stdev(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}") if v else "–"
    t += ["## 1. Clasificador solo, sobre recortes de cajas reales (elección por validación)", "",
          "| Clasificador · variante | Folds | Val: acierto com. | Val: error crít. | Val: top-1 (5 clases) | Test (cota): acierto com. | Test (cota): error crít. | Test (cota): top-1 | Min/fold |",
          "|---|---|---|---|---|---|---|---|---|"]
    orden = sorted(grupos, key=lambda g: -st.mean(r.get("val_acierto_comercial") or 0 for r in grupos[g].values()))
    for g in orden:
        rs = list(grupos[g].values())
        col = lambda k: ms([r[k] for r in rs if r.get(k) is not None])
        cota = lambda k: ms([x for r in rs for x in [next(iter(r["test"].values()))["cota"][k]] if x is not None])   # igual para todo detector
        t.append(f"| **{g}** | {len(rs)} | **{col('val_acierto_comercial')}** | {col('val_error_critico')} | {col('val_acc')} | "
                 f"{cota('acierto_comercial')} | {cota('error_critico')} | {cota('acierto_5clases')} | {col('minutos')} |")
    t += ["", "## 2. Tubería completa en el test fijo: YOLO solo contra YOLO + clasificador (media ± DE entre folds)", "",
          "Δ = (YOLO + clasificador) − (YOLO solo) sobre las mismas fresas; IC 95 % bootstrap por foto y McNemar mid-p, por fold "
          "(se muestra el rango de p entre folds). Con ~150 fresas comerciales solo se distinguen diferencias de ≈ 9–10 puntos.", "",
          "| Detector | Clasificador · variante | Folds | YOLO solo: acierto com. | + clasif.: acierto com. | Δ acierto [IC] | p McNemar | YOLO solo: error crít. | + clasif.: error crít. | Δ error [IC] | p McNemar | Cota (cajas reales): acierto / error |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    dets = sorted({d for r in res.values() for d in r["test"]})
    for d in dets:
        for g in orden:
            rs = [r for r in grupos[g].values() if d in r["test"]]
            if not rs:
                continue
            T = [r["test"][d] for r in rs]
            m_ = lambda k, m: ms([x[k][m] for x in T])
            comp = [x["comparacion"] for x in T]
            dif = lambda k: ms([c[k]["dif"] for c in comp])
            ic = lambda k: f"[{min(c[k]['ic95'][0] for c in comp):.1f}, {max(c[k]['ic95'][1] for c in comp):.1f}]"
            pr = lambda k: f"{min(c[k]['p_mcnemar'] for c in comp):.3f}–{max(c[k]['p_mcnemar'] for c in comp):.3f}"
            t.append(f"| {d} | {g} | {len(rs)} | {m_('yolo', 'acierto_comercial')} | **{m_('dos_etapas', 'acierto_comercial')}** | {dif('acierto')} {ic('acierto')} | {pr('acierto')} "
                     f"| {m_('yolo', 'error_critico')} | **{m_('dos_etapas', 'error_critico')}** | {dif('error_critico')} {ic('error_critico')} | {pr('error_critico')} "
                     f"| {m_('cota', 'acierto_comercial')} / {m_('cota', 'error_critico')} |")
    t += ["", "## 3. Por fold (acierto comercial en test: YOLO solo → YOLO + clasificador; p McNemar del acierto)", "",
          "| Detector | Clasificador · variante | " + " | ".join(f"f{k}" for k in info["folds"]) + " |", "|---" * (2 + len(info["folds"])) + "|"]
    for d in dets:
        for g in orden:
            celdas = []
            for k in info["folds"]:
                r = grupos[g].get(str(k))
                if r and d in r["test"]:
                    x = r["test"][d]
                    celdas.append(f"{x['yolo']['acierto_comercial']} → {x['dos_etapas']['acierto_comercial']} (p {x['comparacion']['acierto']['p_mcnemar']})")
                else:
                    celdas.append("–")
            t.append(f"| {d} | {g} | " + " | ".join(celdas) + " |")
    pend = [k for k in info["trabajos"] if k not in res]
    if pend:
        t += ["", "**Pendientes:** " + ", ".join(pend)]
    t += ["", "Lectura: la segunda etapa se adopta solo si el IC de Δ error crítico queda por debajo de cero sin que baje el acierto "
          "comercial, para el clasificador elegido por validación. La cota (cajas reales) dice cuánto se pierde por detección imperfecta. "
          f"Pesos: release `benchmark-{info['nombre']}`; detecciones en caché: det_*.json. Código: modelos/kaggle_dos_etapas.py."]
    (sal / "RESUMEN.md").write_text("\n".join(t) + "\n", encoding="utf-8")


# ───────────────────────────── principal ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion-de", default="6lotes", help="corrida cuya partición y folds de CV se reusan (cv<K>_<origen>)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--detectores", nargs="+", default=["yolov9t-1024", "yolo12n-1024"], help="pesos de la CV (release benchmark-cv<K>_<origen>)")
    ap.add_argument("--clasificadores", nargs="+", default=CLASIF)
    ap.add_argument("--variantes", nargs="+", default=list(VARIANTES))
    ap.add_argument("--margen", type=float, default=0.15)
    ap.add_argument("--tam", type=int, default=224)
    ap.add_argument("--min-lado", type=int, default=24)
    ap.add_argument("--sobremuestreo", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--horas", type=float, default=11.0)
    ap.add_argument("--salida", default="/kaggle/working/benchmark")
    ap.add_argument("--sin-github", action="store_true")
    ap.add_argument("--secreto", default=None)
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--max-fotos", type=int, default=0, help=argparse.SUPPRESS)   # pruebas rápidas del código
    a = ap.parse_args()
    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"
    fin = time.time() + a.horas * 3600
    ks = a.folds or list(range(1, a.cv + 1))
    origen = a.particion_de
    corrida_cv = f"cv{a.cv}_{origen}"
    nombre = a.nombre or f"dos_etapas_{origen}"
    sal = Path(a.salida) / nombre
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "pesos").mkdir(exist_ok=True)
    rama = kb.RAMA
    subprocess.run(["git", "fetch", "-q", "origin", rama], cwd=REPO, capture_output=True)
    r = subprocess.run(["git", "show", f"origin/{rama}:benchmark/{corrida_cv}/particion.json"], cwd=REPO, capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        sys.exit(f"No está benchmark/{corrida_cv}/particion.json en la rama {rama}: hay que correr antes la validación cruzada.")
    part = json.loads(r.stdout)
    folds = part["folds"]
    fijo = {x: "test" for x in part["test"]} | {x: "train" for xs in folds.values() for x in xs}
    from kaggle_fresas import revisiones
    ds = Path("/tmp/ds_fresas_de")
    kb.particionar(revisiones(), ds, fijo=fijo, sobremuestreo=1)
    import hashlib
    huella = hashlib.sha1(json.dumps([part, a.margen, a.tam, a.min_lado, a.sobremuestreo], sort_keys=True).encode()).hexdigest()[:12]
    trabajos = [f"{c}_{v}-f{k}" for k in ks for c in a.clasificadores for v in a.variantes if not (es_yolo(c) and v != "ce")]
    info = {"nombre": nombre, "origen": origen, "cv": a.cv, "folds": ks, "margen": a.margen, "tam": a.tam, "epochs": a.epochs,
            "detectores": a.detectores, "trabajos": trabajos, "n_test": len(part["test"]), "huella_particion": huella,
            "inicio": time.strftime("%Y-%m-%d %H:%M")}
    escribir(sal / "config.json", info)
    for p in sal.glob("res_*.json"):            # lo que falló se reintenta
        if (leer(p) or {}).get("estado") != "completo":
            p.unlink()
    respaldo = kb.Respaldo(sal, nombre, not a.sin_github, a.secreto, huella)
    log(f"Corrida «{nombre}» · folds {ks} · {len(trabajos)} trabajos · GPU: {torch.cuda.get_device_name(0) if dev == 0 else 'cpu'}")
    for k in ks:
        pend = [t for t in trabajos if t.endswith(f"-f{k}") and not (sal / f"res_{t}.json").exists()]
        if not pend:
            log(f"fold {k}: todo completo, se salta")
            continue
        dk = Path(f"/tmp/de_f{k}")
        kb.armar_fold(ds, folds, k, dk, 1)
        if a.max_fotos:   # pruebas rápidas: deja pocas fotos en cada parte
            for sp in ("train", "val", "test"):
                for p in sorted((dk / "images" / sp).iterdir())[a.max_fotos:]:
                    p.unlink(); (dk / "labels" / sp / f"{p.stem}.txt").unlink()
        rec = Path(f"/tmp/de_rec_f{k}")
        indice, tabla = armar_recortes(dk, rec, a.margen, a.tam, a.sobremuestreo, a.min_lado)
        log(f"fold {k}: recortes " + " · ".join(f"{s} {sum(v.values())}" for s, v in tabla.items()))
        dets = {}
        Path("/tmp/de_detectores").mkdir(exist_ok=True)
        for d in a.detectores:
            peso = Path("/tmp/de_detectores") / f"{d}-f{k}.pt"   # fuera de sal/pesos: no se resube al release
            if not peso.exists():
                urllib.request.urlretrieve(f"https://github.com/{GH}/releases/download/benchmark-{corrida_cv}/{d}-f{k}.pt", peso)
            dets[d] = detecciones(peso, 1024 if "1024" in d else 640, dk, dev, sal / f"det_{d}-f{k}.json")
        for t in pend:
            if time.time() > fin - 20 * 60:
                log(f"{t}: sin tiempo, se deja para otra sesión")
                continue
            clf_n, var = t.rsplit("-f", 1)[0].rsplit("_", 1)   # «mobilenet_v3_small_ordinal-f2» → mobilenet_v3_small, ordinal
            t0 = time.time()
            peso = sal / "pesos" / f"{t}.pt"
            try:
                res = (entrenar_yolo_cls(clf_n, rec, a.epochs, a.tam, dev, peso, sal / "runs") if es_yolo(clf_n)
                       else entrenar_tv(clf_n, var, rec, a.epochs, a.tam, dev, peso))
                clf = Clasificador(clf_n, peso, dev, a.tam)
                res["test"] = {}
                for d, dd in dets.items():
                    filas = evaluar(clf, dd, dk, a.margen, a.tam)
                    escribir(sal / f"fresas_{t}_{d}.json", filas)
                    res["test"][d] = {"yolo": metricas(filas, "yolo"), "dos_etapas": metricas(filas, "dos_etapas"),
                                      "cota": metricas(filas, "cota"), "comparacion": comparar(filas, "yolo", "dos_etapas")}
                res |= {"trabajo": t, "clasificador": clf_n, "variante": var, "fold": k, "estado": "completo",
                        "minutos": round((time.time() - t0) / 60, 1), "recuento": tabla}
                x = res["test"][a.detectores[0]]
                log(f"✓ {t}: val acierto com. {res['val_acierto_comercial']} · test {a.detectores[0]}: YOLO {x['yolo']['acierto_comercial']} → "
                    f"dos etapas {x['dos_etapas']['acierto_comercial']} (p {x['comparacion']['acierto']['p_mcnemar']}) · cota {x['cota']['acierto_comercial']} · {res['minutos']} min")
            except Exception as e:
                import traceback; traceback.print_exc()
                res = {"trabajo": t, "estado": "error", "error": f"{type(e).__name__}: {str(e)[:400]}"}
                log(f"✗ {t}: {res['error']}")
            escribir(sal / f"res_{t}.json", res)
            resumen(sal, info)
            respaldo.sincronizar()
    resumen(sal, info)
    respaldo.sincronizar()
    print((sal / "RESUMEN.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
