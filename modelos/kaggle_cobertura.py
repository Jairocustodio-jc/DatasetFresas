"""Mapas de calor con métrica de cobertura: ¿cuánta superficie de la fresa usa la red al decidir el estado? Kaggle o PC.

    !pip -q install ultralytics
    !git clone -q -b claude/jolly-heisenberg-qa6cj9 https://github.com/Jairocustodio-jc/DatasetFresas
    %cd DatasetFresas
    !python modelos/kaggle_cobertura.py --particion-de 6lotes --fold 1 --clasificador mobilenet_v3_small_ce --detector yolov9t-1024

Para cada fresa comercial del test fijo del fold:
  - Grad-CAM del CLASIFICADOR de recortes (última capa convolucional, objetivo = logit de la clase predicha) sobre el
    recorte de la caja real. Grad-CAM es específico de clase y, en redes con pooling global + capa lineal, coincide con
    CAM y HiResCAM (Draelos & Carr 2020).
  - EigenCAM del DETECTOR (primer componente principal de P3, la entrada de la cabeza), solo de referencia: no distingue
    clase y queda a 1/8 de la resolución.
  - Cobertura dentro de la fresa, aproximada por la elipse inscrita en la caja real (sin segmentar):
      cobertura@τ   = fracción del área de la elipse con mapa ≥ τ·máximo (τ = 0,3 / 0,5 / 0,7);
      entropía norm. = entropía del mapa normalizado dentro de la elipse / log(N)  (1 = uniforme, 0 = un solo punto);
      gini          = concentración del mapa dentro de la elipse (0 = uniforme, 1 = un punto)  [Chalasani et al. 2020];
      masa dentro   = energía del mapa dentro de la elipse / energía total del recorte  [relevance mass accuracy, Arras 2022;
                      energy-based pointing game, Score-CAM].
    Controles: mapa uniforme y mapas aleatorios a la resolución nativa de la capa (7×7 a 224 px).
  - Comparación exploratoria entre fresas bien clasificadas y con error crítico: Mann-Whitney (p por permutación) y delta
    de Cliff. Un mapa amplio no demuestra que la red «use» toda la fresa; solo que la activa. Se dice así en la tesis.
Salida: cobertura_<clasificador>_<detector>_f<k>.{json,md,png} en /kaggle/working (o --salida).
"""
import argparse
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "modelos"))
import kaggle_benchmark as kb
import kaggle_dos_etapas as de
from kaggle_fresas import ORD, iou

TAUS = (0.3, 0.5, 0.7)


# ───────────────────────────── mapas ─────────────────────────────

class GradCAM:
    """Grad-CAM sobre la última capa convolucional de un clasificador (torchvision o yolo-cls)."""

    def __init__(self, nombre, peso, dev, tam):
        import torch
        self.nombre, self.dev, self.tam, self.act, self.grad = nombre, dev, tam, None, None
        if de.es_yolo(nombre):
            from ultralytics import YOLO
            self.m = YOLO(str(peso)).model.to(dev).eval()
            capa = self.m.model[-1].conv                 # Classify: conv → pool → linear; la conv es la última espacial
        else:
            self.m = de.modelo_tv(nombre)
            self.m.load_state_dict(torch.load(peso, map_location="cpu"))
            self.m.to(dev).eval()
            capa = de.capa_cam(self.m, nombre)
        capa.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def entrada(self, rec):
        import torch
        x = rec[:, :, ::-1].astype(np.float32) / 255          # BGR → RGB
        if not de.es_yolo(self.nombre):
            x = (x - np.float32(de.MEDIA)) / np.float32(de.DESV)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32))[None].to(self.dev)

    def __call__(self, rec):
        """Recorte BGR tam×tam → (clase predicha, probabilidades, mapa tam×tam ≥ 0, resolución nativa (h, w))."""
        import torch
        import torch.nn.functional as F
        self.m.zero_grad(set_to_none=True)
        out = self.m(self.entrada(rec))
        logits = out[1] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        k = int(probs.argmax())
        logits[0, k].backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        nat = tuple(cam.shape[2:])
        cam = F.interpolate(cam, size=(self.tam, self.tam), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        return ORD[k], probs, cam, nat


class EigenCAMDetector:
    """EigenCAM de la capa P3 del detector, recortado a la caja (como en kaggle_atencion.py)."""

    def __init__(self, peso, imgsz, dev):
        from ultralytics import YOLO
        self.m, self.imgsz, self.dev, self.feats = YOLO(str(peso)), imgsz, dev, {}
        self.m.predict(np.zeros((64, 64, 3), np.uint8), imgsz=imgsz, device=dev, verbose=False)
        self.m.predictor.model.model.model[-1].register_forward_hook(
            lambda mod, inp, out: self.feats.__setitem__("p3", inp[0][0].detach().float().cpu()))

    def mapa_foto(self, img):
        """Mapa EigenCAM a tamaño de la foto y las detecciones [(caja norm, clase, conf)]."""
        import cv2
        r = self.m.predict(img, imgsz=self.imgsz, conf=0.25, device=self.dev, verbose=False)[0]
        h0, w0 = img.shape[:2]
        a = self.feats["p3"]
        a = a[0] if a.dim() == 4 else a
        C, H, W = a.shape
        A = a.reshape(C, -1).T.numpy()
        A = A - A.mean(0)
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        cam = (A @ vt[0]).reshape(H, W)
        if cam.max() < -cam.min():
            cam = -cam
        cam = np.maximum(cam, 0)
        r_ = min(self.imgsz / h0, self.imgsz / w0)
        nh, nw = round(h0 * r_), round(w0 * r_)
        full = cv2.resize(cam, (W * 8, H * 8), interpolation=cv2.INTER_CUBIC)
        py, px = max((full.shape[0] - nh) // 2, 0), max((full.shape[1] - nw) // 2, 0)
        full = full[py:py + nh, px:px + nw]
        full = cv2.resize(full, (w0, h0), interpolation=cv2.INTER_CUBIC)
        dets = [([b[0] / w0, b[1] / h0, b[2] / w0, b[3] / h0], ORD[int(c)], float(s)) for b, c, s in
                zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        return np.maximum(full, 0), dets, (H, W)


# ───────────────────────────── cobertura ─────────────────────────────

def elipse(tam, caja_px, margen):
    """Máscara de la elipse inscrita en la caja real, en coordenadas del recorte cuadrado con margen."""
    x1, y1, x2, y2 = caja_px
    lado = max(x2 - x1, y2 - y1) * (1 + 2 * margen)
    a, b = (x2 - x1) / lado / 2, (y2 - y1) / lado / 2          # semiejes normalizados al lado del recorte
    yy, xx = np.mgrid[0:tam, 0:tam]
    u, v = (xx + 0.5) / tam - 0.5, (yy + 0.5) / tam - 0.5
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def cobertura(cam, M):
    """Métricas de cobertura de un mapa (≥ 0) dentro de la máscara M."""
    v = cam[M].astype(np.float64)
    tot = float(cam.sum())
    if v.size == 0 or v.max() <= 0:
        return {f"cobertura@{t}": 0.0 for t in TAUS} | {"entropia": 0.0, "gini": 1.0, "masa_dentro": 0.0}
    n = v / v.max()
    p = v / v.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(v.size))
    s = np.sort(v)
    gini = float((2 * np.arange(1, s.size + 1) - s.size - 1).dot(s) / (s.size * s.sum()))
    return {f"cobertura@{t}": round(float((n >= t).mean()), 4) for t in TAUS} | {
        "entropia": round(ent, 4), "gini": round(gini, 4), "masa_dentro": round(float(v.sum() / tot) if tot else 0.0, 4)}


def controles(tam, M, nat, n=100, semilla=0):
    """Cobertura de un mapa uniforme y de mapas aleatorios a la resolución nativa, interpolados como el Grad-CAM."""
    import cv2
    rng = np.random.default_rng(semilla)
    uni = cobertura(np.ones((tam, tam)), M)
    azar = [cobertura(cv2.resize(rng.random(nat).astype(np.float32), (tam, tam), interpolation=cv2.INTER_LINEAR), M) for _ in range(n)]
    return {"uniforme": uni, "aleatorio": {k: round(float(np.mean([a[k] for a in azar])), 4) for k in uni}}


def mann_whitney(x, y, n_perm=10000, semilla=0):
    """U de Mann-Whitney con p bilateral por permutación y delta de Cliff (x contra y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return {"n_x": len(x), "n_y": len(y), "p": None, "cliff": None}
    def u(a, b):
        return float((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())
    u0 = u(x, y)
    rng = np.random.default_rng(semilla)
    todo = np.concatenate([x, y])
    mitad = len(x) * len(y) / 2
    k = 0
    for _ in range(n_perm):
        rng.shuffle(todo)
        if abs(u(todo[:len(x)], todo[len(x):]) - mitad) >= abs(u0 - mitad) - 1e-9:
            k += 1
    return {"n_x": len(x), "n_y": len(y), "U": u0, "p": round((k + 1) / (n_perm + 1), 4),
            "cliff": round(2 * u0 / (len(x) * len(y)) - 1, 3), "mediana_x": round(float(np.median(x)), 4), "mediana_y": round(float(np.median(y)), 4)}


# ───────────────────────────── principal ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion-de", default="6lotes")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--clasificador", default="mobilenet_v3_small_ce", help="trabajo de kaggle_dos_etapas (sin -f<k>)")
    ap.add_argument("--detector", default="yolov9t-1024")
    ap.add_argument("--margen", type=float, default=0.15)
    ap.add_argument("--tam", type=int, default=224)
    ap.add_argument("--errores", type=int, default=6, help="fresas con error crítico en la figura")
    ap.add_argument("--aciertos", type=int, default=4)
    ap.add_argument("--salida", default="/kaggle/working" if Path("/kaggle/working").exists() else ".")
    ap.add_argument("--peso-clasificador", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--peso-detector", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--max-fotos", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    import cv2
    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"
    k, origen = a.fold, a.particion_de
    corrida_cv = f"cv{a.cv}_{origen}"
    subprocess.run(["git", "fetch", "-q", "origin", kb.RAMA], cwd=REPO, capture_output=True)
    part = json.loads(subprocess.run(["git", "show", f"origin/{kb.RAMA}:benchmark/{corrida_cv}/particion.json"], cwd=REPO,
                                     capture_output=True, text=True).stdout)
    fijo = {x: "test" for x in part["test"]} | {x: "train" for xs in part["folds"].values() for x in xs}
    from kaggle_fresas import revisiones
    ds = Path("/tmp/ds_fresas_cob")
    kb.particionar(revisiones(), ds, fijo=fijo, sobremuestreo=1)
    dk = Path(f"/tmp/cob_f{k}")
    kb.armar_fold(ds, part["folds"], k, dk, 1)
    fotos = sorted((dk / "images" / "test").iterdir())
    if a.max_fotos:
        fotos = fotos[:a.max_fotos]
    P = Path("/tmp/cob_pesos")
    P.mkdir(exist_ok=True)
    nombre_clf = a.clasificador.rsplit("_", 1)[0]
    pc = Path(a.peso_clasificador) if a.peso_clasificador else P / f"{a.clasificador}-f{k}.pt"
    pd = Path(a.peso_detector) if a.peso_detector else P / f"{a.detector}-f{k}.pt"
    for p, rel in ((pc, f"benchmark-dos_etapas_{origen}"), (pd, f"benchmark-{corrida_cv}")):
        if not p.exists():
            urllib.request.urlretrieve(f"https://github.com/{kb.GH}/releases/download/{rel}/{p.name}", p)
    gc = GradCAM(nombre_clf, pc, dev, a.tam)
    ed = EigenCAMDetector(pd, 1024 if "1024" in a.detector else 640, dev)

    casos = []
    for p in fotos:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        cajas = [(c, b) for c, b in de.cajas_de(dk, "test", p.stem) if c in de.COMERCIAL]
        if not cajas:
            continue
        mapa_det, dets, nat_det = ed.mapa_foto(img)
        for i, (c, b) in enumerate(cajas):
            px = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
            rec = de.recortar(img, px, a.margen, a.tam)
            pred, probs, cam, nat = gc(rec)
            M = elipse(a.tam, px, a.margen)
            cand = [q for q in dets if iou(b, q[0]) >= 0.5]
            yolo = max(cand, key=lambda q: q[2])[1] if cand else "no_detectada"
            # EigenCAM del detector recortado igual que la foto (mismo cuadrado con margen)
            cx, cy = (px[0] + px[2]) / 2, (px[1] + px[3]) / 2
            lado = max(px[2] - px[0], px[3] - px[1]) * (1 + 2 * a.margen)
            lienzo = np.zeros((int(round(lado)), int(round(lado))), np.float32)
            X1, Y1 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
            sx1, sy1 = max(X1, 0), max(Y1, 0)
            sx2, sy2 = min(X1 + lienzo.shape[1], w), min(Y1 + lienzo.shape[0], h)
            if sx2 > sx1 and sy2 > sy1:
                lienzo[sy1 - Y1:sy2 - Y1, sx1 - X1:sx2 - X1] = mapa_det[sy1:sy2, sx1:sx2]
            cam_det = cv2.resize(lienzo, (a.tam, a.tam))
            resultado = "bien" if pred == c else ("error_critico" if pred in de.CRITICO else "otra_comercial")
            casos.append({"foto": p.name, "i": i, "real": c, "clasificador": pred, "yolo": yolo, "resultado": resultado,
                          "conf": round(float(probs.max()), 3), "lado_px": round(min(px[2] - px[0], px[3] - px[1]), 1),
                          "cobertura": cobertura(cam, M), "cobertura_detector": cobertura(cam_det, M),
                          "_rec": rec, "_cam": cam, "_cam_det": cam_det, "_M": M})
    if not casos:
        sys.exit("No hay fresas comerciales en el test elegido.")
    nat = nat if casos else (7, 7)
    ctrl = controles(a.tam, casos[0]["_M"], nat)
    grupos = {g: [x for x in casos if x["resultado"] == g] for g in ("bien", "error_critico", "otra_comercial")}
    claves = list(casos[0]["cobertura"])
    resumen = {g: {kk: round(float(np.median([x["cobertura"][kk] for x in xs])), 4) for kk in claves} if xs else None for g, xs in grupos.items()}
    resumen_det = {g: {kk: round(float(np.median([x["cobertura_detector"][kk] for x in xs])), 4) for kk in claves} if xs else None for g, xs in grupos.items()}
    pruebas = {}
    for kk in claves:
        bien = [x["cobertura"][kk] for x in grupos["bien"]]
        pruebas[kk] = {"bien_vs_error_critico": mann_whitney(bien, [x["cobertura"][kk] for x in grupos["error_critico"]]),
                       "bien_vs_cualquier_error": mann_whitney(bien, [x["cobertura"][kk] for x in grupos["error_critico"] + grupos["otra_comercial"]])}
    nombre = f"cobertura_{a.clasificador}_{a.detector}_f{k}"
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    out = {"clasificador": a.clasificador, "detector": a.detector, "fold": k, "margen": a.margen, "tam": a.tam,
           "resolucion_nativa_clasificador": list(nat), "resolucion_nativa_detector": list(nat_det), "n": {g: len(xs) for g, xs in grupos.items()},
           "mediana_clasificador": resumen, "mediana_detector_eigencam": resumen_det, "controles": ctrl, "pruebas": pruebas,
           "fresas": [{kk: v for kk, v in x.items() if not kk.startswith("_")} for x in casos]}
    (sal / f"{nombre}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # tabla
    f = lambda d, kk: "–" if d is None else f"{d[kk]:.3f}"
    t = [f"# Cobertura de los mapas de calor · {a.clasificador} · fold {k} · detector {a.detector}", "",
         f"{len(casos)} fresas comerciales del test fijo ({', '.join(f'{g}: {len(xs)}' for g, xs in grupos.items())}). "
         f"Grad-CAM del clasificador a resolución nativa {nat[0]}×{nat[1]} (recorte de {a.tam} px); EigenCAM del detector a "
         f"{nat_det[0]}×{nat_det[1]} (foto a {ed.imgsz} px), solo de referencia. Cobertura medida dentro de la elipse inscrita en la caja real.", "",
         "| Métrica (mediana) | Bien | Error crítico | Otra comercial | Control uniforme | Control aleatorio | p bien vs error crít. | delta de Cliff |",
         "|---|---|---|---|---|---|---|---|"]
    for kk in claves:
        pr = pruebas[kk]["bien_vs_error_critico"]
        t.append(f"| {kk} | {f(resumen['bien'], kk)} | {f(resumen['error_critico'], kk)} | {f(resumen['otra_comercial'], kk)} | "
                 f"{ctrl['uniforme'][kk]:.3f} | {ctrl['aleatorio'][kk]:.3f} | {pr['p'] if pr['p'] is not None else '–'} | {pr['cliff'] if pr['cliff'] is not None else '–'} |")
    t += ["", "EigenCAM del detector (misma elipse; no distingue clase):", "",
          "| Métrica (mediana) | Bien | Error crítico | Otra comercial |", "|---|---|---|---|"]
    for kk in claves:
        t.append(f"| {kk} | {f(resumen_det['bien'], kk)} | {f(resumen_det['error_critico'], kk)} | {f(resumen_det['otra_comercial'], kk)} |")
    t += ["", "Lectura: si los errores críticos tienen cobertura y entropía más bajas (y gini más alto) que los aciertos, con p < 0,05 "
          "y delta de Cliff ≥ 0,33, la red decide por una zona pequeña de la fresa en los errores; es un indicio, no una prueba de "
          "causalidad. Con pocos errores (< 20) solo se detectan efectos grandes. Análisis exploratorio.",
          f"Archivos: {nombre}.json (todas las fresas), {nombre}.png (figura). Código: modelos/kaggle_cobertura.py."]
    (sal / f"{nombre}.md").write_text("\n".join(t) + "\n", encoding="utf-8")

    # figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    random.seed(1)
    sel = grupos["error_critico"][:a.errores] + random.sample(grupos["bien"], min(a.aciertos, len(grupos["bien"])))
    fig, ax = plt.subplots(len(sel), 3, figsize=(7.5, 2.5 * len(sel)), squeeze=False)
    for i, x in enumerate(sel):
        rec = x["_rec"][:, :, ::-1]
        ax[i, 0].imshow(rec)
        ax[i, 0].set_title(f"real: {x['real']} · YOLO: {x['yolo']}", fontsize=7)
        for j, (cm, tit) in enumerate(((x["_cam"], "Grad-CAM clasificador"), (x["_cam_det"], "EigenCAM detector")), 1):
            n = cm / (cm.max() + 1e-8)
            heat = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
            ax[i, j].imshow((0.55 * rec + 0.45 * heat).astype(np.uint8))
            ax[i, j].contour(x["_M"], levels=[0.5], colors="white", linewidths=0.6)
            col = {"bien": "green", "error_critico": "red", "otra_comercial": "orange"}[x["resultado"]]
            cob = x["cobertura" if j == 1 else "cobertura_detector"]
            ax[i, j].set_title(f"{tit}\n{'clasif.: ' + x['clasificador'] if j == 1 else ''} cob@0.5 {cob['cobertura@0.5']:.2f} · entr. {cob['entropia']:.2f}",
                               fontsize=7, color=col if j == 1 else "black")
        for e in ax[i]:
            e.axis("off")
    fig.suptitle(f"{a.clasificador} · fold {k} · verde = bien · rojo = error crítico · elipse = zona medida", fontsize=8)
    plt.tight_layout()
    plt.savefig(sal / f"{nombre}.png", dpi=110)
    print((sal / f"{nombre}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
