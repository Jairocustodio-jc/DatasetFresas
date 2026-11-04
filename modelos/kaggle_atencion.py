"""Mapas de atención (EigenCAM) de dos modelos sobre las fresas comerciales del test del benchmark. Corre en Kaggle o en la PC.

    !pip -q install ultralytics
    !git clone -q -b claude/jolly-heisenberg-qa6cj9 https://github.com/Jairocustodio-jc/DatasetFresas
    %cd DatasetFresas
    !python modelos/kaggle_atencion.py --modelos yolo26n:640 yolo11n-hsv-1024:1024 --errores 6 --aciertos 4

Descarga los pesos del release benchmark-6lotes, rearma el test con la partición guardada en la rama yolo-benchmark y
guarda /kaggle/working/atencion_comerciales.png. EigenCAM: primer componente principal de la capa P3 (stride 8) que entra a
la cabeza de detección; muestra qué zonas activan más la red (no es específico de la clase).
"""
import sys, json, random
from pathlib import Path
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/DatasetFresas/modelos")
from kaggle_fresas import ORD, iou
from ultralytics import YOLO
import argparse, subprocess, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--modelos", nargs="+", default=["yolo26n:640", "yolo11n-hsv-1024:1024"])
ap.add_argument("--errores", type=int, default=6, help="fresas con error crítico del primer modelo")
ap.add_argument("--aciertos", type=int, default=4)
ap.add_argument("--salida", default="/kaggle/working" if Path("/kaggle/working").exists() else ".")
a = ap.parse_args()
MODELOS = [(m.split(":")[0], int(m.split(":")[1])) for m in a.modelos]
S = Path("/tmp/atencion")
REPO = Path(__file__).resolve().parent.parent
(S / "pesos").mkdir(parents=True, exist_ok=True)
for n, _ in MODELOS:
    if not (S / "pesos" / f"{n}.pt").exists():
        urllib.request.urlretrieve(f"https://github.com/Jairocustodio-jc/DatasetFresas/releases/download/benchmark-6lotes/{n}.pt",
                                   S / "pesos" / f"{n}.pt")
subprocess.run(["git", "fetch", "-q", "origin", "yolo-benchmark"], cwd=REPO)
part = json.loads(subprocess.run(["git", "show", "origin/yolo-benchmark:benchmark/6lotes/particion.json"], cwd=REPO,
                                 capture_output=True, text=True).stdout)
import kaggle_benchmark as kb, kaggle_fresas as kf
kb.particionar(kf.revisiones(), S / "ds", fijo={x: sp for sp, xs in part.items() for x in xs})

def preparar(nombre, imgsz):
    m = YOLO(str(S / "pesos" / f"{nombre}.pt"))
    feats = {}
    m.predict(np.zeros((64, 64, 3), np.uint8), imgsz=imgsz, verbose=False)   # inicializa el predictor
    det = m.predictor.model.model.model[-1]
    det.register_forward_hook(lambda mod, inp, out: feats.__setitem__("p3", inp[0][0].detach().float()))
    return m, feats

def eigencam(f):   # f: lista/tensor (B,C,H,W) -> mapa HxW
    a = f[0] if f.dim() == 4 else f
    C, H, W = a.shape
    A = a.reshape(C, -1).T.numpy(); A = A - A.mean(0)
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    cam = (A @ vt[0]).reshape(H, W)
    if cam.max() < -cam.min(): cam = -cam
    cam = np.maximum(cam, 0); return cam / (cam.max() + 1e-8)

def letterbox_crop(cam, shape, imgsz):   # deshace el letterbox de Ultralytics
    h0, w0 = shape; r = min(imgsz / h0, imgsz / w0)
    nh, nw = round(h0 * r), round(w0 * r)
    Hc, Wc = cam.shape; s = imgsz / max(Hc, Wc) if False else None
    full = cv2.resize(cam, (Wc * 8, Hc * 8), interpolation=cv2.INTER_CUBIC)
    py, px = (full.shape[0] - nh) // 2, (full.shape[1] - nw) // 2
    full = full[max(py,0):max(py,0)+nh, max(px,0):max(px,0)+nw]
    return cv2.resize(full, (w0, h0), interpolation=cv2.INTER_CUBIC)

mods = [(n, z, *preparar(n, z)) for n, z in MODELOS]
casos = []
for p in sorted((S / "ds/images/test").iterdir()):
    img = cv2.imread(str(p))[:, :, ::-1]; h0, w0 = img.shape[:2]
    gts = []
    for l in (S / "ds/labels/test" / (p.stem + ".txt")).read_text().split("\n"):
        if l.strip():
            c, x, y, w, h = map(float, l.split())
            if ORD[int(c)] in ("commercial-basic", "commercial-high"):
                gts.append((ORD[int(c)], [x-w/2, y-h/2, x+w/2, y+h/2]))
    if not gts: continue
    res = {}
    for n, z, m, feats in mods:
        r = m.predict(img[:, :, ::-1].copy(), imgsz=z, conf=0.25, verbose=False)[0]
        pr = [([b[0]/w0, b[1]/h0, b[2]/w0, b[3]/h0], ORD[int(c)], float(s)) for b, c, s in
              zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        res[n] = (pr, letterbox_crop(eigencam(feats["p3"]), (h0, w0), z))
    for real, gb in gts:
        if (gb[2]-gb[0]) * w0 < 40: continue
        preds = {}
        for n in res:
            cand = [q for q in res[n][0] if iou(gb, q[0]) >= 0.5]
            preds[n] = max(cand, key=lambda q: q[2])[1] if cand else "—"
        casos.append((p, real, gb, preds, {n: res[n][1] for n in res}, img))
critico = lambda c, n: c[3][n] in ("early-pink", "overripe")
m0 = MODELOS[0][0]
err = [c for c in casos if critico(c, m0)]
ok = [c for c in casos if c[3][m0] == c[1]]
random.seed(1); sel = err[:a.errores] + random.sample(ok, min(a.aciertos, len(ok)))
print(len(casos), "fresas comerciales,", len(err), f"errores críticos de {m0}")
fig, ax = plt.subplots(len(sel), 1 + len(MODELOS), figsize=(2.5 * (1 + len(MODELOS)), 2.6 * len(sel)), squeeze=False)
for i, (p, real, gb, preds, cams, img) in enumerate(sel):
    h0, w0 = img.shape[:2]; mx = 0.25
    bw, bh = gb[2]-gb[0], gb[3]-gb[1]
    x1, y1 = int(max(0, (gb[0]-mx*bw)*w0)), int(max(0, (gb[1]-mx*bh)*h0))
    x2, y2 = int(min(w0, (gb[2]+mx*bw)*w0)), int(min(h0, (gb[3]+mx*bh)*h0))
    crop = img[y1:y2, x1:x2]
    ax[i, 0].imshow(crop); ax[i, 0].set_title(f"real: {real}", fontsize=8)
    for j, (n, _) in enumerate(MODELOS, 1):
        cm = cams[n][y1:y2, x1:x2]; cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
        heat = cv2.applyColorMap((cm * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        ax[i, j].imshow((0.55 * crop + 0.45 * heat).astype(np.uint8))
        col = "green" if preds[n] == real else ("red" if preds[n] in ("early-pink", "overripe") else "orange")
        ax[i, j].set_title(f"{n}: {preds[n]}", fontsize=8, color=col)
    for e in ax[i]: e.axis("off")
fig.suptitle("EigenCAM (capa P3) · verde = bien · rojo = error crítico · naranja = otra comercial", fontsize=9)
plt.tight_layout(); out = Path(a.salida) / "atencion_comerciales.png"; plt.savefig(out, dpi=110); print(out)
