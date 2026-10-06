"""Métricas finales en el test del benchmark (P, R, especificidad, mAP, acierto comercial). Corre en Kaggle con GPU.

    !pip -q install ultralytics
    %cd /kaggle/working
    !rm -rf DatasetFresas
    !git clone -q -b claude/jolly-heisenberg-qa6cj9 https://github.com/Jairocustodio-jc/DatasetFresas
    %cd DatasetFresas
    !python modelos/kaggle_metricas.py

Descarga los pesos del release benchmark-6lotes, rearma el test con la partición guardada (rama yolo-benchmark) y por modelo:
  - mAP@0.5, mAP@0.5:0.95, precisión y recall (Ultralytics, split test);
  - especificidad por clase E = TN / (TN + FP), desde la matriz de confusión en conteos (confianza ≥ 0,25, IoU ≥ 0,45);
  - acierto comercial y error crítico (predicción más segura por fresa comercial, IoU ≥ 0,5, confianza ≥ 0,25).
Guarda /kaggle/working/test_metricas/ (metricas.json + matrices de confusión) y lo sube a la rama yolo-benchmark,
carpeta benchmark/6lotes/test_metricas/, con el token de los Secrets (GITHUB_TOKEN o key_gh_strawberry).
"""
import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GH = "Jairocustodio-jc/DatasetFresas"
POR_DEFECTO = ["yolov8n", "yolov9t-1024", "yolo11n", "yolo26n", "yolo11n-hsv-1024",
               "yolov8x", "yolo11x", "yolov9e", "yolo26x"]


def subir(carpeta, tok):
    """Copia la carpeta a benchmark/6lotes/test_metricas/ de la rama yolo-benchmark sin tocar lo demás."""
    url = f"https://x-access-token:{tok}@github.com/{GH}.git"
    d = Path(tempfile.mkdtemp())
    run = lambda *a: subprocess.run(["git", *a], cwd=d, capture_output=True, text=True)
    run("init", "-q")
    run("config", "user.email", "kaggle@fresas")
    run("config", "user.name", "Kaggle métricas")
    run("fetch", "-q", url, "yolo-benchmark")
    run("checkout", "-q", "-B", "yolo-benchmark", "FETCH_HEAD")
    shutil.copytree(carpeta, d / "benchmark" / "6lotes" / "test_metricas", dirs_exist_ok=True)
    run("add", "-A")
    run("commit", "-q", "-m", "Métricas finales en test (especificidad, acierto comercial)")
    r = run("push", "-q", url, "HEAD:yolo-benchmark")
    print("Subido a GitHub." if r.returncode == 0 else "⚠ No se pudo subir: " + (r.stderr or "").replace(tok, "***")[-300:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=POR_DEFECTO)
    ap.add_argument("--salida", default="/kaggle/working/test_metricas" if Path("/kaggle/working").exists() else "test_metricas")
    ap.add_argument("--sin-github", action="store_true")
    ap.add_argument("--secreto", default=None)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    import kaggle_benchmark as kb
    import kaggle_fresas as kf
    dev = 0 if torch.cuda.is_available() else "cpu"
    S, sal = Path("/tmp/metricas"), Path(a.salida)
    (S / "pesos").mkdir(parents=True, exist_ok=True)
    sal.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "fetch", "-q", "origin", "yolo-benchmark"], cwd=REPO)
    part = json.loads(subprocess.run(["git", "show", "origin/yolo-benchmark:benchmark/6lotes/particion.json"],
                                     cwd=REPO, capture_output=True, text=True).stdout)
    kb.particionar(kf.revisiones(), S / "ds", fijo={x: sp for sp, xs in part.items() for x in xs})
    out = {}
    for m in a.modelos:
        p = S / "pesos" / f"{m}.pt"
        if not p.exists():
            urllib.request.urlretrieve(f"https://github.com/{GH}/releases/download/benchmark-6lotes/{m}.pt", p)
        z = 1024 if "1024" in m else 640
        mod = YOLO(str(p))
        v = mod.val(data=str(S / "ds" / "data.yaml"), split="test", imgsz=z, batch=8, device=dev, plots=True,
                    verbose=False, project=str(S / "val"), name=m, exist_ok=True)
        M = v.confusion_matrix.matrix.astype(float)
        tot, E = M.sum(), {}
        for c in range(M.shape[0] - 1):
            tp = M[c, c]
            fp, fn = M[c, :].sum() - tp, M[:, c].sum() - tp
            E[v.names[c]] = round(100 * (tot - tp - fp - fn) / (tot - tp - fn), 2)
        com = kb.comercial_top1(mod, S / "ds", z, dev)
        out[m] = {"imgsz": z, "map50": round(100 * float(v.box.map50), 2), "map50_95": round(100 * float(v.box.map), 2),
                  "precision": round(100 * float(v.box.mp), 2), "recall": round(100 * float(v.box.mr), 2),
                  "especificidad": round(float(np.mean(list(E.values()))), 2), "especificidad_clase": E,
                  "acierto_comercial": com.get("acierto_comercial"), "error_critico": com.get("error_critico"),
                  "confusion_comercial": com.get("confusion_comercial"), "matriz_conteos": M.astype(int).tolist()}
        for f in ("confusion_matrix_normalized.png", "confusion_matrix.png"):
            if (S / "val" / m / f).exists():
                shutil.copy(S / "val" / m / f, sal / f"{m}_{f}")
        r = out[m]
        print(f"{m:>17}  mAP50 {r['map50']:.2f}  mAP50-95 {r['map50_95']:.2f}  P {r['precision']:.2f}  R {r['recall']:.2f}  "
              f"E {r['especificidad']:.2f}  acierto com. {r['acierto_comercial']}  error crít. {r['error_critico']}", flush=True)
    (sal / "metricas.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Listo:", sal)
    if not a.sin_github:
        tok = kf.token(a.secreto)
        subir(sal, tok) if tok else print("⚠ Sin token: los resultados quedan solo en", sal)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO / "modelos"))
    main()
