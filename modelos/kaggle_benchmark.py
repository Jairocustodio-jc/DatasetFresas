"""Compara YOLOv8, v9, v10, 11, 12 y 26 (nano y el más grande) con los lotes ya revisados. Para dejarlo corriendo de noche en Kaggle.

Notebook de Kaggle (Settings → Accelerator: «GPU T4 x2», Internet: On; Add-ons → Secrets: GITHUB_TOKEN activado):

    !pip -q install ultralytics
    !git clone -q -b claude/jolly-heisenberg-qa6cj9 https://github.com/Jairocustodio-jc/DatasetFresas
    %cd DatasetFresas
    !python modelos/kaggle_benchmark.py

Y luego «Save Version → Save & Run All (Commit)»: así corre solo hasta 12 h aunque cierres el navegador.

Qué compara (mismas condiciones para todos):
  - nano:   yolov8n yolov9t yolov10n yolo11n yolo12n yolo26n   (v9 no tiene «n»: su menor es «t»)
  - grande: yolov8x yolov9e yolov10x yolo11x yolo12x yolo26x   (v9 no tiene «x»: su mayor es «e»)
  - parten de los pesos COCO oficiales; mismos datos y misma partición (40 fotos de validación por lote, semilla 0);
  - imgsz 640, 100 épocas como máximo, paciencia 20, AdamW lr0 0,001, hsv_h 0 (el color define la clase), semilla 0;
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
PESO = {m: 1 for m in NANO} | {m: 6 for m in GRANDE}
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
        carga = sum(PESO[x] for x in pend) / n_gpu
        tope = min(resta, resta * PESO[m] / max(carga, PESO[m]))
        entrenar(m, a, sal, tope, fin, gpu, YOLO)
        torch.cuda.empty_cache()


def entrenar(m, a, sal, tope, fin, gpu, YOLO):
    grande = m in GRANDE
    batch = 8 if grande else 16
    t0 = time.time()
    log(f"[GPU {a.gpu}] {m}: empieza (tope {tope/3600:.1f} h, batch {batch})")
    parado = {"tiempo": False}

    def por_tiempo(tr):
        if time.time() - t0 > tope or time.time() > fin - 10 * 60:
            parado["tiempo"] = True
            tr.stop = True

    res = {"modelo": m, "tamaño": "grande" if grande else "nano", "gpu": gpu, "epochs_max": a.epochs}
    for intento in range(3):
        try:
            modelo = YOLO(f"{m}.pt")
            modelo.add_callback("on_fit_epoch_end", por_tiempo)
            modelo.train(data=a.data, epochs=a.epochs, patience=20, imgsz=640, batch=batch, device=a.dev,
                         optimizer="AdamW", lr0=0.001, nbs=64, seed=0, deterministic=True, workers=a.workers,
                         cache="ram", project=str(sal / "runs"), name=m, exist_ok=True, plots=True, verbose=False,
                         fliplr=0.5, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, amp=True)
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
        v = mb.val(data=a.data, split="val", imgsz=640, batch=8, device=a.dev, plots=True, verbose=False,
                   project=str(run), name="val", exist_ok=True)
        nombres = v.names
        res |= {"map50": round(float(v.box.map50), 4), "map50_95": round(float(v.box.map), 4),
                "precision": round(float(v.box.mp), 4), "recall": round(float(v.box.mr), 4),
                "ap50_por_clase": {nombres[int(c)]: round(float(x), 4) for c, x in zip(v.box.ap_class_index, v.box.ap50)},
                "inferencia_ms": round(float(v.speed.get("inference", 0)), 2)}
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


# ───────────────────────────── respaldo en GitHub ─────────────────────────────

class Respaldo:
    def __init__(self, sal, nombre, activo):
        self.sal, self.nombre, self.tok = sal, nombre, None
        self.dir = Path("/tmp/bench_git")
        self.subidos = set()
        if not activo:
            return
        try:
            from kaggle_secrets import UserSecretsClient
            self.tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
        except Exception:
            self.tok = os.environ.get("GITHUB_TOKEN")
        if not self.tok:
            log("⚠ Sin GITHUB_TOKEN: los resultados quedan solo en /kaggle/working (Output del notebook).")
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
        for p in prev.glob("res_*.json"):   # retomar: lo completo de una corrida anterior no se repite
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
    filas = []
    for m in modelos:
        r = leer(sal / f"res_{m}.json")
        if r is None:
            r = {"modelo": m, "estado": "entrenando" if (sal / "claims" / f"{m}.lock").exists() else "pendiente"}
        filas.append(r)
    cols = ["modelo", "tamaño", "estado", "map50", "map50_95", "precision", "recall", "epochs_hechas", "mejor_epoch",
            "parado_por_tiempo", "minutos", "batch", "params_M", "gflops", "inferencia_ms", "peso_MB", "gpu", "error"]
    with open(sal / "resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    ok = sorted([r for r in filas if r.get("map50_95") is not None], key=lambda r: -r["map50_95"])
    clases = ["unripe", "early-pink", "commercial-basic", "commercial-high", "overripe"]
    t = [f"# Benchmark YOLO · {info['nombre']}", "",
         f"Actualizado {time.strftime('%d/%m/%Y %H:%M')} (hora de Kaggle, UTC). Lotes: {', '.join(info['lotes'])}. "
         f"{info['fotos']} fotos ({info['val']} de validación). GPU: {info['gpus']}.", "",
         "| # | Modelo | mAP50-95 | mAP50 | P | R | Épocas | Min | Params (M) | GFLOPs | ms/img |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ok, 1):
        ep = f"{r.get('epochs_hechas')} (mejor {r.get('mejor_epoch')})" + (" ⏱" if r.get("parado_por_tiempo") else "")
        t.append(f"| {i} | {r['modelo']} | {r['map50_95']:.3f} | {r['map50']:.3f} | {r['precision']:.2f} | {r['recall']:.2f} "
                 f"| {ep} | {r.get('minutos')} | {r.get('params_M', '')} | {r.get('gflops', '')} | {r.get('inferencia_ms', '')} |")
    if ok:
        t += ["", "**AP50 por clase**", "", "| Modelo | " + " | ".join(clases) + " |", "|---" * (len(clases) + 1) + "|"]
        for r in ok:
            ap = r.get("ap50_por_clase", {})
            t.append(f"| {r['modelo']} | " + " | ".join(f"{ap[c]:.2f}" if c in ap else "–" for c in clases) + " |")
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
    ap.add_argument("--modelos", nargs="+", default=NANO + GRANDE)
    ap.add_argument("--nombre", default=None, help="nombre de la corrida (por defecto «<N>lotes»); repetirlo retoma")
    ap.add_argument("--salida", default="/kaggle/working/benchmark")
    ap.add_argument("--sin-github", action="store_true", help="no subir nada (solo /kaggle/working)")
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
    desconocidos = [m for m in a.modelos if m not in PESO]
    if desconocidos:
        sys.exit(f"Modelos desconocidos: {desconocidos}. Opciones: {NANO + GRANDE}")
    import torch
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        sys.exit("No hay GPU. En Kaggle: Settings → Accelerator → GPU T4 x2.")
    gpus = ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu))
    log(f"GPU: {gpus}")

    from kaggle_fresas import revisiones, armar_dataset
    rev = revisiones()
    nombre = a.nombre or f"{len(rev)}lotes"
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

    respaldo = Respaldo(sal, nombre, not a.sin_github)
    ds = Path("/tmp/ds_fresas")
    n = armar_dataset(rev, ds)
    val = sorted(p.name for p in (ds / "images" / "val").iterdir())
    info = {"nombre": nombre, "lotes": sorted(rev), "fotos": n, "val": len(val), "gpus": gpus}
    escribir(sal / "config.json", info | {"modelos": a.modelos, "epochs": a.epochs, "horas": a.horas,
                                          "fotos_validacion": val, "inicio": time.strftime("%Y-%m-%d %H:%M")})
    datas = []
    for g in range(n_gpu):   # una copia por GPU: así no chocan al escribir la caché de etiquetas
        d = Path(f"/tmp/ds_fresas_g{g}")
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(ds, d)
        (d / "data.yaml").write_text((ds / "data.yaml").read_text().replace(str(ds.resolve()), str(d.resolve())))
        datas.append(str(d / "data.yaml"))
    log(f"{n} fotos ({len(val)} de validación) · modelos: {' '.join(a.modelos)}")
    resumen(sal, a.modelos, info)
    respaldo.sincronizar()

    def lanzar(g):
        cmd = [sys.executable, __file__, "--worker", "--gpu", str(g), "--n-gpu", str(n_gpu), "--deadline", str(fin),
               "--data", datas[g], "--salida", str(sal), "--epochs", str(a.epochs),
               "--workers", str(max(2, (os.cpu_count() or 4) // n_gpu)), "--modelos", *a.modelos]
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
