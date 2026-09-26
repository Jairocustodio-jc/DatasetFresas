"""Revisión del dataset de fresas desde el celular.

Paso 1 - ver qué carpetas hay (para identificar la que se arregló):
    python revisar_fresas.py "D:\\新草莓\\DatasetId_360753_1652783343"

Paso 2 - armar los lotes de 300 imágenes con una página de revisión:
    python revisar_fresas.py "D:\\新草莓\\DatasetId_360753_1652783343" --carpeta NOMBRE_SUBCARPETA

Se crea la carpeta "revision" (lote_01 ... lote_10). Cada lote tiene un index.html:
toca una foto para marcarla ✅ bien / ❌ mal / sin marcar, y al final descarga el CSV.
Si hay anotaciones (LabelMe .json o YOLO .txt con el mismo nombre) se dibujan encima.
"""
import argparse
import datetime
import html
import json
import shutil
import sys
from pathlib import Path

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def es_imagen(p):
    return p.is_file() and p.suffix.lower() in EXT_IMG


def listar_carpetas(raiz):
    print(f"\nCarpetas dentro de {raiz}:\n")
    print(f"{'imágenes':>9} {'json':>6} {'txt':>6}  {'última modificación':<19}  carpeta")
    for d in sorted([raiz] + [p for p in raiz.rglob("*") if p.is_dir()]):
        archivos = [f for f in d.iterdir() if f.is_file()]
        imgs = [f for f in archivos if f.suffix.lower() in EXT_IMG]
        if not archivos:
            continue
        n_json = sum(f.suffix.lower() == ".json" for f in archivos)
        n_txt = sum(f.suffix.lower() == ".txt" for f in archivos)
        ultima = max(f.stat().st_mtime for f in archivos)
        fecha = datetime.datetime.fromtimestamp(ultima).strftime("%Y-%m-%d %H:%M")
        print(f"{len(imgs):>9} {n_json:>6} {n_txt:>6}  {fecha:<19}  {d.relative_to(raiz.parent)}")
    print("\nLa carpeta arreglada suele ser la de modificación más reciente y ~3000 imágenes.")
    print('Luego ejecuta de nuevo con --carpeta "<ruta relativa a la raíz>".\n')


def leer_anotacion(img):
    """Devuelve lista de figuras [{'label', 'pts': [[x,y],...] normalizados 0-1}] o []."""
    j = img.with_suffix(".json")
    if j.exists():
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
            w, h = d.get("imageWidth"), d.get("imageHeight")
            if w and h:
                figs = []
                for s in d.get("shapes", []):
                    pts = s.get("points", [])
                    if s.get("shape_type") == "rectangle" and len(pts) == 2:
                        (x1, y1), (x2, y2) = pts
                        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    figs.append({"label": s.get("label", ""),
                                 "pts": [[x / w, y / h] for x, y in pts]})
                return figs
        except (ValueError, KeyError, TypeError):
            pass
    t = img.with_suffix(".txt")
    if t.exists():
        figs = []
        for linea in t.read_text(encoding="utf-8", errors="ignore").splitlines():
            v = linea.split()
            if len(v) == 5:  # YOLO: clase cx cy w h
                c, cx, cy, bw, bh = v[0], *map(float, v[1:])
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                figs.append({"label": c, "pts": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
            elif len(v) > 5 and len(v) % 2 == 1:  # YOLO segmentación
                n = list(map(float, v[1:]))
                figs.append({"label": v[0], "pts": [n[i:i + 2] for i in range(0, len(n), 2)]})
        return figs
    return []


def copiar_imagen(origen, destino, max_lado):
    """Copia reducida (para que pese poco en el celular) si Pillow está instalado."""
    try:
        from PIL import Image
    except ImportError:
        shutil.copy2(origen, destino.with_suffix(origen.suffix.lower()))
        return destino.with_suffix(origen.suffix.lower()).name
    with Image.open(origen) as im:
        im = im.convert("RGB")
        im.thumbnail((max_lado, max_lado))
        im.save(destino.with_suffix(".jpg"), quality=80)
    return destino.with_suffix(".jpg").name


PLANTILLA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fresas - {titulo}</title>
<style>
body{{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee}}
header{{position:sticky;top:0;background:#222;padding:10px 12px;z-index:2;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
header b{{flex:1}}
header button{{padding:6px 10px}}
button,a.btn{{background:#444;color:#fff;border:0;border-radius:8px;padding:8px 12px;font-size:15px;text-decoration:none}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px;padding:6px}}
.c{{position:relative;border:4px solid #333;border-radius:8px;overflow:hidden;background:#000}}
.c.ok{{border-color:#2ecc71}} .c.mal{{border-color:#e74c3c}}
.c img{{width:100%;display:block}}
.c svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
.c .n{{position:absolute;left:0;bottom:0;right:0;background:#000a;font-size:11px;padding:2px 4px;word-break:break-all}}
.c .m{{position:absolute;right:4px;top:4px;font-size:22px}}
#ver{{position:fixed;inset:0;background:#000;display:none;z-index:5;flex-direction:column}}
#ver .img{{flex:1;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}}
#ver .img>div{{position:relative;max-width:100%;max-height:100%}}
#ver img{{max-width:100vw;max-height:calc(100vh - 70px);display:block}}
#ver svg,#ve{{position:absolute;inset:0;width:100%;height:100%}}
.e{{position:absolute;transform:translateY(-100%);color:#000;font-size:12px;font-weight:600;padding:0 3px;border-radius:3px;min-width:6px;min-height:6px;white-space:nowrap;pointer-events:none}}
.c .e{{transform:none;border-radius:50%}}
#ley{{width:100%;display:flex;flex-wrap:wrap;gap:4px 10px;font-size:13px}}
#ley i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-1px}}
#ver .bar{{display:flex;gap:6px;padding:8px}}
#ver .bar button{{flex:1;font-size:20px;padding:12px 0}}
</style></head><body>
<header><b>{titulo} · <span id="cont"></span></b>
<button onclick="mostrarAnot=!mostrarAnot;pintar()">Anotaciones</button>
<button onclick="descargar()">Descargar CSV</button>
{navegacion}<div id="ley"></div></header>
<div class="grid" id="g"></div>
<div id="ver"><div class="img"><div><img id="vi"><svg id="vs" viewBox="0 0 1 1" preserveAspectRatio="none"></svg><div id="ve"></div></div></div>
<div class="bar"><button onclick="mover(-1)">◀</button><button style="background:#27ae60" onclick="marcar('ok')">✅</button>
<button style="background:#c0392b" onclick="marcar('mal')">❌</button><button onclick="cerrar()">✕</button><button onclick="mover(1)">▶</button></div></div>
<script>
const LOTE={lote_json};
const DATOS={datos_json};
const CLAVE="fresas_"+LOTE;
let marcas={{}},mostrarAnot=true,actual=-1;
try{{marcas=JSON.parse(localStorage.getItem(CLAVE)||"{{}}")}}catch(e){{}}
function guardar(){{try{{localStorage.setItem(CLAVE,JSON.stringify(marcas))}}catch(e){{}}}}
const COLORES={{"unripe":"#2ecc71","early-pink":"#ff9ff3","commercial-basic":"#f39c12","commercial-high":"#e74c3c","overripe":"#9b59b6"}};
const EXTRA=["#00d2d3","#feca57","#54a0ff","#ffffff"];
function color(l){{if(!(l in COLORES))COLORES[l]=EXTRA[Object.keys(COLORES).length%EXTRA.length];return COLORES[l];}}
function svgDe(figs){{
  if(!mostrarAnot)return"";
  return figs.map(f=>`<polygon points="${{f.pts.map(p=>p.join(",")).join(" ")}}" fill="none" stroke="${{color(f.label)}}" stroke-width=".006"/>`).join("");
}}
function etqDe(figs,chico){{
  if(!mostrarAnot)return"";
  return figs.map(f=>{{const x=Math.min(...f.pts.map(p=>p[0])),y=Math.min(...f.pts.map(p=>p[1]));
    return `<span class="e" style="left:${{x*100}}%;top:${{y*100}}%;background:${{color(f.label)}}">${{chico?"":f.label}}</span>`}}).join("");
}}
function leyenda(){{
  const n={{}};DATOS.forEach(d=>d.figs.forEach(f=>n[f.label]=(n[f.label]||0)+1));
  document.getElementById("ley").innerHTML=Object.keys(n).sort().map(l=>`<span><i style="background:${{color(l)}}"></i>${{l}} (${{n[l]}})</span>`).join("");
}}
function pintar(){{
  const g=document.getElementById("g");g.innerHTML="";
  DATOS.forEach((d,i)=>{{
    const m=marcas[d.orig]||"";
    const c=document.createElement("div");c.className="c "+m;
    c.innerHTML=`<img loading="lazy" src="${{d.img}}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${{svgDe(d.figs)}}</svg>${{etqDe(d.figs,true)}}
      <span class="m">${{m=="ok"?"✅":m=="mal"?"❌":""}}</span><span class="n">${{i+1}}. ${{d.orig}}</span>`;
    c.onclick=()=>abrir(i);g.appendChild(c);
  }});
  const ok=Object.values(marcas).filter(v=>v=="ok").length,mal=Object.values(marcas).filter(v=>v=="mal").length;
  document.getElementById("cont").textContent=`✅ ${{ok}} · ❌ ${{mal}} · pendientes ${{DATOS.length-ok-mal}}`;
}}
function abrir(i){{actual=i;const d=DATOS[i];document.getElementById("vi").src=d.img;
  document.getElementById("vs").innerHTML=svgDe(d.figs);document.getElementById("ve").innerHTML=etqDe(d.figs,false);document.getElementById("ver").style.display="flex";}}
function cerrar(){{document.getElementById("ver").style.display="none";pintar();}}
function mover(k){{const n=actual+k;if(n>=0&&n<DATOS.length)abrir(n);else cerrar();}}
function marcar(v){{const o=DATOS[actual].orig;marcas[o]=marcas[o]==v?"":v;if(!marcas[o])delete marcas[o];guardar();mover(1);}}
function descargar(){{
  const filas=["archivo,estado"].concat(DATOS.map(d=>`"${{d.orig}}",${{marcas[d.orig]||"sin_revisar"}}`));
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([filas.join("\\n")],{{type:"text/csv"}}));
  a.download=`resultado_${{LOTE}}.csv`;a.click();
}}
leyenda();pintar();
</script></body></html>
"""


def escribir_html(dir_lote, nombres, i, datos):
    nombre = nombres[i]
    nav = '<a class="btn" href="../index.html">Lotes</a>'
    if i > 0:
        nav += f' <a class="btn" href="../{nombres[i - 1]}/index.html">◀</a>'
    if i < len(nombres) - 1:
        nav += f' <a class="btn" href="../{nombres[i + 1]}/index.html">▶</a>'
    (dir_lote / "index.html").write_text(PLANTILLA.format(
        titulo=html.escape(f"{nombre} ({len(datos)} fotos)"), navegacion=nav,
        lote_json=json.dumps(nombre), datos_json=json.dumps(datos, ensure_ascii=False)),
        encoding="utf-8")


def rehacer_html(salida):
    """Regenera los index.html de los lotes existentes con la plantilla actual."""
    dirs = sorted(d for d in salida.glob("lote_*") if (d / "index.html").exists())
    nombres = [d.name for d in dirs]
    cantidades = []
    for i, d in enumerate(dirs):
        texto = (d / "index.html").read_text(encoding="utf-8")
        inicio = texto.index("const DATOS=") + len("const DATOS=")
        datos = json.loads(texto[inicio:texto.index(";\n", inicio)])
        escribir_html(d, nombres, i, datos)
        cantidades.append(len(datos))
    enlaces = "".join(f'<li><a href="{n}/index.html">{n}</a> — {c} fotos</li>'
                      for n, c in zip(nombres, cantidades))
    (salida / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Revisión fresas</title><body style="font-family:system-ui;font-size:20px;background:#111;color:#eee">'
        f'<h2>Revisión de fresas</h2><ul style="line-height:2">{enlaces}</ul>'
        '<style>a{color:#6cf}</style>', encoding="utf-8")
    print(f"Páginas actualizadas: {', '.join(nombres)}")


def armar_lotes(raiz, carpeta, salida, tam, max_lado, solo=None):
    origen = (raiz / carpeta) if carpeta else raiz
    if not origen.is_dir():
        sys.exit(f"No existe la carpeta: {origen}")
    imgs = sorted(p for p in origen.rglob("*") if es_imagen(p))
    if not imgs:
        sys.exit(f"No hay imágenes en {origen}")
    print(f"Carpeta usada: {origen}\nImágenes encontradas: {len(imgs)}")
    salida.mkdir(parents=True, exist_ok=True)
    lotes = [imgs[i:i + tam] for i in range(0, len(imgs), tam)]
    nombres = [f"lote_{i + 1:02d}" for i in range(len(lotes))]
    for i, (nombre, grupo) in enumerate(zip(nombres, lotes)):
        if solo and (i + 1) not in solo:
            continue
        dir_lote = salida / nombre
        (dir_lote / "img").mkdir(parents=True, exist_ok=True)
        datos = []
        for k, img in enumerate(grupo):
            archivo = copiar_imagen(img, dir_lote / "img" / f"{k + 1:03d}", max_lado)
            datos.append({"orig": img.relative_to(origen).as_posix(),
                          "img": f"img/{archivo}", "figs": leer_anotacion(img)})
        escribir_html(dir_lote, nombres, i, datos)
        print(f"  {nombre}: {len(grupo)} imágenes")
    rehacer_html(salida)  # índice y flechas ◀ ▶ solo con los lotes que existen
    print(f"\nListo. Abre {salida / 'index.html'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raiz", type=Path, nargs="?", help="carpeta del dataset")
    ap.add_argument("--rehacer-html", action="store_true",
                    help="solo actualizar las páginas de los lotes ya generados en --salida")
    ap.add_argument("--carpeta", help="subcarpeta a revisar ('.' para la raíz)")
    ap.add_argument("--salida", type=Path, default=Path("revision"))
    ap.add_argument("--lote", type=int, default=300, help="imágenes por lote (300)")
    ap.add_argument("--solo", type=int, nargs="+", help="generar solo estos lotes, ej. --solo 1")
    ap.add_argument("--max-lado", type=int, default=1280, help="tamaño máx. de las copias")
    a = ap.parse_args()
    if a.rehacer_html:
        return rehacer_html(a.salida)
    if a.raiz is None or not a.raiz.is_dir():
        sys.exit(f"No existe: {a.raiz}")
    if a.carpeta is None:
        listar_carpetas(a.raiz)
    else:
        armar_lotes(a.raiz, None if a.carpeta == "." else a.carpeta, a.salida, a.lote, a.max_lado, a.solo)


if __name__ == "__main__":
    main()
