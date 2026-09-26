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


PLANTILLA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fresas - __TITULO__</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee}
header{position:sticky;top:0;background:#222;padding:10px 12px;z-index:2;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
header b{flex:1}
button,a.btn{background:#444;color:#fff;border:0;border-radius:8px;padding:8px 12px;font-size:15px;text-decoration:none}
header button{padding:6px 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px;padding:6px}
.c{position:relative;border:4px solid #333;border-radius:8px;overflow:hidden;background:#000}
.c.ok{border-color:#2ecc71} .c.mal{border-color:#e74c3c}
.c img{width:100%;display:block}
.c svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.c .n{position:absolute;left:0;bottom:0;right:0;background:#000a;font-size:11px;padding:2px 4px;word-break:break-all}
.c .m{position:absolute;right:4px;top:4px;font-size:22px}
.e{position:absolute;transform:translateY(-100%);color:#000;font-size:12px;font-weight:600;padding:0 3px;border-radius:3px;min-width:6px;min-height:6px;white-space:nowrap;pointer-events:none}
.c .e{transform:none;border-radius:50%}
.h{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border:3px solid #fff;border-radius:50%;background:#0008;box-sizing:border-box}
#ley{width:100%;display:flex;flex-wrap:wrap;gap:4px 10px;font-size:13px}
#ley i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-1px}
#ver{position:fixed;inset:0;background:#000;display:none;z-index:5;flex-direction:column}
#ver .img{flex:1;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:0}
#lienzo{position:relative;max-width:100%;max-height:100%}
#lienzo.ed{touch-action:none}
#ver img{max-width:100vw;max-height:calc(100vh - 70px);display:block;user-select:none;-webkit-user-drag:none}
#ver.editando img{max-height:calc(100vh - 130px)}
#ver svg,#ve{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#ver .bar,#edbar{display:flex;gap:6px;padding:8px}
#ver .bar button{flex:1;font-size:20px;padding:12px 0}
#edbar{display:none;flex-wrap:wrap;padding-bottom:0}
#ver.editando #edbar{display:flex}
#edbar button{flex:1;font-size:12px;padding:8px 4px;color:#000;font-weight:600;border:3px solid transparent}
#edbar button.act{border-color:#fff}
#edbar .gris{background:#555;color:#fff}
#aviso{position:absolute;top:8px;left:50%;transform:translateX(-50%);background:#f39c12;color:#000;padding:4px 10px;border-radius:12px;font-size:13px;display:none}
#ver.editando #aviso{display:block}
#ver.zoom .img{display:block;overflow:auto}
#ver.zoom #lienzo{max-width:none;max-height:none;width:max-content}
#ver.zoom img{max-width:none;max-height:none}
</style></head><body>
<header><b>__TITULO__ · <span id="cont"></span></b>
<button onclick="mostrarAnot=!mostrarAnot;pintar()">Anotaciones</button>
<button onclick="descargar()">CSV</button>
<button onclick="exportar()">Correcciones</button>
__NAVEGACION__<div id="ley"></div></header>
<div class="grid" id="g"></div>
<div id="ver"><div class="img"><div id="lienzo"><img id="vi" draggable="false"><svg id="vs" viewBox="0 0 1 1" preserveAspectRatio="none"></svg><div id="ve"></div></div>
<span id="aviso">Editando: toca una caja, arrástrala o mueve sus esquinas</span></div>
<div id="edbar"></div>
<div class="bar"><button onclick="mover(-1)">◀</button><button style="background:#27ae60" onclick="marcar('ok')">✅</button>
<button style="background:#c0392b" onclick="marcar('mal')">❌</button><button id="bed" onclick="alternarEdicion()">✏️</button><button id="bzoom" onclick="cambiarZoom()">1×</button>
<button onclick="cerrar()">✕</button><button onclick="mover(1)">▶</button></div></div>
<script>
const LOTE=__LOTE__;
const DATOS=__DATOS__;
const CLAVE="fresas_"+LOTE;
let marcas={},ediciones={},mostrarAnot=true,actual=-1,editando=false,sel=-1,etiquetaNueva=null,arrastre=null;
try{marcas=JSON.parse(localStorage.getItem(CLAVE)||"{}");ediciones=JSON.parse(localStorage.getItem(CLAVE+"_ed")||"{}")}catch(e){}
function guardar(){try{localStorage.setItem(CLAVE,JSON.stringify(marcas));localStorage.setItem(CLAVE+"_ed",JSON.stringify(ediciones))}catch(e){}}
const COLORES={"unripe":"#2ecc71","early-pink":"#ff9ff3","commercial-basic":"#f39c12","commercial-high":"#e74c3c","overripe":"#9b59b6"};
const EXTRA=["#00d2d3","#feca57","#54a0ff","#ffffff"];
function color(l){if(!(l in COLORES))COLORES[l]=EXTRA[Object.keys(COLORES).length%EXTRA.length];return COLORES[l];}
DATOS.forEach(d=>d.figs.forEach(f=>color(f.label)));
function figsDe(d){return ediciones[d.orig]||d.figs;}
function caja(f){const xs=f.pts.map(p=>p[0]),ys=f.pts.map(p=>p[1]);return[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];}
function deCaja(label,[x1,y1,x2,y2]){return{label,pts:[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]};}
function svgDe(figs,conSel){
  if(!mostrarAnot)return"";
  return figs.map((f,i)=>`<polygon points="${f.pts.map(p=>p.join(",")).join(" ")}" fill="${conSel&&i==sel?"rgba(255,255,255,.15)":"none"}" stroke="${color(f.label)}" stroke-width="${conSel&&i==sel?.01:.006}"/>`).join("");
}
function etqDe(figs,chico,conSel){
  if(!mostrarAnot)return"";
  let h=figs.map(f=>{const[x,y]=caja(f);
    return `<span class="e" style="left:${x*100}%;top:${y*100}%;background:${color(f.label)}">${chico?"":f.label}</span>`}).join("");
  if(conSel&&sel>=0&&figs[sel]){const[x1,y1,x2,y2]=caja(figs[sel]);
    [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].forEach(([x,y])=>h+=`<span class="h" style="left:${x*100}%;top:${y*100}%"></span>`);}
  return h;
}
function leyenda(){
  const n={};DATOS.forEach(d=>figsDe(d).forEach(f=>n[f.label]=(n[f.label]||0)+1));
  document.getElementById("ley").innerHTML=Object.keys(n).sort().map(l=>`<span><i style="background:${color(l)}"></i>${l} (${n[l]})</span>`).join("");
}
function pintar(){
  const g=document.getElementById("g");g.innerHTML="";
  DATOS.forEach((d,i)=>{
    const m=marcas[d.orig]||"",ed=d.orig in ediciones;
    const c=document.createElement("div");c.className="c "+m;
    c.innerHTML=`<img loading="lazy" src="${d.img}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${svgDe(figsDe(d))}</svg>${etqDe(figsDe(d),true)}
      <span class="m">${ed?"✏️":""}${m=="ok"?"✅":m=="mal"?"❌":""}</span><span class="n">${i+1}. ${d.orig}</span>`;
    c.onclick=()=>abrir(i);g.appendChild(c);
  });
  const ok=Object.values(marcas).filter(v=>v=="ok").length,mal=Object.values(marcas).filter(v=>v=="mal").length;
  document.getElementById("cont").textContent=`✅ ${ok} · ❌ ${mal} · ✏️ ${Object.keys(ediciones).length} · pendientes ${DATOS.length-ok-mal}`;
  leyenda();
}
function pintarVisor(){const d=DATOS[actual],f=figsDe(d);
  document.getElementById("vs").innerHTML=svgDe(f,editando);document.getElementById("ve").innerHTML=etqDe(f,false,editando);
  if(editando)pintarEdbar();}
function pintarEdbar(){
  const f=figsDe(DATOS[actual]),actualEtq=sel>=0&&f[sel]?f[sel].label:etiquetaNueva;
  document.getElementById("edbar").innerHTML=Object.keys(COLORES).map(l=>`<button class="${l==actualEtq?"act":""}" style="background:${color(l)}" onclick="ponerEtiqueta('${l}')">${l}</button>`).join("")+
    `<button class="gris" onclick="nuevaCaja()">＋ caja</button><button class="gris" onclick="borrarCaja()">🗑 borrar</button><button class="gris" onclick="restaurar()">↺ original</button>`;
}
let zoom=1,anchoBase=0;
function cambiarZoom(){const vi=document.getElementById("vi"),ver=document.getElementById("ver"),cont=ver.querySelector(".img");
  if(zoom==1)anchoBase=vi.getBoundingClientRect().width;
  const r=vi.getBoundingClientRect(),cx=(cont.clientWidth/2-r.left+cont.getBoundingClientRect().left)/r.width,cy=(cont.clientHeight/2-r.top+cont.getBoundingClientRect().top)/r.height;
  zoom=zoom>=3?1:zoom+1;ver.classList.toggle("zoom",zoom>1);vi.style.width=zoom>1?anchoBase*zoom+"px":"";
  document.getElementById("bzoom").textContent=zoom+"×";
  if(zoom>1){const n=vi.getBoundingClientRect();cont.scrollLeft=cx*n.width-cont.clientWidth/2;cont.scrollTop=cy*n.height-cont.clientHeight/2;}}
function abrir(i){if(zoom>1){zoom=3;cambiarZoom();}actual=i;sel=-1;document.getElementById("vi").src=DATOS[i].img;pintarVisor();document.getElementById("ver").style.display="flex";}
function cerrar(){if(editando)alternarEdicion();document.getElementById("ver").style.display="none";pintar();}
function mover(k){const n=actual+k;if(n>=0&&n<DATOS.length)abrir(n);else cerrar();}
function marcar(v){const o=DATOS[actual].orig;marcas[o]=marcas[o]==v?"":v;if(!marcas[o])delete marcas[o];guardar();mover(1);}
function alternarEdicion(){editando=!editando;sel=-1;
  document.getElementById("ver").classList.toggle("editando",editando);document.getElementById("lienzo").classList.toggle("ed",editando);
  document.getElementById("bed").style.background=editando?"#f39c12":"";pintarVisor();}
function editables(){const d=DATOS[actual];if(!ediciones[d.orig])ediciones[d.orig]=JSON.parse(JSON.stringify(d.figs));return ediciones[d.orig];}
function cambio(){const d=DATOS[actual];if(JSON.stringify(ediciones[d.orig])==JSON.stringify(d.figs))delete ediciones[d.orig];guardar();pintarVisor();}
function ponerEtiqueta(l){etiquetaNueva=l;if(sel>=0){editables()[sel].label=l;cambio();}else pintarEdbar();}
function nuevaCaja(){const f=editables(),l=etiquetaNueva||(f[0]&&f[0].label)||Object.keys(COLORES)[0];
  f.push(deCaja(l,[.42,.42,.58,.58]));sel=f.length-1;cambio();}
function borrarCaja(){if(sel<0)return;editables().splice(sel,1);sel=-1;cambio();}
function restaurar(){if(confirm("¿Volver a las cajas originales de esta foto?")){delete ediciones[DATOS[actual].orig];sel=-1;guardar();pintarVisor();}}
function punto(ev){const r=document.getElementById("vi").getBoundingClientRect();
  return[Math.min(1,Math.max(0,(ev.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(ev.clientY-r.top)/r.height))];}
const lienzo=document.getElementById("lienzo");
lienzo.addEventListener("pointerdown",ev=>{
  if(!editando)return;ev.preventDefault();
  const p=punto(ev),f=figsDe(DATOS[actual]),r=document.getElementById("vi").getBoundingClientRect(),tol=22/r.width,tolY=22/r.height;
  if(sel>=0&&f[sel]){const[x1,y1,x2,y2]=caja(f[sel]);
    const esq=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]].findIndex(([x,y])=>Math.abs(p[0]-x)<tol&&Math.abs(p[1]-y)<tolY);
    if(esq>=0){arrastre={tipo:"esq",esq,caja:[x1,y1,x2,y2]};lienzo.setPointerCapture(ev.pointerId);return;}}
  let mejor=-1,area=9;
  f.forEach((fi,i)=>{const[x1,y1,x2,y2]=caja(fi);if(p[0]>=x1&&p[0]<=x2&&p[1]>=y1&&p[1]<=y2&&(x2-x1)*(y2-y1)<area){mejor=i;area=(x2-x1)*(y2-y1);}});
  sel=mejor;
  if(sel<0&&zoom>1){const c=document.querySelector("#ver .img");arrastre={tipo:"pan",x:ev.clientX,y:ev.clientY,sl:c.scrollLeft,st:c.scrollTop};lienzo.setPointerCapture(ev.pointerId);}
  if(sel>=0){arrastre={tipo:"mover",inicio:p,caja:caja(f[sel])};lienzo.setPointerCapture(ev.pointerId);}
  pintarVisor();
});
lienzo.addEventListener("pointermove",ev=>{
  if(!arrastre)return;
  if(arrastre.tipo=="pan"){const c=document.querySelector("#ver .img");c.scrollLeft=arrastre.sl-(ev.clientX-arrastre.x);c.scrollTop=arrastre.st-(ev.clientY-arrastre.y);return;}
  const p=punto(ev),f=editables();let[x1,y1,x2,y2]=arrastre.caja;
  if(arrastre.tipo=="mover"){let dx=p[0]-arrastre.inicio[0],dy=p[1]-arrastre.inicio[1];
    dx=Math.min(1-x2,Math.max(-x1,dx));dy=Math.min(1-y2,Math.max(-y1,dy));x1+=dx;x2+=dx;y1+=dy;y2+=dy;}
  else{const e=arrastre.esq;if(e==0||e==3)x1=p[0];else x2=p[0];if(e<2)y1=p[1];else y2=p[1];}
  f[sel]=deCaja(f[sel].label,[Math.min(x1,x2),Math.min(y1,y2),Math.max(x1,x2),Math.max(y1,y2)]);
  if(arrastre.tipo=="esq"){const c=arrastre.caja=caja(f[sel]),q=[[c[0],c[1]],[c[2],c[1]],[c[2],c[3]],[c[0],c[3]]];
    arrastre.esq=q.map(([x,y],i)=>[Math.hypot(x-p[0],y-p[1]),i]).sort((a,b)=>a[0]-b[0])[0][1];}
  pintarVisor();
});
function soltar(){if(arrastre){const t=arrastre.tipo;arrastre=null;if(t!="pan")cambio();}}
lienzo.addEventListener("pointerup",soltar);lienzo.addEventListener("pointercancel",soltar);
function bajar(nombre,contenido,tipo){
  const archivo=new File([contenido],nombre,{type:tipo});
  if(navigator.canShare&&navigator.canShare({files:[archivo]})){navigator.share({files:[archivo],title:nombre}).catch(()=>{});return;}
  const a=document.createElement("a");a.href=URL.createObjectURL(archivo);a.download=nombre;a.click();
}
function descargar(){
  const filas=["archivo,estado,editada"].concat(DATOS.map(d=>`"${d.orig}",${marcas[d.orig]||"sin_revisar"},${d.orig in ediciones?"si":"no"}`));
  bajar(`resultado_${LOTE}.csv`,filas.join("\n"),"text/csv");
}
function exportar(){
  const n=Object.keys(ediciones).length;
  if(!n){alert("Todavía no editaste ninguna caja en este lote.");return;}
  bajar(`correcciones_${LOTE}.json`,JSON.stringify({lote:LOTE,correcciones:ediciones},null,1),"application/json");
}
pintar();
</script></body></html>
"""


def escribir_html(dir_lote, nombres, i, datos):
    nombre = nombres[i]
    nav = '<a class="btn" href="../index.html">Lotes</a>'
    if i > 0:
        nav += f' <a class="btn" href="../{nombres[i - 1]}/index.html">◀</a>'
    if i < len(nombres) - 1:
        nav += f' <a class="btn" href="../{nombres[i + 1]}/index.html">▶</a>'
    pagina = (PLANTILLA.replace("__TITULO__", html.escape(f"{nombre} ({len(datos)} fotos)"))
              .replace("__NAVEGACION__", nav).replace("__LOTE__", json.dumps(nombre))
              .replace("__DATOS__", json.dumps(datos, ensure_ascii=False)))
    (dir_lote / "index.html").write_text(pagina, encoding="utf-8")


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


def nombres_yolo(origen):
    """Lista de clases YOLO (classes.txt) buscando hacia arriba desde el origen."""
    for d in [origen, *origen.parents]:
        c = d / "classes.txt"
        if c.exists():
            return [l.strip() for l in c.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def aplicar_correcciones(origen, archivo):
    """Escribe en las anotaciones originales las cajas editadas desde el celular."""
    datos = json.loads(Path(archivo).read_text(encoding="utf-8"))
    correcciones = datos.get("correcciones", datos)
    clases = None
    hechas = 0
    for orig, figs in correcciones.items():
        img = origen / orig
        if not img.exists():
            print(f"  ¡No existe {img}! (¿--carpeta correcta?)")
            continue
        cajas = []
        for f in figs:
            xs, ys = [p[0] for p in f["pts"]], [p[1] for p in f["pts"]]
            cajas.append((f["label"], min(xs), min(ys), max(xs), max(ys)))
        j, t = img.with_suffix(".json"), img.with_suffix(".txt")
        if t.exists() and not j.exists():
            if clases is None:
                clases = nombres_yolo(origen)
            lineas = []
            for label, x1, y1, x2, y2 in cajas:
                c = label if label.isdigit() else (str(clases.index(label)) if label in clases else None)
                if c is None:
                    sys.exit(f"No sé el número de clase de '{label}' (falta classes.txt)")
                lineas.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            respaldo(t)
            t.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        else:
            if j.exists():
                d = json.loads(j.read_text(encoding="utf-8"))
                respaldo(j)
            else:
                from PIL import Image
                with Image.open(img) as im:
                    w, h = im.size
                d = {"version": "5.0.1", "flags": {}, "imagePath": img.name, "imageData": None,
                     "imageHeight": h, "imageWidth": w}
            w, h = d["imageWidth"], d["imageHeight"]
            d["shapes"] = [{"label": label, "points": [[x1 * w, y1 * h], [x2 * w, y2 * h]],
                            "group_id": None, "description": "", "shape_type": "rectangle", "flags": {}}
                           for label, x1, y1, x2, y2 in cajas]
            j.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        hechas += 1
    print(f"Correcciones aplicadas: {hechas} imágenes. Los originales quedaron como .bak")


def respaldo(archivo):
    bak = archivo.with_name(archivo.name + ".bak")
    if not bak.exists():
        shutil.copy2(archivo, bak)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raiz", type=Path, nargs="?", help="carpeta del dataset")
    ap.add_argument("--rehacer-html", action="store_true",
                    help="solo actualizar las páginas de los lotes ya generados en --salida")
    ap.add_argument("--carpeta", help="subcarpeta a revisar ('.' para la raíz)")
    ap.add_argument("--salida", type=Path, default=Path("revision"))
    ap.add_argument("--lote", type=int, default=300, help="imágenes por lote (300)")
    ap.add_argument("--solo", type=int, nargs="+", help="generar solo estos lotes, ej. --solo 1")
    ap.add_argument("--aplicar", metavar="JSON",
                    help="aplicar correcciones_lote_XX.json exportado desde el celular")
    ap.add_argument("--max-lado", type=int, default=1280, help="tamaño máx. de las copias")
    a = ap.parse_args()
    if a.rehacer_html:
        return rehacer_html(a.salida)
    if a.raiz is None or not a.raiz.is_dir():
        sys.exit(f"No existe: {a.raiz}")
    if a.aplicar:
        aplicar_correcciones(a.raiz / (a.carpeta or "."), a.aplicar)
    elif a.carpeta is None:
        listar_carpetas(a.raiz)
    else:
        armar_lotes(a.raiz, None if a.carpeta == "." else a.carpeta, a.salida, a.lote, a.max_lado, a.solo)


if __name__ == "__main__":
    main()
