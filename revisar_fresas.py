"""Revisión del dataset de fresas desde el celular.

    python revisar_fresas.py RAIZ                                   # ver carpetas
    python revisar_fresas.py RAIZ --carpeta dataset_5estados --solo 4
    python revisar_fresas.py RAIZ --carpeta dataset_5estados --lista informe/cambios.csv --prefijo cambios
    python revisar_fresas.py RAIZ --carpeta dataset_5estados --aplicar revision_lote_01.json
    python revisar_fresas.py --rehacer-html

Genera en "revision/" lotes de 300 fotos, cada uno con una página (index.html) para marcar
Bien / Mal / Evaluar y corregir cajas desde el celular. "Enviar" en la página produce
revision_<lote>.json, que --aplicar escribe en las anotaciones (LabelMe .json o YOLO .txt).
"""
import argparse
import datetime
import html
import json
import shutil
import sys
from pathlib import Path

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ORDEN = ["unripe", "early-pink", "commercial-basic", "commercial-high", "overripe"]
METRICAS = ("color_pct", "rojo_intenso_pct")  # medidas por caja que calcula el agente de la PC
UMBRALES_AGENTE = [0.5, 50, 75, 90]              # % de superficie rosa/roja entre estados consecutivos


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
                    fig = {"label": s.get("label", ""), "i": len(figs), "pts": [[x / w, y / h] for x, y in pts]}
                    at = s.get("attributes") or {}
                    if isinstance(at, dict) and any(k in at for k in METRICAS):
                        fig["m"] = {k: at[k] for k in METRICAS if isinstance(at.get(k), (int, float))}
                    figs.append(fig)
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
                figs.append({"label": c, "i": len(figs), "pts": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
            elif len(v) > 5 and len(v) % 2 == 1:  # YOLO segmentación
                n = list(map(float, v[1:]))
                figs.append({"label": v[0], "i": len(figs), "pts": [n[i:i + 2] for i in range(0, len(n), 2)]})
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
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0e0e10">
<title>Fresas · __TITULO__</title>
<style>
:root{--bg:#0e0e10;--sup:#1a1a1d;--lin:#2a2a2e;--tx:#ececec;--mu:#8a8a90;--ok:#34c759;--mal:#ff453a;--ev:#ffd60a;--ac:#ffd60a}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;font:15px/1.4 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--tx)}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer}
a{color:inherit;text-decoration:none}
.mu{color:var(--mu)}
header{position:sticky;top:0;z-index:2;background:var(--bg);padding:14px 16px 10px}
.top{display:flex;align-items:center;gap:10px}
.top .t{font-weight:600;font-size:17px;flex:1}
.ic{width:36px;height:36px;border-radius:50%;display:grid;place-items:center;font-size:20px;color:var(--tx)}
.ic:active{background:var(--sup)}
.prog{height:3px;background:var(--lin);border-radius:2px;margin-top:10px;display:flex;overflow:hidden}
.prog i{display:block;height:100%}
#pok{background:var(--ok)}#pev{background:var(--ev)}#pmal{background:var(--mal)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:3px;padding:0 3px 90px}
.c{position:relative;aspect-ratio:4/3;background:var(--sup);overflow:hidden}
.c img{width:100%;height:100%;object-fit:cover;display:block}
.c svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.c .d{display:none;position:absolute;right:6px;top:6px;width:10px;height:10px;border-radius:50%;box-shadow:0 0 0 2px #0008}
.c.ok .d,.c.mal .d,.c.ev .d{display:block}.c.ok .d{background:var(--ok)}.c.mal .d{background:var(--mal)}.c.ev .d{background:var(--ev)}
.c .p{position:absolute;left:5px;top:3px;font-size:12px;text-shadow:0 0 3px #000}
.fab{position:fixed;left:50%;bottom:calc(18px + env(safe-area-inset-bottom));transform:translateX(-50%);background:var(--tx);color:#000;font-weight:600;padding:12px 26px;border-radius:24px;box-shadow:0 4px 16px #0008}
#hoja{position:fixed;inset:0;z-index:8;background:#0009;display:none;align-items:flex-end}
#hoja.on{display:flex}
.panel{width:100%;background:var(--sup);border-radius:16px 16px 0 0;padding:8px 0 calc(12px + env(safe-area-inset-bottom))}
.fila{display:flex;width:100%;align-items:center;justify-content:space-between;padding:14px 20px;text-align:left}
.fila:active{background:var(--lin)}
.sep{height:1px;background:var(--lin);margin:6px 0}
#ley{display:flex;flex-wrap:wrap;gap:6px 14px;padding:12px 20px 8px;font-size:13px;color:var(--mu)}
#ley i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.sw{width:40px;height:24px;border-radius:12px;background:var(--lin);position:relative;transition:.2s}
.sw::after{content:"";position:absolute;left:3px;top:3px;width:18px;height:18px;border-radius:50%;background:#fff;transition:.2s}
.sw.on{background:var(--ok)}.sw.on::after{left:19px}
#ver{position:fixed;inset:0;background:#000;display:none;z-index:5;flex-direction:column}
.vtop{display:flex;align-items:center;gap:4px;padding:calc(6px + env(safe-area-inset-top)) 8px 6px}
.vtop .pos{font-size:14px;padding:0 6px}
.vtop .sp{flex:1}
#bzoom{font-size:13px;font-weight:600}
#bed.on{background:var(--ac);color:#000}
#bojo .tachar{display:none}#bojo.off{color:var(--ac)}#bojo.off .tachar{display:inline}
#ver .img{flex:1;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:0;touch-action:none;user-select:none;-webkit-user-select:none}
#lienzo{position:relative;transform-origin:0 0;--s:1}
#lienzo.anim{transition:transform .22s ease-out}
#ver img{max-width:100vw;max-height:calc(100vh - 140px);display:block;user-select:none;-webkit-user-drag:none;pointer-events:none}
#ver.editando img{max-height:calc(100vh - 190px)}
#vs,#ve{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#vs polygon{stroke-width:calc(2px / var(--s))}
.e{position:absolute;transform:translateY(-100%);color:#000;font-size:10px;font-weight:600;padding:0 4px;border-radius:3px 3px 0 0;white-space:nowrap;pointer-events:none;opacity:.9}
.e.cs{background:none;padding:0;display:flex;flex-direction:column-reverse;align-items:flex-start;gap:2px;opacity:1}
.e.cs i{font-style:normal;padding:0 4px;border-radius:3px 3px 0 0;opacity:.9}
.e .sg{font-weight:700;padding:0 4px;border-radius:3px;box-shadow:0 0 0 1px #000a}
.chip.sug{box-shadow:inset 0 0 0 1.5px var(--ev);color:var(--tx)}
.h{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border:2px solid #fff;border-radius:50%;background:#0006}
#ve{overflow:hidden}
#bzoom{min-width:44px;width:auto;border-radius:18px;padding:0 8px}
#edbar{display:none;align-items:center;gap:4px;padding:8px 8px 0}
#ver.editando #edbar{display:flex}
.chips{flex:1;display:flex;gap:6px;overflow-x:auto;scrollbar-width:none}
.chip{flex:none;display:flex;align-items:center;gap:6px;padding:6px 10px;border-radius:16px;background:var(--sup);font-size:12px;color:var(--mu)}
.chip i{width:9px;height:9px;border-radius:50%}
.chip.act{color:var(--tx);box-shadow:inset 0 0 0 1.5px var(--tx)}
.vbar{display:flex;gap:8px;padding:10px 10px calc(10px + env(safe-area-inset-bottom))}
.vbar .nav{width:36px;font-size:26px;color:var(--mu)}
.vbar .b{flex:1;padding:13px 0;border-radius:12px;font-weight:600;border:1.5px solid}
.b.mal{border-color:var(--mal);color:var(--mal)}.b.ok{border-color:var(--ok);color:var(--ok)}
.b.mal.on{background:var(--mal);color:#000}.b.ok.on{background:var(--ok);color:#000}
.b.ev{border-color:var(--ev);color:var(--ev)}.b.ev.on{background:var(--ev);color:#000}
.estado{display:flex;align-items:center;gap:8px;margin-top:10px;min-height:30px;font-size:13px;color:var(--mu)}
.estado .txt{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.estado b{font-weight:500;color:var(--tx)}
.estado .pend{color:var(--ev)}
.enviar{flex:none;padding:6px 14px;border-radius:14px;background:var(--sup);color:var(--tx);font-size:13px;font-weight:600}
.enviar.urg{background:var(--ev);color:#000}
#envio{position:fixed;inset:0;z-index:10;background:#000a;display:none;align-items:center;justify-content:center;padding:16px}
#envio.on{display:flex}
.caja-env{width:100%;max-width:360px;background:var(--sup);border-radius:16px;padding:20px}
.caja-env h3{margin:0 0 4px;font-size:17px;font-weight:600}
.caja-env .sub{color:var(--mu);font-size:13px;min-height:18px}
.barra{height:6px;background:var(--lin);border-radius:3px;overflow:hidden;margin:16px 0 18px}
.barra i{display:block;height:100%;width:0;background:var(--ok);border-radius:3px;transition:width .5s ease}
.barra.err i{background:var(--mal)}
.pasos{list-style:none;margin:0;padding:0;font-size:14px}
.pasos li{display:flex;gap:10px;align-items:flex-start;padding:7px 0}
.pasos .ico{flex:none;width:20px;height:20px;border-radius:50%;display:grid;place-items:center;font-size:12px;font-weight:700;background:var(--lin);color:var(--mu)}
.pasos .si .ico{background:var(--ok);color:#000}.pasos .no .ico{background:var(--ev);color:#000}
.pasos small{display:block;color:var(--mu);font-size:12px;margin-top:2px}
.pasos code{font-size:11px;background:#0006;padding:1px 4px;border-radius:4px}
.resumen{display:flex;gap:14px;font-size:13px;margin:10px 0 0;color:var(--mu)}
.resumen b{color:var(--tx);font-weight:600}
.caja-env .acciones{display:flex;gap:8px;margin-top:18px}
.caja-env .acciones button{flex:1;padding:11px 0;border-radius:10px;background:var(--lin);font-weight:600}
.caja-env .acciones .prim{background:var(--tx);color:#000}
#guard{cursor:pointer}
.ok-pc{color:var(--ok)}
#toast{position:fixed;left:50%;top:calc(58px + env(safe-area-inset-top));transform:translate(-50%,-8px);background:#fff;color:#000;font-size:13px;font-weight:600;padding:6px 14px;border-radius:14px;opacity:0;transition:.2s;z-index:9;pointer-events:none;white-space:nowrap}
#toast.on{opacity:1;transform:translate(-50%,0)}#toast.err{background:var(--mal)}
.chipest{font-size:12px;padding:3px 10px;border-radius:12px;background:var(--sup);color:var(--mu);white-space:nowrap}
.chipest.ok{background:var(--ok);color:#000}.chipest.mal{background:var(--mal);color:#000}.chipest.ev{background:var(--ev);color:#000}
.vacio{grid-column:1/-1;text-align:center;color:var(--mu);padding:60px 20px}
</style></head><body>
<header>
 <div class="top"><a class="ic" href="../index.html">‹</a><div class="t">__TITULO__</div><span id="cont" class="mu"></span><button class="ic" onclick="hoja(true)">⋯</button></div>
 <div class="prog"><i id="pok"></i><i id="pev"></i><i id="pmal"></i></div>
 <div class="estado"><span class="txt" id="guard" onclick="verEstadoEnvio()"></span><button class="enviar" id="benv" onclick="enviar()">Enviar</button></div>
</header>
<div id="toast"></div>
<div id="envio" onclick="if(event.target==this&&!enviando)cerrarEnvio()"><div class="caja-env">
 <h3 id="etit">Enviar</h3><div class="sub" id="esub"></div>
 <div class="barra" id="ebarra"><i id="ebar"></i></div>
 <ul class="pasos" id="epasos"></ul><div class="resumen" id="eres"></div>
 <div class="acciones" id="eacc"></div>
</div></div>
<main class="grid" id="g"></main>
<button class="fab" id="fab" onclick="seguir()">Revisar</button>
<div id="hoja" onclick="if(event.target==this)hoja(false)"><div class="panel">
 <div id="ley"></div><div class="sep"></div>
 <button class="fila" onclick="mini=!mini;guardarPref();pintar();hoja(true)"><span>Cajas en miniaturas</span><span id="swmini" class="sw"></span></button>
 <button class="fila" onclick="cambiarFiltro()"><span>Mostrar</span><span class="mu" id="lfiltro"></span></button>
 <button class="fila" onclick="enviar()"><span>Enviar resultados a la PC</span><span class="mu" id="ned"></span></button>
 <button class="fila" onclick="conectarGitHub()"><span>Sincronizar con GitHub</span><span class="mu" id="lgh"></span></button>
 <button class="fila" onclick="$('fimport').click()"><span>Importar revisión</span><span class="mu">de otro dispositivo</span></button>
 <input type="file" id="fimport" accept=".json,application/json" style="display:none" onchange="importar(this)">
 <button class="fila" onclick="descargar()"><span>Descargar tabla</span><span class="mu">CSV</span></button>
 <div class="sep"></div>__NAVEGACION__
 <div class="mu" style="font-size:11px;padding:10px 20px 0">Versión de la página: __VERSION__</div>
</div></div>
<div id="ver">
 <div class="vtop"><button class="ic" onclick="cerrar()">✕</button><span class="pos" id="vpos"></span><span class="chipest" id="vest"></span><span class="sp"></span>
  <button class="ic" id="bojo" title="Mostrar u ocultar cajas" onclick="alternarCajas()"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/><line class="tachar" x1="3" y1="3" x2="21" y2="21"/></svg></button><button class="ic" id="bzoom" onclick="cambiarZoom()">1×</button><button class="ic" id="bed" onclick="alternarEdicion()">✎</button></div>
 <div class="img"><div id="lienzo"><img id="vi" draggable="false"><svg id="vs" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div><div id="ve"></div></div>
 <div id="edbar"><div class="chips" id="chips"></div>
  <button class="ic" title="Nueva caja" onclick="nuevaCaja()">＋</button><button class="ic" title="Borrar caja" onclick="borrarCaja()">⌫</button><button class="ic" title="Volver al original" onclick="restaurar()">↺</button></div>
 <div class="vbar"><button class="nav" onclick="mover(-1)">‹</button><button class="b mal" id="bmal" onclick="marcar('mal')">Mal</button><button class="b ev" id="bev" onclick="marcar('ev')">Evaluar</button>
  <button class="b ok" id="bok" onclick="marcar('ok')">Bien</button><button class="nav" onclick="mover(1)">›</button></div>
</div>
<script>
const LOTE=__LOTE__;
const DATOS=__DATOS__;
const CLAVE="fresas_"+LOTE;
let filtro="todas",orden=[],marcas={},ediciones={},descartadas={},rechazadas={},tiempos={},mini=false,mostrarAnot=true,ocultoTemp=false,actual=-1,editando=false,sel=-1,etiquetaNueva=null,arrastre=null;
try{marcas=JSON.parse(localStorage.getItem(CLAVE)||"{}");ediciones=JSON.parse(localStorage.getItem(CLAVE+"_ed")||"{}");mini=localStorage.getItem("fresas_mini")=="1"}catch(e){}
let enviado={marcas:{},ediciones:{},t:0},aplicados={},enviando=false;
fetch("../aplicados.json?t="+Date.now(),{cache:"no-store"}).then(r=>r.ok?r.json():{}).then(j=>{aplicados=j||{};pintarEstado();}).catch(()=>{});
/* estado de la confirmación desde la PC para el último envío */
function estadoPC(){const a=aplicados[LOTE];if(!enviado.fecha)return a?"antiguo":"";
  if(a&&a.envio_fecha===enviado.fecha)return "si";return a&&a.envio_fecha>enviado.fecha?"si":"no";}
try{enviado=JSON.parse(localStorage.getItem(CLAVE+"_env")||"null")||enviado}catch(e){}
try{navigator.storage&&navigator.storage.persist&&navigator.storage.persist()}catch(e){}
let timerToast=0;
function aviso(txt,err){const t=$("toast");t.textContent=txt;t.classList.toggle("err",!!err);t.classList.add("on");
  clearTimeout(timerToast);timerToast=setTimeout(()=>t.classList.remove("on"),err?4000:1300);}
try{descartadas=JSON.parse(localStorage.getItem(CLAVE+"_desc")||"{}");rechazadas=JSON.parse(localStorage.getItem(CLAVE+"_rech")||"{}");tiempos=JSON.parse(localStorage.getItem(CLAVE+"_t")||"{}")}catch(e){}
function tocar(orig){tiempos[orig]=Date.now();}
function guardar(txt){
  try{localStorage.setItem(CLAVE,JSON.stringify(marcas));localStorage.setItem(CLAVE+"_ed",JSON.stringify(ediciones));localStorage.setItem(CLAVE+"_desc",JSON.stringify(descartadas));localStorage.setItem(CLAVE+"_rech",JSON.stringify(rechazadas));localStorage.setItem(CLAVE+"_t",JSON.stringify(tiempos));
    if(localStorage.getItem(CLAVE)!==JSON.stringify(marcas))throw 0;
    if(txt)aviso(txt);programarSubida();return true;}
  catch(e){aviso("⚠ No se pudo guardar. ¿Modo incógnito?",true);return false;}
}
/* fotos con cambios desde el último envío */
function sinEnviar(){const n=new Set();
  const cmp=(a,b)=>{for(const k of new Set([...Object.keys(a),...Object.keys(b)]))if(JSON.stringify(a[k])!==JSON.stringify(b[k]))n.add(k);};
  cmp(marcas,enviado.marcas||{});cmp(ediciones,enviado.ediciones||{});return n.size;}
function hace(t){const m=Math.round((Date.now()-t)/60000);return m<1?"hace un momento":m<60?`hace ${m} min`:m<1440?`hace ${Math.round(m/60)} h`:`hace ${Math.round(m/1440)} d`;}
function pintarEstado(){
  const h=Object.keys(marcas).length,e=Object.keys(ediciones).length,p=sinEnviar();
  let t=h||e?`<b>✓ ${h} guardada${h==1?"":"s"}</b>${e?` · ${e} ✎`:""}`:"Se guarda solo en este celular";
  const pc=estadoPC();
  if(nubeActiva())t+=nube.estado=="subiendo"?" · ☁ subiendo…":nube.estado=="error"?` · <span class="pend">⚠ sin sincronizar (${nube.err})</span>`:" · ☁ sincronizado";
  if(nubeActiva()&&p==0);else if(p)t+=` · <span class="pend">${p} sin enviar</span>`;
  else if(enviado.t)t+=pc=="si"?` · <span class="ok-pc">✓ aplicado en PC</span>`:` · enviado ${hace(enviado.t)}`;
  $("guard").innerHTML=t;$("benv").style.display=h||e?"":"none";$("benv").classList.toggle("urg",p>=20);
  $("ned").textContent=p?p+" sin enviar":enviado.t?"todo enviado":"";
  if($("lgh"))$("lgh").textContent=nubeActiva()?"conectado ✓":"no conectado";}
function guardarPref(){try{localStorage.setItem("fresas_mini",mini?"1":"0")}catch(e){}}
const COLORES={"unripe":"#34c759","early-pink":"#ff8fd8","commercial-basic":"#ff9f0a","commercial-high":"#ff453a","overripe":"#bf5af2"};
const EXTRA=["#64d2ff","#ffd60a","#0a84ff","#ffffff"];
function color(l){if(!(l in COLORES))COLORES[l]=EXTRA[Object.keys(COLORES).length%EXTRA.length];return COLORES[l];}
DATOS.forEach(d=>d.figs.forEach(f=>color(f.label)));
const $=id=>document.getElementById(id);
/* cada caja sabe de qué foto es (propiedad no enumerable: no se exporta) */
function marcarFoto(figs,orig){figs.forEach(f=>{if(f._o!==orig)Object.defineProperty(f,"_o",{value:orig,writable:true,configurable:true,enumerable:false});});return figs;}
function figsDe(d){return marcarFoto(ediciones[d.orig]||d.figs,d.orig);}
function caja(f){const xs=f.pts.map(p=>p[0]),ys=f.pts.map(p=>p[1]);return[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];}
function deCaja(label,[x1,y1,x2,y2]){return{label,pts:[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]};}
function sugDe(f){if(!f.sug||f.sug===f.label)return null;
  if(f._o&&(rechazadas[f._o]||[]).includes(f.i))return null;   // la persona decidió mantener su etiqueta
  return f.sug;}
/* cajas que el modelo ve y no están anotadas; desaparecen al aceptarlas o descartarlas */
function faltan(d){const ya=figsDe(d);return(d.falta||[]).filter((p,k)=>!(descartadas[d.orig]||[]).includes(k)&&!ya.some(f=>iouN(caja(f),caja(p))>=0.5));}
function iouN(a,b){const ix=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])),iy=Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1])),I=ix*iy;
  const U=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-I;return U?I/U:0;}
/* dos cajas casi idénticas sobre la misma fresa: se avisa en ambas hasta que se borre una */
function dupsDe(figs){const r=new Set();for(let a=0;a<figs.length;a++)for(let b=a+1;b<figs.length;b++)
  if(iouN(caja(figs[a]),caja(figs[b]))>=0.8){r.add(a);r.add(b);}return r;}
function svgFaltan(d){return faltan(d).map(p=>`<polygon points="${p.pts.map(q=>q.join(",")).join(" ")}" fill="rgba(255,255,255,.08)" stroke="#fff" stroke-width="2" vector-effect="non-scaling-stroke" stroke-dasharray="3 4"/>`).join("");}
function svgDe(figs,conSel){
  return figs.map((f,i)=>`<polygon points="${f.pts.map(p=>p.join(",")).join(" ")}" fill="${conSel&&i==sel?"rgba(255,255,255,.12)":"none"}" stroke="${color(f.label)}" stroke-width="2" vector-effect="non-scaling-stroke"${sugDe(f)?' stroke-dasharray="6 4"':""}/>`).join("");
}
/* etiquetas y esquinas en píxeles de pantalla (fuera del zoom, siempre nítidas) */
function etqDe(figs,conSel){
  const r=$("vi").getBoundingClientRect(),v=document.querySelector("#ver .img").getBoundingClientRect();
  const X=x=>(r.left-v.left+x*r.width).toFixed(1)+"px",Y=y=>(r.top-v.top+y*r.height).toFixed(1)+"px";
  const dup=dupsDe(figs);
  let h=figs.map((f,k)=>{const[x,y]=caja(f),s=sugDe(f),dp=dup.has(k);return s||dp
    ?`<span class="e cs" style="left:${X(x)};top:${Y(y)}"><i style="background:${color(f.label)}">${f.label}</i>${s?`<b class="sg" style="background:${color(s)}">¿${s}?</b>`:""}${dp?`<b class="sg" style="background:#ff453a;color:#fff">¿duplicada? borra una</b>`:""}</span>`
    :`<span class="e" style="left:${X(x)};top:${Y(y)};background:${color(f.label)}">${f.label}</span>`}).join("");
  if(conSel&&sel>=0&&figs[sel]){const[x1,y1,x2,y2]=caja(figs[sel]);
    [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].forEach(([x,y])=>h+=`<span class="h" style="left:${X(x)};top:${Y(y)}"></span>`);}
  return h;
}
function verCajas(){return !cargando&&(editando||(mostrarAnot&&!ocultoTemp));}
function pintarCapa(){if(actual<0)return;const d=DATOS[actual];
  $("ve").innerHTML=verCajas()?etqDe(figsDe(d),editando)+etqFaltan(d):"";}
function etqFaltan(d){const r=$("vi").getBoundingClientRect(),v=document.querySelector("#ver .img").getBoundingClientRect();
  return faltan(d).map(p=>{const[x,y]=caja(p);return `<span class="e cs" style="left:${(r.left-v.left+x*r.width).toFixed(1)}px;top:${(r.top-v.top+y*r.height).toFixed(1)}px"><b class="sg" style="background:#fff">¿falta ${p.label}?</b></span>`}).join("");}
let rafCapa=0;
function seguirCapa(ms){cancelAnimationFrame(rafCapa);const fin=performance.now()+ms;
  const paso=()=>{pintarCapa();if(performance.now()<fin)rafCapa=requestAnimationFrame(paso);};paso();}
DATOS.forEach(d=>marcarFoto(d.figs,d.orig));
const HAY_SUG=DATOS.some(d=>d.figs.some(sugDe)||(d.falta||[]).length||dupsDe(d.figs).size);
const FILTROS=HAY_SUG?{todas:"Todas",sug:"Con sugerencia",ev:"A evaluar",mal:"Mal",ok:"Bien",pend:"Sin revisar"}:{todas:"Todas",ev:"A evaluar",mal:"Mal",ok:"Bien",pend:"Sin revisar"};
function pasaFiltro(d){const m=marcas[d.orig];return filtro=="todas"||(filtro=="sug"?figsDe(d).some(sugDe)||faltan(d).length>0||dupsDe(figsDe(d)).size>0:filtro=="pend"?!m:m==filtro);}
function cambiarFiltro(){const k=Object.keys(FILTROS);filtro=k[(k.indexOf(filtro)+1)%k.length];pintar();hoja(true);}
function hoja(on){$("hoja").classList.toggle("on",on);$("lfiltro").textContent=FILTROS[filtro];$("swmini").classList.toggle("on",mini);
  const n={};DATOS.forEach(d=>figsDe(d).forEach(f=>n[f.label]=(n[f.label]||0)+1));
  $("ley").innerHTML=Object.keys(n).sort().map(l=>`<span><i style="background:${color(l)}"></i>${l} ${n[l]}</span>`).join("");}
function pintar(){
  const g=$("g");g.innerHTML="";orden=[];
  DATOS.forEach((d,i)=>{
    if(!pasaFiltro(d))return;orden.push(i);
    const c=document.createElement("div");c.className="c "+(marcas[d.orig]||"");
    c.innerHTML=`<img loading="lazy" src="${d.img}">${mini?`<svg viewBox="0 0 1 1" preserveAspectRatio="none">${svgDe(figsDe(d))}</svg>`:""}<span class="d"></span>${d.orig in ediciones?'<span class="p">✎</span>':(figsDe(d).some(sugDe)||faltan(d).length||dupsDe(figsDe(d)).size)&&!marcas[d.orig]?'<span class="p">?</span>':""}`;
    c.onclick=()=>abrir(i);g.appendChild(c);
  });
  if(!orden.length)g.innerHTML=`<div class="vacio">No hay fotos en «${FILTROS[filtro]}»</div>`;
  const v=Object.values(marcas),ok=v.filter(x=>x=="ok").length,mal=v.filter(x=>x=="mal").length,ev=v.filter(x=>x=="ev").length,t=DATOS.length,hechas=ok+mal+ev;
  $("cont").textContent=(filtro=="todas"?"":FILTROS[filtro]+" · ")+`${hechas}/${t}`;
  $("pok").style.width=ok/t*100+"%";$("pev").style.width=ev/t*100+"%";$("pmal").style.width=mal/t*100+"%";
  pintarEstado();
  $("fab").textContent=filtro!="todas"?"Revisar "+orden.length:hechas==0?"Empezar":hechas>=t?"Listo ✓":"Continuar";
}
function seguir(){if(!orden.length)return;if(filtro!="todas"){abrir(orden[0]);return;}const i=DATOS.findIndex(d=>!marcas[d.orig]);abrir(i<0?0:i);}
function pintarVisor(){const d=DATOS[actual],f=figsDe(d),m=marcas[d.orig];
  $("vs").innerHTML=verCajas()?svgDe(f,editando)+svgFaltan(d):"";pintarCapa();$("bojo").classList.toggle("off",!mostrarAnot);
  $("vpos").textContent=`${orden.indexOf(actual)+1} / ${orden.length}`;
  $("bok").classList.toggle("on",m=="ok");$("bmal").classList.toggle("on",m=="mal");$("bev").classList.toggle("on",m=="ev");
  const ve=$("vest");ve.className="chipest "+(m||"");ve.textContent=(m?{ok:"Bien",mal:"Mal",ev:"Evaluar"}[m]:"Sin marcar")+(d.orig in ediciones?" · ✎":"");
  if(editando)pintarChips();}
function pintarChips(){
  const f=figsDe(DATOS[actual]),act=sel>=0&&f[sel]?f[sel].label:etiquetaNueva;
  const sg=sel>=0&&f[sel]?sugDe(f[sel]):null;
  $("chips").innerHTML=Object.keys(COLORES).map(l=>`<button class="chip ${l==act?"act":""} ${l==sg&&l!=act?"sug":""}" onclick="ponerEtiqueta('${l}')"><i style="background:${color(l)}"></i>${l}</button>`).join("");
}
let cargando=false;
function abrir(i){actual=i;sel=-1;const vi=$("vi");
  vi.onload=vi.onerror=()=>{cargando=false;aplicarVista(false);pintarVisor();};
  if(vi.getAttribute("src")!==DATOS[i].img){cargando=true;vi.src=DATOS[i].img;}$("ver").style.display="flex";reiniciarZoom(false);pintarVisor();}
function cerrar(){if(editando)alternarEdicion();$("ver").style.display="none";pintar();}
function mover(k){const n=orden[orden.indexOf(actual)+k];if(n!==undefined)abrir(n);else cerrar();}
function marcar(v){const o=DATOS[actual].orig;tocar(o);marcas[o]=marcas[o]==v?"":v;if(!marcas[o]){delete marcas[o];guardar("Marca quitada");pintarVisor();return;}
  guardar("✓ Guardado: "+{ok:"Bien",mal:"Mal",ev:"Evaluar"}[v]);
  pintarVisor();setTimeout(()=>mover(1),150);}
function alternarCajas(){mostrarAnot=!mostrarAnot;pintarVisor();aviso(mostrarAnot?"Cajas visibles":"Cajas ocultas · toca el ojo para verlas");}
function alternarEdicion(){editando=!editando;sel=-1;
  $("ver").classList.toggle("editando",editando);$("bed").classList.toggle("on",editando);pintarVisor();setTimeout(()=>aplicarVista(false),0);}
function editables(){const d=DATOS[actual];if(!ediciones[d.orig])ediciones[d.orig]=JSON.parse(JSON.stringify(d.figs));return ediciones[d.orig];}
function cambio(){const d=DATOS[actual];tocar(d.orig);if(JSON.stringify(ediciones[d.orig])==JSON.stringify(d.figs))delete ediciones[d.orig];guardar("✓ Cajas guardadas");pintarVisor();}
function ponerEtiqueta(l){etiquetaNueva=l;if(sel<0){pintarChips();return;}
  const d=DATOS[actual],actualF=figsDe(d)[sel],sg=sugDe(actualF);
  if(sg&&l!==sg&&actualF.i!==undefined){            // eligió otra cosa que la sugerida: la sugerencia se descarta
    const r=rechazadas[d.orig]=rechazadas[d.orig]||[];if(!r.includes(actualF.i))r.push(actualF.i);}
  if(l===actualF.label){guardar(sg?`Se mantiene ${l}`:"");pintarVisor();return;}
  editables()[sel].label=l;cambio();}
function decidirFaltante(fal){
  const d=DATOS[actual],k=d.falta.indexOf(fal);
  if(confirm(`¿Agregar esta caja como «${fal.label}»?\n\nAceptar = agregar · Cancelar = descartar la sugerencia`)){
    const f=editables();f.push({label:fal.label,pts:JSON.parse(JSON.stringify(fal.pts))});sel=f.length-1;cambio();}
  else{(descartadas[d.orig]=descartadas[d.orig]||[]).push(k);guardar("Sugerencia descartada");pintarVisor();}}
function nuevaCaja(){const f=editables(),l=etiquetaNueva||(f[0]&&f[0].label)||Object.keys(COLORES)[0];
  f.push(deCaja(l,[.42,.42,.58,.58]));sel=f.length-1;cambio();}
function borrarCaja(){if(sel<0)return;editables().splice(sel,1);sel=-1;cambio();}
function restaurar(){if(confirm("¿Volver a las cajas originales de esta foto?")){tocar(DATOS[actual].orig);delete ediciones[DATOS[actual].orig];sel=-1;guardar("Cajas originales restauradas");pintarVisor();}}
function punto(ev){const r=$("vi").getBoundingClientRect();
  return[Math.min(1,Math.max(0,(ev.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(ev.clientY-r.top)/r.height))];}

/* ---------- Zoom y gestos: pellizcar, doble toque, arrastrar, deslizar ---------- */
const vista=document.querySelector("#ver .img"),lienzo=$("lienzo"),ZMAX=8;
let esc=1,tx=0,ty=0;
const dedos=new Map();let gesto=null,ultimoToque=null,timerToque=null;
function base(){return{W:vista.clientWidth,H:vista.clientHeight,w:lienzo.offsetWidth,h:lienzo.offsetHeight,ox:lienzo.offsetLeft,oy:lienzo.offsetTop};}
function limitar(){const b=base(),sw=b.w*esc,sh=b.h*esc;
  tx=sw<=b.W?(b.W-sw)/2-b.ox:Math.min(-b.ox,Math.max(b.W-sw-b.ox,tx));
  ty=sh<=b.H?(b.H-sh)/2-b.oy:Math.min(-b.oy,Math.max(b.H-sh-b.oy,ty));}
function aplicarVista(anim,sinLimite){if(!sinLimite)limitar();
  lienzo.classList.toggle("anim",!!anim);lienzo.style.transform=`translate(${tx}px,${ty}px) scale(${esc})`;lienzo.style.setProperty("--s",esc);anim?seguirCapa(260):pintarCapa();
  $("bzoom").textContent=esc<1.05?"1×":(Math.round(esc*10)/10)+"×";}
function reiniciarZoom(anim){esc=1;tx=0;ty=0;aplicarVista(anim);}
/* zoom a "nueva" escala manteniendo fijo el punto de pantalla (px,py) */
function zoomEn(nueva,px,py,anim){const vr=vista.getBoundingClientRect(),b=base();
  const ox=vr.left+b.ox,oy=vr.top+b.oy,cx=(px-ox-tx)/esc,cy=(py-oy-ty)/esc;
  esc=Math.min(ZMAX,Math.max(1,nueva));tx=px-ox-esc*cx;ty=py-oy-esc*cy;aplicarVista(anim);}
function cambiarZoom(){const vr=vista.getBoundingClientRect();
  if(esc>1.05)reiniciarZoom(true);else zoomEn(2.5,vr.left+vr.width/2,vr.top+vr.height/2,true);}
vista.addEventListener("wheel",ev=>{ev.preventDefault();zoomEn(esc*Math.exp(-ev.deltaY*(ev.ctrlKey?.01:.002)),ev.clientX,ev.clientY,false);},{passive:false});
window.addEventListener("resize",()=>{if($("ver").style.display=="flex")aplicarVista(false);});
function dosDedos(){const[a,b]=[...dedos.values()];return{d:Math.hypot(a.x-b.x,a.y-b.y)||1,x:(a.x+b.x)/2,y:(a.y+b.y)/2};}
function iniciarPinza(){clearTimeout(timerPeek);if(ocultoTemp){ocultoTemp=false;pintarVisor();}if(gesto&&(gesto.tipo=="esq"||gesto.tipo=="mover")&&gesto.activo)cambio();
  const m=dosDedos();gesto={tipo:"pinza",d0:m.d,x0:m.x,y0:m.y,e0:esc,tx0:tx,ty0:ty};}
let timerPeek=0;
function iniciarUnDedo(ev){gesto={tipo:"pendiente",x0:ev.clientX,y0:ev.clientY,tx0:tx,ty0:ty,t0:Date.now()};
  if(!editando){const g=gesto;clearTimeout(timerPeek);
    timerPeek=setTimeout(()=>{if(gesto===g&&g.tipo=="pendiente"&&dedos.size==1){g.tipo="peek";ocultoTemp=true;pintarVisor();}},380);return;}
  const p=punto(ev),f=figsDe(DATOS[actual]),r=$("vi").getBoundingClientRect(),tol=22/r.width,tolY=22/r.height;
  if(sel>=0&&f[sel]){const[x1,y1,x2,y2]=caja(f[sel]);
    const esq=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]].findIndex(([x,y])=>Math.abs(p[0]-x)<tol&&Math.abs(p[1]-y)<tolY);
    if(esq>=0){gesto={tipo:"esq",esq,caja:[x1,y1,x2,y2],sx:ev.clientX,sy:ev.clientY};return;}}
  let mejor=-1,area=9;
  f.forEach((fi,i)=>{const[x1,y1,x2,y2]=caja(fi);if(p[0]>=x1&&p[0]<=x2&&p[1]>=y1&&p[1]<=y2&&(x2-x1)*(y2-y1)<area){mejor=i;area=(x2-x1)*(y2-y1);}});
  if(mejor<0){const fal=faltan(DATOS[actual]).find(q=>{const[x1,y1,x2,y2]=caja(q);return p[0]>=x1&&p[0]<=x2&&p[1]>=y1&&p[1]<=y2;});
    if(fal){gesto=null;decidirFaltante(fal);return;}}
  if(mejor>=0){sel=mejor;gesto={tipo:"mover",inicio:p,caja:caja(f[sel]),sx:ev.clientX,sy:ev.clientY};pintarVisor();}
}
vista.addEventListener("pointerdown",ev=>{
  if(ev.pointerType=="mouse"&&ev.button!==0)return;
  try{vista.setPointerCapture(ev.pointerId)}catch(e){}dedos.set(ev.pointerId,{x:ev.clientX,y:ev.clientY});
  if(dedos.size==2)iniciarPinza();else if(dedos.size==1)iniciarUnDedo(ev);
});
vista.addEventListener("pointermove",ev=>{
  if(!dedos.has(ev.pointerId)||!gesto)return;dedos.set(ev.pointerId,{x:ev.clientX,y:ev.clientY});
  if(gesto.tipo=="pinza"){if(dedos.size<2)return;const m=dosDedos(),vr=vista.getBoundingClientRect(),b=base();
    const ox=vr.left+b.ox,oy=vr.top+b.oy,cx=(gesto.x0-ox-gesto.tx0)/gesto.e0,cy=(gesto.y0-oy-gesto.ty0)/gesto.e0;
    esc=Math.min(ZMAX,Math.max(.8,gesto.e0*m.d/gesto.d0));tx=m.x-ox-esc*cx;ty=m.y-oy-esc*cy;
    aplicarVista(false,esc<1);return;}
  const dx=ev.clientX-(gesto.x0??0),dy=ev.clientY-(gesto.y0??0);
  if((gesto.tipo=="pendiente"||gesto.tipo=="peek")&&Math.hypot(dx,dy)>8){clearTimeout(timerPeek);if(ocultoTemp){ocultoTemp=false;pintarVisor();}gesto.tipo=esc>1.02?"pan":"deslizar";}
  if(gesto.tipo=="pan"){tx=gesto.tx0+dx;ty=gesto.ty0+dy;aplicarVista(false);return;}
  if(gesto.tipo=="deslizar"){if(!editando){lienzo.classList.remove("anim");lienzo.style.transform=`translate(${gesto.tx0+dx}px,${gesto.ty0}px)`;pintarCapa();}return;}
  if(gesto.tipo=="esq"||gesto.tipo=="mover"){
    /* un toque para seleccionar no debe mover la caja: solo cuenta si el dedo se desplaza >8 px */
    if(!gesto.activo){if(Math.hypot(ev.clientX-gesto.sx,ev.clientY-gesto.sy)<8)return;gesto.activo=true;
      if(gesto.tipo=="mover")gesto.inicio=punto(ev);}
    const p=punto(ev),f=editables();let[x1,y1,x2,y2]=gesto.caja;
    if(gesto.tipo=="mover"){let mx=p[0]-gesto.inicio[0],my=p[1]-gesto.inicio[1];
      mx=Math.min(1-x2,Math.max(-x1,mx));my=Math.min(1-y2,Math.max(-y1,my));x1+=mx;x2+=mx;y1+=my;y2+=my;}
    else{const e=gesto.esq;if(e==0||e==3)x1=p[0];else x2=p[0];if(e<2)y1=p[1];else y2=p[1];}
    f[sel]={...f[sel],...deCaja(f[sel].label,[Math.min(x1,x2),Math.min(y1,y2),Math.max(x1,x2),Math.max(y1,y2)])};
    if(gesto.tipo=="esq"){const c=gesto.caja=caja(f[sel]),q=[[c[0],c[1]],[c[2],c[1]],[c[2],c[3]],[c[0],c[3]]];
      gesto.esq=q.map(([x,y],i)=>[Math.hypot(x-p[0],y-p[1]),i]).sort((a,b)=>a[0]-b[0])[0][1];}
    pintarVisor();}
});
function toque(ev){
  const ahora=Date.now();
  if(ultimoToque&&ahora-ultimoToque.t<300&&Math.hypot(ev.clientX-ultimoToque.x,ev.clientY-ultimoToque.y)<40){
    clearTimeout(timerToque);ultimoToque=null;
    if(esc>1.05)reiniciarZoom(true);else zoomEn(2.5,ev.clientX,ev.clientY,true);return;}
  ultimoToque={t:ahora,x:ev.clientX,y:ev.clientY};
  timerToque=setTimeout(()=>{ultimoToque=null;
    if(editando&&sel>=0){sel=-1;pintarVisor();}},280);
}
function soltarDedo(ev){
  if(!dedos.has(ev.pointerId))return;dedos.delete(ev.pointerId);clearTimeout(timerPeek);
  if(ocultoTemp){ocultoTemp=false;pintarVisor();}
  const g=gesto;if(!g)return;
  if(g.tipo=="pinza"){
    if(dedos.size==1){const[id,d]=[...dedos.entries()][0];gesto={tipo:"pan",x0:d.x,y0:d.y,tx0:tx,ty0:ty};}
    else gesto=null;
    if(esc<1.05)reiniciarZoom(true);else aplicarVista(true);return;}
  if(dedos.size>0)return;gesto=null;
  if(g.tipo=="esq"||g.tipo=="mover"){if(g.activo)cambio();return;}
  if(g.tipo=="deslizar"){const dx=ev.clientX-g.x0,dy=ev.clientY-g.y0;
    if(!editando&&Math.abs(dx)>60&&Math.abs(dx)>Math.abs(dy)*1.5&&ev.type=="pointerup"){mover(dx<0?1:-1);}else aplicarVista(true);return;}
  if(g.tipo=="pendiente"&&ev.type=="pointerup")toque(ev);
}
vista.addEventListener("pointerup",soltarDedo);vista.addEventListener("pointercancel",soltarDedo);
function descargarArchivo(archivo){const a=document.createElement("a");a.href=URL.createObjectURL(archivo);a.download=archivo.name;a.click();}
/* listo(modo) con modo "compartido" o "descargado"; cancelado() si la persona cierra el menú de compartir */
function bajar(nombre,contenido,tipo,listo,cancelado){
  const archivo=new File([contenido],nombre,{type:tipo});
  if(navigator.canShare&&navigator.canShare({files:[archivo]})){
    navigator.share({files:[archivo],title:nombre}).then(()=>listo&&listo("compartido"))
      .catch(e=>{if(e&&e.name=="AbortError"){cancelado&&cancelado();}else{descargarArchivo(archivo);listo&&listo("descargado");}});
    return;}
  descargarArchivo(archivo);listo&&listo("descargado");
}
function resumenDe(m,ed){const v=Object.values(m);
  return{fotos:v.length,bien:v.filter(x=>x=="bien"||x=="ok").length,mal:v.filter(x=>x=="mal").length,
         evaluar:v.filter(x=>x=="evaluar"||x=="ev").length,cajas:Object.keys(ed).length};}
function htmlResumen(r){return `<span><b>${r.fotos}</b> fotos</span><span><b>${r.bien}</b> bien</span><span><b>${r.mal}</b> mal</span><span><b>${r.evaluar}</b> evaluar</span>${r.cajas?`<span><b>${r.cajas}</b> ✎</span>`:""}`;}
function paso(estado,titulo,detalle){return `<li class="${estado}"><span class="ico">${estado=="si"?"✓":estado=="no"?"!":"·"}</span><span>${titulo}${detalle?`<small>${detalle}</small>`:""}</span></li>`;}
function pasosEnvio(){
  const h=Object.keys(marcas).length+Object.keys(ediciones).length,p=sinEnviar(),pc=estadoPC(),a=aplicados[LOTE];
  return paso(h?"si":"","Guardado en este celular",h?`${Object.keys(marcas).length} fotos marcadas`:"")+
    paso(!enviado.t?"":p?"no":"si","Enviado",!enviado.t?"Todavía no se ha enviado":p?`${p} cambios nuevos sin enviar · último envío ${hace(enviado.t)}`:`${hace(enviado.t)} · revision_${LOTE}.json`)+
    paso(pc=="si"?"si":enviado.t?"no":"","Aplicado en la PC",pc=="si"?`${a.fotos} fotos${a.cajas?`, ${a.cajas} con cajas corregidas`:""} · ${new Date(a.aplicado_fecha).toLocaleString()}`:
      enviado.t?(nubeActiva()?`En la PC: <code>--aplicar github</code> y luego <code>git push</code>`:`En la PC: <code>--aplicar revision_${LOTE}.json</code> y luego <code>git push</code>`):"");
}
function abrirEnvio(tit,sub,pct,err){$("envio").classList.add("on");$("etit").textContent=tit;$("esub").textContent=sub;
  $("ebarra").classList.toggle("err",!!err);$("ebar").style.width=pct+"%";}
function cerrarEnvio(){$("envio").classList.remove("on");}
function verEstadoEnvio(){if(!Object.keys(marcas).length&&!Object.keys(ediciones).length)return;
  const pc=estadoPC(),p=sinEnviar();
  abrirEnvio("Estado del envío",p?"Hay cambios sin enviar":pc=="si"?"Todo está aplicado en la PC":"",p?33:pc=="si"?100:66);
  $("epasos").innerHTML=pasosEnvio();$("eres").innerHTML=enviado.resumen?htmlResumen(enviado.resumen):"";
  $("eacc").innerHTML=(p?`<button class="prim" onclick="enviar()">Enviar ahora</button>`:"")+`<button onclick="cerrarEnvio()">Cerrar</button>`;}
function enviar(){
  if(nubeActiva()){hoja(false);$("epasos").innerHTML="";$("eacc").innerHTML="";$("eres").innerHTML=htmlResumen(datosActuales().resumen);
    abrirEnvio("Sincronizando…","Guardando en GitHub",30);
    sincronizar().then(ok=>{abrirEnvio(ok?"✓ Guardado en GitHub":"No se pudo sincronizar",ok?`rama «resultados», ${LOTE}.json · ${new Date().toLocaleTimeString()}`:nube.err,ok?(estadoPC()=="si"?100:66):100,!ok);
      $("epasos").innerHTML=pasosEnvio();$("eacc").innerHTML=`<button class="prim" onclick="cerrarEnvio()">Listo</button>`;});return;}
  if(!Object.keys(marcas).length&&!Object.keys(ediciones).length){aviso("Todavía no hay nada que enviar");return;}
  if(enviando)return;enviando=true;hoja(false);
  const datos=datosActuales(),fecha=datos.fecha,res=datos.resumen;
  $("epasos").innerHTML="";$("eacc").innerHTML="";$("eres").innerHTML=htmlResumen(res);
  abrirEnvio("Enviando…","Preparando el archivo",8);
  setTimeout(()=>{abrirEnvio("Enviando…","Elige dónde enviarlo (WhatsApp, correo, Drive…)",45);
    const contenido=JSON.stringify(datos,null,1);
    bajar(`revision_${LOTE}.json`,contenido,"application/json",modo=>{
      enviado={marcas:JSON.parse(JSON.stringify(marcas)),ediciones:JSON.parse(JSON.stringify(ediciones)),t:Date.now(),fecha,resumen:res};
      let ok=true;try{localStorage.setItem(CLAVE+"_env",JSON.stringify(enviado))}catch(e){ok=false}
      enviando=false;pintar();
      abrirEnvio("✓ Enviado",modo=="descargado"?`Se descargó revision_${LOTE}.json (${Math.ceil(contenido.length/1024)} KB). Pásalo a la PC.`
        :`revision_${LOTE}.json (${Math.ceil(contenido.length/1024)} KB) compartido.`,66);
      $("epasos").innerHTML=pasosEnvio();
      $("eacc").innerHTML=`<button class="prim" onclick="cerrarEnvio()">Listo</button>`;
      if(!ok)aviso("⚠ Se envió, pero no se pudo guardar el registro del envío",true);
    },()=>{enviando=false;abrirEnvio("Envío cancelado","No se envió nada. Tus marcas siguen guardadas en el celular.",100,true);
      $("epasos").innerHTML="";$("eacc").innerHTML=`<button class="prim" onclick="enviar()">Reintentar</button><button onclick="cerrarEnvio()">Cerrar</button>`;});
  },450);
}
/* ---------- Sincronizar entre dispositivos (sin servidor) ----------
   Combina una revisión (archivo enviado desde otro dispositivo, o la ya aplicada en la PC y publicada)
   con la local. Por foto gana el cambio más reciente; las fotos que solo tiene un lado se agregan. */
function datosActuales(){
  const est={ok:"bien",mal:"mal",ev:"evaluar"},m={};for(const k in marcas)m[k]=est[marcas[k]];
  const originales={};DATOS.forEach(d=>{if(d.orig in ediciones)originales[d.orig]=d.figs;});
  return{formato:2,lote:LOTE,fecha:new Date().toISOString(),marcas:m,correcciones:JSON.parse(JSON.stringify(ediciones)),
         originales,tiempos,rechazadas,descartadas,resumen:resumenDe(m,ediciones)};}
function fusionarRevision(r,yaAplicada){
  if(!r||(r.lote&&r.lote!==LOTE))return 0;
  const inv={bien:"ok",mal:"mal",evaluar:"ev"},rt=r.tiempos||{},base=Date.parse(r.fecha)||0;let n=0;
  const fotos=new Set([...Object.keys(r.marcas||{}),...Object.keys(r.correcciones||{})]);
  for(const o of fotos){
    const tr=rt[o]||base,tl=tiempos[o]||0;if(tr<=tl)continue;
    const m=r.marcas&&r.marcas[o];
    if(m&&inv[m]&&marcas[o]!==inv[m]){marcas[o]=inv[m];n++;}
    if(r.correcciones&&o in r.correcciones&&JSON.stringify(ediciones[o])!==JSON.stringify(r.correcciones[o])){ediciones[o]=r.correcciones[o];n++;}
    tiempos[o]=tr;
    if(yaAplicada){if(marcas[o])enviado.marcas[o]=marcas[o];if(o in ediciones)enviado.ediciones[o]=ediciones[o];}
  }
  for(const[o,l]of Object.entries(r.rechazadas||{})){const a=rechazadas[o]=rechazadas[o]||[];l.forEach(i=>a.includes(i)||a.push(i));}
  for(const[o,l]of Object.entries(r.descartadas||{})){const a=descartadas[o]=descartadas[o]||[];l.forEach(i=>a.includes(i)||a.push(i));}
  if(yaAplicada){try{localStorage.setItem(CLAVE+"_env",JSON.stringify(enviado))}catch(e){}}
  return n;}
/* ---------- Sincronización automática con GitHub (rama «resultados», un archivo <lote>.json) ----------
   El token lo ingresa la persona en cada dispositivo y queda solo en su navegador; la página no trae ninguno. */
const GH=(()=>{const h=location.hostname.match(/^([^.]+)\.github\.io$/i),seg=location.pathname.split("/").filter(Boolean);
  return h&&seg.length?{owner:h[1],repo:seg[0]}:null;})();
const RAMA="resultados";let nube={token:"",sha:undefined,estado:"off",err:""},timerNube=0,subiendo=null;
try{nube.token=localStorage.getItem("fresas_token")||""}catch(e){}
function nubeActiva(){return !!(GH&&nube.token);}
function api(ruta,op={}){return fetch(`https://api.github.com/repos/${GH.owner}/${GH.repo}${ruta}`,{...op,cache:"no-store",
  headers:{Authorization:"Bearer "+nube.token,Accept:"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28",...(op.body?{"Content-Type":"application/json"}:{})}});}
const b64a=t=>btoa(unescape(encodeURIComponent(t))),a64b=b=>decodeURIComponent(escape(atob(b.replace(/\s/g,""))));
function errGH(r,paso){const q=paso?` (${paso})`:"";
  return new Error(r.status==401?"token inválido, vencido o mal copiado"+q
    :r.status==404?`el token no tiene acceso al repositorio ${GH.repo}: en «Repository access» elige «Only select repositories» y márcalo`+q
    :r.status==403?"el token puede ver el repositorio pero no escribir: en Permissions → Repositories, «Contents» debe estar en «Read and write»"+q
    :r.status==422?"GitHub rechazó la operación (422)"+q:"GitHub respondió "+r.status+q);}
async function leerNube(nombre){
  const r=await api(`/contents/${encodeURIComponent(nombre)}?ref=${RAMA}`);if(r.status==404)return null;if(!r.ok)throw errGH(r,"leer lo sincronizado");
  const j=await r.json();let txt=j.content&&j.encoding=="base64"?a64b(j.content):null;
  if(txt===null){const b=await api(`/git/blobs/${j.sha}`);if(!b.ok)throw errGH(b);txt=a64b((await b.json()).content);}
  return{sha:j.sha,datos:JSON.parse(txt)};}
async function asegurarRama(){
  const r=await api(`/git/ref/heads/${RAMA}`);if(r.ok)return;if(r.status!=404)throw errGH(r,"leer la rama");
  const repo=await api("");if(!repo.ok)throw errGH(repo,"abrir el repositorio");
  const def=(await repo.json()).default_branch;
  const b=await api(`/git/ref/heads/${encodeURIComponent(def)}`);if(!b.ok)throw errGH(b,"leer la rama principal");
  const f=await api("/git/refs",{method:"POST",body:JSON.stringify({ref:"refs/heads/"+RAMA,sha:(await b.json()).object.sha})});
  if(!f.ok&&f.status!=422)throw errGH(f,"crear la rama «resultados»");}
function programarSubida(ms){if(!nubeActiva())return;clearTimeout(timerNube);timerNube=setTimeout(()=>sincronizar(),ms??3000);}
/* trae lo de GitHub (otro dispositivo), combina (gana lo más reciente por foto) y sube el resultado */
function sincronizar(){if(!nubeActiva())return Promise.resolve(false);if(subiendo)return subiendo.then(()=>sincronizar());
  subiendo=(async()=>{nube.estado="subiendo";pintarEstado();
    try{await asegurarRama();
      for(let intento=0;intento<3;intento++){
        const x=await leerNube(LOTE+".json");nube.sha=x?x.sha:undefined;
        if(x&&fusionarRevision(x.datos,false)){guardarLocal();pintar();}
        const d=datosActuales();
        if(x&&JSON.stringify({m:x.datos.marcas,c:x.datos.correcciones})===JSON.stringify({m:d.marcas,c:d.correcciones})){registrarEnvioNube(d);break;}
        const r=await api(`/contents/${encodeURIComponent(LOTE+".json")}`,{method:"PUT",body:JSON.stringify({
          message:`Revisión ${LOTE}: ${d.resumen.fotos} fotos (${d.resumen.bien} bien, ${d.resumen.mal} mal, ${d.resumen.evaluar} evaluar)`,
          content:b64a(JSON.stringify(d,null,1)),branch:RAMA,...(nube.sha?{sha:nube.sha}:{})})});
        if(r.status==409||r.status==422)continue;          // otro dispositivo escribió justo ahora: releer y combinar
        if(!r.ok)throw errGH(r,"guardar la revisión");
        nube.sha=(await r.json()).content.sha;registrarEnvioNube(d);break;}
      nube.estado="ok";nube.err="";return true;
    }catch(e){nube.estado="error";nube.err=e.message=="Failed to fetch"?"sin conexión":e.message;
      clearTimeout(timerNube);timerNube=setTimeout(()=>sincronizar(),30000);return false;}
    finally{subiendo=null;pintarEstado();}})();
  return subiendo;}
function guardarLocal(){try{localStorage.setItem(CLAVE,JSON.stringify(marcas));localStorage.setItem(CLAVE+"_ed",JSON.stringify(ediciones));
  localStorage.setItem(CLAVE+"_desc",JSON.stringify(descartadas));localStorage.setItem(CLAVE+"_rech",JSON.stringify(rechazadas));
  localStorage.setItem(CLAVE+"_t",JSON.stringify(tiempos));}catch(e){}}
function registrarEnvioNube(d){enviado={marcas:JSON.parse(JSON.stringify(marcas)),ediciones:JSON.parse(JSON.stringify(ediciones)),
  t:Date.now(),fecha:d.fecha,resumen:d.resumen};try{localStorage.setItem(CLAVE+"_env",JSON.stringify(enviado))}catch(e){}}
async function conectarGitHub(){
  if(!GH){alert("Solo funciona desde la página publicada en GitHub Pages.");return;}
  if(nube.token){if(confirm("¿Dejar de sincronizar con GitHub en este dispositivo? (Lo ya subido queda en GitHub.)")){
    nube.token="";nube.estado="off";try{localStorage.removeItem("fresas_token")}catch(e){}pintarEstado();}return;}
  const t=(prompt("Pega tu token de GitHub (fine-grained, solo el repositorio "+GH.repo+", permiso «Contents: Read and write»). Se guarda solo en este dispositivo.")||"").trim();
  if(!t)return;nube.token=t;try{localStorage.setItem("fresas_token",t)}catch(e){}
  hoja(false);aviso("Conectando con GitHub…");
  if(await sincronizar())aviso("✓ Sincronizado con GitHub");
  else{alert("No se pudo conectar:\n\n"+nube.err+".\n\nCorrige el token en GitHub (o crea uno nuevo) y vuelve a intentarlo.");nube.token="";nube.estado="off";try{localStorage.removeItem("fresas_token")}catch(e){}pintarEstado();}}
window.addEventListener("online",()=>{if(nube.estado=="error")sincronizar();});
document.addEventListener("visibilitychange",()=>{if(document.visibilityState=="visible")programarSubida(300);else if(nubeActiva()&&sinEnviar())sincronizar();});
if(nubeActiva())programarSubida(200);
function importar(input){
  const f=input.files&&input.files[0];input.value="";if(!f)return;
  f.text().then(t=>{let r;try{r=JSON.parse(t)}catch(e){alert("El archivo no es una revisión válida.");return;}
    if(r.lote&&r.lote!==LOTE){alert(`Ese archivo es de «${r.lote}», y este es «${LOTE}». Ábrelo en su lote.`);return;}
    const n=fusionarRevision(r,false);guardar("");hoja(false);pintar();
    aviso(n?`✓ Importado: ${n} cambios de otro dispositivo`:"No había nada nuevo en ese archivo");});}
/* al abrir: trae lo que ya se aplicó en la PC y se publicó (feedback/revision_<lote>.json) */
fetch(`../../feedback/revision_${LOTE}.json?t=${Date.now()}`,{cache:"no-store"}).then(r=>r.ok?r.json():null).then(r=>{
  const n=fusionarRevision(r,true);if(n){guardar("");pintar();aviso(`✓ Sincronizado con la PC: ${n} cambios`);}}).catch(()=>{});
function descargar(){
  const filas=["archivo,estado,editada"].concat(DATOS.map(d=>`"${d.orig}",${{ok:"bien",mal:"mal",ev:"evaluar"}[marcas[d.orig]]||"sin_revisar"},${d.orig in ediciones?"si":"no"}`));
  bajar(`resultado_${LOTE}.csv`,filas.join("\n"),"text/csv");
}
pintar();
</script></body></html>
"""


def titulo_de(nombre):
    """lote_01 -> 'Lote 1', cambios_03 -> 'Cambios 3'."""
    pre, _, num = nombre.rpartition("_")
    return f"{pre.replace('_', ' ').capitalize()} {int(num)}" if pre and num.isdigit() else nombre


def prefijo_de(nombre):
    pre, _, num = nombre.rpartition("_")
    return pre if pre and num.isdigit() else nombre


def escribir_html(dir_lote, nombres, i, datos):
    """nombres: lotes del mismo grupo (prefijo), para las flechas anterior/siguiente."""
    nombre = nombres[i]
    nav = ""
    if i > 0:
        nav += f'<a class="fila" href="../{nombres[i - 1]}/index.html"><span>‹ {html.escape(titulo_de(nombres[i - 1]))}</span></a>'
    if i < len(nombres) - 1:
        nav += f'<a class="fila" href="../{nombres[i + 1]}/index.html"><span>{html.escape(titulo_de(nombres[i + 1]))} ›</span></a>'
    nav += '<a class="fila" href="../index.html"><span>Todos los lotes</span></a>'
    titulo = titulo_de(nombre)
    pagina = (PLANTILLA.replace("__TITULO__", html.escape(titulo))
              .replace("__NAVEGACION__", nav).replace("__LOTE__", json.dumps(nombre))
              .replace("__VERSION__", datetime.datetime.now().astimezone().strftime("%d/%m %H:%M"))
              .replace("__DATOS__", json.dumps(datos, ensure_ascii=False)))
    (dir_lote / "index.html").write_text(pagina, encoding="utf-8")


INDICE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0e0e10"><title>Revisión de fresas</title>
<style>
body{margin:0;font:15px/1.4 system-ui,-apple-system,sans-serif;background:#0e0e10;color:#ececec;padding:24px 16px}
h1{font-size:22px;font-weight:600;margin:8px 4px 20px}
a{display:flex;align-items:center;gap:14px;padding:14px 4px;border-bottom:1px solid #2a2a2e;color:inherit;text-decoration:none}
a b{font-weight:500;width:96px}
h2{font-size:13px;font-weight:600;color:#8a8a90;text-transform:uppercase;letter-spacing:.04em;margin:26px 4px 4px}
.bar{flex:1;height:4px;background:#2a2a2e;border-radius:2px;display:flex;overflow:hidden}
.bar i{display:block;height:100%}
.n{color:#8a8a90;font-size:13px;width:74px;text-align:right;line-height:1.25}
.n em{font-style:normal;color:#ffd60a;font-size:11px}.n em.pc{color:#34c759}
p{color:#8a8a90;font-size:13px;margin:18px 4px}
</style></head><body><h1>Revisión de fresas</h1><div id="l"></div>
<p>Lo que revisas se guarda solo en este celular y en este navegador. Usa «Enviar» en cada lote para pasarlo a la PC.</p>
<p style="font-size:11px">Versión de la página: __VERSION__</p>
<script>
const LOTES=__LOTES__;let APL={};
function pintarIndice(){
let grupo=null;
document.getElementById("l").innerHTML=LOTES.map(([n,t,titulo,g])=>{const cab=g!==grupo&&LOTES.some(x=>x[3]!==g)?`<h2>${g=="lote"?"En orden":g.replace(/_/g," ")}</h2>`:"";grupo=g;let m={},ed={},env={};try{m=JSON.parse(localStorage.getItem("fresas_"+n)||"{}");ed=JSON.parse(localStorage.getItem("fresas_"+n+"_ed")||"{}");env=JSON.parse(localStorage.getItem("fresas_"+n+"_env")||"{}")}catch(e){}
  const pend=JSON.stringify(m)!==JSON.stringify(env.marcas||{})||JSON.stringify(ed)!==JSON.stringify(env.ediciones||{});
  const a=APL[n],enPC=!pend&&env.fecha&&a&&a.envio_fecha>=env.fecha;
  const v=Object.values(m),ok=v.filter(x=>x=="ok").length,mal=v.filter(x=>x=="mal").length,ev=v.filter(x=>x=="ev").length;
  return cab+`<a href="${n}/index.html"><b>${titulo}</b><span class="bar"><i style="width:${ok/t*100}%;background:#34c759"></i><i style="width:${ev/t*100}%;background:#ffd60a"></i><i style="width:${mal/t*100}%;background:#ff453a"></i></span><span class="n">${ok+mal+ev==t?"✓":ok+mal+ev+"/"+t}${pend?'<br><em>sin enviar</em>':enPC?'<br><em class="pc">✓ en PC</em>':env.fecha?'<br><em>falta aplicar</em>':""}</span></a>`}).join("");
}
pintarIndice();
fetch("aplicados.json?t="+Date.now(),{cache:"no-store"}).then(r=>r.ok?r.json():{}).then(j=>{APL=j||{};pintarIndice();}).catch(()=>{});
</script></body></html>
"""


def leer_datos(dir_lote):
    texto = (dir_lote / "index.html").read_text(encoding="utf-8")
    inicio = texto.index("const DATOS=") + len("const DATOS=")
    return json.loads(texto[inicio:texto.index(";\n", inicio)])


def rehacer_html(salida):
    """Regenera los index.html de los lotes existentes con la plantilla actual."""
    dirs = sorted(d for d in salida.iterdir()
                  if d.is_dir() and (d / "index.html").exists() and (d / "img").is_dir())
    grupos = {}
    for d in dirs:
        grupos.setdefault(prefijo_de(d.name), []).append(d)
    orden = sorted(grupos, key=lambda g: (g != "lote", g))  # «lote» primero
    lotes = []
    for g in orden:
        nombres = [d.name for d in grupos[g]]
        for i, d in enumerate(grupos[g]):
            datos = leer_datos(d)
            for foto in datos:  # lotes generados antes de guardar el índice de cada caja
                for k, f in enumerate(foto["figs"]):
                    f.setdefault("i", k)
            escribir_html(d, nombres, i, datos)
            lotes.append([d.name, len(datos), titulo_de(d.name), g])
    (salida / "index.html").write_text(INDICE.replace("__LOTES__", json.dumps(lotes, ensure_ascii=False))
                                       .replace("__VERSION__", datetime.datetime.now().astimezone().strftime("%d/%m %H:%M")),
                                       encoding="utf-8")
    print(f"Páginas actualizadas: {', '.join(l[0] for l in lotes)}")


def listar_imagenes(origen):
    """Fotos a revisar. Si la carpeta tiene fotos en su raíz, solo esas (así no entran copias
    como yolo/images/); si no, se busca en subcarpetas, saltando las de exportación YOLO."""
    imgs = sorted(p for p in origen.iterdir() if es_imagen(p))
    if not imgs:
        imgs = sorted(p for p in origen.rglob("*")
                      if es_imagen(p) and "yolo" not in (x.lower() for x in p.relative_to(origen).parts))
    return imgs


def leer_lista(archivo):
    """Nombres de imagen desde un .txt (uno por línea) o un .csv (columna «imagen» o la primera)."""
    import csv
    ruta = Path(archivo)
    texto = ruta.read_text(encoding="utf-8-sig")
    if ruta.suffix.lower() == ".csv":
        filas = list(csv.reader(texto.splitlines()))
        if not filas:
            return []
        cab = [c.strip().lower() for c in filas[0]]
        col = cab.index("imagen") if "imagen" in cab else 0
        valores = [f[col] for f in filas[1:] if len(f) > col]
    else:
        valores = texto.splitlines()
    vistos, nombres = set(), []
    for v in valores:
        v = Path(v.strip().replace("\\", "/")).name
        if v and v not in vistos:
            vistos.add(v)
            nombres.append(v)
    return nombres


def armar_lotes(raiz, carpeta, salida, tam, max_lado, solo=None, lista=None, prefijo="lote", criterios=None):
    origen = (raiz / carpeta) if carpeta else raiz
    if not origen.is_dir():
        sys.exit(f"No existe la carpeta: {origen}")
    imgs = listar_imagenes(origen)
    if not imgs:
        sys.exit(f"No hay imágenes en {origen}")
    print(f"Carpeta usada: {origen}\nImágenes encontradas: {len(imgs)}")
    if lista:
        por_nombre = {p.name: p for p in imgs}
        por_nombre.update({p.stem: p for p in imgs if p.stem not in por_nombre})
        pedidos = leer_lista(lista)
        imgs = [por_nombre[n] for n in pedidos if n in por_nombre]
        faltan = [n for n in pedidos if n not in por_nombre]
        print(f"Lista {lista}: {len(pedidos)} nombres, {len(imgs)} encontrados"
              + (f", {len(faltan)} no encontrados (ej. {faltan[0]})" if faltan else ""))
        if not imgs:
            sys.exit("Ninguna imagen de la lista está en la carpeta.")
    salida.mkdir(parents=True, exist_ok=True)
    lotes = [imgs[i:i + tam] for i in range(0, len(imgs), tam)]
    nombres = [f"{prefijo}_{i + 1:02d}" for i in range(len(lotes))]
    print(f"{len(lotes)} lotes de hasta {tam} fotos")
    if criterios:
        print("Usando tus criterios de feedback/criterios.json para sugerir correcciones.")
    for i, (nombre, grupo) in enumerate(zip(nombres, lotes)):
        if solo and (i + 1) not in solo:
            continue
        dir_lote = salida / nombre
        (dir_lote / "img").mkdir(parents=True, exist_ok=True)
        datos = []
        for k, img in enumerate(grupo):
            archivo = copiar_imagen(img, dir_lote / "img" / f"{k + 1:03d}", max_lado)
            figs = leer_anotacion(img)
            if criterios:
                sugerir(figs, criterios)
            datos.append({"orig": img.relative_to(origen).as_posix(),
                          "img": f"img/{archivo}", "figs": figs})
        escribir_html(dir_lote, nombres, i, datos)
        print(f"  {nombre}: {len(grupo)} imágenes")
    rehacer_html(salida)  # índice y flechas ◀ ▶ solo con los lotes que existen
    print(f"\nListo. Abre {salida / 'index.html'}")


def clase_segun(m, criterios):
    """Estado que corresponde a una caja según los umbrales aprendidos (None si faltan medidas)."""
    k = 0
    for frontera in criterios["fronteras"]:
        if frontera is None:
            return None
        v = m.get(frontera["medida"])
        if v is None:
            return None
        if v >= frontera["umbral"]:
            k += 1
        else:
            break
    return ORDEN[k]


def sugerir(figs, criterios):
    for f in figs:
        m = f.get("m")
        if not m or f["label"] not in ORDEN:
            continue
        c = clase_segun(m, criterios)
        if c and c != f["label"]:
            f["sug"] = c


def mejor_umbral(valores_bajo, valores_alto, referencia):
    """Umbral que mejor separa dos estados consecutivos (menos errores). Entre los que empatan
    se elige el más cercano al del agente (referencia), para no inventar diferencias donde
    los datos no las muestran. Devuelve (umbral, errores)."""
    todos = sorted(set(valores_bajo) | set(valores_alto))
    candidatos = [(a + b) / 2 for a, b in zip(todos, todos[1:])] + todos + [referencia]
    def errores(t):
        return sum(v >= t for v in valores_bajo) + sum(v < t for v in valores_alto)
    minimo = min(errores(t) for t in candidatos)
    iguales = [t for t in candidatos if errores(t) == minimo]
    # dentro del tramo sin cambio de errores, acercarse todo lo posible a la referencia
    t = min(iguales, key=lambda t: abs(t - referencia))
    lo = max([v for v in todos if v < t], default=None)
    hi = min([v for v in todos if v >= t], default=None)
    if lo is not None and hi is not None and lo < referencia <= hi:
        t = referencia
    elif hi is not None and lo is not None and referencia > hi:
        t = hi
    elif lo is not None and referencia <= lo:
        t = lo + 1e-6 if errores(lo + 1e-6) == minimo else t
    return t, errores(t)


def aprender_criterios(origen, archivos, destino):
    """Aprende tus umbrales a partir de las fotos que ya revisaste y aplicaste."""
    cajas, cambios = [], {}
    for archivo in archivos:
        datos = json.loads(Path(archivo).read_text(encoding="utf-8"))
        for orig in datos.get("marcas", {}):
            j = (origen / orig).with_suffix(".json")
            if not j.exists():
                continue
            d = json.loads(j.read_text(encoding="utf-8"))
            for sh in d.get("shapes", []):
                at = sh.get("attributes") or {}
                if sh.get("label") in ORDEN and isinstance(at, dict):
                    cajas.append((sh["label"], {k: at[k] for k in METRICAS if isinstance(at.get(k), (int, float))}))
        for orig, figs in datos.get("correcciones", {}).items():
            antes = datos.get("originales", {}).get(orig)
            if not antes:
                continue
            for f in asignar_indices(figs, antes) if not any("i" in x for x in figs) else figs:
                o = next((x for x in antes if x.get("i") == f.get("i")), None) if "i" in f else None
                if o and o["label"] != f["label"]:
                    cambios[(o["label"], f["label"])] = cambios.get((o["label"], f["label"]), 0) + 1
    con_medidas = [(l, m) for l, m in cajas if m]
    print(f"\nCajas revisadas por ti: {len(cajas)} ({len(con_medidas)} con medidas del agente)")
    if cambios:
        print("Etiquetas que cambiaste (antes -> tu corrección):")
        for (a, b), n in sorted(cambios.items(), key=lambda x: -x[1]):
            print(f"  {a:>17} -> {b:<17} {n}")
    if not con_medidas:
        print("Las anotaciones no tienen 'attributes' con color_pct: no se pueden aprender umbrales.")
        return None
    fronteras = []
    print("\nFrontera                            agente    tú (medida)                 errores")
    for k in range(len(ORDEN) - 1):
        bajo = [m for l, m in con_medidas if l == ORDEN[k]]
        alto = [m for l, m in con_medidas if l == ORDEN[k + 1]]
        mejor = None
        for medida in METRICAS:
            vb = [m[medida] for m in bajo if medida in m]
            va = [m[medida] for m in alto if medida in m]
            if len(vb) < 5 or len(va) < 5:
                continue
            t, err = mejor_umbral(vb, va, UMBRALES_AGENTE[k] if medida == "color_pct" else
                                  (sum(vb) / len(vb) + sum(va) / len(va)) / 2)
            if t is not None and (mejor is None or err / (len(vb) + len(va)) < mejor["error"]):
                mejor = {"medida": medida, "umbral": round(t, 1), "error": round(err / (len(vb) + len(va)), 3),
                         "n": len(vb) + len(va)}
        fronteras.append(mejor)
        nombre = f"{ORDEN[k]} | {ORDEN[k + 1]}"
        if mejor:
            print(f"  {nombre:<34} {UMBRALES_AGENTE[k]:>5}%   {mejor['umbral']:>5} ({mejor['medida']:<16}) "
                  f"{mejor['error'] * 100:5.1f}% de {mejor['n']}")
        else:
            print(f"  {nombre:<34} {UMBRALES_AGENTE[k]:>5}%   (pocos ejemplos, sin sugerencias)")
    criterios = {"fronteras": fronteras, "orden": ORDEN, "umbrales_agente": UMBRALES_AGENTE,
                 "cajas": len(con_medidas), "cambios": {f"{a} -> {b}": n for (a, b), n in cambios.items()},
                 "fuentes": [Path(a).name for a in archivos],
                 "fecha": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(criterios, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGuardado en {destino}. Los próximos lotes que generes marcarán las cajas donde tu criterio")
    print("no coincide con la etiqueta actual (filtro «Con sugerencia» en la página).")
    return criterios


def nombres_yolo(origen):
    """Lista de clases YOLO (classes.txt) buscando hacia arriba desde el origen."""
    for d in [origen, *origen.parents]:
        c = d / "classes.txt"
        if c.exists():
            return [l.strip() for l in c.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


VERSION_LABELME = "4.0.0-beta.7"  # la de los JSON del dataset (AnyLabeling)
META_NUEVA = {"score": None, "group_id": None, "description": "", "difficult": False,
              "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": []}


def a_caja(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def misma_caja(a, b, tol=1e-4):
    return all(abs(u - v) <= tol for u, v in zip(a_caja(a), a_caja(b)))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a_caja(a)
    bx1, by1, bx2, by2 = a_caja(b)
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0


def asignar_indices(figs, originales, minimo=0.3):
    """Da a cada caja editada el índice de la caja original con la que más se superpone
    (emparejamiento voraz por IoU ≥ minimo). Las que no se parecen a ninguna quedan como nuevas."""
    pares = sorted(((iou(o["pts"], f["pts"]), o.get("i", k), j)
                    for k, o in enumerate(originales) for j, f in enumerate(figs)), reverse=True)
    usados_o, asignado = set(), {}
    for v, i, j in pares:
        if v < minimo or i in usados_o or j in asignado:
            continue
        usados_o.add(i)
        asignado[j] = i
    return [dict(f, i=asignado[j]) if j in asignado else {k: v for k, v in f.items() if k != "i"}
            for j, f in enumerate(figs)]


def fusionar_json(d, figs, originales):
    """Aplica las ediciones del celular sobre las cajas del JSON sin perder sus metadatos.

    Con «originales» (formato 2) es una fusión a tres bandas por índice de caja: solo se
    escribe lo que la persona cambió en el celular; si no tocó la etiqueta o la posición,
    se respeta lo que tenga ahora el JSON (que pudo actualizarse después de generar el lote).
    Devuelve (shapes nuevas, mapa viejo->nuevo, índices nuevos a recalcular).
    """
    w, h = d["imageWidth"], d["imageHeight"]
    previas = d.get("shapes", [])
    if originales is not None and figs and not any("i" in f for f in figs):
        # editada con una versión anterior de la página (sin índice por caja): se empareja cada
        # caja con la original que más se le superpone, no por su orden en la lista
        figs = asignar_indices(figs, originales)
    orig_por_i = {f.get("i", k): f for k, f in enumerate(originales)} if originales is not None else None
    nuevas, mapa, recalcular = [], {k: None for k in range(len(previas))}, []
    if orig_por_i is not None:
        # una caja «nueva» que ya está en el JSON (p. ej. al aplicar dos veces el mismo envío) se reconoce
        # por su posición y conserva sus metadatos en vez de borrarse y volver a crearse
        usados = {f.get("i") for f in figs if f.get("i") is not None}
        sobrantes = [k for k in range(len(previas)) if k not in usados and k not in orig_por_i
                     and len(previas[k].get("points", [])) >= 2]
        figs = [dict(f) for f in figs]
        for f in figs:
            if f.get("i") is not None or not sobrantes:
                continue
            def norm(sh):
                return [[x / w, y / h] for x, y in sh["points"]]
            k = max(sobrantes, key=lambda k: iou(f["pts"], norm(previas[k])))
            if iou(f["pts"], norm(previas[k])) >= 0.9:
                f["i"] = k
                sobrantes.remove(k)
    for k, f in enumerate(figs):
        i = f.get("i") if orig_por_i is not None else (k if k < len(previas) else None)
        x1, y1, x2, y2 = a_caja(f["pts"])
        if i is not None and i < len(previas):
            base = json.loads(json.dumps(previas[i]))
            o = orig_por_i.get(i) if orig_por_i is not None else None
            if o is None or f["label"] != o["label"]:
                base["label"] = f["label"]
            # un desplazamiento mínimo (IoU ≥ 0,97) se toma como toque accidental y se ignora
            if o is None or not (misma_caja(f["pts"], o["pts"]) or iou(f["pts"], o["pts"]) >= 0.97):
                antes = base.get("points")
                base["points"] = [[x1 * w, y1 * h], [x2 * w, y2 * h]]
                base["shape_type"] = "rectangle"
                if antes != base["points"]:
                    base["attributes"] = {}  # sus métricas ya no corresponden a la caja
                    recalcular.append(len(nuevas))
            mapa[i] = len(nuevas)
        else:
            base = json.loads(json.dumps(META_NUEVA))
            base["label"] = f["label"]
            base["points"] = [[x1 * w, y1 * h], [x2 * w, y2 * h]]
            recalcular.append(len(nuevas))
        nuevas.append(base)
    return nuevas, mapa, recalcular


def exportar_yolo(j, d, origen):
    """Regenera yolo/labels/<foto>.txt desde el JSON, si la carpeta tiene export YOLO."""
    dir_yolo = origen / "yolo"
    clases_txt = dir_yolo / "classes.txt"
    if not clases_txt.exists():
        return False
    clases = [l.strip() for l in clases_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    W, H = d["imageWidth"], d["imageHeight"]
    lineas = []
    for sh in d["shapes"]:
        if sh["label"] not in clases:
            sys.exit(f"'{sh['label']}' no está en {clases_txt}")
        (x1, y1), (x2, y2) = [min(p[0] for p in sh["points"]), min(p[1] for p in sh["points"])], \
                             [max(p[0] for p in sh["points"]), max(p[1] for p in sh["points"])]
        lineas.append(f"{clases.index(sh['label'])} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                      f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    destino = dir_yolo / "labels" / (j.stem + ".txt")
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        respaldo(destino)
    destino.write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
    return True


def aplicar_correcciones(origen, archivo, salida=Path("revision")):
    """Escribe en las anotaciones originales lo revisado desde el celular."""
    datos = json.loads(Path(archivo).read_text(encoding="utf-8"))
    correcciones = datos.get("correcciones", {} if "marcas" in datos else datos)
    originales = datos.get("originales", {}) if datos.get("formato", 1) >= 2 else None
    lote = datos.get("lote", "lote")
    if datos.get("marcas"):
        csv = Path(archivo).with_name(f"resultado_{lote}.csv")
        with open(csv, "w", encoding="utf-8-sig") as f:
            f.write("archivo,estado,editada\n")
            for orig, est in sorted(datos["marcas"].items()):
                f.write(f'"{orig}",{est},{"si" if orig in correcciones else "no"}\n')
        cuenta = {e: list(datos["marcas"].values()).count(e) for e in ("bien", "mal", "evaluar")}
        print(f"Marcas: {cuenta['bien']} bien, {cuenta['mal']} mal, {cuenta['evaluar']} evaluar -> {csv}")
    if correcciones and originales is None:
        print("Aviso: archivo de una versión anterior de la página; las cajas se emparejan por posición.")
    clases = None
    hechas, yolo, mapas = 0, 0, {}
    for orig, figs in correcciones.items():
        img = origen / orig
        if not img.exists():
            print(f"  ¡No existe {img}! (¿--carpeta correcta?)")
            continue
        j, t = img.with_suffix(".json"), img.with_suffix(".txt")
        if t.exists() and not j.exists():  # dataset solo YOLO
            if clases is None:
                clases = nombres_yolo(origen)
            lineas = []
            for f in figs:
                label, (x1, y1, x2, y2) = f["label"], a_caja(f["pts"])
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
                d = {"version": VERSION_LABELME, "flags": {}, "shapes": [], "imagePath": img.name,
                     "imageData": None, "imageHeight": h, "imageWidth": w}
            d["shapes"], mapa, recalcular = fusionar_json(
                d, figs, originales.get(orig) if originales is not None else None)
            j.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            cambio_estructura = any(v != k for k, v in mapa.items()) or len(d["shapes"]) != len(mapa)
            if recalcular or cambio_estructura:
                mapas[orig] = {"mapa": {str(k): v for k, v in mapa.items()}, "recalcular": recalcular}
            yolo += exportar_yolo(j, d, origen)
        hechas += 1
    if hechas:
        print(f"Cajas corregidas en {hechas} imágenes. Los archivos anteriores quedaron como .bak")
        if yolo:
            print(f"Export YOLO actualizado: {yolo} archivos en {origen / 'yolo' / 'labels'}")
    else:
        print("No hay cajas editadas en este archivo.")
    if mapas:
        destino = Path(archivo).with_name(f"mapa_indices_{lote}.json")
        destino.write_text(json.dumps(mapas, ensure_ascii=False, indent=1), encoding="utf-8")
        n = sum(len(m["recalcular"]) for m in mapas.values())
        print(f"{destino}: {len(mapas)} imágenes con cajas movidas, nuevas o borradas "
              f"({n} cajas sin métricas: hay que recalcular sus atributos)")
    registrar_aplicado(salida, lote, datos, hechas)


def registrar_aplicado(salida, lote, datos, cajas):
    """Anota en revision/aplicados.json que este envío ya se aplicó; la página lo muestra
    como «✓ Aplicado en PC» después del git push."""
    if not salida.is_dir():
        return
    ruta = salida / "aplicados.json"
    try:
        registro = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        registro = {}
    marcas = list(datos.get("marcas", {}).values())
    registro[lote] = {"envio_fecha": datos.get("fecha", ""),
                      "aplicado_fecha": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                      "fotos": len(marcas), "bien": marcas.count("bien"), "mal": marcas.count("mal"),
                      "evaluar": marcas.count("evaluar"), "cajas": cajas}
    ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Confirmación guardada en {ruta}. Súbela para verla en el celular:\n"
          f'    git add -A; git commit -m "Aplicado {lote}"; git push')


def aplicar_github(origen, salida):
    """Trae de la rama «resultados» (donde sincroniza la página) cada <lote>.json y lo aplica."""
    import subprocess
    repo = Path(__file__).resolve().parent

    def git(*args):
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True)
        if r.returncode:
            raise RuntimeError(r.stderr.decode("utf-8", "replace").strip())
        return r.stdout.decode("utf-8")
    try:
        git("fetch", "-q", "origin", "resultados")
    except RuntimeError:
        sys.exit("Todavía no hay nada sincronizado en GitHub (no existe la rama «resultados»).")
    try:
        registro = json.loads((salida / "aplicados.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        registro = {}
    nombres = [n for n in git("ls-tree", "--name-only", "origin/resultados").split() if n.endswith(".json")]
    destino = repo / "feedback"
    destino.mkdir(exist_ok=True)
    aplicados = 0
    for n in sorted(nombres):
        datos = json.loads(git("show", f"origin/resultados:{n}"))
        lote = datos.get("lote", Path(n).stem)
        if datos.get("fecha", "") <= registro.get(lote, {}).get("envio_fecha", ""):
            print(f"{lote}: sin cambios desde la última vez")
            continue
        archivo = destino / f"revision_{lote}.json"
        archivo.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n== {lote} ==")
        aplicar_correcciones(origen, archivo, salida)
        aplicados += 1
    if aplicados:
        print('\nListo. Sube la confirmación: git add -A; git commit -m "Aplicado desde GitHub"; git push')


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
    ap.add_argument("--lista", metavar="ARCHIVO",
                    help="armar lotes solo con estas imágenes (.txt una por línea o .csv con columna «imagen»)")
    ap.add_argument("--prefijo", default=None,
                    help="nombre de los lotes (por defecto «lote», o el nombre del archivo de --lista)")
    ap.add_argument("--aplicar", metavar="JSON",
                    help="aplicar revision_lote_XX.json enviado desde el celular, o «github» para traer lo sincronizado")
    ap.add_argument("--feedback", metavar="JSON", nargs="+",
                    help="aprender tus criterios de revision_lote_XX.json ya aplicados (escribe feedback/criterios.json)")
    ap.add_argument("--max-lado", type=int, default=1280, help="tamaño máx. de las copias")
    a = ap.parse_args()
    if a.rehacer_html:
        return rehacer_html(a.salida)
    if a.raiz is None or not a.raiz.is_dir():
        sys.exit(f"No existe: {a.raiz}")
    ruta_criterios = Path(__file__).resolve().parent / "feedback" / "criterios.json"
    if a.feedback:
        aprender_criterios(a.raiz / (a.carpeta or "."), a.feedback, ruta_criterios)
        return
    if a.aplicar == "github":
        aplicar_github(a.raiz / (a.carpeta or "."), a.salida)
    elif a.aplicar:
        aplicar_correcciones(a.raiz / (a.carpeta or "."), a.aplicar, a.salida)
    elif a.carpeta is None:
        listar_carpetas(a.raiz)
    else:
        prefijo = a.prefijo or (Path(a.lista).stem if a.lista else "lote")
        prefijo = "".join(c if c.isalnum() else "_" for c in prefijo).strip("_").lower() or "lote"
        criterios = None
        if ruta_criterios.exists():
            criterios = json.loads(ruta_criterios.read_text(encoding="utf-8"))
        armar_lotes(a.raiz, None if a.carpeta == "." else a.carpeta, a.salida, a.lote, a.max_lado,
                    a.solo, a.lista, prefijo, criterios)


if __name__ == "__main__":
    main()
