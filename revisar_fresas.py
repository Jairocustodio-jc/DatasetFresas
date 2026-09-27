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
                    figs.append({"label": s.get("label", ""), "i": len(figs),
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
#ver .img{flex:1;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:0;touch-action:none;user-select:none;-webkit-user-select:none}
#lienzo{position:relative;transform-origin:0 0;--s:1}
#lienzo.anim{transition:transform .22s ease-out}
#ver img{max-width:100vw;max-height:calc(100vh - 140px);display:block;user-select:none;-webkit-user-drag:none;pointer-events:none}
#ver.editando img{max-height:calc(100vh - 190px)}
#ver svg,#ve{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#vs polygon{stroke-width:calc(2px / var(--s))}
.e{position:absolute;transform:translateY(-100%);color:#000;font-size:10px;font-weight:600;padding:0 4px;border-radius:3px 3px 0 0;white-space:nowrap;pointer-events:none;opacity:.9}
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
#toast{position:fixed;left:50%;top:calc(58px + env(safe-area-inset-top));transform:translate(-50%,-8px);background:#fff;color:#000;font-size:13px;font-weight:600;padding:6px 14px;border-radius:14px;opacity:0;transition:.2s;z-index:9;pointer-events:none;white-space:nowrap}
#toast.on{opacity:1;transform:translate(-50%,0)}#toast.err{background:var(--mal)}
.chipest{font-size:12px;padding:3px 10px;border-radius:12px;background:var(--sup);color:var(--mu);white-space:nowrap}
.chipest.ok{background:var(--ok);color:#000}.chipest.mal{background:var(--mal);color:#000}.chipest.ev{background:var(--ev);color:#000}
.vacio{grid-column:1/-1;text-align:center;color:var(--mu);padding:60px 20px}
</style></head><body>
<header>
 <div class="top"><a class="ic" href="../index.html">‹</a><div class="t">__TITULO__</div><span id="cont" class="mu"></span><button class="ic" onclick="hoja(true)">⋯</button></div>
 <div class="prog"><i id="pok"></i><i id="pev"></i><i id="pmal"></i></div>
 <div class="estado"><span class="txt" id="guard"></span><button class="enviar" id="benv" onclick="enviar()">Enviar</button></div>
</header>
<div id="toast"></div>
<main class="grid" id="g"></main>
<button class="fab" id="fab" onclick="seguir()">Revisar</button>
<div id="hoja" onclick="if(event.target==this)hoja(false)"><div class="panel">
 <div id="ley"></div><div class="sep"></div>
 <button class="fila" onclick="mini=!mini;guardarPref();pintar();hoja(true)"><span>Cajas en miniaturas</span><span id="swmini" class="sw"></span></button>
 <button class="fila" onclick="cambiarFiltro()"><span>Mostrar</span><span class="mu" id="lfiltro"></span></button>
 <button class="fila" onclick="enviar()"><span>Enviar resultados a la PC</span><span class="mu" id="ned"></span></button>
 <button class="fila" onclick="descargar()"><span>Descargar tabla</span><span class="mu">CSV</span></button>
 <div class="sep"></div>__NAVEGACION__
</div></div>
<div id="ver">
 <div class="vtop"><button class="ic" onclick="cerrar()">✕</button><span class="pos" id="vpos"></span><span class="chipest" id="vest"></span><span class="sp"></span>
  <button class="ic" id="bzoom" onclick="cambiarZoom()">1×</button><button class="ic" id="bed" onclick="alternarEdicion()">✎</button></div>
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
let filtro="todas",orden=[],marcas={},ediciones={},mini=false,mostrarAnot=true,actual=-1,editando=false,sel=-1,etiquetaNueva=null,arrastre=null;
try{marcas=JSON.parse(localStorage.getItem(CLAVE)||"{}");ediciones=JSON.parse(localStorage.getItem(CLAVE+"_ed")||"{}");mini=localStorage.getItem("fresas_mini")=="1"}catch(e){}
let enviado={marcas:{},ediciones:{},t:0};
try{enviado=JSON.parse(localStorage.getItem(CLAVE+"_env")||"null")||enviado}catch(e){}
try{navigator.storage&&navigator.storage.persist&&navigator.storage.persist()}catch(e){}
let timerToast=0;
function aviso(txt,err){const t=$("toast");t.textContent=txt;t.classList.toggle("err",!!err);t.classList.add("on");
  clearTimeout(timerToast);timerToast=setTimeout(()=>t.classList.remove("on"),err?4000:1300);}
function guardar(txt){
  try{localStorage.setItem(CLAVE,JSON.stringify(marcas));localStorage.setItem(CLAVE+"_ed",JSON.stringify(ediciones));
    if(localStorage.getItem(CLAVE)!==JSON.stringify(marcas))throw 0;
    if(txt)aviso(txt);return true;}
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
  if(p)t+=` · <span class="pend">${p} sin enviar</span>`;else if(enviado.t)t+=` · enviado ${hace(enviado.t)}`;
  $("guard").innerHTML=t;$("benv").style.display=h||e?"":"none";$("benv").classList.toggle("urg",p>=20);
  $("ned").textContent=p?p+" sin enviar":enviado.t?"todo enviado":"";}
function guardarPref(){try{localStorage.setItem("fresas_mini",mini?"1":"0")}catch(e){}}
const COLORES={"unripe":"#34c759","early-pink":"#ff8fd8","commercial-basic":"#ff9f0a","commercial-high":"#ff453a","overripe":"#bf5af2"};
const EXTRA=["#64d2ff","#ffd60a","#0a84ff","#ffffff"];
function color(l){if(!(l in COLORES))COLORES[l]=EXTRA[Object.keys(COLORES).length%EXTRA.length];return COLORES[l];}
DATOS.forEach(d=>d.figs.forEach(f=>color(f.label)));
const $=id=>document.getElementById(id);
function figsDe(d){return ediciones[d.orig]||d.figs;}
function caja(f){const xs=f.pts.map(p=>p[0]),ys=f.pts.map(p=>p[1]);return[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];}
function deCaja(label,[x1,y1,x2,y2]){return{label,pts:[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]};}
function svgDe(figs,conSel){
  return figs.map((f,i)=>`<polygon points="${f.pts.map(p=>p.join(",")).join(" ")}" fill="${conSel&&i==sel?"rgba(255,255,255,.12)":"none"}" stroke="${color(f.label)}" stroke-width="2" vector-effect="non-scaling-stroke"/>`).join("");
}
/* etiquetas y esquinas en píxeles de pantalla (fuera del zoom, siempre nítidas) */
function etqDe(figs,conSel){
  const r=$("vi").getBoundingClientRect(),v=document.querySelector("#ver .img").getBoundingClientRect();
  const X=x=>(r.left-v.left+x*r.width).toFixed(1)+"px",Y=y=>(r.top-v.top+y*r.height).toFixed(1)+"px";
  let h=figs.map(f=>{const[x,y]=caja(f);return `<span class="e" style="left:${X(x)};top:${Y(y)};background:${color(f.label)}">${f.label}</span>`}).join("");
  if(conSel&&sel>=0&&figs[sel]){const[x1,y1,x2,y2]=caja(figs[sel]);
    [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].forEach(([x,y])=>h+=`<span class="h" style="left:${X(x)};top:${Y(y)}"></span>`);}
  return h;
}
function pintarCapa(){if(actual<0)return;$("ve").innerHTML=mostrarAnot||editando?etqDe(figsDe(DATOS[actual]),editando):"";}
let rafCapa=0;
function seguirCapa(ms){cancelAnimationFrame(rafCapa);const fin=performance.now()+ms;
  const paso=()=>{pintarCapa();if(performance.now()<fin)rafCapa=requestAnimationFrame(paso);};paso();}
const FILTROS={todas:"Todas",ev:"A evaluar",mal:"Mal",ok:"Bien",pend:"Sin revisar"};
function pasaFiltro(d){const m=marcas[d.orig];return filtro=="todas"||(filtro=="pend"?!m:m==filtro);}
function cambiarFiltro(){const k=Object.keys(FILTROS);filtro=k[(k.indexOf(filtro)+1)%k.length];pintar();hoja(true);}
function hoja(on){$("hoja").classList.toggle("on",on);$("lfiltro").textContent=FILTROS[filtro];$("swmini").classList.toggle("on",mini);
  const n={};DATOS.forEach(d=>figsDe(d).forEach(f=>n[f.label]=(n[f.label]||0)+1));
  $("ley").innerHTML=Object.keys(n).sort().map(l=>`<span><i style="background:${color(l)}"></i>${l} ${n[l]}</span>`).join("");}
function pintar(){
  const g=$("g");g.innerHTML="";orden=[];
  DATOS.forEach((d,i)=>{
    if(!pasaFiltro(d))return;orden.push(i);
    const c=document.createElement("div");c.className="c "+(marcas[d.orig]||"");
    c.innerHTML=`<img loading="lazy" src="${d.img}">${mini?`<svg viewBox="0 0 1 1" preserveAspectRatio="none">${svgDe(figsDe(d))}</svg>`:""}<span class="d"></span>${d.orig in ediciones?'<span class="p">✎</span>':""}`;
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
  $("vs").innerHTML=mostrarAnot||editando?svgDe(f,editando):"";pintarCapa();
  $("vpos").textContent=`${orden.indexOf(actual)+1} / ${orden.length}`;
  $("bok").classList.toggle("on",m=="ok");$("bmal").classList.toggle("on",m=="mal");$("bev").classList.toggle("on",m=="ev");
  const ve=$("vest");ve.className="chipest "+(m||"");ve.textContent=(m?{ok:"Bien",mal:"Mal",ev:"Evaluar"}[m]:"Sin marcar")+(d.orig in ediciones?" · ✎":"");
  if(editando)pintarChips();}
function pintarChips(){
  const f=figsDe(DATOS[actual]),act=sel>=0&&f[sel]?f[sel].label:etiquetaNueva;
  $("chips").innerHTML=Object.keys(COLORES).map(l=>`<button class="chip ${l==act?"act":""}" onclick="ponerEtiqueta('${l}')"><i style="background:${color(l)}"></i>${l}</button>`).join("");
}
function abrir(i){actual=i;sel=-1;$("vi").onload=()=>aplicarVista(false);$("vi").src=DATOS[i].img;$("ver").style.display="flex";reiniciarZoom(false);pintarVisor();}
function cerrar(){if(editando)alternarEdicion();$("ver").style.display="none";pintar();}
function mover(k){const n=orden[orden.indexOf(actual)+k];if(n!==undefined)abrir(n);else cerrar();}
function marcar(v){const o=DATOS[actual].orig;marcas[o]=marcas[o]==v?"":v;if(!marcas[o]){delete marcas[o];guardar("Marca quitada");pintarVisor();return;}
  guardar("✓ Guardado: "+{ok:"Bien",mal:"Mal",ev:"Evaluar"}[v]);
  pintarVisor();setTimeout(()=>mover(1),150);}
function alternarEdicion(){editando=!editando;sel=-1;
  $("ver").classList.toggle("editando",editando);$("bed").classList.toggle("on",editando);pintarVisor();setTimeout(()=>aplicarVista(false),0);}
function editables(){const d=DATOS[actual];if(!ediciones[d.orig])ediciones[d.orig]=JSON.parse(JSON.stringify(d.figs));return ediciones[d.orig];}
function cambio(){const d=DATOS[actual];if(JSON.stringify(ediciones[d.orig])==JSON.stringify(d.figs))delete ediciones[d.orig];guardar("✓ Cajas guardadas");pintarVisor();}
function ponerEtiqueta(l){etiquetaNueva=l;if(sel>=0){editables()[sel].label=l;cambio();}else pintarChips();}
function nuevaCaja(){const f=editables(),l=etiquetaNueva||(f[0]&&f[0].label)||Object.keys(COLORES)[0];
  f.push(deCaja(l,[.42,.42,.58,.58]));sel=f.length-1;cambio();}
function borrarCaja(){if(sel<0)return;editables().splice(sel,1);sel=-1;cambio();}
function restaurar(){if(confirm("¿Volver a las cajas originales de esta foto?")){delete ediciones[DATOS[actual].orig];sel=-1;guardar("Cajas originales restauradas");pintarVisor();}}
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
function iniciarPinza(){if(gesto&&(gesto.tipo=="esq"||gesto.tipo=="mover"))cambio();
  const m=dosDedos();gesto={tipo:"pinza",d0:m.d,x0:m.x,y0:m.y,e0:esc,tx0:tx,ty0:ty};}
function iniciarUnDedo(ev){gesto={tipo:"pendiente",x0:ev.clientX,y0:ev.clientY,tx0:tx,ty0:ty,t0:Date.now()};
  if(!editando)return;
  const p=punto(ev),f=figsDe(DATOS[actual]),r=$("vi").getBoundingClientRect(),tol=22/r.width,tolY=22/r.height;
  if(sel>=0&&f[sel]){const[x1,y1,x2,y2]=caja(f[sel]);
    const esq=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]].findIndex(([x,y])=>Math.abs(p[0]-x)<tol&&Math.abs(p[1]-y)<tolY);
    if(esq>=0){gesto={tipo:"esq",esq,caja:[x1,y1,x2,y2]};return;}}
  let mejor=-1,area=9;
  f.forEach((fi,i)=>{const[x1,y1,x2,y2]=caja(fi);if(p[0]>=x1&&p[0]<=x2&&p[1]>=y1&&p[1]<=y2&&(x2-x1)*(y2-y1)<area){mejor=i;area=(x2-x1)*(y2-y1);}});
  if(mejor>=0){sel=mejor;gesto={tipo:"mover",inicio:p,caja:caja(f[sel])};pintarVisor();}
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
  if(gesto.tipo=="pendiente"&&Math.hypot(dx,dy)>8)gesto.tipo=esc>1.02?"pan":"deslizar";
  if(gesto.tipo=="pan"){tx=gesto.tx0+dx;ty=gesto.ty0+dy;aplicarVista(false);return;}
  if(gesto.tipo=="deslizar"){if(!editando){lienzo.classList.remove("anim");lienzo.style.transform=`translate(${gesto.tx0+dx}px,${gesto.ty0}px)`;pintarCapa();}return;}
  if(gesto.tipo=="esq"||gesto.tipo=="mover"){
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
    if(editando){if(sel>=0){sel=-1;pintarVisor();}}else{mostrarAnot=!mostrarAnot;pintarVisor();}},280);
}
function soltarDedo(ev){
  if(!dedos.has(ev.pointerId))return;dedos.delete(ev.pointerId);
  const g=gesto;if(!g)return;
  if(g.tipo=="pinza"){
    if(dedos.size==1){const[id,d]=[...dedos.entries()][0];gesto={tipo:"pan",x0:d.x,y0:d.y,tx0:tx,ty0:ty};}
    else gesto=null;
    if(esc<1.05)reiniciarZoom(true);else aplicarVista(true);return;}
  if(dedos.size>0)return;gesto=null;
  if(g.tipo=="esq"||g.tipo=="mover"){cambio();return;}
  if(g.tipo=="deslizar"){const dx=ev.clientX-g.x0,dy=ev.clientY-g.y0;
    if(!editando&&Math.abs(dx)>60&&Math.abs(dx)>Math.abs(dy)*1.5&&ev.type=="pointerup"){mover(dx<0?1:-1);}else aplicarVista(true);return;}
  if(g.tipo=="pendiente"&&ev.type=="pointerup")toque(ev);
}
vista.addEventListener("pointerup",soltarDedo);vista.addEventListener("pointercancel",soltarDedo);
function bajar(nombre,contenido,tipo,listo){
  const archivo=new File([contenido],nombre,{type:tipo});
  if(navigator.canShare&&navigator.canShare({files:[archivo]})){navigator.share({files:[archivo],title:nombre}).then(()=>listo&&listo()).catch(()=>{});return;}
  const a=document.createElement("a");a.href=URL.createObjectURL(archivo);a.download=nombre;a.click();listo&&listo();
}
function enviar(){
  if(!Object.keys(marcas).length&&!Object.keys(ediciones).length){aviso("Todavía no hay nada que enviar");return;}
  const est={ok:"bien",mal:"mal",ev:"evaluar"},m={};for(const k in marcas)m[k]=est[marcas[k]];
  const originales={};DATOS.forEach(d=>{if(d.orig in ediciones)originales[d.orig]=d.figs;});
  const datos={formato:2,lote:LOTE,fecha:new Date().toISOString(),marcas:m,correcciones:ediciones,originales};
  bajar(`revision_${LOTE}.json`,JSON.stringify(datos,null,1),"application/json",()=>{
    enviado={marcas:JSON.parse(JSON.stringify(marcas)),ediciones:JSON.parse(JSON.stringify(ediciones)),t:Date.now()};
    try{localStorage.setItem(CLAVE+"_env",JSON.stringify(enviado))}catch(e){}
    hoja(false);pintar();aviso("✓ Archivo listo: revision_"+LOTE+".json");});
}
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
.n em{font-style:normal;color:#ffd60a;font-size:11px}
p{color:#8a8a90;font-size:13px;margin:18px 4px}
</style></head><body><h1>Revisión de fresas</h1><div id="l"></div>
<p>Lo que revisas se guarda solo en este celular y en este navegador. Usa «Enviar» en cada lote para pasarlo a la PC.</p>
<script>
const LOTES=__LOTES__;
let grupo=null;
document.getElementById("l").innerHTML=LOTES.map(([n,t,titulo,g])=>{const cab=g!==grupo&&LOTES.some(x=>x[3]!==g)?`<h2>${g=="lote"?"En orden":g.replace(/_/g," ")}</h2>`:"";grupo=g;let m={},ed={},env={};try{m=JSON.parse(localStorage.getItem("fresas_"+n)||"{}");ed=JSON.parse(localStorage.getItem("fresas_"+n+"_ed")||"{}");env=JSON.parse(localStorage.getItem("fresas_"+n+"_env")||"{}")}catch(e){}
  const pend=JSON.stringify(m)!==JSON.stringify(env.marcas||{})||JSON.stringify(ed)!==JSON.stringify(env.ediciones||{});
  const v=Object.values(m),ok=v.filter(x=>x=="ok").length,mal=v.filter(x=>x=="mal").length,ev=v.filter(x=>x=="ev").length;
  return cab+`<a href="${n}/index.html"><b>${titulo}</b><span class="bar"><i style="width:${ok/t*100}%;background:#34c759"></i><i style="width:${ev/t*100}%;background:#ffd60a"></i><i style="width:${mal/t*100}%;background:#ff453a"></i></span><span class="n">${ok+mal+ev==t?"✓":ok+mal+ev+"/"+t}${pend?'<br><em>sin enviar</em>':""}</span></a>`}).join("");
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
    (salida / "index.html").write_text(INDICE.replace("__LOTES__", json.dumps(lotes, ensure_ascii=False)),
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


def armar_lotes(raiz, carpeta, salida, tam, max_lado, solo=None, lista=None, prefijo="lote"):
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


VERSION_LABELME = "4.0.0-beta.7"  # la de los JSON del dataset (AnyLabeling)
META_NUEVA = {"score": None, "group_id": None, "description": "", "difficult": False,
              "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": []}


def a_caja(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def misma_caja(a, b, tol=1e-4):
    return all(abs(u - v) <= tol for u, v in zip(a_caja(a), a_caja(b)))


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
        originales = None  # editada con una versión anterior de la página: emparejar por posición
    orig_por_i = {f.get("i", k): f for k, f in enumerate(originales)} if originales is not None else None
    nuevas, mapa, recalcular = [], {k: None for k in range(len(previas))}, []
    for k, f in enumerate(figs):
        i = f.get("i") if orig_por_i is not None else (k if k < len(previas) else None)
        x1, y1, x2, y2 = a_caja(f["pts"])
        if i is not None and i < len(previas):
            base = json.loads(json.dumps(previas[i]))
            o = orig_por_i.get(i) if orig_por_i is not None else None
            if o is None or f["label"] != o["label"]:
                base["label"] = f["label"]
            if o is None or not misma_caja(f["pts"], o["pts"]):
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


def aplicar_correcciones(origen, archivo):
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
                    help="aplicar revision_lote_XX.json enviado desde el celular")
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
        prefijo = a.prefijo or (Path(a.lista).stem if a.lista else "lote")
        prefijo = "".join(c if c.isalnum() else "_" for c in prefijo).strip("_").lower() or "lote"
        armar_lotes(a.raiz, None if a.carpeta == "." else a.carpeta, a.salida, a.lote, a.max_lado,
                    a.solo, a.lista, prefijo)


if __name__ == "__main__":
    main()
