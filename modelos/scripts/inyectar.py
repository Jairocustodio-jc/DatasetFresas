"""Escribe en revision/<lote>/index.html las sugerencias del modelo: f.sug (estado) y d.falta (cajas faltantes)."""
import json,sys
from comparar import *
modelo,lote,t_cls,t_det=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
ruta=REPO/"revision"/lote/"index.html";txt=ruta.read_text(encoding="utf-8")
i=txt.index("const DATOS=")+12;j=txt.index(";\n",i);D=json.loads(txt[i:j])
preds=predecir(modelo,lote);ns=nf=0
for d in D:
    for f in d["figs"]:f.pop("sug",None);f.pop("sc",None)
    d.pop("falta",None)
    sug,faltan=comparar(d["figs"],preds[d["orig"]],t_cls,t_det)
    for k,(c,p) in sug.items():d["figs"][k]["sug"]=c;d["figs"][k]["sc"]=p;ns+=1
    if faltan:
        d["falta"]=[{"label":p["cls"],"conf":round(p["conf"],2),"pts":[[p["box"][0],p["box"][1]],[p["box"][2],p["box"][1]],[p["box"][2],p["box"][3]],[p["box"][0],p["box"][3]]]} for p in faltan];nf+=len(faltan)
ruta.write_text(txt[:i]+json.dumps(D,ensure_ascii=False)+txt[j:],encoding="utf-8")
json.dump({d["orig"]:{"sug":{k:[f["label"],f["sug"],f["sc"]] for k,f in enumerate(d["figs"]) if "sug" in f},"falta":d.get("falta",[])} for d in D},open(f"sug_{lote}.json","w"),ensure_ascii=False)
print(f"{lote}: {ns} sugerencias de estado, {nf} posibles cajas faltantes, en {sum(bool(any('sug' in f for f in d['figs']) or d.get('falta')) for d in D)} fotos")
