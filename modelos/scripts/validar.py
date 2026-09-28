import json,sys
from comparar import *
import importlib.util
spec=importlib.util.spec_from_file_location("rf",str(REPO/"revisar_fresas.py"));rf=importlib.util.module_from_spec(spec);spec.loader.exec_module(rf)
fb=json.loads((REPO/"feedback/revision_lote_01.json").read_text(encoding="utf-8"))
val={p.stem+".png" for p in Path("ds/images/val").glob("*.jpg")}
D={d["orig"]:d for d in datos("lote_01")}
preds=predecir(sys.argv[1],"lote_01",val)
for t in (0.4,0.5,0.6,0.7):
    tot=bien=errores_agente=detectados=0;falt=0
    for orig in val:
        agente=D[orig]["figs"]                       # lo que mostraba la página (etiqueta automática)
        final=fb["correcciones"].get(orig,agente)   # tu corrección
        fi=rf.asignar_indices(final,agente) if orig in fb["correcciones"] else [dict(f) for f in final]
        verdad={f.get("i"):f["label"] for f in fi if f.get("i") is not None}
        sug,faltan=comparar(agente,preds[orig],t,t);falt+=len(faltan)
        for k,(c,p) in sug.items():
            tot+=1;bien+=verdad.get(agente[k].get("i",k))==c
        for f in agente:
            v=verdad.get(f.get("i"));
            if v and v!=f["label"]:
                errores_agente+=1;detectados+=agente.index(f) in sug and sug[agente.index(f)][0]==v
    print(f"umbral {t}: {tot} sugerencias de etiqueta, correctas {bien} ({bien/max(tot,1):.0%}); errores del agente detectados {detectados}/{errores_agente}; cajas 'faltantes' {falt}")
