"""Compara predicciones YOLO con cajas existentes -> sugerencias de etiqueta / cajas faltantes."""
import json,sys
from pathlib import Path
from ultralytics import YOLO
REPO=Path("/home/user/DatasetFresas");ORD=["unripe","early-pink","commercial-basic","commercial-high","overripe"]
def datos(l):
    s=(REPO/f"revision/{l}/index.html").read_text(encoding="utf-8");i=s.index("const DATOS=")+12;return json.loads(s[i:s.index(";\n",i)])
def caja(pts):xs=[p[0] for p in pts];ys=[p[1] for p in pts];return min(xs),min(ys),max(xs),max(ys)
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]));I=ix*iy
    U=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-I;return I/U if U else 0
def predecir(modelo,lote,fotos=None):
    m=YOLO(modelo);res={}
    for d in datos(lote):
        if fotos and d["orig"] not in fotos:continue
        r=m.predict(str(REPO/"revision"/lote/d["img"]),imgsz=640,conf=0.05,iou=0.5,verbose=False,device="cpu")[0]
        W,H=r.orig_shape[1],r.orig_shape[0]
        res[d["orig"]]=[{"box":[b[0]/W,b[1]/H,b[2]/W,b[3]/H],"cls":ORD[int(c)],"conf":float(p),
                          "probs":None} for b,c,p in zip(r.boxes.xyxy.tolist(),r.boxes.cls.tolist(),r.boxes.conf.tolist())]
    return res
def comparar(figs,preds,t_cls=0.5,t_det=0.5):
    """figs: cajas actuales. Devuelve sugerencias de etiqueta por índice y cajas faltantes."""
    boxes=[caja(f["pts"]) for f in figs];sug={};usadas=set()
    for k,b in enumerate(boxes):
        cand=[(iou(b,p["box"]),j,p) for j,p in enumerate(preds)];cand=[c for c in cand if c[0]>=0.5]
        if not cand:continue
        # la clase con mayor confianza entre predicciones que coinciden con la caja
        v,j,p=max(cand,key=lambda c:c[2]["conf"]);usadas.update(c[1] for c in cand)
        if p["cls"]!=figs[k]["label"] and p["conf"]>=t_cls:sug[k]=(p["cls"],round(p["conf"],2))
    faltan=[p for j,p in enumerate(preds) if j not in usadas and p["conf"]>=t_det and all(iou(p["box"],b)<0.3 for b in boxes)]
    # quitar faltantes duplicados
    dedup=[]
    for p in sorted(faltan,key=lambda p:-p["conf"]):
        if all(iou(p["box"],q["box"])<0.4 for q in dedup):dedup.append(p)
    return sug,dedup
