import json,random,shutil
from pathlib import Path
REPO=Path("/home/user/DatasetFresas");ORD=["unripe","early-pink","commercial-basic","commercial-high","overripe"]
def datos(l):
    s=(REPO/f"revision/{l}/index.html").read_text(encoding="utf-8");i=s.index("const DATOS=")+12;return json.loads(s[i:s.index(";\n",i)])
fb=json.loads((REPO/"feedback/revision_lote_01.json").read_text(encoding="utf-8"))
D1=datos("lote_01");random.seed(0)
fotos=[d for d in D1 if d["orig"] in fb["marcas"]]
random.shuffle(fotos);val=set(d["orig"] for d in fotos[:40])
out=Path("ds");shutil.rmtree(out,ignore_errors=True)
n=0
for d in fotos:
    figs=fb["correcciones"].get(d["orig"],d["figs"])
    sp="val" if d["orig"] in val else "train"
    (out/"images"/sp).mkdir(parents=True,exist_ok=True);(out/"labels"/sp).mkdir(parents=True,exist_ok=True)
    stem=Path(d["orig"]).stem
    shutil.copy(REPO/"revision/lote_01"/d["img"],out/"images"/sp/f"{stem}.jpg")
    L=[]
    for f in figs:
        xs=[p[0] for p in f["pts"]];ys=[p[1] for p in f["pts"]];x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys)
        L.append(f"{ORD.index(f['label'])} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}");n+=1
    (out/"labels"/sp/f"{stem}.txt").write_text("\n".join(L)+"\n")
(out/"data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames: {ORD}\n")
print("fotos",len(fotos),"val",len(val),"cajas",n)
