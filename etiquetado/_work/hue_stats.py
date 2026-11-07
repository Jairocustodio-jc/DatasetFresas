import sys, os, random, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import *
from pigment import analyze, CFG, ellipse_mask, masks_of, segment_fruit

random.seed(3)
st = stems(); random.shuffle(st)
fruit_green = []   # pixeles del nucleo de frutos 'unripe'
outside     = []   # pixeles fuera de la mascara del fruto (fondo: hojas, suelo)
red_core    = []
n=0
for s in st[:120]:
    jp = os.path.join(SRC, s + ".json")
    j, boxes = load_boxes(jp)
    im = imread_u(os.path.join(SRC, s + ".png"))
    H0,W0 = im.shape[:2]
    for b in boxes:
        if (b['x2']-b['x1']) < 45 or (b['y2']-b['y1']) < 45: continue
        p=CFG['PAD']
        crop = im[max(0,int(b['y1'])-p):min(H0,int(b['y2'])+p), max(0,int(b['x1'])-p):min(W0,int(b['x2'])+p)]
        if crop.size==0: continue
        sc = CFG['MAXSIDE']/max(crop.shape[:2])
        if sc<1: crop = cv2.resize(crop,(int(crop.shape[1]*sc),int(crop.shape[0]*sc)),interpolation=cv2.INTER_AREA)
        sm = cv2.medianBlur(crop,3)
        fr,how = segment_fruit(sm)
        ch = masks_of(sm)
        h,w = sm.shape[:2]
        core = (ellipse_mask(h,w,0.28,0.28) & fr).astype(np.uint8)
        sel = core>0
        if sel.sum()<50: continue
        Hh,Ss,Vv,Aa,Ll,Bb = ch['H'][sel],ch['S'][sel],ch['V'][sel],ch['A'][sel],ch['L'][sel],ch['B'][sel]
        rec = np.stack([Hh,Ss,Vv,Aa,Ll,Bb],1)
        if b['label']=='unripe': fruit_green.append(rec)
        if b['label'] in ('overripe','commercial-high'): red_core.append(rec)
        bg = (fr==0)
        if bg.sum()>50:
            idx = np.random.choice(bg.sum(), min(400,bg.sum()), replace=False)
            outside.append(np.stack([ch[k][bg][idx] for k in 'HSVAL'],1) if False else
                           np.stack([ch['H'][bg][idx],ch['S'][bg][idx],ch['V'][bg][idx],ch['A'][bg][idx],ch['L'][bg][idx],ch['B'][bg][idx]],1))
        n+=1
    if n>300: break
def rep(name, arrs):
    if not arrs: print(name,"vacio"); return
    X=np.concatenate(arrs)
    print(f"--- {name}  n={len(X)}")
    for i,c in enumerate(['H','S','V','a*','L','b*']):
        q=np.percentile(X[:,i],[5,25,50,75,95])
        print(f"   {c:3s} p5={q[0]:6.1f} p25={q[1]:6.1f} p50={q[2]:6.1f} p75={q[3]:6.1f} p95={q[4]:6.1f}")
rep("FRUTO VERDE (nucleo unripe)", fruit_green)
rep("FRUTO ROJO (nucleo ripe)", red_core)
rep("FONDO (hojas/suelo)", outside)
# Distribucion de H solo para pixeles verdosos (a*<125) en cada grupo
for name,arrs in [("fruto-verde",fruit_green),("fondo",outside)]:
    if not arrs: continue
    X=np.concatenate(arrs); g=X[X[:,3]<126]
    if len(g)==0: continue
    print(f"[{name}] pixeles a*<126: n={len(g)}  H p5/25/50/75/95 =", np.percentile(g[:,0],[5,25,50,75,95]).round(1),
          " S p50=",np.percentile(g[:,1],50).round(1)," V p50=",np.percentile(g[:,2],50).round(1), " b* p50=",np.percentile(g[:,5],50).round(1))
