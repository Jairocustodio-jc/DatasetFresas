import sys, os, random, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import *
from pigment import analyze, CFG

def panel(stem, bi, T=130):
    im = imread_u(os.path.join(SRC, stem + ".png"))
    j, boxes = load_boxes(os.path.join(SRC, stem + ".json"))
    b = boxes[bi]
    p = CFG['PAD']; H,W = im.shape[:2]
    x1,y1 = max(0,int(b['x1'])-p), max(0,int(b['y1'])-p)
    x2,y2 = min(W,int(b['x2'])+p), min(H,int(b['y2'])+p)
    crop = im[y1:y2, x1:x2]
    m, d = analyze(crop)
    c = d['crop']
    def show(img): 
        h,w=img.shape[:2]; s=T/max(h,w)
        r=cv2.resize(img,(max(1,int(w*s)),max(1,int(h*s))))
        cv = np.full((T,T,3),40,np.uint8); hh,ww=r.shape[:2]
        cv[(T-hh)//2:(T-hh)//2+hh,(T-ww)//2:(T-ww)//2+ww]=r; return cv
    seg = c.copy(); seg[d['surface']==0] = (seg[d['surface']==0]*0.25).astype(np.uint8)
    ov = c.copy()
    ov[d['colored']>0] = (0.45*ov[d['colored']>0] + 0.55*np.array([255,0,255])).astype(np.uint8)
    ov[d['leaf']>0]    = (0.5*ov[d['leaf']>0]    + 0.5*np.array([0,255,0])).astype(np.uint8)
    strip = np.hstack([show(c), show(seg), show(ov)])
    txt = np.full((34, strip.shape[1], 3), 15, np.uint8)
    cv2.putText(txt, f"{stem}#{bi} {b['label'][:12]}", (3,13), 0, 0.38, (255,255,255),1)
    cv2.putText(txt, f"col={m['pct_colored']:.0f}% red={m['pct_red']:.0f}% a90={m.get('a_p90',0)} {m['how'][:9]}", (3,29), 0, 0.35, (120,220,255),1)
    return np.vstack([strip, txt]), m

random.seed(11)
st = stems(); random.shuffle(st)
sel = []
need = {c: 4 for c in CLASSES}
for s in st:
    j, boxes = load_boxes(os.path.join(SRC, s + ".json"))
    for i,b in enumerate(boxes):
        if need.get(b['label'],0) > 0 and (b['x2']-b['x1'])>40:
            need[b['label']] -= 1; sel.append((s,i,b['label']))
    if sum(need.values())==0: break
sel.sort(key=lambda t: CLASSES.index(t[2]))
panels=[panel(s,i)[0] for s,i,_ in sel]
rows=[np.hstack(panels[k:k+4]) for k in range(0,len(panels),4)]
w=max(r.shape[1] for r in rows)
rows=[np.pad(r,((0,0),(0,w-r.shape[1]),(0,0))) for r in rows]
imwrite_u(os.path.join(WORK,"diag1.png"), np.vstack(rows))
print("ok", len(panels))
