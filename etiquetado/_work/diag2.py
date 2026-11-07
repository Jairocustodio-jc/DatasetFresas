import sys, os, random, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import *
from pigment import analyze, CFG

def get_crop(stem, bi):
    im = imread_u(os.path.join(SRC, stem + ".png"))
    j, boxes = load_boxes(os.path.join(SRC, stem + ".json")); b = boxes[bi]
    p = CFG['PAD']; H,W = im.shape[:2]
    x1,y1 = max(0,int(b['x1'])-p), max(0,int(b['y1'])-p)
    x2,y2 = min(W,int(b['x2'])+p), min(H,int(b['y2'])+p)
    return im[y1:y2, x1:x2], b['label']

def fit(img, T):
    h,w = img.shape[:2]; s = T/max(h,w)
    r = cv2.resize(img, (max(1,int(w*s)), max(1,int(h*s))))
    cv_ = np.full((T,T,3), 40, np.uint8); hh,ww = r.shape[:2]
    cv_[(T-hh)//2:(T-hh)//2+hh, (T-ww)//2:(T-ww)//2+ww] = r; return cv_

def panel(stem, bi, T=120):
    crop, lab = get_crop(stem, bi)
    m, d = analyze(crop)
    c = d['crop']
    seg = c.copy(); seg[d['surface']==0] = (seg[d['surface']==0]*0.2).astype(np.uint8)
    o1 = c.copy(); o1[d['m_pink']>0] = (0.4*o1[d['m_pink']>0] + 0.6*np.array([255,0,255])).astype(np.uint8)
    o2 = c.copy(); o2[d['m_red']>0]  = (0.4*o2[d['m_red']>0]  + 0.6*np.array([0,255,255])).astype(np.uint8)
    strip = np.hstack([fit(c,T), fit(seg,T), fit(o1,T), fit(o2,T)])
    txt = np.full((32, strip.shape[1], 3), 15, np.uint8)
    cv2.putText(txt, f"{stem}#{bi} {lab[:13]}", (3,13), 0, 0.36, (255,255,255),1)
    cv2.putText(txt, f"pink={m['pct_pink']:.0f} red={m['pct_red']:.0f} deep={m['pct_deep']:.0f} core={m['pct_core']:.0f} {m['how'][:8]}", (3,28), 0, 0.33, (120,220,255),1)
    return np.vstack([strip, txt])

sel = [("2015",1),("2015",2),("2015",4),("1092",6),("1917",2),("198",2),("435",6),
       ("2015",3),("1539",0),("209",0),("916",0),("2015",0),("1092",0),("262",0),
       ("1567",1),("1092",1),("1092",2),("1567",0),("1539",1),("2091",0)]
panels=[panel(*s) for s in sel]
rows=[np.hstack(panels[k:k+4]) for k in range(0,len(panels),4)]
w=max(r.shape[1] for r in rows)
rows=[np.pad(r,((0,0),(0,w-r.shape[1]),(0,0))) for r in rows]
imwrite_u(os.path.join(WORK,"diag2.png"), np.vstack(rows)); print("ok")
