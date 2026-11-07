import sys, os, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import *
from pigment import CFG, segment_fruit, masks_of, ellipse_mask

THS=[128,132,136,140,145]
def get_crop(stem, bi):
    im = imread_u(os.path.join(SRC, stem + ".png")); j,boxes = load_boxes(os.path.join(SRC, stem+".json")); b=boxes[bi]
    p=CFG['PAD']; H,W=im.shape[:2]
    return im[max(0,int(b['y1'])-p):min(H,int(b['y2'])+p), max(0,int(b['x1'])-p):min(W,int(b['x2'])+p)], b['label']
def fit(img,T):
    h,w=img.shape[:2]; s=T/max(h,w); r=cv2.resize(img,(max(1,int(w*s)),max(1,int(h*s))))
    c=np.full((T,T,3),40,np.uint8); hh,ww=r.shape[:2]; c[(T-hh)//2:(T-hh)//2+hh,(T-ww)//2:(T-ww)//2+ww]=r; return c
def row(stem,bi,T=118):
    crop,lab=get_crop(stem,bi)
    s=CFG['MAXSIDE']/max(crop.shape[:2])
    if s<1: crop=cv2.resize(crop,(int(crop.shape[1]*s),int(crop.shape[0]*s)),interpolation=cv2.INTER_AREA)
    sm=cv2.medianBlur(crop,3); fr,how=segment_fruit(sm); ch=masks_of(sm)
    H,S,V,A=ch['H'],ch['S'],ch['V'],ch['A']
    leaf=((H>=38)&(H<=100)&(S>=45)).astype(np.uint8)
    leaf=cv2.morphologyEx(leaf,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    dark=((V<85)).astype(np.uint8)
    surf=(fr&(1-ch['spec'])&(1-leaf)&(1-dark)).astype(np.uint8)
    As=cv2.medianBlur(A.astype(np.uint8),7).astype(np.int16)
    imgs=[fit(sm,T)]
    segv=sm.copy(); segv[surf==0]=(segv[surf==0]*0.18).astype(np.uint8); imgs.append(fit(segv,T))
    pcts=[]
    for t in THS:
        m=((As>=t)&(surf>0)).astype(np.uint8)
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)); m=(m&surf).astype(np.uint8)
        pct=100.0*m.sum()/max(1,surf.sum()); pcts.append(pct)
        o=sm.copy(); o[m>0]=(0.35*o[m>0]+0.65*np.array([255,0,255])).astype(np.uint8); imgs.append(fit(o,T))
    strip=np.hstack(imgs)
    txt=np.full((30,strip.shape[1],3),15,np.uint8)
    cv2.putText(txt,f"{stem}#{bi} {lab[:12]}",(3,12),0,0.36,(255,255,255),1)
    cv2.putText(txt," ".join(f"a{t}:{p:.0f}%" for t,p in zip(THS,pcts)),(150,12),0,0.36,(120,220,255),1)
    cv2.putText(txt,"crop | superficie | "+" | ".join(f"a>={t}" for t in THS),(3,26),0,0.33,(150,150,150),1)
    return np.vstack([strip,txt])
sel=[("1539",0),("916",0),("2015",0),("262",0),("2091",0),("435",6),("198",2),("209",0),("1092",0),("1567",1)]
out=np.vstack([row(*s) for s in sel])
imwrite_u(os.path.join(WORK,"sweep.png"),out); print("ok",out.shape)
