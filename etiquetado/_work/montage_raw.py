import sys, os, random, cv2, numpy as np
sys.path.insert(0, r"D:\新草莓\DatasetId_360753_1652783343\_work")
from common import *
random.seed(7)
st = stems()
picks = []
random.shuffle(st)
by = {c: [] for c in CLASSES}
for s in st:
    j, boxes = load_boxes(os.path.join(SRC, s + ".json"))
    for b in boxes:
        if len(by[b['label']]) < 8 and (b['x2']-b['x1'])>28:
            by[b['label']].append((s, b))
    if all(len(v) >= 8 for v in by.values()): break
T=120; rows=[]
for c in CLASSES:
    row=[]
    for s, b in by[c][:8]:
        im = imread_u(os.path.join(SRC, s + ".png"))
        x1,y1,x2,y2 = [int(round(v)) for v in (b['x1'],b['y1'],b['x2'],b['y2'])]
        crop = im[max(0,y1):y2, max(0,x1):x2]
        h,w = crop.shape[:2]; sc = T/max(h,w)
        crop = cv2.resize(crop, (max(1,int(w*sc)), max(1,int(h*sc))))
        canvas = np.full((T,T,3), 30, np.uint8)
        ch,cw = crop.shape[:2]; canvas[(T-ch)//2:(T-ch)//2+ch, (T-cw)//2:(T-cw)//2+cw] = crop
        cv2.putText(canvas, s, (2,12), 0, 0.35, (255,255,255), 1)
        row.append(canvas)
    lab = np.full((T, 150, 3), 20, np.uint8)
    cv2.putText(lab, c, (3, 60), 0, 0.45, (200,255,200), 1)
    rows.append(np.hstack([lab]+row))
out = np.vstack(rows)
imwrite_u(os.path.join(WORK,"montage_raw.png"), out)
print("saved", out.shape)
