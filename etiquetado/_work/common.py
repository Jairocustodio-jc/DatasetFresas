import cv2, numpy as np, json, os, glob
BASE = r"D:\新草莓\DatasetId_360753_1652783343"
WORK = os.path.join(BASE, "_work")
SRC  = os.path.join(BASE, "data_anylabeling_2")
# nombres en ingles: los usan los scripts de la pasada automatica (extract/candidates/apply),
# que trabajan sobre data_anylabeling_2. El dataset final ya esta en espanol.
CLASSES = ["unripe","early-pink","commercial-basic","commercial-high","overripe"]
CLASSES_ES = ["verde_inmadura", "rosa_temprano", "comercial_basico", "comercial_alto", "madurez_plena"]
ES = ["0_Verde_inmadura","1_Rosa_temprano","2_Comercial_basico","3_Comercial_alto","4_Sobremadura"]
def imread_u(path, flags=cv2.IMREAD_COLOR):
    a = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(a, flags)
def imwrite_u(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok: buf.tofile(path)
    return ok
def load_boxes(jpath):
    j = json.load(open(jpath, encoding='utf-8'))
    out = []
    for i, s in enumerate(j['shapes']):
        xs = [p[0] for p in s['points']]; ys = [p[1] for p in s['points']]
        out.append(dict(idx=i, label=s['label'], x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)))
    return j, out
def stems():
    return sorted([os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(SRC,"*.json"))], key=lambda s:int(s) if s.isdigit() else 1e9)
