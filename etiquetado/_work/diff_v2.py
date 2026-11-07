import json,glob,os,collections
base=r"D:\新草莓\DatasetId_360753_1652783343"
def load(d):
    out={}
    for f in glob.glob(os.path.join(base,d,"*.json")):
        j=json.load(open(f,encoding='utf-8'))
        labs={}
        for s in j['shapes']:
            xs=[p[0] for p in s['points']]; ys=[p[1] for p in s['points']]
            labs.setdefault((round(min(xs)),round(min(ys)),round(max(xs)),round(max(ys))),[]).append(s['label'])
        out[os.path.basename(f)]=labs
    return out
a=load("dataset_relabeled_v2"); b=load("data_anylabeling_2")
chg=collections.Counter(); nfiles=0; added=0; removed=0; files_changed=[]
for f in set(a)&set(b):
    A,B=a[f],b[f]; c=0
    for box in set(A)&set(B):
        for l1,l2 in zip(sorted(A[box]),sorted(B[box])):
            if l1!=l2: chg[(l1,l2)]+=1; c+=1
    na=len(set(A)-set(B)); nb=len(set(B)-set(A))
    removed+=na; added+=nb
    if c or na or nb: nfiles+=1; files_changed.append((f,c,na,nb))
print("MANUAL EDITS by user (dataset_relabeled_v2 -> data_anylabeling_2)")
print(f"files touched: {nfiles}; label flips: {sum(chg.values())}; boxes removed/moved: {removed}; boxes added/moved: {added}")
for k,v in chg.most_common(): print(f"   {k[0]:16s} -> {k[1]:16s} {v}")
print("sample files:", files_changed[:15])
# image stats
import cv2,random
imgs=glob.glob(os.path.join(base,"Images","*.png"))
random.seed(0)
sizes=collections.Counter()
for p in random.sample(imgs,40):
    im=cv2.imread(p); sizes[im.shape]+=1
print("image shapes:", sizes)
