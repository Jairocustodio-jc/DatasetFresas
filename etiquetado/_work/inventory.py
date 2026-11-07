import json, os, glob, collections, sys
base = r"D:\新草莓\DatasetId_360753_1652783343"
def scan(d):
    cnt = collections.Counter(); files=0; boxes=0; vers=collections.Counter(); npts=collections.Counter()
    per_file={}
    for f in glob.glob(os.path.join(base,d,"*.json")):
        try: j=json.load(open(f,encoding='utf-8'))
        except Exception as e: continue
        if 'shapes' not in j: continue
        files+=1; vers[j.get('version','?')]+=1
        labs=[]
        for s in j['shapes']:
            cnt[s['label']]+=1; boxes+=1; npts[len(s['points'])]+=1
            labs.append((s['label'], tuple(round(c,1) for p in s['points'] for c in p)))
        per_file[os.path.basename(f)]=labs
    return cnt, files, boxes, vers, npts, per_file
res={}
for d in ["Images_AnyLabeling_5cls","dataset_anylabeling","dataset_relabeled_v2","data_anylabeling_2"]:
    cnt,files,boxes,vers,npts,pf = scan(d)
    res[d]=pf
    print(f"=== {d}: {files} json, {boxes} boxes, versions={dict(vers)}, pts={dict(npts)}")
    for k,v in cnt.most_common(): print(f"    {v:6d}  {k}")
# compare data_anylabeling_2 vs dataset_anylabeling (bbox geometry + labels)
a=res["dataset_anylabeling"]; b=res["data_anylabeling_2"]
common=set(a)&set(b)
print(f"\ncommon files {len(common)}; only in dataset_anylabeling: {len(set(a)-set(b))}; only in data_anylabeling_2: {len(set(b)-set(a))}")
print("missing in _2:", sorted(set(a)-set(b))[:10])
def norm(labs):
    out=[]
    for l,pts in labs:
        xs=pts[0::2]; ys=pts[1::2]
        out.append((l,(round(min(xs)),round(min(ys)),round(max(xs)),round(max(ys)))))
    return out
same=0; lab_diff=0; geo_diff=0; diff_examples=[]
labchange=collections.Counter()
for f in common:
    na={}; nb={}
    for l,box in norm(a[f]): na.setdefault(box,[]).append(l)
    for l,box in norm(b[f]): nb.setdefault(box,[]).append(l)
    if set(na)!=set(nb):
        geo_diff+=1
        if len(diff_examples)<5: diff_examples.append((f,len(na),len(nb)))
    for box in set(na)&set(nb):
        for l1,l2 in zip(sorted(na[box]),sorted(nb[box])):
            if l1!=l2: lab_diff+=1; labchange[(l1,l2)]+=1
            else: same+=1
print(f"files with geometry/box-set differences: {geo_diff}; example {diff_examples}")
print(f"identical-box labels equal: {same}, changed: {lab_diff}")
for k,v in labchange.most_common(20): print("   ",k,v)
