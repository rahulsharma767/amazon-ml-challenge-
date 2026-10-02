"""Phase-2 training/inference utilities.

The module is intentionally streaming-oriented: source-2/source-3 rows live in
SQLite indexes and candidate features are generated only for the candidates
actually scored by the model.
"""
import csv, json, math, os, random
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from candidates import find_candidates
from advanced_features import pair_features, FEATURE_COLUMNS
from labels import load_ground_truth_compact

MODEL_FEATURES = FEATURE_COLUMNS + ["source_is_s2","source_is_s3"]

def row_from_index(index, source, suffix):
    attrs=index.get_attrs(suffix)
    if attrs is None: return None
    return {"entity_id":f"{source}-{suffix}","business_name":attrs[0] or "","business_address":attrs[1] or "","country":attrs[2] or ""}

def meta_feature_row(s1, s2, base):
    f=pair_features(s1,s2,base)
    src=s2["entity_id"].split("-",1)[0]
    f["source_is_s2"]=int(src=="S2"); f["source_is_s3"]=int(src=="S3")
    return f

def candidate_rows_for_s1(s1, indexes_by_source, freq_caps=None, max_per_source=None, protected_ids=None):
    """Build bounded training candidates without one SQLite query per row.

    ``max_per_source`` is a real bound now. We first rank candidates using
    cheap blocking evidence, fetch attributes in SQL batches for a bounded
    shortlist, then use RapidFuzz to select hard negatives. Ground-truth
    positives supplied in ``protected_ids`` are always retained when they
    are reachable through blocking, even if they fall outside the shortlist.
    """
    out = []
    protected_ids = protected_ids or set()
    from rapidfuzz.fuzz import ratio

    for source, index in indexes_by_source.items():
        cand = find_candidates(s1, index, freq_caps=freq_caps)
        if not cand:
            continue

        # Cheap ranking: multiple independent blocking keys and lower bucket
        # frequency are stronger evidence. This avoids materializing attrs for
        # every candidate when a blocking union is large.
        ranked = []
        protected_suffixes = []
        for suffix, info in cand.items():
            eid = f"{source}-{suffix}"
            key_count = len(info["key_types"])
            cheap = (key_count * 1000.0) - float(info.get("min_freq", 999999))
            if eid in protected_ids:
                protected_suffixes.append(suffix)
            ranked.append((cheap, suffix, info))

        ranked.sort(key=lambda x: x[0], reverse=True)
        if max_per_source is None:
            shortlist = ranked
        else:
            # Extra room improves hard-negative quality while remaining
            # bounded. Protected positives are added separately.
            shortlist = ranked[:max(4 * max_per_source, max_per_source)]

        suffixes = [x[1] for x in shortlist]
        for suffix in protected_suffixes:
            if suffix not in suffixes:
                suffixes.append(suffix)

        if hasattr(index, "get_attrs_many"):
            attrs_map = index.get_attrs_many(suffixes)
        else:
            attrs_map = {str(s): index.get_attrs(s) for s in suffixes}

        items = []
        for _, suffix, info in shortlist:
            attrs = attrs_map.get(str(suffix))
            if attrs is None:
                continue
            r = {
                "entity_id": f"{source}-{suffix}",
                "business_name": attrs[0] or "",
                "business_address": attrs[1] or "",
                "country": attrs[2] or "",
            }
            meta = {**info,
                    "blocking_name_key": int(any(k.startswith(("NP","NS","FL","F2","FML","UN")) for k in info["key_types"])),
                    "blocking_address_key": int(any(k.startswith(("A","N")) for k in info["key_types"]))}
            n1, n2 = str(s1.get("business_name","")), str(r.get("business_name",""))
            a1, a2 = str(s1.get("business_address","")), str(r.get("business_address",""))
            pre = .65 * ratio(n1, n2) + .35 * ratio(a1, a2)
            items.append((pre, r, meta))

        items.sort(key=lambda x: x[0], reverse=True)
        if max_per_source is not None:
            protected = [x for x in items if x[1]["entity_id"] in protected_ids]
            protected_ids_source = {x[1]["entity_id"] for x in protected}
            kept = protected + [x for x in items if x[1]["entity_id"] not in protected_ids_source]
            items = kept[:max_per_source + len(protected)]

        count = len(items)
        for rank, (_, r, meta) in enumerate(items, 1):
            meta["candidate_rank"] = rank
            meta["candidate_count"] = count
            out.append((r, meta))
    return out

def build_training_matrix(s1_rows,indexes_by_source,gt,sample_ids,hard_negatives=30,max_candidates_per_source=100,seed=42):
    rng=random.Random(seed); X=[]; y=[]; groups=[]; sources=[]; pair_ids=[]
    stats={"groups":0,"pairs":0,"positives":0,"negatives":0}
    for s1 in s1_rows:
        sid=s1["entity_id"]
        if sid not in sample_ids: continue
        positives=set(gt.get(sid,()))
        candidates=candidate_rows_for_s1(s1,indexes_by_source,max_per_source=max_candidates_per_source,protected_ids=positives)
        # Always retain all retrieved positives; among negatives retain the hardest by pre-ranking.
        pos=[]; neg=[]
        for r,m in candidates:
            if r["entity_id"] in positives: pos.append((r,m))
            else: neg.append((r,m))
        if not candidates: continue
        keep=pos+neg[:hard_negatives]
        for r,m in keep:
            X.append([meta_feature_row(s1,r,m)[c] for c in MODEL_FEATURES]); y.append(int(r["entity_id"] in positives)); groups.append(sid); sources.append(r["entity_id"].split("-",1)[0]); pair_ids.append(r["entity_id"])
        stats["groups"]+=1; stats["pairs"]+=len(keep); stats["positives"]+=len(pos); stats["negatives"]+=len(keep)-len(pos)
    return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.int8),np.asarray(groups),np.asarray(sources),np.asarray(pair_ids),stats

def macro_fbeta(pred_by_s1, gt, beta=.5):
    b2=beta*beta; vals=[]
    all_ids=set(gt)|set(pred_by_s1)
    for sid in all_ids:
        p=set(pred_by_s1.get(sid,set())); g=set(gt.get(sid,set()))
        tp=len(p&g); fp=len(p-g); fn=len(g-p)
        if tp==0:
            vals.append(1.0 if not p and not g else 0.0); continue
        precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
        vals.append((1+b2)*precision*recall/(b2*precision+recall) if precision+recall else 0.0)
    return float(np.mean(vals)) if vals else 0.0

def threshold_predictions(ids,probs,threshold,margin=0.0,source_thresholds=None):
    grouped=defaultdict(list)
    for sid,eid,src,p in zip(ids["s1"],ids["eid"],ids["source"],probs): grouped[sid].append((eid,src,float(p)))
    out={}
    for sid,items in grouped.items():
        items.sort(key=lambda x:x[2],reverse=True)
        accepted=[]
        for i,(eid,src,p) in enumerate(items):
            th=source_thresholds.get(src,threshold) if source_thresholds else threshold
            if p < th: continue
            if i==0 and len(items)>1 and margin>0 and (p-items[1][2])<margin: continue
            accepted.append(eid)
        out[sid]=set(accepted)
    return out

def optimize_threshold(model,X,ids,gt):
    probs=model.predict_proba(X)[:,1]
    best=(-1,None)
    # Coarse then fine grid; OOF only.
    for t in np.arange(.05,.951,.025):
        pred=threshold_predictions(ids,probs,float(t),margin=0.0)
        score=macro_fbeta(pred,gt)
        if score>best[0]: best=(score,float(t))
    lo=max(.01,best[1]-.03); hi=min(.99,best[1]+.03)
    for t in np.arange(lo,hi+.0001,.0025):
        pred=threshold_predictions(ids,probs,float(t),margin=0.0)
        score=macro_fbeta(pred,gt)
        if score>best[0]: best=(score,float(t))
    return best[1],best[0],probs

def save_model(model,path,metadata):
    import joblib
    Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,path)
    with open(str(path)+".json","w",encoding="utf-8") as f: json.dump(metadata,f,indent=2)
