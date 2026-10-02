#!/usr/bin/env python3
"""Train the final pairwise matcher with group-aware OOF threshold calibration."""
import argparse, os, random, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "modules"))
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from data_loader import read_tsv
from indexer import build_index
from labels import load_ground_truth_compact
from phase2 import build_training_matrix, MODEL_FEATURES, macro_fbeta, threshold_predictions, save_model

def make_model(seed):
    return LGBMClassifier(
        n_estimators=1000, learning_rate=.03, num_leaves=63, max_depth=-1,
        min_child_samples=40, subsample=.9, colsample_bytree=.9,
        reg_lambda=2.0, reg_alpha=.1, objective='binary', verbosity=-1,
        random_state=seed, n_jobs=-1
    )

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-dir',required=True); ap.add_argument('--work-dir',default='artifacts'); ap.add_argument('--sample-s1',type=int,default=60000); ap.add_argument('--hard-negatives',type=int,default=40); ap.add_argument('--max-candidates',type=int,default=0); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    wd=Path(args.work_dir); wd.mkdir(parents=True,exist_ok=True)
    s1=list(read_tsv(os.path.join(args.train_dir,'train_source1.tsv')))
    rng=random.Random(args.seed); rng.shuffle(s1); sample=s1[:min(args.sample_s1,len(s1))]; sample_ids={r['entity_id'] for r in sample}
    gt=load_ground_truth_compact(os.path.join(args.train_dir,'train_ground_truth.tsv'))
    indexes={}
    for src in ('S2','S3'):
        path=os.path.join(args.train_dir,f'train_source{src[1:]}.tsv'); indexes[src]=build_index(path,src,limit=None,backend='sqlite',db_path=str(wd/f'{src.lower()}_full.sqlite'))
    print(f'Building training pairs for {len(sample):,} S1 entities...', flush=True)
    X,y,groups,sources,pair_ids,stats=build_training_matrix(sample,indexes,gt,sample_ids,args.hard_negatives,(args.max_candidates or None),args.seed)
    np.savez_compressed(wd/'training_matrix.npz',X=X,y=y,groups=groups,sources=sources,pair_ids=pair_ids)
    with open(wd/'training_stats.json','w') as f: json.dump(stats,f,indent=2)
    print('Training matrix:',stats,'shape',X.shape)
    gkf=GroupKFold(n_splits=5); oof=np.zeros(len(y),dtype=np.float32)
    for fold,(tr,va) in enumerate(gkf.split(X,y,groups),1):
        m=make_model(args.seed+fold); m.fit(X[tr],y[tr]); oof[va]=m.predict_proba(X[va])[:,1]; print('fold',fold,'train',len(tr),'valid',len(va))
    # OOF identity-aware calibration.
    best=(-1,None,None); oof_ids={'s1':groups,'eid':pair_ids,'source':sources}
    for t in np.arange(.05,.951,.025):
        pred=threshold_predictions(oof_ids,oof,float(t)); sc=macro_fbeta(pred,gt)
        if sc>best[0]: best=(sc,float(t),0.0)
    lo=max(.01,best[1]-.03); hi=min(.99,best[1]+.03)
    for t in np.arange(lo,hi+.0001,.0025):
        pred=threshold_predictions(oof_ids,oof,float(t)); sc=macro_fbeta(pred,gt)
        if sc>best[0]: best=(sc,float(t),0.0)
    # Margin is only useful for ambiguous top candidates; optimize a small grid on OOF.
    base_t=best[1]
    for margin in (0,.01,.02,.03,.05,.08,.12):
        pred=threshold_predictions(oof_ids,oof,base_t,margin=margin); sc=macro_fbeta(pred,gt)
        if sc>best[0]: best=(sc,base_t,margin)
    print('OOF best Macro F0.5=',best)
    final=make_model(args.seed); final.fit(X,y)
    save_model(final,wd/'final_model.joblib',{'feature_columns':MODEL_FEATURES,'threshold':best[1],'top_margin':best[2],'oof_macro_f0_5':best[0],'training_stats':stats,'sample_s1':len(sample),'hard_negatives':args.hard_negatives,'max_candidates_per_source':args.max_candidates})
    print('Saved',wd/'final_model.joblib')
if __name__=='__main__': main()
