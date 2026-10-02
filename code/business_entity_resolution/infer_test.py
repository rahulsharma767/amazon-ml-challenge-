#!/usr/bin/env python3
"""Frozen test inference and required output generation.

Usage:
python infer_test.py --test-dir dataset/test --model artifacts/final_model.joblib --out output
"""
import argparse,csv,json,os,sys
from pathlib import Path
import numpy as np
import joblib
sys.path.insert(0,str(Path(__file__).parent/'modules'))
from data_loader import read_tsv
from indexer import build_index
from candidates import find_candidates
from phase2 import row_from_index, meta_feature_row, MODEL_FEATURES, threshold_predictions

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--test-dir',required=True); ap.add_argument('--model',required=True); ap.add_argument('--out',default='output'); ap.add_argument('--db-dir',default='artifacts/test_indexes'); args=ap.parse_args()
 out=Path(args.out); out.mkdir(parents=True,exist_ok=True); db=Path(args.db_dir); db.mkdir(parents=True,exist_ok=True)
 model=joblib.load(args.model); meta=json.load(open(str(args.model)+'.json',encoding='utf-8')); threshold=float(meta['threshold']); margin=float(meta.get('top_margin',0))
 indexes={}
 for src in ('S2','S3'):
  indexes[src]=build_index(os.path.join(args.test_dir,f'test_source{src[1:]}.tsv'),src,limit=None,backend='sqlite',db_path=str(db/f'{src.lower()}_full.sqlite'))
 with open(out/'matching_results.tsv','w',encoding='utf-8',newline='') as fm, open(out/'candidate_pairs.tsv','w',encoding='utf-8',newline='') as fc:
  wm=csv.writer(fm,delimiter='\t',lineterminator='\n'); wc=csv.writer(fc,delimiter='\t',lineterminator='\n')
  wm.writerow(['source1_entity_id','matched_entity_ids']); wc.writerow(['source1_entity_id','candidate_entity_ids'])
  total=0; total_c=0; total_m=0
  for s1 in read_tsv(os.path.join(args.test_dir,'test_source1.tsv')):
   sid=s1['entity_id']; rows=[]; candidate_ids=[]
   for src,index in indexes.items():
    cand=find_candidates(s1,index)
    items=[]
    for suffix,info in cand.items():
     r=row_from_index(index,src,suffix)
     if not r: continue
     meta2={**info,'blocking_name_key':int(any(k.startswith(('NP','NS','FL','F2','FML','UN')) for k in info['key_types'])),'blocking_address_key':int(any(k.startswith(('A','N')) for k in info['key_types'])),'candidate_count':len(cand)}
     items.append((r,meta2))
    # rank within source using a cheap deterministic pre-score before model scoring
    from rapidfuzz.fuzz import ratio
    items.sort(key=lambda x:.65*ratio(s1.get('business_name',''),x[0].get('business_name',''))+.35*ratio(s1.get('business_address',''),x[0].get('business_address','')),reverse=True)
    for rank,(r,m) in enumerate(items,1):
     m['candidate_rank']=rank; rows.append((r,m)); candidate_ids.append(r['entity_id'])
   X=np.asarray([[meta_feature_row(s1,r,m)[c] for c in MODEL_FEATURES] for r,m in rows],dtype=np.float32)
   probs=model.predict_proba(X)[:,1] if len(rows) else np.array([],dtype=float)
   accepted=[]
   if len(rows):
    order=np.argsort(-probs)
    for pos in order:
     p=float(probs[pos]);
     if p < threshold: continue
     if margin>0 and pos==order[0] and len(order)>1 and p-float(probs[order[1]])<margin: continue
     accepted.append((rows[pos][0]['entity_id'],p))
   accepted_ids=[]
   for eid,_ in sorted(accepted,key=lambda x:x[0]):
    if eid not in accepted_ids: accepted_ids.append(eid)
   wc.writerow([sid,','.join(dict.fromkeys(candidate_ids))]); wm.writerow([sid,','.join(accepted_ids)])
   total+=1; total_c+=len(candidate_ids); total_m+=len(accepted_ids)
   if total%10000==0: print(f'processed {total:,} S1; mean candidates={total_c/total:.1f}; mean matches={total_m/total:.2f}',flush=True)
 print('DONE',total,'S1; mean candidates',total_c/max(total,1),'mean matches',total_m/max(total,1))
if __name__=='__main__': main()
