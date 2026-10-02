#!/usr/bin/env python3
"""Strict local validator matching the official output contract."""
import argparse,csv,os,sys

def read(path):
 with open(path,encoding='utf-8',newline='') as f: return list(csv.DictReader(f,delimiter='\t'))
def ids(path,col):
 out=set()
 for r in read(path): out.update(x for x in (r[col] or '').split(',') if x)
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--matching',required=True); ap.add_argument('--candidate',required=True); ap.add_argument('--test-dir',required=True); a=ap.parse_args(); errs=[]
 m=read(a.matching); c=read(a.candidate)
 s1rows=read(os.path.join(a.test_dir,'test_source1.tsv')); s2=ids(os.path.join(a.test_dir,'test_source2.tsv'),'entity_id'); s3=ids(os.path.join(a.test_dir,'test_source3.tsv'),'entity_id'); valid=s2|s3; expected=[r['entity_id'] for r in s1rows]
 if len(m)!=len(expected): errs.append(f'matching row count {len(m)} != test S1 {len(expected)}')
 if len(c)!=len(expected): errs.append(f'candidate row count {len(c)} != test S1 {len(expected)}')
 def check(rows,col,name):
  seen=set()
  for r in rows:
   sid=r.get('source1_entity_id','')
   if sid in seen: errs.append(f'duplicate S1 in {name}: {sid}')
   seen.add(sid)
   vals=[x for x in (r.get(col) or '').split(',') if x]
   if len(vals)!=len(set(vals)): errs.append(f'duplicate IDs for {sid} in {name}')
   bad=[x for x in vals if x not in valid]
   if bad: errs.append(f'invalid IDs for {sid} in {name}: {bad[:5]}')
  if seen!=set(expected): errs.append(f'{name} S1 IDs do not exactly equal test S1 IDs')
 check(m,'matched_entity_ids','matching'); check(c,'candidate_entity_ids','candidate')
 cand={r['source1_entity_id']:set(x for x in (r['candidate_entity_ids'] or '').split(',') if x) for r in c}
 for r in m:
  sid=r['source1_entity_id']; vals=set(x for x in (r['matched_entity_ids'] or '').split(',') if x)
  if not vals.issubset(cand.get(sid,set())): errs.append(f'matches outside candidate set for {sid}')
 if errs:
  print('FAIL'); [print(f'{i}. {e}') for i,e in enumerate(errs,1)]; return 1
 print('PASS: outputs satisfy challenge structural contract'); return 0
if __name__=='__main__': sys.exit(main())
