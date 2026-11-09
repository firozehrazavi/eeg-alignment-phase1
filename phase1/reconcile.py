"""Read-only CSV reconciliation. python -m phase1.reconcile --output new.json"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def reconcile(root):
    files=['results_loso_v6b.csv','results_within_v6b.csv','partA_capacity.csv',
           'comparison_v6b.csv','landmark_rules_summary.csv','partC_confidence.csv']
    data={f:list(csv.DictReader((root/f).open(encoding='utf8'))) for f in files}
    def mean(rows,column): return statistics.mean(float(r[column]) for r in rows)
    loso=[r for r in data[files[0]] if r['config']=='route_c']
    within=[r for r in data[files[1]] if r['config']=='route_c']
    if len(loso)!=10 or len({r['test_subject'] for r in loso})!=10:
        raise ValueError('Expected ten distinct LOSO subjects')
    if len(within)!=30 or len({(r['subject'],r['seed']) for r in within})!=30:
        raise ValueError('Expected thirty unique within subject/seed rows')
    capacity={}
    for name in ['score n=64','score n=128','score n=all']:
        rows=[r for r in data[files[2]] if r['setting']==name]
        if len(rows)!=10 or len({r['subject'] for r in rows})!=10:
            raise ValueError('Incomplete landmark sweep')
        capacity[name]={c:mean(rows,c) for c in ['top1','top5']}
    return {'historical_only':True,'strict_inductive_certified':False,
        'hashes':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
        'within_route_c':{c:mean(within,c) for c in ['top1','top5']},
        'loso_route_c':{c:mean(loso,c) for c in ['plain_top1','plain_top5','csls_top1','csls_top5','recovery_top1','recovery_top5']},
        'capacity':capacity,
        'resolution':'44.7 = 64 landmarks; 46.45 = 128/all landmarks; selection provenance not certified',
        'missing_calibration_csvs':[f for f in ['calibration_v9.csv','calibration_v9b.csv'] if not (root/f).exists()]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=reconcile(a.root)
    with a.output.open('x',encoding='utf8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(result['resolution'])


if __name__=='__main__': main()
