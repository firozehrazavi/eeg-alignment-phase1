"""python -m phase1.smoke --output <new-directory> (synthetic only)."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch
from torch import nn
import torch.nn.functional as F
from .experiment import paired_train, evaluate


class TinyAligner(nn.Module):
    """Small smoke-only architecture; never a scientific baseline result."""
    def __init__(self):
        super().__init__()
        self.eeg=nn.Sequential(nn.Flatten(),nn.Linear(12,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,8))
        self.proj=nn.Linear(10,8)
        self.log_scale=nn.Parameter(torch.tensor(1.))
    def img(self,targets): return self.proj(targets[0])
    def scale(self): return self.log_scale.exp().clamp(max=100)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(91)
    eeg=torch.randn(3,40,3,4)
    targets=[torch.randn(40,10)]
    ids=[f'train-{i}' for i in range(40)]
    concepts=[f'concept-{i//2}' for i in range(40)]
    runs,enorm,tnorms,split=paired_train(eeg,ids,concepts,targets,seed=0,epochs=3,batch=24,
                                         eval_every=1,model_factory=TinyAligner)
    test=torch.randn(200,3,4); gallery=[torch.randn(200,10)]
    test_ids=[f'test-{i:03d}' for i in range(200)]
    result={'synthetic':True,'scientific_result':False,'torch':torch.__version__,
            'python':platform.python_version(),'seed':0,'split':split,
            'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'arms':{}}
    for name,r in runs.items():
        result['arms'][name]={'history':r['history'],'best_epoch':r['best_epoch'],
            'synthetic_200_way':evaluate(r['model'],test,gallery,enorm,tnorms,test_ids,test_ids)}
    # A second independently constructed run must produce identical histories.
    repeat,_,_,_=paired_train(eeg,ids,concepts,targets,seed=0,epochs=3,batch=24,
                              eval_every=1,model_factory=TinyAligner)
    for name in runs:
        assert repeat[name]['history']==runs[name]['history'], 'Non-reproducible paired run'
    result['repeat_exact']=True
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path('phase1').glob('*.py'))}
    (args.output/'smoke.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'synthetic':True,'repeat_exact':True,'output':str(args.output/'smoke.json')}))


if __name__=='__main__': main()
