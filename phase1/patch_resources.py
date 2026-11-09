"""Runtime memory telemetry and CU estimates, never an authoritative billing API."""
import json
import os
import time
import gc
from pathlib import Path
import torch


class BudgetExceeded(RuntimeError):
    pass


def runtime_profile():
    if not torch.cuda.is_available():
        return dict(device='cpu',precision='fp32',adapter_chunk=2,extraction_batch=1)
    free,total=torch.cuda.mem_get_info()
    return dict(device='cuda',gpu=torch.cuda.get_device_name(),total_vram_gib=total/2**30,
                free_vram_gib=free/2**30,precision='bf16' if torch.cuda.is_bf16_supported() else 'fp16',
                adapter_chunk=8 if free>=20*2**30 else 4 if free>=12*2**30 else 1,
                extraction_batch=1)


def estimate_cu(seconds, rate):
    if seconds<0 or rate<=0:
        raise ValueError('Need nonnegative duration and positive user-observed CU/hour')
    return seconds/3600*rate


def retry_oom(operation, chunk):
    """Retry smaller adapter chunks; never alter contrastive batch or precision."""
    while True:
        failed=False
        try:
            return operation(chunk),chunk
        except torch.cuda.OutOfMemoryError:
            if chunk<=1:
                raise RuntimeError('OOM at chunk=1. Checkpoint retained; use more VRAM or a new matched config.') from None
            failed=True
        if failed:
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            chunk=max(1,chunk//2)
            print(f'CUDA OOM: retrying adapter chunk={chunk}; contrastive batch unchanged',flush=True)


class BudgetMeter:
    """Shared notebook-session clock via env; cap is an estimate, not billing control.

    Checks at batch boundaries; a notebook supervisor can terminate on wall cap.
    PHASE1_CU_RATE must come from Resources, never inferred from GPU name.
    """
    def __init__(self):
        self.start=float(os.environ.get('PHASE1_SESSION_START',time.time()))
        self.rate=float(os.environ.get('PHASE1_CU_RATE','0'))
        self.limit=float(os.environ.get('PHASE1_CU_LIMIT','0'))
        self.seconds_limit=float(os.environ.get('PHASE1_SECONDS_LIMIT','0'))
        self.prior=float(os.environ.get('PHASE1_CU_PRIOR','0'))
        self.log=os.environ.get('PHASE1_RESOURCE_LOG')
        self.last=0.

    def check(self, stage, force=False):
        seconds=max(0.,time.time()-self.start)
        spent=self.prior+estimate_cu(seconds,self.rate) if self.rate>0 else None
        row=dict(timestamp=time.time(),stage=stage,session_seconds=seconds,
                 estimated_cu=spent,observed_cu_per_hour=self.rate or None)
        if torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info()
            row.update(gpu=torch.cuda.get_device_name(),free_vram_gib=free/2**30,
                       allocated_gib=torch.cuda.memory_allocated()/2**30,
                       reserved_gib=torch.cuda.memory_reserved()/2**30,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        if force or time.time()-self.last>=60:
            print(json.dumps(row),flush=True)
            if self.log:
                Path(self.log).parent.mkdir(parents=True,exist_ok=True)
                with open(self.log,'a') as f:f.write(json.dumps(row)+'\n')
            self.last=time.time()
        if self.limit and spent is not None and spent>=self.limit:
            raise BudgetExceeded('Estimated pilot CU cap reached; resume only after reviewing budget')
        if self.seconds_limit and seconds>=self.seconds_limit:
            raise BudgetExceeded('Pilot wall-time cap reached; checkpoints are retained')
        return row
