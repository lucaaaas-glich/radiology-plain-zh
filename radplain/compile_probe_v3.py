"""Bounded OLD-report runtime experiment; restore all settings on exit.

The existing worker's generate callback closes over its already-loaded HFBackend.
This adapter reads that known, owned callback contract to avoid a second model
allocation. It is not a general callback adapter or a final inference setting.
"""
import copy
import inspect
import signal
import time
from .kernel_probe_v3 import prepare, activate


def backend_from_worker(generate):
    backend=inspect.getclosurevars(generate).nonlocals.get('model')
    if type(backend).__name__!='HFBackend' or not backend.load_in_4bit:
        raise ValueError('Expected the pinned V3 worker HFBackend closure')
    return backend


def explain(report,generate):
    from transformers import CompileConfig
    import torch
    activate(prepare(),True)
    backend=backend_from_worker(generate)
    model=backend.model
    original=copy.deepcopy(model.generation_config)
    prompt=[{'role':'system','content':'忠实翻译这份影像报告为简体中文，保留部位、数字、否定、可能性和时间。只输出译文。'},
            {'role':'user','content':report}]
    results={}
    start=time.perf_counter()
    def timeout(*args):raise TimeoutError('Static compile probe exceeded 240 seconds')
    old_handler=signal.signal(signal.SIGALRM,timeout)
    try:
        results['dynamic']=generate(prompt,max_tokens=96)
        model.generation_config.cache_implementation='static'
        model.generation_config.max_cache_len=12288
        model.generation_config.compile_config=CompileConfig(fullgraph=True,mode='reduce-overhead')
        signal.alarm(240)
        results['static_cold']=generate(prompt,max_tokens=96)
        results['static_warm']=generate(prompt,max_tokens=96)
        signal.alarm(0)
        results['same_output']=len({results[k]['text'] for k in ['dynamic','static_cold','static_warm']})==1
    except Exception as error:
        results['probe_error']=type(error).__name__+': '+str(error)[:1000]
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        model.generation_config=original
        # Static buffers and compiled graph can be reclaimed; keep model weights.
        if hasattr(model,'_cache'):del model._cache
        if hasattr(model,'_compiled_call'):del model._compiled_call
        torch._dynamo.reset();torch.cuda.empty_cache()
    return dict(results,version='static-compile-development-probe',seconds=time.perf_counter()-start,
                source_report=report,restored_original_generation_config=True)
