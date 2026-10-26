"""Development-only official FLA activation in the existing allocated process.

No installed files or model weights are edited. The two Transformers functions
are re-decorated with the SAME upstream fallback decorator after an isolated
package directory is made available. Final arms must share this same runtime.
"""
import importlib
import inspect
import functools
import json
import os
from pathlib import Path
import sys
import time

STATE = None


def reference_wrapper(function):
    """Force the actual undecorated Torch reference, even in a fresh FLA process."""
    raw=inspect.unwrap(function)
    parameters=tuple(inspect.signature(raw).parameters)
    @functools.wraps(raw)
    def call(*args,**kwargs):
        return raw(*args,**{k:v for k,v in kwargs.items() if k in parameters})
    return call


def prepare():
    global STATE
    if STATE is not None:
        return STATE
    import torch
    target=Path(os.environ.get('RADZH_V3_ACCEL_PATH','/home/Student/s4905914/radiology-plain-zh/v3/acceleration/python'))
    if not target.is_dir():raise ValueError('Isolated acceleration packages are missing')
    sys.path.insert(0,str(target));importlib.invalidate_caches()
    from importlib.metadata import version
    expected={'fla-core':'0.5.2','flash-linear-attention':'0.5.2','einops':'0.8.2'}
    if any(version(k)!=v for k,v in expected.items()):raise ValueError('Acceleration dependency pin mismatch')
    import fla.ops.gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    from transformers.integrations import use_kernel_func_from_hub_with_fallback
    names={'torch_chunk_gated_delta_rule':'chunk_gated_delta_rule',
           'torch_recurrent_gated_delta_rule':'fused_recurrent_gated_delta_rule'}
    original={name:getattr(module,name) for name in names}
    reference={name:reference_wrapper(function) for name,function in original.items()}
    fast={}
    implementations={}
    def fla_implementation(function,seen=None):
        seen=set() if seen is None else seen
        if id(function) in seen:return None
        seen.add(id(function))
        if getattr(function,'__module__','').startswith('fla.'):
            return function.__module__+'.'+function.__name__
        for cell in getattr(function,'__closure__',None) or ():
            value=cell.cell_contents
            if inspect.isfunction(value):
                found=fla_implementation(value,seen)
                if found:return found
        return None
    for name,kernel in names.items():
        raw=inspect.unwrap(original[name])
        if raw.__code__.co_name!=name:raise ValueError('Cannot identify the upstream reference function')
        fast[name]=use_kernel_func_from_hub_with_fallback(kernel,'fla')(raw)
        implementations[name]=fla_implementation(fast[name])
        if not implementations[name]:raise ValueError('The new wrapper did not select an FLA implementation')
    g=torch.Generator(device='cuda').manual_seed(37102603)
    def sample(t):
        q,k,v=[torch.randn((1,t,4,128),device='cuda',dtype=torch.bfloat16,generator=g)*.1 for _ in range(3)]
        gate=-torch.rand((1,t,4),device='cuda',generator=g)
        beta=torch.rand((1,t,4),device='cuda',dtype=torch.bfloat16,generator=g)
        return (q,k,v),dict(g=gate,beta=beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
    checks={}
    for name,t in [('torch_chunk_gated_delta_rule',16),('torch_recurrent_gated_delta_rule',1)]:
        args,kwargs=sample(t)
        expected_output=reference[name](*args,**kwargs)
        accelerated=fast[name](*args,**kwargs)
        values=[]
        for a,b in zip(expected_output,accelerated):
            if a is None or b is None:raise ValueError('Kernel did not return required state')
            if a.shape!=b.shape or not torch.isfinite(b).all():raise ValueError('Kernel shape/nonfinite failure')
            diff=(a.float()-b.float())
            relative=(diff.square().mean().sqrt()/(a.float().square().mean().sqrt()+1e-8)).item()
            maximum=diff.abs().max().item()
            if relative>.05 or maximum>.05:raise ValueError('Kernel numerical probe exceeds predeclared tolerance')
            values.append({'relative_rms_error':relative,'max_abs_error':maximum})
        checks[name]=values
    STATE={'module':module,'original':original,'reference':reference,'fast':fast,'checks':checks,'packages':expected,'implementations':implementations}
    return STATE


def activate(state,fast):
    for name,function in state['fast' if fast else 'reference'].items():setattr(state['module'],name,function)


def explain(report,generate):
    start=time.perf_counter()
    state=prepare()
    prompt=[{'role':'system','content':'把影像报告忠实翻译成简体中文，只输出译文，保留数字、否定和不确定性。'},
            {'role':'user','content':report}]
    try:
        activate(state,False)
        reference=generate(prompt,max_tokens=128)
        activate(state,True)
        accelerated=generate(prompt,max_tokens=128)
    except Exception:
        for name,function in state['original'].items():setattr(state['module'],name,function)
        raise
    note={'packages':state['packages'],'numerical_probe':state['checks'],'implementations':state['implementations'],
          'scope':'Two upstream Gated Delta Rule functions, reference decorator rebuilt with isolated FLA; causal convolution remains the original fallback.',
          'same_model_weights':True,'both_final_arms_must_share_runtime':True}
    Path('acceleration_activation.json').write_text(json.dumps(note,indent=2)+'\n')
    return {'version':'fla-runtime-development-probe','source_report':report,'reference':reference,'accelerated':accelerated,
            'same_output':reference['text']==accelerated['text'],'activation':note,'seconds':time.perf_counter()-start}
