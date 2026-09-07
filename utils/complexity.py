
import copy 
import time 

import torch 
import torch .nn as nn 

from models import model_output 

class _InferenceWrapper (nn .Module ):

    def __init__ (self ,model ):
        super ().__init__ ()
        self .model =model 

    def forward (self ,x ):
        from models .ladsnet_offset import offset_logits_from_model 
        raw =self .model (x )
        offset_logits =offset_logits_from_model (self .model )
        if offset_logits is not None :
            return offset_logits 
        return model_output (raw )

def _count_params (model ):
    return sum (p .numel ()for p in model .parameters ())

def _profile_thop (model ,inputs ):
    from thop import profile 

    model =copy .deepcopy (model ).eval ()
    with torch .no_grad ():
        macs ,params =profile (model ,inputs =inputs ,verbose =False )
    return int (macs ),int (params )

def _resolve_device (device ):
    if device is None :
        return torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')
    return torch .device (device )

def _sync_device (device ):
    if device .type =='cuda':
        torch .cuda .synchronize (device )

def compute_model_complexity (
model ,
input_h =512 ,
input_w =512 ,
device ='cpu',
inference_only =True ,
):
    try :
        import thop 
    except ImportError as exc :
        raise ImportError (
        'MACs profiling requires `thop`. Install with: pip install thop'
        )from exc 

    device =_resolve_device (device )
    probe =_InferenceWrapper (model )if inference_only else copy .deepcopy (model )
    probe =probe .to (device ).eval ()
    inputs =(torch .randn (1 ,3 ,input_h ,input_w ,device =device ),)

    params =_count_params (model )
    try :
        macs ,thop_params =_profile_thop (probe ,inputs )
        if thop_params >0 :
            params =thop_params 
    except Exception as exc :
        raise RuntimeError (f'MACs profiling failed: {exc}')from exc 

    return {
    'params':params ,
    'macs':macs ,
    'flops':macs *2 ,
    'input_size':(input_h ,input_w ),
    }

def compute_inference_latency (
model ,
input_h =512 ,
input_w =512 ,
device =None ,
warmup =10 ,
repeats =100 ,
inference_only =True ,
):
    device =_resolve_device (device )
    probe =_InferenceWrapper (model )if inference_only else model 
    probe =probe .to (device ).eval ()
    x =torch .randn (1 ,3 ,input_h ,input_w ,device =device )

    with torch .no_grad ():
        for _ in range (warmup ):
            probe (x )
        _sync_device (device )

        timings =[]
        for _ in range (repeats ):
            _sync_device (device )
            start =time .perf_counter ()
            probe (x )
            _sync_device (device )
            timings .append ((time .perf_counter ()-start )*1000.0 )

    timings_tensor =torch .tensor (timings )
    return {
    'latency_ms':float (timings_tensor .mean ().item ()),
    'latency_std_ms':float (timings_tensor .std (unbiased =False ).item ()),
    'latency_device':str (device ),
    'latency_warmup':warmup ,
    'latency_repeats':repeats ,
    'input_size':(input_h ,input_w ),
    }

def compute_model_profile (
model ,
input_h =512 ,
input_w =512 ,
macs_device ='cpu',
latency_device =None ,
inference_only =True ,
warmup =10 ,
repeats =100 ,
):
    stats =compute_model_complexity (
    model ,
    input_h =input_h ,
    input_w =input_w ,
    device =macs_device ,
    inference_only =inference_only ,
    )
    stats .update (compute_inference_latency (
    model ,
    input_h =input_h ,
    input_w =input_w ,
    device =latency_device ,
    warmup =warmup ,
    repeats =repeats ,
    inference_only =inference_only ,
    ))
    return stats 

def format_params (num_params ):
    if num_params >=1e9 :
        return f'{num_params / 1e9:.2f} G'
    if num_params >=1e6 :
        return f'{num_params / 1e6:.2f} M'
    if num_params >=1e3 :
        return f'{num_params / 1e3:.2f} K'
    return str (num_params )

def format_macs (macs ):
    if macs >=1e9 :
        return f'{macs / 1e9:.2f} GMacs'
    if macs >=1e6 :
        return f'{macs / 1e6:.2f} MMacs'
    if macs >=1e3 :
        return f'{macs / 1e3:.2f} KMacs'
    return f'{macs} MACs'

def format_latency (latency_ms ,std_ms =None ):
    if std_ms is None :
        return f'{latency_ms:.2f} ms'
    return f'{latency_ms:.2f} ± {std_ms:.2f} ms'

def format_complexity (stats ):
    h ,w =stats ['input_size']
    parts =[
    f'Input={w}x{h}',
    f'Params={format_params(stats["params"])}',
    f'MACs={format_macs(stats["macs"])}',
    f'FLOPs={format_macs(stats["flops"])}',
    ]
    if 'latency_ms'in stats :
        parts .append (
        f'Latency={format_latency(stats["latency_ms"], stats.get("latency_std_ms"))}'
        f' ({stats.get("latency_device", "cpu")})'
        )
    return ', '.join (parts )

def print_complexity (stats ,prefix =''):
    line =format_complexity (stats )
    print (f'{prefix}{line}'if prefix else line )
    return stats 
