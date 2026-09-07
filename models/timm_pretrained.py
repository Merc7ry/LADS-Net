
import os 

import torch 

def _extract_state_dict (checkpoint ):
    if not isinstance (checkpoint ,dict ):
        return checkpoint 
    for key in ('state_dict','model','model_state_dict'):
        if key in checkpoint and isinstance (checkpoint [key ],dict ):
            return checkpoint [key ]
    return checkpoint 

def _strip_prefix (state_dict ,prefix ):
    if not prefix :
        return state_dict 
    if not all (k .startswith (prefix )for k in state_dict ):
        return state_dict 
    return {k [len (prefix ):]:v for k ,v in state_dict .items ()}

def load_timm_checkpoint (model ,checkpoint_path ,strict =False ):
    if not checkpoint_path :
        raise ValueError ('checkpoint_path is required')
    if not os .path .isfile (checkpoint_path ):
        raise FileNotFoundError (f'Pretrained checkpoint not found: {checkpoint_path}')

    checkpoint =torch .load (checkpoint_path ,map_location ='cpu',weights_only =False )
    state_dict =_extract_state_dict (checkpoint )
    state_dict =_strip_prefix (state_dict ,'module.')
    state_dict =_strip_prefix (state_dict ,'backbone.')

    try :
        from timm .models import load_checkpoint 
        load_checkpoint (model ,checkpoint_path ,strict =strict )
        return 
    except Exception :
        pass 

    missing ,unexpected =model .load_state_dict (state_dict ,strict =strict )
    if missing and strict :
        raise RuntimeError (
        f'Failed to load `{checkpoint_path}`: missing keys={missing}, '
        f'unexpected keys={unexpected}')
