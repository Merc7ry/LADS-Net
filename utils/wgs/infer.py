
import torch 
import torch .nn .functional as F 

P4_MAP_INDEX =0 

def align_output_maps (output_maps ,size ):
    h ,w =size 
    aligned =[]
    for out in output_maps :
        if out .shape [-2 :]!=(h ,w ):
            out =F .interpolate (
            out ,size =(h ,w ),mode ='bilinear',align_corners =False )
        aligned .append (out )
    return aligned 

def model_output_maps (model ,x ):
    outputs =model (x )
    if not isinstance (outputs ,(list ,tuple )):
        return [outputs ]
    if len (outputs )==1 :
        return [outputs [0 ]]
    return align_output_maps (list (outputs ),x .shape [-2 :])

def resolve_infer_mode (mode ,wgs_module =None ):
    if mode in (None ,''):
        return 'wf'if wgs_module is not None else 'p4'
    return mode 

def logits_from_maps (maps ,wgs_module =None ,mode ='wf'):
    if len (maps )==1 :
        return maps [0 ]
    if mode =='p4':
        return maps [P4_MAP_INDEX ]
    if mode =='last':
        return maps [-1 ]
    if mode =='wf'and wgs_module is not None :
        return wgs_module .fuse_all (maps )
    if mode =='wf':
        return torch .stack (maps ,dim =0 ).mean (dim =0 )
    raise ValueError (f'Unknown infer mode: {mode}')

def wgs_predict_logits (model ,x ,wgs_module =None ,mode ='wf'):
    mode =resolve_infer_mode (mode ,wgs_module )
    maps =model_output_maps (model ,x )
    from models .ladsnet_offset import maps_with_final_offset 
    maps =maps_with_final_offset (model ,maps )
    return logits_from_maps (maps ,wgs_module ,mode =mode )
