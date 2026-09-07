
from .dice import WGSDiceLoss 
from .fusion import WeightedFusion 
from .infer import align_output_maps ,logits_from_maps ,wgs_predict_logits 
from .loss import CombinatorialMutationsLossModule 

WGS_DEFAULT_OPS =['add','mul','wf','concat']

WGS_OP_ALIASES ={
'add':'add',
'mult':'mul',
'mul':'mul',
'awf':'wf',
'wf':'wf',
'weighted_fusion':'wf',
'concat':'concat',
}

def resolve_wgs_ops (ops =None ,default =None ,supervision ='wgs'):
    if ops :
        resolved =[]
        for op in ops :
            key =op .lower ().replace ('-','_')
            if key not in WGS_OP_ALIASES :
                valid =sorted (set (WGS_OP_ALIASES .keys ()))
                raise ValueError (
                f'Unknown WGS op `{op}`. Valid aliases: {valid}')
            canonical =WGS_OP_ALIASES [key ]
            if canonical not in resolved :
                resolved .append (canonical )
        return resolved 
    if default is not None :
        return list (default )
    if supervision =='wgs':
        return list (WGS_DEFAULT_OPS )
    return ['add']

__all__ =[
'WGSDiceLoss',
'WeightedFusion',
'CombinatorialMutationsLossModule',
'align_output_maps',
'logits_from_maps',
'wgs_predict_logits',
'WGS_DEFAULT_OPS',
'WGS_OP_ALIASES',
'resolve_wgs_ops',
]
