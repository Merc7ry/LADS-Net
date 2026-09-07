
import torch 

from datasets .uavid_base import IMAGENET_MEAN ,IMAGENET_STD 
from models import model_output 

_MEAN =torch .tensor (IMAGENET_MEAN ,dtype =torch .float32 ).view (3 ,1 ,1 )
_STD =torch .tensor (IMAGENET_STD ,dtype =torch .float32 ).view (3 ,1 ,1 )

def slide_window_starts (length ,crop ,stride ):
    if length <=crop :
        return [0 ]
    starts =list (range (0 ,length -crop +1 ,stride ))
    last =length -crop 
    if starts [-1 ]!=last :
        starts .append (last )
    return sorted (set (starts ))

def _normalize_patch (patch ,device ):
    mean =_MEAN .to (device )
    std =_STD .to (device )
    return (patch -mean )/std 

@torch .no_grad ()
def slide_predict_logits (model ,image ,crop_size ,stride ,device ,wgs_module =None ,
wgs_infer_mode ='wf'):
    from utils .wgs import wgs_predict_logits 

    model .eval ()
    if wgs_module is not None :
        wgs_module .eval ()

    _ ,height ,width =image .shape 
    crop =crop_size 
    image =image .to (device )
    orig_h ,orig_w =height ,width 

    if height <crop or width <crop :
        pad_h =max (crop -height ,0 )
        pad_w =max (crop -width ,0 )
        image =torch .nn .functional .pad (image ,(0 ,pad_w ,0 ,pad_h ))
        height ,width =image .shape [1 ],image .shape [2 ]

    logits_sum =None 
    count =torch .zeros ((height ,width ),device =device ,dtype =torch .float32 )

    for y in slide_window_starts (height ,crop ,stride ):
        for x in slide_window_starts (width ,crop ,stride ):
            patch =image [:,y :y +crop ,x :x +crop ]
            patch_in =_normalize_patch (patch ,device ).unsqueeze (0 )
            if wgs_module is not None :
                logits =wgs_predict_logits (
                model ,patch_in ,wgs_module ,mode =wgs_infer_mode )
            else :
                raw =model (patch_in )
                from models .ladsnet_offset import offset_logits_from_model 
                offset_logits =offset_logits_from_model (model )
                logits =(
                offset_logits if offset_logits is not None 
                else model_output (raw ))
            if logits_sum is None :
                num_classes =logits .shape [1 ]
                logits_sum =torch .zeros (
                (num_classes ,height ,width ),device =device ,dtype =torch .float32 )
            logits_sum [:,y :y +crop ,x :x +crop ]+=logits [0 ]
            count [y :y +crop ,x :x +crop ]+=1.0 

    logits_sum =logits_sum /count .clamp (min =1.0 ).unsqueeze (0 )
    return logits_sum [:,:orig_h ,:orig_w ]

@torch .no_grad ()
def slide_predict (model ,image ,crop_size ,stride ,device ,wgs_module =None ,
wgs_infer_mode ='wf'):
    logits =slide_predict_logits (
    model ,image ,crop_size ,stride ,device ,
    wgs_module =wgs_module ,wgs_infer_mode =wgs_infer_mode )
    return logits .argmax (dim =0 ).cpu ().numpy ().astype ('uint8')
