
import inspect 

import torch .nn as nn 

from modules .offset_learning import OffsetLearning 

from .ladsnet_pvt_network import LADS_Net 

_BACKBONES ={
'LADS_Net':LADS_Net ,
}

_OFFSET_HEAD_NAME ='out4'

DEFAULT_KERNEL_SIZES =(1 ,3 ,5 )

def resolve_kernel_sizes (kernel_sizes =None ):
    if kernel_sizes is None :
        return DEFAULT_KERNEL_SIZES 
    return tuple (kernel_sizes )

def _offset_learning_ctor (num_classes ,embed_dims ,use_cls_offset =True ,use_feat_offset =True ):
    params =inspect .signature (OffsetLearning .__init__ ).parameters 
    kwargs ={}
    if 'use_cls_offset'in params :
        kwargs ['use_cls_offset']=use_cls_offset 
    if 'use_feat_offset'in params :
        kwargs ['use_feat_offset']=use_feat_offset 
    return OffsetLearning (num_classes ,embed_dims ,**kwargs )

class PlainOffsetHead (nn .Module ):

    def __init__ (
    self ,
    num_classes ,
    embed_dims ,
    use_cls_offset =True ,
    use_feat_offset =True ,
    ):
        super ().__init__ ()
        self .plain =nn .Conv2d (embed_dims ,num_classes ,kernel_size =1 )
        self .offset =_offset_learning_ctor (
        num_classes ,
        embed_dims ,
        use_cls_offset =use_cls_offset ,
        use_feat_offset =use_feat_offset ,
        )
        self ._feat =None 

    @property 
    def in_channels (self ):
        return self .plain .in_channels 

    def forward (self ,x ):
        self ._feat =x 
        return self .plain (x )

    def forward_offset (self ,x =None ):
        feat =self ._feat if x is None else x 
        if feat is None :
            raise RuntimeError (
            'PlainOffsetHead.forward must run before forward_offset()')
        return self .offset (feat )

def find_plain_offset_head (model ):
    head =getattr (model ,_OFFSET_HEAD_NAME ,None )
    return head if isinstance (head ,PlainOffsetHead )else None 

def offset_logits_from_model (model ):
    head =find_plain_offset_head (model )
    if head is None :
        return None 
    return head .forward_offset ()

def maps_with_final_offset (model ,maps ):
    offset_p4 =offset_logits_from_model (model )
    if offset_p4 is None :
        return maps 
    maps =list (maps )
    maps [0 ]=offset_p4 
    return maps 

def _attach_offset_head (
backbone ,
num_classes ,
use_cls_offset =True ,
use_feat_offset =True ,
):
    head =getattr (backbone ,_OFFSET_HEAD_NAME )
    embed_dims =head .in_channels 
    setattr (
    backbone ,
    _OFFSET_HEAD_NAME ,
    PlainOffsetHead (
    num_classes ,
    embed_dims ,
    use_cls_offset =use_cls_offset ,
    use_feat_offset =use_feat_offset ,
    ),
    )
    return backbone 

class LADSNetOffsetOutput :
    @staticmethod 
    def build (
    name ,
    num_classes =8 ,
    use_offset =True ,
    use_cls_offset =True ,
    use_feat_offset =True ,
    kernel_sizes =None ,
    simple_decoder =False ,
    use_gag =True ,
    use_mkira =True ,
    **kwargs ,
    ):
        if name not in _BACKBONES :
            raise ValueError (f'Unknown model `{name}`. Available: {list(_BACKBONES)}')
        resolved_kernels =resolve_kernel_sizes (kernel_sizes )
        backbone =_BACKBONES [name ](
        num_classes =num_classes ,
        kernel_sizes =resolved_kernels ,
        simple_decoder =simple_decoder ,
        use_gag =use_gag ,
        use_mkira =use_mkira ,
        **kwargs ,
        )
        if use_offset :
            _attach_offset_head (
            backbone ,
            num_classes ,
            use_cls_offset =use_cls_offset ,
            use_feat_offset =use_feat_offset ,
            )
        return backbone 
