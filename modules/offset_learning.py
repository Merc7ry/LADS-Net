
import torch 
import torch .nn as nn 

def _trunc_normal_ (tensor ,std =0.02 ):
    nn .init .trunc_normal_ (tensor ,std =std )

class OffsetLearning (nn .Module ):

    def __init__ (
    self ,
    num_classes ,
    embed_dims ,
    init_std =0.02 ,
    use_cls_offset =True ,
    use_feat_offset =True ,
    ):
        super ().__init__ ()
        self .num_classes =num_classes 
        self .use_cls_offset =use_cls_offset 
        self .use_feat_offset =use_feat_offset 
        self .cls_repr =nn .Parameter (torch .randn (1 ,num_classes ,embed_dims ))
        self .init_std =init_std 
        self .mask_norm =nn .LayerNorm (num_classes )
        self .cls_offset_proj =(
        nn .Linear (embed_dims ,embed_dims ,bias =False )
        if use_cls_offset else None )
        self .feat_offset_proj =(
        nn .Linear (embed_dims ,embed_dims ,bias =False )
        if use_feat_offset else None )
        self .init_weights ()

    def init_weights (self ):
        _trunc_normal_ (self .cls_repr ,std =self .init_std )
        for proj in (self .cls_offset_proj ,self .feat_offset_proj ):
            if proj is not None :
                _trunc_normal_ (proj .weight ,std =self .init_std )

    def forward (self ,x ):
        b ,c ,h ,w =x .shape 
        cls_repr =self .cls_repr .expand (b ,-1 ,-1 )
        img_feat =x .permute (0 ,2 ,3 ,1 ).contiguous ().view (b ,h *w ,c )

        coupled_attn =(img_feat @cls_repr .transpose (1 ,2 )).permute (0 ,2 ,1 )

        if self .use_cls_offset :
            cls_attn =coupled_attn .softmax (dim =2 )
            cls_offset =self .cls_offset_proj (cls_attn @img_feat )
            aligned_cls_repr =cls_repr +cls_offset 
        else :
            aligned_cls_repr =cls_repr 

        if self .use_feat_offset :
            pos_attn =coupled_attn .softmax (dim =1 )
            feat_offset =self .feat_offset_proj (pos_attn .transpose (1 ,2 )@cls_repr )
            aligned_img_feat =img_feat +feat_offset 
        else :
            aligned_img_feat =img_feat 

        masks =aligned_img_feat @aligned_cls_repr .transpose (1 ,2 )
        masks =self .mask_norm (masks )
        return masks .permute (0 ,2 ,1 ).contiguous ().view (b ,-1 ,h ,w )
