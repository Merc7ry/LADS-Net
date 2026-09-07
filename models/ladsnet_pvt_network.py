
import os 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 

from .ladsnet_network import (
ChannelAttention ,
GroupedAttentionGate ,
SpatialAttention ,
mk_irb_bottleneck ,
)
from .pvtv2 import (
pvt_v2_b0 ,
pvt_v2_b1 ,
pvt_v2_b2 ,
pvt_v2_b3 ,
pvt_v2_b4 ,
pvt_v2_b5 ,
)
from .pretrain_paths import resolve_pretrain_root 

PVT_ENCODER_CONFIG ={
'pvt_v2_b0':{
'factory':pvt_v2_b0 ,
'embed_dims':[32 ,64 ,160 ,256 ],
'pretrain_file':'pvt_v2_b0.pth',
},
'pvt_v2_b1':{
'factory':pvt_v2_b1 ,
'embed_dims':[64 ,128 ,320 ,512 ],
'pretrain_file':'pvt_v2_b1.pth',
},
'pvt_v2_b2':{
'factory':pvt_v2_b2 ,
'embed_dims':[64 ,128 ,320 ,512 ],
'pretrain_file':'pvt_v2_b2.pth',
},
'pvt_v2_b3':{
'factory':pvt_v2_b3 ,
'embed_dims':[64 ,128 ,320 ,512 ],
'pretrain_file':'pvt_v2_b3.pth',
},
'pvt_v2_b4':{
'factory':pvt_v2_b4 ,
'embed_dims':[64 ,128 ,320 ,512 ],
'pretrain_file':'pvt_v2_b4.pth',
},
'pvt_v2_b5':{
'factory':pvt_v2_b5 ,
'embed_dims':[64 ,128 ,320 ,512 ],
'pretrain_file':'pvt_v2_b5.pth',
},
}

def _decoder_channels (embed_dims ):
    c1 ,c2 ,c3 ,c4 =embed_dims 
    return [c1 ,c1 ,c2 ,c3 ,c4 ]

def _default_pretrain_root():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(resolve_pretrain_root(), 'pvt'),
        os.path.join(here, '..', 'pretrained_pth', 'pvt'),
    ]
    for path in candidates:
        norm = os.path.normpath(path)
        if os.path.isdir(norm):
            return norm
    return os.path.normpath(candidates[0])

def _load_pvt_pretrained (backbone ,encoder_name ,pretrain_path =None ):
    cfg =PVT_ENCODER_CONFIG [encoder_name ]
    if pretrain_path is None :
        pretrain_path =os .path .join (_default_pretrain_root (),cfg ['pretrain_file'])
    if not os .path .isfile (pretrain_path ):
        print (f'Warning: PVT pretrained weights not found: {pretrain_path}')
        return 
    state =torch .load (pretrain_path ,map_location ='cpu')
    model_dict =backbone .state_dict ()
    filtered ={k :v for k ,v in state .items ()if k in model_dict }
    model_dict .update (filtered )
    backbone .load_state_dict (model_dict ,strict =False )
    print (f'Loaded PVT pretrained weights: {pretrain_path} ({len(filtered)} keys)')

def _simple_conv_block (in_ch ,out_ch ):
    return nn .Sequential (
    nn .Conv2d (in_ch ,out_ch ,kernel_size =3 ,padding =1 ,bias =False ),
    nn .BatchNorm2d (out_ch ),
    nn .ReLU (inplace =True ),
    )

class LADS_Net (nn .Module ):

    def __init__ (
    self ,
    num_classes =8 ,
    encoder ='pvt_v2_b2',
    pretrain =True ,
    pretrain_path =None ,
    kernel_sizes =(1 ,3 ,5 ),
    expansion_factor =2 ,
    gag_kernel =3 ,
    simple_decoder =False ,
    use_gag =True ,
    use_mkira =True ,
    **kwargs ,
    ):
        super ().__init__ ()
        if encoder not in PVT_ENCODER_CONFIG :
            raise ValueError (
            f'Unknown encoder `{encoder}`. '
            f'Available: {list(PVT_ENCODER_CONFIG)}')

        cfg =PVT_ENCODER_CONFIG [encoder ]
        self .encoder_name =encoder 
        self .simple_decoder =simple_decoder 
        self .use_gag =bool (use_gag )and not simple_decoder 
        self .use_mkira =bool (use_mkira )and not simple_decoder 
        self .encoder =cfg ['factory']()
        channels =_decoder_channels (cfg ['embed_dims'])

        if pretrain :
            _load_pvt_pretrained (self .encoder ,encoder ,pretrain_path )

        if simple_decoder :
            self .decoder1 =_simple_conv_block (channels [4 ],channels [3 ])
            self .decoder2 =_simple_conv_block (channels [3 ],channels [2 ])
            self .decoder3 =_simple_conv_block (channels [2 ],channels [1 ])
            self .decoder4 =_simple_conv_block (channels [1 ],channels [0 ])
            self .decoder5 =_simple_conv_block (channels [0 ],channels [0 ])
        else :
            if self .use_gag :
                self .AG1 =GroupedAttentionGate (
                F_g =channels [3 ],F_l =channels [3 ],F_int =channels [3 ]//2 ,
                kernel_size =gag_kernel ,groups =max (channels [3 ]//2 ,1 ))
                self .AG2 =GroupedAttentionGate (
                F_g =channels [2 ],F_l =channels [2 ],F_int =channels [2 ]//2 ,
                kernel_size =gag_kernel ,groups =max (channels [2 ]//2 ,1 ))
                self .AG3 =GroupedAttentionGate (
                F_g =channels [1 ],F_l =channels [1 ],F_int =channels [1 ]//2 ,
                kernel_size =gag_kernel ,groups =max (channels [1 ]//2 ,1 ))
                self .AG4 =GroupedAttentionGate (
                F_g =channels [0 ],F_l =channels [0 ],F_int =max (channels [0 ]//2 ,1 ),
                kernel_size =gag_kernel ,groups =max (channels [0 ]//2 ,1 ))

            self .decoder1 =mk_irb_bottleneck (
            channels [4 ],channels [3 ],1 ,1 ,
            expansion_factor =expansion_factor ,dw_parallel =True ,add =True ,
            kernel_sizes =kernel_sizes )
            self .decoder2 =mk_irb_bottleneck (
            channels [3 ],channels [2 ],1 ,1 ,
            expansion_factor =expansion_factor ,dw_parallel =True ,add =True ,
            kernel_sizes =kernel_sizes )
            self .decoder3 =mk_irb_bottleneck (
            channels [2 ],channels [1 ],1 ,1 ,
            expansion_factor =expansion_factor ,dw_parallel =True ,add =True ,
            kernel_sizes =kernel_sizes )
            self .decoder4 =mk_irb_bottleneck (
            channels [1 ],channels [0 ],1 ,1 ,
            expansion_factor =expansion_factor ,dw_parallel =True ,add =True ,
            kernel_sizes =kernel_sizes )
            self .decoder5 =mk_irb_bottleneck (
            channels [0 ],channels [0 ],1 ,1 ,
            expansion_factor =expansion_factor ,dw_parallel =True ,add =True ,
            kernel_sizes =kernel_sizes )

            if self .use_mkira :
                self .CA1 =ChannelAttention (channels [4 ],ratio =16 )
                self .CA2 =ChannelAttention (channels [3 ],ratio =16 )
                self .CA3 =ChannelAttention (channels [2 ],ratio =16 )
                self .CA4 =ChannelAttention (channels [1 ],ratio =8 )
                self .CA5 =ChannelAttention (channels [0 ],ratio =4 )
                self .SA =SpatialAttention ()

        self .out1 =nn .Conv2d (channels [2 ],num_classes ,kernel_size =1 )
        self .out2 =nn .Conv2d (channels [1 ],num_classes ,kernel_size =1 )
        self .out3 =nn .Conv2d (channels [0 ],num_classes ,kernel_size =1 )
        self .out4 =nn .Conv2d (channels [0 ],num_classes ,kernel_size =1 )

        self .channels =channels 
        if simple_decoder :
            decoder_type ='simple-upsampling'
        else :
            parts =['MKIR'if tuple (kernel_sizes )!=(3 ,)else 'IRB']
            if self .use_gag :
                parts .append ('GAG')
            if self .use_mkira :
                parts .append ('MKIRA')
            decoder_type ='+'.join (parts )
        print (
        f'LADS_Net created with encoder={encoder}, channels={channels}, '
        f'decoder={decoder_type}, kernel_sizes={tuple(kernel_sizes)}')

    def _forward_simple (self ,x1 ,x2 ,x3 ,x4 ):
        t1 =F .interpolate (x1 ,scale_factor =2 ,mode ='bilinear',align_corners =False )
        t2 =x1 
        t3 =x2 
        t4 =x3 
        out =x4 

        out =F .relu (F .interpolate (
        self .decoder1 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        out =torch .add (out ,t4 )

        out =F .relu (F .interpolate (
        self .decoder2 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        p1 =F .interpolate (self .out1 (out ),scale_factor =(8 ,8 ),mode ='bilinear',align_corners =False )
        out =torch .add (out ,t3 )

        out =F .relu (F .interpolate (
        self .decoder3 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        p2 =F .interpolate (self .out2 (out ),scale_factor =(4 ,4 ),mode ='bilinear',align_corners =False )
        out =torch .add (out ,t2 )

        out =F .relu (F .interpolate (
        self .decoder4 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        p3 =F .interpolate (self .out3 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False )
        out =torch .add (out ,t1 )

        out =F .relu (F .interpolate (
        self .decoder5 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        p4 =self .out4 (out )
        return [p4 ,p3 ,p2 ,p1 ]

    def _apply_mkira (self ,out ,ca ):
        if not self .use_mkira :
            return out 
        out =ca (out )*out 
        out =self .SA (out )*out 
        return out 

    def _apply_gag_add (self ,out ,skip ,ag ):
        if self .use_gag :
            skip =ag (g =out ,x =skip )
        return torch .add (out ,skip )

    def _forward_mk_decoder (self ,x1 ,x2 ,x3 ,x4 ):
        t1 =F .interpolate (x1 ,scale_factor =2 ,mode ='bilinear',align_corners =False )
        t2 =x1 
        t3 =x2 
        t4 =x3 
        out =x4 

        out =self ._apply_mkira (out ,self .CA1 if self .use_mkira else None )
        out =F .relu (F .interpolate (
        self .decoder1 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        out =self ._apply_gag_add (out ,t4 ,self .AG1 if self .use_gag else None )

        out =self ._apply_mkira (out ,self .CA2 if self .use_mkira else None )
        out =F .relu (F .interpolate (
        self .decoder2 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        p1 =F .interpolate (self .out1 (out ),scale_factor =(8 ,8 ),mode ='bilinear',align_corners =False )
        out =self ._apply_gag_add (out ,t3 ,self .AG2 if self .use_gag else None )

        out =self ._apply_mkira (out ,self .CA3 if self .use_mkira else None )
        out =F .relu (F .interpolate (
        self .decoder3 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        p2 =F .interpolate (self .out2 (out ),scale_factor =(4 ,4 ),mode ='bilinear',align_corners =False )
        out =self ._apply_gag_add (out ,t2 ,self .AG3 if self .use_gag else None )

        out =self ._apply_mkira (out ,self .CA4 if self .use_mkira else None )
        out =F .relu (F .interpolate (
        self .decoder4 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        p3 =F .interpolate (self .out3 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False )
        out =self ._apply_gag_add (out ,t1 ,self .AG4 if self .use_gag else None )

        out =self ._apply_mkira (out ,self .CA5 if self .use_mkira else None )
        out =F .relu (F .interpolate (
        self .decoder5 (out ),scale_factor =(2 ,2 ),mode ='bilinear',align_corners =False ))
        p4 =self .out4 (out )
        return [p4 ,p3 ,p2 ,p1 ]

    def forward (self ,x ):
        if x .shape [1 ]==1 :
            x =x .repeat (1 ,3 ,1 ,1 )

        x1 ,x2 ,x3 ,x4 =self .encoder (x )
        if self .simple_decoder :
            return self ._forward_simple (x1 ,x2 ,x3 ,x4 )
        return self ._forward_mk_decoder (x1 ,x2 ,x3 ,x4 )
