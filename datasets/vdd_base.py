
import os 

import cv2 
import numpy as np 
from torch .utils .data import Dataset 

from .uavid_base import (
IMAGE_EXTENSIONS ,
MASK_EXTENSIONS ,
UAVID_PATCH_SIZE ,
_ensure_patch_size ,
get_uavid_train_transform ,
get_uavid_val_transform ,
)

VDD_PATCH_SIZE =UAVID_PATCH_SIZE 
VDD_NATIVE_W =4000 
VDD_NATIVE_H =3000 
VDD_NUM_CLASSES =7 
VDD_BG_CLASS =0 

VDD_CLASSES =[
'Other',
'Wall',
'Road',
'Vegetation',
'Vehicle',
'Roof',
'Water',
]

VDD_PALETTE =[
(128 ,128 ,128 ),
(128 ,0 ,0 ),
(128 ,64 ,128 ),
(0 ,128 ,0 ),
(64 ,0 ,128 ),
(192 ,128 ,64 ),
(0 ,128 ,128 ),
]

get_vdd_train_transform =get_uavid_train_transform 
get_vdd_val_transform =get_uavid_val_transform 

def decode_vdd_mask (mask ,ignore_index =255 ):
    mask =np .asarray (mask )
    if mask .ndim ==3 :
        if mask .shape [2 ]==1 :
            mask =mask [:,:,0 ]
        elif np .all (mask [:,:,0 ]==mask [:,:,1 ])and np .all (mask [:,:,1 ]==mask [:,:,2 ]):
            mask =mask [:,:,0 ]
        else :
            raise ValueError (
            f'VDD masks are expected to be index PNG, got RGB shape {mask.shape}')

    idx =mask .astype (np .int32 )
    out =np .full (idx .shape ,VDD_BG_CLASS ,dtype =np .uint8 )
    valid =(idx >=0 )&(idx <VDD_NUM_CLASSES )
    out [valid ]=idx [valid ].astype (np .uint8 )
    out [idx ==ignore_index ]=ignore_index 
    out [(idx <0 )|(idx >=VDD_NUM_CLASSES )]=ignore_index 
    return out 

def vdd_mask_to_rgb (mask ,palette =None ,ignore_index =255 ):
    if palette is None :
        palette =VDD_PALETTE 
    mask =np .asarray (mask )
    h ,w =mask .shape 
    rgb =np .zeros ((h ,w ,3 ),dtype =np .uint8 )
    for class_id ,color in enumerate (palette ):
        rgb [mask ==class_id ]=color 
    if ignore_index is not None :
        rgb [mask ==ignore_index ]=(0 ,0 ,0 )
    return rgb 

def load_vdd_mask_rgb_display (mask_path ,palette =None ,ignore_index =255 ,size =None ):
    if palette is None :
        palette =VDD_PALETTE 
    raw =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
    if raw is None :
        raise FileNotFoundError (f'Failed to load mask: {mask_path}')
    if raw .ndim ==3 :
        raw =raw [:,:,0 ]
    class_mask =decode_vdd_mask (raw ,ignore_index =ignore_index )
    out =vdd_mask_to_rgb (class_mask ,palette ,ignore_index )
    if size is not None :
        w ,h =size 
        out =cv2 .resize (out ,(w ,h ),interpolation =cv2 .INTER_NEAREST )
    return out 

class VDDDataset (Dataset ):

    CLASSES =VDD_CLASSES 
    NUM_CLASSES =VDD_NUM_CLASSES 
    IGNORE_INDEX =255 

    def __init__ (self ,img_dir ,mask_dir ,input_w =VDD_PATCH_SIZE ,
    input_h =VDD_PATCH_SIZE ,patch_size =VDD_PATCH_SIZE ,
    transform =None ,is_train =True ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .is_train =is_train 
        self .patch_size =patch_size 
        if input_w !=input_h or input_w !=patch_size :
            raise ValueError (
            f'VDD patch must be square {patch_size}x{patch_size}, '
            f'got input_w={input_w}, input_h={input_h}')

        if not os .path .isdir (img_dir ):
            raise FileNotFoundError (f'Image directory not found: {img_dir}')
        if not os .path .isdir (mask_dir ):
            raise FileNotFoundError (f'Mask directory not found: {mask_dir}')

        self .img_ids =sorted ([
        os .path .splitext (f )[0 ]
        for f in os .listdir (img_dir )
        if f .lower ().endswith (IMAGE_EXTENSIONS )
        ])
        if not self .img_ids :
            raise RuntimeError (f'No images found in {img_dir}')

        if transform is None :
            self .transform =(
            get_uavid_train_transform (patch_size )
            if is_train else get_uavid_val_transform ())
        else :
            self .transform =transform 

    def __len__ (self ):
        return len (self .img_ids )

    def _resolve_path (self ,directory ,file_id ,extensions ):
        for ext in extensions :
            path =os .path .join (directory ,file_id +ext )
            if os .path .exists (path ):
                return path 
        raise FileNotFoundError (
        f'File not found for id `{file_id}` in `{directory}` '
        f'with extensions {extensions}')

    def __getitem__ (self ,idx ):
        img_id =self .img_ids [idx ]
        img_path =self ._resolve_path (self .img_dir ,img_id ,IMAGE_EXTENSIONS )
        mask_path =self ._resolve_path (self .mask_dir ,img_id ,MASK_EXTENSIONS )

        img =cv2 .imread (img_path )
        if img is None :
            raise FileNotFoundError (f'Failed to load image: {img_path}')
        img =cv2 .cvtColor (img ,cv2 .COLOR_BGR2RGB )

        raw_mask =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
        if raw_mask is None :
            raise FileNotFoundError (f'Failed to load mask: {mask_path}')
        if raw_mask .ndim ==3 :
            raw_mask =raw_mask [:,:,0 ]
        mask =decode_vdd_mask (raw_mask ,ignore_index =self .IGNORE_INDEX )

        img ,mask =_ensure_patch_size (img ,mask ,self .patch_size )
        augmented =self .transform (image =img ,mask =mask )
        return augmented ['image'],augmented ['mask'].long ()
