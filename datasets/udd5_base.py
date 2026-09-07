
import os 

import cv2 
import numpy as np 
from PIL import Image 
from torch .utils .data import Dataset 

from .uavid_base import (
IMAGE_EXTENSIONS ,
MASK_EXTENSIONS ,
UAVID_PATCH_SIZE ,
_ensure_patch_size ,
get_uavid_train_transform ,
get_uavid_val_transform ,
)

UDD5_PATCH_SIZE =UAVID_PATCH_SIZE 
UDD5_NATIVE_W =4096 
UDD5_NATIVE_H =2160 
UDD5_NUM_CLASSES =5 

UDD5_CLASSES =[
'Vegetation',
'Building',
'Road',
'Vehicle',
'Other',
]

UDD5_PALETTE =[
(107 ,142 ,35 ),
(102 ,102 ,156 ),
(128 ,64 ,128 ),
(0 ,0 ,142 ),
(0 ,0 ,0 ),
]

UDD5_COLOR_MAP ={color :i for i ,color in enumerate (UDD5_PALETTE )}

get_udd5_train_transform =get_uavid_train_transform 
get_udd5_val_transform =get_uavid_val_transform 

def decode_udd5_mask (mask ,ignore_index =255 ):
    mask =np .asarray (mask )
    if mask .ndim ==3 :
        if mask .shape [2 ]==1 :
            mask =mask [:,:,0 ]
        elif np .all (mask [:,:,0 ]==mask [:,:,1 ])and np .all (
        mask [:,:,1 ]==mask [:,:,2 ]
        ):
            mask =mask [:,:,0 ]
        else :
            rgb =mask [:,:,:3 ].astype (np .uint8 )
            h ,w ,_ =rgb .shape 
            out =np .full ((h ,w ),ignore_index ,dtype =np .uint8 )
            for color ,class_id in UDD5_COLOR_MAP .items ():
                out [np .all (rgb ==np .array (color ,dtype =np .uint8 ),axis =-1 )]=class_id 
            return out 

    idx =mask .astype (np .int32 )
    out =np .full (idx .shape ,ignore_index ,dtype =np .uint8 )
    valid =(idx >=0 )&(idx <UDD5_NUM_CLASSES )
    out [valid ]=idx [valid ].astype (np .uint8 )
    out [idx ==ignore_index ]=ignore_index 
    return out 

def udd5_mask_to_rgb (mask ,palette =None ,ignore_index =255 ):
    if palette is None :
        palette =UDD5_PALETTE 
    mask =np .asarray (mask )
    h ,w =mask .shape 
    rgb =np .zeros ((h ,w ,3 ),dtype =np .uint8 )
    for class_id ,color in enumerate (palette ):
        rgb [mask ==class_id ]=color 
    if ignore_index is not None :
        rgb [mask ==ignore_index ]=(0 ,0 ,0 )
    return rgb 

def load_udd5_mask_rgb_display (mask_path ,palette =None ,ignore_index =255 ,size =None ):
    if palette is None :
        palette =UDD5_PALETTE 
    with Image .open (mask_path )as mask_im :
        if mask_im .mode in ('P','L'):
            raw =np .array (mask_im )
        else :
            raw =np .array (mask_im .convert ('RGB'))
    class_mask =decode_udd5_mask (raw ,ignore_index =ignore_index )
    out =udd5_mask_to_rgb (class_mask ,palette ,ignore_index )
    if size is not None :
        w ,h =size 
        out =cv2 .resize (out ,(w ,h ),interpolation =cv2 .INTER_NEAREST )
    return out 

class UDD5Dataset (Dataset ):

    CLASSES =UDD5_CLASSES 
    NUM_CLASSES =UDD5_NUM_CLASSES 
    IGNORE_INDEX =255 

    def __init__ (
    self ,
    img_dir ,
    mask_dir ,
    input_w =UDD5_PATCH_SIZE ,
    input_h =UDD5_PATCH_SIZE ,
    patch_size =UDD5_PATCH_SIZE ,
    transform =None ,
    is_train =True ,
    ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .is_train =is_train 
        self .patch_size =patch_size 
        if input_w !=input_h or input_w !=patch_size :
            raise ValueError (
            f'UDD5 patch must be square {patch_size}x{patch_size}, '
            f'got input_w={input_w}, input_h={input_h}'
            )

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
            if is_train 
            else get_uavid_val_transform ()
            )
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
        f'with extensions {extensions}'
        )

    def __getitem__ (self ,idx ):
        img_id =self .img_ids [idx ]
        img_path =self ._resolve_path (self .img_dir ,img_id ,IMAGE_EXTENSIONS )
        mask_path =self ._resolve_path (self .mask_dir ,img_id ,MASK_EXTENSIONS )

        img =cv2 .imread (img_path )
        if img is None :
            raise FileNotFoundError (f'Failed to load image: {img_path}')
        img =cv2 .cvtColor (img ,cv2 .COLOR_BGR2RGB )

        with Image .open (mask_path )as mask_im :
            if mask_im .mode in ('P','L'):
                raw_mask =np .array (mask_im )
            else :
                raw_mask =np .array (mask_im .convert ('RGB'))
        mask =decode_udd5_mask (raw_mask ,ignore_index =self .IGNORE_INDEX )

        img ,mask =_ensure_patch_size (img ,mask ,self .patch_size )
        augmented =self .transform (image =img ,mask =mask )
        return augmented ['image'],augmented ['mask'].long ()
