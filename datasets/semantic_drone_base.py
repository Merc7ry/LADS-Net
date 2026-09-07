
import os 

import cv2 
import numpy as np 
from PIL import Image 
from torch .utils .data import Dataset 

from .uavid_base import (
IMAGE_EXTENSIONS ,
MASK_EXTENSIONS ,
_ensure_patch_size ,
_pad_if_needed ,
_random_crop ,
_random_scale ,
get_uavid_train_transform ,
get_uavid_val_transform ,
)

SEMANTIC_DRONE_PATCH_SIZE =512 
SEMANTIC_DRONE_NATIVE_W =6000 
SEMANTIC_DRONE_NATIVE_H =4000 
SEMANTIC_DRONE_NUM_CLASSES =24 
SEMANTIC_DRONE_BG_CLASS =0 

SEMANTIC_DRONE_CLASSES =[
'unlabeled',
'paved-area',
'dirt',
'grass',
'gravel',
'water',
'rocks',
'pool',
'vegetation',
'roof',
'wall',
'window',
'door',
'fence',
'fence-pole',
'person',
'dog',
'car',
'bicycle',
'tree',
'bald-tree',
'ar-marker',
'obstacle',
'conflicting',
]

SEMANTIC_DRONE_PALETTE =[
(0 ,0 ,0 ),
(128 ,64 ,128 ),
(130 ,76 ,0 ),
(0 ,102 ,0 ),
(112 ,103 ,87 ),
(28 ,42 ,168 ),
(48 ,41 ,30 ),
(0 ,50 ,89 ),
(107 ,142 ,35 ),
(70 ,70 ,70 ),
(102 ,102 ,156 ),
(254 ,228 ,12 ),
(254 ,148 ,12 ),
(190 ,153 ,153 ),
(153 ,153 ,153 ),
(255 ,22 ,96 ),
(102 ,51 ,0 ),
(9 ,143 ,150 ),
(119 ,11 ,32 ),
(51 ,51 ,0 ),
(190 ,250 ,190 ),
(112 ,150 ,146 ),
(2 ,135 ,115 ),
(255 ,0 ,0 ),
]

SEMANTIC_DRONE_COLOR_MAP ={
color :i for i ,color in enumerate (SEMANTIC_DRONE_PALETTE )
}

def get_semantic_drone_train_transform (patch_size =SEMANTIC_DRONE_PATCH_SIZE ):
    import albumentations as A 
    from albumentations .pytorch import ToTensorV2 

    return A .Compose ([
    A .HorizontalFlip (p =0.5 ),
    A .VerticalFlip (p =0.5 ),
    _random_scale (0.5 ,2.0 ),
    _pad_if_needed (
    patch_size ,patch_size ,fill =0 ,fill_mask =SEMANTIC_DRONE_BG_CLASS ),
    _random_crop (patch_size ,patch_size ),
    A .RandomBrightnessContrast (
    brightness_limit =0.2 ,contrast_limit =0.2 ,p =1.0 ),
    A .HueSaturationValue (
    hue_shift_limit =20 ,sat_shift_limit =30 ,val_shift_limit =20 ,p =1.0 ),
    A .Normalize (mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )),
    ToTensorV2 (),
    ])

get_semantic_drone_val_transform =get_uavid_val_transform 

def decode_semantic_drone_mask (mask ,color_map =None ,ignore_index =255 ):
    if color_map is None :
        color_map =SEMANTIC_DRONE_COLOR_MAP 
    mask =np .asarray (mask )

    if mask .ndim ==3 :
        if mask .shape [2 ]==1 :
            return decode_semantic_drone_mask (mask [:,:,0 ],color_map ,ignore_index )

        rgb =mask [:,:,:3 ].astype (np .uint8 )
        if np .all (rgb [:,:,0 ]==rgb [:,:,1 ])and np .all (
        rgb [:,:,1 ]==rgb [:,:,2 ]
        ):
            return decode_semantic_drone_mask (rgb [:,:,0 ],color_map ,ignore_index )

        h ,w ,_ =rgb .shape 
        out =np .full ((h ,w ),ignore_index ,dtype =np .uint8 )
        for color ,class_id in color_map .items ():
            match =(
            (rgb [...,0 ]==color [0 ])
            &(rgb [...,1 ]==color [1 ])
            &(rgb [...,2 ]==color [2 ])
            )
            out [match ]=class_id 
        return out 

    idx =mask .astype (np .int32 )
    out =np .full (idx .shape ,ignore_index ,dtype =np .uint8 )
    valid =(idx >=0 )&(idx <SEMANTIC_DRONE_NUM_CLASSES )
    out [valid ]=idx [valid ].astype (np .uint8 )
    out [idx ==ignore_index ]=ignore_index 
    return out 

def semantic_drone_mask_to_rgb (mask ,palette =None ,ignore_index =255 ):
    if palette is None :
        palette =SEMANTIC_DRONE_PALETTE 
    mask =np .asarray (mask )
    h ,w =mask .shape 
    rgb =np .zeros ((h ,w ,3 ),dtype =np .uint8 )
    for class_id ,color in enumerate (palette ):
        rgb [mask ==class_id ]=color 
    if ignore_index is not None :
        rgb [mask ==ignore_index ]=(0 ,0 ,0 )
    return rgb 

def load_semantic_drone_mask_rgb_display (mask_path ,palette =None ,ignore_index =255 ,
size =None ):
    if palette is None :
        palette =SEMANTIC_DRONE_PALETTE 
    with Image .open (mask_path )as mask_im :
        if mask_im .mode in ('P','L'):
            raw =np .array (mask_im )
        else :
            raw =np .array (mask_im .convert ('RGB'))
    class_mask =decode_semantic_drone_mask (raw ,ignore_index =ignore_index )
    out =semantic_drone_mask_to_rgb (class_mask ,palette ,ignore_index )
    if size is not None :
        w ,h =size 
        out =cv2 .resize (out ,(w ,h ),interpolation =cv2 .INTER_NEAREST )
    return out 

def save_semantic_drone_mask_palette (path ,mask_index ):
    flat =[]
    for c in SEMANTIC_DRONE_PALETTE :
        flat .extend (c )
    while len (flat )<768 :
        flat .extend ([0 ,0 ,0 ])
    im =Image .fromarray (np .asarray (mask_index ,dtype =np .uint8 ),mode ='L').convert ('P')
    im .putpalette (flat )
    im .save (path )

class SemanticDroneDataset (Dataset ):

    CLASSES =SEMANTIC_DRONE_CLASSES 
    NUM_CLASSES =SEMANTIC_DRONE_NUM_CLASSES 
    IGNORE_INDEX =255 

    def __init__ (
    self ,
    img_dir ,
    mask_dir ,
    input_w =SEMANTIC_DRONE_PATCH_SIZE ,
    input_h =SEMANTIC_DRONE_PATCH_SIZE ,
    patch_size =SEMANTIC_DRONE_PATCH_SIZE ,
    transform =None ,
    is_train =True ,
    ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .is_train =is_train 
        self .patch_size =patch_size 
        if input_w !=input_h or input_w !=patch_size :
            raise ValueError (
            f'Semantic Drone patch must be square {patch_size}x{patch_size}, '
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

        paired =[]
        missing =0 
        for img_id in self .img_ids :
            try :
                self ._resolve_path (self .mask_dir ,img_id ,MASK_EXTENSIONS )
                paired .append (img_id )
            except FileNotFoundError :
                missing +=1 
        if not paired :
            raise RuntimeError (
            f'No image/mask pairs in `{img_dir}` and `{mask_dir}`. '
            f'Image and mask files must share the same stem.')
        if missing :
            print (
            f'Warning: skipped {missing} images without matching masks '
            f'in {mask_dir}')
        self .img_ids =paired 

        if transform is None :
            self .transform =(
            get_semantic_drone_train_transform (patch_size )
            if is_train 
            else get_semantic_drone_val_transform ()
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
        f'with extensions {extensions}')

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
        mask =decode_semantic_drone_mask (raw_mask ,ignore_index =self .IGNORE_INDEX )

        img ,mask =_ensure_patch_size (img ,mask ,self .patch_size )
        augmented =self .transform (image =img ,mask =mask )
        return augmented ['image'],augmented ['mask'].long ()
