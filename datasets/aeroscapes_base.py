
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
AEROSCAPES_PATCH_SIZE =UAVID_PATCH_SIZE 
AEROSCAPES_CROP_W =AEROSCAPES_PATCH_SIZE 
AEROSCAPES_CROP_H =AEROSCAPES_PATCH_SIZE 
AEROSCAPES_NATIVE_W =1280 
AEROSCAPES_NATIVE_H =720 
AEROSCAPES_BG_CLASS =0 
AEROSCAPES_COLOR_MAP ={
(0 ,0 ,0 ):0 ,
(192 ,128 ,128 ):1 ,
(0 ,128 ,0 ):2 ,
(128 ,128 ,128 ):3 ,
(128 ,0 ,0 ):4 ,
(0 ,0 ,128 ):5 ,
(192 ,0 ,128 ):6 ,
(192 ,0 ,0 ):7 ,
(192 ,128 ,0 ):8 ,
(0 ,64 ,0 ):9 ,
(128 ,128 ,0 ):10 ,
(0 ,128 ,128 ):11 ,
}

AEROSCAPES_CLASSES =[
'Background',
'Person',
'Bike',
'Car',
'Drone',
'Boat',
'Animal',
'Obstacle',
'Construction',
'Vegetation',
'Road',
'Sky',
]

AEROSCAPES_PALETTE =[
(0 ,0 ,0 ),
(192 ,128 ,128 ),
(0 ,128 ,0 ),
(128 ,128 ,128 ),
(128 ,0 ,0 ),
(0 ,0 ,128 ),
(192 ,0 ,128 ),
(192 ,0 ,0 ),
(192 ,128 ,0 ),
(0 ,64 ,0 ),
(128 ,128 ,0 ),
(0 ,128 ,128 ),
]

def _is_aeroscapes_rgb_mask (rgb ):
    hits =0 
    for color in AEROSCAPES_COLOR_MAP :
        if color ==(0 ,0 ,0 ):
            continue 
        if np .any (np .all (rgb ==np .array (color ,dtype =np .uint8 ),axis =-1 )):
            hits +=1 
    return hits >=2 

def decode_aeroscapes_mask (mask ,color_map =None ,ignore_index =255 ):
    if color_map is None :
        color_map =AEROSCAPES_COLOR_MAP 
    mask =np .asarray (mask )

    if mask .ndim ==2 :
        idx =mask .astype (np .int32 )
        uniq =set (np .unique (idx ).tolist ())

        if uniq .issubset (set (range (1 ,12 ))|{255 ,0 })and 0 not in (uniq -{255 }):
            out =np .full (idx .shape ,AEROSCAPES_BG_CLASS ,dtype =np .uint8 )
            valid =(idx >=1 )&(idx <=11 )
            out [valid ]=idx [valid ].astype (np .uint8 )
            out [idx ==255 ]=ignore_index 
            return out 
        out =idx .astype (np .uint8 )
        out [(out <0 )|(out >11 )]=ignore_index 
        return out 

    if mask .ndim ==3 :
        if mask .shape [2 ]==1 :
            return decode_aeroscapes_mask (mask [:,:,0 ],color_map ,ignore_index )

        rgb =mask [:,:,:3 ].astype (np .uint8 )
        if np .all (rgb [:,:,0 ]==rgb [:,:,1 ])and np .all (rgb [:,:,1 ]==rgb [:,:,2 ]):
            return decode_aeroscapes_mask (rgb [:,:,0 ],color_map ,ignore_index )

        if _is_aeroscapes_rgb_mask (rgb ):
            h ,w ,_ =rgb .shape 
            class_mask =np .full ((h ,w ),ignore_index ,dtype =np .uint8 )
            for color ,class_id in color_map .items ():
                class_mask [np .all (rgb ==np .array (color ,dtype =np .uint8 ),axis =-1 )]=class_id 
            return class_mask 

        return decode_aeroscapes_mask (rgb [:,:,0 ],color_map ,ignore_index )

    raise ValueError (f'Unsupported mask shape: {mask.shape}')

def aeroscapes_mask_to_rgb (mask ,palette =None ,ignore_index =255 ):
    if palette is None :
        palette =AEROSCAPES_PALETTE 
    mask =np .asarray (mask )
    h ,w =mask .shape 
    rgb =np .zeros ((h ,w ,3 ),dtype =np .uint8 )
    for class_id ,color in enumerate (palette ):
        rgb [mask ==class_id ]=color 
    if ignore_index is not None :
        rgb [mask ==ignore_index ]=(0 ,0 ,0 )
    return rgb 

def apply_aeroscapes_patch_geom (class_mask ,y ,x ,patch_size =AEROSCAPES_PATCH_SIZE ):
    return class_mask [y :y +patch_size ,x :x +patch_size ].copy ()

def load_aeroscapes_mask_rgb_display (mask_path ,palette =None ,ignore_index =255 ,
crop_box =None ,size =None ):
    if palette is None :
        palette =AEROSCAPES_PALETTE 
    raw =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
    if raw is None :
        raise FileNotFoundError (f'Failed to load mask: {mask_path}')

    if raw .ndim ==3 and raw .shape [2 ]>=3 :
        rgb =cv2 .cvtColor (raw ,cv2 .COLOR_BGR2RGB )
        class_mask =decode_aeroscapes_mask (rgb ,ignore_index =ignore_index )
    else :
        if raw .ndim ==3 :
            raw =raw [:,:,0 ]
        class_mask =decode_aeroscapes_mask (raw ,ignore_index =ignore_index )

    if crop_box is not None :
        y ,x ,patch_size =crop_box 
        class_mask =apply_aeroscapes_patch_geom (class_mask ,y ,x ,patch_size )

    out =aeroscapes_mask_to_rgb (class_mask ,palette ,ignore_index )
    if size is not None :
        w ,h =size 
        out =cv2 .resize (out ,(w ,h ),interpolation =cv2 .INTER_NEAREST )
    return out 

get_aeroscapes_train_transform =get_uavid_train_transform 
get_aeroscapes_val_transform =get_uavid_val_transform 

class AeroScapesDataset (Dataset ):

    COLOR_MAP =AEROSCAPES_COLOR_MAP 
    CLASSES =AEROSCAPES_CLASSES 
    NUM_CLASSES =12 
    IGNORE_INDEX =255 

    def __init__ (self ,img_dir ,mask_dir ,input_w =AEROSCAPES_PATCH_SIZE ,
    input_h =AEROSCAPES_PATCH_SIZE ,patch_size =AEROSCAPES_PATCH_SIZE ,
    transform =None ,is_train =True ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .is_train =is_train 
        self .patch_size =patch_size 
        if input_w !=input_h or input_w !=patch_size :
            raise ValueError (
            f'AeroScapes patch must be square {patch_size}x{patch_size}, '
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
        if raw_mask .ndim ==3 and raw_mask .shape [2 ]>=3 :
            mask =decode_aeroscapes_mask (
            cv2 .cvtColor (raw_mask ,cv2 .COLOR_BGR2RGB ),
            ignore_index =self .IGNORE_INDEX )
        elif raw_mask .ndim ==3 :
            mask =decode_aeroscapes_mask (
            raw_mask [:,:,0 ],ignore_index =self .IGNORE_INDEX )
        else :
            mask =decode_aeroscapes_mask (raw_mask ,ignore_index =self .IGNORE_INDEX )

        img ,mask =_ensure_patch_size (img ,mask ,self .patch_size )
        augmented =self .transform (image =img ,mask =mask )
        return augmented ['image'],augmented ['mask'].long ()
