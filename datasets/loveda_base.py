
from __future__ import annotations 

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
crop_patch ,
get_uavid_val_transform ,
grid_crop_positions ,
)

LOVEDA_PATCH_SIZE =512 
LOVEDA_NATIVE_W =1024 
LOVEDA_NATIVE_H =1024 
LOVEDA_NUM_CLASSES =7 
LOVEDA_STRIDE =512 

LOVEDA_CLASSES =[
'background',
'building',
'road',
'water',
'barren',
'forest',
'agriculture',
]

LOVEDA_PALETTE =[
(255 ,255 ,255 ),
(255 ,0 ,0 ),
(255 ,255 ,0 ),
(0 ,0 ,255 ),
(159 ,129 ,183 ),
(0 ,255 ,0 ),
(255 ,195 ,128 ),
]

LOVEDA_COLOR_MAP ={color :i for i ,color in enumerate (LOVEDA_PALETTE )}

def get_loveda_train_transform (patch_size =LOVEDA_PATCH_SIZE ):
    import albumentations as A 
    from albumentations .pytorch import ToTensorV2 

    return A .Compose ([
    A .HorizontalFlip (p =0.5 ),
    A .VerticalFlip (p =0.5 ),
    _random_scale (0.5 ,2.0 ),
    _pad_if_needed (patch_size ,patch_size ,fill =0 ,fill_mask =255 ),
    _random_crop (patch_size ,patch_size ),
    A .RandomBrightnessContrast (
    brightness_limit =0.2 ,contrast_limit =0.2 ,p =1.0 ),
    A .HueSaturationValue (
    hue_shift_limit =20 ,sat_shift_limit =30 ,val_shift_limit =20 ,p =1.0 ),
    A .Normalize (mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )),
    ToTensorV2 (),
    ])

get_loveda_val_transform =get_uavid_val_transform 

def decode_loveda_mask (mask ,ignore_index =255 ,reduce_zero_label =True ):
    mask =np .asarray (mask )
    if mask .ndim ==3 :
        if mask .shape [2 ]==1 :
            return decode_loveda_mask (
            mask [:,:,0 ],ignore_index ,reduce_zero_label )
        rgb =mask [:,:,:3 ].astype (np .uint8 )
        if np .all (rgb [:,:,0 ]==rgb [:,:,1 ])and np .all (
        rgb [:,:,1 ]==rgb [:,:,2 ]
        ):
            return decode_loveda_mask (
            rgb [:,:,0 ],ignore_index ,reduce_zero_label )
        h ,w ,_ =rgb .shape 
        out =np .full ((h ,w ),ignore_index ,dtype =np .uint8 )
        for color ,class_id in LOVEDA_COLOR_MAP .items ():
            match =(
            (rgb [...,0 ]==color [0 ])
            &(rgb [...,1 ]==color [1 ])
            &(rgb [...,2 ]==color [2 ])
            )
            out [match ]=class_id 
        return out 

    idx =mask .astype (np .int32 )
    out =np .full (idx .shape ,ignore_index ,dtype =np .uint8 )
    if reduce_zero_label :

        has_class7 =(idx ==LOVEDA_NUM_CLASSES ).any ()
        has_official =((idx >=1 )&(idx <=LOVEDA_NUM_CLASSES )).any ()
        if idx .max ()==0 :
            return out 
        if has_class7 or has_official :
            official =(idx >=1 )&(idx <=LOVEDA_NUM_CLASSES )
            out [official ]=(idx [official ]-1 ).astype (np .uint8 )
            out [idx ==0 ]=ignore_index 
        else :

            reduced =(idx >=0 )&(idx <LOVEDA_NUM_CLASSES )
            out [reduced ]=idx [reduced ].astype (np .uint8 )
        out [idx ==ignore_index ]=ignore_index 
        return out 

    valid =(idx >=0 )&(idx <LOVEDA_NUM_CLASSES )
    out [valid ]=idx [valid ].astype (np .uint8 )
    out [idx ==ignore_index ]=ignore_index 
    return out 

def loveda_mask_to_rgb (mask ,palette =None ,ignore_index =255 ):
    if palette is None :
        palette =LOVEDA_PALETTE 
    mask =np .asarray (mask )
    h ,w =mask .shape 
    rgb =np .zeros ((h ,w ,3 ),dtype =np .uint8 )
    for class_id ,color in enumerate (palette ):
        rgb [mask ==class_id ]=color 
    if ignore_index is not None :
        rgb [mask ==ignore_index ]=(0 ,0 ,0 )
    return rgb 

def load_loveda_mask_rgb_display (mask_path ,palette =None ,ignore_index =255 ,
size =None ):
    if palette is None :
        palette =LOVEDA_PALETTE 
    with Image .open (mask_path )as mask_im :
        if mask_im .mode in ('P','L'):
            raw =np .array (mask_im )
        else :
            raw =np .array (mask_im .convert ('RGB'))
    class_mask =decode_loveda_mask (raw ,ignore_index =ignore_index )
    out =loveda_mask_to_rgb (class_mask ,palette ,ignore_index )
    if size is not None :
        w ,h =size 
        out =cv2 .resize (out ,(w ,h ),interpolation =cv2 .INTER_NEAREST )
    return out 

def _list_image_files (directory ):
    if not os .path .isdir (directory ):
        return []
    return sorted (
    f for f in os .listdir (directory )
    if f .lower ().endswith (IMAGE_EXTENSIONS )and not f .startswith ('.')
    )

def _resolve_mask_path (mask_dir ,stem ):
    for ext in MASK_EXTENSIONS :
        cand =os .path .join (mask_dir ,stem +ext )
        if os .path .isfile (cand ):
            return cand 
    return None 

def _resolve_child (parent ,*names ):
    if not os .path .isdir (parent ):
        return None 
    entries ={e .lower ():e for e in os .listdir (parent )}
    for name in names :
        key =name .lower ()
        if key in entries :
            path =os .path .join (parent ,entries [key ])
            if os .path .isdir (path ):
                return path 
    return None 

def discover_loveda_pairs (data_root ,split ):
    data_root =os .path .abspath (data_root )
    split_aliases ={
    'train':('Train','train','TRAIN'),
    'val':('Val','val','Validation','validation','VAL'),
    'test':('Test','test','TEST'),
    }
    aliases =split_aliases .get (split .lower (),(split ,))
    split_dir =None 
    for name in aliases :
        cand =_resolve_child (data_root ,name )
        if cand is not None :
            split_dir =cand 
            break 
    if split_dir is None :
        raise FileNotFoundError (
        f'LoveDA split `{split}` not found under {data_root}. '
        f'Expected one of {aliases}.')

    pairs =[]
    domains =[]
    for domain in ('Urban','Rural','urban','rural'):
        d =_resolve_child (split_dir ,domain )
        if d is not None :
            domains .append (d )
    if not domains :
        domains =[split_dir ]

    for domain_dir in domains :
        img_dir =_resolve_child (
        domain_dir ,'images_png','images','img','imgs')
        mask_dir =_resolve_child (
        domain_dir ,'masks_png','masks','mask','labels','label')
        if img_dir is None :
            continue 
        domain_tag =os .path .basename (domain_dir )
        for name in _list_image_files (img_dir ):
            stem ,_ =os .path .splitext (name )
            img_path =os .path .join (img_dir ,name )
            mask_path =None 
            if mask_dir is not None :
                mask_path =_resolve_mask_path (mask_dir ,stem )
            if mask_path is None and split .lower ()!='test':
                raise FileNotFoundError (
                f'No LoveDA mask for `{name}` in {mask_dir}')
            uid =f'{domain_tag}_{stem}'if domain_tag .lower ()in (
            'urban','rural')else stem 
            pairs .append ((img_path ,mask_path ,uid ))

    if not pairs :
        raise RuntimeError (
        f'No LoveDA images found for split `{split}` under {split_dir}')
    return pairs 

def count_loveda_images (data_root ,split ):
    try :
        return len (discover_loveda_pairs (data_root ,split ))
    except (FileNotFoundError ,RuntimeError ):
        return 0 

def slide_positions (h ,w ,patch_size =LOVEDA_PATCH_SIZE ,stride =LOVEDA_STRIDE ):
    if h <patch_size or w <patch_size :
        return grid_crop_positions (h ,w ,patch_size )

    def starts (length ):
        xs =list (range (0 ,length -patch_size +1 ,stride ))
        last =length -patch_size 
        if not xs or xs [-1 ]!=last :
            xs .append (last )
        return sorted (set (xs ))

    return [(y ,x )for y in starts (h )for x in starts (w )]

class LoveDADataset (Dataset ):

    CLASSES =LOVEDA_CLASSES 
    NUM_CLASSES =LOVEDA_NUM_CLASSES 
    IGNORE_INDEX =255 

    def __init__ (
    self ,
    data_root ,
    split ='train',
    input_w =LOVEDA_PATCH_SIZE ,
    input_h =LOVEDA_PATCH_SIZE ,
    patch_size =LOVEDA_PATCH_SIZE ,
    stride =LOVEDA_STRIDE ,
    transform =None ,
    is_train =None ,
    ):
        self .data_root =data_root 
        self .split =split .lower ()
        self .patch_size =patch_size 
        self .stride =stride 
        if is_train is None :
            is_train =self .split =='train'
        self .is_train =is_train 

        if input_w !=input_h or input_w !=patch_size :
            raise ValueError (
            f'LoveDA patch must be square {patch_size}x{patch_size}, '
            f'got input_w={input_w}, input_h={input_h}')

        pairs =discover_loveda_pairs (data_root ,self .split )
        self .samples =[]
        for img_path ,mask_path ,uid in pairs :

            positions =slide_positions (
            LOVEDA_NATIVE_H ,LOVEDA_NATIVE_W ,patch_size ,stride )
            if not positions :
                raise RuntimeError (
                f'No sliding windows for {img_path} with '
                f'patch={patch_size}, stride={stride}')
            for y ,x in positions :
                self .samples .append ((img_path ,mask_path ,uid ,y ,x ))

        if transform is None :
            self .transform =(
            get_loveda_train_transform (patch_size )
            if is_train 
            else get_loveda_val_transform ()
            )
        else :
            self .transform =transform 

    def __len__ (self ):
        return len (self .samples )

    def __getitem__ (self ,idx ):
        img_path ,mask_path ,uid ,y ,x =self .samples [idx ]

        img =cv2 .imread (img_path )
        if img is None :
            raise FileNotFoundError (f'Failed to load image: {img_path}')
        img =cv2 .cvtColor (img ,cv2 .COLOR_BGR2RGB )

        if mask_path is None :
            mask =np .full (img .shape [:2 ],self .IGNORE_INDEX ,dtype =np .uint8 )
        else :
            with Image .open (mask_path )as mask_im :
                if mask_im .mode in ('P','L'):
                    raw_mask =np .array (mask_im )
                else :
                    raw_mask =np .array (mask_im .convert ('RGB'))
            mask =decode_loveda_mask (raw_mask ,ignore_index =self .IGNORE_INDEX )

        if img .shape [:2 ]!=mask .shape [:2 ]:
            raise ValueError (
            f'Size mismatch {uid}: image {img.shape[:2]} vs mask {mask.shape[:2]}')

        h ,w =img .shape [:2 ]
        if h !=LOVEDA_NATIVE_H or w !=LOVEDA_NATIVE_W :

            positions =slide_positions (h ,w ,self .patch_size ,self .stride )
            if not positions :
                raise ValueError (
                f'{uid} too small ({w}x{h}) for crop {self.patch_size}')

            if (y ,x )not in positions :
                y ,x =positions [idx %len (positions )]

        img_c ,mask_c =crop_patch (img ,mask ,y ,x ,self .patch_size )
        img_c ,mask_c =_ensure_patch_size (img_c ,mask_c ,self .patch_size )
        augmented =self .transform (image =img_c ,mask =mask_c )
        return augmented ['image'],augmented ['mask'].long ()
