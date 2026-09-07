
import os 

import albumentations as A 
import cv2 
import numpy as np 
from albumentations .pytorch import ToTensorV2 
from torch .utils .data import Dataset 

IMAGE_EXTENSIONS =('.tif','.tiff','.png','.jpg','.jpeg')
MASK_EXTENSIONS =('.png','.tif','.tiff')

def _try_create (candidates ,label ):
    errors =[]
    for factory in candidates :
        try :
            return factory ()
        except (TypeError ,ValueError )as exc :
            errors .append (str (exc ))
    raise RuntimeError (
    f'Failed to create {label} with albumentations {A.__version__}: '
    +' | '.join (errors ))

UAVID_PATCH_SIZE =512 
UAVID_NATIVE_W =3840 
UAVID_NATIVE_H =2160 
UAVID_BG_CLASS =0 

UAVID_COLOR_MAP ={
(0 ,0 ,0 ):0 ,
(128 ,0 ,0 ):1 ,
(128 ,64 ,128 ):2 ,
(0 ,128 ,0 ):3 ,
(128 ,128 ,0 ):4 ,
(64 ,0 ,128 ):5 ,
(192 ,0 ,192 ):6 ,
(64 ,64 ,0 ):7 ,
}

UAVID_CLASSES =[
'Background',
'Building',
'Road',
'Tree',
'Static car',
'Moving car',
'Human',
'Low vegetation',
]

UAVID_PALETTE =[
(0 ,0 ,0 ),
(128 ,0 ,0 ),
(128 ,64 ,128 ),
(0 ,128 ,0 ),
(128 ,128 ,0 ),
(64 ,0 ,128 ),
(192 ,0 ,192 ),
(64 ,64 ,0 ),
]

IMAGENET_MEAN =np .array ([0.485 ,0.456 ,0.406 ],dtype =np .float32 )
IMAGENET_STD =np .array ([0.229 ,0.224 ,0.225 ],dtype =np .float32 )

def denormalize_image (image ):
    img =np .asarray (image ,dtype =np .float32 )
    if img .ndim ==3 and img .shape [0 ]==3 :
        img =img .transpose (1 ,2 ,0 )
    img =img *IMAGENET_STD +IMAGENET_MEAN 
    return np .clip (img *255.0 ,0 ,255 ).astype (np .uint8 )

def grid_crop_positions (h ,w ,patch_size =UAVID_PATCH_SIZE ):
    n_rows =h //patch_size 
    n_cols =w //patch_size 
    if n_rows ==0 or n_cols ==0 :
        return []

    start_y =(h -n_rows *patch_size )//2 
    start_x =(w -n_cols *patch_size )//2 
    return [
    (start_y +i *patch_size ,start_x +j *patch_size )
    for i in range (n_rows )
    for j in range (n_cols )
    ]

def crop_patch (img ,mask ,y ,x ,patch_size =UAVID_PATCH_SIZE ):
    return (
    img [y :y +patch_size ,x :x +patch_size ].copy (),
    mask [y :y +patch_size ,x :x +patch_size ].copy (),
    )

def _random_scale (scale_min =0.5 ,scale_max =2.0 ):
    low =scale_min -1.0 
    high =scale_max -1.0 
    return _try_create ([
    lambda :A .RandomScale (scale_limit =(low ,high ),p =1.0 ),
    lambda :A .RandomScale (scale_limit =(low ,high )),
    ],'RandomScale')

def _pad_if_needed (min_h ,min_w ,fill =0 ,fill_mask =0 ):
    return _try_create ([
    lambda :A .PadIfNeeded (
    min_height =min_h ,min_width =min_w ,
    border_mode =cv2 .BORDER_CONSTANT ,fill =fill ,fill_mask =fill_mask ),
    lambda :A .PadIfNeeded (
    min_height =min_h ,min_width =min_w ,
    border_mode =cv2 .BORDER_CONSTANT ,value =fill ,mask_value =fill_mask ),
    ],'PadIfNeeded')

def _random_crop (crop_h ,crop_w ):
    return _try_create ([
    lambda :A .RandomCrop (height =crop_h ,width =crop_w ),
    lambda :A .RandomCrop (crop_h ,crop_w ),
    ],'RandomCrop')

def _is_uavid_rgb_mask (rgb ):
    hits =0 
    for color in UAVID_COLOR_MAP :
        if np .any (np .all (rgb ==np .array (color ,dtype =np .uint8 ),axis =-1 )):
            hits +=1 
    return hits >=2 

def decode_uavid_mask (mask ,color_map =None ,ignore_index =255 ):
    if color_map is None :
        color_map =UAVID_COLOR_MAP 
    mask =np .asarray (mask )

    if mask .ndim ==2 :
        out =mask .astype (np .uint8 )
        out [(out <0 )|(out >7 )]=ignore_index 
        return out 

    if mask .ndim ==3 :
        if mask .shape [2 ]==1 :
            return decode_uavid_mask (mask [:,:,0 ],color_map ,ignore_index )

        rgb =mask [:,:,:3 ].astype (np .uint8 )
        if np .all (rgb [:,:,0 ]==rgb [:,:,1 ])and np .all (rgb [:,:,1 ]==rgb [:,:,2 ]):
            return decode_uavid_mask (rgb [:,:,0 ],color_map ,ignore_index )

        if _is_uavid_rgb_mask (rgb ):
            h ,w ,_ =rgb .shape 
            class_mask =np .full ((h ,w ),ignore_index ,dtype =np .uint8 )
            for color ,class_id in color_map .items ():
                class_mask [np .all (rgb ==np .array (color ,dtype =np .uint8 ),axis =-1 )]=class_id 
            return class_mask 

        return decode_uavid_mask (rgb [:,:,0 ],color_map ,ignore_index )

    raise ValueError (f'Unsupported mask shape: {mask.shape}')

def uavid_mask_to_rgb (mask ,palette =None ,ignore_index =255 ):
    if palette is None :
        palette =UAVID_PALETTE 
    mask =np .asarray (mask )
    h ,w =mask .shape 
    rgb =np .zeros ((h ,w ,3 ),dtype =np .uint8 )
    for class_id ,color in enumerate (palette ):
        rgb [mask ==class_id ]=color 
    if ignore_index is not None :
        rgb [mask ==ignore_index ]=(0 ,0 ,0 )
    return rgb 

def apply_uavid_patch_geom (class_mask ,y ,x ,patch_size =UAVID_PATCH_SIZE ):
    return class_mask [y :y +patch_size ,x :x +patch_size ].copy ()

def load_uavid_mask_rgb_display (mask_path ,palette =None ,ignore_index =255 ,
crop_box =None ,size =None ):
    if palette is None :
        palette =UAVID_PALETTE 
    raw =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
    if raw is None :
        raise FileNotFoundError (f'Failed to load mask: {mask_path}')

    if raw .ndim ==3 and raw .shape [2 ]>=3 :
        rgb =cv2 .cvtColor (raw ,cv2 .COLOR_BGR2RGB )
        if _is_uavid_rgb_mask (rgb ):
            class_mask =decode_uavid_mask (rgb ,ignore_index =ignore_index )
        else :
            class_mask =decode_uavid_mask (rgb ,ignore_index =ignore_index )
    else :
        if raw .ndim ==3 :
            raw =raw [:,:,0 ]
        class_mask =decode_uavid_mask (raw ,ignore_index =ignore_index )

    if crop_box is not None :
        y ,x ,patch_size =crop_box 
        class_mask =apply_uavid_patch_geom (class_mask ,y ,x ,patch_size )

    out =uavid_mask_to_rgb (class_mask ,palette ,ignore_index )
    if size is not None :
        w ,h =size 
        out =cv2 .resize (out ,(w ,h ),interpolation =cv2 .INTER_NEAREST )
    return out 

def get_uavid_train_transform (patch_size =UAVID_PATCH_SIZE ):
    return A .Compose ([
    A .HorizontalFlip (p =0.5 ),
    _random_scale (0.5 ,2.0 ),
    _pad_if_needed (patch_size ,patch_size ,fill =0 ,fill_mask =UAVID_BG_CLASS ),
    _random_crop (patch_size ,patch_size ),
    A .RandomBrightnessContrast (
    brightness_limit =0.2 ,contrast_limit =0.2 ,p =1.0 ),
    A .HueSaturationValue (
    hue_shift_limit =20 ,sat_shift_limit =30 ,val_shift_limit =20 ,p =1.0 ),
    A .Normalize (mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )),
    ToTensorV2 (),
    ])

def get_uavid_val_transform ():
    return A .Compose ([
    A .Normalize (mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )),
    ToTensorV2 (),
    ])

def _ensure_patch_size (img ,mask ,patch_size ):
    h ,w =img .shape [:2 ]
    if h ==patch_size and w ==patch_size :
        return img ,mask 
    img =cv2 .resize (
    img ,(patch_size ,patch_size ),interpolation =cv2 .INTER_LINEAR )
    mask =cv2 .resize (
    mask ,(patch_size ,patch_size ),interpolation =cv2 .INTER_NEAREST )
    return img ,mask 

class UAVidDataset (Dataset ):

    COLOR_MAP =UAVID_COLOR_MAP 
    CLASSES =UAVID_CLASSES 
    NUM_CLASSES =8 
    IGNORE_INDEX =255 

    def __init__ (self ,img_dir ,mask_dir ,input_w =UAVID_PATCH_SIZE ,
    input_h =UAVID_PATCH_SIZE ,patch_size =UAVID_PATCH_SIZE ,
    transform =None ,is_train =True ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .is_train =is_train 
        self .patch_size =patch_size 
        if input_w !=input_h or input_w !=patch_size :
            raise ValueError (
            f'UAVid patch must be square {patch_size}x{patch_size}, '
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
            mask =decode_uavid_mask (
            cv2 .cvtColor (raw_mask ,cv2 .COLOR_BGR2RGB ),
            ignore_index =self .IGNORE_INDEX )
        elif raw_mask .ndim ==3 :
            mask =decode_uavid_mask (
            raw_mask [:,:,0 ],ignore_index =self .IGNORE_INDEX )
        else :
            mask =decode_uavid_mask (raw_mask ,ignore_index =self .IGNORE_INDEX )

        img ,mask =_ensure_patch_size (img ,mask ,self .patch_size )
        augmented =self .transform (image =img ,mask =mask )
        return augmented ['image'],augmented ['mask'].long ()
