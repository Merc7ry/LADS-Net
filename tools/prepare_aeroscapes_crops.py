
import argparse 
import os 
import stat 

import cv2 
import numpy as np 
from tqdm import tqdm 

IMAGE_EXTENSIONS =('.jpg','.jpeg','.png','.tif','.tiff')
MASK_EXTENSIONS =('.png','.tif','.tiff','.jpg','.jpeg')

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

def decode_aeroscapes_mask (mask ,color_map =None ,ignore_index =255 ):
    if color_map is None :
        color_map =AEROSCAPES_COLOR_MAP 
    mask =np .asarray (mask )

    if mask .ndim ==2 :
        idx =mask .astype (np .int32 )
        out =np .full (idx .shape ,0 ,dtype =np .uint8 )
        valid =(idx >=0 )&(idx <=11 )
        out [valid ]=idx [valid ].astype (np .uint8 )
        out [(idx <0 )|(idx >11 )]=ignore_index 
        return out 

    if mask .ndim ==3 :
        if mask .shape [2 ]==1 :
            return decode_aeroscapes_mask (mask [:,:,0 ],color_map ,ignore_index )
        rgb =mask [:,:,:3 ].astype (np .uint8 )
        if np .all (rgb [:,:,0 ]==rgb [:,:,1 ])and np .all (rgb [:,:,1 ]==rgb [:,:,2 ]):
            return decode_aeroscapes_mask (rgb [:,:,0 ],color_map ,ignore_index )
        h ,w ,_ =rgb .shape 
        class_mask =np .zeros ((h ,w ),dtype =np .uint8 )
        for color ,class_id in sorted (color_map .items (),key =lambda kv :kv [1 ]):
            class_mask [np .all (rgb ==np .array (color ,dtype =np .uint8 ),axis =-1 )]=class_id 
        return class_mask 

    raise ValueError (f'Unsupported mask shape: {mask.shape}')

_SPLIT_FILES ={
'train':('trn.txt','train.txt','trainval.txt'),
'val':('val.txt',),
}

def _resolve_path (directory ,file_id ,extensions ):
    for ext in extensions :
        path =os .path .join (directory ,file_id +ext )
        if os .path .exists (path ):
            return path 
    raise FileNotFoundError (
    f'File not found for id `{file_id}` in `{directory}`')

def _load_split_ids (data_root ,split ):
    imagesets_dir =os .path .join (data_root ,'ImageSets')
    if not os .path .isdir (imagesets_dir ):
        raise FileNotFoundError (f'ImageSets not found: {imagesets_dir}')

    for fname in _SPLIT_FILES [split ]:
        split_path =os .path .join (imagesets_dir ,fname )
        if not os .path .isfile (split_path ):
            continue 
        with open (split_path ,'r',encoding ='utf-8')as f :
            ids =[os .path .splitext (line .strip ())[0 ]for line in f if line .strip ()]
        if ids :
            return ids 
        raise RuntimeError (f'Empty split file: {split_path}')

    tried =[os .path .join (imagesets_dir ,f )for f in _SPLIT_FILES [split ]]
    raise FileNotFoundError (f'No split file for `{split}`. Tried: {tried}')

def _load_mask_for_crop (mask_path ,mask_dir_name ):
    raw =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
    if raw is None :
        raise FileNotFoundError (f'Failed to load mask: {mask_path}')

    if mask_dir_name =='Visualizations':
        if raw .ndim ==2 :
            raw =cv2 .cvtColor (raw ,cv2 .COLOR_GRAY2BGR )
        elif raw .ndim ==3 and raw .shape [2 ]==4 :
            raw =raw [:,:,:3 ]
        return raw 

    return _load_class_mask (mask_path ,mask_dir_name )

def _load_class_mask (mask_path ,mask_dir_name ):
    raw =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
    if raw is None :
        raise FileNotFoundError (f'Failed to load mask: {mask_path}')

    if raw .ndim ==3 :
        if raw .shape [2 ]>=3 :
            raw =cv2 .cvtColor (raw ,cv2 .COLOR_BGR2RGB )
            if (np .all (raw [:,:,0 ]==raw [:,:,1 ])
            and np .all (raw [:,:,1 ]==raw [:,:,2 ])):
                return raw [:,:,0 ].astype (np .uint8 )
            return decode_aeroscapes_mask (raw )
        return raw [:,:,0 ].astype (np .uint8 )
    return raw .astype (np .uint8 )

def _random_crop_box (h ,w ,crop_h ,crop_w ,rng ):
    if h <crop_h or w <crop_w :
        raise ValueError (f'Image {w}x{h} smaller than crop {crop_w}x{crop_h}')
    y =int (rng .integers (0 ,h -crop_h +1 ))
    x =int (rng .integers (0 ,w -crop_w +1 ))
    return y ,x 

def _clear_dir (path ):
    os .makedirs (path ,exist_ok =True )
    for name in os .listdir (path ):
        fp =os .path .join (path ,name )
        if os .path .isfile (fp )or os .path .islink (fp ):
            try :
                os .chmod (fp ,stat .S_IWRITE )
                os .remove (fp )
            except OSError :
                pass 

def _prepare_output_dir (dst_root ,split ):
    out_img_dir =os .path .join (dst_root ,split ,'images')
    out_mask_dir =os .path .join (dst_root ,split ,'masks')
    for d in (out_img_dir ,out_mask_dir ):
        _clear_dir (d )
    return out_img_dir ,out_mask_dir 

def _process_split (src_root ,dst_root ,split ,crop_size ,patches_per_image ,seed ,
mask_dir_name ):
    img_dir =os .path .join (src_root ,'JPEGImages')
    mask_dir =os .path .join (src_root ,mask_dir_name )
    if not os .path .isdir (mask_dir ):
        raise FileNotFoundError (f'Mask directory not found: {mask_dir}')

    out_img_dir ,out_mask_dir =_prepare_output_dir (dst_root ,split )
    ids =_load_split_ids (src_root ,split )
    rng =np .random .default_rng (seed +(0 if split =='train'else 1 ))

    for img_id in tqdm (ids ,desc =f'{split}',ncols =100 ):
        img_path =_resolve_path (img_dir ,img_id ,IMAGE_EXTENSIONS )
        mask_path =_resolve_path (mask_dir ,img_id ,MASK_EXTENSIONS )

        img =cv2 .imread (img_path )
        if img is None :
            raise FileNotFoundError (f'Failed to load image: {img_path}')
        mask =_load_mask_for_crop (mask_path ,mask_dir_name )

        h ,w =img .shape [:2 ]
        if mask .shape [:2 ]!=(h ,w ):
            mask =cv2 .resize (mask ,(w ,h ),interpolation =cv2 .INTER_NEAREST )

        for k in range (patches_per_image ):
            y ,x =_random_crop_box (h ,w ,crop_size ,crop_size ,rng )
            patch_img =img [y :y +crop_size ,x :x +crop_size ]
            patch_mask =mask [y :y +crop_size ,x :x +crop_size ]

            out_id =f'{img_id}_p{k:02d}'
            cv2 .imwrite (os .path .join (out_img_dir ,out_id +'.jpg'),patch_img )
            cv2 .imwrite (os .path .join (out_mask_dir ,out_id +'.png'),patch_mask )

    return len (ids )*patches_per_image 

def main ():
    parser =argparse .ArgumentParser (description ='Prepare AeroScapes 512x512 random crops')
    parser .add_argument ('--src_root',type =str ,default=None)
    parser .add_argument ('--dst_root',type =str ,default=None)
    parser .add_argument ('--crop_size',type =int ,default =512 )
    parser .add_argument ('--patches_per_image',type =int ,default =1 )
    parser .add_argument ('--seed',type =int ,default =42 )
    parser .add_argument ('--mask_dir',type =str ,default ='Visualizations',
    choices =['SegmentationClass','Visualizations'])
    args =parser .parse_args ()

    print (f'Source: {args.src_root}')
    print (f'Dest  : {args.dst_root}')
    print (f'Mask  : {args.mask_dir}')
    print (f'Crop  : {args.crop_size}x{args.crop_size}, {args.patches_per_image} patches/image')

    train_n =_process_split (
    args .src_root ,args .dst_root ,'train',
    args .crop_size ,args .patches_per_image ,args .seed ,args .mask_dir )
    val_n =_process_split (
    args .src_root ,args .dst_root ,'val',
    args .crop_size ,args .patches_per_image ,args .seed ,args .mask_dir )

    print (f'Done. train={train_n} patches, val={val_n} patches')
    print (f'Output: {args.dst_root}')

if __name__ =='__main__':
    main ()
