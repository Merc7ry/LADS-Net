
from __future__ import annotations 

import argparse 
import os 

import numpy as np 
from PIL import Image 
from tqdm import tqdm 

IMAGE_EXTENSIONS =('.png','.jpg','.jpeg','.tif','.tiff','.bmp')

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

def decode_semantic_drone_mask (mask ,ignore_index =255 ):
    mask =np .asarray (mask )
    if mask .ndim ==3 :
        if mask .shape [2 ]==1 :
            return decode_semantic_drone_mask (mask [:,:,0 ],ignore_index )
        rgb =mask [:,:,:3 ].astype (np .uint8 )
        if np .all (rgb [:,:,0 ]==rgb [:,:,1 ])and np .all (
        rgb [:,:,1 ]==rgb [:,:,2 ]
        ):
            return decode_semantic_drone_mask (rgb [:,:,0 ],ignore_index )
        h ,w ,_ =rgb .shape 
        out =np .full ((h ,w ),ignore_index ,dtype =np .uint8 )
        for color ,class_id in SEMANTIC_DRONE_COLOR_MAP .items ():
            match =(
            (rgb [...,0 ]==color [0 ])
            &(rgb [...,1 ]==color [1 ])
            &(rgb [...,2 ]==color [2 ])
            )
            out [match ]=class_id 
        return out 
    idx =mask .astype (np .int32 )
    out =np .full (idx .shape ,ignore_index ,dtype =np .uint8 )
    valid =(idx >=0 )&(idx <24 )
    out [valid ]=idx [valid ].astype (np .uint8 )
    out [idx ==ignore_index ]=ignore_index 
    return out 

def save_semantic_drone_mask_palette (path ,mask_index ):
    Image .fromarray (np .asarray (mask_index ,dtype =np .uint8 ),mode ='L').save (path )

def list_pairs (img_dir ,mask_dir ):
    if not os .path .isdir (img_dir ):
        raise FileNotFoundError (f'Image directory not found: {img_dir}')
    if not os .path .isdir (mask_dir ):
        raise FileNotFoundError (f'Mask directory not found: {mask_dir}')

    pairs =[]
    for name in sorted (os .listdir (img_dir )):
        if name .startswith ('.'):
            continue 
        if not name .lower ().endswith (IMAGE_EXTENSIONS ):
            continue 
        stem ,_ =os .path .splitext (name )
        src_path =os .path .join (img_dir ,name )
        gt_path =None 
        for ext in IMAGE_EXTENSIONS :
            cand =os .path .join (mask_dir ,stem +ext )
            if os .path .isfile (cand ):
                gt_path =cand 
                break 
        if gt_path is None :
            raise FileNotFoundError (f'No label for `{name}` in {mask_dir}')
        pairs .append ((src_path ,gt_path ,stem ))
    if not pairs :
        raise RuntimeError (f'No image files found in {img_dir}')
    return pairs 

def load_image (path ):
    with Image .open (path )as im :
        return np .array (im .convert ('RGB'))

def load_mask_index (path ):
    with Image .open (path )as im :
        if im .mode in ('P','L'):
            raw =np .array (im )
        else :
            raw =np .array (im .convert ('RGB'))
    return decode_semantic_drone_mask (raw )

def clear_dir (path ):
    os .makedirs (path ,exist_ok =True )
    for name in os .listdir (path ):
        fp =os .path .join (path ,name )
        if os .path .isfile (fp ):
            os .remove (fp )

def allocate_counts (n_images ,n_patches ):
    base =n_patches //n_images 
    rem =n_patches %n_images 
    return [base +(1 if i <rem else 0 )for i in range (n_images )]

def crop_patches (pairs ,out_img_dir ,out_mask_dir ,n_patches ,crop_size ,seed ):
    clear_dir (out_img_dir )
    clear_dir (out_mask_dir )
    rng =np .random .default_rng (seed )
    counts =allocate_counts (len (pairs ),n_patches )
    crop_h =crop_w =crop_size 
    saved =0 
    global_idx =0 

    pbar =tqdm (total =n_patches ,desc =os .path .basename (os .path .dirname (out_img_dir )))
    for (src_path ,gt_path ,stem ),n_crop in zip (pairs ,counts ):
        if n_crop <=0 :
            continue 
        img =load_image (src_path )
        mask =load_mask_index (gt_path )
        if img .shape [:2 ]!=mask .shape [:2 ]:
            raise ValueError (
            f'Size mismatch {stem}: image {img.shape[:2]} vs mask {mask.shape[:2]}')

        h ,w =img .shape [:2 ]
        if h <crop_h or w <crop_w :
            raise ValueError (f'{stem} too small ({w}x{h}) for crop {crop_size}')

        for _ in range (n_crop ):
            y =int (rng .integers (0 ,h -crop_h +1 ))
            x =int (rng .integers (0 ,w -crop_w +1 ))
            img_c =img [y :y +crop_h ,x :x +crop_w ]
            mask_c =mask [y :y +crop_h ,x :x +crop_w ]
            global_idx +=1 
            out_name =f'{stem}_p{global_idx:05d}_{y:04d}_{x:04d}.png'
            Image .fromarray (img_c ).save (os .path .join (out_img_dir ,out_name ))
            save_semantic_drone_mask_palette (
            os .path .join (out_mask_dir ,out_name ),mask_c )
            saved +=1 
            pbar .update (1 )
    pbar .close ()
    return saved 

def main ():
    parser =argparse .ArgumentParser (
    description ='Random-crop Semantic Drone to 512x512 train/val patches')
    parser .add_argument (
    '--src_root',
    type =str ,
    default=None,
    help ='Dataset root containing training_set/',
    )
    parser .add_argument (
    '--img_dir',
    type =str ,
    default =None ,
    help ='Override image dir (default: src_root/training_set/images)',
    )
    parser .add_argument (
    '--mask_dir',
    type =str ,
    default =None ,
    help ='Override mask dir (default: .../gt/semantic/label_images)',
    )
    parser .add_argument (
    '--dst_root',
    type =str ,
    default=None,
    help ='Output root with train/val subfolders',
    )
    parser .add_argument ('--crop_size',type =int ,default =512 )
    parser .add_argument ('--train_patches',type =int ,default =2400 )
    parser .add_argument ('--val_patches',type =int ,default =800 )
    parser .add_argument ('--seed',type =int ,default =42 )
    args =parser .parse_args ()

    img_dir =args .img_dir or os .path .join (
    args .src_root ,'training_set','images')
    mask_dir =args .mask_dir or os .path .join (
    args .src_root ,'training_set','gt','semantic','label_images')

    pairs =list_pairs (img_dir ,mask_dir )
    print (f'source images: {len(pairs)}')
    print (f'  images: {img_dir}')
    print (f'  masks:  {mask_dir}')

    n_train =crop_patches (
    pairs ,
    os .path .join (args .dst_root ,'train','images'),
    os .path .join (args .dst_root ,'train','masks'),
    args .train_patches ,
    args .crop_size ,
    seed =args .seed ,
    )
    n_val =crop_patches (
    pairs ,
    os .path .join (args .dst_root ,'val','images'),
    os .path .join (args .dst_root ,'val','masks'),
    args .val_patches ,
    args .crop_size ,
    seed =args .seed +1 ,
    )
    print (f'Done. train={n_train}, val={n_val}')
    print (f'Output: {args.dst_root}/train|val/{{images,masks}}')

if __name__ =='__main__':
    main ()
