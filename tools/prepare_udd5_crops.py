
from __future__ import annotations 

import argparse 
import os 
from collections import defaultdict 

import cv2 
import numpy as np 
from PIL import Image 
from tqdm import tqdm 

def list_pairs (split_dir ):
    src_dir =os .path .join (split_dir ,'src')
    gt_dir =os .path .join (split_dir ,'gt')
    if not os .path .isdir (src_dir )or not os .path .isdir (gt_dir ):
        raise FileNotFoundError (f'Need src/ and gt/ under {split_dir}')

    pairs =[]
    for name in sorted (os .listdir (src_dir )):
        if name .startswith ('.'):
            continue 
        stem ,_ =os .path .splitext (name )
        src_path =os .path .join (src_dir ,name )
        gt_path =None 
        for ext in ('.png','.PNG','.jpg','.JPG','.tif'):
            cand =os .path .join (gt_dir ,stem +ext )
            if os .path .isfile (cand ):
                gt_path =cand 
                break 
        if gt_path is None :
            raise FileNotFoundError (f'No GT for {name} in {gt_dir}')
        pairs .append ((src_path ,gt_path ,stem ))
    return pairs 

def load_image (path ):
    with Image .open (path )as im :
        return np .array (im .convert ('RGB'))

def load_mask_index (path ):
    with Image .open (path )as im :
        if im .mode =='P':
            arr =np .array (im ,dtype =np .uint8 )
        elif im .mode =='L':
            arr =np .array (im ,dtype =np .uint8 )
        else :
            arr =np .array (im .convert ('RGB'))
            if arr .ndim ==3 and np .all (arr [:,:,0 ]==arr [:,:,1 ])and np .all (
            arr [:,:,1 ]==arr [:,:,2 ]
            ):
                arr =arr [:,:,0 ].astype (np .uint8 )
            else :
                raise ValueError (f'Unsupported color GT (need index/palette): {path}')
    return arr 

def save_mask_palette (path ,mask_index ):
    palette =[
    (107 ,142 ,35 ),
    (102 ,102 ,156 ),
    (128 ,64 ,128 ),
    (0 ,0 ,142 ),
    (0 ,0 ,0 ),
    ]
    flat =[]
    for c in palette :
        flat .extend (c )
    while len (flat )<768 :
        flat .extend ([0 ,0 ,0 ])
    im =Image .fromarray (np .asarray (mask_index ,dtype =np .uint8 ),mode ='L').convert ('P')
    im .putpalette (flat )
    im .save (path )

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

def crop_split (pairs ,out_img_dir ,out_mask_dir ,n_patches ,crop_size ,seed ):
    clear_dir (out_img_dir )
    clear_dir (out_mask_dir )
    rng =np .random .default_rng (seed )
    counts =allocate_counts (len (pairs ),n_patches )
    crop_h =crop_w =crop_size 
    saved =0 
    per_stem =defaultdict (int )

    cache ={}

    pbar =tqdm (total =n_patches ,desc =os .path .basename (os .path .dirname (out_img_dir )))
    for (src_path ,gt_path ,stem ),n_crop in zip (pairs ,counts ):
        if n_crop <=0 :
            continue 
        if stem not in cache :
            img =load_image (src_path )
            mask =load_mask_index (gt_path )
            if img .shape [:2 ]!=mask .shape [:2 ]:
                raise ValueError (
                f'Size mismatch {stem}: image {img.shape[:2]} vs mask {mask.shape[:2]}'
                )
            cache [stem ]=(img ,mask )
        img ,mask =cache [stem ]
        h ,w =img .shape [:2 ]
        if h <crop_h or w <crop_w :
            raise ValueError (f'{stem} too small ({w}x{h}) for {crop_size}')

        for _ in range (n_crop ):
            y =int (rng .integers (0 ,h -crop_h +1 ))
            x =int (rng .integers (0 ,w -crop_w +1 ))
            img_c =img [y :y +crop_h ,x :x +crop_w ]
            mask_c =mask [y :y +crop_h ,x :x +crop_w ]
            per_stem [stem ]+=1 
            out_name =f'{stem}_{per_stem[stem]:02d}_{y:04d}_{x:04d}.png'
            cv2 .imwrite (
            os .path .join (out_img_dir ,out_name ),
            cv2 .cvtColor (img_c ,cv2 .COLOR_RGB2BGR ),
            )
            save_mask_palette (os .path .join (out_mask_dir ,out_name ),mask_c )
            saved +=1 
            pbar .update (1 )

        del cache [stem ]
    pbar .close ()
    return saved 

def main ():
    parser =argparse .ArgumentParser (description ='Random-crop UDD5 to 512x512')
    parser .add_argument (
    '--src_root',
    type =str ,
    default=None,
    )
    parser .add_argument ('--dst_root',type =str ,default=None)
    parser .add_argument ('--crop_size',type =int ,default =512 )
    parser .add_argument ('--train_patches',type =int ,default =2400 )
    parser .add_argument ('--val_patches',type =int ,default =800 )
    parser .add_argument ('--seed',type =int ,default =42 )
    args =parser .parse_args ()

    train_pairs =list_pairs (os .path .join (args .src_root ,'train'))
    val_pairs =list_pairs (os .path .join (args .src_root ,'val'))
    print (f'train images: {len(train_pairs)}, val images: {len(val_pairs)}')

    n_train =crop_split (
    train_pairs ,
    os .path .join (args .dst_root ,'train','images'),
    os .path .join (args .dst_root ,'train','masks'),
    args .train_patches ,
    args .crop_size ,
    seed =args .seed ,
    )
    n_val =crop_split (
    val_pairs ,
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
