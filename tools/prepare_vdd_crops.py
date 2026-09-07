
import argparse 
import os 

import cv2 
import numpy as np 
from tqdm import tqdm 

IMAGE_EXTENSIONS =('.jpg','.jpeg','.png','.tif','.tiff','.JPG','.JPEG','.PNG')
MASK_EXTENSIONS =('.png','.PNG','.tif','.tiff','.jpg','.jpeg')

def _resolve_path (directory ,file_id ,extensions ):
    for ext in extensions :
        path =os .path .join (directory ,file_id +ext )
        if os .path .exists (path ):
            return path 
    raise FileNotFoundError (
    f'File not found for id `{file_id}` in `{directory}`')

def _load_split_ids (data_root ,split ):
    split_path =os .path .join (data_root ,'metadata',f'{split}.txt')
    if not os .path .isfile (split_path ):
        raise FileNotFoundError (f'Split file not found: {split_path}')

    ids =[]
    with open (split_path ,'r',encoding ='utf-8')as f :
        for line in f :
            line =line .strip ()
            if not line :
                continue 
            parts =line .split ()
            src_path =parts [0 ]
            img_id =os .path .splitext (os .path .basename (src_path ))[0 ]
            ids .append (img_id )
    if not ids :
        raise RuntimeError (f'Empty split file: {split_path}')
    return ids 

def _load_mask (mask_path ):
    raw =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
    if raw is None :
        raise FileNotFoundError (f'Failed to load mask: {mask_path}')
    if raw .ndim ==3 :
        return raw [:,:,0 ].astype (np .uint8 )
    return raw .astype (np .uint8 )

def _coverage_crop_boxes (h ,w ,crop_size ,n_patches ,rng ,grid_h =3 ,grid_w =4 ):
    max_y =h -crop_size 
    max_x =w -crop_size 
    if max_y <0 or max_x <0 :
        raise ValueError (f'Image {w}x{h} smaller than crop {crop_size}x{crop_size}')

    cell_h =h /grid_h 
    cell_w =w /grid_w 
    coverage =np .zeros ((grid_h ,grid_w ),dtype =np .int32 )
    boxes =[]

    for _ in range (n_patches ):
        min_cov =coverage .min ()
        candidates =np .argwhere (coverage ==min_cov )
        gy ,gx =candidates [int (rng .integers (0 ,len (candidates )))]

        y_lo =int (gy *cell_h )
        y_hi =int (min ((gy +1 )*cell_h ,h )-crop_size )
        x_lo =int (gx *cell_w )
        x_hi =int (min ((gx +1 )*cell_w ,w )-crop_size )

        y_lo =min (max (y_lo ,0 ),max_y )
        y_hi =max (y_lo ,min (y_hi ,max_y ))
        x_lo =min (max (x_lo ,0 ),max_x )
        x_hi =max (x_lo ,min (x_hi ,max_x ))

        y =int (rng .integers (y_lo ,y_hi +1 ))if y_hi >y_lo else y_lo 
        x =int (rng .integers (x_lo ,x_hi +1 ))if x_hi >x_lo else x_lo 
        boxes .append ((y ,x ))

        y2 ,x2 =y +crop_size ,x +crop_size 
        for iy in range (grid_h ):
            cy1 ,cy2 =int (iy *cell_h ),int ((iy +1 )*cell_h )
            for ix in range (grid_w ):
                cx1 ,cx2 =int (ix *cell_w ),int ((ix +1 )*cell_w )
                if not (y2 <=cy1 or y >=cy2 or x2 <=cx1 or x >=cx2 ):
                    coverage [iy ,ix ]+=1 

    return boxes 

def _process_split (src_root ,dst_root ,split ,crop_size ,patches_per_image ,seed ):
    img_dir =os .path .join (src_root ,split ,'src')
    mask_dir =os .path .join (src_root ,split ,'gt')
    out_img_dir =os .path .join (dst_root ,split ,'images')
    out_mask_dir =os .path .join (dst_root ,split ,'masks')
    os .makedirs (out_img_dir ,exist_ok =True )
    os .makedirs (out_mask_dir ,exist_ok =True )

    ids =_load_split_ids (src_root ,split )
    rng =np .random .default_rng (seed +(0 if split =='train'else 1 ))
    total =0 

    for img_id in tqdm (ids ,desc =split ,ncols =100 ):
        img_path =_resolve_path (img_dir ,img_id ,IMAGE_EXTENSIONS )
        mask_path =_resolve_path (mask_dir ,img_id ,MASK_EXTENSIONS )

        img =cv2 .imread (img_path )
        if img is None :
            raise FileNotFoundError (f'Failed to load image: {img_path}')
        mask =_load_mask (mask_path )

        h ,w =img .shape [:2 ]
        if mask .shape [:2 ]!=(h ,w ):
            mask =cv2 .resize (mask ,(w ,h ),interpolation =cv2 .INTER_NEAREST )

        boxes =_coverage_crop_boxes (h ,w ,crop_size ,patches_per_image ,rng )
        for k ,(y ,x )in enumerate (boxes ):
            patch_img =img [y :y +crop_size ,x :x +crop_size ]
            patch_mask =mask [y :y +crop_size ,x :x +crop_size ]
            out_id =f'{img_id}_p{k:02d}'
            cv2 .imwrite (os .path .join (out_img_dir ,out_id +'.jpg'),patch_img )
            cv2 .imwrite (os .path .join (out_mask_dir ,out_id +'.png'),patch_mask )
            total +=1 

    return len (ids ),total 

def main ():
    parser =argparse .ArgumentParser (description ='Prepare VDD 512x512 coverage-balanced crops')
    parser .add_argument ('--src_root',type =str ,default=None)
    parser .add_argument ('--dst_root',type =str ,default=None)
    parser .add_argument ('--crop_size',type =int ,default =512 )
    parser .add_argument ('--patches_per_image',type =int ,default =10 )
    parser .add_argument ('--seed',type =int ,default =42 )
    args =parser .parse_args ()

    print (f'Source: {args.src_root}')
    print (f'Dest  : {args.dst_root}')
    print (f'Crop  : {args.crop_size}x{args.crop_size}, {args.patches_per_image} patches/image')

    train_n ,train_p =_process_split (
    args .src_root ,args .dst_root ,'train',
    args .crop_size ,args .patches_per_image ,args .seed )
    val_n ,val_p =_process_split (
    args .src_root ,args .dst_root ,'val',
    args .crop_size ,args .patches_per_image ,args .seed )

    print (f'Done. train={train_n} images -> {train_p} patches')
    print (f'      val={val_n} images -> {val_p} patches')
    print (f'Output: {args.dst_root}')

if __name__ =='__main__':
    main ()
