
import os 

from .uavid_base import IMAGE_EXTENSIONS 

PATCH_DATASETS =(
'uavid','aeroscapes','vdd','udd5',
'vaihingen','potsdam','semantic_drone','loveda',
)

def count_images (directory ):
    if not os .path .isdir (directory ):
        return 0 
    return sum (
    1 for f in os .listdir (directory )
    if f .lower ().endswith (IMAGE_EXTENSIONS )
    )

def _resolve_child_dir (parent ,name ):
    direct =os .path .join (parent ,name )
    if os .path .isdir (direct ):
        return direct 
    if not os .path .isdir (parent ):
        return direct 
    target =name .lower ()
    for entry in os .listdir (parent ):
        if entry .lower ()==target and os .path .isdir (os .path .join (parent ,entry )):
            return os .path .join (parent ,entry )
    return direct 

def resolve_patch_split_dirs (data_root ,split ):
    split_dir =_resolve_child_dir (os .path .abspath (data_root ),split )
    img_dir =_resolve_child_dir (split_dir ,'images')
    mask_dir =_resolve_child_dir (split_dir ,'masks')
    return img_dir ,mask_dir 

def split_has_samples (data_root ,split ):
    img_dir ,mask_dir =resolve_patch_split_dirs (data_root ,split )
    if os .path .isdir (img_dir )and os .path .isdir (mask_dir )and count_images (img_dir )>0 :
        return True 
    try :
        from .loveda_base import count_loveda_images 
        return count_loveda_images (data_root ,split )>0 
    except Exception :
        return False 

def describe_data_root (data_root ):
    data_root =os .path .abspath (data_root )
    lines =[f'Data root: {data_root}']
    for split in ('train','val'):
        img_dir ,mask_dir =resolve_patch_split_dirs (data_root ,split )
        img_n =count_images (img_dir )
        mask_n =0 
        if os .path .isdir (mask_dir ):
            mask_n =sum (
            1 for f in os .listdir (mask_dir )
            if f .lower ().endswith (('.png','.tif','.tiff','.jpg','.jpeg'))
            )
        if img_n ==0 :
            try :
                from .loveda_base import count_loveda_images 
                loveda_n =count_loveda_images (data_root ,split )
                if loveda_n >0 :
                    lines .append (
                    f'  {split}: loveda_tiles={loveda_n} '
                    f'(~{loveda_n * 4} sliding 512 patches)')
                    continue 
            except Exception :
                pass 
        lines .append (
        f'  {split}: images={img_n} ({img_dir}), masks={mask_n} ({mask_dir})')
    return '\n'.join (lines )

def val_split_status (data_root ):
    data_root =os .path .abspath (data_root )
    img_dir ,mask_dir =resolve_patch_split_dirs (data_root ,'val')
    if os .path .isdir (img_dir )and os .path .isdir (mask_dir ):
        img_n =count_images (img_dir )
        if img_n >0 :
            return True ,f'{img_n} val samples at {img_dir}'
    try :
        from .loveda_base import count_loveda_images 
        loveda_n =count_loveda_images (data_root ,'val')
        if loveda_n >0 :
            return True ,(
            f'{loveda_n} LoveDA val tiles '
            f'(~{loveda_n * 4} sliding 512 patches)')
    except Exception :
        pass 
    if not os .path .isdir (img_dir ):
        return False ,f'missing directory: {img_dir}'
    if not os .path .isdir (mask_dir ):
        return False ,f'missing directory: {mask_dir}'
    return False ,f'no images found in: {img_dir}'
