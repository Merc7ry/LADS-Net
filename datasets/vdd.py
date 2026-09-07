
import os 

from .patch_layout import resolve_patch_split_dirs 
from .vdd_base import VDD_PATCH_SIZE ,VDDDataset as VDDDatasetBase 

class VDDDataset (VDDDatasetBase ):

    DATASET_NAME ='vdd'

    def __init__ (self ,data_root ,split ='train',input_w =VDD_PATCH_SIZE ,
    input_h =VDD_PATCH_SIZE ,transform =None ):
        split =split .lower ()
        if split not in ('train','val','test'):
            raise ValueError (f'Unsupported split: {split}')

        img_dir ,mask_dir =resolve_patch_split_dirs (data_root ,split )
        super ().__init__ (
        img_dir =img_dir ,
        mask_dir =mask_dir ,
        input_w =input_w ,
        input_h =input_h ,
        patch_size =VDD_PATCH_SIZE ,
        transform =transform ,
        is_train =(split =='train'),
        )
        self .data_root =data_root 
        self .split =split 
