
from .isprs_base import ISPRS_PATCH_SIZE ,ISPRSPatchDataset ,isprs_rgb_image_id_to_mask_id 
from .patch_layout import resolve_patch_split_dirs 

class PotsdamDataset (ISPRSPatchDataset ):

    DATASET_NAME ='potsdam'

    def __init__ (
    self ,
    data_root ,
    split ='train',
    input_w =ISPRS_PATCH_SIZE ,
    input_h =ISPRS_PATCH_SIZE ,
    transform =None ,
    ):
        split =split .lower ()
        if split not in ('train','val','test'):
            raise ValueError (f'Unsupported split: {split}')

        img_dir ,mask_dir =resolve_patch_split_dirs (data_root ,split )
        super ().__init__ (
        img_dir =img_dir ,
        mask_dir =mask_dir ,
        input_w =input_w ,
        input_h =input_h ,
        patch_size =ISPRS_PATCH_SIZE ,
        transform =transform ,
        is_train =(split =='train'),
        mask_id_fn =isprs_rgb_image_id_to_mask_id ,
        )
        self .data_root =data_root 
        self .split =split 
