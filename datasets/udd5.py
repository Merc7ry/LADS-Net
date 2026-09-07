
from .patch_layout import resolve_patch_split_dirs 
from .udd5_base import UDD5_PATCH_SIZE ,UDD5Dataset as UDD5DatasetBase 

class UDD5Dataset (UDD5DatasetBase ):

    DATASET_NAME ='udd5'

    def __init__ (
    self ,
    data_root ,
    split ='train',
    input_w =UDD5_PATCH_SIZE ,
    input_h =UDD5_PATCH_SIZE ,
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
        patch_size =UDD5_PATCH_SIZE ,
        transform =transform ,
        is_train =(split =='train'),
        )
        self .data_root =data_root 
        self .split =split 
