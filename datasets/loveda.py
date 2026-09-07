
from .loveda_base import (
LOVEDA_PATCH_SIZE ,
LOVEDA_STRIDE ,
LoveDADataset as LoveDADatasetBase ,
)

class LoveDADataset (LoveDADatasetBase ):

    DATASET_NAME ='loveda'

    def __init__ (
    self ,
    data_root ,
    split ='train',
    input_w =LOVEDA_PATCH_SIZE ,
    input_h =LOVEDA_PATCH_SIZE ,
    transform =None ,
    stride =LOVEDA_STRIDE ,
    ):
        super ().__init__ (
        data_root =data_root ,
        split =split ,
        input_w =input_w ,
        input_h =input_h ,
        patch_size =LOVEDA_PATCH_SIZE ,
        stride =stride ,
        transform =transform ,
        is_train =(split .lower ()=='train'),
        )
