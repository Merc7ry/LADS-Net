
import os 

from .aeroscapes import AeroScapesDataset 
from .aeroscapes_base import (
AEROSCAPES_CLASSES ,
AEROSCAPES_NATIVE_H ,
AEROSCAPES_NATIVE_W ,
AEROSCAPES_PALETTE ,
)
from .isprs_base import (
ISPRS_CLASSES ,
ISPRS_NATIVE_H ,
ISPRS_NATIVE_W ,
ISPRS_PALETTE ,
)
from .loveda import LoveDADataset 
from .loveda_base import (
LOVEDA_CLASSES ,
LOVEDA_NATIVE_H ,
LOVEDA_NATIVE_W ,
LOVEDA_PALETTE ,
)
from .patch_layout import PATCH_DATASETS ,describe_data_root ,split_has_samples ,val_split_status 
from .potsdam import PotsdamDataset 
from .uavid import UAVidDataset 
from .uavid_base import UAVID_CLASSES ,UAVID_NATIVE_H ,UAVID_NATIVE_W ,UAVID_PALETTE 
from .udd5 import UDD5Dataset 
from .udd5_base import UDD5_CLASSES ,UDD5_NATIVE_H ,UDD5_NATIVE_W ,UDD5_PALETTE 
from .semantic_drone import SemanticDroneDataset 
from .semantic_drone_base import (
SEMANTIC_DRONE_CLASSES ,
SEMANTIC_DRONE_NATIVE_H ,
SEMANTIC_DRONE_NATIVE_W ,
SEMANTIC_DRONE_PALETTE ,
)
from .vaihingen import VaihingenDataset 
from .vdd import VDDDataset 
from .vdd_base import VDD_CLASSES ,VDD_NATIVE_H ,VDD_NATIVE_W ,VDD_PALETTE 

_DATASET_REGISTRY ={
'uavid':UAVidDataset ,
'aeroscapes':AeroScapesDataset ,
'vdd':VDDDataset ,
'udd5':UDD5Dataset ,
'vaihingen':VaihingenDataset ,
'potsdam':PotsdamDataset ,
'semantic_drone':SemanticDroneDataset ,
'loveda':LoveDADataset ,
}

_DATASET_INFO ={
'uavid':{
'num_classes':8 ,
'classes':UAVID_CLASSES ,
'palette':UAVID_PALETTE ,
'ignore_index':255 ,
'default_input_size':(512 ,512 ),
'native_image_size':(UAVID_NATIVE_H ,UAVID_NATIVE_W ),
'default_export_width':UAVID_NATIVE_W ,
},
'aeroscapes':{
'num_classes':12 ,
'classes':AEROSCAPES_CLASSES ,
'palette':AEROSCAPES_PALETTE ,
'ignore_index':255 ,
'default_input_size':(512 ,512 ),
'native_image_size':(AEROSCAPES_NATIVE_H ,AEROSCAPES_NATIVE_W ),
'default_export_width':AEROSCAPES_NATIVE_W ,
},
'vdd':{
'num_classes':7 ,
'classes':VDD_CLASSES ,
'palette':VDD_PALETTE ,
'ignore_index':255 ,
'default_input_size':(512 ,512 ),
'native_image_size':(VDD_NATIVE_H ,VDD_NATIVE_W ),
'default_export_width':VDD_NATIVE_W ,
},
'udd5':{
'num_classes':5 ,
'classes':UDD5_CLASSES ,
'palette':UDD5_PALETTE ,
'ignore_index':255 ,
'default_input_size':(512 ,512 ),
'native_image_size':(UDD5_NATIVE_H ,UDD5_NATIVE_W ),
'default_export_width':UDD5_NATIVE_W ,
},
'vaihingen':{
'num_classes':6 ,
'classes':ISPRS_CLASSES ,
'palette':ISPRS_PALETTE ,
'ignore_index':255 ,
'default_input_size':(256 ,256 ),
'native_image_size':(ISPRS_NATIVE_H ,ISPRS_NATIVE_W ),
'default_export_width':ISPRS_NATIVE_W ,
},
'potsdam':{
'num_classes':6 ,
'classes':ISPRS_CLASSES ,
'palette':ISPRS_PALETTE ,
'ignore_index':255 ,
'default_input_size':(256 ,256 ),
'native_image_size':(ISPRS_NATIVE_H ,ISPRS_NATIVE_W ),
'default_export_width':ISPRS_NATIVE_W ,
},
'semantic_drone':{
'num_classes':24 ,
'classes':SEMANTIC_DRONE_CLASSES ,
'palette':SEMANTIC_DRONE_PALETTE ,
'ignore_index':255 ,
'default_input_size':(512 ,512 ),
'native_image_size':(SEMANTIC_DRONE_NATIVE_H ,SEMANTIC_DRONE_NATIVE_W ),
'default_export_width':SEMANTIC_DRONE_NATIVE_W ,
},
'loveda':{
'num_classes':7 ,
'classes':LOVEDA_CLASSES ,
'palette':LOVEDA_PALETTE ,
'ignore_index':255 ,
'default_input_size':(512 ,512 ),
'native_image_size':(LOVEDA_NATIVE_H ,LOVEDA_NATIVE_W ),
'default_export_width':LOVEDA_NATIVE_W ,
},
}

_DATASETS_CUSTOM_DEFAULT =set (PATCH_DATASETS )

def get_dataset_info (name ):
    name =name .lower ()
    if name not in _DATASET_INFO :
        raise ValueError (
        f'Unknown dataset `{name}`. Available: {list(_DATASET_INFO)}')
    return _DATASET_INFO [name ].copy ()

def has_val_split (name ,data_root ):
    del name 
    return split_has_samples (data_root ,'val')

def build_dataset (name ,data_root ,split ='train',input_w =512 ,input_h =512 ,
transform =None ):
    name =name .lower ()
    if name not in _DATASET_REGISTRY :
        raise ValueError (
        f'Unknown dataset `{name}`. Available: {list(_DATASET_REGISTRY)}')

    info =_DATASET_INFO [name ]
    default_h ,default_w =info .get ('default_input_size',(input_h ,input_w ))
    if input_w ==512 and input_h ==512 and name in _DATASETS_CUSTOM_DEFAULT :
        input_w ,input_h =default_w ,default_h 

    return _DATASET_REGISTRY [name ](
    data_root =data_root ,
    split =split ,
    input_w =input_w ,
    input_h =input_h ,
    transform =transform ,
    )
