
import torch 
import torch .nn as nn 
import torch .nn .functional as F 

class WeightedFusion (nn .Module ):

    def __init__ (self ,num_stages ,num_classes ):
        super ().__init__ ()
        self .weight_convs =nn .ModuleList ([
        nn .Conv2d (num_classes ,1 ,kernel_size =1 )
        for _ in range (num_stages )
        ])

    def forward (self ,predictions ):
        weight_maps =[conv (pred )for pred ,conv in zip (predictions ,self .weight_convs )]
        weights =F .softmax (torch .stack (weight_maps ,dim =1 ),dim =1 )
        preds =torch .stack (predictions ,dim =1 )
        return torch .sum (weights *preds ,dim =1 )
