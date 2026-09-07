import torch 
import torch .nn as nn 
import torch .nn .functional as F 

class JointLoss (nn .Module ):

    def __init__ (self ,num_classes ,ignore_index =255 ):
        super ().__init__ ()
        self .num_classes =num_classes 
        self .ignore_index =ignore_index 
        self .ce =nn .CrossEntropyLoss (ignore_index =ignore_index )

    def forward (self ,inputs ,targets ):
        loss_ce =self .ce (inputs ,targets )

        prob =F .softmax (inputs ,dim =1 ).clamp (1e-6 ,1 -1e-6 )
        valid_mask =targets !=self .ignore_index 
        targets_masked =targets *valid_mask 
        target_one_hot =F .one_hot (
        targets_masked .long (),self .num_classes ).permute (0 ,3 ,1 ,2 ).float ()

        mask_expand =valid_mask .unsqueeze (1 ).expand_as (prob ).float ()
        prob =prob *mask_expand 
        target_one_hot =target_one_hot *mask_expand 

        intersection =(prob *target_one_hot ).sum (dim =(2 ,3 ))
        union =prob .sum (dim =(2 ,3 ))+target_one_hot .sum (dim =(2 ,3 ))
        dice_score =(2.0 *intersection +1.0 )/(union +1.0 )
        loss_dice =1 -dice_score .mean ()
        return loss_ce +loss_dice 
