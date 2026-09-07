
import torch 
import torch .nn as nn 

class WGSDiceLoss (nn .Module ):
    def __init__ (self ,n_classes ,ignore_index =None ):
        super ().__init__ ()
        self .n_classes =n_classes 
        self .ignore_index =ignore_index 

    def _one_hot_encoder (self ,input_tensor ):
        tensor_list =[]
        for i in range (self .n_classes ):
            tensor_list .append ((input_tensor ==i ).unsqueeze (1 ))
        return torch .cat (tensor_list ,dim =1 ).float ()

    def _dice_loss (self ,score ,target ):
        smooth =1e-5 
        intersect =torch .sum (score *target )
        y_sum =torch .sum (target *target )
        z_sum =torch .sum (score *score )
        return 1.0 -(2.0 *intersect +smooth )/(z_sum +y_sum +smooth )

    def forward (self ,inputs ,target ,weight =None ,softmax =False ):
        if softmax :
            inputs =torch .softmax (inputs ,dim =1 )
        if self .ignore_index is not None :
            valid =(target !=self .ignore_index ).unsqueeze (1 ).float ()
            target =target .clone ()
            target [target ==self .ignore_index ]=0 
        else :
            valid =None 

        target =self ._one_hot_encoder (target )
        if valid is not None :
            inputs =inputs *valid 
            target =target *valid 

        if weight is None :
            weight =[1.0 ]*self .n_classes 

        loss =0.0 
        for i in range (self .n_classes ):
            dice =self ._dice_loss (inputs [:,i ],target [:,i ])
            loss +=dice *weight [i ]
        return loss /self .n_classes 
