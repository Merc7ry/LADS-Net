
from itertools import combinations 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 

from .fusion import WeightedFusion 

class CombinatorialMutationsLossModule (nn .Module ):

    def __init__ (
    self ,
    original_num_maps ,
    num_classes ,
    selecetd_num_maps ,
    operations =None ,
    use_learnable_weights =True ,
    supervision ='wgs',
    lc1 =0.3 ,
    lc2 =0.7 ,
    ):
        super ().__init__ ()
        if operations is None :
            operations =['add','sub','mul','concat','weighted_fusion','avg','max']

        self .num_maps =selecetd_num_maps 
        self .num_classes =num_classes 
        self .operations =list (operations )
        self .use_learnable_weights =use_learnable_weights 
        self .supervision =supervision 
        self .lc1 =lc1 
        self .lc2 =lc2 

        if use_learnable_weights :
            self .original_weights =nn .ParameterList ([
            nn .Parameter (torch .zeros (1 ))
            for _ in range (original_num_maps )
            ])

        self .combination_indices ={}
        if use_learnable_weights :
            self .synthesized_weights =nn .ModuleDict ()

        for op in self .operations :
            comb_list =[]
            for k in range (2 ,selecetd_num_maps +1 ):
                comb_list .extend (list (combinations (range (selecetd_num_maps ),k )))
            self .combination_indices [op ]=comb_list 

            if use_learnable_weights :
                self .synthesized_weights [op ]=nn .ParameterList ([
                nn .Parameter (torch .zeros (1 ))
                for _ in range (len (comb_list ))
                ])

        self .weighted_fusion_modules =nn .ModuleDict ({
        str (k ):WeightedFusion (k ,num_classes )
        for k in range (2 ,selecetd_num_maps +1 )
        })

        self .concat_convs =nn .ModuleDict ()
        if 'concat'in self .operations :
            for idx ,comb in enumerate (self .combination_indices ['concat']):
                in_ch =len (comb )*num_classes 
                self .concat_convs [str (idx )]=nn .Conv2d (in_ch ,num_classes ,kernel_size =1 )

    def _compute_all_weights (self ):
        if not self .use_learnable_weights :
            return None ,None 
        orig_vals =[F .softplus (w )for w in self .original_weights ]
        synth_vals ={
        op :[F .softplus (p )for p in plist ]
        for op ,plist in self .synthesized_weights .items ()
        }
        return orig_vals ,synth_vals 

    def _mutate (self ,op ,comb ,output_maps ,combo_idx =None ):
        if op =='add':
            return sum (output_maps [c ]for c in comb )
        if op =='avg':
            return sum (output_maps [c ]for c in comb )/len (comb )
        if op =='sub':
            return output_maps [comb [0 ]]-sum (output_maps [c ]for c in comb [1 :])
        if op =='mul':
            mutated =output_maps [comb [0 ]]
            for c in comb [1 :]:
                mutated =mutated *output_maps [c ]
            return mutated 
        if op =='concat':
            cat =torch .cat ([output_maps [c ]for c in comb ],dim =1 )
            return self .concat_convs [str (combo_idx )](cat )
        if op in ('weighted_fusion','wf'):
            mod =self .weighted_fusion_modules [str (len (comb ))]
            return mod ([output_maps [c ]for c in comb ])
        if op =='max':
            return torch .stack ([output_maps [c ]for c in comb ],dim =0 ).max (dim =0 ).values 
        raise ValueError (f'Unsupported op: {op}')

    def fuse_all (self ,output_maps ):
        return self .weighted_fusion_modules [str (self .num_maps )](output_maps )

    def forward (self ,output_maps ,label_batch =None ,ce_loss =None ,dice_loss =None ):
        device =output_maps [0 ].device 
        deep_supervision_loss =0.0 
        mutation_loss =0.0 
        fused_logits =[]

        if not self .use_learnable_weights :
            orig_weights =[torch .ones (1 ,device =device )for _ in output_maps ]
            syn_weights ={
            op :[torch .ones (1 ,device =device )for _ in combos ]
            for op ,combos in self .combination_indices .items ()
            }
        else :
            orig_weights ,syn_weights =self ._compute_all_weights ()

        infer_only =label_batch is None and ce_loss is None and dice_loss is None 

        for i ,fmap in enumerate (output_maps ):
            if infer_only :
                continue 
            loss_ce =ce_loss (fmap ,label_batch .long ())
            loss_dice =dice_loss (fmap ,label_batch ,softmax =True )
            combined_loss =self .lc1 *loss_ce +self .lc2 *loss_dice 
            deep_supervision_loss +=orig_weights [i ]*combined_loss 

        for op in self .operations :
            combos =self .combination_indices [op ]
            w_list =syn_weights [op ]
            for idx ,comb in enumerate (combos ):
                mutated =self ._mutate (op ,comb ,output_maps ,combo_idx =idx )
                fused_logits .append (mutated )
                if infer_only :
                    continue 
                loss_ce =ce_loss (mutated ,label_batch .long ())
                loss_dice =dice_loss (mutated ,label_batch ,softmax =True )
                combined_loss =self .lc1 *loss_ce +self .lc2 *loss_dice 
                mutation_loss +=w_list [idx ]*combined_loss 

        if infer_only :
            return fused_logits 

        if self .supervision in ('mutation','wgs'):
            final_loss =deep_supervision_loss +mutation_loss 
        else :
            final_loss =deep_supervision_loss 
        return final_loss ,deep_supervision_loss ,mutation_loss 

    def save_weights (self ,save_path ):
        torch .save (self .state_dict (),save_path )

    def load_weights (self ,load_path ,map_location ='cpu'):
        self .load_state_dict (torch .load (load_path ,map_location =map_location ))
