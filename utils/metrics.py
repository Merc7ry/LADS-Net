import numpy as np 
from scipy .ndimage import binary_erosion ,distance_transform_edt 

def binary_assd (pred_mask ,gt_mask ):
    pred_mask =pred_mask .astype (bool )
    gt_mask =gt_mask .astype (bool )

    if not pred_mask .any ()and not gt_mask .any ():
        return 0.0 
    if not pred_mask .any ()or not gt_mask .any ():
        return float ('nan')

    pred_surface =pred_mask ^binary_erosion (pred_mask ,border_value =0 )
    gt_surface =gt_mask ^binary_erosion (gt_mask ,border_value =0 )
    if not pred_surface .any ()and not gt_surface .any ():
        return 0.0 
    if not pred_surface .any ()or not gt_surface .any ():
        return float ('nan')

    dt_gt =distance_transform_edt (~gt_surface )
    dt_pred =distance_transform_edt (~pred_surface )
    return (dt_gt [pred_surface ].mean ()+dt_pred [gt_surface ].mean ())/2.0 

def compute_pixel_metrics (preds ,gts ,num_classes ,ignore_bg =None ):
    iou_list =[]
    f1_list =[]

    for cls in range (num_classes ):
        if ignore_bg is not None and cls ==ignore_bg :
            iou_list .append (float ('nan'))
            f1_list .append (float ('nan'))
            continue 

        pred_inds =preds ==cls 
        gt_inds =gts ==cls 

        if gt_inds .sum ()==0 and pred_inds .sum ()==0 :
            iou_list .append (float ('nan'))
            f1_list .append (float ('nan'))
            continue 

        intersection =np .logical_and (pred_inds ,gt_inds ).sum ()
        union =np .logical_or (pred_inds ,gt_inds ).sum ()

        tp =intersection 
        fp =pred_inds .sum ()-tp 
        fn =gt_inds .sum ()-tp 

        iou =intersection /union if union >0 else 0.0 
        f1 =2 *tp /(2 *tp +fp +fn )if (2 *tp +fp +fn )>0 else 0.0 
        iou_list .append (iou )
        f1_list .append (f1 )

    valid_iou =[x for x in iou_list if not np .isnan (x )]
    valid_f1 =[x for x in f1_list if not np .isnan (x )]
    miou =np .mean (valid_iou )if valid_iou else 0.0 
    mf1 =np .mean (valid_f1 )if valid_f1 else 0.0 
    pa =(preds ==gts ).sum ()/preds .size 
    return iou_list ,miou ,f1_list ,mf1 ,pa 

def compute_asd_metrics (pred_maps ,gt_maps ,num_classes ,ignore_index =255 ):
    asd_accum =[[]for _ in range (num_classes )]

    for pred_map ,gt_map in zip (pred_maps ,gt_maps ):
        valid =gt_map !=ignore_index 
        for cls in range (num_classes ):
            pred_mask =(pred_map ==cls )&valid 
            gt_mask =(gt_map ==cls )&valid 
            if not gt_mask .any ()and not pred_mask .any ():
                continue 
            asd =binary_assd (pred_mask ,gt_mask )
            if np .isfinite (asd ):
                asd_accum [cls ].append (asd )

    asd_list =[np .mean (vals )if vals else float ('nan')for vals in asd_accum ]
    valid_asd =[x for x in asd_list if not np .isnan (x )]
    masd =np .mean (valid_asd )if valid_asd else 0.0 
    return asd_list ,masd 

def compute_metrics (
preds ,
gts ,
num_classes ,
pred_maps =None ,
gt_maps =None ,
ignore_index =255 ,
ignore_bg =None ,
):
    iou_list ,miou ,f1_list ,mf1 ,pa =compute_pixel_metrics (
    preds ,gts ,num_classes ,ignore_bg =ignore_bg )
    if pred_maps is not None and gt_maps is not None :
        asd_list ,masd =compute_asd_metrics (
        pred_maps ,gt_maps ,num_classes ,ignore_index =ignore_index )
    else :
        asd_list =[float ('nan')]*num_classes 
        masd =0.0 
    return iou_list ,miou ,f1_list ,mf1 ,asd_list ,masd ,pa 
