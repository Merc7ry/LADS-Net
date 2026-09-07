
import os 

import cv2 
import numpy as np 
import torch 
import torch .nn .functional as F 

COLORMAP_CHOICES ={
'jet':cv2 .COLORMAP_JET ,
'turbo':cv2 .COLORMAP_TURBO ,
'viridis':cv2 .COLORMAP_VIRIDIS ,
'hot':cv2 .COLORMAP_HOT ,
}

def logits_to_probs (logits ):
    if isinstance (logits ,(list ,tuple )):
        logits =logits [0 ]
    return torch .softmax (logits ,dim =1 )

def prob_map_to_heatmap (prob_map ,colormap =cv2 .COLORMAP_JET ):
    prob =np .asarray (prob_map ,dtype =np .float32 )
    prob =np .clip (prob ,0.0 ,1.0 )
    gray =(prob *255.0 ).astype (np .uint8 )
    colored_bgr =cv2 .applyColorMap (gray ,colormap )
    return cv2 .cvtColor (colored_bgr ,cv2 .COLOR_BGR2RGB )

def overlay_heatmap (image_rgb ,heatmap_rgb ,alpha =0.55 ):
    image_rgb =np .asarray (image_rgb ,dtype =np .uint8 )
    heatmap_rgb =np .asarray (heatmap_rgb ,dtype =np .uint8 )
    if image_rgb .shape [:2 ]!=heatmap_rgb .shape [:2 ]:
        heatmap_rgb =cv2 .resize (
        heatmap_rgb ,(image_rgb .shape [1 ],image_rgb .shape [0 ]),
        interpolation =cv2 .INTER_LINEAR )
    return cv2 .addWeighted (image_rgb ,1.0 -alpha ,heatmap_rgb ,alpha ,0.0 )

def _sanitize_name (name ):
    safe =''.join (ch if ch .isalnum ()or ch in '-_+'else '_'for ch in name )
    return safe .strip ('_')or 'class'

def save_single_class_heatmap (
image_rgb ,
prob_map ,
class_id ,
class_name ,
save_dir ,
colormap =cv2 .COLORMAP_JET ,
overlay_alpha =0.55 ,
save_overlay =True ,
):
    os .makedirs (save_dir ,exist_ok =True )
    heatmap_rgb =prob_map_to_heatmap (prob_map ,colormap =colormap )
    stem =f'heatmap_{class_id:02d}_{_sanitize_name(class_name)}'

    pure_path =os .path .join (save_dir ,f'{stem}.png')
    cv2 .imwrite (pure_path ,cv2 .cvtColor (heatmap_rgb ,cv2 .COLOR_RGB2BGR ))

    paths ={'heatmap':pure_path }
    if save_overlay :
        overlay_rgb =overlay_heatmap (image_rgb ,heatmap_rgb ,alpha =overlay_alpha )
        overlay_path =os .path .join (save_dir ,f'{stem}_overlay.png')
        cv2 .imwrite (overlay_path ,cv2 .cvtColor (overlay_rgb ,cv2 .COLOR_RGB2BGR ))
        paths ['overlay']=overlay_path 
    return paths 

def save_class_heatmaps (
image_rgb ,
probs ,
class_names ,
save_dir ,
colormap =cv2 .COLORMAP_JET ,
overlay_alpha =0.55 ,
save_overlay =True ,
ignore_index =255 ,
gt_mask =None ,
):
    os .makedirs (save_dir ,exist_ok =True )

    if isinstance (probs ,torch .Tensor ):
        probs =probs .detach ().cpu ().numpy ()
    probs =np .asarray (probs ,dtype =np .float32 )

    input_path =os .path .join (save_dir ,'input.png')
    cv2 .imwrite (input_path ,cv2 .cvtColor (image_rgb ,cv2 .COLOR_RGB2BGR ))

    saved ={'input':input_path ,'classes':[]}
    num_classes =probs .shape [0 ]
    for class_id in range (num_classes ):
        class_name =class_names [class_id ]if class_id <len (class_names )else f'class_{class_id}'
        prob_map =probs [class_id ]
        if gt_mask is not None and ignore_index is not None :
            valid =np .asarray (gt_mask )!=ignore_index 
            if not valid .any ():
                continue 
        paths =save_single_class_heatmap (
        image_rgb =image_rgb ,
        prob_map =prob_map ,
        class_id =class_id ,
        class_name =class_name ,
        save_dir =save_dir ,
        colormap =colormap ,
        overlay_alpha =overlay_alpha ,
        save_overlay =save_overlay ,
        )
        saved ['classes'].append ({
        'class_id':class_id ,
        'class_name':class_name ,
        **paths ,
        })

    grid_path =save_class_heatmap_grid (
    image_rgb =image_rgb ,
    probs =probs ,
    class_names =class_names ,
    save_path =os .path .join (save_dir ,'heatmap_grid.png'),
    colormap =colormap ,
    overlay_alpha =overlay_alpha ,
    )
    saved ['grid']=grid_path 
    return saved 

def save_class_heatmap_grid (
image_rgb ,
probs ,
class_names ,
save_path ,
colormap =cv2 .COLORMAP_JET ,
overlay_alpha =0.55 ,
thumb_size =256 ,
):
    if isinstance (probs ,torch .Tensor ):
        probs =probs .detach ().cpu ().numpy ()
    probs =np .asarray (probs ,dtype =np .float32 )

    h ,w =image_rgb .shape [:2 ]
    num_classes =probs .shape [0 ]
    panels =[]

    def _resize_panel (panel ):
        return cv2 .resize (panel ,(thumb_size ,thumb_size ),interpolation =cv2 .INTER_LINEAR )

    def _titled (panel ,title ):
        header_h =28 
        header =np .ones ((header_h ,thumb_size ,3 ),dtype =np .uint8 )*255 
        cv2 .putText (
        header ,title ,(8 ,20 ),cv2 .FONT_HERSHEY_SIMPLEX ,0.5 ,
        (0 ,0 ,0 ),1 ,cv2 .LINE_AA )
        body =_resize_panel (panel )
        return np .vstack ([header ,body ])

    panels .append (_titled (image_rgb ,'Input'))

    for class_id in range (num_classes ):
        class_name =class_names [class_id ]if class_id <len (class_names )else f'class_{class_id}'
        heatmap_rgb =prob_map_to_heatmap (probs [class_id ],colormap =colormap )
        overlay_rgb =overlay_heatmap (image_rgb ,heatmap_rgb ,alpha =overlay_alpha )
        title =f'{class_id}:{class_name[:12]}'
        panels .append (_titled (overlay_rgb ,title ))

    cols =3 
    rows =int (np .ceil (len (panels )/cols ))
    while len (panels )<rows *cols :
        panels .append (np .ones ((thumb_size +28 ,thumb_size ,3 ),dtype =np .uint8 )*255 )

    row_imgs =[]
    for row_idx in range (rows ):
        row_panels =panels [row_idx *cols :(row_idx +1 )*cols ]
        row_imgs .append (np .hstack (row_panels ))
    canvas =np .vstack (row_imgs )

    os .makedirs (os .path .dirname (save_path )or '.',exist_ok =True )
    cv2 .imwrite (save_path ,cv2 .cvtColor (canvas ,cv2 .COLOR_RGB2BGR ))
    return save_path 

def predict_class_probs (model ,image_tensor ,wgs_module =None ,wgs_infer_mode ='wf'):
    from utils .wgs .infer import wgs_predict_logits 

    model .eval ()
    if wgs_module is not None :
        wgs_module .eval ()
    with torch .no_grad ():
        logits =wgs_predict_logits (
        model ,image_tensor ,wgs_module ,mode =wgs_infer_mode )
    probs =logits_to_probs (logits )[0 ]
    if probs .shape [-2 :]!=image_tensor .shape [-2 :]:
        probs =F .interpolate (
        probs .unsqueeze (0 ),
        size =image_tensor .shape [-2 :],
        mode ='bilinear',
        align_corners =False ,
        )[0 ]
    return probs .cpu ().numpy ()
