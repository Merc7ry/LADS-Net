import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets import (
    aeroscapes_mask_to_rgb,
    build_dataset,
    get_dataset_info,
    isprs_mask_to_rgb,
    loveda_mask_to_rgb,
    semantic_drone_mask_to_rgb,
    uavid_mask_to_rgb,
    udd5_mask_to_rgb,
    vdd_mask_to_rgb,
)
from datasets.patch_layout import PATCH_DATASETS
from models import ALL_MODEL_NAMES, DEFAULT_ENCODER, build_model, model_output
from models.ladsnet_offset import resolve_kernel_sizes
from utils import compute_metrics
from utils.complexity import compute_model_profile, print_complexity
from utils.wgs import CombinatorialMutationsLossModule, resolve_wgs_ops, wgs_predict_logits
from utils.wgs.infer import resolve_infer_mode
from utils.viz import save_viz_triplet


def _pred_to_rgb(dataset, pred, palette, ignore_index):
    if dataset == 'aeroscapes':
        return aeroscapes_mask_to_rgb(pred, palette, ignore_index)
    if dataset == 'vdd':
        return vdd_mask_to_rgb(pred, palette, ignore_index)
    if dataset == 'udd5':
        return udd5_mask_to_rgb(pred, palette, ignore_index)
    if dataset in ('vaihingen', 'potsdam'):
        return isprs_mask_to_rgb(pred, palette, ignore_index)
    if dataset == 'semantic_drone':
        return semantic_drone_mask_to_rgb(pred, palette, ignore_index)
    if dataset == 'loveda':
        return loveda_mask_to_rgb(pred, palette, ignore_index)
    return uavid_mask_to_rgb(pred, palette, ignore_index)


def _cap_val_batch_size(dataset, input_w, input_h, batch_size):
    if dataset not in PATCH_DATASETS:
        return batch_size
    pixels = input_w * input_h
    if pixels >= 1280 * 704:
        cap = 1
    elif pixels >= 512 * 512:
        cap = 2
    else:
        cap = 4
    if batch_size > cap:
        print(f'Val batch capped: {batch_size} -> {cap} ({input_w}x{input_h})')
        return cap
    return batch_size


@torch.no_grad()
def run_validation(model, dataloader, device, num_classes, dataset, palette,
                   ignore_index=255, max_viz_count=50, wgs_module=None,
                   wgs_infer_mode='wf', viz_output='validation_viz_results'):
    model.eval()
    if wgs_module is not None:
        wgs_module.eval()
    all_preds = []
    all_gts = []
    pred_maps = []
    gt_maps = []
    viz_count = 0
    os.makedirs(viz_output, exist_ok=True)

    for images, masks in tqdm(dataloader, desc='Val', ncols=100):
        images = images.to(device, non_blocking=True)
        masks_cpu = masks
        if wgs_module is not None:
            outputs = wgs_predict_logits(
                model, images, wgs_module, mode=wgs_infer_mode)
        else:
            outputs = model(images)
            outputs = model_output(outputs)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_gts.append(masks_cpu)
        pred_maps.append(preds.numpy())
        gt_maps.append(masks_cpu.numpy())
        if viz_count < max_viz_count:
            for i in range(preds.shape[0]):
                if viz_count >= max_viz_count:
                    break
                save_viz_triplet(
                    images[i].cpu(),
                    _pred_to_rgb(dataset, preds[i].numpy(), palette, ignore_index),
                    _pred_to_rgb(dataset, masks_cpu[i].numpy(), palette, ignore_index),
                    os.path.join(viz_output, f'{viz_count:04d}.png'),
                )
                viz_count += 1

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_gts = torch.cat(all_gts, dim=0).numpy()
    return compute_metrics(
        all_preds, all_gts, num_classes,
        pred_maps=pred_maps, gt_maps=gt_maps, ignore_index=ignore_index)


def main():
    parser = argparse.ArgumentParser(description='Validate LADS-Net')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='uavid',
                        choices=['uavid', 'aeroscapes', 'vdd', 'udd5',
                                 'vaihingen', 'potsdam', 'semantic_drone',
                                 'loveda'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='LADS_Net',
                        choices=ALL_MODEL_NAMES)
    parser.add_argument('--encoder', type=str, default=DEFAULT_ENCODER)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_w', type=int, default=512)
    parser.add_argument('--input_h', type=int, default=512)
    parser.add_argument('--gag_kernel', type=int, default=3)
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=None)
    parser.add_argument('--simple_decoder', action='store_true')
    parser.add_argument('--no_gag', action='store_true')
    parser.add_argument('--no_msair', action='store_true')
    parser.add_argument('--no_offset', action='store_true')
    parser.add_argument('--no_cls_offset', action='store_true')
    parser.add_argument('--no_feat_offset', action='store_true')
    parser.add_argument('--no_wgs', action='store_true')
    parser.add_argument('--wgs_infer', type=str, default='wf',
                        choices=['p4', 'wf', 'last'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--viz_count', type=int, default=50)
    parser.add_argument('--viz_output', type=str, default='validation_viz_results')
    parser.add_argument('--profile_macs', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f'Error: checkpoint not found: {args.model_path}')
        return

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)

    model_name = checkpoint.get('model_name', args.model)
    if model_name in ('LADS_Net_PVT',):
        model_name = 'LADS_Net'
    encoder = checkpoint.get('encoder', args.encoder)
    dataset = checkpoint.get('dataset', args.dataset)
    use_offset = checkpoint.get('use_offset', not args.no_offset)
    use_cls_offset = checkpoint.get('use_cls_offset', not args.no_cls_offset)
    use_feat_offset = checkpoint.get('use_feat_offset', not args.no_feat_offset)
    kernel_sizes = checkpoint.get('kernel_sizes', args.kernel_sizes)
    if kernel_sizes is not None:
        kernel_sizes = resolve_kernel_sizes(kernel_sizes)
    simple_decoder = checkpoint.get('simple_decoder', args.simple_decoder)
    use_gag = checkpoint.get('use_gag', not args.no_gag)
    use_msair = checkpoint.get(
        'use_msair', checkpoint.get('use_mkira', not args.no_msair))
    use_wgs = checkpoint.get('use_wgs', not args.no_wgs)
    wgs_supervision = checkpoint.get('wgs_supervision', 'wgs')
    wgs_infer_raw = checkpoint.get('wgs_infer', args.wgs_infer)

    if 'input_w' in checkpoint and args.input_w == 512:
        args.input_w = checkpoint['input_w']
    if 'input_h' in checkpoint and args.input_h == 512:
        args.input_h = checkpoint['input_h']

    info = get_dataset_info(dataset)
    num_classes = info['num_classes']
    palette = info['palette']
    ignore_index = info['ignore_index']
    default_h, default_w = info.get('default_input_size', (args.input_h, args.input_w))
    if args.input_w == 512 and args.input_h == 512 and dataset in PATCH_DATASETS:
        args.input_w, args.input_h = default_w, default_h

    args.batch_size = _cap_val_batch_size(
        dataset, args.input_w, args.input_h, args.batch_size)
    print(f'Validation input: {args.input_w}x{args.input_h}, batch_size={args.batch_size}')

    val_set = build_dataset(
        dataset, args.data_dir, split=args.split,
        input_w=args.input_w, input_h=args.input_h)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    model_kwargs = dict(
        encoder=encoder,
        pretrain=False,
        gag_kernel=args.gag_kernel,
        use_offset=use_offset,
        img_size=args.input_h,
        simple_decoder=simple_decoder,
        use_gag=use_gag,
        use_msair=use_msair,
    )
    if not simple_decoder and kernel_sizes is not None:
        model_kwargs['kernel_sizes'] = kernel_sizes
    if use_offset:
        model_kwargs['use_cls_offset'] = use_cls_offset
        model_kwargs['use_feat_offset'] = use_feat_offset
    model = build_model(model_name, num_classes=num_classes, **model_kwargs).to(device)

    wgs_module = None
    if use_wgs:
        ops = resolve_wgs_ops(
            checkpoint.get('wgs_operations'),
            supervision=wgs_supervision)
        wgs_module = CombinatorialMutationsLossModule(
            original_num_maps=4,
            num_classes=num_classes,
            selecetd_num_maps=4,
            operations=ops,
            use_learnable_weights=True,
            supervision=wgs_supervision,
        ).to(device)
        if 'wgs_loss_state_dict' in checkpoint:
            wgs_module.load_state_dict(checkpoint['wgs_loss_state_dict'])

    wgs_infer = resolve_infer_mode(wgs_infer_raw, wgs_module)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    print(f'Loaded checkpoint: {args.model_path} (encoder={encoder}, dataset={dataset})')

    if args.profile_macs:
        print_complexity(compute_model_profile(
            model, input_h=args.input_h, input_w=args.input_w,
            macs_device='cpu', latency_device=device, inference_only=True))

    print(f' Model   : {model_name} + {encoder}')
    metrics = run_validation(
        model, val_loader, device, num_classes, dataset, palette,
        ignore_index=ignore_index, max_viz_count=args.viz_count,
        wgs_module=wgs_module, wgs_infer_mode=wgs_infer,
        viz_output=args.viz_output)
    print(metrics)


if __name__ == '__main__':
    main()
