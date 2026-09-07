import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets import build_dataset, describe_data_root, get_dataset_info, val_split_status
from datasets.patch_layout import PATCH_DATASETS
from models import (
    ALL_MODEL_NAMES,
    DEFAULT_ENCODER,
    build_model,
    model_output,
    model_output_maps,
)
from models.ladsnet_offset import resolve_kernel_sizes
from utils import JointLoss, compute_metrics
from utils.complexity import compute_model_profile, print_complexity
from utils.wgs import (
    resolve_wgs_ops,
    CombinatorialMutationsLossModule,
    WGSDiceLoss,
    logits_from_maps,
)


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


def _seg_loss(criterion, outputs, masks):
    if isinstance(outputs, (list, tuple)):
        return sum(criterion(o, masks) for o in outputs)
    return criterion(outputs, masks)


def _build_criterion(args, num_classes, ignore_index, device):
    if args.loss == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
    return JointLoss(num_classes=num_classes, ignore_index=ignore_index).to(device)


def _build_optimizer(args, model, extra_modules=None):
    params = list(model.parameters())
    if extra_modules:
        for mod in extra_modules:
            if mod is not None:
                params += list(mod.parameters())
    if args.optimizer == 'adam':
        return optim.Adam(params, lr=args.lr)
    return optim.AdamW(params, lr=args.lr, weight_decay=1e-4)


def _build_wgs_module(args, num_classes, device):
    supervision = args.wgs_supervision
    operations = resolve_wgs_ops(args.wgs_ops, supervision=supervision)
    return CombinatorialMutationsLossModule(
        original_num_maps=4,
        num_classes=num_classes,
        selecetd_num_maps=4,
        operations=operations,
        use_learnable_weights=True,
        supervision=supervision,
    ).to(device)


def _build_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
    return None


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None,
                    wgs_module=None, wgs_ce=None, wgs_dice=None):
    from models.ladsnet_offset import offset_logits_from_model

    model.train()
    if wgs_module is not None:
        wgs_module.train()
    total_loss = 0.0
    valid_batches = 0
    pbar = tqdm(dataloader, desc='Train', ncols=100)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            if wgs_module is not None:
                output_maps = model_output_maps(model, images)
                loss, _, _ = wgs_module(output_maps, masks, wgs_ce, wgs_dice)
                offset_logits = offset_logits_from_model(model)
                if offset_logits is not None:
                    loss = loss + _seg_loss(criterion, offset_logits, masks)
            else:
                outputs = model(images)
                offset_logits = offset_logits_from_model(model)
                if offset_logits is not None and isinstance(outputs, (list, tuple)):
                    loss = _seg_loss(criterion, offset_logits, masks)
                    if len(outputs) > 1:
                        loss = loss + _seg_loss(criterion, outputs[1:], masks)
                else:
                    loss = _seg_loss(
                        criterion,
                        offset_logits if offset_logits is not None
                        else model_output(outputs),
                        masks)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f'Warning: NaN/Inf loss ({loss_val}), skip batch.')
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if wgs_module is not None:
                torch.nn.utils.clip_grad_norm_(wgs_module.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if wgs_module is not None:
                torch.nn.utils.clip_grad_norm_(wgs_module.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss_val
        valid_batches += 1
        pbar.set_postfix({'Loss': f'{loss_val:.4f}'})

    return total_loss / valid_batches if valid_batches else 0.0


@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes, ignore_index,
             wgs_module=None, wgs_ce=None, wgs_dice=None, wgs_infer_mode='wf'):
    from models.ladsnet_offset import maps_with_final_offset, offset_logits_from_model

    model.eval()
    if wgs_module is not None:
        wgs_module.eval()
    total_loss = 0.0
    valid_batches = 0
    all_preds = []
    all_gts = []
    pred_maps = []
    gt_maps = []

    for images, masks in tqdm(dataloader, desc='Val', ncols=100):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if wgs_module is not None:
            output_maps = model_output_maps(model, images)
            loss_val = wgs_module(output_maps, masks, wgs_ce, wgs_dice)[0].item()
            offset_logits = offset_logits_from_model(model)
            if offset_logits is not None:
                loss_val = loss_val + _seg_loss(criterion, offset_logits, masks).item()
            pred_maps_logits = maps_with_final_offset(model, output_maps)
            logits = logits_from_maps(pred_maps_logits, wgs_module, mode=wgs_infer_mode)
        else:
            outputs = model(images)
            offset_logits = offset_logits_from_model(model)
            logits = offset_logits if offset_logits is not None else model_output(outputs)
            loss_val = _seg_loss(criterion, logits, masks).item()
        if not math.isfinite(loss_val):
            continue
        total_loss += loss_val
        valid_batches += 1
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_gts.append(masks.cpu())
        pred_maps.append(preds.cpu().numpy())
        gt_maps.append(masks.cpu().numpy())

    if valid_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    val_loss = total_loss / valid_batches
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_gts = torch.cat(all_gts, dim=0).numpy()
    _, miou, _, mf1, _, masd, pa = compute_metrics(
        all_preds, all_gts, num_classes,
        pred_maps=pred_maps, gt_maps=gt_maps, ignore_index=ignore_index)
    return val_loss, miou, mf1, masd, pa


def main():
    parser = argparse.ArgumentParser(description='Train LADS-Net')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='uavid',
                        choices=['uavid', 'aeroscapes', 'vdd', 'udd5',
                                 'vaihingen', 'potsdam', 'semantic_drone',
                                 'loveda'])
    parser.add_argument('--model', type=str, default='LADS_Net',
                        choices=ALL_MODEL_NAMES)
    parser.add_argument('--encoder', type=str, default=DEFAULT_ENCODER)
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--encoder_pretrain_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--loss', type=str, default='joint', choices=['ce', 'joint'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine'])
    parser.add_argument('--input_w', type=int, default=512)
    parser.add_argument('--input_h', type=int, default=512)
    parser.add_argument('--gag_kernel', type=int, default=3)
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=None)
    parser.add_argument('--simple_decoder', action='store_true')
    parser.add_argument('--no_gag', action='store_true')
    parser.add_argument('--no_mkira', action='store_true')
    parser.add_argument('--no_offset', action='store_true')
    parser.add_argument('--no_cls_offset', action='store_true')
    parser.add_argument('--no_feat_offset', action='store_true')
    parser.add_argument('--no_wgs', action='store_true')
    parser.add_argument('--wgs_supervision', type=str, default='wgs',
                        choices=['wgs', 'mutation', 'deep_supervision'])
    parser.add_argument('--wgs_infer', type=str, default='wf',
                        choices=['p4', 'wf', 'last'])
    parser.add_argument('--wgs_ops', type=str, nargs='+', default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--name', type=str, default='LADSNet')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--profile_macs', action='store_true')
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(describe_data_root(args.data_dir))

    info = get_dataset_info(args.dataset)
    num_classes = info['num_classes']
    ignore_index = info['ignore_index']
    default_h, default_w = info.get('default_input_size', (args.input_h, args.input_w))
    if args.input_w == 512 and args.input_h == 512 and args.dataset in PATCH_DATASETS:
        args.input_w, args.input_h = default_w, default_h

    train_set = build_dataset(
        args.dataset, args.data_dir, split='train',
        input_w=args.input_w, input_h=args.input_h)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(train_set, **loader_kwargs)

    val_loader = None
    val_enabled, val_reason = val_split_status(args.data_dir)
    if val_enabled:
        val_set = build_dataset(
            args.dataset, args.data_dir, split='val',
            input_w=args.input_w, input_h=args.input_h)
        val_bs = _cap_val_batch_size(
            args.dataset, args.input_w, args.input_h, args.batch_size)
        val_loader = DataLoader(
            val_set, batch_size=val_bs, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        print(
            f'Validation enabled: {len(val_set)} samples, '
            f'patch={args.input_w}x{args.input_h}, batch_size={val_bs}, '
            f'every {args.val_every} epoch(s)')
    else:
        print(f'Validation disabled: {val_reason}')

    use_offset = not args.no_offset
    use_cls_offset = not args.no_cls_offset
    use_feat_offset = not args.no_feat_offset
    use_wgs = not args.no_wgs
    simple_decoder = args.simple_decoder
    use_gag = not args.no_gag
    use_mkira = not args.no_mkira
    kernel_sizes = (
        resolve_kernel_sizes(args.kernel_sizes) if not simple_decoder else None)
    if simple_decoder:
        decoder_name = 'simple-upsampling'
    else:
        block = 'IRB' if kernel_sizes == (3,) else 'MKIR'
        parts = [block]
        if use_gag:
            parts.append('GAG')
        if use_mkira:
            parts.append('MKIRA')
        decoder_name = '+'.join(parts)

    head_name = '1x1 conv (all heads)'
    if use_offset:
        branches = []
        if use_cls_offset:
            branches.append('cls')
        if use_feat_offset:
            branches.append('feat')
        branch_label = '+'.join(branches) if branches else 'none'
        head_name = f'OffsetLearning(P4 predict, {branch_label})'
    wgs_ops = (
        resolve_wgs_ops(args.wgs_ops, supervision=args.wgs_supervision)
        if use_wgs else None)
    output_name = (
        f'WGS [{args.wgs_supervision}] ops={wgs_ops} / infer={args.wgs_infer}'
        if use_wgs else 'P4-only supervision')
    print(
        f'Initializing {args.model} (encoder={args.encoder}, decoder={decoder_name}, '
        f'head={head_name}, supervision={output_name})...')

    model_kwargs = dict(
        encoder=args.encoder,
        pretrain=not args.no_pretrain,
        pretrain_path=args.encoder_pretrain_path,
        gag_kernel=args.gag_kernel,
        use_offset=use_offset,
        img_size=args.input_h,
        simple_decoder=simple_decoder,
        use_gag=use_gag,
        use_mkira=use_mkira,
    )
    if not simple_decoder:
        model_kwargs['kernel_sizes'] = kernel_sizes
    if use_offset:
        model_kwargs['use_cls_offset'] = use_cls_offset
        model_kwargs['use_feat_offset'] = use_feat_offset
    model = build_model(args.model, num_classes=num_classes, **model_kwargs).to(device)

    if args.profile_macs:
        print_complexity(compute_model_profile(
            model, input_h=args.input_h, input_w=args.input_w,
            macs_device='cpu', latency_device=device, inference_only=True),
            prefix='Complexity: ')

    wgs_module = None
    wgs_ce = None
    wgs_dice = None
    if use_wgs:
        wgs_module = _build_wgs_module(args, num_classes, device)
        wgs_ce = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
        wgs_dice = WGSDiceLoss(num_classes, ignore_index=ignore_index).to(device)

    criterion = _build_criterion(args, num_classes, ignore_index, device)
    optimizer = _build_optimizer(args, model, extra_modules=[wgs_module])
    scheduler = _build_scheduler(args, optimizer)
    scaler = amp.GradScaler() if args.amp else None

    start_epoch = 1
    best_miou = -1.0
    best_train_loss = float('inf')
    track_val_miou = val_loader is not None

    if args.resume and os.path.exists(args.resume):
        print(f'Resuming from: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_miou = checkpoint.get('best_miou', -1.0)
        best_train_loss = checkpoint.get('best_train_loss', float('inf'))
        track_val_miou = checkpoint.get('track_val_miou', track_val_miou)
        if wgs_module is not None and 'wgs_loss_state_dict' in checkpoint:
            wgs_module.load_state_dict(checkpoint['wgs_loss_state_dict'])

    save_root = os.path.join('runs', args.name)
    os.makedirs(save_root, exist_ok=True)

    print(f'\nStart training: samples={len(train_set)}, patch={args.input_w}x{args.input_h}')

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\nEpoch [{epoch}/{args.epochs}] LR: {optimizer.param_groups[0]["lr"]:.8f}')
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            wgs_module=wgs_module, wgs_ce=wgs_ce, wgs_dice=wgs_dice)
        if scheduler is not None:
            scheduler.step()
        print(f'Train Loss: {train_loss:.4f}')

        val_loss = None
        val_miou = None
        val_mf1 = None
        val_masd = None
        if val_loader is not None and epoch % args.val_every == 0:
            val_loss, val_miou, val_mf1, val_masd, val_pa = validate(
                model, val_loader, criterion, device, num_classes,
                ignore_index, wgs_module=wgs_module,
                wgs_ce=wgs_ce, wgs_dice=wgs_dice,
                wgs_infer_mode=args.wgs_infer)
            print(
                f'Val Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, '
                f'mF1: {val_mf1:.4f}, mASD: {val_masd:.4f}, PA: {val_pa:.4f}')

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'val_mf1': val_mf1,
            'val_masd': val_masd,
            'best_miou': best_miou,
            'best_train_loss': best_train_loss,
            'track_val_miou': track_val_miou,
            'dataset': args.dataset,
            'num_classes': num_classes,
            'model_name': args.model,
            'encoder': args.encoder,
            'use_offset': use_offset,
            'use_cls_offset': use_cls_offset if use_offset else None,
            'use_feat_offset': use_feat_offset if use_offset else None,
            'kernel_sizes': list(kernel_sizes) if kernel_sizes else None,
            'simple_decoder': simple_decoder,
            'use_gag': use_gag,
            'use_mkira': use_mkira,
            'use_wgs': use_wgs,
            'wgs_supervision': args.wgs_supervision if use_wgs else None,
            'wgs_operations': wgs_ops,
            'wgs_infer': args.wgs_infer if use_wgs else None,
            'input_w': args.input_w,
            'input_h': args.input_h,
            'optimizer': args.optimizer,
            'loss': args.loss,
            'scheduler': args.scheduler,
            'lr': args.lr,
        }
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if wgs_module is not None:
            ckpt['wgs_loss_state_dict'] = wgs_module.state_dict()
        torch.save(ckpt, os.path.join(save_root, 'latest.pth'))

        improved = False
        if track_val_miou and val_miou is not None and val_miou > best_miou:
            best_miou = val_miou
            ckpt['best_miou'] = best_miou
            improved = True
        elif (not track_val_miou) and train_loss < best_train_loss:
            best_train_loss = train_loss
            ckpt['best_train_loss'] = best_train_loss
            improved = True
        if improved:
            torch.save(ckpt, os.path.join(save_root, 'best.pth'))
            print(f'Saved best.pth (mIoU={best_miou:.4f})')

    print('Training finished.')


if __name__ == '__main__':
    main()
