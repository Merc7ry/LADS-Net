import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets import get_dataset_info
from models import ALL_MODEL_NAMES, DEFAULT_ENCODER, build_model
from models.ladsnet_offset import resolve_kernel_sizes
from utils.complexity import compute_model_profile, format_complexity, print_complexity


def main():
    parser = argparse.ArgumentParser(description='Profile LADS-Net')
    parser.add_argument('--model', type=str, default='LADS_Net',
                        choices=ALL_MODEL_NAMES)
    parser.add_argument('--dataset', type=str, default='uavid',
                        choices=['uavid', 'aeroscapes', 'vdd', 'udd5',
                                 'vaihingen', 'potsdam', 'semantic_drone',
                                 'loveda'])
    parser.add_argument('--encoder', type=str, default=DEFAULT_ENCODER)
    parser.add_argument('--input_w', type=int, default=512)
    parser.add_argument('--input_h', type=int, default=512)
    parser.add_argument('--no_offset', action='store_true')
    parser.add_argument('--no_cls_offset', action='store_true')
    parser.add_argument('--no_feat_offset', action='store_true')
    parser.add_argument('--simple_decoder', action='store_true')
    parser.add_argument('--no_gag', action='store_true')
    parser.add_argument('--no_msair', action='store_true')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=None)
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--encoder_pretrain_path', type=str, default=None)
    parser.add_argument('--macs_device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--latency_device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()

    info = get_dataset_info(args.dataset)
    num_classes = info['num_classes']
    default_h, default_w = info.get('default_input_size', (args.input_h, args.input_w))
    if args.input_w == 512 and args.input_h == 512:
        args.input_w, args.input_h = default_w, default_h

    latency_device = None if args.latency_device == 'auto' else args.latency_device
    use_offset = not args.no_offset
    kernel_sizes = (
        resolve_kernel_sizes(args.kernel_sizes) if not args.simple_decoder else None)
    model_kwargs = dict(
        encoder=args.encoder,
        pretrain=not args.no_pretrain,
        pretrain_path=args.encoder_pretrain_path,
        use_offset=use_offset,
        img_size=args.input_h,
        simple_decoder=args.simple_decoder,
        use_gag=not args.no_gag,
        use_msair=not args.no_msair,
    )
    if not args.simple_decoder:
        model_kwargs['kernel_sizes'] = kernel_sizes
    if use_offset:
        model_kwargs['use_cls_offset'] = not args.no_cls_offset
        model_kwargs['use_feat_offset'] = not args.no_feat_offset
    model = build_model(args.model, num_classes=num_classes, **model_kwargs)

    stats = compute_model_profile(
        model,
        input_h=args.input_h,
        input_w=args.input_w,
        macs_device=args.macs_device,
        latency_device=latency_device,
        inference_only=True,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print('=' * 60)
    print(f'Model   : {args.model}')
    print(f'Dataset : {args.dataset} ({num_classes} classes)')
    print(f'Encoder : {args.encoder}')
    print_complexity(stats)
    print('=' * 60)
    print(format_complexity(stats))


if __name__ == '__main__':
    main()
