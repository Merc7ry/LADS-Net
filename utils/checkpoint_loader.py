import torch

from models import build_model
from models.ladsnet_offset import resolve_kernel_sizes
from utils.wgs import CombinatorialMutationsLossModule, resolve_wgs_ops
from utils.wgs.infer import resolve_infer_mode


def load_model_from_checkpoint(
    checkpoint_path,
    device,
    model_name=None,
    encoder=None,
    gag_kernel=3,
    no_offset=False,
    no_cls_offset=False,
    no_feat_offset=False,
    kernel_sizes=None,
    simple_decoder=False,
    no_gag=False,
    no_mkira=False,
    no_wgs=False,
    wgs_infer='wf',
    img_size=512,
):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    resolved_name = checkpoint.get('model_name', model_name or 'LADS_Net')
    if resolved_name in ('LADS_Net_PVT',):
        resolved_name = 'LADS_Net'
    resolved_encoder = checkpoint.get('encoder', encoder or 'pvt_v2_b2')

    dataset = checkpoint.get('dataset', 'uavid')
    use_offset = checkpoint.get('use_offset', not no_offset)
    use_cls_offset = checkpoint.get('use_cls_offset', not no_cls_offset)
    use_feat_offset = checkpoint.get('use_feat_offset', not no_feat_offset)
    resolved_kernel_sizes = checkpoint.get('kernel_sizes', kernel_sizes)
    if resolved_kernel_sizes is not None:
        resolved_kernel_sizes = resolve_kernel_sizes(resolved_kernel_sizes)
    simple_decoder = checkpoint.get('simple_decoder', simple_decoder)
    use_gag = checkpoint.get('use_gag', not no_gag)
    use_mkira = checkpoint.get('use_mkira', not no_mkira)
    use_wgs = checkpoint.get('use_wgs', not no_wgs)

    input_h = checkpoint.get('input_h', img_size)
    input_w = checkpoint.get('input_w', img_size)
    num_classes = checkpoint.get('num_classes')
    if num_classes is None:
        from datasets import get_dataset_info
        num_classes = get_dataset_info(dataset)['num_classes']

    model_kwargs = dict(
        encoder=resolved_encoder,
        pretrain=False,
        gag_kernel=gag_kernel,
        use_offset=use_offset,
        img_size=input_h,
        simple_decoder=simple_decoder,
        use_gag=use_gag,
        use_mkira=use_mkira,
    )
    if not simple_decoder and resolved_kernel_sizes is not None:
        model_kwargs['kernel_sizes'] = resolved_kernel_sizes
    if use_offset:
        model_kwargs['use_cls_offset'] = use_cls_offset
        model_kwargs['use_feat_offset'] = use_feat_offset
    model = build_model(
        resolved_name, num_classes=num_classes, **model_kwargs).to(device)

    wgs_module = None
    wgs_supervision = checkpoint.get('wgs_supervision', 'wgs')
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

    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    if wgs_module is not None:
        wgs_module.eval()

    meta = {
        'model_name': resolved_name,
        'encoder': resolved_encoder,
        'dataset': dataset,
        'num_classes': num_classes,
        'input_h': input_h,
        'input_w': input_w,
        'use_wgs': use_wgs,
        'wgs_infer': resolve_infer_mode(
            checkpoint.get('wgs_infer', wgs_infer),
            wgs_module),
    }
    return model, wgs_module, meta
