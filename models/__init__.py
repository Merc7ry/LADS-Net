from .ladsnet_pvt_network import LADS_Net, PVT_ENCODER_CONFIG
from .ladsnet_offset import LADSNetOffsetOutput

ALL_MODEL_NAMES = ['LADS_Net']
DEFAULT_ENCODER = 'pvt_v2_b2'

__all__ = [
    'LADS_Net',
    'PVT_ENCODER_CONFIG',
    'LADSNetOffsetOutput',
    'ALL_MODEL_NAMES',
    'build_model',
    'model_output',
    'model_output_maps',
    'DEFAULT_ENCODER',
]


def build_model(name='LADS_Net', num_classes=8, use_offset=True, **kwargs):
    name = name.strip()
    if name != 'LADS_Net':
        raise ValueError(f'Unknown model `{name}`. Available: {ALL_MODEL_NAMES}')
    return LADSNetOffsetOutput.build(
        name,
        num_classes=num_classes,
        use_offset=use_offset,
        **kwargs,
    )


def model_output_maps(model, x):
    from utils.wgs.infer import model_output_maps as _maps
    return _maps(model, x)


def model_output(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 2 and isinstance(outputs[0], (list, tuple)):
            return outputs[1]
        return outputs[0]
    return outputs
