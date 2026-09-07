import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRETRAIN_ROOT_CANDIDATES = [
    os.path.join(_ROOT, 'pretrained_pth'),
    os.environ.get('LADSNET_PRETRAIN_DIR', ''),
]


def resolve_pretrain_root():
    for path in PRETRAIN_ROOT_CANDIDATES:
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    return os.path.abspath(os.path.join(_ROOT, 'pretrained_pth'))


def resolve_pretrain_file(explicit_path=None, *filenames, subdirs=('',)):
    if explicit_path:
        if os.path.isfile(explicit_path):
            return os.path.abspath(explicit_path)
        raise FileNotFoundError(
            f'Pretrained checkpoint not found: {explicit_path}')

    roots = [p for p in PRETRAIN_ROOT_CANDIDATES if p and os.path.isdir(p)]
    if not roots:
        roots = [os.path.join(_ROOT, 'pretrained_pth')]

    for root in roots:
        for subdir in subdirs:
            base = os.path.join(root, subdir) if subdir else root
            for name in filenames:
                path = os.path.join(base, name)
                if os.path.isfile(path):
                    return os.path.abspath(path)
    return None


def format_pretrain_search_hint(*filenames, subdirs=('',)):
    lines = []
    for root in PRETRAIN_ROOT_CANDIDATES:
        if not root:
            continue
        for subdir in subdirs:
            base = os.path.join(root, subdir) if subdir else root
            for name in filenames:
                lines.append(os.path.join(base, name))
    return '\n'.join(f'  - {p}' for p in lines)
