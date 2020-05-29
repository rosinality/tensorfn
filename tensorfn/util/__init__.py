from tensorfn.util.config import (
    read_config,
    preset_argparser,
    load_config,
    load_arg_config,
)
from tensorfn.util.ensure import ensure_tuple
from tensorfn.util.lazy_extension import LazyExtension


def load_wandb():
    try:
        import wandb

    except ImportError:
        wandb = None

    return wandb
