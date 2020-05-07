from tensorfn.util.config import (
    read_config,
    preset_argparser,
    load_config,
    load_arg_config,
)


def load_wandb():
    try:
        import wandb

    except ImportError:
        wandb = None

    return wandb
