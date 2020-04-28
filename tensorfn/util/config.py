import ast
import argparse
from collections.abc import Mapping
from copy import deepcopy
import os
from pprint import pprint
import sys

from pyhocon import ConfigFactory, ConfigMissingException

from tensorfn.distributed import is_primary


def read_config(config_file, overrides=(), strict=False):
    conf = ConfigFactory.parse_file(config_file)

    for override in overrides:
        key, value = override.split("=", 1)

        try:
            original = conf.get(key)

        except ConfigMissingException:
            if strict:
                raise KeyError(f"Config '{key}' is missing")

        new = ast.literal_eval(value)

        if strict and type(original) != type(new):
            if not (
                isinstance(original, (tuple, list)) and isinstance(new, (tuple, list))
            ):
                expected_type = type(original).__name__
                new_type = type(new).__name__

                raise ValueError(
                    (
                        f"{new} of type '{new_type}' for {key} is incompatible with"
                        f" expected type '{expected_type}'"
                    )
                )

        conf.put(key, new)

    return conf.as_plain_ordered_dict()


def preset_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--ckpt", type=str)

    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_machine", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    return parser


def load_config(config_model, config, overrides=(), show=True):
    conf = config_model(**read_config(config, overrides=overrides))

    if show and is_primary():
        pprint(conf.dict())

    return conf


def load_arg_config(config_model, show=True):
    parser = preset_argparser()
    args = parser.parse_args()

    conf = load_config(config_model, args.conf, args.opts, show)

    conf.n_gpu = args.n_gpu
    conf.n_machine = args.n_machine
    conf.machine_rank = args.machine_rank
    conf.dist_url = args.dist_url
    conf.ckpt = args.ckpt

    return conf
