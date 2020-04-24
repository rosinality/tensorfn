import ast
import argparse
from collections.abc import Mapping
from copy import deepcopy
from pprint import pprint

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

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--ckpt", type=str)
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

    conf.ckpt = args.ckpt
    conf.local_rank = args.local_rank

    return conf
