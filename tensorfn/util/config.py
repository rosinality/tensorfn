import ast
import argparse
from collections.abc import Mapping
from copy import deepcopy
import os
from pprint import pprint
import sys

from pyhocon import ConfigFactory, ConfigTree

from tensorfn.distributed import is_primary


def read_config(config_file, overrides=(), strict=False):
    conf = ConfigFactory.parse_file(config_file)

    if len(overrides) > 0:
        for override in overrides:
            conf_overrides = ConfigFactory.parse_string(override)
            conf = ConfigTree.merge_configs(conf, conf_overrides)

    return conf.as_plain_ordered_dict()


def preset_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--ckpt", type=str)

    parser = add_distributed_args(parser)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    return parser


def add_distributed_args(parser):
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_machine", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    return parser


def load_config(config_model, config, overrides=(), show=True):
    conf = config_model(**read_config(config, overrides=overrides))

    if show and is_primary():
        pprint(conf.dict())

    return conf


def load_arg_config(config_model, show=False):
    parser = preset_argparser()
    args = parser.parse_args()

    conf = load_config(config_model, args.conf, args.opts, show)

    conf.n_gpu = args.n_gpu
    conf.n_machine = args.n_machine
    conf.machine_rank = args.machine_rank
    conf.dist_url = args.dist_url
    conf.ckpt = args.ckpt

    return conf
