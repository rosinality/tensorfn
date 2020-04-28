from typing import Tuple, Union

from pydantic import BaseModel, validator, StrictStr, StrictFloat, StrictBool
from torch import optim

from tensorfn.config import Config
from tensorfn import optim as tensor_optim


class SGD(Config):
    type: StrictStr

    lr: StrictFloat
    momentum: StrictFloat = 0.0
    dampening: StrictFloat = 0.0
    weight_decay: StrictFloat = 0.0
    nesterov: StrictBool = False

    @validator("type")
    def check_type(cls, v):
        if v != "sgd":
            raise ValueError("Optimizer type not match for sgd")

        return v

    def make(self, params):
        return optim.SGD(
            params,
            self.lr,
            self.momentum,
            self.dampening,
            self.weight_decay,
            self.nesterov,
        )


class Adam(Config):
    type: StrictStr

    lr: StrictFloat = 0.001
    betas: Tuple[StrictFloat, StrictFloat] = (0.9, 0.999)
    eps: StrictFloat = 1e-8
    weight_decay: StrictFloat = 0
    amsgrad: StrictBool = False

    @validator("type")
    def check_type(cls, v):
        if v != "adam":
            raise ValueError("Optimizer type not match for adam")

        return v

    def make(self, params):
        return optim.Adam(
            params, self.lr, self.betas, self.eps, self.weight_decay, self.amsgrad
        )


class AdamW(Config):
    type: StrictStr

    lr: StrictFloat = 0.001
    betas: Tuple[StrictFloat, StrictFloat] = (0.9, 0.999)
    eps: StrictFloat = 1e-8
    weight_decay: StrictFloat = 0
    amsgrad: StrictBool = False

    @validator("type")
    def check_type(cls, v):
        if v != "adamw":
            raise ValueError("Optimizer type not match for adam")

        return v

    def make(self, params):
        return optim.AdamW(
            params, self.lr, self.betas, self.eps, self.weight_decay, self.amsgrad
        )


class LAMB(Config):
    type: StrictStr

    lr: StrictFloat = 0.001
    betas: Tuple[StrictFloat, StrictFloat] = (0.9, 0.999)
    eps: StrictFloat = 1e-6
    weight_decay: StrictFloat = 0

    @validator("type")
    def check_type(cls, v):
        if v != "lamb":
            raise ValueError("Optimizer type not match for adam")

        return v

    def make(self, params):
        return tensor_optim.LAMB(
            params, self.lr, self.betas, self.eps, self.weight_decay
        )


def make_optimizer(config, params):
    return config.make(params)


Optimizer = Union[SGD, Adam, AdamW, LAMB]
