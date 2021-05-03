from typing import Tuple, Union

from pydantic import BaseModel, validator, StrictStr, StrictBool
from torch import optim

from tensorfn.config import Config, TypedConfig, override
from tensorfn import optim as tensor_optim


class SGD(Config):
    type: StrictStr

    lr: float
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: StrictBool = False

    @validator("type")
    def check_type(cls, v):
        if v != "sgd":
            raise ValueError("Optimizer type not match for sgd")

        return v

    def make(self, params, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )

        return optim.SGD(params, **argument)


class Adam(Config):
    type: StrictStr

    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: StrictBool = False

    @validator("type")
    def check_type(cls, v):
        if v != "adam":
            raise ValueError("Optimizer type not match for adam")

        return v

    def make(self, params, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

        return optim.Adam(params, **argument)


class AdamW(Config):
    type: StrictStr

    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: StrictBool = False

    @validator("type")
    def check_type(cls, v):
        if v != "adamw":
            raise ValueError("Optimizer type not match for adam")

        return v

    def make(self, params, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

        return optim.AdamW(params, **argument)


class LAMB(Config):
    type: StrictStr

    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-6
    weight_decay: float = 0

    @validator("type")
    def check_type(cls, v):
        if v != "lamb":
            raise ValueError("Optimizer type not match for adam")

        return v

    def make(self, params, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        return tensor_optim.LAMB(params, **argument)


class RMSpropTF(TypedConfig):
    __type__ = "rmsprop_tf"

    lr: float = 0.01
    alpha: float = 0.9
    eps: float = 1e-10
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: StrictBool = False
    decoupled_decay: StrictBool = False
    lr_in_momentum: StrictBool = True

    def make(self, params, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
            decoupled_decay=self.decoupled_decay,
            lr_in_momentum=self.lr_in_momentum,
        )

        return tensor_optim.RMSpropTF(params, **argument)


def make_optimizer(config, params):
    return config.make(params)


Optimizer = Union[SGD, Adam, AdamW, LAMB, RMSpropTF]
