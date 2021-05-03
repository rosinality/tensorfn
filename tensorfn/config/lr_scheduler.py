from typing import Tuple, Union, Sequence

from pydantic import BaseModel, validator, StrictStr, StrictInt, StrictBool

from tensorfn.config import Config, TypedConfig, override
from tensorfn.optim import lr_scheduler


class Constant(Config):
    type: StrictStr

    @validator("type")
    def check_type(cls, v):
        if v != "constant":
            raise ValueError("Optimizer options not match for constant scheduler")

        return v

    def make(self, optimizer):
        return lr_scheduler.ConstantScheduler(optimizer)


class Cycle(Config):
    type: StrictStr

    lr: float
    n_iter: StrictInt = 0
    initial_multiplier: float = 4e-2
    final_multiplier: float = 1e-5
    warmup: StrictInt = 0
    plateau: StrictInt = 0
    decay: Sequence[StrictStr] = ("linear", "cos")

    @validator("type")
    def check_type(cls, v):
        if v != "cycle":
            raise ValueError("Optimizer options not match for cycle")

        return v

    def make(self, optimizer, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            n_iter=self.n_iter,
            initial_multiplier=self.initial_multiplier,
            final_multiplier=self.final_multiplier,
            warmup=self.warmup,
            plateau=self.plateau,
            decay=self.decay,
        )

        return lr_scheduler.cycle_scheduler(optimizer, **argument)


class Step(Config):
    type: StrictStr

    lr: float
    milestones: Sequence[StrictInt]
    gamma: float = 0.1
    warmup: StrictInt = 0
    warmup_multiplier = 4e-2

    @validator("type")
    def check_type(cls, v):
        if v != "step":
            raise ValueError("Optimizer options not match for cycle")

        return v

    def make(self, optimizer, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            milestones=self.milestones,
            gamma=self.gamma,
            warmup=self.warmup,
            warmup_multiplier=self.warmup_multiplier,
        )

        return lr_scheduler.step_scheduler(optimizer, **argument)


class Exp(TypedConfig):
    __type__ = "exp"

    lr: float
    step: StrictInt
    max_iter: StrictInt = 0
    gamma: float = 0.97
    warmup: StrictInt = 0
    warmup_multiplier: float = 4e-2

    def make(self, optimizer, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            step=self.step,
            max_iter=self.max_iter,
            gamma=self.gamma,
            warmup=self.warmup,
            warmup_multiplier=self.warmup_multiplier,
        )

        return lr_scheduler.exp_scheduler(optimizer, **argument)


class ExpEpoch(TypedConfig):
    __type__ = "exp_epoch"

    lr: float
    epoch: float
    max_iter: StrictInt = 0
    gamma: float = 0.97
    warmup: StrictInt = 0
    warmup_multiplier: float = 4e-2

    def make(self, optimizer, epoch_step, **kwargs):
        argument = override(
            kwargs,
            lr=self.lr,
            max_iter=self.max_iter,
            gamma=self.gamma,
            warmup=self.warmup,
            warmup_multiplier=self.warmup_multiplier,
        )

        return lr_scheduler.exp_scheduler(
            optimizer, step=int(epoch_step * self.epoch), **argument
        )


class LRFind(Config):
    type: StrictStr

    lr_min: float
    lr_max: float
    n_iter: StrictInt
    linear: StrictBool = False

    @validator("type")
    def check_type(cls, v):
        if v != "lr_find":
            raise ValueError("Optimizer options not match for cycle")

        return v

    def make(self, optimizer, **kwargs):
        argument = override(
            kwargs,
            lr_min=self.lr_min,
            lr_max=self.lr_max,
            n_iter=self.n_iter,
            linear=self.linear,
        )

        return lr_scheduler.lr_finder(optimizer, **argument)


Scheduler = Union[Constant, Cycle, Step, ExpEpoch, Exp, LRFind]
