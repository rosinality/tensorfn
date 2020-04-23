from typing import Tuple, Union, List

from pydantic import BaseModel, validator, StrictStr, StrictFloat, StrictInt, StrictBool

from tensorfn.config import Config


class Cycle(Config):
    type: StrictStr

    lr: StrictFloat
    n_iter: StrictInt
    initial_multiplier: StrictFloat = 4e-2
    final_multiplier: StrictFloat = 1e-5
    warmup: StrictInt = 500
    plateau: StrictInt = 0
    decay: List[StrictStr]

    @validator("type")
    def check_type(cls, v):
        if v != "cycle":
            raise ValueError("Optimizer options not match for cycle")

        return v


class Step(Config):
    type: StrictStr

    lr: StrictFloat
    milestones: List[StrictInt]
    gamma: StrictFloat = 0.1
    warmup: StrictInt = 0
    warmup_multiplier = 4e-2

    @validator("type")
    def check_type(cls, v):
        if v != "step":
            raise ValueError("Optimizer options not match for cycle")

        return v


class LRFind(Config):
    type: StrictStr

    lr_min: StrictFloat
    lr_max: StrictFloat
    n_iter: StrictInt
    linear: StrictBool = False

    @validator("type")
    def check_type(cls, v):
        if v != "lr_find":
            raise ValueError("Optimizer options not match for cycle")

        return v


Scheduler = Union[Cycle, Step, LRFind]
