from typing import Union

from pydantic import StrictStr
from tensorfn.config import TypedConfig
from tensorfn import checker


class NSML(TypedConfig):
    __type__ = "nsml"

    def make(self, save_fn, load_fn=None):
        return checker.Checker(save_fn, load_fn, "nsml")


class Native(TypedConfig):
    __type__ = "native"
    path: StrictStr = "checkpoint"

    def make(self, save_fn, load_fn=None):
        return checker.Checker(save_fn, load_fn, "native", path=self.path)


Checker = Union[NSML, Native]
