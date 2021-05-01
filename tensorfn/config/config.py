import sys
import inspect
import functools
import typing
from typing import Optional, Union

from pydantic import (
    BaseModel,
    create_model,
    validator,
    StrictStr,
    StrictInt,
    StrictBool,
)


class Config(BaseModel):
    class Config:
        extra = "forbid"


class MainConfig(BaseModel):
    class Config:
        extra = "forbid"

    n_gpu: Optional[StrictInt]
    n_machine: Optional[StrictInt]
    machine_rank: Optional[StrictInt]
    dist_url: Optional[StrictStr]
    distributed: Optional[StrictBool]
    ckpt: Optional[StrictStr]


class TypedConfig(BaseModel):
    class Config:
        extra = "forbid"

    type: StrictStr

    @validator("type")
    def check_type(cls, v):
        if v != cls.__type__:
            raise ValueError("Options does not match for " + cls.__type__)

        return v


CONFIG_REGISTRY = {}


def config_model(name=None, namespace=None, exclude=(), use_type=False):
    def _decorate(fn):
        if name is None:
            fn_name = fn.__name__

        else:
            fn_name = name

        if namespace not in CONFIG_REGISTRY:
            CONFIG_REGISTRY[namespace] = {}

        if fn_name in CONFIG_REGISTRY[namespace]:
            prev_fn = CONFIG_REGISTRY[namespace][fn_name][1]
            raise KeyError(f"Conflict occured in config registry: {prev_fn} vs {fn}")

        CONFIG_REGISTRY[namespace][fn_name] = (
            use_type,
            fn,
            inspect.signature(fn),
            exclude,
        )

        return fn

    return _decorate


def _check_type(type_name):
    @validator("type", allow_reuse=True)
    def check_type(cls, v):
        if v != type_name:
            raise ValueError(f"Type does not match for {type_name}")

        return v

    return check_type


class StrictConfig:
    extra = "forbid"


def make_model_from_signature(name, init_fn, signature, exclude, type_name=None):
    params = {}

    if type_name is not None:
        params["type"] = (StrictStr, ...)

    for k, v in signature.parameters.items():
        if k in exclude:
            continue

        annotation = v.annotation
        if annotation is inspect._empty:
            annotation = typing.Any

        if v.default is inspect._empty:
            params[k] = (annotation, ...)

        else:
            params[k] = (annotation, v.default)

    def _params(self):
        values = self.dict()

        if type_name is not None:
            values.pop("type")

        return values

    @functools.wraps(init_fn)
    def _init_fn(self, *args, **kwargs):
        params = self.params()
        params.update(kwargs)

        return init_fn(*args, **params)

    validators = {"params": _params, "make": _init_fn}

    if type_name is not None:
        validators["check_type"] = _check_type(type_name)

    model = create_model(
        name,
        __config__=StrictConfig,
        __validators__=validators,
        __module__=__name__,
        **params,
    )

    setattr(sys.modules[__name__], name, model)

    return model


CONFIG_MODEL_REGISTRY = {}


def get_models(namespace):
    names = CONFIG_REGISTRY[namespace].keys()
    for i, name in enumerate(names):
        model = get_model(name, namespace)

        if i == 0:
            models = Union[model]

        else:
            models = Union[models, model]

    return models


def get_model(name, namespace=None):
    if namespace not in CONFIG_MODEL_REGISTRY:
        CONFIG_MODEL_REGISTRY[namespace] = {}

    if name in CONFIG_MODEL_REGISTRY[namespace]:
        return CONFIG_MODEL_REGISTRY[namespace][name]

    use_type, init_fn, signature, exclude = CONFIG_REGISTRY[namespace][name]
    model = make_model_from_signature(
        name, init_fn, signature, exclude, name if use_type else None
    )
    CONFIG_MODEL_REGISTRY[namespace][name] = model

    return model


def override(overrides, **defaults):
    result = {}

    for k, v in defaults.items():
        result[k] = overrides.get(k, v)

    return result
