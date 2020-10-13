import sys
import inspect
import functools
import typing

from pydantic import BaseModel, create_model


class Config(BaseModel):
    class Config:
        extra = "forbid"


CONFIG_REGISTRY = {}


def config_model(name=None, exclude=(), use_type=False):
    def _decorate(fn):
        if name is None:
            fn_name = fn.__name__

        else:
            fn_name = name

        if fn_name in CONFIG_REGISTRY:
            prev_fn = CONFIG_REGISTRY[fn_name][1]
            raise KeyError(f"Conflict occured in config registry: {prev_fn} vs {fn}")

        CONFIG_REGISTRY[fn_name] = (use_type, fn, inspect.signature(fn), exclude)

        return fn

    return _decorate


def _check_type(type_name):
    @validator("type")
    def check_type(cls, v):
        if v != type_name:
            raise ValueError(f"Type does not match for {type_name}")

        return v

    return check_type


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

    model = create_model(name, __validators__=validators, __module__=__name__, **params)

    setattr(sys.modules[__name__], name, model)

    return model


CONFIG_MODEL_REGISTRY = {}


def get_model(name):
    if name in CONFIG_MODEL_REGISTRY:
        return CONFIG_MODEL_REGISTRY[name]

    use_type, init_fn, signature, exclude = CONFIG_REGISTRY[name]
    model = make_model_from_signature(
        name, init_fn, signature, exclude, name if use_type else None
    )
    CONFIG_MODEL_REGISTRY[name] = model

    return model
