import re
import sys
import collections
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
    ValidationError,
    ArbitraryTypeError,
    dataclasses,
)
from torch.distributed.launcher.api import LaunchConfig

# LaunchConfig = dataclasses.dataclass(LaunchConfig)


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
    launch_config: typing.Any


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
    arbitrary_types_allowed = True


def make_model_from_signature(
    name, init_fn, signature, exclude, type_name=None, strict=True
):
    params = {}

    if type_name is not None:
        params["type"] = (StrictStr, ...)

    for k, v in signature.parameters.items():
        if k in exclude:
            continue

        if v.kind == v.VAR_POSITIONAL or v.kind == v.VAR_KEYWORD:
            strict = False

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
        pos_replace = list(signature.parameters.keys())[: len(args)]
        for pos in pos_replace:
            params.pop(pos)

        return init_fn(*args, **params)

    validators = {"params": _params, "make": _init_fn}

    if type_name is not None:
        validators["check_type"] = _check_type(type_name)

    if strict:
        config = StrictConfig

    else:
        config = None

    model = create_model(
        name,
        __config__=config,
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


def resolve_module(path):
    from importlib import import_module

    sub_path = path.split(".")
    module = None

    for i in reversed(range(len(sub_path))):
        try:
            mod = ".".join(sub_path[:i])
            module = import_module(mod)

        except (ModuleNotFoundError, ImportError):
            continue

        if module is not None:
            break

    obj = module

    for sub in sub_path[i:]:
        mod = f"{mod}.{sub}"

        if not hasattr(obj, sub):
            try:
                import_module(mod)

            except (ModuleNotFoundError, ImportError) as e:
                raise ImportError(
                    f"Encountered error: '{e}' when loading module '{path}'"
                ) from e

        obj = getattr(obj, sub)

    return obj


def instance_traverse(node, *args, recursive=True, instantiate=False):
    if isinstance(node, collections.abc.Sequence) and not isinstance(node, str):
        seq = [
            instance_traverse(i, recursive=recursive, instantiate=instantiate)
            for i in node
        ]

        return seq

    if isinstance(node, collections.abc.Mapping):
        target_key = "__target"
        init_key = "__init"
        fn_key = "__fn"
        validate_key = "__validate"
        partial_key = "__partial"
        args_key = "__args"

        exclude_keys = {
            target_key,
            init_key,
            fn_key,
            validate_key,
            partial_key,
            args_key,
        }

        if target_key in node or init_key in node or fn_key in node:
            return_fn = False
            partial = node.get(partial_key, False)
            do_validate = node.get(validate_key, True)

            if init_key in node:
                target = node.get(init_key)

            elif fn_key in node:
                target = node.get(fn_key)

                if len([k for k in node if k not in exclude_keys]) > 0:
                    partial = True

                else:
                    return_fn = True
                    do_validate = False

            else:
                target = node.get(target_key)

            obj = resolve_module(target)
            signature = inspect.signature(obj)

            if instantiate:
                if args_key in node:
                    args_node = node[args_key]

                    if len(args_node) > len(args):
                        args_init = []

                        for a in args_node[len(args) :]:
                            args_init.append(
                                instance_traverse(
                                    a, recursive=recursive, instantiate=instantiate
                                )
                            )

                        args = list(args) + args_init

                pos_replace = list(signature.parameters.keys())[: len(args)]

                kwargs = {}
                for k, v in node.items():
                    if k in exclude_keys:
                        continue

                    if k in pos_replace:
                        continue

                    kwargs[k] = instance_traverse(
                        v, recursive=recursive, instantiate=instantiate
                    )

                if return_fn:
                    return obj

                elif partial:
                    return functools.partial(obj, *args, **kwargs)

                else:
                    return obj(*args, **kwargs)

            else:
                rest = {}

                args_replaced = []
                if args_key in node:
                    for arg, k in zip(node[args_key], signature.parameters.keys()):
                        rest[k] = arg
                        args_replaced.append(k)

                for k, v in node.items():
                    if k in exclude_keys:
                        continue

                    rest[k] = instance_traverse(
                        v, recursive=recursive, instantiate=instantiate
                    )

                    if k in args_replaced:
                        raise TypeError(
                            f"{target} got multiple values for argument '{k}'"
                        )

                if do_validate:
                    name = "instance." + target

                    if partial:
                        rest_key = list(rest.keys())
                        exclude = []

                        for k in signature.parameters.keys():
                            if k not in rest_key:
                                exclude.append(k)

                        model = make_model_from_signature(
                            name, obj, signature, exclude, strict=False
                        )

                    else:
                        model = make_model_from_signature(name, obj, signature, ())

                    try:
                        model.validate(rest)

                    except ValidationError as e:
                        arbitrary_flag = True

                        for error in e.errors():
                            if error["type"] != "type_error.arbitrary_type":
                                arbitrary_flag = False

                                break

                        if not arbitrary_flag:
                            raise ValueError(
                                f"Validation for {target} with {v} is failed:\n{e}"
                            ) from e

                """return_dict = {
                    validate_key: do_validate,
                    partial_key: partial,
                    **rest,
                }

                if target_key in node:
                    return_dict[target_key] = target

                elif init_key in node:
                    return_dict[init_key] = target

                elif fn_key in node:
                    return_dict[fn_key] = target"""

                for arg in args_replaced:
                    del rest[arg]

                return_dict = {**node, **rest}

                return return_dict

        else:
            mapping = {}

            for k, v in node.items():
                mapping[k] = instance_traverse(
                    v, recursive=recursive, instantiate=instantiate
                )

            return mapping

    else:
        return node


class Instance(dict):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        v_new = instance_traverse(v)
        instance = cls(v_new)

        return instance

    def make(self, *args):
        return instance_traverse(self, *args, instantiate=True)

    def instantiate(self, *args):
        return self.make(*args)


def instantiate(instance, *args):
    try:
        return instance.make(*args)

    except AttributeError:
        return instance_traverse(instance, *args, instantiate=True)
