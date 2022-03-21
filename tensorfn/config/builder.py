import ast
import builtins
import os
import pydoc
import uuid
import importlib
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from typing import Union, Tuple

from tensorfn.config.config import resolve_module

CFG_PACKAGE_NAME = "tensorfn._conf_loader"


def str_to_import(name):
    obj = pydoc.locate(name)

    if obj is None:
        obj = resolve_module(name)

    return obj


def validate_syntax(filename):
    with open(filename) as f:
        code = f.read()

    try:
        ast.parse(code)

    except SyntaxError as e:
        raise SyntaxError(f"{filename} has syntax error") from e


def random_package_name(filename):
    # generate a random package name when loading config files
    return CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)


def import_to_str(obj):
    module, qualname = obj.__module__, obj.__qualname__

    module_parts = module.split(".")

    for i in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:i])
        candid = f"{prefix}.{qualname}"

        try:
            if str_to_import(candid) is obj:
                return candid

        except ImportError:
            pass

    return f"{module}.{qualname}"


def build(__key, __name, *args, **kwargs):
    node = {__key: __name}

    if len(args) > 0:
        node["__args"] = args

    node = {**node, **kwargs}

    return node


def build_init(__name, *args, **kwargs):
    return build("__init", __name, *args, **kwargs)


def build_fn(__name, *args, **kwargs):
    return build("__fn", __name, *args, **kwargs)


class Init:
    def __init__(self, name, fn=False):
        self.name = name
        self.fn = fn

    def __call__(self, *args, **kwargs):
        if self.fn:
            return build_fn(self.name, *args, **kwargs)

        return build_init(self.name, *args, **kwargs)


class LazyCall:
    def __getitem__(self, obj):
        fn = False

        if isinstance(obj, tuple):
            obj, fn = obj

        if not isinstance(obj, str):
            obj = import_to_str(obj)

        return Init(obj, fn)


class LazyFn:
    def __getitem__(self, obj):
        if not isinstance(obj, str):
            obj = import_to_str(obj)

        return Init(obj, True)


@contextmanager
def patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. support other storage system through PathManager
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)

        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)

        cur_name = relative_import_path.lstrip(".")

        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)

        # NOTE: directory import is not handled. Because then it's unclear
        # if such import should produce python module or DictConfig. This can
        # be discussed further if needed.
        if not cur_file.endswith(".py"):
            cur_file += ".py"

        if not os.path.isfile(cur_file):
            raise ImportError(
                f"cannot import name {relative_import_path} from "
                f"{original_file}: {cur_file} has to exist"
            )

        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            validate_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(
                random_package_name(cur_file), None, origin=cur_file
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file

            with open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)

            # for name in fromlist:  # turn imported dict into DictConfig automatically
            #     val = _cast_to_config(module.__dict__[name])
            #     module.__dict__[name] = val

            return module

        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


class PyConfig:
    @staticmethod
    def load(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Load a config file.
        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return. If not given, return all keys
                (whose values are config objects) in a dict.
        """
        has_keys = keys is not None
        filename = filename.replace("/./", "/")  # redundant

        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")

        if filename.endswith(".py"):
            validate_syntax(filename)

            with patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": random_package_name(filename),
                }
                with open(filename) as f:
                    content = f.read()
                # Compile first with filename to:
                # 1. make filename appears in stacktrace
                # 2. make load_rel able to find its parent's (possibly remote) location
                exec(compile(content, filename, "exec"), module_namespace)

            ret = module_namespace

        ret = ret["conf"].to_dict()

        if has_keys:
            return tuple(ret[a] for a in keys)

        return ret


def unfold_field(x):
    if isinstance(x, Sequence) and not isinstance(x, str):
        return [unfold_field(i) for i in x]

    if isinstance(x, Mapping):
        res = {}

        for k, v in x.items():
            res[k] = unfold_field(v)

        return res

    return x


class Field(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)

        except AttributeError:
            try:
                return self[key]

            except KeyError:
                raise AttributeError(key)

    def __setattr__(self, key, value):
        try:
            object.__getattribute__(self, key)

        except AttributeError:
            try:
                self[key] = value

            except:
                raise AttributeError(key)

        else:
            object.__setattr__(self, key, value)

    def __delattr__(self, key):
        try:
            object.__getattribute__(self, key)

        except AttributeError:
            try:
                del self[key]

            except KeyError:
                raise AttributeError(key)

        else:
            object.__delattr__(self, key)

    def __repr__(self):
        return f"{self.__class__.__name__}({dict.__repr__(self)})"

    def to_dict(self):
        return unfold_field(self)


L = LazyCall()
F = LazyFn()
field = Field
