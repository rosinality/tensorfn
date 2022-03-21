import os
import sys

try:
    import private

except ImportError:
    private = None

try:
    from rich.pretty import pretty_repr

    pformat = pretty_repr

except ImportError:
    from pprint import pformat

from tensorfn import distributed as dist
from tensorfn.checker.backend import Local


class Checker:
    def __init__(self, storages=None, reporters=None):
        self.storages = storages
        self.reporters = reporters

    def catalog(self, conf):
        if not dist.is_primary():
            return

        if not isinstance(conf, dict):
            conf = conf.dict()

        conf = pformat(conf)

        argvs = " ".join([os.path.basename(sys.executable)] + sys.argv)

        template = f"""{argvs}

{conf}"""
        template = template.encode("utf-8")

        for storage in self.storages:
            storage.save(template, "catalog.txt")

    def save(self, data, name):
        if dist.is_primary():
            for storage in self.storages:
                storage.save(data, name)

    def checkpoint(self, obj, name):
        if dist.is_primary():
            for storage in self.storages:
                storage.checkpoint(obj, name)

    def log(self, step=None, **kwargs):
        if dist.is_primary():
            for reporter in self.reporters:
                reporter.log(step, **kwargs)
