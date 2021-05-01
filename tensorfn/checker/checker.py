import os

try:
    import private

except ImportError:
    private = None

from tensorfn import distributed as dist
from tensorfn.checker.backend import Local


class Checker:
    def __init__(self, storages=None, reporters=None):
        self.storages = storages
        self.reporters = reporters

    def save(self, data, name):
        if dist.is_primary():
            for storage in self.storages:
                storage.save(data, name)

    def checkpoint(self, obj, name):
        if dist.is_primary():
            for storage in self.storages:
                storage.checkpoint(obj, name)

    def log(self, step, **kwargs):
        if dist.is_primary():
            for reporter in self.reporters:
                reporter.log(step, **kwargs)
