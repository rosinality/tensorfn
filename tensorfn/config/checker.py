from typing import Union, Optional, List

from pydantic import StrictStr, StrictBool, StrictInt
from tensorfn.config import TypedConfig, override, Config
from tensorfn import checker


class Local(TypedConfig):
    __type__ = "local"
    path: StrictStr
    keep: StrictInt = -1

    def make(self, **kwargs):
        argument = override(kwargs, path=self.path, keep=self.keep)

        return checker.Local(**argument)


class S3(TypedConfig):
    __type__ = "s3"
    bucket: StrictStr
    path: StrictStr
    access_key: StrictStr
    secret_key: StrictStr
    keep: StrictInt = -1
    endpoint: Optional[StrictStr]
    show_progress: StrictBool

    def make(self, **kwargs):
        argument = override(
            kwargs,
            bucket=self.bucket,
            path=self.path,
            access_key=self.access_key,
            secret_key=self.secret_key,
            keep=self.keep,
            endpoint=self.endpoint,
            show_progress=self.show_progress,
        )

        return checker.S3(**argument)


class Logger(TypedConfig):
    __type__ = "logger"

    def make(self, formatter=None):
        return checker.Logger(formatter)


class WandB(TypedConfig):
    __type__ = "wandb"
    project: StrictStr
    group: Optional[StrictStr] = None
    name: Optional[StrictStr] = None
    notes: Optional[StrictStr] = None
    resume: Optional[Union[StrictBool, StrictStr]] = None
    tags: Optional[List[StrictStr]] = None
    id: Optional[StrictStr] = None

    def make(self, **kwargs):
        argument = override(
            kwargs,
            project=self.project,
            group=self.group,
            name=self.name,
            notes=self.notes,
            resume=self.resume,
            tags=self.tags,
            id=self.id,
        )

        return checker.WandB(**argument)


class NSML(TypedConfig):
    __type__ = "nsml"

    def make(self):
        return checker.NSML()


Storage = Union[Local, S3]
Reporter = Union[Logger, NSML, WandB]


class Checker(Config):
    storage: Union[Storage, List[Storage]] = Local(type="local", path="experiment")
    reporter: Union[Reporter, List[Reporter]] = Logger(type="logger")

    def make(self, storage=None, reporter=None):
        if storage is None:
            if not isinstance(self.storage, list):
                storage_list = [self.storage]

            else:
                storage_list = self.storage

            storages = []

            for storage in storage_list:
                storages.append(storage.make())

        if reporter is None:
            if not isinstance(self.reporter, list):
                reporter_list = [self.reporter]

            else:
                reporter_list = self.reporter

            reporters = []

            for reporter in reporter_list:
                reporters.append(reporter.make())

        return checker.Checker(storages, reporters)
