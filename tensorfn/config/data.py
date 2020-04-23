from typing import Tuple, Union, List, Optional, Any, Callable

from pydantic import BaseModel, StrictInt, StrictBool
from torch.utils.data import DataLoader

from tensorfn.config import Config


class DataLoader(Config):
    dataset: Any
    batch_size: StrictInt = 1
    shuffle: StrictBool = False
    num_workers: StrictInt = 0
    pin_memory: StrictBool = False
    drop_last: StrictBool = False
    timeout: StrictInt = 0

    def make(
        self,
        dataset,
        sampler=None,
        batch_sampler=None,
        collate_fn=None,
        worker_init_fn=None,
        multiprocessing_context=None,
    ):
        return DataLoader(
            self.dataset,
            self.batch_size,
            self.shuffle,
            sampler,
            batch_sampler,
            self.num_workers,
            collate_fn,
            self.pin_memory,
            self.drop_last,
            self.timeout,
            worker_init_fn,
            multiprocessing_context,
        )


def make_dataloader(
    config,
    dataset,
    sampler=None,
    batch_sampler=None,
    collate_fn=None,
    worker_init_fn=None,
    multiprocessing_context=None,
):
    return config.make(
        dataset,
        sampler,
        batch_sampler,
        collate_fn,
        worker_init_fn,
        multiprocessing_context,
    )
