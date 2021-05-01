# shamelessly took from detectron2
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/logger.py

import functools
import logging
import sys
import pprint

from termcolor import colored
from tabulate import tabulate

try:
    from rich.logging import RichHandler

except ImportError:
    RichHandler = None

from tensorfn import distributed as dist


class ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def wrap_log_record_factory(factory):
    def wrapper(
        name, level, fn, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs
    ):
        if not isinstance(msg, str):
            msg = pprint.pformat(msg)

        return factory(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)

    return wrapper


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def get_logger(distributed_rank=None, *, mode="rich", name="main", abbrev_name=None):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """
    if distributed_rank is None:
        distributed_rank = dist.get_rank()

    logging.setLogRecordFactory(wrap_log_record_factory(logging.getLogRecordFactory()))

    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    if mode == "rich" and RichHandler is None:
        mode = "color"

    if distributed_rank == 0:
        if mode == "color":
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        elif mode == "rich":
            logger.addHandler(
                RichHandler(level=logging.DEBUG, log_time_format="%m/%d %H:%M:%S")
            )

        elif mode == "plain":
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%m/%d %H:%M:%S",
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    return logger


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.
    Args:
        small_dict (dict): a result dictionary of only a few items.
    Returns:
        str: the table as a string.
    """

    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )

    return table
