from datetime import datetime
import io
import os
import functools
import sys
import re

import torch
from termcolor import colored
from tabulate import tabulate

try:
    import boto3
    from tqdm import tqdm

except ImportError:
    boto3 = None


from tensorfn import distributed as dist, get_logger, nsml


def torch_serialize(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)

    return buf.read()


class Storage:
    def __init__(self, keep=-1):
        self.keep = keep
        self._saved_checkpoints = []
        self._saved_checkpoints_value = []

    def checkpoint(self, obj, path, value=None):
        if value is not None:
            exps = re.findall("<.+?>", path)
            path = path
            for exp in exps:
                if "value" in exp:
                    path = path.replace(exp, f"{{{exp[1:-1]}}}", 1)
            path = path.format(value=value)

        keep = self.keep - 1

        if self.keep > 0:
            if len(self._saved_checkpoints) > keep:
                for head in self._saved_checkpoints[:-keep]:
                    self._remove(head)

                self._saved_checkpoints = self._saved_checkpoints[-keep:]

            if len(self._saved_checkpoints_value) > keep:
                sorted_k = sorted(
                    enumerate(self._saved_checkpoints_value), key=lambda x: x[1][1]
                )
                bottom_k = sorted_k[:-keep]

                for _, (bottom, _) in bottom_k:
                    self._remove(bottom)

                updated = []
                keep_ids = [i[0] for i in sorted_k[-keep:]]
                for i, record in enumerate(self._saved_checkpoints_value):
                    if i in keep_ids:
                        updated.append(record)

                self._saved_checkpoints_value = updated

        binary = torch_serialize(obj)
        self.save(binary, path)

        if value is None:
            self._saved_checkpoints.append(path)

        else:
            self._saved_checkpoints_value.append((path, value))

    def get_directory(self, path):
        # dup = len(self.list(path)) + 1
        # path = f"{path}/{str(dup).zfill(5)}"
        key = datetime.now().astimezone().isoformat().replace(":", ".")
        path = f"{path}/{key}"

        return path


class Local(Storage):
    def __init__(self, path, keep=-1):
        super().__init__(keep)

        root, child = os.path.split(path)
        if root == "":
            root = "."

        path = os.path.join(root, child)

        self.path = self.get_directory(path)

    def list(self, path):
        try:
            dirs = os.listdir(path)

        except FileNotFoundError:
            dirs = []

        return dirs

    def save(self, data, name):
        if isinstance(data, bytes):
            flag = "wb"

        else:
            flag = "w"

        target_path = os.path.join(self.path, name)

        os.makedirs(os.path.split(target_path)[0], exist_ok=True)

        with open(target_path, flag) as f:
            f.write(data)

    def load(self, name):
        pass


def progress_callback(pbar):
    def wrap(bytes_amount):
        pbar.update(bytes_amount)

    return wrap


class S3(Storage):
    def __init__(
        self,
        bucket,
        path,
        access_key,
        secret_key,
        keep=-1,
        endpoint=None,
        show_progress=True,
    ):
        super().__init__(keep)

        if boto3 is None:
            raise ImportError("boto3 should be installed for S3 storage")

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
        )
        self.bucket = bucket
        self.path = self.get_directory(path)
        self.show_progress = show_progress

    def list(self, path):
        if path[-1] != "/":
            path += "/"

        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=path, Delimiter="/")

        try:
            prefixes = []

            for prefix in resp["CommonPrefixes"]:
                prefixes.append(prefix["Prefix"])

        except KeyError:
            prefixes = []

        return prefixes

    def save(self, data, name):
        buf = io.BytesIO(data)
        size = len(data)

        self._save(buf, name, size)

    def _remove(self, name):
        target_path = f"{self.path}/{name}"

        self.s3.delete_object(Bucket=self.bucket, Key=target_path)

    def _save(self, buf, name, size):
        target_path = f"{self.path}/{name}"

        if self.show_progress:
            with tqdm(total=size, unit="B", unit_scale=True, desc=target_path) as pbar:
                self.s3.upload_fileobj(
                    buf, self.bucket, target_path, Callback=progress_callback(pbar)
                )

        else:
            self.s3.upload_fileobj(buf, self.bucket, target_path)


def default_formatter(step, **kwargs):
    panels = [f"step: {step}"]

    for k, v in kwargs.items():
        if isinstance(v, float):
            panels.append(f"{k}: {v:.3f}")

        else:
            panels.append(f"{k}: {v}")

    return "; ".join(panels)


class Logger:
    def __init__(self, formatter=None):
        if formatter is None:
            formatter = default_formatter

        self.logger = get_logger()
        self.formatter = formatter

    def log(self, step, **kwargs):
        self.logger.info(self.formatter(step, **kwargs))


class NSML:
    def log(self, step, **kwargs):
        nsml.report(summary=True, step=step, **kwargs)
