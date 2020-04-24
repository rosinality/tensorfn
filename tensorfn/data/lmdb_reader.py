import io
import pickle

import lmdb
import torch


def pickle_reader(byte_str):
    return pickle.loads(byte_str)


def torch_reader(byte_str):
    return torch.load(io.BytesIO(byte_str), map_location=lambda storage, loc: storage)


def raw_reader(byte_str):
    return byte_str


def str_reader(byte_str):
    return byte_str.decode('utf-8')


class LMDBReader:
    def __init__(self, path, reader='torch', map_size=1024 ** 4, max_readers=126):
        self.env = lmdb.open(
            path,
            map_size,
            readonly=True,
            create=False,
            readahead=False,
            lock=False,
            max_readers=max_readers,
        )

        if not self.env:
            raise IOError(f'Cannot open lmdb dataset {path}')

        self.reader = self.get_reader(reader)

        try:
            self.length = int(self.get(b'length', 'str'))

        except KeyError:
            self.length = 0

    def get_reader(self, reader):
        if isinstance(reader, str):
            read_fn = {
                'pickle': pickle_reader,
                'torch': torch_reader,
                'raw': raw_reader,
                'str': str_reader,
            }[reader]

        elif callable(reader):
            read_fn = reader

        else:
            raise ValueError(
                'reader should be "pickle", "torch", "raw", "str" or callable'
            )

        return read_fn

    def get(self, key, reader=None):
        if reader is not None:
            read_fn = self.get_reader(reader)

        else:
            read_fn = self.reader

        with self.env.begin(write=False) as txn:
            value = txn.get(key)

        if value is None:
            raise KeyError(f'lmdb dataset does not have key {key}')

        return read_fn(value)

    def __len__(self):
        return self.length

    def __iter__(self):
        i = 0

        while i < self.length:
            yield self.__getitem__(i)
            i += 1

    def __getitem__(self, index):
        return self.get(str(index).encode('utf-8'))

    def __del__(self):
        self.env.close()
