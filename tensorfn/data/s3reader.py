try:
    import boto3
    from tqdm import tqdm

except ImportError:
    boto3 = None

from tensorfn.data.lmdb_reader import get_reader


class S3Reader:
    def __init__(
        self,
        bucket,
        path=None,
        reader="torch",
        access_key=None,
        secret_key=None,
        endpoint=None,
    ):
        if boto3 is None:
            raise ImportError("boto3 should be installed for S3 storage")

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
        )

        self.bucket = bucket
        self.path = path
        self.reader = self.get_reader(reader)
        self.length = None

    def open(self):
        try:
            self.length = int(self.get("length", "str"))

        except KeyError:
            self.length = 0

    def get_reader(self, reader):
        return get_reader(reader)

    def get(self, key, reader=None):
        if self.path is None:
            path_key = key

        else:
            path_key = f"{self.path}/{key}"

        return self.get_path(path_key, reader)

    def get_path(self, path_key, reader=None):
        if reader is not None:
            read_fn = self.get_reader(reader)

        else:
            read_fn = self.reader

        try:
            value = self.s3.get_object(Bucket=self.bucket, Key=path_key)["Body"].read()

        except self.s3.exceptions.NoSuchKey as e:
            raise KeyError(f"S3 bucket {self.bucket} does not have key {path_key}")

        return read_fn(value)

    def __len__(self):
        if self.length is None:
            self.open()

        return self.length

    def __iter__(self):
        i = 0

        while i < self.length:
            yield self.__getitem__(i)
            i += 1

    def __getitem__(self, index):
        return self.get(str(index))
