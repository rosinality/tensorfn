import dataclasses


def _set_attr(cls, name, value):
    if name in cls.__dict__:
        return True

    setattr(cls, name, value)

    return False


def _to_fn(self, *args, **kwargs):
    fields = dataclasses.fields(self)
    new_batch = {}

    for field in fields:
        value = getattr(self, field.name)

        if hasattr(value, "to"):
            new_value = value.to(*args, **kwargs)

        else:
            new_value = value

        new_batch[field.name] = new_value

    return self.__class__(**new_batch)


def batch(cls):
    _set_attr(cls, "to", _to_fn)

    return dataclasses.dataclass(cls)


if __name__ == "__main__":
    import torch
    from typing import List

    @batch
    class Test:
        input: torch.Tensor
        label: List[int]

    abc = Test(input=torch.tensor([1, 2]), label=[0, 1])
    abc_cuda = abc.to("cuda")
    print(abc.to("cuda"))
    print(id(abc.input), id(abc.label))
    print(id(abc_cuda.input), id(abc_cuda.label))
