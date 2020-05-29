from collections import abc
from itertools import repeat


def ensure_tuple(x, n_item):
    if isinstance(x, abc.Iterable):
        try:
            if len(x) != n_item:
                raise ValueError(
                    f"length of {x} (length: {len(x)}) does not match with the condition. expected length: {n_item}"
                )

        except TypeError:
            pass

        return x

    return tuple(repeat(x, n_item))


if __name__ == "__main__":
    print(ensure_tuple(range(2), 2))
    print(list(ensure_tuple((i ** 2 for i in range(5)), 2)))
    print(ensure_tuple(3, 2))
    print(ensure_tuple(range(2), 3))
