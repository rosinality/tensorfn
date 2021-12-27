import copy


def repeat(object, times):
    copied = []

    for _ in range(times):
        if isinstance(object, (list, tuple)):
            for obj in object:
                copied.append(copy.deepcopy(obj))

        else:
            copied.append(copy.deepcopy(object))

    return copied
