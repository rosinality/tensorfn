import os

try:
    import nsml

except ImportError:
    nsml = None

from tensorfn import distributed as dist


class NSMLBackend:
    def bind(self, save_fn, load_fn):
        def save_wrapper(name):
            save_fn(os.path.join(name, "checkpoint"))

        nsml.bind(save=save_wrapper, load=load_fn)

    def save(self, name):
        nsml.save(name)

    def log(self, step, **kwargs):
        nsml.report(summary=True, step=step, **kwargs)


class NativeBackend:
    def __init__(self, path):
        self.path = path

    def bind(self, save_fn, load_fn):
        self.save_fn = save_fn
        self.load_fn = load_fn

    def save(self, name):
        os.makedirs(self.path, exist_ok=True)
        self.save_fn(os.path.join(self.path, name))

    def load(self, name):
        pass

    def log(self, step, **kwargs):
        meters = [str(step)]

        for k, v in kwargs.items():
            meters.append(f"{k}: {v}")

        print("; ".join(meters))


class Checker:
    def __init__(self, save_fn, load_fn=None, backend="native", **kwargs):
        self.save_fn = save_fn
        self.load_fn = load_fn

        if nsml is not None and nsml.IS_ON_NSML:
            backend = "nsml"

        self.backend = {"native": NativeBackend, "nsml": NSMLBackend}.get(backend)(
            **kwargs
        )
        self.backend.bind(save_fn, load_fn)

    def save(self, name):
        if dist.is_primary():
            self.backend.save(name)

    def load(self, name):
        if dist.is_primary():
            return self.backend.load(name)

    def log(self, step, **kwargs):
        if dist.is_primary():
            self.backend.log(step, **kwargs)
