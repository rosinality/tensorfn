from torch.utils import cpp_extension as cpp_ext


class LazyExtension:
    def __init__(self, name, sources):
        self.name = name
        self.sources = sources
        self.loaded = False
        self.extension = None

    def get(self):
        if self.extension is None:
            self.extension = cpp_ext.load(self.name, sources=self.sources)

        return self.extension
