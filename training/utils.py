def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def exists(val):
    return val is not None


class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.total += value * count
        self.count += count

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count
