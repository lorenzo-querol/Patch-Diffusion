def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def exists(val):
    return val is not None
