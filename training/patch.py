import torch


def get_patches(images, patch_size, padding=None):
    """Extract random patches of square `patch_size`
    from the input images and return them along with their positions.

    Proposed in https://openreview.net/forum?id=iv2sTQtbst.

    :param images: Input images of shape (batch_size, channels, resolution, resolution)
    :param patch_size: Size of the patches to be extracted
    :param padding: Padding to be added to the images before extracting patches
    :return: Patches with positions of shape (batch_size, channels + 2, patch_size, patch_size)
    """
    device = images.device
    batch_size, resolution = images.size(0), images.size(2)

    if padding is not None:
        padded = torch.zeros((images.size(0), images.size(1), images.size(2) + padding * 2, images.size(3) + padding * 2), dtype=images.dtype, device=device)
        padded[:, :, padding:-padding, padding:-padding] = images
    else:
        padded = images

    h, w = padded.size(2), padded.size(3)
    th, tw = patch_size, patch_size
    if w == tw and h == th:
        i = torch.zeros((batch_size,), device=device).long()
        j = torch.zeros((batch_size,), device=device).long()
    else:
        i = torch.randint(0, h - th + 1, (batch_size,), device=device)
        j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

    rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
    columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
    padded = padded.permute(1, 0, 2, 3)
    padded = padded[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]], columns[:, None]]
    padded = padded.permute(1, 0, 2, 3)

    x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    x_pos = x_pos + j.view(-1, 1, 1, 1)
    y_pos = y_pos + i.view(-1, 1, 1, 1)
    x_pos = (x_pos / (resolution - 1) - 0.5) * 2.0
    y_pos = (y_pos / (resolution - 1) - 0.5) * 2.0
    images_pos = torch.cat((x_pos, y_pos), dim=1)

    return torch.cat([padded, images_pos], dim=1)
