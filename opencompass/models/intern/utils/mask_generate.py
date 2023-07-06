import math

import torch


def cosine_mask_weight_v1(cur_iter, max_iter):
    # cur_iter: [1, max_iter]
    lr = (1 - math.cos(min(cur_iter, max_iter) / max_iter * math.pi * 0.5))
    return lr


def cosine_mask_weight_v2(cur_iter, max_iter):
    # cur_iter: [1, max_iter]
    lr = (1 - math.cos(min(cur_iter, max_iter) / max_iter * math.pi)) * 0.5
    return lr


def cosine_mask_weight_v3(cur_iter, max_iter):
    # cur_iter: [1, max_iter]
    lr = ((1 - math.cos(min(cur_iter, max_iter) / max_iter * math.pi)) *
          0.5)**2
    return lr


def generate_scaleup_mask(cur_iter,
                          warumup_iter,
                          max_iter,
                          ori_ch,
                          scaleup_ch,
                          max_ch,
                          mode='v1',
                          device='cuda'):
    """cur_iter (int): [1, max_iter] ori_ch (int): channel of original model
    scaleup_ch (int): channel to scale up max_ch (int): max channel mode (str):

    cosine or sine.
    """
    assert warumup_iter <= max_iter
    if mode == 'v1':
        cur_weight = cosine_mask_weight_v1(max(cur_iter - warumup_iter, 0),
                                           max_iter - warumup_iter)
    elif mode == 'v2':
        cur_weight = cosine_mask_weight_v2(max(cur_iter - warumup_iter, 0),
                                           max_iter - warumup_iter)
    elif mode == 'v3':
        cur_weight = cosine_mask_weight_v3(max(cur_iter - warumup_iter, 0),
                                           max_iter - warumup_iter)
    else:
        raise NameError
    assert cur_weight >= 0 and cur_weight <= 1

    assert ori_ch <= max_ch
    assert scaleup_ch <= max_ch
    assert ori_ch <= scaleup_ch
    mask = torch.zeros((1, 1, max_ch), dtype=torch.float, device=device)
    mask[:, :, :ori_ch] = 1.0
    mask[:, :, ori_ch:scaleup_ch] = cur_weight

    # mask = mask.to(device)
    return mask, cur_weight


def get_hidden_dim(dim):
    multiple_of = 256
    hidden_dim = int(8 * dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim
