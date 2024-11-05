# coding=utf-8

import numpy as np
import torch
import torch.nn as nn


def get_len_mask(
    b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device
) -> torch.Tensor:
    attn_mask = torch.ones((b, max_len, max_len), device=device)
     

    pass
