import torch

def add_coord_channels(x):
    B, C, H, W = x.shape
    device = x.device

    row = torch.linspace(0, 1, H, device=device).view(1,1,H,1).repeat(B,1,1,W)
    col = torch.linspace(0, 1, W, device=device).view(1,1,1,W).repeat(B,1,H,1)

    return torch.cat([x, row, col], dim=1)
