import torch
import torch.nn.functional as F


def compute_kerple_bias(r1_param, r2_param, num_heads, distances):
    """
    Compute KERPLE positional bias.

        B[i, j] = -r1 * log(1 + r2 * |dist(i, j)|)
    """
    r1 = torch.clamp(F.softplus(r1_param), max=10.0)
    r2 = torch.clamp(F.softplus(r2_param), max=10.0)

    bias = -r1.view(num_heads, 1, 1) * torch.log1p(
        torch.clamp(r2.view(num_heads, 1, 1) * distances.unsqueeze(0), max=1e4)
    )
    return bias
