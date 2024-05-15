import torch


def time_aug(X, t_range=(0, 1)):
    # assume (batch, time, channel)
    t_len = X.shape[1]
    time_axis = torch.linspace(*t_range, t_len, device=X.device)
    time_component = torch.repeat_interleave(time_axis[None, :, None], len(X), dim=0)
    return torch.cat([time_component, X], dim=-1)


def znorm(x, dim=-1, eps=1e-8):
    if torch.is_tensor(x):
        return (x - x.mean(dim, keepdim=True)) / (eps + x.std(dim, keepdim=True))
    else:
        return (x - x.mean(dim, keepdims=True)) / (eps + x.std(dim, keepdims=True))


def minmaxnorm(x, dim=-1, eps=1e-8):
    if torch.is_tensor(x):
        return (x - x.amin(dim, keepdim=True)) / (
            eps + x.amax(dim, keepdim=True) - x.amin(dim, keepdim=True)
        )
    else:
        return (x - x.amin(dim, keepdims=True)) / (
            eps + x.amax(dim, keepdims=True) - x.amin(dim, keepdims=True)
        )


def unorm(x, dim=-1, eps=1e-8):
    return x / (eps + torch.norm(x, p=2, dim=dim, keepdim=True))


def max_abs_norm(x, dim=-1, eps=1e-8):
    return x / (eps + torch.max(torch.abs(x), dim=dim, keepdim=True).values)