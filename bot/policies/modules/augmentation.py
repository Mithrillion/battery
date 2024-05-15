from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SolarRandomNoiseAugmenter(nn.Module):
    def __init__(
        self, noise_std: float = 0.01, solar_indices: Optional[Tuple[int]] = None
    ):
        super().__init__()
        self.noise_std = noise_std
        self.applied_indices = solar_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.applied_indices is not None:
            x_solar_subset = x[:, self.applied_indices]
            solar_nonzero_indices = (
                torch.abs(x_solar_subset - (-1.0)) > 1e-4
            )
            x_solar_subset[solar_nonzero_indices] += (
                torch.randn_like(
                    x_solar_subset[solar_nonzero_indices], device=x.device
                )
                * self.noise_std
            )
            x_solar_subset = torch.clamp(x_solar_subset, -1.0, 1.0)
            x[:, self.applied_indices] = x_solar_subset
            return x
        else:
            return x
