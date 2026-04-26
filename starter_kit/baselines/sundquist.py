#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging

# External modules
import torch

# Internal modules
from .utils import estimate_relative_humidity, approximate_surface_pressure


main_logger = logging.getLogger(__name__)


class SundquistNetwork(torch.nn.Module):
    _EPSILON = 1e-6

    def __init__(self):
        super().__init__()
        self.register_parameter(
            "logit_critical_level",
            torch.nn.Parameter(torch.zeros(7, 1, 1,))
        )
        self.register_buffer(
            "pressure_levels",
            torch.tensor(
                [1000_00, 850_00, 700_00, 500_00, 250_00, 100_00, 50_00]
            ).reshape(-1, 1, 1)
        )

    def forward(
            self,
            input_level: torch.Tensor,
            input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        # Cloud cover at levels
        level_rh = estimate_relative_humidity(
            temperature=input_level[:, 0:1],
            specific_humidity=input_level[:, 1:2],
            pressure=self.pressure_levels
        )
        level_ratio = (
            (1 - level_rh)
            / torch.sigmoid(self.logit_critical_level)
        )
        level_cloud_cover = 1 - torch.sqrt(level_ratio + self._EPSILON)
        level_cloud_cover = torch.clamp(level_cloud_cover, 0, 1)

        # Mask where pressure is lower than surface pressure
        surface_pressure = approximate_surface_pressure(
            input_auxiliary[:, 1:2]
        )
        valid_mask = self.pressure_levels < surface_pressure.unsqueeze(2)

        # Randomly distributed cloud cover
        level_clear_sky_fraction = 1 - level_cloud_cover * valid_mask
        total_cloud_cover = 1 - torch.prod(level_clear_sky_fraction, dim=2)
        return total_cloud_cover
