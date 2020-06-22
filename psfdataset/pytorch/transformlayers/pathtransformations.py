# -----------------------------------------------------------
# Pytroch layers to compute time incorporated and invisibility reset
# transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
from typing import Tuple

import torch

from ...types import StructureDescription


class TimeIncorporatedTransformation(torch.nn.Module):
    """
    Compute the time incorporated transformation of the path.

    Takes an array of the form [batch][frame]...[coords]
    Adds an extra dimension to coords for time, advancing time linearly from 0
    to 1 along the frame axis.

    Example:
    The path 2,8,4 turns into
        (2,0),(8,1),(4,2)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = torch.tensor([i / (x.shape[1] - 1) for i in range(x.shape[1])])
        if not x.is_floating_point():
            x = x.to(torch.float)
        output = torch.cat(
            (x, time.repeat((x.shape[0], 1) + x.shape[2:-1] +
                            (1, )).transpose(1, -1)),
            dim=-1)
        return output

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Parameters
        ----------
        input_shape : tuple of ints
            shape of the expected input array
        """
        if len(input_shape) < 3:
            raise Exception("The input array must be at least 3 dimensional")
        return input_shape[:-1] + (input_shape[-1] + 1, )

    def explain(self,
                input_structure: StructureDescription) -> StructureDescription:
        output_structure = input_structure.copy()
        if isinstance(output_structure[-1], list):
            output_structure[-1].append("t")
        elif isinstance(output_structure[-1], int):
            output_structure[-1] += 1
        return output_structure


class InvisibilityResetTransformation(torch.nn.Module):
    """
    Computes the invisibility reset transformation of the path.

    Takes an array of the form [batch][frame]...[coords]
    Adds a visibility dimension to coords and two extra time steps to the path.
    The visibility coordinate is set to 1 for all original steps of the path,
    and 0 for the two new steps. The first of the two new steps is a copy of
    the last step of the original path, the second one is equal to zero.

    Example:
    The Path 1,2,3 turns into
        (1,1),(2,1),(3,1),(3,0),(0,0)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extended = torch.cat(
            (x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype)), dim=-1)
        extra = torch.cat(
            (x[:, -1].unsqueeze(1),
             torch.zeros(
                 (x.shape[0], 1) + x.shape[2:-1] + (1, ), dtype=x.dtype)),
            dim=-1)
        extended = torch.cat((extended, extra), dim=1)
        extra = torch.zeros(extra.shape, dtype=x.dtype)
        extended = torch.cat((extended, extra), dim=1)
        return extended

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Parameters
        ----------
        input_shape : tuple of ints
            shape of the expected input array
        """
        if len(input_shape) < 3:
            raise Exception("The input array must be at least 3 dimensional")
        return (input_shape[0], input_shape[1] +
                2) + input_shape[2:-1] + (input_shape[-1] + 1, )

    def explain(self,
                input_structure: StructureDescription) -> StructureDescription:
        output_structure = input_structure.copy()
        if isinstance(output_structure[-1], list):
            output_structure[-1].append("vis")
        elif isinstance(output_structure[-1], int):
            output_structure[-1] += 1
        return output_structure
