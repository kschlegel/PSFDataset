# -----------------------------------------------------------
# Pytorch layer to compute the multi-delayed transformation.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
from typing import Tuple

import torch

from ...types import StructureDescription


class MultiDelayedTransformation(torch.nn.Module):
    """
    Compute the multi-delayed transformation of a path.

    Expects an at least 3-dimensional array of the form
    [batch][frame]...[coords]
    This is a variant of the lead-lag transformation which instead of advancing
    time one component at a time it advances time by one in each component
    every step so that each element of the multi-delayed path contains the last
    #delay elements of the original path. Pads with zeros at the ends.
    Returns an array of the form [batch][frame]...[(delay+1)*coords] which
    contains #delay extra timesteps. The [coords] dimension contains the
    original coords and a copy of the original path, delayed by i timesteps,
    e.g. with original coords [x,y] and a delay of 2 the new coords dimension
    contains [x_t,y_t,x_{t-1},y_{t-1},x_{t-2},y_{t-2}].

    Example:
    The path 1,2,3 with delay 1 turns into
        (1,0),(2,1),(3,2),(0,3)
    """
    def __init__(self, delay: int) -> None:
        """
        Parameters
        ----------
        delay : int, optional (default is 1)
            How many timesteps to delay the path
        """
        super().__init__()
        self._delay = delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = []
        for i in range(self._delay + 1):
            pad_front = torch.zeros((x.shape[0], i) + x.shape[2:],
                                    dtype=x.dtype)
            pad_back = torch.zeros((x.shape[0], self._delay - i) + x.shape[2:],
                                   dtype=x.dtype)
            rows.append(torch.cat((pad_front, x, pad_back), dim=1))
        return torch.cat(rows, dim=-1)

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
                self._delay) + input_shape[2:-1] + (input_shape[-1] *
                                                    (self._delay + 1), )

    def explain(self,
                input_structure: StructureDescription) -> StructureDescription:
        output_structure = input_structure.copy()
        # test for both here to tell mypy both are fine
        if (isinstance(output_structure[-1], list)
                and isinstance(input_structure[-1], list)):
            for i in range(self._delay):
                output_structure[-1].extend(
                    [c + "_d" + str(i + 1) for c in input_structure[-1]])
        elif isinstance(output_structure[-1], int):
            output_structure[-1] = input_structure[-1] * (self._delay + 1)
        return output_structure
