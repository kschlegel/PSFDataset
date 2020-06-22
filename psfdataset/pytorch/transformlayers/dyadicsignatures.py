# -----------------------------------------------------------
# Pytroch layers to compute dyadic signatures.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
from typing import Tuple, List

import torch
import signatory

from ...types import StructureDescription, StructureElement


class DyadicSignatures(torch.nn.Module):
    """
    Take signatures of (dyadic intervals of) a path.

    Takes a tensor of the form [batch][frame][element][coords]. Optionally
    splits the frame axis into (possibly half overlapping) dyadic intervals.
    Computes the signatures for each channel. Returns a tensor of the form
    [batch][channel][dyadic_piece][signature_terms].
    """
    def __init__(self,
                 dyadic_levels: int,
                 signature_level: int,
                 overlapping: bool = False,
                 drop_zeroth_term: bool = True) -> None:
        """
        Parameters
        ----------
        dyadic_levels : int
            number of dyadic splits to do. Level 0 corresponds to just the
            original path, 1 is the path and its halfs, etc.
        signature_level: int, optional (default is 2)
            level of signatures to be computed
        overlapping: bool, optional (default is False)
            Whether to take the dyadic intervals half overlapping each other
        """
        super().__init__()
        self._dyadic_levels = dyadic_levels
        self._signature_level = signature_level
        self._overlapping = overlapping
        self._drop_zeroth_term = drop_zeroth_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            x = x.to(torch.float)
        sigs = []
        for i in range(x.shape[2]):
            path = signatory.Path(x[:, :, i],
                                  self._signature_level,
                                  scalar_term=not self._drop_zeroth_term)
            ch_sigs = []
            for dy_lvl in range(self._dyadic_levels + 1):
                frames_per_piece = x.shape[1] / 2**dy_lvl
                if self._overlapping:
                    num_pieces = 2**(dy_lvl + 1) - 1
                    step_size = frames_per_piece / 2
                else:
                    num_pieces = 2**dy_lvl
                    step_size = frames_per_piece
                dy_sigs = []
                for j in range(num_pieces):
                    start_frame = j * step_size
                    end_frame = start_frame + frames_per_piece
                    start_frame = int(start_frame)
                    end_frame = int(end_frame)
                    if j == num_pieces - 1 and end_frame != x.shape[1]:
                        end_frame = x.shape[1]
                    dy_sigs.append(path.signature(start_frame, end_frame))
                # dy_sigs contains elements of the form [batch][sig_terms]
                # stack over dyadic pieces dimension
                ch_sigs.append(torch.stack(dy_sigs, dim=1))
            # sigs contains elements of the form [batch][dy_piece][sig_terms]
            # concatenate over the diadic pice dimension to get all dyadic
            # pieces across levels into the dyadic pice dimension
            sigs.append(torch.cat(ch_sigs, dim=1))
        # sigs contains elements of the form [batch][dyadic_piece][sig_terms]
        # with all dyadic pieces in the dyadic piece dimension
        # stack over channel dimension
        output = torch.stack(sigs, dim=1)
        return output

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Parameters
        ----------
        input_shape : tuple of ints
            shape of the expected input array
        """
        if len(input_shape) != 4:
            raise Exception("The input array must be 4 dimensional")
        if self._overlapping:
            dyadic_pieces = (2**(self._dyadic_levels + 2) -
                             self._dyadic_levels - 3)
        else:
            dyadic_pieces = 2**(self._dyadic_levels + 1) - 1
        signature_terms = signatory.signature_channels(
            input_shape[3],
            self._signature_level,
            scalar_term=not self._drop_zeroth_term)
        return (input_shape[0], input_shape[2], dyadic_pieces, signature_terms)

    def explain(self,
                input_structure: StructureDescription) -> StructureDescription:
        """
        Expected input structure: [time, elements, D]
        """
        output_structure: List[StructureElement] = [
            input_structure[1], [], input_structure[2]
        ]
        for dyadic_level in range(self._dyadic_levels + 1):
            if self._overlapping:
                num_pieces = 2**(dyadic_level + 1) - 1
            else:
                num_pieces = 2**dyadic_level
            # MyPy does not realise this has been set to an empty list above
            output_structure[1].extend([  # type: ignore
                "lvl" + str(dyadic_level) + "p" + str(i)
                for i in range(num_pieces)
            ])
        if isinstance(input_structure[2], list):
            output_structure[2] = len(input_structure[2])
        output_structure[2] = signatory.signature_channels(
            output_structure[2],
            self._signature_level,
            scalar_term=not self._drop_zeroth_term)
        return output_structure
