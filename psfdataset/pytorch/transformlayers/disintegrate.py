import torch


class Disintegrate(torch.nn.Module):
    """
    Simple class to add another dimension to the given input.

    This class simply is a wrapper for torch.unsqueeze(-1), adding another
    singluar dimension to the end of the vector, thus turning a n-dimensional
    path into n 1-dimensional paths.
    Provides output_shape() and explain() methods to propagate input structure
    through.
    """
    def forward(self, x):
        x = x.unsqueeze(-1)
        return x

    def output_shape(self, input_shape):
        """
        Parameters
        ----------
        input_shape : tuple of ints
            shape of the expected input array
        """
        return input_shape + (1, )

    def explain(self, input_structure):
        output_structure = input_structure.copy()
        output_structure.append(1)
        return output_structure
