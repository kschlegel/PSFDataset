import torch


class TransformBlock(torch.nn.Module):
    def __init__(self, transforms):
        """
        """
        super().__init__()
        self._transforms = transforms

    def forward(self, x):
        for tr in self._transforms:
            x = tr(x)
        return x

    def output_shape(self, input_shape):
        """
        Parameters
        ----------
        input_shape : tuple of ints
            shape of the expected input array
        """
        for tr in self._transforms:
            input_shape = tr.explain(input_shape)
        return input_shape

    def explain(self, input_structure):
        output_structure = input_structure.copy()
        for tr in self._transforms:
            output_structure = tr.explain(output_structure)
        return output_structure
