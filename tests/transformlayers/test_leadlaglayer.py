# -----------------------------------------------------------
# Test time incorporated and invisibility reset transformlayers.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import torch

from psfdataset.pytorch.transformlayers import MultiDelayedTransformation


class TestLeadLagTransformLayer:
    def test_MultiDelayedTransformation(self):
        x = torch.tensor([[[1, 2], [2, 3], [3, 4], [4, 5]]],
                         dtype=torch.float,
                         requires_grad=True)
        mdt = MultiDelayedTransformation(1)
        out = mdt(x)

        expected = torch.tensor([[[1, 2, 0, 0], [2, 3, 1, 2], [3, 4, 2, 3],
                                  [4, 5, 3, 4], [0, 0, 4, 5]]])
        assert torch.all(torch.eq(out, expected))
        assert out.shape == mdt.output_shape(x.shape)

        out.backward(torch.ones([1, 5, 4], dtype=torch.float))
        expected_grad = torch.tensor([[[2, 2], [2, 2], [2, 2], [2, 2]]])
        assert torch.all(torch.eq(x.grad, expected_grad))

        # Test with multiple time step delay
        x = torch.tensor([[[1, 2], [2, 3], [3, 4], [4, 5]]],
                         dtype=torch.float,
                         requires_grad=True)
        mdt = MultiDelayedTransformation(2)
        out = mdt(x)

        exp = torch.tensor([[[1, 2, 0, 0, 0, 0], [2, 3, 1, 2, 0, 0],
                             [3, 4, 2, 3, 1, 2], [4, 5, 3, 4, 2, 3],
                             [0, 0, 4, 5, 3, 4], [0, 0, 0, 0, 4, 5]]])
        assert torch.all(torch.eq(out, exp))
        assert out.shape == mdt.output_shape(x.shape)

        out.backward(torch.ones(out.shape, dtype=torch.float))
        expected_grad = torch.tensor([[[3, 3], [3, 3], [3, 3], [3, 3]]])
        assert torch.all(torch.eq(x.grad, expected_grad))

        # Test with different array shapes
        x = torch.tensor([[[[1], [2], [3]], [[4], [5], [6]]]],
                         dtype=torch.float)
        mdt = MultiDelayedTransformation(1)
        out = mdt(x)

        exp = torch.tensor([[[[1, 0], [2, 0], [3, 0]], [[4, 1], [5, 2], [6,
                                                                         3]],
                             [[0, 4], [0, 5], [0, 6]]]])
        assert torch.all(torch.eq(out, exp))
        assert out.shape == mdt.output_shape(x.shape)

        # Test with different array shapes
        x = torch.tensor([[[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]]],
                         dtype=torch.float)
        mdt = MultiDelayedTransformation(1)
        out = mdt(x)

        exp = torch.tensor([[[[[1, 0]], [[2, 0]], [[3, 0]]],
                             [[[4, 1]], [[5, 2]], [[6, 3]]],
                             [[[0, 4]], [[0, 5]], [[0, 6]]]]])
        assert torch.all(torch.eq(out, exp))
        assert out.shape == mdt.output_shape(x.shape)
