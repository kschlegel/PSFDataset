# -----------------------------------------------------------
# Test time incorporated and invisibility reset transformlayers.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import torch

from psfdataset.pytorch.transformlayers import TimeIncorporatedTransformation
from psfdataset.pytorch.transformlayers import InvisibilityResetTransformation


class TestPathTransformations:
    def test_TimeIncorporatedTransform(self):
        x = torch.tensor([[[2], [4], [8]]],
                         dtype=torch.float,
                         requires_grad=True)
        ti = TimeIncorporatedTransformation()
        out = ti(x)

        expected = torch.tensor([[[2, 0], [4, 0.5], [8, 1]]])
        assert torch.all(torch.eq(out, expected))
        assert out.shape == ti.output_shape(x.shape)

        out.backward(torch.ones([1, 3, 2], dtype=torch.float))
        expected_grad = torch.tensor([[[1], [1], [1]]])
        assert torch.all(torch.eq(x.grad, expected_grad))

    def test_InvisibilityResetTransform(self):
        x = torch.tensor([[[1], [2], [3]]],
                         dtype=torch.float,
                         requires_grad=True)
        ir = InvisibilityResetTransformation()
        out = ir(x)

        expected = torch.tensor([[[1, 1], [2, 1], [3, 1], [3, 0], [0, 0]]])
        assert torch.all(torch.eq(out, expected))
        assert out.shape == ir.output_shape(x.shape)

        out.backward(torch.ones([1, 5, 2], dtype=torch.float))
        expected_grad = torch.tensor([[[1], [1], [2]]])
        assert torch.all(torch.eq(x.grad, expected_grad))
