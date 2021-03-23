# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for the fenchel_young module."""

import torch
from absl.testing import absltest
from perturbations_torch import fenchel_young as fy


def ranks(inputs, dim=-1):
    """Returns the ranks of the input values among the given axis."""
    return 1 + inputs.argsort(dim).argsort(dim).type(inputs.dtype)


class FenchelYoungTest(absltest.TestCase):
    """Testing the gradients obtained by the FenchelYoungLoss class."""

    def test_gradients(self):

        loss_fn = fy.FenchelYoungLoss(
            ranks, num_samples=10000, sigma=0.1, batched=False
        )

        theta = torch.FloatTensor([1, 20, 7.3, 7.35])
        y_true = torch.FloatTensor([1, 4, 3, 2])
        y_hard_minimum = torch.FloatTensor([1, 4, 2, 3])
        y_perturbed_minimum = loss_fn.perturbed(theta)

        def get_grad(y, theta):
            theta.requires_grad = True
            loss_fn(y, theta).backward()
            g = theta.grad.clone()
            theta.grad.zero_()
            return g

        g_true = get_grad(y_true, theta)
        g_hard_minimum = get_grad(y_hard_minimum, theta)
        g_perturbed_minimum = get_grad(y_perturbed_minimum, theta)
        assert torch.allclose(g_true[:2], torch.FloatTensor([0, 0]))
        assert torch.norm(g_perturbed_minimum) < torch.norm(g_hard_minimum)
        assert torch.norm(g_hard_minimum) < torch.norm(g_true)
        for g in [g_true, g_hard_minimum, g_perturbed_minimum]:
            assert torch.allclose(g.sum(), torch.FloatTensor([0.0]))


if __name__ == "__main__":
    absltest.main()
