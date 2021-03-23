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
"""Tests for differentiable_programming.perturbations."""

from absl.testing import parameterized
from absl.testing import absltest
from perturbations_torch import perturbations
import torch


def reduce_sign_any(input_tensor, dim=-1):
    """A logical or of the signs of a tensor along an axis.

    Args:
    input_tensor: Tensor<float> of any shape.
    axis: the axis along which we want to compute a logical or of the signs of
        the values.

    Returns:
    A Tensor<float>, which as the same shape as the input tensor, but without the
    axis on which we reduced.
    """
    boolean_sign = torch.any(
        ((torch.sign(input_tensor) + 1) / 2.0).bool(), dim=dim
    )
    return boolean_sign.type(input_tensor.dtype) * 2.0 - 1.0


class PerturbationsTest(parameterized.TestCase):
    """Testing the perturbations module."""

    def assertAllEqual(self, a, b):
        print(a)
        print(b)
        assert torch.all(a == b)

    def setUp(self):
        super(PerturbationsTest, self).setUp()
        torch.manual_seed(0)

    @parameterized.parameters([perturbations._GUMBEL, perturbations._NORMAL])
    def test_sample_noise_with_gradients(self, noise):
        shape = (3, 2, 4)
        samples, gradients = perturbations.sample_noise_with_gradients(
            noise, shape
        )
        assert list(samples.shape) == list(shape)
        assert list(gradients.shape) == list(shape)

    def test_sample_noise_with_gradients_raise(self):
        with self.assertRaises(ValueError):
            _, _ = perturbations.sample_noise_with_gradients(
                "unknown", (3, 2, 4)
            )

    @parameterized.parameters([1e-3, 1e-2, 1e-1])
    def test_perturbed_reduce_sign_any(self, sigma):
        input_tensor = torch.FloatTensor(
            [[-0.3, -1.2, 1.6], [-0.4, -2.4, -1.0]]
        )
        soft_reduce_any = perturbations.perturbed(reduce_sign_any, sigma=sigma)
        output_tensor = soft_reduce_any(input_tensor, dim=-1)
        assert torch.allclose(
            output_tensor, torch.FloatTensor([1.0, -1.0]), atol=0.01
        )

    def test_perturbed_reduce_sign_any_gradients(self):
        # We choose a point where the gradient should be above noise, that is
        # to say the distance to 0 along one direction is about sigma.
        sigma = 0.1
        input_tensor = torch.FloatTensor(
            [[-0.6, -1.2, 0.5 * sigma], [-2 * sigma, -2.4, -1.0]]
        )
        soft_reduce_any = perturbations.perturbed(reduce_sign_any, sigma=sigma)
        input_tensor.requires_grad = True
        output_tensor = soft_reduce_any(input_tensor)
        output_tensor.sum().backward()
        gradient = input_tensor.grad
        # The two values that could change the soft logical or should be the one
        # with real positive impact on the final values.
        assert gradient[0, 2] > 0.0
        assert gradient[1, 0] > 0.0
        # The value that is more on the fence should bring more gradient than any
        # other one.
        assert torch.all(gradient <= gradient[0, 2])

    def test_unbatched_rank_one_raise(self):
        with self.assertRaises(ValueError):
            input_tensor = torch.FloatTensor([-0.6, -0.5, 0.5])
            dim = len(input_tensor)
            n = 10000000

            argmax = lambda t: torch.nn.functional.one_hot(
                torch.argmax(t, 1), dim
            )
            soft_argmax = perturbations.perturbed(
                argmax, sigma=0.5, num_samples=n
            )
            _ = soft_argmax(input_tensor)

    def test_perturbed_argmax_gradients_without_minibatch(self):
        input_tensor = torch.FloatTensor([-0.6, -0.5, 0.5])
        dim = len(input_tensor)
        eps = 1e-2
        n = 10000000

        argmax = lambda t: torch.nn.functional.one_hot(
            t.argmax(1), dim
        ).float()
        soft_argmax = perturbations.perturbed(
            argmax, sigma=0.5, num_samples=n, batched=False
        )
        norm_argmax = lambda t: (soft_argmax(t) ** 2).sum()

        w = torch.randn(input_tensor.shape)
        w /= torch.linalg.norm(w)
        var = input_tensor
        var.requires_grad = True
        value = norm_argmax(var)

        value.backward()
        grad = var.grad
        grad = grad.view(input_tensor.shape)

        value_minus = norm_argmax(input_tensor - eps * w)
        value_plus = norm_argmax(input_tensor + eps * w)

        lhs = (w * grad).sum()
        rhs = (value_plus - value_minus) * 1.0 / (2 * eps)
        assert torch.all(torch.abs(lhs - rhs) < 0.05)

    def test_perturbed_argmax_gradients_with_minibatch(self):
        input_tensor = torch.FloatTensor(
            [[-0.6, -0.7, 0.5], [0.9, -0.6, -0.5]]
        )
        dim = input_tensor.shape[-1]
        eps = 1e-2
        n = 10000000

        argmax = lambda t: torch.nn.functional.one_hot(
            t.argmax(-1), dim
        ).float()
        soft_argmax = perturbations.perturbed(argmax, sigma=2.5, num_samples=n)
        norm_argmax = lambda t: (soft_argmax(t) ** 2).sum()

        w = torch.randn(input_tensor.shape)
        w /= torch.linalg.norm(w)
        var = input_tensor
        var.requires_grad = True
        value = norm_argmax(var)

        value.backward()
        grad = var.grad
        grad = grad.view(input_tensor.shape)

        value_minus = norm_argmax(input_tensor - eps * w)
        value_plus = norm_argmax(input_tensor + eps * w)

        lhs = (w * grad).sum()
        rhs = (value_plus - value_minus) * 1.0 / (2 * eps)
        assert torch.all(torch.abs(lhs - rhs) < 0.05)


if __name__ == "__main__":
    absltest.main()
