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

"""Implementation of a Fenchel-Young loss using perturbation techniques."""

import torch
from typing import Callable, Optional
from perturbations_torch import perturbations


class FenchelYoungLoss(torch.nn.Module):
    """Implementation of a Fenchel Young loss."""

    def __init__(
        self,
        func=None,
        num_samples=1000,
        sigma=0.01,
        noise=perturbations._NORMAL,
        batched=True,
        maximize=True,
        reduction="sum",
    ):
        super().__init__()
        self._batched = batched
        self._maximize = maximize
        self.func = func
        self.perturbed = perturbations.perturbed(
            func=func,
            num_samples=num_samples,
            sigma=sigma,
            noise=noise,
            batched=batched,
        )
        self.reduction = reduction

    def forward(self, y_true, theta):
        class FYTorch(torch.autograd.Function):
            @staticmethod
            def forward(ctx, theta):
                diff = self.perturbed(theta) - y_true.type(theta.dtype)
                if not self._maximize:
                    diff = -diff
                # Computes per-example loss for batched inputs. If the total loss for the
                # batch is the desired output, use `SUM` or `SUM_OVER_BATCH` as reduction.
                if self._batched:
                    loss = (diff.view(diff.shape[0], -1) ** 2).sum(-1)
                else:
                    loss = (diff ** 2).sum()
                ctx.save_for_backward(diff)
                return loss

            @staticmethod
            def backward(ctx, dy):
                diff, = ctx.saved_tensors
                if self._batched:  # dy has shape (batch_size,) in this case.
                    dy = dy.view([dy.shape[0]] + [1] * (len(diff.shape) - 1))
                return dy * diff

        return FYTorch.apply(theta)
