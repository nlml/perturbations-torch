import torch
import functools
from typing import Tuple


_GUMBEL = "gumbel"
_NORMAL = "normal"
SUPPORTED_NOISES = (_GUMBEL, _NORMAL)


def sample_noise_with_gradients(noise, shape):
    """Samples a noise tensor according to a distribution with its gradient.

  Args:
   noise: (str) a type of supported noise distribution.
   shape: tf.Tensor<int>, the shape of the tensor to sample.

  Returns:
   A tuple Tensor<float>[shape], Tensor<float>[shape] that corresponds to the
   sampled noise and the gradient of log the underlying probability
   distribution function. For instance, for a gaussian noise (normal), the
   gradient is equal to the noise itself.

  Raises:
   ValueError in case the requested noise distribution is not supported.
   See perturbations.SUPPORTED_NOISES for the list of supported distributions.
  """
    if noise not in SUPPORTED_NOISES:
        raise ValueError(
            "{} noise is not supported. Use one of [{}]".format(
                noise, SUPPORTED_NOISES
            )
        )

    if noise == _GUMBEL:
        sampler = torch.distributions.gumbel.Gumbel(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = 1 - torch.exp(-samples)
    elif noise == _NORMAL:
        sampler = torch.distributions.normal.Normal(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = samples

    return samples, gradients


def perturbed(
    func=None, num_samples=1000, sigma=0.05, noise=_NORMAL, batched=True
):
    """Turns a function into a differentiable one via perturbations.

  The input function has to be the solution to a linear program for the trick
  to work. For instance the maximum function, the logical operators or the ranks
  can be expressed as solutions to some linear programs on some polytopes.
  If this condition is violated though, the result would not hold and there is
  no guarantee on the validity of the obtained gradients.

  This function can be used directly or as a decorator.

  Args:
   func: the function to be turned into a perturbed and differentiable one.
    Four I/O signatures for func are currently supported:
     If batched is True,
      (1) input [B, D1, ..., Dk], output [B, D1, ..., Dk], k >= 1
      (2) input [B, D1, ..., Dk], output [B], k >= 1
     If batched is False,
      (3) input [D1, ..., Dk], output [D1, ..., Dk], k >= 1
      (4) input [D1, ..., Dk], output [], k >= 1.
   num_samples: the number of samples to use for the expectation computation.
   sigma: the scale of the perturbation.
   noise: a string representing the noise distribution to be used to sample
    perturbations.
   batched: whether inputs to the perturbed function will have a leading batch
    dimension (True) or consist of a single example (False). Defaults to True.

  Returns:
   a function has the same signature as func but that can be back propagated.
  """
    # This is a trick to have the decorator work both with and without arguments.
    if func is None:
        return functools.partial(
            perturbed,
            num_samples=num_samples,
            sigma=sigma,
            noise=noise,
            batched=batched,
        )

    @functools.wraps(func)
    def wrapper(input_tensor, *args, **kwargs):
        def forward(input_tensor, *args, **kwargs):
            class PerturbedTch(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input_tensor):
                    """
                    In the forward pass we receive a Tensor containing the input and return
                    a Tensor containing the output. ctx is a context object that can be used
                    to stash information for backward computation. You can cache arbitrary
                    objects for use in the backward pass using the ctx.save_for_backward method.
                    """
                    original_input_shape = input_tensor.shape
                    orig_shape = torch.LongTensor(list(original_input_shape))
                    if batched:
                        if not len(original_input_shape) >= 2:
                            raise ValueError(
                                "Batched inputs must have at least rank two"
                            )
                    else:
                        input_tensor = input_tensor.unsqueeze(0)
                    input_shape = input_tensor.shape  # [B, D1, ... Dk], k >= 1
                    perturbed_input_shape = [num_samples] + list(input_shape)

                    noises = sample_noise_with_gradients(
                        noise, perturbed_input_shape
                    )
                    additive_noise, noise_gradient = [
                        noise.to(input_tensor.device).type(input_tensor.dtype)
                        for noise in noises
                    ]
                    perturbed_input = (
                        input_tensor.unsqueeze(0) + sigma * additive_noise
                    )

                    # [N, B, D1, ..., Dk] -> [NB, D1, ..., Dk].
                    flat_batch_dim_shape = [-1] + list(input_shape[1:])
                    perturbed_input = perturbed_input.view(
                        flat_batch_dim_shape
                    )
                    # Calls user-defined function in a perturbation agnostic manner.
                    perturbed_output = func(perturbed_input, *args, **kwargs)
                    # [NB, D1, ..., Dk] ->  [N, B, D1, ..., Dk].
                    perturbed_input = perturbed_input.view(
                        perturbed_input_shape
                    )
                    # Either
                    #   (Default case): [NB, D1, ..., Dk] ->  [N, B, D1, ..., Dk]
                    # or
                    #   (Full-reduce case) [NB] -> [N, B]
                    perturbed_output_shape = (
                        [num_samples] + [-1] + list(perturbed_output.shape[1:])
                    )
                    perturbed_output = perturbed_output.view(
                        perturbed_output_shape
                    )

                    forward_output = perturbed_output.mean(0)
                    if not batched:  # Removes dummy batch dimension.
                        forward_output = forward_output[0]
                    ctx.save_for_backward(
                        orig_shape, noise_gradient, perturbed_output
                    )
                    return forward_output
                    # ctx.save_for_backward(original_input_shape)
                    # return input_tensor.clamp(min=0)

                @staticmethod
                def backward(ctx, dy):
                    original_input_shape, noise_gradient, perturbed_output = (
                        ctx.saved_tensors
                    )
                    # perturbed_input_shape = [num_samples] if batched else [num_samples, 1] + list(original_input_shape)
                    # perturbed_input_rank = len(perturbed_input_shape)
                    perturbed_input_rank = len(original_input_shape) + (
                        1 if batched else 2
                    )

                    """Compute the gradient of the expectation via integration by parts."""
                    output, noise_grad = perturbed_output, noise_gradient
                    # Adds dummy feature/channel dimension internally.
                    if perturbed_input_rank > len(output.shape):
                        dy = dy.unsqueeze(-1)
                        output = output.unsqueeze(-1)
                    # Adds dummy batch dimension internally.
                    if not batched:
                        dy = dy.unsqueeze(0)
                    # Flattens [D1, ..., Dk] to a single feat dim [D].
                    flatten = lambda t: t.view(t.shape[0], t.shape[1], -1)
                    dy = dy.view(dy.shape[0], -1)  # (B, D)
                    output = flatten(output)  # (N, B, D)
                    noise_grad = flatten(noise_grad)  # (N, B, D)

                    g = torch.einsum(
                        "nbd,nb->bd",
                        noise_grad,
                        torch.einsum("nbd,bd->nb", output, dy),
                    )
                    g /= sigma * num_samples
                    return g.view(*original_input_shape)

            return PerturbedTch.apply(input_tensor)

        return forward(input_tensor, *args, **kwargs)

    return wrapper
