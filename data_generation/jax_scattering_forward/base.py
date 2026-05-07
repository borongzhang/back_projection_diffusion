from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import jax
import jax.numpy as jnp


class BaseOperator(ABC):
    """Base class for JAX measurement operators.

    Notes vs Torch:
        Randomness in JAX is explicit: pass ``rng`` to ``__call__``.
        ``gradient`` uses ``jax.grad`` on a scalar batch loss.
    """

    def __init__(
        self,
        sigma_noise: float = 0.0,
        unnorm_shift: float = 0.0,
        unnorm_scale: float = 1.0,
        device: str = "cuda",
    ) -> None:
        self.sigma_noise = float(sigma_noise)
        self.unnorm_shift = float(unnorm_shift)
        self.unnorm_scale = float(unnorm_scale)
        self._device = self._get_device(device)

    @staticmethod
    def _get_device(device: str) -> jax.Device:
        """Return the requested JAX device when available."""
        device_kind = "gpu" if device == "cuda" else "cpu"
        try:
            devices = jax.devices(device_kind)
        except RuntimeError:
            devices = []
        return devices[0] if devices else jax.devices()[0]

    @abstractmethod
    def forward(self, inputs, **kwargs):
        """Apply the forward measurement map."""
        raise NotImplementedError

    def __call__(
        self,
        inputs: Mapping[str, jax.Array],
        rng: jax.Array | None = None,
        **kwargs,
    ):
        target = inputs["target"]
        output = self.forward(target, **kwargs)
        if self.sigma_noise == 0.0:
            return output
        if rng is None:
            rng = jax.random.PRNGKey(0)
        noise = jax.random.normal(rng, shape=output.shape, dtype=output.dtype)
        return output + self.sigma_noise * noise

    def loss(self, pred, observation, **kwargs):
        """Return the batchwise squared L2 data misfit."""
        diff = self.forward(pred, **kwargs) - observation
        diff = jnp.reshape(diff, (diff.shape[0], -1))
        return jnp.sum(diff * diff, axis=1)

    def loss_m(self, measurements, observation):
        """Return the batchwise squared L2 misfit in measurement space."""
        diff = measurements - observation
        diff = jnp.reshape(diff, (diff.shape[0], -1))
        return jnp.sum(diff * diff, axis=1)

    def gradient(self, pred, observation, return_loss: bool = False):
        """Return the gradient of the summed batch loss with respect to pred."""

        def scalar_loss(p):
            return jnp.sum(self.loss(p, observation))

        grad = jax.grad(scalar_loss)(pred)
        if return_loss:
            return grad, scalar_loss(pred)
        return grad

    def gradient_m(self, measurements, observation):
        """Return the gradient of the summed measurement-space loss."""

        def scalar_loss(m):
            return jnp.sum(self.loss_m(m, observation))

        return jax.grad(scalar_loss)(measurements)

    def unnormalize(self, inputs):
        return (inputs + self.unnorm_shift) * self.unnorm_scale

    def normalize(self, inputs):
        return inputs / self.unnorm_scale - self.unnorm_shift

    @staticmethod
    def as_output_array(inputs):
        """Convert operator outputs to JAX arrays for downstream code."""
        return jnp.asarray(inputs)

    def close(self):
        """Release resources held by the operator."""
