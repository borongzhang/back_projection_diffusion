"""Alpha-conditioned denoiser models."""

from collections.abc import Callable, Sequence
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import layers

Array = jax.Array
Initializer = nn.initializers.Initializer
PrecisionLike = (None | str | jax.lax.Precision |
                 tuple[str, str] |
                 tuple[jax.lax.Precision, jax.lax.Precision])


def default_init(scale: float = 1e-10) -> Initializer:
  """Return a variance scaling initializer with the given scale."""
  return nn.initializers.variance_scaling(
    scale=scale, mode="fan_avg", distribution="uniform"
  )


class AdaptiveScale(nn.Module):
  """Adaptively scaled input based on embedding.

  Conditional information is projected to two vectors of length c,
  where c is the number of channels of x. Then, x is scaled channel-
  wise by the first vector and offset by the second vector.
  """
  act_fun: Callable[[Array], Array] = nn.swish
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, emb: Array) -> Array:
    """Applied adaptive scaling to the channel dimension."""
    assert emb.ndim == 2, (
      "The embedding dimension must be 2, instead it was: " + str(emb.ndim)
    )
    affine = nn.Dense(
      features=x.shape[-1] * 2,
      kernel_init=default_init(1.0),
      precision=self.precision,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
    )
    scale_params = affine(self.act_fun(emb))
    scale_params = scale_params.reshape(
      scale_params.shape[:1] + (x.ndim - 2) * (1,) +
      scale_params.shape[1:]
    )
    scale, bias = jnp.split(scale_params, 2, axis=-1)
    return x * (scale + 1.0) + bias


class ConvBlock(nn.Module):
  """Basic two-layer convolution block with adaptive scaling.

  Main path:
    GroupNorm --> Swish --> Conv_Squeeze -->
    GroupNorm --> FiLM --> Swish --> Conv_Expand.
  """
  squeeze_channels: int
  out_channels: int
  kernel_size: tuple[int, ...]
  padding: str = "CIRCULAR"
  film_act_fun: Callable[[Array], Array] = nn.swish
  act_fun: Callable[[Array], Array] = nn.swish
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, emb: Array, is_training: bool) -> Array:
    """Applied convolution block operations to the input."""
    h = x
    h = nn.GroupNorm(min(h.shape[-1] // 4, 32))(h)
    h = self.act_fun(h)
    h = layers.ConvLayer(
      features=self.squeeze_channels,
      kernel_size=self.kernel_size,
      padding=self.padding,
      kernel_init=default_init(1.0),
      precision=self.precision,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      name="conv_squeeze",
    )(h)
    # (batch, 80, 80, squeeze_channels)
    h = nn.GroupNorm(min(h.shape[-1] // 4, 32))(h)
    h = AdaptiveScale(act_fun=self.film_act_fun)(h, emb)
    h = self.act_fun(h)
    h = layers.ConvLayer(
      features=self.out_channels,
      kernel_size=self.kernel_size,
      padding=self.padding,
      kernel_init=default_init(1.0),
      precision=self.precision,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      name="conv_expand",
    )(h)
    # (batch, 80, 80, out_channels)
    return layers.CombineResidualWithSkip(
      project_skip=True,
      dtype=self.dtype,
      precision=self.precision,
      param_dtype=self.param_dtype,
    )(residual=h, skip=x)


class FourierEmbedding(nn.Module):
  """Applied Fourier embedding."""
  dims: int = 64
  max_freq: float = 2e4
  projection: bool = True
  act_fun: Callable[[Array], Array] = nn.swish
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """Computed Fourier embedding for the input."""
    assert x.ndim == 1
    logfreqs = jnp.linspace(0, jnp.log(self.max_freq), self.dims // 2)
    x = jnp.pi * jnp.exp(logfreqs)[None, :] * x[:, None]
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    if self.projection:
      x = nn.Dense(
        features=2 * self.dims,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
      )(x)
      x = self.act_fun(x)
      x = nn.Dense(
        features=self.dims,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
      )(x)
    return x


class MergeChannelCond(nn.Module):
  """Base class for merging conditional inputs along channels.

  Attributes:
    embed_iter: Number of merging iterations.
    kernel_size: Convolutional kernel size.
    padding: Padding method for convolutions.
  """
  embed_iter: int
  kernel_size: Sequence[int]
  padding: str = "CIRCULAR"
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32


class InterpConvMerge(MergeChannelCond):
  """
  Merged conditional inputs via interpolation and convolutions.

  Args:
    embed_iter: Number of embedding iterations.
    kernel_size: Convolutional kernel size.
  """
  embed_iter: int
  kernel_size: tuple[int, ...]

  @nn.compact
  def __call__(self, x: Array, cond: dict[str, Array],
               fstars: list[nn.Module]) -> Array:
    """Merged conditional inputs with the main input."""
    values = []
    for n, (key, value) in enumerate(sorted(cond.items())):
      values.append(fstars[n](value))
    y = nn.LayerNorm()(jnp.concatenate(values, axis=-1))
    # (batch, height, width, num_cond_features)
    for _ in range(self.embed_iter):
      tmp = nn.Conv(
        features=6,
        kernel_size=self.kernel_size,
        padding="CIRCULAR"
      )(y)
      tmp = nn.swish(tmp)
      tmp = nn.Conv(
        features=6,
        kernel_size=self.kernel_size,
        padding="CIRCULAR"
      )(tmp)
      y = jnp.concatenate([y, tmp], axis=-1)
      # (batch, height, width, num_features + 6)
    x = jnp.concatenate([x, y], axis=-1)
    # (batch, height, width, input_channels + num_features)
    return x


class FStarNet(nn.Module):
  """
  Neural network with conditional embedding and multiple conv layers.

  Attributes:
    fstars: List of modules representing the back projection operator.
    out_channels: Number of output channels.
    squeeze_ratio: Ratio for channel squeeze.
    noise_embed_dim: Dimension of the noise embedding.
    cond_embed_iter: Number of iterations for conditional embedding.
    cond_merging_fn: Function to merge conditional channels.
    num_conv: Number of convolutional blocks.
    num_feature: Number of features in conv layers.
    dtype: Input data type.
    param_dtype: Parameter data type.
  """
  fstars: list[nn.Module]
  out_channels: int = 1
  squeeze_ratio: int = 8
  noise_embed_dim: int = 64
  cond_embed_iter: int = 10
  cond_merging_fn: type[MergeChannelCond] = InterpConvMerge
  num_conv: int = 12
  num_feature: int = 96
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, sigma: Array,
               cond: dict[str, Array] | None = None,
               *, is_training: bool) -> Array:
    """Executed forward pass through the network."""
    kernel_dim = x.ndim - 2
    cond = cond or {}
    y = self.cond_merging_fn(
      embed_iter=self.cond_embed_iter,
      kernel_size=(3,) * kernel_dim,
    )(x, cond, self.fstars)
    # (batch, 80, 80, cond_embed_dim + ?)
    emb = FourierEmbedding(dims=self.noise_embed_dim)(sigma)
    y = layers.ConvLayer(
      features=self.num_feature,
      kernel_size=kernel_dim * (3,),
      kernel_init=default_init(1.0),
      name="conv_in",
      padding="CIRCULAR",
    )(y)
    # (batch, 80, 80, num_feature = 96)
    for n in range(self.num_conv):
      y = ConvBlock(
        squeeze_channels=self.num_feature // self.squeeze_ratio,
        out_channels=self.num_feature,
        kernel_size=kernel_dim * (3,),
        padding="CIRCULAR",
        name=f"conv{n}",
      )(y, emb, is_training=is_training)
    y = layers.ConvLayer(
      features=self.out_channels,
      kernel_size=kernel_dim * (3,),
      kernel_init=default_init(1.0),
      name="conv_out",
      padding="CIRCULAR",
    )(y)
    # (batch, 80, 80, out_channels = 1)
    return y


class PreconditionedDenoiser(FStarNet):
  """Preconditioned denoising model as in Karras et al."""
  sigma_data: float = 1.0

  @nn.compact
  def __call__(self, x: Array, sigma: Array,
               cond: dict[str, Array] | None = None,
               *, is_training: bool) -> Array:
    """Executed preconditioned denoising."""
    if sigma.ndim < 1:
      sigma = jnp.broadcast_to(sigma, (x.shape[0],))
    if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
        "sigma must be 1D and have the same batch dim as x "
        f"({x.shape[0]})!"
      )
    total_var = jnp.square(self.sigma_data) + jnp.square(sigma)
    c_skip = jnp.square(self.sigma_data) / total_var
    c_out = sigma * self.sigma_data / jnp.sqrt(total_var)
    c_in = 1 / jnp.sqrt(total_var)
    c_noise = 0.25 * jnp.log(sigma)
    c_in = jnp.expand_dims(c_in,
      axis=np.arange(x.ndim - 1, dtype=np.int32) + 1)
    c_out = jnp.expand_dims(c_out, axis=np.arange(x.ndim - 1) + 1)
    c_skip = jnp.expand_dims(c_skip, axis=np.arange(x.ndim - 1) + 1)
    f_x = super().__call__(jnp.multiply(c_in, x), c_noise, cond,
                           is_training=is_training)
    return jnp.multiply(c_skip, x) + jnp.multiply(c_out, f_x)
