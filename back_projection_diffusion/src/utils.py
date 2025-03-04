import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from scipy.ndimage import geometric_transform
from scipy.ndimage import gaussian_filter
import h5py
import natsort
import tensorflow as tf


def rotation_index(n: int) -> Array:
  """Creates an array of indices based on rotating rows and columns of a 2D array.

  Args:
    n: Dimension of the square grid (n x n).

  Returns:
    jnp.ndarray: A concatenated array of rotation indices.
  """
  index = jnp.reshape(jnp.arange(0, n**2, 1), [n, n])
    
  # Rotate the grid for each shift and concatenate the results.
  return jnp.concatenate(
    [jnp.roll(index, shift=[-i, -i], axis=[0, 1]) for i in range(n)], 0
  )
    
def sparse_polar_to_cartesian(neta: int, nx: int) -> BCOO:
  """Converts a Cartesian grid to polar coordinates and creates a sparse matrix.

  Args:
    neta: The size of the Cartesian grid (neta x neta).
    nx: The size of the polar grid (nx x nx).

  Returns:
    BCOO: A sparse matrix in BCOO format representing the transformation.
  """
  def shift_func(coords: tuple) -> tuple:
    """Transforms Cartesian coordinates to polar coordinates.

    Args:
      coords: A pair of coordinates (i, j) in the Cartesian grid.

    Returns:
      tuple: Polar coordinates (theta, rho) scaled to match the transformation.
    """
    i, j = coords[0], coords[1]
    # Calculate the radial distance rho.
    rho = 2 * np.sqrt((i - neta / 2) ** 2 + (j - neta / 2) ** 2) * nx / neta
    # Calculate the angular coordinate theta.
    # Adjust theta to fit within [0, 2π] using modulo.
    theta = ((np.arctan2((neta / 2 - j), (i - neta / 2))) % (2 * np.pi)) * \
            nx / (2 * np.pi)
    # Return the scaled polar coordinates.
    return theta, rho + neta // 2

  cart_mat = np.zeros((neta**2, nx, nx))
  # Loop through each pixel in the polar grid (nx x nx).
  for i in range(nx):
    # Prints progress to track computation for each row.
    print(f"\rProcessing row {i + 1} / {nx}", end="", flush=True)
    for j in range(nx):
      # Creates a dummy matrix with a single pixel set to 1 at (i, j).
      mat_dummy = np.zeros((nx, nx))
      mat_dummy[i, j] = 1
      # Pads the matrix to prevent shifting out of bound.
      pad_dummy = np.pad(mat_dummy, ((0, 0), (neta // 2, neta // 2)), "edge")
      # Maps each output point to the corresponding input coordinates using spline
      # interpolation.
      cart_mat[:, i, j] = geometric_transform(
        pad_dummy,
        shift_func,
        output_shape=[neta, neta],
        mode="grid-wrap"
      ).flatten()

  cart_mat = np.reshape(cart_mat, (neta**2, nx**2))
  # Thresholds the matrix to remove small values.
  cart_mat = np.where(np.abs(cart_mat) > 0.001, cart_mat, 0)
  # Converts the dense matrix to a sparse BCOO format.
  return BCOO.fromdense(cart_mat)

def sparse_cartesian_to_polar(neta: int, nx: int) -> BCOO:
  """Converts a polar grid to a cartesian grid and creates a sparse matrix.

  Args:
    neta: The size of the cartesian grid (neta x neta).
    nx: The size of the polar grid (nx x nx).

  Returns:
    BCOO: A sparse matrix in BCOO format representing the transformation from
      polar to cartesian coordinates.
  """
  def shift_func(coords: tuple) -> tuple:
    """Transforms polar coordinates to cartesian coordinates.

    Args:
      coords: A pair of coordinates (theta_index, rho_index) in the polar grid.

    Returns:
      tuple: Cartesian coordinates (i, j) scaled to match the transformation.
    """
    # Calculate the angular coordinate theta from its scaled index.
    theta = 2 * np.pi * coords[0] / nx
    # Calculate the radial distance rho from its scaled index.
    rho = coords[1] / 2
    # Convert polar coordinates to cartesian coordinates.
    i = nx / 2 + rho * np.cos(theta)
    j = nx / 2 - rho * np.sin(theta)
    # Return the cartesian coordinates.
    return i, j

  polar_mat = np.zeros((nx**2, neta, neta))
  # Loop through each pixel in the cartesian grid.
  for i in range(neta):
    for j in range(neta):
      # Create a dummy matrix with a single pixel set to 1 at (i, j).
      mat_dummy = np.zeros((neta, neta))
      mat_dummy[i, j] = 1
      # Map each output point to input coordinates using spline interpolation.
      polar_mat[:, i, j] = geometric_transform(
        mat_dummy,
        shift_func,
        output_shape=[nx, nx],
        order=3,
        mode="nearest"
      ).flatten()

  polar_mat = np.reshape(polar_mat, (nx**2, neta**2))
  # Threshold the matrix to remove small values.
  polar_mat = np.where(np.abs(polar_mat) > 0.001, polar_mat, 0)
  # Convert the dense matrix to a sparse BCOO format.
  return BCOO.fromdense(polar_mat)

def compute_f_adj(freq: float) -> jnp.ndarray:
  """Computes the analytical adjoint of the forward map for nx = neta = 80.

  Args:
    freq: The frequency for which to compute the adjoint.

  Returns:
    jnp.ndarray: The adjoint of the forward map as a complex matrix.
  """
  omega = 2 * jnp.pi * freq

  X, Y = jnp.meshgrid(jnp.linspace(0, 1, 80) - 0.5,
                      jnp.linspace(0, 1, 80) - 0.5)
  pa, qa = jnp.meshgrid(jnp.linspace(jnp.pi, 3 * jnp.pi, 80),
                        jnp.linspace(jnp.pi, 3 * jnp.pi, 80))

  p1 = jnp.cos(pa).ravel()
  p2 = jnp.sin(pa).ravel()
  q1 = jnp.cos(qa).ravel()
  q2 = jnp.sin(qa).ravel()

  X_flat = X.ravel()
  Y_flat = Y.ravel()

  # Compute the complex exponential components.
  F = (jnp.exp(1j * jnp.pi / 4) / jnp.sqrt(8 * jnp.pi * omega) *
       (omega**2 * jnp.exp(1j * omega * 0.5) / jnp.sqrt(0.5)) *
       jnp.exp(1j * omega * ((p1[:, None] - q1[:, None]) *
                              X_flat +
                              (p2[:, None] - q2[:, None]) *
                              Y_flat)) / 79**2)
  return F.conj().T

def load_eta_data(data_path: str, N: int, blur_sigma: float,
                  normalize: bool = False):
  """Loads and processes η data from an HDF5 file.

  Args:
    data_path: Path to the folder containing 'eta.h5'.
    N: Number of samples to load.
    blur_sigma: Sigma value for the gaussian blur.
    normalize: If True, normalizes the data and returns the normalization
      constants.

  Returns:
    If normalize is True:
      tuple: (eta, mean_eta, std_eta) where eta is the normalized data.
    Otherwise:
      np.ndarray: The processed (but not normalized) eta data.
  """
  with h5py.File(data_path, "r") as f:
    raw_eta = f[list(f.keys())[0]][:N, :]
    # Determine the spatial dimension (assumes square images).
    neta = int(np.sqrt(raw_eta.shape[1]))
    # Reshape raw data into (N, neta, neta).
    eta = raw_eta.reshape(-1, neta, neta)
    # Apply gaussian blur by transposing each image.
    blur_fn = lambda x: gaussian_filter(x, sigma=blur_sigma)
    eta = np.stack([blur_fn(img.T) for img in eta]).astype("float32")

  if normalize:
    # Compute normalization constants.
    mean_eta = np.mean(eta, axis=0)
    std_eta = np.std(eta)
    # Normalize the data.
    eta = (eta - mean_eta) / std_eta
    return eta, mean_eta, std_eta
  else:
    return eta

def load_scatter_data(data_path: str, N: int,
                      scatter_norm_constants: list = None):
  """Loads and processes scatter data from an HDF5 file.

  The file is assumed to contain real parts (keys 3, 4, 5) and imaginary parts
  (keys 0, 1, 2) after natural sorting of keys.

  Args:
    data_path: Path to the folder containing the scatter HDF5 file.
    N: Number of samples to load.
    scatter_norm_constants: If None, compute per-channel normalization
      constants, normalize the data, and return the constants. Otherwise,
      use the provided values.

  Returns:
    If scatter_norm_constants is None:
      tuple: (scatter, computed_norm_constants) where scatter is the normalized
        data.
    Otherwise:
      np.ndarray: The normalized scatter data.
  """
  with h5py.File(data_path, "r") as f:
    keys = natsort.natsorted(f.keys())
    # Process the real part from keys 3, 4, 5.
    tmp1 = f[keys[3]][:N, :]
    tmp2 = f[keys[4]][:N, :]
    tmp3 = f[keys[5]][:N, :]
    scatter_re = np.stack((tmp1, tmp2, tmp3), axis=-1)
    # Process the imaginary part from keys 0, 1, 2.
    tmp1 = f[keys[0]][:N, :]
    tmp2 = f[keys[1]][:N, :]
    tmp3 = f[keys[2]][:N, :]
    scatter_im = np.stack((tmp1, tmp2, tmp3), axis=-1)
    # Combine real and imaginary parts along a new axis.
    scatter = np.stack((scatter_re, scatter_im), axis=-2).astype("float32")

  if scatter_norm_constants is None:
    computed_norm_constants = []
    for ch in range(scatter.shape[-1]):
      mean_ch = np.mean(scatter[:, :, :, ch])
      std_ch = np.std(scatter[:, :, :, ch])
      computed_norm_constants.append((mean_ch, std_ch))
      scatter[:, :, :, ch] = (scatter[:, :, :, ch] - mean_ch) / std_ch
    return scatter, computed_norm_constants
  else:
    for ch in range(scatter.shape[-1]):
      mean_ch, std_ch = scatter_norm_constants[ch]
      scatter[:, :, :, ch] = (scatter[:, :, :, ch] - mean_ch) / std_ch
    return scatter

def create_dataset(eta: np.ndarray, scatter: np.ndarray, batch_size: int,
                   repeat: bool = True):
  """Creates a tensorflow dataset from eta and scatter data.

  The dictionary 'x' contains the eta data and 'cond' contains the scatter
  channels.

  Args:
    eta: Array containing eta data.
    scatter: Array containing scatter data with at least 4 dimensions.
    batch_size: The batch size for the dataset.
    repeat: If True, repeats the dataset indefinitely.

  Returns:
    Iterator: An iterator over the numpy batches from the dataset.
  """
  dict_data = {
    "x": eta,
    "cond": {
      "channel:scatter0": scatter[:, :, :, 0],
      "channel:scatter1": scatter[:, :, :, 1],
      "channel:scatter2": scatter[:, :, :, 2]
    }
  }
  dataset = tf.data.Dataset.from_tensor_slices(dict_data)
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset.as_numpy_iterator()
