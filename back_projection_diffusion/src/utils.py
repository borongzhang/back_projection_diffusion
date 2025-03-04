import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
from scipy.ndimage import geometric_transform
from scipy.ndimage import gaussian_filter
import h5py
import natsort

def rotationindex(n):
    """
    Creates an array of indices based on rotating rows and columns of a 2D array.

    Args:
        n (int): Dimension of the square grid (n x n).

    Returns:
        jnp.ndarray: A concatenated array of rotation indices.
    """
    index = jnp.reshape(jnp.arange(0, n**2, 1), [n, n])
    
    # Rotate the grid for each shift and concatenate the results
    return jnp.concatenate([jnp.roll(index, shift=[-i, -i], axis=[0, 1]) for i in range(n)], 0)
    
def SparsePolarToCartesian(neta, nx):
    """
    Converts a Cartesian grid to a polar coordinate system and creates a sparse representation of the transformation.

    Args:
        neta (int): The size of the Cartesian grid (neta x neta).
        nx (int): The size of the polar grid (nx x nx).

    Returns:
        sparse.BCOO: A sparse matrix in BCOO format representing the transformation 
                     from Cartesian to polar coordinates.
    """

    def shift_func(coords):
        """
        Transforms Cartesian coordinates to polar coordinates.

        Args:
            coords (tuple): A pair of coordinates (i, j) in the Cartesian grid.

        Returns:
            tuple: Polar coordinates (theta, rho) scaled to match the transformation.
        """
        i, j = coords[0], coords[1]

        # Calculate the radial distance `rho`.
        rho = 2 * np.sqrt((i - neta / 2) ** 2 + (j - neta / 2) ** 2) * nx / neta

        # Calculate the angular coordinate `theta`.
        # Adjust `theta` to fit within [0, 2π] using modulo.
        theta = ((np.arctan2((neta / 2 - j), (i - neta / 2))) % (2 * np.pi)) * nx / (2 * np.pi)

        # Return the scaled polar coordinates.
        return theta, rho + neta // 2

    cart_mat = np.zeros((neta**2, nx, nx))

    # Loop through each pixel in the polar grid (nx x nx).
    for i in range(nx):
        print(f"\rProcessing row {i + 1} / {nx}", end='', flush=True)  # Print progress to track computation for each row.

        for j in range(nx):
            # Create a dummy matrix with a single pixel set to 1 at position (i, j).
            mat_dummy = np.zeros((nx, nx))
            mat_dummy[i, j] = 1

            # Pad the matrix to prevent shifting out of bound.
            pad_dummy = np.pad(mat_dummy, ((0, 0), (neta // 2, neta // 2)), 'edge')

            # The given mapping function is used to find, for each point in the output, the corresponding coordinates in the input. 
            # The value of the input at those coordinates is determined by spline interpolation of the requested order.
            cart_mat[:, i, j] = geometric_transform(
                pad_dummy,
                shift_func,
                output_shape=[neta, neta],
                mode='grid-wrap'
            ).flatten()

    cart_mat = np.reshape(cart_mat, (neta**2, nx**2))

    # Threshold the matrix to remove small values (set values < 0.001 to 0).
    cart_mat = np.where(np.abs(cart_mat) > 0.001, cart_mat, 0)

    # Convert the dense matrix to a sparse BCOO format for efficient storage and computation.
    return sparse.BCOO.fromdense(cart_mat)

def SparseCartesianToPolar(neta, nx):
    """
    Converts a polar grid to a Cartesian grid and creates a sparse representation of the transformation.

    Args:
        neta (int): The size of the Cartesian grid (neta x neta).
        nx (int): The size of the polar grid (nx x nx).

    Returns:
        sparse.BCOO: A sparse matrix in BCOO format representing the transformation 
                     from polar to Cartesian coordinates.
    """

    def shift_func(coords):
        """
        Transforms polar coordinates to Cartesian coordinates.

        Args:
            coords (tuple): A pair of coordinates (theta_index, rho_index) in the polar grid.

        Returns:
            tuple: Cartesian coordinates (i, j) scaled to match the transformation.
        """
        # Calculate the angular coordinate (theta) from its scaled index.
        theta = 2 * np.pi * coords[0] / nx

        # Calculate the radial distance (rho) from its scaled index.
        rho = coords[1] / 2

        # Convert polar coordinates to Cartesian coordinates (i, j).
        i = nx / 2 + rho * np.cos(theta)
        j = nx / 2 - rho * np.sin(theta)

        # Return the Cartesian coordinates.
        return i, j

    polar_mat = np.zeros((nx**2, neta, neta))

    # Loop through each pixel in the Cartesian grid (neta x neta).
    for i in range(neta):
        for j in range(neta):
            # Create a dummy matrix with a single pixel set to 1 at position (i, j).
            mat_dummy = np.zeros((neta, neta))
            mat_dummy[i, j] = 1

            # The given mapping function is used to find, for each point in the output, the corresponding coordinates in the input. 
            # The value of the input at those coordinates is determined by spline interpolation of the requested order.
            polar_mat[:, i, j] = geometric_transform(
                mat_dummy,
                shift_func,
                output_shape=[nx, nx],
                order=3,   
                mode='nearest'   
            ).flatten()

    polar_mat = np.reshape(polar_mat, (nx**2, neta**2))

    # Threshold the matrix to remove small values (set values < 0.001 to 0).
    polar_mat = np.where(np.abs(polar_mat) > 0.001, polar_mat, 0)

    # Convert the dense matrix to a sparse BCOO format for efficient storage and computation.
    return sparse.BCOO.fromdense(polar_mat)

def compute_F_adj(freq):
    """
    Computes the analytical adjoint of the forward map for a given frequency for nx = neta = 80.

    Args:
        freq (float): The frequency for which to compute the adjoint.

    Returns:
        jnp.ndarray: The adjoint of the forward map as a complex matrix.
    """
    omega = 2 * jnp.pi * freq

    X, Y = jnp.meshgrid(jnp.linspace(0, 1, 80) - 0.5, jnp.linspace(0, 1, 80) - 0.5)
    pa, qa = jnp.meshgrid(jnp.linspace(jnp.pi, 3 * jnp.pi, 80), jnp.linspace(jnp.pi, 3 * jnp.pi, 80))

    p1 = jnp.cos(pa).ravel()
    p2 = jnp.sin(pa).ravel()
    q1 = jnp.cos(qa).ravel()
    q2 = jnp.sin(qa).ravel()

    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Complex exponential components
    F = jnp.exp(1j * jnp.pi / 4) / jnp.sqrt(8 * jnp.pi * omega) * (omega**2 * jnp.exp(1j * omega * 0.5) / jnp.sqrt(0.5)) * \
        jnp.exp(1j * omega * ((p1[:, None] - q1[:, None]) * X_flat + (p2[:, None] - q2[:, None]) * Y_flat)) / 79**2

    return F.conj().T

def load_eta_data(data_path, N, blur_sigma, normalize=False):
    """
    Load and process η (eta) data from an HDF5 file.

    Parameters:
        data_path (str): Path to the folder containing 'eta.h5'.
        N (int): Number of samples to load.
        blur_sigma (float): Sigma value for the Gaussian blur.
        normalize (bool): If True, normalize the data and return the normalization constants.

    Returns:
        If normalize is True:
            eta (np.ndarray): Normalized eta data.
            mean_eta (np.ndarray): Mean computed from the data.
            std_eta (float): Standard deviation computed from the data.
        If normalize is False:
            eta (np.ndarray): Processed (but not normalized) eta data.
    """
    # Open the file and load the raw η data
    with h5py.File(f'{data_path}/eta.h5', 'r') as f:
        raw_eta = f[list(f.keys())[0]][:N, :]
        # Determine the spatial dimension (assumes square images)
        neta = int(np.sqrt(raw_eta.shape[1]))
        # Reshape raw data into (N, neta, neta)
        eta = raw_eta.reshape(-1, neta, neta)
        # Apply Gaussian blur (transposing each image as in the original code)
        blur_fn = lambda x: gaussian_filter(x, sigma=blur_sigma)
        eta = np.stack([blur_fn(img.T) for img in eta]).astype('float32')
    
    if normalize:
        # Compute normalization constants
        mean_eta = np.mean(eta, axis=0)
        std_eta = np.std(eta)
        # Normalize the data
        eta = (eta - mean_eta) / std_eta
        return eta, mean_eta, std_eta
    else:
        return eta
        
     
def load_scatter_data(data_path, N, scatter_norm_constants=None):
    """
    Load and preprocess scatter data from an HDF5 file.

    The scatter data file is assumed to contain both real and imaginary parts.
    The real part is taken from keys[3], keys[4], keys[5] and the imaginary part
    from keys[0], keys[1], keys[2] (after natural sorting the keys).

    Parameters:
        data_path (str): Path to the folder containing the scatter HDF5 file (e.g., 'scatter.h5').
        N (int): Number of samples to load.
        scatter_norm_constants (list of tuples or None): 
            If None, compute per-channel normalization constants (mean and std), normalize the data,
            and return both the normalized data and the computed constants.
            If provided, it should be a list like [(mean0, std0), (mean1, std1), (mean2, std2)].
            The function will then normalize the scatter data using these constants.

    Returns:
        If scatter_norm_constants is None:
            scatter (np.ndarray): Normalized scatter data.
            computed_norm_constants (list): List of (mean, std) tuples for each scatter channel.
        Else:
            scatter (np.ndarray): Normalized scatter data.
    """
    # Open the HDF5 file and read the keys in natural sorted order.
    with h5py.File(f'{data_path}/scatter.h5', 'r') as f:
        keys = natsort.natsorted(f.keys())
        
        # Process the real part from keys[3], keys[4], keys[5]
        tmp1 = f[keys[3]][:N, :]
        tmp2 = f[keys[4]][:N, :]
        tmp3 = f[keys[5]][:N, :]
        scatter_re = np.stack((tmp1, tmp2, tmp3), axis=-1)
        
        # Process the imaginary part from keys[0], keys[1], keys[2]
        tmp1 = f[keys[0]][:N, :]
        tmp2 = f[keys[1]][:N, :]
        tmp3 = f[keys[2]][:N, :]
        scatter_im = np.stack((tmp1, tmp2, tmp3), axis=-1)
        
        # Combine real and imaginary parts along a new axis (second-to-last)
        scatter = np.stack((scatter_re, scatter_im), axis=-2).astype('float32')

    # If no normalization constants are provided, compute them from the data.
    if scatter_norm_constants is None:
        computed_norm_constants = []
        for ch in range(scatter.shape[-1]):
            mean_ch = np.mean(scatter[:, :, :, ch])
            std_ch = np.std(scatter[:, :, :, ch])
            computed_norm_constants.append((mean_ch, std_ch))
            scatter[:, :, :, ch] = (scatter[:, :, :, ch] - mean_ch) / std_ch
        return scatter, computed_norm_constants
    else:
        # Normalize using the provided normalization constants.
        for ch in range(scatter.shape[-1]):
            mean_ch, std_ch = scatter_norm_constants[ch]
            scatter[:, :, :, ch] = (scatter[:, :, :, ch] - mean_ch) / std_ch
        return scatter
       
def create_dataset(eta, scatter, batch_size, repeat=True):
    """
    Create a TensorFlow dataset from eta and scatter data.

    The function builds a dictionary where:
      - "x" corresponds to the eta data.
      - "cond" is another dictionary with keys:
          "channel:scatter0": scatter[..., 0],
          "channel:scatter1": scatter[..., 1],
          "channel:scatter2": scatter[..., 2].
    
    The dataset is then batched, prefetched, and optionally repeated.

    Parameters:
        eta (np.ndarray): Array containing eta data.
        scatter (np.ndarray): Array containing scatter data with at least 4 dimensions.
        batch_size (int): The batch size for the dataset.
        repeat (bool): If True, the dataset is repeated indefinitely.

    Returns:
        An iterator over the numpy batches from the dataset.
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

