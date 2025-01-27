import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
from scipy.ndimage import geometric_transform

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


