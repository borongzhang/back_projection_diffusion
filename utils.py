import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
from scipy.ndimage import geometric_transform

def compute_F_adj(freq):
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
    
def data_transform_index(n):
    index = np.reshape(np.arange(0, n**2, 1), [n, n])
    return np.stack([np.roll(index[i,:], shift=-i, axis=0) for i in range(n)], 0).flatten()
    
def rotationindex(n):
    index = jnp.reshape(jnp.arange(0, n**2, 1), [n, n])
    return jnp.concatenate([jnp.roll(index, shift=[-i,-i], axis=[0,1]) for i in range(n)], 0)
    
    
def SparsePolarToCartesian(neta, nx):
    
    def CartesianToPolar(coords):
        i, j = coords[0], coords[1]
        # Calculate the radial distance with a scaling factor.
        rho = 2 * np.sqrt((i - neta / 2) ** 2 + (j - neta / 2) ** 2) * nx / neta
        # Calculate the angle in radians and adjust the scale to fit the specified range.
        theta = ((np.arctan2((neta / 2 - j), (i - neta / 2))) % (2 * np.pi)) * nx / np.pi / 2
        return theta, rho + neta // 2

    cart_mat = np.zeros((neta**2, nx, nx))

    for i in range(nx):
        print(i, end = ' ')
        for j in range(nx):
            mat_dummy = np.zeros((nx, nx))
            mat_dummy[i, j] = 1
            pad_dummy = np.pad(mat_dummy, ((0, 0), (neta // 2, neta // 2)), 'edge')
            cart_mat[:, i, j] = geometric_transform(pad_dummy, CartesianToPolar, output_shape=[neta, neta], mode='grid-wrap').flatten()
    
    cart_mat = np.reshape(cart_mat, (neta**2, nx**2))
    cart_mat = np.where(np.abs(cart_mat) > 0.001, cart_mat, 0)
    
    return sparse.BCOO.fromdense(cart_mat)
    
def SparseCatesianToPolar(neta, nx):
    
    def PolarToCartesian(coords):
        theta = 2*np.pi*coords[0] / nx 
        rho = coords[1] / 2
        i = nx/2 + rho*np.cos(theta)
        j = nx/2 - rho*np.sin(theta)
        return i, j
    

    polar_mat = np.zeros((nx**2, neta, neta))

    for i in range(neta):
        for j in range(neta):
            mat_dummy = np.zeros((neta, neta))
            mat_dummy[i, j] = 1
            polar_mat[:, i, j] = geometric_transform(mat_dummy, PolarToCartesian, output_shape=[nx, nx], order=3, mode='nearest').flatten()
    
    polar_mat = np.reshape(polar_mat, (nx**2, neta**2))
    polar_mat = np.where(np.abs(polar_mat) > 0.001, polar_mat, 0)
    
    return sparse.BCOO.fromdense(polar_mat)
    