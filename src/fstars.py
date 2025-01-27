import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import flax.linen as nn
from numpy import cos, sin, exp, newaxis

class analytical_fstar(nn.Module):
    F_adj: jnp.ndarray
    
    def __call__(self, inputs):
        """
        Args:
            inputs (jnp.ndarray): Input scattering data of shape [batch_size, 6400, 2],
                                  where the last dimension represents real and imaginary parts.

        Returns:
            jnp.ndarray: Outputs of shape [batch_size, 80, 80, 2] by the application of the equivariant adjoint forward operaor,
                         where the last dimension contains the real and imaginary parts of the result.
        """
        Lambda = jnp.reshape(inputs,[-1, 6400, 2])
        Lambda_Complex = Lambda[..., 0] + 1j * Lambda[..., 1]  
        alpha = jnp.einsum('ij,aj->ai', self.F_adj, Lambda_Complex)
        alpha_real = jnp.reshape(np.real(alpha),[-1, 80, 80, 1])
        alpha_imag = jnp.reshape(np.imag(alpha),[-1, 80, 80, 1])
        return jnp.concatenate([alpha_real, alpha_imag], axis = -1)
        
class equinet_fstar(nn.Module):
    nx: int
    neta: int
    cart_mat: jnp.ndarray
    r_index: np.ndarray 

    def setup(self):
        kernel_shape = (self.nx, self.nx)
        p_shape = (1, self.nx)
        
        self.pre1 = self.param('pre1', nn.initializers.uniform(), p_shape)
        self.pre2 = self.param('pre2', nn.initializers.uniform(), p_shape)
        self.pre3 = self.param('pre3', nn.initializers.uniform(), p_shape)
        self.pre4 = self.param('pre4', nn.initializers.uniform(), p_shape)

        self.post1 = self.param('post1', nn.initializers.uniform(), p_shape)
        self.post2 = self.param('post2', nn.initializers.uniform(), p_shape)
        self.post3 = self.param('post3', nn.initializers.uniform(), p_shape)
        self.post4 = self.param('post4', nn.initializers.uniform(), p_shape)
        
        self.cos_kernel1 = self.param('cos_kernel1', nn.initializers.uniform(), kernel_shape)
        self.sin_kernel1 = self.param('sin_kernel1', nn.initializers.uniform(), kernel_shape)
        self.cos_kernel2 = self.param('cos_kernel2', nn.initializers.uniform(), kernel_shape)
        self.sin_kernel2 = self.param('sin_kernel2', nn.initializers.uniform(), kernel_shape)
        self.cos_kernel3 = self.param('cos_kernel3', nn.initializers.uniform(), kernel_shape)
        self.sin_kernel3 = self.param('sin_kernel3', nn.initializers.uniform(), kernel_shape)
        self.cos_kernel4 = self.param('cos_kernel4', nn.initializers.uniform(), kernel_shape)
        self.sin_kernel4 = self.param('sin_kernel4', nn.initializers.uniform(), kernel_shape)

    def __call__(self, inputs):
        """
        Args:
            inputs (jnp.ndarray): Input scattering data of shape [batch_size, 6400, 2],
                                  where the last dimension represents real and imaginary parts.

        Returns:
            jnp.ndarray: Outputs of shape [batch_size, 80, 80, 1] by the application of the equivariantly appeoximated adjoint forward operaor.
        """
        # Separate real and imaginary parts of inputs
        R, I = inputs[:, :, 0], inputs[:, :, 1]
        
        rdata = lambda d: jnp.take(d, self.r_index)
        
        Rs = jax.vmap(rdata)(R)
        Rs = jnp.reshape(Rs, [-1, self.nx, self.nx])
        Is = jax.vmap(rdata)(I)
        Is = jnp.reshape(Is, [-1, self.nx, self.nx])
        
        def helper(pre, post, kernel2, kernel1, data):
            return jnp.matmul(post, jnp.multiply(kernel2, jnp.matmul(jnp.multiply(data, pre), kernel1)))  
        
        output_polar = helper(self.pre1, self.post1, self.cos_kernel1, self.cos_kernel2, Rs) \
                     + helper(self.pre2, self.post2, self.sin_kernel1, self.sin_kernel2, Rs) \
                     + helper(self.pre3, self.post3, self.cos_kernel3, self.sin_kernel3, Is) \
                     + helper(self.pre4, self.post4, self.sin_kernel4, self.cos_kernel4, Is)
        
        output_polar = jnp.reshape(output_polar, (-1, self.nx**2, 1))
        
        # Convert from polar to Cartesian coordinates
        def polar_to_cart(x):
            x = self.cart_mat @ x
            return jnp.reshape(x, (self.neta, self.neta, 1))
            
        return jax.vmap(polar_to_cart)(output_polar)

def build_permutation_indices(L, l):
    delta = 2**(L-l-1)
    tmp = np.tile(np.arange(2)*delta, delta)
    tmp += np.repeat(np.arange(delta), 2)
    tmp = np.tile(tmp, 2**l)
    tmp += np.repeat(np.arange(2**l)*(2**(L-l)), 2**(L-l))
    return jnp.asarray(tmp)

# Precomputing indices used for redistributing blocks according to the transformation represented by x -> M*xM.
def build_switch_indices(L):
    L = L // 2
    tmp = np.arange(2**L)*(2**L)
    tmp = np.tile(tmp, 2**L)
    tmp += np.repeat(np.arange(2**L), 2**L)
    return jnp.asarray(tmp)

class V(nn.Module):
    r: int

    @nn.compact
    def __call__(self, x):
        
        n, s = x.shape[1], x.shape[2]

        init_fn = nn.initializers.glorot_uniform()
        vr1 = self.param('vr1', init_fn, (n, s, self.r))
        vi1 = self.param('vi1', init_fn, (n, s, self.r))
        vr2 = self.param('vr2', init_fn, (n, s, self.r))
        vi2 = self.param('vi2', init_fn, (n, s, self.r))
        vr3 = self.param('vr3', init_fn, (n, s, self.r))
        vi3 = self.param('vi3', init_fn, (n, s, self.r))
        vr4 = self.param('vr4', init_fn, (n, s, self.r))
        vi4 = self.param('vi4', init_fn, (n, s, self.r))

        x_re, x_im = x[..., 0], x[..., 1]

        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, vr1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, vr1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, vi1)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, vi1)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, vi2)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, vr2)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, vr2)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, vi2)
        y_re = y_re_1+y_re_2+y_re_3+y_re_4
        
        y_im_1 = jnp.einsum('...iaj,ajk->...iak', x_im, vr3)
        y_im_1 = jnp.einsum('abj...i,bjk->abk...i', y_im_1, vr3)
        y_im_2 = jnp.einsum('...iaj,ajk->...iak', x_im, vi3)
        y_im_2 = jnp.einsum('abj...i,bjk->abk...i', y_im_2, vi3)
        y_im_3 = jnp.einsum('...iaj,ajk->...iak', x_re, vi4)
        y_im_3 = jnp.einsum('abj...i,bjk->abk...i', y_im_3, vr4)
        y_im_4 = jnp.einsum('...iaj,ajk->...iak', x_re, vr4)
        y_im_4 = jnp.einsum('abj...i,bjk->abk...i', y_im_4, vi4)
        y_im = y_im_1+y_im_2+y_im_3+y_im_4
        
        y = jnp.stack([y_re, y_im], axis=-1)
        
        return y

class H(nn.Module):
    perm_idx: jnp.ndarray
    
    @nn.compact
    def __call__(self, x):
        # Placeholder for actual input shape dependent variables
        m = x.shape[1] // 2
        s = x.shape[2] * 2

        # Define weights
        init_fn = nn.initializers.glorot_uniform()
        hr1 = self.param('hr1', init_fn, (m, s, s))
        hi1 = self.param('hi1', init_fn, (m, s, s))
        hr2 = self.param('hr2', init_fn, (m, s, s))
        hi2 = self.param('hi2', init_fn, (m, s, s))
        hr3 = self.param('hr3', init_fn, (m, s, s))
        hi3 = self.param('hi3', init_fn, (m, s, s))
        hr4 = self.param('hr4', init_fn, (m, s, s))
        hi4 = self.param('hi4', init_fn, (m, s, s))

        # Apply permutations
        x = x.take(self.perm_idx, axis=1).take(self.perm_idx, axis=3)
        
        # Reshape operation
        x = x.reshape((-1, m, s, m, s, 2))
        # Split real and imaginary parts for processing
        x_re, x_im = x[..., 0], x[..., 1]
        
        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, hr1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, hr1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, hi1)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, hi1)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, hi2)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, hr2)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, hr2)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, hi2)
        y_re = y_re_1+y_re_2+y_re_3+y_re_4
        
        y_im_1 = jnp.einsum('...iaj,ajk->...iak', x_im, hr3)
        y_im_1 = jnp.einsum('abj...i,bjk->abk...i', y_im_1, hr3)
        y_im_2 = jnp.einsum('...iaj,ajk->...iak', x_im, hi3)
        y_im_2 = jnp.einsum('abj...i,bjk->abk...i', y_im_2, hi3)
        y_im_3 = jnp.einsum('...iaj,ajk->...iak', x_re, hi4)
        y_im_3 = jnp.einsum('abj...i,bjk->abk...i', y_im_3, hr4)
        y_im_4 = jnp.einsum('...iaj,ajk->...iak', x_re, hr4)
        y_im_4 = jnp.einsum('abj...i,bjk->abk...i', y_im_4, hi4)
        y_im = y_im_1+y_im_2+y_im_3+y_im_4
        
        y = jnp.stack([y_re, y_im], axis=-1)

        n = m * 2
        r = s // 2
        y = y.reshape((-1, n, r, n, r, 2))

        return y

class M(nn.Module):
    @nn.compact
    def __call__(self, x):
        n, r = x.shape[1], x.shape[2]

        # Initialize weights
        init_fn = nn.initializers.glorot_uniform()
        mr1 = self.param('mr1', init_fn, (n, r, r))
        mi1 = self.param('mi1', init_fn, (n, r, r))
        mr2 = self.param('mr2', init_fn, (n, r, r))
        mi2 = self.param('mi2', init_fn, (n, r, r))
        mr3 = self.param('mr3', init_fn, (n, r, r))
        mi3 = self.param('mi3', init_fn, (n, r, r))
        mr4 = self.param('mr4', init_fn, (n, r, r))
        mi4 = self.param('mi4', init_fn, (n, r, r))

        x_re, x_im = x[..., 0], x[..., 1]

        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, mr1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, mr1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, mi1)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, mi1)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, mi2)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, mr2)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, mr2)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, mi2)
        y_re = y_re_1+y_re_2+y_re_3+y_re_4
        
        y_im_1 = jnp.einsum('...iaj,ajk->...iak', x_im, mr3)
        y_im_1 = jnp.einsum('abj...i,bjk->abk...i', y_im_1, mr3)
        y_im_2 = jnp.einsum('...iaj,ajk->...iak', x_im, mi3)
        y_im_2 = jnp.einsum('abj...i,bjk->abk...i', y_im_2, mi3)
        y_im_3 = jnp.einsum('...iaj,ajk->...iak', x_re, mi4)
        y_im_3 = jnp.einsum('abj...i,bjk->abk...i', y_im_3, mr4)
        y_im_4 = jnp.einsum('...iaj,ajk->...iak', x_re, mr4)
        y_im_4 = jnp.einsum('abj...i,bjk->abk...i', y_im_4, mi4)
        y_im = y_im_1+y_im_2+y_im_3+y_im_4
        
        y = jnp.stack([y_re, y_im], axis=-1)

        return y

class G(nn.Module):
    perm_idx: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        # Dimensions need to be dynamically inferred from 'x'
        m = x.shape[1] // 2
        s = x.shape[2] * 2

        # Initialize weights
        init_fn = nn.initializers.glorot_uniform()
        gr1 = self.param('gr1', init_fn, (m, s, s))
        gi1 = self.param('gi1', init_fn, (m, s, s))
        gr2 = self.param('gr2', init_fn, (m, s, s))
        gi2 = self.param('gi2', init_fn, (m, s, s))
        gr3 = self.param('gr3', init_fn, (m, s, s))
        gi3 = self.param('gi3', init_fn, (m, s, s))
        gr4 = self.param('gr4', init_fn, (m, s, s))
        gi4 = self.param('gi4', init_fn, (m, s, s))

        # Reshape and perform operations
        x = x.reshape((-1, m, s, m, s, 2))
        x_re, x_im = x[..., 0], x[..., 1]

        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, gr1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, gr1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, gi1)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, gi1)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, gi2)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, gr2)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, gr2)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, gi2)
        y_re = y_re_1+y_re_2+y_re_3+y_re_4
        
        y_im_1 = jnp.einsum('...iaj,ajk->...iak', x_im, gr3)
        y_im_1 = jnp.einsum('abj...i,bjk->abk...i', y_im_1, gr3)
        y_im_2 = jnp.einsum('...iaj,ajk->...iak', x_im, gi3)
        y_im_2 = jnp.einsum('abj...i,bjk->abk...i', y_im_2, gi3)
        y_im_3 = jnp.einsum('...iaj,ajk->...iak', x_re, gi4)
        y_im_3 = jnp.einsum('abj...i,bjk->abk...i', y_im_3, gr4)
        y_im_4 = jnp.einsum('...iaj,ajk->...iak', x_re, gr4)
        y_im_4 = jnp.einsum('abj...i,bjk->abk...i', y_im_4, gi4)
        y_im = y_im_1+y_im_2+y_im_3+y_im_4

        y = jnp.stack([y_re, y_im], axis=-1)

        # Final reshape and permutation
        n, r = m * 2, s // 2
        y = y.reshape((-1, n, r, n, r, 2))
        y = y.take(self.perm_idx, axis=1).take(self.perm_idx, axis=3)

        return y

class U(nn.Module):
    s: int  # Size parameter

    @nn.compact
    def __call__(self, x):
        # Extracting the shapes for weight initialization
        n, r = x.shape[1], x.shape[2]
        nx = n*self.s
        
        # Weight initialization
        init_fn = nn.initializers.glorot_uniform()
        ur1 = self.param('ur1', init_fn, (n, r, self.s))
        ui1 = self.param('ui1', init_fn, (n, r, self.s))
        ur2 = self.param('ur2', init_fn, (n, r, self.s))
        ui2 = self.param('ui2', init_fn, (n, r, self.s))
        ur3 = self.param('ur3', init_fn, (n, r, self.s))
        ui3 = self.param('ui3', init_fn, (n, r, self.s))
        ur4 = self.param('ur4', init_fn, (n, r, self.s))
        ui4 = self.param('ui4', init_fn, (n, r, self.s))

        # Splitting real and imaginary parts
        x_re, x_im = x[..., 0], x[..., 1]

        # Performing the einsum operations
        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, ur1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, ur1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, ui2)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, ui2)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, ui3)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, ur3)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, ur4)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, ui4)
        # Final sum of y_re components
        y_re = y_re_1 + y_re_2 + y_re_3 + y_re_4

        return y_re.reshape((-1, nx, nx))

class b_equinet_fstar(nn.Module):
    L: int
    s: int
    r: int
    NUM_RESNET: int
    cart_mat: jnp.ndarray
    r_index: jnp.ndarray
    
    def setup(self):
        self.n = 2**self.L
        self.nx = (2**self.L)*self.s
        self.neta = (2**self.L)*self.s
        self.V = V(self.r)
        self.Hs = [H(build_permutation_indices(self.L, l)) for l in range(self.L-1, self.L//2-1, -1)]
        self.Ms = [M() for _ in range(self.NUM_RESNET)]
        self.Gs = [G(build_permutation_indices(self.L, l)) for l in range(self.L//2, self.L)]
        self.U = U(self.s)
        self.switch_idx = build_switch_indices(self.L)

    def __call__(self, inputs):
        """
        Args:
            inputs (jnp.ndarray): Input scattering data of shape [batch_size, 6400, 2],
                                  where the last dimension represents real and imaginary parts.

        Returns:
            jnp.ndarray: Outputs of shape [batch_size, 80, 80, 1] by the application of the equivariantly appeoximated adjoint forward operaor.
                            Compressed by the butterfly factorizaiton.
        """
        y = inputs.take(self.r_index, axis=1)
        y = jnp.reshape(y, (-1, self.n, self.s, self.n, self.s, 2))
        
        y = self.V(y)
        for h in self.Hs:
            y = h(y)
        y = y.take(self.switch_idx, axis=1).take(self.switch_idx, axis=3)

        for m in self.Ms:
            y = m(y) if m is self.Ms[-1] else y + nn.relu(m(y))
            
        for g in self.Gs:
            y = g(y)
        y = self.U(y)
        
        y = jnp.diagonal(y, axis1 = 1, axis2 = 2)
        output_polar = jnp.reshape(y, (-1, self.nx**2, 1))
 
        def polar_to_cart(x):
            x = self.cart_mat @ x
            return jnp.reshape(x, (self.neta, self.neta, 1))
        
        return jax.vmap(polar_to_cart)(output_polar)


class DMLayer(nn.Module):
    output_dim: int  

    @nn.compact
    def __call__(self, x):
        # x shape expected: [batch_size, a, b]
        batch_size = x.shape[0]

        # Define kernel: [a, b, c]
        kernel_shape = (x.shape[-2], x.shape[-1], self.output_dim)
        kernel = self.param('kernel', nn.initializers.uniform(), kernel_shape)
        
        bias_shape = (1, x.shape[-2], self.output_dim)  # Broadcastable shape for bias
        bias = self.param('bias', nn.initializers.uniform(), bias_shape)

        b = jnp.einsum('ijk,jkl->ijl', x, kernel, optimize=True)
        b += bias

        return b
        
class switchnet_fstar(nn.Module):
    L1: int
    L2x: int
    L2y: int
    Nw1: int
    Nb1: int
    Nw2x: int
    Nw2y: int
    Nb2x: int
    Nb2y: int
    r: int

    @nn.compact
    def __call__(self, x):
        """
        Args:
            inputs (jnp.ndarray): Input scattering data of shape [batch_size, 6400, 2],
                                  where the last dimension represents real and imaginary parts.

        Returns:
            jnp.ndarray: Outputs of shape [batch_size, 80, 80, 1] by the application of the adjoint forward operaor.
                            Approximated by the SwitchNet.
        """
        batch_size = x.shape[0]

        # First set of operations (Reshape, Permute, DMLayer)
        x = x.reshape((batch_size, self.Nb1, self.Nw1, self.Nb1, self.Nw1, 2))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((batch_size, self.Nb1**2, 2*self.Nw1**2))
        x = DMLayer(self.Nb2x*self.Nb2y*self.r)(x)
        x = x.reshape((batch_size, self.Nb1*self.Nb1, self.Nb2x*self.Nb2y, self.r))
        x = x.transpose((0, 3, 1, 2))

        # Second set of operations (Reshape, DMLayer)
        x = x.reshape((batch_size, self.Nb2x*self.Nb2y*self.Nb1**2, self.r))
        x = DMLayer(self.r)(x)
        x = x.reshape((batch_size, self.Nb2x*self.Nb2y, self.Nb1**2*self.r))

        # Third set of operations (DMLayer, Reshape, Permute)
        x = DMLayer(2*self.Nw2x*self.Nw2y)(x)
        x = x.reshape((batch_size, self.Nb2x, self.Nb2y, self.Nw2x, self.Nw2y, 2))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((batch_size, self.Nb2x*self.Nw2x, self.Nb2y*self.Nw2y, 2))
        
        return x

    