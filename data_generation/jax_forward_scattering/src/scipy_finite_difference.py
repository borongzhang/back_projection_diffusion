from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.sparse import diags, eye, kron, spdiags


class FiniteDifferenceParams(NamedTuple):
    """Finite-difference grid and PML parameters."""

    sigma_max: np.float32
    x: np.ndarray
    y: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    nxi: int
    nyi: int
    nx: int
    ny: int
    npml: int
    h: np.float32
    xi: np.ndarray
    yi: np.ndarray
    xi_grid: np.ndarray
    yi_grid: np.ndarray


def init_params(lx, ly, nxi, nyi, npml, sigma_max):
    """Initialize the finite-difference grid and PML-extended grid."""
    hx = lx / (nxi - 1)
    hy = ly / (nyi - 1)
    if not np.isclose(float(hx), float(hy)):
        raise ValueError("This code assumes hx == hy.")

    h = hx
    nx = nxi + 2 * npml
    ny = nyi + 2 * npml

    xi = h * np.arange(nxi, dtype=np.float32) - lx / 2.0
    yi = h * np.arange(nyi, dtype=np.float32) - ly / 2.0

    x_left = xi[0] + h * np.arange(-npml, 0, dtype=np.float32)
    x_right = xi[-1] + h * np.arange(1, npml + 1, dtype=np.float32)
    x = np.concatenate([x_left, xi, x_right])

    y_left = yi[0] + h * np.arange(-npml, 0, dtype=np.float32)
    y_right = yi[-1] + h * np.arange(1, npml + 1, dtype=np.float32)
    y = np.concatenate([y_left, yi, y_right])

    xi_grid, yi_grid = np.meshgrid(xi, yi, indexing="xy")
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    return FiniteDifferenceParams(
        sigma_max=np.float32(sigma_max),
        x=x,
        y=y,
        x_grid=x_grid,
        y_grid=y_grid,
        nxi=int(nxi),
        nyi=int(nyi),
        nx=int(nx),
        ny=int(ny),
        npml=int(npml),
        h=np.float32(h),
        xi=xi,
        yi=yi,
        xi_grid=xi_grid,
        yi_grid=yi_grid,
    )


def finite_difference_weights(z, x, derivative_order):
    """Return finite-difference weights using Fornberg's recursion."""
    x = np.asarray(x, dtype=float)
    n = len(x) - 1
    dtype = complex if np.iscomplexobj(x) or np.iscomplexobj(z) else float
    weights = np.zeros((n + 1, derivative_order + 1), dtype=dtype)
    c1 = 1.0
    c4 = x[0] - z
    weights[0, 0] = 1.0

    for i in range(1, n + 1):
        max_order = min(i, derivative_order)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(max_order, 0, -1):
                    weights[i, k] = (
                        c1 * (k * weights[i - 1, k - 1] - c5 * weights[i - 1, k]) / c2
                    )
                weights[i, 0] = -c1 * c5 * weights[i - 1, 0] / c2
            for k in range(max_order, 0, -1):
                weights[j, k] = (c4 * weights[j, k] - k * weights[j, k - 1]) / c3
            weights[j, 0] = c4 * weights[j, 0] / c3

        c1 = c2

    return weights[:, -1]


def stiffness_matrix(num_points, spacing, order):
    """Build a one-dimensional second-derivative matrix."""
    half_width = order // 2
    stencil = np.arange(order + 1)
    base_weights = (
        np.asarray(finite_difference_weights(half_width, stencil, 2)).ravel()
        / spacing**2
    )
    data = np.tile(base_weights[:, None], (1, num_points))
    offsets = np.arange(-half_width, half_width + 1)
    second_derivative = spdiags(data, offsets, num_points, num_points).tolil()

    boundary_stencil = np.arange(order + 3)
    for i in range(1, half_width):
        left_weights = (
            np.asarray(finite_difference_weights(i, boundary_stencil, 2)).ravel()
            / spacing**2
        )
        second_derivative[i - 1, 0 : order + 2] = left_weights[1:]

        right_weights = (
            np.asarray(
                finite_difference_weights(order + 2 - i, boundary_stencil, 2)
            ).ravel()
            / spacing**2
        )
        row = num_points - i
        start = num_points - (order + 2)
        second_derivative[row, start:num_points] = right_weights[:-1]

    return second_derivative.tocsr()


def first_order_difference_matrix_1d(num_points, spacing, order):
    """Build a one-dimensional first-derivative matrix."""
    half_width = order // 2
    stencil = np.arange(order + 1)
    base_weights = (
        np.asarray(finite_difference_weights(half_width, stencil, 1)).ravel() / spacing
    )
    data = np.tile(base_weights[:, None], (1, num_points))
    offsets = np.arange(-half_width, half_width + 1)
    first_derivative = spdiags(data, offsets, num_points, num_points).tolil()

    boundary_stencil = np.arange(order + 3)
    for i in range(1, half_width):
        left_weights = (
            np.asarray(finite_difference_weights(i, boundary_stencil, 1)).ravel()
            / spacing
        )
        first_derivative[i - 1, 0 : order + 2] = left_weights[1:]

        right_weights = (
            np.asarray(
                finite_difference_weights(order + 2 - i, boundary_stencil, 1)
            ).ravel()
            / spacing
        )
        row = num_points - i
        start = num_points - (order + 2)
        first_derivative[row, start:num_points] = right_weights[:-1]

    return first_derivative.tocsr()


def pml_profile(nx, ny, npml, factor):
    """Return quadratic PML profiles and their derivatives."""
    t = np.linspace(0.0, 1.0, npml)
    sigma_x = np.zeros((ny, nx), dtype=np.float32)
    sigma_y = np.zeros((ny, nx), dtype=np.float32)
    sigma_x_prime = np.zeros((ny, nx), dtype=np.float32)
    sigma_y_prime = np.zeros((ny, nx), dtype=np.float32)

    sigma_x[:, :npml] = (factor * t[::-1] ** 2)[None, :]
    sigma_x[:, nx - npml :] = (factor * t**2)[None, :]
    sigma_y[:npml, :] = (factor * t[::-1] ** 2)[:, None]
    sigma_y[ny - npml :, :] = (factor * t**2)[:, None]

    sigma_x_prime[:, :npml] = (-2 * factor * t[::-1])[None, :]
    sigma_x_prime[:, nx - npml :] = (2 * factor * t)[None, :]
    sigma_y_prime[:npml, :] = (-2 * factor * t[::-1])[:, None]
    sigma_y_prime[ny - npml :, :] = (2 * factor * t)[:, None]

    return sigma_x, sigma_y, sigma_x_prime, sigma_y_prime


def _pml_fd_operators(nx, ny, npml, spacing, factor, order):
    """Build derivative operators and PML coefficient arrays."""
    sigma_x, sigma_y, sigma_x_prime, sigma_y_prime = pml_profile(nx, ny, npml, factor)

    dxx_1d = stiffness_matrix(nx, spacing, order)
    dyy_1d = stiffness_matrix(ny, spacing, order)
    dx_1d = first_order_difference_matrix_1d(nx, spacing, order)
    dy_1d = first_order_difference_matrix_1d(ny, spacing, order)

    identity_y = eye(ny, format="csr")
    identity_x = eye(nx, format="csr")
    dx = kron(dx_1d, identity_y, format="csr")
    dy = kron(identity_x, dy_1d, format="csr")
    dxx = kron(dxx_1d, identity_y, format="csr")
    dyy = kron(identity_x, dyy_1d, format="csr")

    sigma_x_flat = np.asarray(sigma_x).ravel(order="F")
    sigma_y_flat = np.asarray(sigma_y).ravel(order="F")
    sigma_x_prime_flat = np.asarray(sigma_x_prime).ravel(order="F")
    sigma_y_prime_flat = np.asarray(sigma_y_prime).ravel(order="F")

    return (
        dx,
        dy,
        dxx,
        dyy,
        sigma_x_flat,
        sigma_y_flat,
        sigma_x_prime_flat,
        sigma_y_prime_flat,
    )


def helmholtz_matrix_off_diagonal(nx, ny, npml, spacing, factor, order, omega):
    """Build the off-diagonal part of the PML Helmholtz matrix."""
    n = nx * ny
    (
        dx,
        dy,
        dxx,
        dyy,
        sigma_x,
        sigma_y,
        sigma_x_prime,
        sigma_y_prime,
    ) = _pml_fd_operators(nx, ny, npml, spacing, factor, order)

    x_pml_scale = 1.0 - 1j / omega * sigma_x
    y_pml_scale = 1.0 - 1j / omega * sigma_y
    derivative_scale = 1j / (omega * (npml - 1) * spacing)

    helmholtz_matrix = (
        diags(
            derivative_scale * sigma_x_prime / x_pml_scale**3,
            0,
            shape=(n, n),
            format="csr",
        )
        @ dx
        + diags(
            derivative_scale * sigma_y_prime / y_pml_scale**3,
            0,
            shape=(n, n),
            format="csr",
        )
        @ dy
        + diags(1.0 / x_pml_scale**2, 0, shape=(n, n), format="csr") @ dxx
        + diags(1.0 / y_pml_scale**2, 0, shape=(n, n), format="csr") @ dyy
    )

    return helmholtz_matrix.tocsr()


def extend_model(perturbation, nx_inner, ny_inner, npml):
    """Embed the perturbation inside the PML-extended computational grid."""
    perturbation = np.asarray(perturbation)
    if perturbation.ndim == 1:
        perturbation = perturbation.reshape((ny_inner, nx_inner), order="F")
    elif perturbation.ndim != 2:
        raise ValueError("extend_model expects a 1D or 2D input.")

    nx = perturbation.shape[1] + 2 * npml
    ny = perturbation.shape[0] + 2 * npml
    extended = np.zeros((ny, nx), dtype=perturbation.dtype)
    extended[npml : npml + ny_inner, npml : npml + nx_inner] = perturbation
    return extended.ravel(order="F")


def adjoint_extend_model(gradient_ext_flat, nx_inner, ny_inner, npml):
    """Restrict an extended-grid adjoint field to the physical domain."""
    gradient_ext = np.asarray(gradient_ext_flat).reshape(
        (ny_inner + 2 * npml, nx_inner + 2 * npml),
        order="F",
    )
    return gradient_ext[npml : npml + ny_inner, npml : npml + nx_inner].copy()
