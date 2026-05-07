from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from scipy import special
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .base import BaseOperator
from .scipy_finite_difference import (
    adjoint_extend_model,
    extend_model,
    helmholtz_matrix_off_diagonal,
    init_params,
)


def _to_backend_array(array, dtype=None):
    """Move JAX-compatible inputs to NumPy arrays for the SciPy backend."""
    return np.asarray(array, dtype=dtype)


def _to_jax_array(array):
    """Return JAX arrays from the public operator API."""
    return jnp.asarray(array)


def _flatten_fortran(matrix):
    return _to_backend_array(matrix).ravel(order="F")


def point_source_green_function(radius, omega):
    """Return the two-dimensional Helmholtz Green's function."""
    return (1j / 4) * special.hankel1(0, omega * radius)


def incident_plane_wave(x_grid, y_grid, directions, omega):
    """Evaluate incident plane waves on the computational grid."""
    x_flat = _flatten_fortran(x_grid).reshape((-1, 1))
    y_flat = _flatten_fortran(y_grid).reshape((-1, 1))
    phase = x_flat * directions[:, 0] + y_flat * directions[:, 1]
    return np.exp(1j * omega * phase).astype(np.complex64)


class Scattering(BaseOperator):
    """JAX-facing forward map for the 2D time-harmonic Helmholtz problem.

    Public methods accept and return JAX arrays for Plug-and-Play / DPS
    sampling loops. The current backend uses NumPy/SciPy sparse solves, so
    these methods are not JIT- or VMAP-compatible yet; use the explicit
    Jacobian and adjoint-Jacobian applications instead of autodiff through
    the PDE solve.
    """

    def __init__(
        self,
        lx: float = 1.0,
        ly: float = 1.0,
        nx: int = 80,
        ny: int = 80,
        frequencies: Sequence[float] | None = None,
        num_directions: int = 80,
        receiver_radius: float = 1.0,
        order: int = 2,
        npml: int = 20,
        sigma_max: int = 80,
        sigma_noise: float = 0.0,
        unnorm_shift: float = 1.0,
        unnorm_scale: float = 0.5,
        device: str = "cuda",
        **legacy_kwargs,
    ) -> None:
        lx = legacy_kwargs.pop("Lx", lx)
        ly = legacy_kwargs.pop("Ly", ly)
        nx = legacy_kwargs.pop("Nx", nx)
        ny = legacy_kwargs.pop("Ny", ny)
        frequencies = legacy_kwargs.pop("freqs", frequencies)
        num_directions = legacy_kwargs.pop("numTrans", num_directions)
        receiver_radius = legacy_kwargs.pop("sensorRadius", receiver_radius)
        sigma_max = legacy_kwargs.pop("sigmaMax", sigma_max)
        if legacy_kwargs:
            names = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {names}.")

        super().__init__(sigma_noise, unnorm_shift, unnorm_scale, device)
        self.backend = "scipy"
        self.lx = float(lx)
        self.ly = float(ly)
        self.nx = int(nx)
        self.ny = int(ny)
        self.hx = self.lx / (self.nx - 1)
        self.hy = self.ly / (self.ny - 1)
        self.num_directions = int(num_directions)
        self.receiver_radius = float(receiver_radius)
        self.order = int(order)
        self.npml = int(npml)
        self.sigma_max = int(sigma_max)

        if frequencies is None:
            frequencies = (10,)
        self.frequencies = list(frequencies)
        self.angular_frequencies = [
            2.0 * np.pi * frequency for frequency in self.frequencies
        ]
        self.num_freqs = len(self.frequencies)

        self.finite_difference_params = []
        self.incident_fields = []
        self.helmholtz_offdiagonal_matrices = []
        self.projection_matrices = []

        for omega in self.angular_frequencies:
            params = init_params(
                self.lx,
                self.ly,
                self.nx,
                self.ny,
                self.npml,
                self.sigma_max,
            )
            self.finite_difference_params.append(params)

            d_theta = 2.0 * np.pi / self.num_directions
            theta = np.linspace(
                np.pi,
                3.0 * np.pi - d_theta,
                self.num_directions,
            )
            directions = np.column_stack((np.cos(theta), np.sin(theta))).astype(
                np.float32
            )

            incident_field = incident_plane_wave(
                params.x_grid,
                params.y_grid,
                directions,
                omega,
            )
            self.incident_fields.append(incident_field)

            helmholtz_offdiag = helmholtz_matrix_off_diagonal(
                int(params.nx),
                int(params.ny),
                int(params.npml),
                float(params.h),
                float(params.sigma_max),
                self.order,
                omega,
            )
            self.helmholtz_offdiagonal_matrices.append(csr_matrix(helmholtz_offdiag))
            self.projection_matrices.append(
                self._build_projection_matrix(
                    params,
                    self.num_directions,
                    self.receiver_radius,
                )
            )

    @property
    def Nx(self):
        """Compatibility alias for older notebooks; use ``nx`` in new code."""
        return self.nx

    @property
    def Ny(self):
        """Compatibility alias for older notebooks; use ``ny`` in new code."""
        return self.ny

    @property
    def numTrans(self):
        """Compatibility alias; use ``num_directions`` in new code."""
        return self.num_directions

    @staticmethod
    def _resolve_frequency_index(frequency_index, legacy_kwargs):
        if "i" in legacy_kwargs:
            if frequency_index is not None:
                raise TypeError("Use only one of 'frequency_index' or 'i'.")
            frequency_index = legacy_kwargs.pop("i")
        if legacy_kwargs:
            names = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {names}.")
        if frequency_index is None:
            raise TypeError("frequency_index is required.")
        return frequency_index

    def _build_projection_matrix(self, params, num_receivers, radius):
        """Build interpolation from the computational grid to receivers."""
        x = params.x
        y = params.y
        nx_grid = x.size
        ny_grid = y.size

        dtheta = 2.0 * np.pi / num_receivers
        receiver_angles = dtheta * np.arange(num_receivers)
        receiver_directions = np.column_stack(
            (np.cos(receiver_angles), np.sin(receiver_angles))
        ).astype(np.float32)
        receiver_locations = (radius * receiver_directions).astype(np.float32)

        projection_tensor = np.zeros(
            (num_receivers, nx_grid, ny_grid),
            dtype=np.float32,
        )
        query_points = np.column_stack(
            [receiver_locations[:, 1], receiver_locations[:, 0]]
        )

        for ix in range(nx_grid):
            for iy in range(ny_grid):
                basis = np.zeros((nx_grid, ny_grid), dtype=np.float32)
                basis[ix, iy] = 1.0
                interpolator = RegularGridInterpolator(
                    (y, x),
                    basis.T,
                    bounds_error=False,
                    fill_value=0.0,
                )
                projection_tensor[:, ix, iy] = interpolator(query_points)

        return csr_matrix(projection_tensor.reshape((num_receivers, nx_grid * ny_grid)))

    def _prepare_batch(self, perturbation):
        perturbation = _to_backend_array(perturbation)
        if perturbation.ndim == 4 and perturbation.shape[-1] == 1:
            return perturbation[..., 0]
        if perturbation.ndim == 3:
            return perturbation
        if perturbation.ndim == 2 and perturbation.shape[1] == self.nx * self.ny:
            return perturbation.reshape(
                (perturbation.shape[0], self.ny, self.nx),
                order="F",
            )
        raise ValueError(
            "Expected perturbation with shape (B, Ny, Nx, 1), "
            f"(B, Ny, Nx), or (B, Nx*Ny). Got {perturbation.shape}."
        )

    def _assemble_helmholtz_matrix(self, perturbation_ext, frequency_index):
        omega = self.angular_frequencies[frequency_index]
        refractive_index = (1.0 + perturbation_ext).astype(np.float32)
        return (
            self.helmholtz_offdiagonal_matrices[frequency_index]
            + omega**2 * diags(refractive_index, 0, format="csr")
        ).tocsr()

    def _forward_state(self, perturbation_sample, frequency_index):
        perturbation_ext = _to_backend_array(
            extend_model(perturbation_sample, self.nx, self.ny, self.npml),
            dtype=np.float32,
        )
        scattered_field, helmholtz_matrix = self._solve_forward(
            perturbation_ext,
            frequency_index,
            return_matrix=True,
        )
        total_field = self.incident_fields[frequency_index] + scattered_field
        return perturbation_ext, scattered_field, helmholtz_matrix, total_field

    def _solve_forward(self, perturbation_ext, frequency_index, return_matrix=False):
        perturbation_ext = _to_backend_array(
            perturbation_ext, dtype=np.float32
        ).reshape((-1,))
        omega = self.angular_frequencies[frequency_index]
        incident_field = self.incident_fields[frequency_index]

        helmholtz_matrix = self._assemble_helmholtz_matrix(
            perturbation_ext,
            frequency_index,
        )
        rhs = (
            (-(omega**2)) * perturbation_ext.reshape((-1, 1)) * incident_field
        ).astype(np.complex64)
        scattered_field = _to_backend_array(spsolve(helmholtz_matrix, rhs))

        if return_matrix:
            return scattered_field, helmholtz_matrix
        return scattered_field

    def _jacobian_action_single(
        self,
        perturbation_sample,
        delta_perturbation_sample,
        frequency_index=None,
        **legacy_kwargs,
    ):
        frequency_index = self._resolve_frequency_index(
            frequency_index,
            legacy_kwargs,
        )
        _, _, helmholtz_matrix, total_field = self._forward_state(
            perturbation_sample,
            frequency_index,
        )
        delta_perturbation_ext = _to_backend_array(
            extend_model(
                delta_perturbation_sample,
                self.nx,
                self.ny,
                self.npml,
            ),
            dtype=np.float32,
        )

        omega = self.angular_frequencies[frequency_index]
        rhs_linearized = (
            (-(omega**2)) * delta_perturbation_ext.reshape((-1, 1)) * total_field
        ).astype(np.complex64)
        delta_scattered_field = _to_backend_array(
            spsolve(helmholtz_matrix, rhs_linearized)
        )

        return self.projection_matrices[frequency_index] @ delta_scattered_field

    def _adjoint_jacobian_action_single(
        self,
        perturbation_sample,
        data_residual_sample,
        frequency_index=None,
        **legacy_kwargs,
    ):
        frequency_index = self._resolve_frequency_index(
            frequency_index,
            legacy_kwargs,
        )
        _, _, helmholtz_matrix, total_field = self._forward_state(
            perturbation_sample,
            frequency_index,
        )
        omega = self.angular_frequencies[frequency_index]

        rhs_adjoint = self.projection_matrices[frequency_index].T @ _to_backend_array(
            data_residual_sample
        )
        adjoint_field = _to_backend_array(
            spsolve(helmholtz_matrix.T.conjugate(), rhs_adjoint)
        )

        gradient_ext = (-(omega**2)) * np.real(
            np.sum(np.conj(total_field) * adjoint_field, axis=1)
        )
        gradient = adjoint_extend_model(
            gradient_ext,
            self.nx,
            self.ny,
            self.npml,
        )
        return gradient.astype(np.float32)

    def _forward_single(self, perturbation_sample, frequency_index):
        perturbation_ext = extend_model(
            perturbation_sample,
            self.nx,
            self.ny,
            self.npml,
        )
        scattered_field = self._solve_forward(perturbation_ext, frequency_index)
        scattering_data = self.projection_matrices[frequency_index] @ _to_backend_array(
            scattered_field
        )
        return scattering_data

    def forward(self, perturbation, unnormalize: bool = True):
        """Return scattering data for each perturbation and frequency."""
        perturbation = self._prepare_batch(perturbation)
        if unnormalize:
            perturbation = self.unnormalize(perturbation)

        batch_size = perturbation.shape[0]
        all_frequencies = []
        for frequency_index in range(self.num_freqs):
            frequency_outputs = []
            for batch_index in range(batch_size):
                frequency_outputs.append(
                    self._forward_single(
                        perturbation[batch_index],
                        frequency_index,
                    )
                )
            all_frequencies.append(np.stack(frequency_outputs, axis=0))

        return _to_jax_array(np.stack(all_frequencies, axis=-1))

    def jacobian_action(self, perturbation, delta_perturbation):
        """Apply the Jacobian of the forward map to a perturbation update."""
        perturbation = self._prepare_batch(perturbation)
        delta_perturbation = self._prepare_batch(delta_perturbation)

        batch_size = perturbation.shape[0]
        all_frequencies = []
        for frequency_index in range(self.num_freqs):
            frequency_outputs = []
            for batch_index in range(batch_size):
                frequency_outputs.append(
                    self._jacobian_action_single(
                        perturbation[batch_index],
                        delta_perturbation[batch_index],
                        frequency_index,
                    )
                )
            all_frequencies.append(np.stack(frequency_outputs, axis=0))

        return _to_jax_array(np.stack(all_frequencies, axis=-1))

    def adjoint_jacobian_action(self, perturbation, data_residual):
        """Apply the adjoint Jacobian of the forward map to data residuals."""
        perturbation = self._prepare_batch(perturbation)
        data_residual = _to_backend_array(data_residual)

        if data_residual.ndim != 4:
            raise ValueError(
                "Expected data_residual with shape "
                f"(B, Ntheta, Ntheta, num_freqs). Got {data_residual.shape}."
            )

        batch_size = perturbation.shape[0]
        gradients = np.zeros((batch_size, self.ny, self.nx, 1), dtype=np.float32)

        for batch_index in range(batch_size):
            gradient_batch = np.zeros((self.ny, self.nx), dtype=np.float32)
            for frequency_index in range(self.num_freqs):
                gradient_batch += self._adjoint_jacobian_action_single(
                    perturbation[batch_index],
                    data_residual[batch_index, :, :, frequency_index],
                    frequency_index,
                )
            gradients[batch_index, ..., 0] = gradient_batch

        return _to_jax_array(gradients)

    def gradient(
        self,
        pred,
        observation,
        return_loss: bool = False,
        unnormalize: bool = True,
    ):
        """Return the data-misfit gradient via the adjoint-state method."""
        perturbation = self._prepare_batch(pred)

        if unnormalize:
            perturbation = self.unnormalize(perturbation)

        predicted_data = _to_backend_array(
            self.forward(perturbation, unnormalize=False)
        )
        residual = predicted_data - _to_backend_array(observation)

        gradients = self.adjoint_jacobian_action(perturbation, residual)
        loss = 0.5 * np.linalg.norm(residual) ** 2

        if unnormalize:
            gradients = gradients * self.unnorm_scale

        if return_loss:
            return gradients, _to_jax_array(loss)
        return gradients
