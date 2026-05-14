"""
Axion Field Simulation
======================

Numba-accelerated lattice simulation of axion field dynamics during a
first-order cosmological phase transition (PT).

Physics
-------
Solves the classical axion equation of motion on a 3D periodic lattice
in a radiation-dominated FRW background:

    θ'' + 2 H_conf θ' - ∇²θ + a² m²(τ,x) sin θ = 0

where primes denote conformal-time derivatives, H_conf = 1/τ is the
conformal Hubble rate, and m²(τ,x) is the spatially varying mass field
driven by bubble nucleation.

Cosmology convention
--------------------
The phase transition occurs at cosmic time t_PT = 1/(2 H_PT), at which
point the scale factor is normalised to a(t_PT) = 1.  Conformal time
runs from τ = 0 at t = 0, with τ_PT = 1/H_PT.

Nucleation
----------
Bubble nucleation follows an exponential rate Γ(t) = Γ_0 exp(β(t - t_PT))
with Γ_0 = H_PT^4.  Nucleation events are sampled stochastically via
Monte Carlo tracer particles distributed uniformly in the comoving volume.
Each nucleated bubble expands at constant wall speed v_w in comoving
coordinates.

Time integration
----------------
Leapfrog (Störmer-Verlet) with an adaptive CFL timestep.  The mass field
between bubble snapshots is interpolated using a smooth cubic Hermite
kernel to avoid discontinuous accelerations.

Threading
---------
BLAS/MKL/OpenBLAS are forced to single-threaded mode to prevent lock
contention; Numba uses all available cores for the parallelised kernels.
Adjust `numba.set_num_threads` below to match your hardware.

Usage
-----
    from axion_sim import create_simulation_pt_at_zero

    sim, tau_final = create_simulation_pt_at_zero(
        H_PT=0.7, beta=5.6, start_time=0.0, end_time=10.0/0.7,
        Ngrid=64, num_tracers=100_000, enable_nucleation=True,
        theta0_initial=1.0, v_bubble=0.6,
    )
    results = sim.run_simulation(tau_final)
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numba as nb
import pyfftw

# ---------------------------------------------------------------------------
# Threading: suppress BLAS/MKL multi-threading; let Numba own the cores.
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Set to the number of physical cores on your machine.
numba.set_num_threads(8)

pyfftw.config.NUM_THREADS = 1
np.seterr(all='ignore')

try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    pass

FLOAT_TYPE = np.float32
COMPLEX_TYPE = np.complex64


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class SimulationConfig:
    """All parameters for a single axion field simulation run."""
    # Lattice
    Ngrid: int = 16
    boxSize_comoving: float = 8.0 * np.pi

    # Physics
    m0: float = 1.0
    v_bubble: float = 0.3
    delta_w: float = 0.15
    theta0_initial: float = 2.0

    # Cosmology
    a0: float = 1.0
    tau0: float = 1.0
    Gamma_0: float = 1e-6
    beta: float = 0.1
    t_0: float = 0.0

    # Nucleation control
    enable_nucleation: bool = True
    mb: float = 0.0          # background (false-vacuum) mass; 0 = no-mass outside bubbles

    # Numerical
    cfl_factor: float = 0.5
    friction_factor: float = 0.1
    num_tracers: int = 100_000

    # Performance
    spatial_hash_cells: int = 8
    energy_save_interval: int = 2
    checkpoint_interval: int = 0
    use_fft_laplacian: bool = False  # FFT Laplacian is faster for large grids

    def __post_init__(self):
        if self.Ngrid <= 0 or (self.Ngrid & (self.Ngrid - 1)) != 0:
            self.Ngrid = 2 ** int(np.ceil(np.log2(self.Ngrid)))
            warnings.warn(f"Ngrid rounded up to {self.Ngrid} (nearest power of two)")


# ===========================================================================
# Numba kernels
# ===========================================================================

@nb.njit(cache=True)
def create_true_vacuum(shape, m0_sq, dtype):
    """Return a uniform mass field equal to m0² (fully converted volume)."""
    return np.full(shape, m0_sq, dtype=dtype)


@nb.njit(parallel=True, fastmath=True, cache=True)
def calculate_mass_field_snapshot_subsampled(X, Y, Z, bubble_centers,
                                              bubble_radii_comoving,
                                              m0_sq, mb_sq, box_size):
    """
    Compute the spatially varying mass-squared field m²(x) at a given instant.

    Each voxel is subsampled at 8 points (2×2×2) to smooth the bubble wall
    across the lattice spacing.  Points inside any bubble get m0², points
    outside get mb².  The voxel value is the volume-weighted average.

    Parameters
    ----------
    X, Y, Z : (Nx, Ny, Nz) float32 arrays
        Comoving coordinates of lattice sites.
    bubble_centers : (N_b, 3) float32
        Comoving centres of all nucleated bubbles.
    bubble_radii_comoving : (N_b,) float32
        Comoving radii at the current conformal time.
    m0_sq, mb_sq : float
        True-vacuum and false-vacuum mass-squared values.
    box_size : float
        Comoving box size (periodic boundary condition wrapping).
    """
    Nx, Ny, Nz = X.shape
    mass_squared = np.full((Nx, Ny, Nz), mb_sq, dtype=X.dtype)
    dx = box_size / Nx
    num_bubbles = len(bubble_centers)

    if num_bubbles == 0:
        return mass_squared

    sub_offsets = np.array([-0.25, 0.25], dtype=X.dtype) * dx
    offsets = np.array([[sx, sy, sz]
                        for sx in sub_offsets
                        for sy in sub_offsets
                        for sz in sub_offsets], dtype=X.dtype)

    N_total = Nx * Ny * Nz
    for idx in nb.prange(N_total):
        i = idx // (Ny * Nz)
        j = (idx // Nz) % Ny
        k = idx % Nz

        x0, y0, z0 = X[i, j, k], Y[i, j, k], Z[i, j, k]
        points_inside = 0

        for off in offsets:
            px, py, pz = x0 + off[0], y0 + off[1], z0 + off[2]
            for b in range(num_bubbles):
                center = bubble_centers[b]
                radius_sq = bubble_radii_comoving[b] ** 2

                dx_ = px - center[0]
                dy_ = py - center[1]
                dz_ = pz - center[2]
                dx_ -= box_size * round(dx_ / box_size)
                dy_ -= box_size * round(dy_ / box_size)
                dz_ -= box_size * round(dz_ / box_size)

                if dx_**2 + dy_**2 + dz_**2 <= radius_sq:
                    points_inside += 1
                    break

        if points_inside > 0:
            frac = points_inside / 8.0
            mass_squared[i, j, k] = frac * m0_sq + (1.0 - frac) * mb_sq

    return mass_squared


@nb.njit(parallel=False, fastmath=True, cache=True)
def get_bubble_mask_snapshot(X, Y, Z, bubble_centers, bubble_radii_comoving,
                              box_size):
    """Boolean mask: True where a lattice site is inside any bubble."""
    Nx, Ny, Nz = X.shape
    mask = np.zeros((Nx, Ny, Nz), dtype=nb.boolean)
    num_bubbles = len(bubble_centers)

    if num_bubbles == 0:
        return mask

    for i in nb.prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for b in range(num_bubbles):
                    center = bubble_centers[b]
                    radius = bubble_radii_comoving[b]

                    dx = X[i, j, k] - center[0]
                    dy = Y[i, j, k] - center[1]
                    dz = Z[i, j, k] - center[2]
                    dx -= box_size * round(dx / box_size)
                    dy -= box_size * round(dy / box_size)
                    dz -= box_size * round(dz / box_size)

                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        mask[i, j, k] = True
                        break
    return mask


@nb.njit(parallel=False, fastmath=True, cache=True)
def evolve_bubbles_vectorized(radii_comoving, velocities, birth_times,
                               count, current_tau):
    """Update comoving bubble radii: R(τ) = v_w × (τ - τ_nucleation)."""
    for i in nb.prange(count):
        radii_comoving[i] = velocities[i] * (current_tau - birth_times[i])


@nb.njit(fastmath=True, cache=True)
def assign_to_hash_cell(pos, box_size, num_cells):
    normalized = (pos + box_size / 2) % box_size
    cell_idx = int(normalized / box_size * num_cells)
    return min(max(cell_idx, 0), num_cells - 1)


@nb.njit(parallel=True, fastmath=True, cache=True)
def find_surviving_tracers_simple(all_tracers, active_indices,
                                   bubble_centers, bubble_radii_comoving,
                                   box_size, num_hash_cells):
    """
    Return the subset of active tracers not yet engulfed by any bubble.

    Parameters
    ----------
    all_tracers : (N_total, 3) float32
        All tracer positions in comoving coordinates.
    active_indices : (N_active,) int64
        Indices into all_tracers that are still in the false vacuum.
    bubble_centers : (N_b, 3) float32
    bubble_radii_comoving : (N_b,) float32
    box_size : float
    num_hash_cells : int  (unused; kept for API compatibility)
    """
    num_active = len(active_indices)
    num_bubbles = len(bubble_centers)

    if num_bubbles == 0 or num_active == 0:
        return active_indices

    survived_mask = np.ones(num_active, dtype=nb.boolean)

    for idx in nb.prange(num_active):
        pos = all_tracers[active_indices[idx]]

        for b in range(num_bubbles):
            center = bubble_centers[b]
            radius_sq = bubble_radii_comoving[b] ** 2

            dx = pos[0] - center[0]
            dy = pos[1] - center[1]
            dz = pos[2] - center[2]
            dx -= box_size * round(dx / box_size)
            dy -= box_size * round(dy / box_size)
            dz -= box_size * round(dz / box_size)

            if dx*dx + dy*dy + dz*dz <= radius_sq:
                survived_mask[idx] = False
                break

    return active_indices[survived_mask]


# ===========================================================================
# Cosmology
# ===========================================================================

class CosmologyManagerPTAtZeroStartAtZero:
    """
    Radiation-dominated FRW cosmology with PT at τ_PT = 1/H_PT.

    Convention: t = 0, τ = 0 at the start of the simulation;
    a(τ_PT) = 1 so that a(τ) = H_PT τ everywhere.
    """

    def __init__(self, H_PT: float):
        self.H_PT = H_PT
        self.t_PT = 1.0 / (2.0 * H_PT)
        self.a_PT = 1.0
        self.tau_PT = 1.0 / H_PT

    def scale_factor(self, tau: float) -> float:
        return self.H_PT * tau

    def cosmic_time_from_conformal(self, tau: float) -> float:
        return 0.5 * self.H_PT * tau**2

    def conformal_time_from_cosmic(self, t_cosmic: float) -> float:
        return np.sqrt(2 * max(t_cosmic, 0.0) / self.H_PT)


# ===========================================================================
# FFT backend
# ===========================================================================

class FFTWBackend:
    """Thin wrapper around pyFFTW with single-threaded, pre-planned transforms."""

    def __init__(self):
        self.shape = None
        self.input_array = None
        self.output_array = None
        self.fftn_obj = None
        self.ifftn_obj = None

        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(300)

    def setup(self, shape, dtype):
        self.shape = shape
        self.dtype = np.complex64 if np.issubdtype(dtype, np.floating) else dtype

        self.input_array = pyfftw.empty_aligned(shape, dtype=self.dtype)
        self.output_array = pyfftw.empty_aligned(shape, dtype=self.dtype)

        self.fftn_obj = pyfftw.FFTW(
            self.input_array, self.output_array,
            axes=range(len(shape)),
            direction='FFTW_FORWARD',
            threads=1,
            flags=('FFTW_MEASURE',)
        )
        self.ifftn_obj = pyfftw.FFTW(
            self.output_array, self.input_array,
            axes=range(len(shape)),
            direction='FFTW_BACKWARD',
            normalise_idft=True,
            threads=1,
            flags=('FFTW_MEASURE',)
        )

    def fftn(self, array):
        self.input_array[...] = np.ascontiguousarray(array, dtype=self.dtype)
        return self.fftn_obj()

    def ifftn(self, array):
        self.output_array[...] = np.ascontiguousarray(array, dtype=self.dtype)
        return self.ifftn_obj()


class FFTManager:
    def __init__(self, preferred_backend: str = 'fftw'):
        try:
            self.backend = FFTWBackend()
        except ImportError:
            raise RuntimeError("pyFFTW is required.")

    def setup(self, shape, dtype):
        self.backend.setup(shape, dtype)

    def fftn(self, array):
        return self.backend.fftn(array)

    def ifftn(self, array):
        return self.backend.ifftn(array)


# ===========================================================================
# Bubble manager
# ===========================================================================

class OptimizedBubbleManager:
    """Dynamically-growing array of nucleated bubble centres, velocities, and radii."""

    def __init__(self, box_size: float, initial_capacity: int = 10_000,
                 growth_factor: float = 1.5):
        self.box_size = box_size
        self.count = 0
        self.capacity = initial_capacity
        self.growth_factor = growth_factor

        self.centers = np.zeros((self.capacity, 3), dtype=FLOAT_TYPE)
        self.radii_comoving = np.zeros(self.capacity, dtype=FLOAT_TYPE)
        self.velocities = np.zeros(self.capacity, dtype=FLOAT_TYPE)
        self.birth_times = np.zeros(self.capacity, dtype=FLOAT_TYPE)

    def _resize_if_needed(self):
        if self.count >= self.capacity:
            new_cap = int(self.capacity * self.growth_factor)
            self.centers.resize((new_cap, 3), refcheck=False)
            self.radii_comoving.resize(new_cap, refcheck=False)
            self.velocities.resize(new_cap, refcheck=False)
            self.birth_times.resize(new_cap, refcheck=False)
            self.capacity = new_cap

    def add_bubble(self, center: np.ndarray, velocity: float, birth_time: float):
        self._resize_if_needed()
        idx = self.count
        wrapped = center.copy()
        for dim in range(3):
            wrapped[dim] = (wrapped[dim] + self.box_size / 2) % self.box_size - self.box_size / 2
        self.centers[idx] = wrapped
        self.radii_comoving[idx] = 0.0
        self.velocities[idx] = velocity
        self.birth_times[idx] = birth_time
        self.count += 1

    def evolve_bubbles(self, tau: float):
        if self.count == 0:
            return
        evolve_bubbles_vectorized(
            self.radii_comoving, self.velocities, self.birth_times,
            self.count, tau
        )

    def get_centers_and_radii_comoving(self, scale_factor: float = None
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return np.array([]).reshape(0, 3), np.array([])
        return self.centers[:self.count].copy(), self.radii_comoving[:self.count].copy()

    def get_centers_and_radii_physical(self, scale_factor: float
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return np.array([]).reshape(0, 3), np.array([])
        return (self.centers[:self.count].copy(),
                self.radii_comoving[:self.count] * scale_factor)


# ===========================================================================
# Tracer manager
# ===========================================================================

class NucleationTracerManager:
    """
    Monte Carlo tracer particles for tracking the false-vacuum volume fraction.

    Tracers are placed uniformly in the comoving box.  A tracer is removed
    when it is engulfed by a bubble, giving a stochastic estimator of the
    false-vacuum fraction f_fv(τ) = N_active / N_total.
    """

    def __init__(self, num_tracers: int, box_size: float,
                 H_PT: float = 1.0, Ngrid: int = 16, num_hash_cells: int = 8):
        self.box_size = box_size
        self.H_PT = H_PT
        self.Ngrid = Ngrid
        self.num_hash_cells = num_hash_cells

        print(f"\n{'='*60}")
        print("Nucleation Tracer Configuration")
        print(f"  Numba threads : {nb.get_num_threads()}")
        print(f"  Box size      : {box_size:.3f} (comoving)")
        print(f"  Tracers       : {num_tracers}")
        print(f"  Hash cells    : {num_hash_cells}^3 = {num_hash_cells**3}")
        print(f"{'='*60}\n")

        self.num_total_tracers = num_tracers
        self.tracers = ((np.random.rand(num_tracers, 3) - 0.5) *
                        box_size).astype(FLOAT_TYPE)
        self.active_indices = np.arange(num_tracers, dtype=np.int64)

    @property
    def num_active_tracers(self) -> int:
        return len(self.active_indices)

    def get_active_zone_volume(self) -> float:
        return self.box_size ** 3

    def get_zone_info(self) -> Dict[str, float]:
        hubble_radius = 1.0 / self.H_PT
        return {
            'zone_size': self.box_size,
            'zone_volume': self.box_size ** 3,
            'box_volume': self.box_size ** 3,
            'volume_fraction': 1.0,
            'hubble_volumes': (self.box_size / hubble_radius) ** 3,
            'cells_across': self.box_size / (self.box_size / self.Ngrid),
        }

    def get_false_vacuum_fraction(self) -> float:
        if self.num_total_tracers == 0:
            return 0.0
        return self.num_active_tracers / self.num_total_tracers

    def get_active_tracer_coords(self):
        return self.tracers[self.active_indices]

    def update_state(self, bubble_centers: np.ndarray,
                     bubble_radii_comoving: np.ndarray):
        if self.num_active_tracers == 0:
            return
        self.active_indices = find_surviving_tracers_simple(
            self.tracers, self.active_indices,
            bubble_centers, bubble_radii_comoving,
            self.box_size, self.num_hash_cells
        )


# ===========================================================================
# Mass field manager
# ===========================================================================

class MassFieldManager:
    """
    Time-interpolated mass field m²(τ, x).

    Snapshots of the bubble configuration are computed at intervals
    dtau_bubble and the mass field is smoothly interpolated between them
    using a cubic Hermite (smoothstep) kernel.  A per-step cache avoids
    redundant recomputation when the same τ is queried more than once.
    """

    def __init__(self, X, Y, Z, m0_sq, mb_sq, box_size, dtau_theta,
                 v_bubble=0.5):
        self.X, self.Y, self.Z = X, Y, Z
        self.m0_sq, self.mb_sq = m0_sq, mb_sq
        self.box_size = box_size

        dx_grid = box_size / X.shape[0]
        self.dtau_bubble = max(dtau_theta,
                               0.3 * dx_grid / max(v_bubble, 1e-10))

        self.prev_mass = None
        self.prev_tau = None
        self.next_mass = None
        self.next_tau = None
        self.cached_mass_squared = None
        self.cache_tau = None
        self.cache_tolerance = 1e-6

    def get_mass_field(self, bubble_centers, bubble_radii_comoving, tau,
                       false_frac=None, phase_transition_complete=False):
        if (self.cached_mass_squared is not None and
                self.cache_tau is not None and
                abs(tau - self.cache_tau) < self.cache_tolerance):
            return self.cached_mass_squared

        if phase_transition_complete or (false_frac is not None and false_frac <= 0.0):
            mass = create_true_vacuum(self.X.shape, self.m0_sq, self.X.dtype)
            self.cached_mass_squared = mass
            self.cache_tau = tau
            return mass

        if self.prev_mass is None:
            mass = calculate_mass_field_snapshot_subsampled(
                self.X, self.Y, self.Z, bubble_centers, bubble_radii_comoving,
                self.m0_sq, self.mb_sq, self.box_size
            )
            self.prev_mass = mass.copy()
            self.next_mass = mass.copy()
            self.prev_tau = tau
            self.next_tau = tau + self.dtau_bubble
            self.cached_mass_squared = mass
            self.cache_tau = tau
            return mass

        while tau >= self.next_tau:
            self.prev_mass = self.next_mass.copy()
            self.prev_tau = self.next_tau
            self.next_mass = calculate_mass_field_snapshot_subsampled(
                self.X, self.Y, self.Z, bubble_centers, bubble_radii_comoving,
                self.m0_sq, self.mb_sq, self.box_size
            )
            self.next_tau += self.dtau_bubble

        alpha = (tau - self.prev_tau) / max(self.next_tau - self.prev_tau, 1e-12)
        alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        mass = (1.0 - alpha_smooth) * self.prev_mass + alpha_smooth * self.next_mass
        self.cached_mass_squared = mass
        self.cache_tau = tau
        return mass

    def get_cached_mass_squared(self):
        return self.cached_mass_squared

    def invalidate_cache(self):
        self.cache_tau = None


# ===========================================================================
# Energy calculator
# ===========================================================================

class EnergyCalculator:
    """
    Volume-averaged energy components of the axion field.

    All energies are normalised by the comoving box volume so they scale
    as the physical energy density times a³.
    """

    def __init__(self, config: SimulationConfig, field_evolver):
        self.config = config
        self.field_evolver = field_evolver
        self.m0 = config.m0
        self.dx = field_evolver.dx
        self.use_fft = config.use_fft_laplacian

    def _gradient_density_fft(self, theta, a_tau):
        theta_k = self.field_evolver.fft.fftn(theta)
        grad_sq = np.real(self.field_evolver.fft.ifftn(
            self.field_evolver.K_SQUARED * np.abs(theta_k)**2
        ))
        return 0.5 * grad_sq / (a_tau**2 * self.m0**2)

    def _gradient_density_fd(self, theta, a_tau):
        dx = self.dx

        def grad4(f, axis):
            r = lambda s: np.roll(f, shift=s, axis=axis)
            return (-r(2) + 8*r(1) - 8*r(-1) + r(-2)) / (12 * dx)

        grad_sq = sum(grad4(theta, ax)**2 for ax in range(3))
        return 0.5 * grad_sq / (a_tau**2 * self.m0**2)

    def calculate_energy_components(self, tau: float,
                                     mass_squared: np.ndarray,
                                     theta: np.ndarray,
                                     theta_prime: np.ndarray,
                                     bubble_mask: Optional[np.ndarray] = None
                                     ) -> Dict[str, float]:
        a_tau = self.field_evolver.cosmology.scale_factor(tau)

        kinetic_d  = 0.5 * theta_prime**2 / (a_tau**2 * self.m0**2)
        potential_d = (mass_squared / self.m0**2) * (1 - np.cos(theta))
        gradient_d = (self._gradient_density_fft(theta, a_tau) if self.use_fft
                      else self._gradient_density_fd(theta, a_tau))
        total_d = kinetic_d + gradient_d + potential_d

        vol_norm = self.config.boxSize_comoving**3
        dV = self.field_evolver.dV

        bubble_energy = 0.0
        bubble_vol_frac = 0.0
        if bubble_mask is not None:
            bubble_energy = float(np.sum(total_d[bubble_mask]) * dV / vol_norm)
            bubble_vol_frac = float(np.sum(bubble_mask) / bubble_mask.size)

        return {
            'total_energy':    float(np.sum(total_d) * dV / vol_norm),
            'kinetic_energy':  float(np.sum(kinetic_d) * dV / vol_norm),
            'gradient_energy': float(np.sum(gradient_d) * dV / vol_norm),
            'potential_energy': float(np.sum(potential_d) * dV / vol_norm),
            'bubble_energy':   bubble_energy,
            'bubble_volume_fraction': bubble_vol_frac,
            'scale_factor':    float(a_tau),
        }


# ===========================================================================
# Field evolver
# ===========================================================================

class FieldEvolver:
    """
    Leapfrog integration of the axion field on a cubic periodic lattice.

    Provides two half-steps (part1 / part2) that together form one full
    Störmer-Verlet step at second-order accuracy.
    """

    def __init__(self, config: SimulationConfig, fft_manager: FFTManager,
                 cosmology):
        self.config = config
        self.fft = fft_manager
        self.cosmology = cosmology
        self.parent_simulation = None

        self._setup_grids()
        self._setup_k_grids()

        self.theta       = np.full(self.grid_shape, config.theta0_initial, dtype=FLOAT_TYPE)
        self.theta_prime = np.zeros(self.grid_shape, dtype=FLOAT_TYPE)

        dtau_theta = self.get_recommended_timestep(config.tau0)
        self.mass_manager = MassFieldManager(
            self.X, self.Y, self.Z,
            config.m0**2, config.mb**2,
            config.boxSize_comoving, dtau_theta, config.v_bubble
        )
        self.use_fft_laplacian = config.use_fft_laplacian

    def _setup_grids(self):
        self.grid_shape = (self.config.Ngrid,) * 3
        self.dx = self.config.boxSize_comoving / self.config.Ngrid
        self.dV = self.dx ** 3
        x = np.linspace(-self.config.boxSize_comoving / 2,
                         self.config.boxSize_comoving / 2,
                         self.config.Ngrid, endpoint=False, dtype=FLOAT_TYPE)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')

    def _setup_k_grids(self):
        k = 2 * np.pi * np.fft.fftfreq(self.config.Ngrid, self.dx)
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K_SQUARED = self.KX**2 + self.KY**2 + self.KZ**2

    def get_mass_field(self, *args, **kwargs):
        return self.mass_manager.get_mass_field(*args, **kwargs)

    def _laplacian_fft(self, field):
        field_k = self.fft.fftn(field)
        return np.real(self.fft.ifftn(-self.K_SQUARED * field_k))

    def _laplacian_fd(self, field):
        dx = self.dx

        def lap4(f, axis):
            r = lambda s: np.roll(f, s, axis=axis)
            return (-r(2) + 16*r(1) - 30*f + 16*r(-1) - r(-2)) / (12 * dx**2)

        return lap4(field, 0) + lap4(field, 1) + lap4(field, 2)

    def _laplacian(self, field):
        return self._laplacian_fft(field) if self.use_fft_laplacian else self._laplacian_fd(field)

    def evolve_step_part1(self, tau, dtau, bubble_centers, bubble_radii_comoving):
        """First half of the leapfrog step: advance θ by dtau using θ' at τ."""
        mass_sq = self.get_mass_field(bubble_centers, bubble_radii_comoving, tau)
        H_conf  = 1.0 / max(tau, 1e-12)
        a       = self.cosmology.scale_factor(tau)

        accel = (self._laplacian(self.theta)
                 - a**2 * mass_sq * np.sin(self.theta)
                 - 2.0 * H_conf * self.theta_prime)

        self.theta_prime += 0.5 * dtau * accel
        self.theta       += dtau * self.theta_prime

    def evolve_step_part2(self, tau_new, dtau, bubble_centers, bubble_radii_comoving):
        """Second half of the leapfrog step: kick θ' using the force at τ + dtau."""
        mass_sq_new = self.get_mass_field(bubble_centers, bubble_radii_comoving, tau_new)
        H_conf_new  = 1.0 / max(tau_new, 1e-12)
        a_new       = self.cosmology.scale_factor(tau_new)

        accel_new = (self._laplacian(self.theta)
                     - a_new**2 * mass_sq_new * np.sin(self.theta)
                     - 2.0 * H_conf_new * self.theta_prime)

        self.theta_prime += 0.5 * dtau * accel_new

    def get_recommended_timestep(self, tau) -> float:
        k_max = np.sqrt(3) * (np.pi / self.dx)
        omega_max = np.sqrt(k_max**2 + (self.config.m0 * self.cosmology.scale_factor(tau))**2)
        dtau_cfl      = self.config.cfl_factor / omega_max
        dtau_friction = self.config.friction_factor * max(tau, 1e-6)
        return min(dtau_cfl, dtau_friction)


# ===========================================================================
# Main simulation class
# ===========================================================================

class AxionSimulation:
    """
    Full axion field simulation: nucleation + field evolution + diagnostics.

    Initialise via the factory function `create_simulation_pt_at_zero` rather
    than directly, so that the cosmology and config are set up consistently.
    """

    def __init__(self, config: SimulationConfig, cosmology,
                 fft_backend: str = 'fftw', H_PT: float = None,
                 nucleation_mode: str = 'standard'):
        self.config = config
        self.cosmology = cosmology
        self.H_PT = H_PT if H_PT is not None else getattr(config, 'H_PT', 1.0)
        self.nucleation_mode = nucleation_mode

        self.fft_manager  = FFTManager(fft_backend)
        self.field_evolver = FieldEvolver(config, self.fft_manager, cosmology)

        if config.enable_nucleation:
            self.bubble_manager = OptimizedBubbleManager(config.boxSize_comoving)
            self.tracer_manager = NucleationTracerManager(
                config.num_tracers, config.boxSize_comoving,
                H_PT=self.H_PT, Ngrid=config.Ngrid,
                num_hash_cells=config.spatial_hash_cells
            )
            self.nucleation_volume = self.tracer_manager.get_active_zone_volume()
            self.zone_info = self.tracer_manager.get_zone_info()
        else:
            self.bubble_manager   = OptimizedBubbleManager(config.boxSize_comoving)
            self.tracer_manager   = None
            self.nucleation_volume = 0.0
            self.zone_info        = {}

        self.tracers_nucleated = 0
        self.tracers_engulfed  = 0

        self.fft_manager.setup(self.field_evolver.grid_shape, FLOAT_TYPE)
        self.field_evolver.parent_simulation = self
        self.energy_calculator = EnergyCalculator(config, self.field_evolver)

        self.phase_transition_complete    = False
        self.current_false_vacuum_fraction = 1.0

        self.simulation_data = {
            'times_conformal':              [],
            'times_cosmic':                 [],
            'bubble_counts':                [],
            'false_vacuum_fractions_main':  [],
            'energy_history':               [],
            'nucleation_volume':            self.nucleation_volume,
            'box_volume':                   config.boxSize_comoving**3,
            'zone_info':                    self.zone_info,
            'H_PT':                         self.H_PT,
        }
        self.checkpoint_snapshots = []

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _nucleation_rate(self, t_cosmic: float) -> float:
        return self.config.Gamma_0 * np.exp(self.config.beta * (t_cosmic - self.config.t_0))

    def _calculate_and_save_energy(self, tau: float):
        centers, radii = self.bubble_manager.get_centers_and_radii_comoving()
        mass_sq = self.field_evolver.mass_manager.get_cached_mass_squared()
        if mass_sq is None:
            mass_sq = self.field_evolver.get_mass_field(
                centers, radii, tau,
                self.current_false_vacuum_fraction,
                self.phase_transition_complete
            )
        bubble_mask = None
        if self.bubble_manager.count > 0:
            bubble_mask = get_bubble_mask_snapshot(
                self.field_evolver.X, self.field_evolver.Y, self.field_evolver.Z,
                centers, radii, self.config.boxSize_comoving
            )
        energy = self.energy_calculator.calculate_energy_components(
            tau, mass_sq,
            self.field_evolver.theta,
            self.field_evolver.theta_prime,
            bubble_mask
        )
        energy['num_bubbles'] = self.bubble_manager.count
        energy['tau']         = tau
        energy['t_cosmic']    = self.cosmology.cosmic_time_from_conformal(tau)
        self.simulation_data['energy_history'].append(energy)

    def _save_checkpoint(self, tau: float):
        centers, radii = self.bubble_manager.get_centers_and_radii_comoving()
        self.checkpoint_snapshots.append({
            'tau':                   tau,
            't_cosmic':              self.cosmology.cosmic_time_from_conformal(tau),
            'theta':                 self.field_evolver.theta.copy(),
            'theta_prime':           self.field_evolver.theta_prime.copy(),
            'bubble_centers':        centers.copy(),
            'bubble_radii':          radii.copy(),
            'false_vacuum_fraction': self.current_false_vacuum_fraction,
        })

    def _save_snapshot(self, tau: float):
        self.simulation_data['times_conformal'].append(tau)
        self.simulation_data['times_cosmic'].append(
            self.cosmology.cosmic_time_from_conformal(tau))
        self.simulation_data['bubble_counts'].append(self.bubble_manager.count)
        self.simulation_data['false_vacuum_fractions_main'].append(
            self.current_false_vacuum_fraction)

    def _finalize_results(self) -> Dict[str, Any]:
        return {
            **self.simulation_data,
            'checkpoint_snapshots': self.checkpoint_snapshots,
            'config':               self.config,
            'cosmology':            self.cosmology,
            'tracers_nucleated':    self.tracers_nucleated,
            'tracers_engulfed':     self.tracers_engulfed,
        }

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run_simulation(self, tau_final: float, save_interval: int = 10,
                       progress_interval: int = 100,
                       progress_bar: bool = False) -> Dict[str, Any]:
        """
        Run the simulation from τ₀ to τ_final.

        Parameters
        ----------
        tau_final : float
            Final conformal time.
        save_interval : int
            Number of timesteps between lightweight snapshot saves.
        progress_interval : int
            Number of timesteps between stdout / log-file progress updates.
        progress_bar : bool
            Unused; kept for API compatibility.

        Returns
        -------
        dict
            Keys: times_conformal, times_cosmic, bubble_counts,
            false_vacuum_fractions_main, energy_history, config, cosmology,
            tracers_nucleated, tracers_engulfed, checkpoint_snapshots, …
        """
        if tau_final <= self.config.tau0:
            raise ValueError("tau_final must be greater than tau0")

        config     = self.config
        field      = self.field_evolver
        bubble_mgr = self.bubble_manager
        tracer_mgr = self.tracer_manager
        cosmology  = self.cosmology
        energy_calc = self.energy_calculator

        tau        = config.tau0
        step_count = 0
        est_dtau   = field.get_recommended_timestep(tau)
        nsteps     = max(1, int((tau_final - tau) / est_dtau))

        energy_interval    = max(1, config.energy_save_interval)
        checkpoint_interval = max(0, config.checkpoint_interval)
        save_interval       = max(1, save_interval)
        progress_interval   = max(1, progress_interval)

        nucleation_volume    = getattr(self, 'nucleation_volume', None)
        num_tracers          = config.num_tracers
        box_size             = config.boxSize_comoving
        tracers_exist        = (tracer_mgr is not None and
                                tracer_mgr.num_active_tracers > 0)

        log_path = getattr(self, 'performance_log_path', 'simulation_progress.log')
        self.performance_log_path = log_path
        log_file = open(log_path, 'a', buffering=1)

        try:
            t_loop_start = time.time()

            while tau < tau_final:
                dtau = field.get_recommended_timestep(tau)
                if tau + dtau > tau_final:
                    dtau = tau_final - tau

                centers_start, radii_start = bubble_mgr.get_centers_and_radii_comoving(
                    cosmology.scale_factor(tau))

                field.evolve_step_part1(tau, dtau, centers_start, radii_start)

                # -- nucleation subcycle --
                tau_sub = 0.0
                vol_per_tracer_com = ((nucleation_volume / float(num_tracers))
                                      if num_tracers > 0 else 0.0)

                while tau_sub < dtau:
                    dtau_nucl = dtau - tau_sub

                    if tracers_exist and tracer_mgr.num_active_tracers > 0:
                        cur_tau  = tau + tau_sub
                        self.current_false_vacuum_fraction = tracer_mgr.get_false_vacuum_fraction()
                        t_cos    = cosmology.cosmic_time_from_conformal(cur_tau)
                        gamma_t  = self._nucleation_rate(t_cos)
                        a        = cosmology.scale_factor(cur_tau)

                        vol_per_tracer_phys = vol_per_tracer_com * a**3
                        dt_cos              = dtau_nucl * a
                        p_nuc               = gamma_t * vol_per_tracer_phys * dt_cos

                        # Cap so that at most ~10 % of tracers nucleate per sub-step
                        if p_nuc > 0.1:
                            dtau_nucl = dtau_nucl * (0.1 / p_nuc)
                            dt_cos    = dtau_nucl * a
                            p_nuc     = gamma_t * vol_per_tracer_phys * dt_cos

                        if p_nuc > 0.0:
                            active_coords  = tracer_mgr.get_active_tracer_coords()
                            rolls          = np.random.rand(len(active_coords)).astype(FLOAT_TYPE)
                            nucleated_mask = rolls < p_nuc

                            if nucleated_mask.any():
                                for center in active_coords[nucleated_mask]:
                                    bubble_mgr.add_bubble(center, config.v_bubble,
                                                          tau + tau_sub)
                                tracer_mgr.active_indices = \
                                    tracer_mgr.active_indices[~nucleated_mask]
                                self.tracers_nucleated += int(nucleated_mask.sum())

                    tau_sub += dtau_nucl

                tau_new = tau + dtau
                bubble_mgr.evolve_bubbles(tau_new)

                centers_end, radii_end = bubble_mgr.get_centers_and_radii_comoving(
                    cosmology.scale_factor(tau_new))
                field.evolve_step_part2(tau_new, dtau, centers_end, radii_end)

                if tracers_exist and tracer_mgr.num_active_tracers > 0:
                    N_before = tracer_mgr.num_active_tracers
                    tracer_mgr.update_state(centers_end, radii_end)
                    self.tracers_engulfed += N_before - tracer_mgr.num_active_tracers
                    self.current_false_vacuum_fraction = tracer_mgr.get_false_vacuum_fraction()

                if self.current_false_vacuum_fraction <= 0.0:
                    self.phase_transition_complete = True

                if step_count % save_interval == 0:
                    self._save_snapshot(tau_new)

                    if step_count % energy_interval == 0:
                        centers_now, radii_now = bubble_mgr.get_centers_and_radii_comoving(
                            cosmology.scale_factor(tau_new))
                        mass_sq = field.mass_manager.get_cached_mass_squared()
                        if mass_sq is None:
                            mass_sq = field.get_mass_field(
                                centers_now, radii_now, tau_new,
                                self.current_false_vacuum_fraction,
                                self.phase_transition_complete)

                        bm = (None if bubble_mgr.count == 0 else
                              get_bubble_mask_snapshot(
                                  field.X, field.Y, field.Z,
                                  centers_now, radii_now, box_size))

                        energy = energy_calc.calculate_energy_components(
                            tau_new, mass_sq, field.theta, field.theta_prime, bm)
                        energy['num_bubbles'] = bubble_mgr.count
                        energy['tau']         = tau_new
                        energy['t_cosmic']    = cosmology.cosmic_time_from_conformal(tau_new)
                        self.simulation_data['energy_history'].append(energy)

                    if checkpoint_interval > 0 and step_count % checkpoint_interval == 0:
                        self._save_checkpoint(tau_new)

                if step_count % progress_interval == 0:
                    elapsed = time.time() - t_loop_start
                    try:
                        sys.stdout.write(
                            f"\rStep {step_count}/{nsteps}  "
                            f"τ={tau_new:.5g}  "
                            f"bubbles={bubble_mgr.count}  "
                            f"fv={self.current_false_vacuum_fraction:.4f}")
                        sys.stdout.flush()
                    except Exception:
                        pass
                    log_file.write(
                        f"{time.asctime()} | step={step_count} | "
                        f"tau={tau_new:.6g} | bubbles={bubble_mgr.count} | "
                        f"fv_frac={self.current_false_vacuum_fraction:.6g} | "
                        f"elapsed_s={elapsed:.2f}\n")

                tau = tau_new
                step_count += 1

            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass

            self._save_snapshot(tau)
            try:
                self._calculate_and_save_energy(tau)
            except Exception:
                pass

            return self._finalize_results()

        finally:
            try:
                log_file.close()
            except Exception:
                pass


# ===========================================================================
# Factory
# ===========================================================================

def create_simulation_pt_at_zero(H_PT: float, beta: float,
                                  start_time: float, end_time: float,
                                  nucleation_mode: str = 'standard',
                                  **kwargs):
    """
    Create an AxionSimulation with the PT-at-zero cosmology convention.

    Parameters
    ----------
    H_PT : float
        Hubble rate at the phase transition in units of m_0.
    beta : float
        Exponential nucleation rate parameter β (same units as H_PT).
    start_time : float
        Simulation start in cosmic time.
    end_time : float
        Simulation end in cosmic time.
    nucleation_mode : str
        Reserved for future use; pass 'standard'.
    **kwargs
        Forwarded to SimulationConfig (e.g. Ngrid, num_tracers, theta0_initial).

    Returns
    -------
    sim : AxionSimulation
    tau_final : float
        Conformal time corresponding to end_time.
    """
    cosmology = CosmologyManagerPTAtZeroStartAtZero(H_PT)
    tau_initial = cosmology.conformal_time_from_cosmic(start_time)
    tau_final   = cosmology.conformal_time_from_cosmic(end_time)

    config = SimulationConfig(
        tau0    = tau_initial,
        a0      = cosmology.scale_factor(tau_initial),
        Gamma_0 = H_PT**4,
        t_0     = cosmology.t_PT,
        beta    = beta,
        **kwargs
    )

    sim = AxionSimulation(config, cosmology,
                          fft_backend='fftw',
                          H_PT=H_PT,
                          nucleation_mode=nucleation_mode)
    return sim, tau_final
