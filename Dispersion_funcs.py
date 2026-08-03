"""Velocity-dispersion measurements with iterative aperture convergence.

This module provides the public measurement portion of the velocity-dispersion
pipeline. It measures a cluster velocity dispersion inside an aperture that is
iteratively updated from the measured dispersion itself. Scaling-relation,
velocity-bias, and sample-specific fitting routines are intentionally excluded.

The module depends on two project-level helpers:

``get_velocity_dispersion_data``
    Selects galaxies for a cluster and removes interlopers for a supplied
    projected aperture.

``rho_crit_z``
    Returns the critical density at the cluster redshift and is used only when
    an optional initial mass is converted to an initial aperture.
"""

from collections.abc import Callable

import numpy as np
from astropy import units as u
from astropy.stats import biweight_midvariance
from numpy.typing import ArrayLike, NDArray

from escape_analysis_functions import get_velocity_dispersion_data
from escape_theory_functions import rho_crit_z

__all__ = [
    "biweight_sigma_1d",
    "bootstrap_sigma_with_v_errors",
    "make_sigma_to_r200_carlberg",
    "converge_aperture_data",
    "calculate_converged_sigma_data",
]


def biweight_sigma_1d(velocities: ArrayLike, c: float = 9.0) -> float:
    """Estimate a robust one-dimensional velocity dispersion.

    The dispersion is the square root of the Astropy biweight midvariance.
    Non-finite velocities are removed before the calculation.

    Parameters
    ----------
    velocities : array-like
        Line-of-sight galaxy velocities in km/s.
    c : float, default=9.0
        Tuning constant for the biweight estimator. Larger values reduce the
        aggressiveness of the downweighting.

    Returns
    -------
    float
        Biweight velocity dispersion in km/s. Returns ``numpy.nan`` when fewer
        than two finite velocities remain or the biweight variance is invalid.
    """
    velocities = np.asarray(velocities, dtype=float).ravel()
    velocities = velocities[np.isfinite(velocities)]
    if len(velocities) < 2:
        return np.nan

    variance = float(biweight_midvariance(velocities, c=c))
    if not np.isfinite(variance) or variance < 0.0:
        return np.nan
    return float(np.sqrt(variance))


def bootstrap_sigma_with_v_errors(
    velocities: ArrayLike,
    velocity_errors: ArrayLike | float | None = None,
    c: float = 9.0,
    n_resamples: int = 20000,
    seed: int | None = None,
) -> dict:
    """Bootstrap a biweight dispersion while perturbing velocity errors.

    Every bootstrap realization resamples the galaxies with replacement and
    adds an independent Gaussian velocity perturbation to each selected
    galaxy. This combines finite-sampling uncertainty and specified
    line-of-sight velocity measurement errors.

    Parameters
    ----------
    velocities : array-like
        Line-of-sight galaxy velocities in km/s.
    velocity_errors : array-like, float, or None, default=None
        One-sigma velocity errors in km/s. A scalar is applied to every
        galaxy. An array must have the same shape as ``velocities``. ``None``
        applies no velocity perturbation.
    c : float, default=9.0
        Tuning constant passed to :func:`biweight_sigma_1d`.
    n_resamples : int, default=20000
        Number of bootstrap realizations.
    seed : int or None, default=None
        Seed for NumPy's random-number generator.

    Returns
    -------
    dict
        Dictionary containing:

        ``sigma_hat``
            Biweight dispersion of the unperturbed input sample in km/s.
        ``err_std``
            Standard deviation of the bootstrap distribution in km/s.
        ``err_low``, ``err_high``
            Distances from ``sigma_hat`` to the 16th and 84th percentiles.
        ``ci16``, ``ci84``
            The 16th and 84th percentiles in km/s.
        ``bootstrap_distribution``
            Finite bootstrap dispersion realizations in km/s.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes, contain negative errors,
        contain fewer than two usable velocities, or produce too few finite
        bootstrap dispersions.
    """
    velocities = np.asarray(velocities, dtype=float).ravel()

    if velocity_errors is None:
        velocity_errors = np.zeros_like(velocities)
    else:
        velocity_errors = np.asarray(velocity_errors, dtype=float)
        if velocity_errors.ndim == 0:
            velocity_errors = np.full_like(velocities, float(velocity_errors))
        else:
            velocity_errors = velocity_errors.ravel()

    if velocities.shape != velocity_errors.shape:
        raise ValueError("velocities and velocity_errors must have matching shapes.")
    if np.any(velocity_errors[np.isfinite(velocity_errors)] < 0.0):
        raise ValueError("velocity_errors must be non-negative.")
    if int(n_resamples) < 2:
        raise ValueError("n_resamples must be at least 2.")

    valid = np.isfinite(velocities) & np.isfinite(velocity_errors)
    velocities = velocities[valid]
    velocity_errors = velocity_errors[valid]

    if len(velocities) < 2:
        raise ValueError("At least two finite velocities are required.")

    sigma_hat = biweight_sigma_1d(velocities, c=c)
    if not np.isfinite(sigma_hat):
        raise ValueError("The input sample has an invalid velocity dispersion.")

    rng = np.random.default_rng(seed)
    bootstrap = np.full(int(n_resamples), np.nan)

    for i in range(int(n_resamples)):
        indices = rng.integers(0, len(velocities), size=len(velocities))
        draw = velocities[indices] + rng.normal(0.0, velocity_errors[indices])
        bootstrap[i] = biweight_sigma_1d(draw, c=c)

    bootstrap = bootstrap[np.isfinite(bootstrap)]
    if len(bootstrap) < 2:
        raise ValueError("The bootstrap produced too few finite dispersions.")

    ci16, ci84 = np.percentile(bootstrap, [16.0, 84.0])
    return {
        "sigma_hat": float(sigma_hat),
        "err_std": float(np.std(bootstrap, ddof=1)),
        "err_low": float(max(0.0, sigma_hat - ci16)),
        "err_high": float(max(0.0, ci84 - sigma_hat)),
        "ci16": float(ci16),
        "ci84": float(ci84),
        "bootstrap_distribution": bootstrap,
    }


def make_sigma_to_r200_carlberg(cosmo) -> Callable[[ArrayLike, float], NDArray | float]:
    """Build the Carlberg et al. conversion from dispersion to physical r200.

    The returned function evaluates

    ``r200 = sqrt(3) * sigma / [10 H(z)]``,

    with ``sigma`` in km/s and ``H(z)`` in km/s/Mpc.

    Parameters
    ----------
    cosmo : astropy.cosmology.Cosmology
        Astropy cosmology used to evaluate the Hubble parameter at the cluster
        redshift.

    Returns
    -------
    callable
        Function with signature ``sigma_to_r200(sigma, z)``. It accepts a
        scalar or array of velocity dispersions in km/s and returns physical
        r200 values in Mpc.
    """
    def sigma_to_r200(sigma: ArrayLike, z: float) -> NDArray | float:
        sigma = np.asarray(sigma, dtype=float)
        r200 = np.sqrt(3.0) * sigma / (10.0 * cosmo.H(z).value)
        return float(r200) if r200.ndim == 0 else r200

    return sigma_to_r200


def converge_aperture_data(
    cluster_positional_data: ArrayLike,
    galaxy_positional_data: ArrayLike,
    coremin_cut: float,
    velocity_cut: float,
    cosmo_params: ArrayLike,
    cosmo_name: str,
    sigma_to_r200: Callable,
    initial_aperture_mpc: float,
    aperture_factor: float = 1.0,
    tol: float = 0.01,
    max_iter: int = 50,
    min_galaxies: int = 10,
    biweight_c: float = 9.0,
    verbose: bool = False,
) -> dict:
    """Iteratively converge the aperture used to measure velocity dispersion.

    Starting from ``initial_aperture_mpc``, the function repeatedly selects
    galaxies and removes interlopers using
    :func:`get_velocity_dispersion_data`, measures the biweight dispersion,
    converts that dispersion to r200, and updates the aperture according to

    ``R_new = aperture_factor * r200(sigma_v)``.

    Iteration stops when the fractional aperture change is smaller than
    ``tol`` or when ``max_iter`` iterations have been attempted.

    Parameters
    ----------
    cluster_positional_data : array-like
        Cluster position ``(RA_deg, Dec_deg, z)``. Right ascension and
        declination are in decimal degrees and redshift is dimensionless.
    galaxy_positional_data : array-like
        Galaxy catalog passed to :func:`get_velocity_dispersion_data`.
        For the standard observational path, this is an ``(N, 3)`` array
        containing ``RA_deg``, ``Dec_deg``, and galaxy redshift.
    coremin_cut : float
        Inner radius used by the shifting-gapper interloper-removal routine,
        expressed as a fraction of the current aperture.
    velocity_cut : float
        Absolute line-of-sight velocity cut in km/s.
    cosmo_params : array-like
        Cosmological parameters expected by the project projection utilities.
    cosmo_name : str
        Name of the cosmological model expected by the project utilities.
    sigma_to_r200 : callable
        Function with signature ``sigma_to_r200(sigma, z)`` that converts a
        velocity dispersion in km/s to physical r200 in Mpc.
    initial_aperture_mpc : float
        Starting projected aperture in physical Mpc.
    aperture_factor : float, default=1.0
        Multiplicative factor applied to r200 when defining the next aperture.
        A value of 1 measures within r200.
    tol : float, default=0.01
        Fractional aperture-change tolerance required for convergence.
    max_iter : int, default=50
        Maximum number of aperture iterations.
    min_galaxies : int, default=10
        Minimum number of selected galaxies required for a valid dispersion.
    biweight_c : float, default=9.0
        Tuning constant used by the biweight dispersion estimator.
    verbose : bool, default=False
        Print the starting aperture, final aperture, sigma-derived r200,
        iteration count, and convergence status.

    Returns
    -------
    dict
        Dictionary containing:

        ``r``, ``v``
            Final projected radii in Mpc and line-of-sight velocities in km/s.
        ``sigma``
            Final biweight velocity dispersion in km/s.
        ``r200``
            r200 inferred from the final dispersion in physical Mpc.
        ``R_start``, ``R_aperture``
            Starting and final membership apertures in physical Mpc.
        ``N``
            Number of final galaxies between 0.2 times the final aperture and
            the final aperture.
        ``N_members``
            Total number of galaxies in the final cleaned sample.
        ``n_iterations``
            Number of aperture updates attempted.
        ``converged``
            Whether the fractional aperture-change criterion was met.
        ``aperture_history_mpc``, ``sigma_history_kms``
            Aperture and dispersion values recorded during iteration.

        If no valid final sample is available, numerical outputs are ``nan``
        or empty arrays and ``converged`` is ``False``.

    Raises
    ------
    ValueError
        If the initial aperture or convergence settings are invalid.
    """
    cluster_positional_data = np.asarray(cluster_positional_data, dtype=float).ravel()
    if len(cluster_positional_data) != 3:
        raise ValueError("cluster_positional_data must contain (RA, Dec, z).")

    initial_aperture_mpc = float(initial_aperture_mpc)
    if not np.isfinite(initial_aperture_mpc) or initial_aperture_mpc <= 0.0:
        raise ValueError("initial_aperture_mpc must be finite and positive.")
    if not 0.0 < tol < 1.0:
        raise ValueError("tol must lie between 0 and 1.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1.")
    if int(min_galaxies) < 2:
        raise ValueError("min_galaxies must be at least 2.")
    if aperture_factor <= 0.0:
        raise ValueError("aperture_factor must be positive.")

    cluster_redshift = float(cluster_positional_data[2])

    def measure(aperture_mpc: float):
        radii, velocities = get_velocity_dispersion_data(
            cluster_positional_data,
            galaxy_positional_data,
            aperture_mpc,
            coremin_cut,
            velocity_cut,
            cosmo_params,
            cosmo_name,
        )
        radii = np.asarray(radii, dtype=float)
        velocities = np.asarray(velocities, dtype=float)

        if len(velocities) < min_galaxies:
            return radii, velocities, np.nan

        sigma = biweight_sigma_1d(velocities, c=biweight_c)
        return radii, velocities, sigma

    aperture = initial_aperture_mpc
    last_valid_aperture = aperture
    converged = False
    aperture_history = []
    sigma_history = []

    for iteration in range(1, int(max_iter) + 1):
        radii, velocities, sigma = measure(aperture)
        aperture_history.append(float(aperture))
        sigma_history.append(float(sigma) if np.isfinite(sigma) else np.nan)

        if not np.isfinite(sigma):
            aperture = last_valid_aperture
            break

        last_valid_aperture = aperture
        r200_new = float(sigma_to_r200(sigma, cluster_redshift))
        next_aperture = float(aperture_factor * r200_new)

        if not np.isfinite(next_aperture) or next_aperture <= 0.0:
            aperture = last_valid_aperture
            break

        fractional_change = abs(next_aperture - aperture) / aperture
        aperture = next_aperture

        if fractional_change < tol:
            converged = True
            break

    n_iterations = iteration
    radii, velocities, sigma = measure(aperture)

    if not np.isfinite(sigma) and aperture != last_valid_aperture:
        aperture = last_valid_aperture
        radii, velocities, sigma = measure(aperture)
        converged = False

    if not np.isfinite(sigma):
        if verbose:
            print(
                "Aperture convergence failed: "
                f"start={initial_aperture_mpc:.3f} Mpc, "
                f"last={aperture:.3f} Mpc"
            )
        return {
            "r": np.array([]),
            "v": np.array([]),
            "sigma": np.nan,
            "r200": np.nan,
            "R_start": initial_aperture_mpc,
            "R_aperture": float(aperture),
            "N": 0,
            "N_members": 0,
            "n_iterations": n_iterations,
            "converged": False,
            "aperture_history_mpc": np.asarray(aperture_history),
            "sigma_history_kms": np.asarray(sigma_history),
        }

    r200 = float(sigma_to_r200(sigma, cluster_redshift))
    radial_count = int(np.sum((radii > 0.2 * aperture) & (radii < aperture)))

    if verbose:
        print(
            "Aperture convergence: "
            f"start={initial_aperture_mpc:.3f} Mpc -> "
            f"final={aperture:.3f} Mpc; "
            f"sigma-derived r200={r200:.3f} Mpc; "
            f"iterations={n_iterations}; converged={converged}"
        )

    return {
        "r": radii,
        "v": velocities,
        "sigma": float(sigma),
        "r200": r200,
        "R_start": initial_aperture_mpc,
        "R_aperture": float(aperture),
        "N": radial_count,
        "N_members": int(len(velocities)),
        "n_iterations": n_iterations,
        "converged": bool(converged),
        "aperture_history_mpc": np.asarray(aperture_history),
        "sigma_history_kms": np.asarray(sigma_history),
    }


def _resolve_initial_aperture(
    cluster_redshift: float,
    cosmo_params: ArrayLike,
    cosmo_name: str,
    initial_aperture_mpc: float | None,
    initial_mass_msun: float | None,
) -> tuple[float, str]:
    """Resolve the starting aperture for the public measurement function."""
    if initial_aperture_mpc is not None and initial_mass_msun is not None:
        raise ValueError(
            "Pass initial_aperture_mpc or initial_mass_msun, not both."
        )

    if initial_aperture_mpc is not None:
        aperture = float(initial_aperture_mpc)
        source = "user_aperture"
    elif initial_mass_msun is not None:
        mass = float(initial_mass_msun)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("initial_mass_msun must be finite and positive.")

        rho_critical = rho_crit_z(
            cluster_redshift, cosmo_params, cosmo_name
        ).to_value(u.Msun / u.Mpc**3)

        aperture = (
            3.0 * mass / (4.0 * np.pi * 200.0 * rho_critical)
        ) ** (1.0 / 3.0)
        source = "initial_mass"
    else:
        aperture = 2.0
        source = "default_2_mpc"

    if not np.isfinite(aperture) or aperture <= 0.0:
        raise ValueError("The resolved initial aperture is invalid.")

    return float(aperture), source


def _invalid_sigma_result(convergence: dict, reason: str) -> dict:
    """Construct a consistent invalid-result dictionary."""
    empty = np.array([])
    return {
        "status": "invalid",
        "reason": reason,
        "sigma_hat": np.nan,
        "err_low": np.nan,
        "err_high": np.nan,
        "ci16": np.nan,
        "ci84": np.nan,
        "bootstrap_ci16": np.nan,
        "bootstrap_ci84": np.nan,
        "bootstrap_err_low": np.nan,
        "bootstrap_err_high": np.nan,
        "aperture_sigma_std": np.nan,
        "within_membership_std": np.nan,
        "bootstrap_distribution": empty,
        "r200": np.nan,
        "r200_ci16": np.nan,
        "r200_ci84": np.nan,
        "R_start": convergence.get("R_start", np.nan),
        "R_aperture": convergence.get("R_aperture", np.nan),
        "N": 0,
        "N_members": 0,
        "n_iterations": convergence.get("n_iterations", 0),
        "converged": False,
        "initialization_source": None,
        "aperture_history_mpc": convergence.get(
            "aperture_history_mpc", empty
        ),
        "sigma_history_kms": convergence.get("sigma_history_kms", empty),
        # Backward-compatible aliases used by older notebooks.
        "between_draw_std": np.nan,
        "within_draw_std": np.nan,
        "pooled_distribution": empty,
        "all_N": [0],
    }


def calculate_converged_sigma_data(
    cluster_positional_data: ArrayLike,
    galaxy_positional_data: ArrayLike,
    cosmo_params: ArrayLike,
    cosmo_name: str,
    sigma_to_r200: Callable | None = None,
    cosmo=None,
    initial_aperture_mpc: float | None = None,
    initial_mass_msun: float | None = None,
    coremin_cut: float = 0.44,
    velocity_cut: float = 4500.0,
    aperture_factor: float = 1.0,
    tol: float = 0.01,
    max_iter: int = 50,
    min_galaxies: int = 10,
    biweight_c: float = 9.0,
    n_resamples: int = 20000,
    seed: int | None = 42,
    velocity_errors: ArrayLike | float | None = 30.0,
    include_aperture_error: bool = True,
    verbose: bool = False,
) -> dict:
    """Measure a cluster velocity dispersion in a self-consistent aperture.

    This is the public, sample-independent entry point. It accepts the sky
    position and redshift of one cluster directly; it does not require a
    cluster-table index or contain survey-specific cuts.

    The calculation has three stages:

    1. Starting from an initial aperture, repeatedly select galaxies, remove
       interlopers, measure the biweight dispersion, and update the aperture
       from the supplied sigma-to-r200 relation.
    2. Bootstrap the final cleaned membership while perturbing the specified
       galaxy velocity errors.
    3. Optionally estimate sensitivity to the aperture by remeasuring the
       dispersion at radii shifted by the fractional bootstrap uncertainty
       and add that term in quadrature to the lower and upper errors.

    Parameters
    ----------
    cluster_positional_data : array-like
        Position of one cluster as ``(RA_deg, Dec_deg, z)``. Right ascension
        and declination are in decimal degrees and redshift is dimensionless.
    galaxy_positional_data : array-like
        Galaxy catalog passed to :func:`get_velocity_dispersion_data`.
        For the standard observational path, provide an ``(N, 3)`` array with
        columns ``RA_deg``, ``Dec_deg``, and galaxy redshift.
    cosmo_params : array-like
        Cosmological parameters expected by the project projection utilities.
    cosmo_name : str
        Name of the cosmological model expected by the project utilities.
    sigma_to_r200 : callable or None, default=None
        Function with signature ``sigma_to_r200(sigma, z)`` returning physical
        r200 in Mpc. When omitted, ``cosmo`` must be supplied and the Carlberg
        relation is constructed automatically.
    cosmo : astropy.cosmology.Cosmology or None, default=None
        Astropy cosmology used only to construct the default Carlberg
        sigma-to-r200 relation when ``sigma_to_r200`` is not provided.
    initial_aperture_mpc : float or None, default=None
        Explicit starting aperture in physical Mpc. It affects only the
        starting point of the fixed-point iteration.
    initial_mass_msun : float or None, default=None
        Optional positive linear M200 in solar masses used only to calculate
        the starting aperture. It is not used after initialization. Do not
        supply this together with ``initial_aperture_mpc``.
    coremin_cut : float, default=0.44
        Inner shifting-gapper protection radius expressed as a fraction of the
        current aperture.
    velocity_cut : float, default=4500.0
        Absolute line-of-sight velocity cut in km/s.
    aperture_factor : float, default=1.0
        Factor multiplying sigma-derived r200 to define the measurement
        aperture. A value of 1 measures within r200.
    tol : float, default=0.01
        Fractional aperture-change threshold for convergence.
    max_iter : int, default=50
        Maximum number of aperture iterations.
    min_galaxies : int, default=10
        Minimum number of cleaned galaxies required to measure a dispersion.
    biweight_c : float, default=9.0
        Tuning constant for the biweight dispersion estimator.
    n_resamples : int, default=20000
        Number of bootstrap realizations.
    seed : int or None, default=42
        Seed for the bootstrap random-number generator.
    velocity_errors : array-like, float, or None, default=30.0
        One-sigma line-of-sight velocity errors in km/s. A scalar is assigned
        to every final member. An array must match the number of final members.
        ``None`` applies no velocity perturbation.
    include_aperture_error : bool, default=True
        Estimate an additional aperture-sensitivity term and add it in
        quadrature to the bootstrap lower and upper errors.
    verbose : bool, default=False
        Print the starting and final aperture diagnostics.

    Returns
    -------
    dict
        Dictionary containing:

        ``status``, ``reason``
            ``"ok"`` and an empty reason on success, otherwise ``"invalid"``
            and a description of the failure.
        ``sigma_hat``
            Biweight velocity dispersion of the final membership in km/s.
        ``err_low``, ``err_high``
            Final asymmetric errors in km/s, including the aperture term when
            requested.
        ``ci16``, ``ci84``
            Final interval defined by ``sigma_hat - err_low`` and
            ``sigma_hat + err_high``.
        ``bootstrap_ci16``, ``bootstrap_ci84``
            Raw 16th and 84th percentiles of the bootstrap distribution.
        ``bootstrap_err_low``, ``bootstrap_err_high``
            Bootstrap-only asymmetric errors in km/s.
        ``aperture_sigma_std``
            Half the difference between dispersions measured at the lower and
            upper aperture perturbations, in km/s.
        ``within_membership_std``
            Standard deviation of the bootstrap distribution in km/s.
        ``bootstrap_distribution``
            Bootstrap dispersion realizations in km/s.
        ``r200``, ``r200_ci16``, ``r200_ci84``
            Sigma-derived physical r200 and interval in Mpc.
        ``R_start``, ``R_aperture``
            Starting and final membership apertures in physical Mpc.
        ``N``
            Number of final galaxies between 0.2 times the final aperture and
            the final aperture.
        ``N_members``
            Total number of galaxies in the final cleaned membership.
        ``n_iterations``, ``converged``
            Aperture-convergence diagnostics.
        ``initialization_source``
            One of ``"user_aperture"``, ``"initial_mass"``, or
            ``"default_2_mpc"``.
        ``aperture_history_mpc``, ``sigma_history_kms``
            Values recorded during the aperture iteration.

        Backward-compatible aliases ``between_draw_std``,
        ``within_draw_std``, ``pooled_distribution``, and ``all_N`` are also
        returned for older notebooks.

    Raises
    ------
    ValueError
        If required inputs are invalid or neither ``sigma_to_r200`` nor
        ``cosmo`` is supplied.
    """
    cluster_positional_data = np.asarray(cluster_positional_data, dtype=float).ravel()
    if len(cluster_positional_data) != 3:
        raise ValueError("cluster_positional_data must contain (RA, Dec, z).")
    if not np.all(np.isfinite(cluster_positional_data)):
        raise ValueError("cluster_positional_data must contain finite values.")

    if sigma_to_r200 is None:
        if cosmo is None:
            raise ValueError("Pass sigma_to_r200 or cosmo.")
        sigma_to_r200 = make_sigma_to_r200_carlberg(cosmo)

    initial_aperture, initialization_source = _resolve_initial_aperture(
        cluster_redshift=float(cluster_positional_data[2]),
        cosmo_params=cosmo_params,
        cosmo_name=cosmo_name,
        initial_aperture_mpc=initial_aperture_mpc,
        initial_mass_msun=initial_mass_msun,
    )

    convergence = converge_aperture_data(
        cluster_positional_data=cluster_positional_data,
        galaxy_positional_data=galaxy_positional_data,
        coremin_cut=coremin_cut,
        velocity_cut=velocity_cut,
        cosmo_params=cosmo_params,
        cosmo_name=cosmo_name,
        sigma_to_r200=sigma_to_r200,
        initial_aperture_mpc=initial_aperture,
        aperture_factor=aperture_factor,
        tol=tol,
        max_iter=max_iter,
        min_galaxies=min_galaxies,
        biweight_c=biweight_c,
        verbose=verbose,
    )

    if not np.isfinite(convergence["sigma"]):
        result = _invalid_sigma_result(
            convergence, "A finite dispersion could not be measured."
        )
        result["initialization_source"] = initialization_source
        return result

    velocities = np.asarray(convergence["v"], dtype=float)
    velocity_info = bootstrap_sigma_with_v_errors(
        velocities=velocities,
        velocity_errors=velocity_errors,
        c=biweight_c,
        n_resamples=n_resamples,
        seed=seed,
    )

    sigma_hat = float(velocity_info["sigma_hat"])
    bootstrap_err_low = float(velocity_info["err_low"])
    bootstrap_err_high = float(velocity_info["err_high"])
    aperture_sigma_std = 0.0

    if include_aperture_error and sigma_hat > 0.0:
        fractional_error = (
            0.5 * (bootstrap_err_low + bootstrap_err_high) / sigma_hat
        )
        final_aperture = float(convergence["R_aperture"])
        trial_radii = (
            max(0.1, final_aperture * (1.0 - fractional_error)),
            final_aperture * (1.0 + fractional_error),
        )
        trial_sigmas = []

        for trial_radius in trial_radii:
            try:
                _, trial_velocities = get_velocity_dispersion_data(
                    cluster_positional_data,
                    galaxy_positional_data,
                    trial_radius,
                    coremin_cut,
                    velocity_cut,
                    cosmo_params,
                    cosmo_name,
                )
            except ValueError:
                continue

            if len(trial_velocities) < min_galaxies:
                continue

            trial_sigma = biweight_sigma_1d(
                trial_velocities, c=biweight_c
            )
            if np.isfinite(trial_sigma):
                trial_sigmas.append(trial_sigma)

        if len(trial_sigmas) == 2:
            aperture_sigma_std = 0.5 * abs(trial_sigmas[1] - trial_sigmas[0])

    err_low = float(np.hypot(bootstrap_err_low, aperture_sigma_std))
    err_high = float(np.hypot(bootstrap_err_high, aperture_sigma_std))
    ci16 = float(max(0.0, sigma_hat - err_low))
    ci84 = float(sigma_hat + err_high)
    redshift = float(cluster_positional_data[2])

    r200 = float(convergence["r200"])
    r200_ci16 = float(sigma_to_r200(ci16, redshift))
    r200_ci84 = float(sigma_to_r200(ci84, redshift))
    distribution = velocity_info["bootstrap_distribution"]

    return {
        "status": "ok",
        "reason": "",
        "sigma_hat": sigma_hat,
        "err_low": err_low,
        "err_high": err_high,
        "ci16": ci16,
        "ci84": ci84,
        "bootstrap_ci16": float(velocity_info["ci16"]),
        "bootstrap_ci84": float(velocity_info["ci84"]),
        "bootstrap_err_low": bootstrap_err_low,
        "bootstrap_err_high": bootstrap_err_high,
        "aperture_sigma_std": float(aperture_sigma_std),
        "within_membership_std": float(velocity_info["err_std"]),
        "bootstrap_distribution": distribution,
        "r200": r200,
        "r200_ci16": r200_ci16,
        "r200_ci84": r200_ci84,
        "R_start": float(convergence["R_start"]),
        "R_aperture": float(convergence["R_aperture"]),
        "N": int(convergence["N"]),
        "N_members": int(convergence["N_members"]),
        "n_iterations": int(convergence["n_iterations"]),
        "converged": bool(convergence["converged"]),
        "initialization_source": initialization_source,
        "aperture_history_mpc": convergence["aperture_history_mpc"],
        "sigma_history_kms": convergence["sigma_history_kms"],
        # Backward-compatible aliases used by older notebooks.
        "between_draw_std": float(aperture_sigma_std),
        "within_draw_std": float(velocity_info["err_std"]),
        "pooled_distribution": distribution,
        "all_N": [int(convergence["N"])],
    }
