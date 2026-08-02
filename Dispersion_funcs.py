"""Velocity-dispersion measurements with iterative aperture convergence.

This module contains only the public measurement code. Scaling-relation and
velocity-bias fitting routines are intentionally excluded.
"""

import numpy as np
from astropy import units as u
from astropy.stats import biweight_midvariance

from escape_analysis_functions import get_velocity_dispersion_data
from escape_theory_functions import rho_crit_z


def biweight_sigma_1d(velocities, c=9.0):
    """Return the 1D biweight velocity dispersion in km/s.

    Non-finite values are removed. ``nan`` is returned when the remaining
    sample is too small or the biweight variance is invalid.
    """
    velocities = np.asarray(velocities, dtype=float)
    velocities = velocities[np.isfinite(velocities)]
    if len(velocities) < 2:
        return np.nan

    variance = float(biweight_midvariance(velocities, c=c))
    if not np.isfinite(variance) or variance < 0:
        return np.nan
    return float(np.sqrt(variance))


def bootstrap_sigma_with_v_errors(
    gal_v,
    gal_v_err=None,
    c=9.0,
    n_resamples=20000,
    seed=None,
):
    """Bootstrap a biweight dispersion while perturbing velocity errors.

    ``gal_v`` and ``gal_v_err`` are in km/s. A scalar error is broadcast to
    every galaxy; ``None`` applies no velocity perturbation.
    """
    gal_v = np.asarray(gal_v, dtype=float).ravel()
    if gal_v_err is None:
        gal_v_err = np.zeros_like(gal_v)
    else:
        gal_v_err = np.asarray(gal_v_err, dtype=float)
        gal_v_err = (
            np.full_like(gal_v, float(gal_v_err))
            if gal_v_err.ndim == 0
            else gal_v_err.ravel()
        )

    if gal_v.shape != gal_v_err.shape:
        raise ValueError("gal_v and gal_v_err must have matching shapes.")
    if np.any(gal_v_err[np.isfinite(gal_v_err)] < 0):
        raise ValueError("Velocity errors must be non-negative.")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")

    valid = np.isfinite(gal_v) & np.isfinite(gal_v_err)
    gal_v, gal_v_err = gal_v[valid], gal_v_err[valid]
    if len(gal_v) < 2:
        raise ValueError("At least two finite velocities are required.")

    sigma_hat = biweight_sigma_1d(gal_v, c=c)
    if not np.isfinite(sigma_hat):
        raise ValueError("The input sample has an invalid dispersion.")

    rng = np.random.default_rng(seed)
    boot = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        indices = rng.integers(0, len(gal_v), size=len(gal_v))
        v_draw = gal_v[indices] + rng.normal(0.0, gal_v_err[indices])
        boot[i] = biweight_sigma_1d(v_draw, c=c)

    finite_boot = boot[np.isfinite(boot)]
    if len(finite_boot) < 2:
        raise ValueError("The bootstrap produced too few finite dispersions.")

    ci16, ci84 = np.percentile(finite_boot, [16, 84])
    return {
        "sigma_hat": float(sigma_hat),
        "err_std": float(np.std(finite_boot, ddof=1)),
        "err_low": float(sigma_hat - ci16),
        "err_high": float(ci84 - sigma_hat),
        "ci16": float(ci16),
        "ci84": float(ci84),
        "bootstrap_distribution": boot,
    }


def make_sigma_to_r200_carlberg(cosmo):
    """Return ``r200 = sqrt(3) * sigma / [10 H(z)]`` in physical Mpc."""
    def sigma_to_r200(sigma, z):
        sigma = np.asarray(sigma, dtype=float)
        r200 = np.sqrt(3.0) * sigma / (10.0 * cosmo.H(z).value)
        return float(r200) if r200.ndim == 0 else r200

    return sigma_to_r200


def converge_aperture_data(
    cluster_positional_data,
    galaxy_positional_data,
    coremin_cut,
    cut,
    cosmo_params,
    cosmo_name,
    sigma_to_r200,
    R_init,
    aperture_factor=1.0,
    tol=0.01,
    max_iter=50,
    min_gal=10,
    verbose=False,
):
    """
    Iteratively converge the velocity-dispersion aperture.

    At each iteration, galaxies are selected within the current aperture,
    interlopers are removed, sigma_v is measured, and the aperture is
    updated using

        R_new = aperture_factor * r200(sigma_v).

    Parameters
    ----------
    R_init : float
        Starting aperture in Mpc.
    verbose : bool, default=False
        If True, print the starting and final aperture values.

    Returns
    -------
    result : dict
        Converged membership, dispersion, r200, aperture, and diagnostics.
    """
    z = cluster_positional_data[2]
    R_start = float(R_init)

    def measure(R):
        r, v = get_velocity_dispersion_data(
            cluster_positional_data,
            galaxy_positional_data,
            R,
            coremin_cut,
            cut,
            cosmo_params,
            cosmo_name,
        )

        if len(v) < min_gal:
            return None, None, np.nan

        return (
            np.asarray(r, dtype=float),
            np.asarray(v, dtype=float),
            biweight_sigma_1d(v),
        )

    R = R_start
    R_previous = R
    converged = False
    n_iterations = 0

    for iteration in range(1, max_iter + 1):
        n_iterations = iteration
        r, v, sigma = measure(R)

        if not np.isfinite(sigma):
            R = R_previous
            break

        r200_new = sigma_to_r200(sigma, z)
        R_new = aperture_factor * r200_new

        if np.isfinite(R) and abs(R_new - R) / R < tol:
            R = R_new
            converged = True
            break

        R_previous = R
        R = R_new

    r, v, sigma = measure(R)

    if not np.isfinite(sigma):
        R = R_previous
        r, v, sigma = measure(R)

    if not np.isfinite(sigma):
        if verbose:
            print(
                f"Aperture convergence failed: "
                f"start={R_start:.3f} Mpc, "
                f"last valid={R:.3f} Mpc"
            )

        return {
            "r": np.array([]),
            "v": np.array([]),
            "sigma": np.nan,
            "r200": np.nan,
            "R_start": R_start,
            "R_aperture": float(R),
            "N": 0,
            "n_iterations": n_iterations,
            "converged": False,
        }

    r200 = float(sigma_to_r200(sigma, z))
    N = int(np.sum((r > 0.2 * R) & (r < R)))

    if verbose:
        print(
            f"Aperture convergence: "
            f"start={R_start:.3f} Mpc -> "
            f"final={R:.3f} Mpc; "
            f"sigma-derived r200={r200:.3f} Mpc; "
            f"iterations={n_iterations}; "
            f"converged={converged}"
        )

    return {
        "r": r,
        "v": v,
        "sigma": float(sigma),
        "r200": r200,
        "R_start": R_start,
        "R_aperture": float(R),
        "N": N,
        "n_iterations": n_iterations,
        "converged": converged,
    }


def calculate_converged_sigma_data(
    index,
    cluster_positional_data_all,
    galaxy_positional_data,
    sample_use,
    cosmo_params,
    cosmo_name,
    coremin_cut = 0.44,
    velocity_cut = 4500,
    sigma_to_r200=None,
    cosmo=None,
    M200_init=None,
    aperture_factor=1.0,
    tol=0.01,
    max_iter=50,
    min_gal=10,
    n_resamples=20_000,
    seed=42,
    gal_v_err_use=30.0,
    include_aperture_error=True,
    verbose=False,
):
    """Measure a converged observed-cluster velocity dispersion.

    The aperture is converged once. The final membership is then bootstrapped
    with velocity perturbations. Optionally, an aperture-sensitivity term is
    added in quadrature to the bootstrap errors.

    ``cluster_positional_data_all`` must contain rows of at least
    ``[name, RA, Dec, z]``. ``M200_init``, when supplied, is a linear mass in
    solar masses and is used only to initialize the aperture.
    """
    if sigma_to_r200 is None:
        if cosmo is None:
            raise ValueError("Pass either sigma_to_r200 or cosmo.")
        sigma_to_r200 = make_sigma_to_r200_carlberg(cosmo)

    try:
        cluster_row = cluster_positional_data_all[index]
    except IndexError as exc:
        raise IndexError(f"Cluster index {index} is outside the table.") from exc
    if len(cluster_row) < 4:
        raise ValueError("Cluster rows must contain [name, RA, Dec, z].")

    cluster_positional_data = (
        float(cluster_row[1]),
        float(cluster_row[2]),
        float(cluster_row[3]),
    )

    if M200_init is None:
        R_init = 2.0
    else:
        if not np.isfinite(M200_init) or M200_init <= 0:
            raise ValueError("M200_init must be a positive linear mass.")
        rho_crit = rho_crit_z(
            cluster_positional_data[2], cosmo_params, cosmo_name
        ).to_value(u.Msun / u.Mpc**3)
        R_init = (
            3.0 * M200_init / (4.0 * np.pi * 200.0 * rho_crit)
        ) ** (1.0 / 3.0)

    convergence = converge_aperture_data(
        cluster_positional_data,
        galaxy_positional_data,
        coremin_cut,
        velocity_cut,
        cosmo_params,
        cosmo_name,
        sigma_to_r200,
        R_init,
        aperture_factor=aperture_factor,
        tol=tol,
        max_iter=max_iter,
        min_gal=min_gal,
        verbose=verbose
    )

    if not np.isfinite(convergence["sigma"]):
        return {
            "sigma_hat": np.nan,
            "err_low": np.nan,
            "err_high": np.nan,
            "ci16": np.nan,
            "ci84": np.nan,
            "aperture_sigma_std": np.nan,
            "between_draw_std": np.nan,
            "within_draw_std": np.nan,
            "bootstrap_distribution": np.array([]),
            "pooled_distribution": np.array([]),
            "all_N": [0],
            "r200": np.nan,
            "R_aperture": convergence["R_aperture"],
            "converged": False,
        }

    velocities = np.asarray(convergence["v"], dtype=float)
    velocity_info = bootstrap_sigma_with_v_errors(
        velocities,
        gal_v_err=np.full(len(velocities), gal_v_err_use),
        n_resamples=n_resamples,
        seed=seed,
    )

    sigma_hat = velocity_info["sigma_hat"]
    err_low = velocity_info["err_low"]
    err_high = velocity_info["err_high"]
    aperture_sigma_std = 0.0

    if include_aperture_error and sigma_hat > 0:
        fractional_error = 0.5 * (err_low + err_high) / sigma_hat
        aperture = convergence["R_aperture"]
        trial_radii = (
            max(0.1, aperture * (1.0 - fractional_error)),
            aperture * (1.0 + fractional_error),
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

            if len(trial_velocities) >= min_gal:
                trial_sigma = biweight_sigma_1d(trial_velocities)
                if np.isfinite(trial_sigma):
                    trial_sigmas.append(trial_sigma)

        if len(trial_sigmas) == 2:
            aperture_sigma_std = 0.5 * abs(trial_sigmas[1] - trial_sigmas[0])
            err_low = float(np.hypot(err_low, aperture_sigma_std))
            err_high = float(np.hypot(err_high, aperture_sigma_std))

    distribution = velocity_info["bootstrap_distribution"]
    return {
        "sigma_hat": float(sigma_hat),
        "err_low": float(err_low),
        "err_high": float(err_high),
        "ci16": float(velocity_info["ci16"]),
        "ci84": float(velocity_info["ci84"]),
        "aperture_sigma_std": float(aperture_sigma_std),
        "between_draw_std": float(aperture_sigma_std),
        "within_draw_std": float(velocity_info["err_std"]),
        "bootstrap_distribution": distribution,
        "pooled_distribution": distribution,
        "all_N": [int(convergence["N"])],
        "r200": float(convergence["r200"]),
        "R_aperture": float(convergence["R_aperture"]),
        "converged": bool(convergence["converged"]),
    }



