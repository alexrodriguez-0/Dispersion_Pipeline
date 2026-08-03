# Galaxy Cluster Velocity Dispersion Pipeline

Tools for measuring galaxy-cluster line-of-sight velocity dispersions from spectroscopic redshift catalogs using an iteratively converged \(r_{200}\) aperture.

The pipeline begins with a projected aperture, selects cluster galaxies and removes interlopers, measures a robust biweight velocity dispersion, converts that dispersion to \(r_{200}\), and repeats until the aperture converges. Uncertainties are estimated by bootstrap resampling while perturbing the galaxy velocities by their measurement errors.

> **Please cite:** Rodriguez et al. 2026a, in prep., if this pipeline is used in scientific work.

---

## Repository layout

```text
Dispersion_pipeline2/
├── Dispersion_funcs.py
├── run_dispersion_example.ipynb
└── README.md
```

- `Dispersion_funcs.py` contains the public measurement functions.
- `run_dispersion_example.ipynb` demonstrates the pipeline for the galaxy cluster A7 using HeCS and HeCS-SZ spectroscopic-redshift data.

The main user-facing function is:

```python
from Dispersion_funcs import calculate_converged_sigma_data
```

---

## Required dependency: Escape Velocity Library

This repository depends on the [Escape Velocity Library — 2026 Edition](https://github.com/alexrodriguez-0/Escape_Velocity_Library-2026-Edition).

Only two functions from that repository are required:

```python
from escape_analysis_functions import get_velocity_dispersion_data
from escape_theory_functions import rho_crit_z
```

They are used as follows:

- `get_velocity_dispersion_data` constructs the projected cluster phase space, applies the radial and velocity selections, and removes interlopers for a supplied aperture.
- `rho_crit_z` converts an optional initial \(M_{200}\) estimate into an initial \(r_{200}\). It is not needed when the default or a user-supplied initial aperture is used.

Clone both repositories:

```bash
git clone https://github.com/alexrodriguez-0/Escape_Velocity_Library-2026-Edition.git
git clone https://github.com/alexrodriguez-0/Dispersion_pipeline2.git
```

Then add the escape-library function directory to your Python path. For example, in a notebook:

```python
from pathlib import Path
import sys

escape_function_path = Path(
    "/path/to/Escape_Velocity_Library-2026-Edition/Function_Libraries"
)

if str(escape_function_path) not in sys.path:
    sys.path.insert(0, str(escape_function_path))
```

Run the notebook from the dispersion-repository directory, or similarly add that directory to `sys.path`.

---

## Installation

A typical environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas astropy matplotlib emcee jupyter
```

The dispersion module directly uses NumPy and Astropy. Additional packages are required when importing the helper modules from the Escape Velocity Library.

---

## Method overview

For a cluster at redshift \(z\), the pipeline performs the following steps:

1. Begin with an initial projected aperture \(R\).
2. Select galaxies within that aperture and an absolute line-of-sight velocity cut.
3. Remove interlopers using the shifting-gapper procedure implemented by `get_velocity_dispersion_data`.
4. Measure the line-of-sight dispersion using the biweight estimator.
5. Convert the measured dispersion to \(r_{200}\) using the Carlberg-style relation

\[
r_{200}=\frac{\sqrt{3}\,\sigma_v}{10H(z)}.
\]

6. Update the aperture according to

\[
R_{\rm new}=f_{\rm ap}\,r_{200}(\sigma_v),
\]

where `aperture_factor` is \(f_{\rm ap}\). The default `aperture_factor=1` measures the dispersion within \(r_{200}\).

7. Repeat until

\[
\frac{|R_{\rm new}-R|}{R}<{\tt tol}.
\]

8. Bootstrap the final cleaned membership, perturbing the selected velocities by their supplied measurement errors.
9. Optionally estimate an additional aperture-sensitivity uncertainty by remeasuring the dispersion at slightly smaller and larger apertures.

The aperture convergence used here is independent of the escape-mass aperture procedure. No escape-velocity mass is required to measure the dispersion.

---

## Required input format

### Cluster position

The cluster center is supplied as:

```python
cluster_positional_data = (
    cluster_ra_deg,
    cluster_dec_deg,
    cluster_redshift,
)
```

Right ascension and declination are in decimal degrees. Redshift is dimensionless.

### Galaxy catalog

For the standard observational mode, the galaxy catalog should have shape

```text
(N_galaxies, 3)
```

with columns:

```text
RA_deg   DEC_deg   redshift
```

For example:

```python
import numpy as np

galaxy_positional_data = np.genfromtxt(
    "/path/to/galaxy_spectroscopic_catalog.txt"
)
```

The input catalog should cover a sufficiently large region around the cluster that the converged aperture is not limited by the catalog boundary.

---

## Quick start: A7 example

The included notebook `run_dispersion_example.ipynb` demonstrates the measurement for A7.

A minimal version is:

```python
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from Dispersion_funcs import (
    calculate_converged_sigma_data,
    make_sigma_to_r200_carlberg,
)

omega_m = 0.3
h = 0.7

cosmo_name = "FlatLambdaCDM"
cosmo = FlatLambdaCDM(
    H0=100.0 * h,
    Om0=omega_m,
    name=cosmo_name,
)

# Parameter convention expected by the Escape Velocity Library helpers.
cosmo_params = [cosmo.Om0, cosmo.h]

galaxy_positional_data = np.genfromtxt(
    "/path/to/Rines_galaxy_data.txt"
)

cluster_positional_data = (
    2.9385416666666666,
    32.415694444444444,
    0.106,
)

sigma_to_r200 = make_sigma_to_r200_carlberg(cosmo)

result = calculate_converged_sigma_data(
    cluster_positional_data=cluster_positional_data,
    galaxy_positional_data=galaxy_positional_data,
    cosmo_params=cosmo_params,
    cosmo_name=cosmo_name,
    sigma_to_r200=sigma_to_r200,
    coremin_cut=0.44,
    velocity_cut=4500.0,
    velocity_errors=30.0,
    n_resamples=20000,
    verbose=True,
)

if result["status"] == "ok":
    print(
        "Velocity dispersion: "
        f"{result['sigma_hat']:.1f} "
        f"+{result['err_high']:.1f} "
        f"-{result['err_low']:.1f} km/s"
    )
    print(f"Converged r200: {result['r200']:.3f} Mpc")
    print(f"Final number of members: {result['N_members']}")
else:
    print(f"Measurement failed: {result['reason']}")
```

With `verbose=True`, the pipeline reports the starting aperture, final aperture, sigma-derived \(r_{200}\), number of iterations, and convergence status.

---

## Initial aperture

The initial aperture controls only the starting point of the fixed-point iteration. It is not imposed on the final result.

There are three initialization options.

### Default initialization

When neither an aperture nor a mass is provided:

```python
initial_aperture_mpc = None
initial_mass_msun = None
```

the calculation starts at 2 Mpc.

### User-supplied aperture

```python
initial_aperture_mpc = 1.5
```

The value is in physical Mpc.

### Initial mass

```python
initial_mass_msun = 1.0e15
```

This must be a positive linear mass in solar masses, not \(\log_{10}M\). It is used only to convert the initial mass to an initial \(r_{200}\). It does not otherwise enter the dispersion measurement.

Do not supply both `initial_aperture_mpc` and `initial_mass_msun`.

A useful stability test is to repeat the calculation from several starting apertures and verify that they converge to similar final values.

---

## Important parameters

### `coremin_cut`

```python
coremin_cut = 0.44
```

Inner protection radius used by the shifting-gapper interloper rejection, expressed as a fraction of the current aperture.

### `velocity_cut`

```python
velocity_cut = 4500.0
```

Maximum absolute line-of-sight velocity in km/s considered during the phase-space selection. Some clusters may require a smaller cut when high-velocity foreground or background structures contaminate the sample.

### `aperture_factor`

```python
aperture_factor = 1.0
```

Sets the final measurement aperture relative to the sigma-derived \(r_{200}\). The default measures within \(r_{200}\).

### `tol`

```python
tol = 0.01
```

Fractional aperture-change tolerance. The default requires the aperture to stabilize to within 1%.

### `velocity_errors`

```python
velocity_errors = 30.0
```

One-sigma line-of-sight velocity uncertainty in km/s. A scalar assigns the same error to every final member. Use `None` or `0.0` to omit velocity perturbations.

### `n_resamples`

```python
n_resamples = 20000
```

Number of bootstrap realizations. Smaller values can be used for testing, while final measurements should use enough realizations for stable uncertainty estimates.

### `include_aperture_error`

```python
include_aperture_error = True
```

When enabled, the pipeline estimates how the measured dispersion changes under small perturbations of the final aperture and adds this term in quadrature to the bootstrap errors.

---

## Main outputs

`calculate_converged_sigma_data` returns a dictionary. Important entries include:

```python
result["status"]
result["reason"]

result["sigma_hat"]
result["err_low"]
result["err_high"]
result["ci16"]
result["ci84"]

result["r200"]
result["r200_ci16"]
result["r200_ci84"]

result["R_start"]
result["R_aperture"]
result["n_iterations"]
result["converged"]

result["N_members"]
result["N"]

result["bootstrap_distribution"]
result["aperture_history_mpc"]
result["sigma_history_kms"]
```

Here:

- `sigma_hat` is the biweight velocity dispersion of the final cleaned membership.
- `err_low` and `err_high` include the optional aperture-sensitivity term.
- `r200` is the physical radius inferred from the final dispersion.
- `N_members` is the total final cleaned membership.
- `N` counts final galaxies between \(0.2R_{\rm aperture}\) and \(R_{\rm aperture}\).
- `aperture_history_mpc` and `sigma_history_kms` record the convergence process.

The raw bootstrap-only quantities are also returned as:

```python
result["bootstrap_ci16"]
result["bootstrap_ci84"]
result["bootstrap_err_low"]
result["bootstrap_err_high"]
```

---

## Convergence diagnostics

A successful numerical measurement does not necessarily guarantee that the selected phase-space membership is physically appropriate. Recommended checks include:

1. Confirm that `result["status"] == "ok"`.
2. Inspect `result["converged"]`.
3. Compare `R_start`, `R_aperture`, and `r200`.
4. Verify that the final sample contains a reasonable number of galaxies.
5. Repeat the measurement from different initial apertures.
6. Inspect the radius-velocity phase space when contamination or substructure is suspected.

The convergence history can be plotted simply with:

```python
import matplotlib.pyplot as plt

plt.plot(
    result["aperture_history_mpc"],
    marker="o",
)
plt.xlabel("Iteration")
plt.ylabel("Aperture [Mpc]")
plt.tight_layout()
plt.show()
```

---

## Public functions

The module exposes:

```python
biweight_sigma_1d
bootstrap_sigma_with_v_errors
make_sigma_to_r200_carlberg
converge_aperture_data
calculate_converged_sigma_data
```

Most users should call only `make_sigma_to_r200_carlberg` and `calculate_converged_sigma_data`.

---

## Data acknowledgement

The A7 example uses galaxy spectroscopy from the HeCS and HeCS-SZ cluster samples. Users should cite the appropriate original data publications when using those catalogs or derived data products.

---

## Citation

If this pipeline contributes to a publication, please cite:

```text
Rodriguez et al. 2026a, in preparation.
```

The associated methodology and scientific analysis are still in preparation. Citation information will be updated when the paper becomes publicly available.

This dispersion pipeline also relies on code from the [Escape Velocity Library — 2026 Edition](https://github.com/alexrodriguez-0/Escape_Velocity_Library-2026-Edition). Users should follow the citation guidance in that repository when using its components.
