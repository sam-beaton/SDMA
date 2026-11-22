
"""
Validate the EDF of a Gaussian-smoothed rank-1 variance
estimator in three spatial dimensions.

Simulates homogeneous, isotropic, spatially correlated
Gaussian fields corresponding to K independent pipelines.
The pipeline sum is generated directly using Gaussian
closure, avoiding unnecessary simulation of K separate
fields.

The raw estimator s_hat(v) = u(v)^2 is smoothed with
3D Gaussian kernels of varying FWHM.

Compares:
    1. Empirical EDF across independent Monte Carlo batches.
    2. Exact discrete covariance-based EDF.
    3. Continuous-field EDF from Section 6.1.

The empirical EDF is calculated from the average sampling
variance across batches. Approximate 95% Monte Carlo
confidence intervals are obtained using a t interval for
the mean batch variance, transformed onto the EDF scale.

Purpose: Extend the validated 1D experiment to 3D while
keeping Q = I and the true variance s = K constant.

References: SDMA working notes, Sections 5.8 and 6.1.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import t


# --------------------------------------------------
# Simulation parameters
# --------------------------------------------------

K = 20
GRID_SIZE = 33

N_SEEDS = 50
N_SIM = 10000
SEED_GENERATOR = 2026

F_D = 3.0
F_V_VALUES = [0, 1, 2, 3, 4, 5, 6, 8, 10]

shape = (GRID_SIZE,) * 3
centre = (GRID_SIZE // 2,) * 3
s_true = K


def fwhm_to_sigma(fwhm):
    """Convert Gaussian FWHM to standard deviation."""
    return fwhm / np.sqrt(8 * np.log(2))


# --------------------------------------------------
# Construct intrinsic spatial correlation
# --------------------------------------------------

sigma_d = fwhm_to_sigma(F_D)

# Discrete impulse at the centre of the 3D grid
impulse = np.zeros(shape)
impulse[centre] = 1.0

# Gaussian kernel used to generate spatially correlated data
data_kernel = gaussian_filter(
    impulse,
    sigma=sigma_d,
    mode="wrap",
)

# Marginal variance of filtered unit-variance white noise
data_variance = np.sum(data_kernel**2)

# Circular autocovariance of the discrete data kernel
data_covariance = np.fft.ifftn(
    np.abs(np.fft.fftn(data_kernel))**2
).real

# Normalised spatial correlation function
data_correlation = data_covariance / data_variance


# --------------------------------------------------
# Precompute variance-smoothing kernels and theory
# --------------------------------------------------

kernels = []
theory_discrete = []
theory_continuous = []

for f_v in F_V_VALUES:

    if f_v == 0:
        weights = impulse.copy()
    else:
        weights = gaussian_filter(
            impulse,
            sigma=fwhm_to_sigma(f_v),
            mode="wrap",
        )

    kernels.append(weights)

    # Circular autocorrelation of the smoothing weights
    weight_autocorrelation = np.fft.ifftn(
        np.abs(np.fft.fftn(weights))**2
    ).real

    # Exact discrete covariance calculation
    variance_factor = np.sum(
        weight_autocorrelation * data_correlation**2
    )

    theory_discrete.append(1.0 / variance_factor)

    # Continuous-field approximation for D = 3
    theory_continuous.append(
        (1 + 2 * (f_v / F_D)**2)**1.5
    )

theory_discrete = np.asarray(theory_discrete)
theory_continuous = np.asarray(theory_continuous)


# --------------------------------------------------
# Monte Carlo simulation across independent seeds
# --------------------------------------------------

# Reproducibly generate independent batch seeds
seed_rng = np.random.default_rng(SEED_GENERATOR)

seeds = seed_rng.integers(
    0,
    2**32,
    size=N_SEEDS,
    dtype=np.uint64,
)

# Store the mean and sampling variance from each batch
batch_means = np.zeros((N_SEEDS, len(F_V_VALUES)))
batch_variances = np.zeros((N_SEEDS, len(F_V_VALUES)))

for b, seed in enumerate(seeds):

    rng = np.random.default_rng(int(seed))

    estimates = np.zeros((N_SIM, len(F_V_VALUES)))

    for sim in range(N_SIM):

        # Equivalent to summing K independent Gaussian
        # pipeline fields with identical spatial covariance
        white_noise = rng.standard_normal(shape)

        u = gaussian_filter(
            white_noise,
            sigma=sigma_d,
            mode="wrap",
        )

        # Normalise so that Var[u(v)] = K
        u *= np.sqrt(s_true / data_variance)

        # Raw rank-1 variance estimator
        raw_s = u**2

        # Smoothed estimate at the central voxel
        for i, weights in enumerate(kernels):
            estimates[sim, i] = np.sum(raw_s * weights)

    # Summarise this independent Monte Carlo batch
    batch_means[b] = estimates.mean(axis=0)
    batch_variances[b] = estimates.var(axis=0, ddof=1)

    print(f"Completed batch {b + 1}/{N_SEEDS}")


# --------------------------------------------------
# Aggregate results across independent seeds
# --------------------------------------------------

# Average sampling variance across batches
mean_variance = batch_variances.mean(axis=0)

# EDF calculated from averaged variance
empirical_edf = 2 * s_true**2 / mean_variance

# Standard error of the mean batch variance
se_variance = (
    batch_variances.std(axis=0, ddof=1)
    / np.sqrt(N_SEEDS)
)

# Approximate 95% confidence intervals for variance
t_crit = t.ppf(0.975, df=N_SEEDS - 1)

var_lower = mean_variance - t_crit * se_variance
var_upper = mean_variance + t_crit * se_variance

if np.any(var_lower <= 0):
    raise ValueError(
        "Non-positive variance confidence bound. "
        "Increase the number of simulations or batches."
    )

# Transform confidence intervals onto EDF scale
# The reciprocal reverses the interval endpoints
edf_lower = 2 * s_true**2 / var_upper
edf_upper = 2 * s_true**2 / var_lower


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

print("\nRaw estimator diagnostic:")
print(f"Empirical mean:     {batch_means[:, 0].mean():.4f}")
print(f"Expected mean:      {s_true:.4f}")
print(f"Empirical variance: {mean_variance[0]:.4f}")
print(f"Expected variance:  {2 * s_true**2:.4f}")


# --------------------------------------------------
# Print EDF results
# --------------------------------------------------

print("\nAveraged EDF results across independent seeds:\n")

for i, f_v in enumerate(F_V_VALUES):

    print(
        f"F_V={f_v:>2} | "
        f"Empirical={empirical_edf[i]:>8.3f} | "
        f"Discrete={theory_discrete[i]:>8.3f} | "
        f"Continuous={theory_continuous[i]:>8.3f} | "
        f"95% CI=[{edf_lower[i]:.3f}, "
        f"{edf_upper[i]:.3f}]"
    )


# --------------------------------------------------
# Plot results
# --------------------------------------------------

plt.figure(figsize=(9, 5.5))

plt.plot(
    F_V_VALUES,
    empirical_edf,
    "o-",
    label="Empirical EDF (averaged across seeds)",
)

plt.fill_between(
    F_V_VALUES,
    edf_lower,
    edf_upper,
    alpha=0.2,
    label="95% Monte Carlo CI",
)

plt.plot(
    F_V_VALUES,
    theory_discrete,
    "s--",
    label="Discrete theory",
)

plt.plot(
    F_V_VALUES,
    theory_continuous,
    "^-.",
    label="Continuous approximation",
)

plt.xlabel("Variance smoothing FWHM (voxels)")
plt.ylabel("Effective degrees of freedom")

plt.title(
    f"3D Gaussian fields (F_D={F_D}, "
    f"{N_SEEDS} seeds, {N_SIM} realisations per seed)"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# --------------------------------------------------
# Save and display figure
# --------------------------------------------------

output_dir = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "SIMULATIONS"
)

output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "rank1_spatial_correlation_3d_edf.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()