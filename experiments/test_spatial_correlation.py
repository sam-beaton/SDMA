
"""
Validate the EDF of a spatially smoothed rank-1 variance estimator
when the input Gaussian fields are spatially correlated.

Generates repeated null multiverses with K independent pipelines.
Each pipeline is spatially smoothed to produce a Gaussian random
field with specified intrinsic FWHM, F_D, and then rescaled to
unit marginal variance.

The raw estimator s_hat(v) = (1'y(v))^2 is subsequently smoothed
with a Gaussian kernel of FWHM F_V.

The empirical EDF at a fixed voxel is compared with:
(1) the exact discrete-weight covariance calculation, and
(2) the continuous-field approximation from the working notes.

Purpose: Test how intrinsic spatial correlation reduces the
effective information gained by smoothing the variance estimate.

Reference: SDMA working notes, Sections 5.3-5.8 and 6.1.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t

# Simulation parameters
K = 20
J = 1001

F_D = 3.0
F_V_VALUES = [0, 1, 2, 3, 4, 5, 6, 8, 10]

centre = J // 2
s_true = K

def fwhm_to_sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

sigma_d = fwhm_to_sigma(F_D)

# Discrete kernel for generating spatial correlation
impulse = np.zeros(J)
impulse[centre] = 1.0
data_kernel = gaussian_filter1d(
    impulse, sigma=sigma_d, mode="wrap"
)

# Exact marginal variance of the smoothed white noise
data_variance = np.sum(data_kernel**2)

# Circular autocorrelation of the discrete data kernel
data_covariance = np.fft.ifft(
    np.abs(np.fft.fft(data_kernel))**2
).real
data_correlation = data_covariance / data_variance

# Precompute theoretical EDF values
theory_discrete = []
theory_continuous = []

for f_v in F_V_VALUES:
    if f_v == 0:
        weights = impulse.copy()
    else:
        weights = gaussian_filter1d(
            impulse,
            sigma=fwhm_to_sigma(f_v),
            mode="wrap",
        )

    # Covariance of squared Gaussian values uses correlation squared
    weight_autocorrelation = np.fft.ifft(
        np.abs(np.fft.fft(weights))**2
    ).real

    variance_factor = np.sum(
        weight_autocorrelation * data_correlation**2
    )

    theory_discrete.append(1.0 / variance_factor)

    theory_continuous.append(
        (1 + 2 * (f_v / F_D)**2)**0.5
    )


# --------------------------------------------------
# Monte Carlo simulation across independent seeds
# --------------------------------------------------

N_SEEDS = 50
N_SIM = 10000

# Reproducibly select independent batch seeds
seed_rng = np.random.default_rng(2026)
seeds = seed_rng.integers(
    0, 2**32, size=N_SEEDS, dtype=np.uint64
)

# Store empirical variances for each seed and FWHM
batch_variances = np.zeros((N_SEEDS, len(F_V_VALUES)))
batch_means = np.zeros((N_SEEDS, len(F_V_VALUES)))

for b, seed in enumerate(seeds):

    rng = np.random.default_rng(int(seed))
    estimates = np.zeros((N_SIM, len(F_V_VALUES)))

    for sim in range(N_SIM):
        Y = rng.standard_normal((K, J))

        Y = gaussian_filter1d(
            Y, sigma=sigma_d, axis=1, mode="wrap"
        )
        Y /= np.sqrt(data_variance)

        u = Y.sum(axis=0)
        raw_s = u**2

        for i, f_v in enumerate(F_V_VALUES):
            if f_v == 0:
                estimates[sim, i] = raw_s[centre]
            else:
                estimates[sim, i] = gaussian_filter1d(
                    raw_s,
                    sigma=fwhm_to_sigma(f_v),
                    mode="wrap",
                )[centre]

    batch_means[b] = estimates.mean(axis=0)
    batch_variances[b] = estimates.var(axis=0, ddof=1)

    print(f"Completed batch {b + 1}/{N_SEEDS}")


# --------------------------------------------------
# Aggregate results across seeds
# --------------------------------------------------

# Mean sampling variance across batches
mean_variance = batch_variances.mean(axis=0)

# EDF from averaged variance, rather than averaged EDF
empirical_edf = 2 * s_true**2 / mean_variance

# Standard error of the mean variance across batches
se_variance = (
    batch_variances.std(axis=0, ddof=1)
    / np.sqrt(N_SEEDS)
)

# Approximate 95% confidence interval for variance
t_crit = t.ppf(0.975, df=N_SEEDS - 1)

var_lower = mean_variance - t_crit * se_variance
var_upper = mean_variance + t_crit * se_variance

# Transform variance interval into EDF interval
# (reciprocal reverses interval endpoints)
edf_lower = 2 * s_true**2 / var_upper
edf_upper = 2 * s_true**2 / var_lower


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nAveraged results across independent seeds:\n")

for i, f_v in enumerate(F_V_VALUES):
    print(
        f"F_V={f_v:>2} | "
        f"Empirical={empirical_edf[i]:>7.3f} | "
        f"Discrete={theory_discrete[i]:>7.3f} | "
        f"95% CI=[{edf_lower[i]:.3f}, "
        f"{edf_upper[i]:.3f}]"
    )

print("\nRaw estimator diagnostic:")
print("Mean:", batch_means[:, 0].mean())
print("Expected mean:", s_true)
print("Variance:", mean_variance[0])
print("Expected variance:", 2 * s_true**2)

plt.figure(figsize=(8, 5))
plt.plot(
    F_V_VALUES,
    empirical_edf,
    "o-",
    label="Empirical EDF (averaged across seeds)"
)

plt.fill_between(
    F_V_VALUES,
    edf_lower,
    edf_upper,
    alpha=0.2,
    label="95% Monte Carlo CI"
)
plt.plot(F_V_VALUES, theory_discrete, "s--", label="Discrete theory")
plt.plot(F_V_VALUES, theory_continuous, "^-.", label="Continuous approximation")

plt.xlabel("Variance smoothing FWHM (voxels)")
plt.ylabel("Effective degrees of freedom")
plt.title(f"Spatially correlated Gaussian fields (F_D={F_D})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save figure
output_dir = Path(__file__).resolve().parents[1] / "figures" / "SIMULATIONS"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "rank1_spatial_correlation_edf.png",
    dpi=300,
    bbox_inches="tight"
)

# Show figure
plt.show()