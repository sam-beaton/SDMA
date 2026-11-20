
"""
Investigate Gaussian smoothing of the rank-1 variance estimator.

Generates repeated null multiverses with K independent pipelines
and spatially independent voxels. At each voxel, the pipeline
values are summed and squared to estimate s = 1'Q1.

Gaussian smoothing is applied at different FWHMs. The sampling
variance of the smoothed estimator at a fixed voxel is estimated
across independent realisations and compared with the exact
discrete-kernel prediction:

    Var[s_hat_h] = 2s^2 * sum(w^2)
    EDF = 1 / sum(w^2)

Purpose: Validate the variance reduction and effective degrees
of freedom (EDF) achieved through Gaussian smoothing, before
introducing spatial correlation or heterogeneous variance.

Reference: SDMA working notes, Sections 5.8 and 6.1.
"""

from pathlib import Path
import sys
import contextlib
import io

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Import Jeremy's existing simulation generator
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation_generator import generate_simulation


# --------------------------------------------------
# Simulation parameters
# --------------------------------------------------

K = 20
J = 1001
N_SIM = 10000

FWHMS = [0, 1, 2, 3, 4, 5, 6, 8, 10]

s_true = K  # Q = I
centre = J // 2

empirical_edf = []
theoretical_edf = []


# --------------------------------------------------
# Run simulations
# --------------------------------------------------

for fwhm in FWHMS:

    sigma = fwhm / np.sqrt(8 * np.log(2))

    estimates = []

    for sim in range(N_SIM):

        with contextlib.redirect_stdout(io.StringIO()):
            Y = generate_simulation(
                case="Null",
                K=K,
                J=J,
                seed=sim,
            )

        # Sum across pipelines
        u = np.sum(Y, axis=0)

        # Raw rank-1 variance estimate
        raw_s = u**2

        # Smooth the raw variance image
        if fwhm == 0:
            smoothed_s = raw_s
        else:
            smoothed_s = gaussian_filter1d(
                raw_s,
                sigma=sigma,
                mode="reflect",
            )

        # Record the estimate at the centre voxel
        estimates.append(smoothed_s[centre])

    estimates = np.asarray(estimates)

    # Empirical sampling variance across realisations
    empirical_variance = np.var(estimates, ddof=1)

    # EDF calculated from the known true variance
    edf_emp = 2 * s_true**2 / empirical_variance

    # Construct the exact discrete smoothing weights
    impulse = np.zeros(J)
    impulse[centre] = 1.0

    if fwhm == 0:
        weights = impulse
    else:
        weights = gaussian_filter1d(
            impulse,
            sigma=sigma,
            mode="reflect",
        )

    edf_theory = 1 / np.sum(weights**2)

    empirical_edf.append(edf_emp)
    theoretical_edf.append(edf_theory)

    print(
        f"FWHM={fwhm:>2} | "
        f"Empirical EDF={edf_emp:>7.3f} | "
        f"Theoretical EDF={edf_theory:>7.3f}"
    )


# --------------------------------------------------
# Plot results
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    FWHMS,
    theoretical_edf,
    "o-",
    label="Theoretical EDF (discrete kernel)",
)

plt.plot(
    FWHMS,
    empirical_edf,
    "s--",
    label="Empirical EDF (simulation)",
)

plt.xlabel("Gaussian smoothing FWHM (voxels)")
plt.ylabel("Effective degrees of freedom")
plt.title("EDF of the smoothed rank-1 variance estimator")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output_dir = ROOT / "figures" / "SIMULATIONS"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "rank1_gaussian_smoothing_edf.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
