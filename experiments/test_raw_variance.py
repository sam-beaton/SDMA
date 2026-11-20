"""
Validate the raw rank-1 estimator of 1'Q1.

Uses Lefort-Besnard et al.'s simulation generator to produce null
Z-statistic maps from K pipelines, with either independent or
compound-symmetric inter-pipeline correlation Q.

At each voxel, the pipeline values are summed and squared to form
the raw estimator s_hat(v) = (1'y(v))^2 of s = 1'Q1.

The empirical mean, variance and relative standard deviation of
s_hat across independent simulated voxels are compared against
the theoretical results:

    E[s_hat] = s
    Var(s_hat) = 2s^2
    Relative SD = sqrt(2)

Purpose: Verify the single-voxel estimator's theoretical properties
before introducing spatial pooling or Gaussian smoothing.

Reference: SDMA working notes, Sections 4.1-4.5.
"""

from pathlib import Path
import sys
import numpy as np

# Import Jeremy's existing simulation generator
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation_generator import generate_simulation

# Simulation parameters
K = 20
J = 20000

# Test independent and correlated pipelines
for case, rho in [
    ("Null", 0.0),
    ("Null correlated", 0.8),
]:
    Y = generate_simulation(
        case=case,
        K=K,
        J=J,
        corr=rho,
        seed=1,
    )

    # Sum across pipelines at each voxel
    u = np.sum(Y, axis=0)

    # Raw rank-1 estimator of s = 1'Q1
    s_hat = u**2

    # Theoretical values
    s_true = K * (1 + (K - 1) * rho)
    theoretical_variance = 2 * s_true**2

    # Empirical values across simulated voxels
    empirical_mean = np.mean(s_hat)
    empirical_variance = np.var(s_hat)
    empirical_relative_sd = (
        np.std(s_hat) / empirical_mean
    )

    print(f"\nCase: {case}")
    print(f"True s: {s_true:.4f}")
    print(f"Empirical mean: {empirical_mean:.4f}")
    print(f"Theoretical variance: {theoretical_variance:.4f}")
    print(f"Empirical variance: {empirical_variance:.4f}")
    print(f"Empirical relative SD: {empirical_relative_sd:.4f}")
    print(f"Theoretical relative SD: {np.sqrt(2):.4f}")
