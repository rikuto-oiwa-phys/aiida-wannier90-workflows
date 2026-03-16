"""Tests for Closest Wannier plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np


def test_plot_cwf_fit_raw():
    """Test plotting the Closest Wannier fitting curve."""
    from aiida_wannier90_workflows.utils.workflows.plot.bands import plot_cwf_fit_raw

    energies = np.linspace(-5.0, 5.0, 21)
    projectability = np.exp(-0.5 * energies**2)

    ax = plot_cwf_fit_raw(
        energies,
        projectability,
        fit_mu_min=-2.0,
        fit_mu_max=2.0,
        fit_sigma_min=0.4,
        fit_sigma_max=0.3,
        opt_mu_min=-2.0,
        opt_mu_max=1.1,
        opt_sigma_min=0.4,
        opt_sigma_max=0.3,
        delta=1e-12,
    )

    assert ax.get_xlabel() == "Energy [eV]"
    assert ax.get_ylabel() == "Projectability / weight"
    assert len(ax.lines) >= 6

    plt.close(ax.figure)
