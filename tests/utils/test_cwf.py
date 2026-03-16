"""Tests for Closest Wannier fitting utilities."""

import numpy as np

from aiida_wannier90_workflows.utils.cwf import (
    cwf_window_function,
    fit_cwf_parameters_from_contents,
)


def test_fit_cwf_parameters_from_contents():
    """Test fitting CWF parameters from synthetic eig/amn contents."""
    energies = np.linspace(-5.0, 5.0, 41)
    mu_min = -3.0
    mu_max = 2.0
    sigma_min = 0.4
    sigma_max = 0.2
    projectability = cwf_window_function(
        energies,
        mu_min=mu_min,
        mu_max=mu_max,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    projectability = np.clip(projectability, 0.0, 1.0)

    eig_content = "\n".join(
        f"{band_index} 1 {energy:.16f}"
        for band_index, energy in enumerate(energies, start=1)
    )
    amn_lines = ["generated for test", f"{len(energies)} 1 1"]
    for band_index, value in enumerate(projectability, start=1):
        amn_lines.append(f"1 {band_index} 1 {np.sqrt(value):.16f} 0.0")

    parameters = fit_cwf_parameters_from_contents(
        eig_content=eig_content,
        amn_content="\n".join(amn_lines),
    )

    assert abs(parameters["cwf_mu_min"] - mu_min) < 1e-2
    assert abs(parameters["cwf_mu_max"] - (mu_max - 3.0 * sigma_max)) < 1e-2
    assert abs(parameters["cwf_sigma_min"] - sigma_min) < 1e-2
    assert abs(parameters["cwf_sigma_max"] - sigma_max) < 1e-2
