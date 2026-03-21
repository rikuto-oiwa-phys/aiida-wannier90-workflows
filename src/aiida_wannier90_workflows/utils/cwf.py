"""Utilities for fitting Closest Wannier parameters."""

import typing as ty

import numpy as np

__all__ = (
    "cwf_window_function",
    "fit_cwf_parameters",
    "fit_cwf_parameters_from_contents",
    "fit_cwf_parameters_raw",
    "read_amn",
    "read_eig",
)


def read_eig(eig_content: str) -> np.ndarray:
    """Read the contents of a ``seedname.eig`` file."""
    eig_data = []
    for line in eig_content.splitlines():
        line = line.strip()
        if not line:
            continue
        band_index, kpoint_index, energy = line.split()
        eig_data.append((int(band_index), int(kpoint_index), float(energy)))

    if not eig_data:
        raise ValueError("Empty eig content")

    num_bands = max(row[0] for row in eig_data)
    num_kpoints = max(row[1] for row in eig_data)
    energies = np.zeros((num_kpoints, num_bands))
    for band_index, kpoint_index, energy in eig_data:
        energies[kpoint_index - 1, band_index - 1] = energy

    return energies


def read_amn(amn_content: str) -> np.ndarray:
    """Read the contents of a ``seedname.amn`` file."""
    lines = [line for line in amn_content.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("Invalid amn content")

    num_bands, num_kpoints, num_wann = (int(value) for value in lines[1].split())
    amn_data = np.genfromtxt(lines[2:]).reshape(num_kpoints, num_wann, num_bands, 5)

    return np.transpose(amn_data[:, :, :, 3] + 1j * amn_data[:, :, :, 4], axes=(0, 2, 1))


def fermi_dist_func(energy: ty.Union[float, np.ndarray], kb_t: float):
    """Return the Fermi-Dirac distribution."""
    if kb_t == 0.0:
        return np.where(np.asarray(energy) < 0.0, 1.0, 0.0)

    return 0.5 * (1.0 - np.tanh(0.5 * np.asarray(energy) / kb_t))


def cwf_window_function(
    energy: ty.Union[float, np.ndarray],
    mu_min: float,
    mu_max: float,
    sigma_min: float,
    sigma_max: float,
    delta: float = 1e-12,
):
    """Return the Closest Wannier window function."""
    return fermi_dist_func(mu_min - energy, sigma_min) + fermi_dist_func(energy - mu_max, sigma_max) - 1.0 + delta


def fit_cwf_parameters_raw(
    energies: np.ndarray,
    amn: np.ndarray,
    sigma_factor: float = 3.0,
    delta: float = 1e-12,
    fermi_energy: ty.Optional[float] = None,
    return_data: bool = False,
) -> dict:
    """Fit Closest Wannier parameters from raw energy and AMN arrays."""
    import lmfit

    fixed_mu_min = -300.0
    fixed_sigma_min = 0.0

    projectability = np.real(np.diagonal(amn @ amn.transpose(0, 2, 1).conjugate(), axis1=1, axis2=2))

    energies_flat = energies.reshape(-1)
    projectability_flat = projectability.reshape(-1)

    model = lmfit.Model(
        lambda energy, mu_min, width, sigma_min, sigma_max: cwf_window_function(
            energy, mu_min, mu_min + width, sigma_min, sigma_max, delta
        )
    )
    params = lmfit.Parameters()
    if fermi_energy is None:
        width_init = max(float(np.max(energies_flat) - fixed_mu_min) + 10.0, 1e-6)
    else:
        width_init = max(float(fermi_energy) - fixed_mu_min, 1e-6)

    params.add("mu_min", value=fixed_mu_min, vary=False)
    params.add(
        "width",
        value=width_init,
        min=1e-6,
        max=np.inf,
    )
    params.add("sigma_min", value=fixed_sigma_min, vary=False)
    params.add("sigma_max", value=1.0, min=0.0, max=30.0)

    result_fit = model.fit(projectability_flat, params, energy=energies_flat)

    mu_min_fit = result_fit.params["mu_min"].value
    mu_max_fit = mu_min_fit + result_fit.params["width"].value
    sigma_min_fit = result_fit.params["sigma_min"].value
    sigma_max_fit = result_fit.params["sigma_max"].value

    result = {
        "cwf_mu_min": float(mu_min_fit),
        "cwf_mu_max": float(mu_max_fit - sigma_factor * sigma_max_fit),
        "cwf_sigma_min": float(sigma_min_fit),
        "cwf_sigma_max": float(sigma_max_fit),
    }

    if return_data:
        data = {
            "energies": energies_flat,
            "projectability": projectability_flat,
            "fit_mu_min": float(mu_min_fit),
            "fit_mu_max": float(mu_max_fit),
            "fit_sigma_min": float(sigma_min_fit),
            "fit_sigma_max": float(sigma_max_fit),
            "delta": float(delta),
        }
        return result, data

    return result


def fit_cwf_parameters_from_contents(
    eig_content: str,
    amn_content: str,
    sigma_factor: float = 3.0,
    delta: float = 1e-12,
    fermi_energy: ty.Optional[float] = None,
    return_data: bool = False,
) -> dict:
    """Fit Closest Wannier parameters from retrieved ``eig`` and ``amn`` contents."""
    energies = read_eig(eig_content)
    amn = read_amn(amn_content)

    if energies.shape[:2] != amn.shape[:2]:
        raise ValueError("Inconsistent eig/amn contents: " f"energies shape={energies.shape}, amn shape={amn.shape}")

    return fit_cwf_parameters_raw(
        energies=energies,
        amn=amn,
        sigma_factor=sigma_factor,
        delta=delta,
        fermi_energy=fermi_energy,
        return_data=return_data,
    )


def fit_cwf_parameters(
    eig_content: str,
    amn_content: str,
    sigma_factor: float = 3.0,
    delta: float = 1e-12,
    fermi_energy: ty.Optional[float] = None,
    return_data: bool = False,
):
    """Compatibility wrapper for fitting Closest Wannier parameters."""
    return fit_cwf_parameters_from_contents(
        eig_content=eig_content,
        amn_content=amn_content,
        sigma_factor=sigma_factor,
        delta=delta,
        fermi_energy=fermi_energy,
        return_data=return_data,
    )
