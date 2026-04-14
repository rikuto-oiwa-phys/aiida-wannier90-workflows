"""Functions for analyzing the band distances of WorkChain results."""

import typing as ty

import numpy as np
import pandas as pd

from aiida import orm
from aiida.common.links import LinkType


def _safe_get_nested_output(node, path: ty.Iterable[str]):
    """Return nested output if available, otherwise ``None``."""
    try:
        current = node.outputs
    except AttributeError:
        return None
    try:
        for part in path:
            current = current[part]
    except (AttributeError, KeyError, TypeError):
        return None
    return current


def _iter_called_descendants_with_labels(node):
    """Yield direct called descendants as ``(label, node)`` pairs."""
    for link in node.base.links.get_outgoing().all():
        if link.link_type not in (LinkType.CALL_CALC, LinkType.CALL_WORK):
            continue
        yield link.link_label, link.node


def _get_called_descendants_by_prefix(node, prefix: str):
    """Return called descendants whose call link label starts with ``prefix``."""
    matches = []
    for label, child in _iter_called_descendants_with_labels(node):
        if label.startswith(prefix):
            matches.append((label, child))
    matches.sort(key=lambda item: item[0])
    return matches


def _get_bands_shape_product(bands):
    """Return the product of the stored bands shape, or ``None`` if unavailable."""
    try:
        return np.prod(bands.base.attributes.all["array|bands"])
    except (KeyError, TypeError):
        return None


def _append_bands_candidate(candidates, source: str, bands):
    """Append a bands candidate if present."""
    if bands is not None:
        candidates.append((source, bands))


def _deduplicate_bands_candidates(candidates):
    """Return candidate list without duplicate bands nodes, preserving order."""
    unique = []
    seen = set()

    for source, bands in candidates:
        pk = getattr(bands, "pk", None)
        key = pk if pk is not None else id(bands)
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, bands))

    return unique


def _collect_wannier_bands_candidates(workchain):
    """Return ordered bands candidates for workchains used in bands distance analysis."""
    from aiida.plugins import WorkflowFactory

    Wannier90BandsWorkChain = WorkflowFactory("wannier90_workflows.bands")
    Wannier90OptimizeWorkChain = WorkflowFactory("wannier90_workflows.optimize")

    candidates = []

    if workchain.process_class == Wannier90OptimizeWorkChain:
        _append_bands_candidate(
            candidates,
            "outputs.wannier90_optimal.interpolated_bands",
            _safe_get_nested_output(workchain, ("wannier90_optimal", "interpolated_bands")),
        )
        _append_bands_candidate(
            candidates,
            "outputs.band_structure",
            _safe_get_nested_output(workchain, ("band_structure",)),
        )

        for label, child in _get_called_descendants_by_prefix(workchain, "wannier90_plot"):
            _append_bands_candidate(
                candidates,
                f"child:{label}.interpolated_bands",
                _safe_get_nested_output(child, ("interpolated_bands",)),
            )
            _append_bands_candidate(
                candidates,
                f"child:{label}.band_structure",
                _safe_get_nested_output(child, ("band_structure",)),
            )

        for label, child in reversed(_get_called_descendants_by_prefix(workchain, "wannier90_optimize_iteration")):
            _append_bands_candidate(
                candidates,
                f"child:{label}.interpolated_bands",
                _safe_get_nested_output(child, ("interpolated_bands",)),
            )
            _append_bands_candidate(
                candidates,
                f"child:{label}.band_structure",
                _safe_get_nested_output(child, ("band_structure",)),
            )

        for label, child in _get_called_descendants_by_prefix(workchain, "wannier90"):
            _append_bands_candidate(
                candidates,
                f"child:{label}.interpolated_bands",
                _safe_get_nested_output(child, ("interpolated_bands",)),
            )
            _append_bands_candidate(
                candidates,
                f"child:{label}.band_structure",
                _safe_get_nested_output(child, ("band_structure",)),
            )

    elif workchain.process_class == Wannier90BandsWorkChain:
        _append_bands_candidate(
            candidates,
            "outputs.wannier90.interpolated_bands",
            _safe_get_nested_output(workchain, ("wannier90", "interpolated_bands")),
        )
        _append_bands_candidate(
            candidates,
            "outputs.band_structure",
            _safe_get_nested_output(workchain, ("band_structure",)),
        )
        for label, child in _get_called_descendants_by_prefix(workchain, "wannier90"):
            _append_bands_candidate(
                candidates,
                f"child:{label}.interpolated_bands",
                _safe_get_nested_output(child, ("interpolated_bands",)),
            )
            _append_bands_candidate(
                candidates,
                f"child:{label}.band_structure",
                _safe_get_nested_output(child, ("band_structure",)),
            )
    else:
        _append_bands_candidate(
            candidates,
            "outputs.interpolated_bands",
            _safe_get_nested_output(workchain, ("interpolated_bands",)),
        )
        _append_bands_candidate(
            candidates,
            "outputs.band_structure",
            _safe_get_nested_output(workchain, ("band_structure",)),
        )

    return _deduplicate_bands_candidates(candidates)


def _get_resolved_fermi_energy(workchain):
    """Get Fermi energy for bands distance, rescuing from child workchains if needed."""
    from aiida.plugins import WorkflowFactory

    from aiida_wannier90_workflows.utils.workflows.plot.bands import (
        get_workchain_fermi_energy,
    )

    Wannier90OptimizeWorkChain = WorkflowFactory("wannier90_workflows.optimize")

    try:
        return get_workchain_fermi_energy(workchain), "workchain"
    except (KeyError, ValueError, AttributeError):
        pass

    candidates = []
    if workchain.process_class == Wannier90OptimizeWorkChain:
        candidates.extend(_get_called_descendants_by_prefix(workchain, "wannier90_plot"))
        candidates.extend(_get_called_descendants_by_prefix(workchain, "wannier90_optimize_iteration"))
        candidates.extend(_get_called_descendants_by_prefix(workchain, "wannier90"))
    else:
        candidates.extend(_get_called_descendants_by_prefix(workchain, "wannier90"))

    for label, child in candidates:
        try:
            return get_workchain_fermi_energy(child), f"child:{label}"
        except (KeyError, ValueError, AttributeError):
            continue

    raise ValueError("no entries found")


def _extract_exclude_bands_from_process(node):
    """Return ``exclude_bands`` from a process input namespace if available."""
    try:
        if "parameters" in node.inputs:
            return node.inputs["parameters"].get_dict().get("exclude_bands", [])
    except (AttributeError, KeyError, TypeError):
        pass

    try:
        return node.inputs["wannier90"]["parameters"].get_dict().get("exclude_bands", [])
    except (AttributeError, KeyError, TypeError):
        pass

    return []


def _get_resolved_exclude_bands(workchain):
    """Get ``exclude_bands`` for bands distance, rescuing from child workchains if needed."""
    from aiida.plugins import WorkflowFactory

    Wannier90OptimizeWorkChain = WorkflowFactory("wannier90_workflows.optimize")

    candidates = []
    if workchain.process_class == Wannier90OptimizeWorkChain:
        candidates.extend(_get_called_descendants_by_prefix(workchain, "wannier90_plot"))
        candidates.extend(_get_called_descendants_by_prefix(workchain, "wannier90_optimize_iteration"))
        candidates.extend(_get_called_descendants_by_prefix(workchain, "wannier90"))
    else:
        candidates.extend(_get_called_descendants_by_prefix(workchain, "wannier90"))

    for _, child in candidates:
        exclude_bands = _extract_exclude_bands_from_process(child)
        if exclude_bands:
            return exclude_bands

    return []


def _get_resolved_wannier_bands(workchain):
    """Return bands node and source for workchains used in bands distance analysis."""
    candidates = _collect_wannier_bands_candidates(workchain)

    if not candidates:
        raise ValueError("no bands outputs found")

    for source, bands in candidates:
        shape_product = _get_bands_shape_product(bands)
        if shape_product == 0:
            continue
        return bands, source

    raise ValueError("all candidate bands outputs are empty")


def _select_best_candidate_by_distance(
    workchain,
    bands_dft_node,
    fermi_energy,
    exclude_list_dft,
):
    """Select the bands candidate with the smallest Ef+2eV distance."""
    from aiida.plugins import WorkflowFactory

    from aiida_wannier90_workflows.utils.bands.distance import bands_distance

    Wannier90OptimizeWorkChain = WorkflowFactory("wannier90_workflows.optimize")

    if workchain.process_class != Wannier90OptimizeWorkChain:
        return _get_resolved_wannier_bands(workchain)

    candidates = _collect_wannier_bands_candidates(workchain)
    if not candidates:
        raise ValueError("no bands outputs found")

    best = None
    best_metric = None

    for source, bands in candidates:
        shape_product = _get_bands_shape_product(bands)
        if shape_product == 0:
            continue

        try:
            dist = bands_distance(bands_dft_node, bands, fermi_energy, exclude_list_dft)
            metric = float(dist[2, 1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue

        if best_metric is None or metric < best_metric:
            best_metric = metric
            best = (bands, source)

    if best is not None:
        return best

    raise ValueError("all candidate bands outputs are unusable")


def _format_diagnosis_message(diagnosis: dict) -> str:
    """Format a compact user-facing diagnosis string."""
    parts = []
    if diagnosis["missing"]:
        parts.append("; ".join(diagnosis["missing"]))
    else:
        parts.append("missing required Wannier bands outputs")
    parts.append(f"rescue={diagnosis['can_rescue']}")
    parts.append(f"fermi_source={diagnosis['fermi_energy_source']}")
    parts.append(f"bands_source={diagnosis['bands_source']}")
    return "; ".join(parts)


def diagnose_bandsdist_workchain(workchain) -> dict:
    """Inspect whether a workchain can be used by ``bands_distance_for_group``.

    Returns a dictionary with resolved data sources and a short rescue summary.
    """
    diagnosis = {
        "pk": workchain.pk,
        "process_label": workchain.process_label,
        "formula": workchain.inputs.structure.get_formula() if "structure" in workchain.inputs else None,
        "fermi_energy_source": None,
        "bands_source": None,
        "can_rescue": False,
        "missing": [],
    }

    try:
        _, source = _get_resolved_fermi_energy(workchain)
        diagnosis["fermi_energy_source"] = source
    except (KeyError, ValueError, AttributeError) as exc:
        diagnosis["missing"].append(f"fermi_energy: {exc}")

    try:
        _, source = _get_resolved_wannier_bands(workchain)
        diagnosis["bands_source"] = source
    except (KeyError, ValueError, AttributeError) as exc:
        diagnosis["missing"].append(f"bands: {exc}")

    diagnosis["can_rescue"] = diagnosis["fermi_energy_source"] is not None and diagnosis["bands_source"] is not None

    return diagnosis


def bands_distance_for_group(  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
    wan_group: ty.Union[orm.Group, str],
    dft_group: ty.Union[orm.Group, str],
    match_by_formula: bool = False,
) -> pd.DataFrame:
    """Calculate bands distance for a group of DFT Calculation and Wannier WorkChain.

    :param wan_group: [description]
    :type wan_group: ty.Union[orm.Group, str]
    :param dft_group: [description]
    :type dft_group: ty.Union[orm.Group, str]
    :return: [description]
    :rtype: pd.DataFrame
    """
    from aiida.plugins import WorkflowFactory

    from aiida_quantumespresso.calculations.pw import PwCalculation
    from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    from aiida_wannier90.calculations import Wannier90Calculation

    from aiida_wannier90_workflows.utils.bands.distance import bands_distance
    from aiida_wannier90_workflows.utils.workflows.group import get_mapping_for_group

    Wannier90BandsWorkChain = WorkflowFactory("wannier90_workflows.bands")
    Wannier90OptimizeWorkChain = WorkflowFactory("wannier90_workflows.optimize")

    if isinstance(wan_group, str):
        wan_group = orm.load_group(wan_group)
    if isinstance(dft_group, str):
        dft_group = orm.load_group(dft_group)

    if wan_group.nodes[0].process_class != Wannier90OptimizeWorkChain:
        mapping = get_mapping_for_group(wan_group, dft_group, match_by_formula)

    columns = [
        "formula",
        "wan_workchain",
        "dft_workchain",
        "fermi_energy",
    ]
    mu_range = range(0, 6)
    columns.extend([f"bands_dist_ef+{mu}" for mu in mu_range])
    columns.extend([f"bands_maxdist_ef+{mu}" for mu in mu_range])
    columns.extend([f"bands_maxdist2_ef+{mu}" for mu in mu_range])
    print(columns)

    result = []
    for wan_wc in wan_group.nodes:
        structure = wan_wc.inputs.structure
        formula = structure.get_formula()

        if wan_wc.process_class == Wannier90OptimizeWorkChain and "optimize_reference_bands" in wan_wc.inputs:
            bands_wc = (
                wan_wc.inputs.optimize_reference_bands.base.links.get_incoming(link_label_filter="band_structure")
                .one()
                .node
            )
        else:
            bands_wc = mapping[wan_wc]

        if bands_wc is None:
            msg = f"! Cannot find DFT bands for {wan_wc.process_label}<{wan_wc.pk}> of {formula}"
            print(msg)
            continue

        if bands_wc.process_class in (PwBaseWorkChain, PwCalculation):
            if "output_band" not in bands_wc.outputs:
                print(f"! Skip DFT {bands_wc.process_label}<{bands_wc.pk}> of {formula}: " "missing output_band")
                continue
            bands_dft_node = bands_wc.outputs.output_band
        elif bands_wc.process_class == PwBandsWorkChain:
            if "band_structure" not in bands_wc.outputs:
                print(f"! Skip DFT {bands_wc.process_label}<{bands_wc.pk}> of {formula}: " "missing band_structure")
                continue
            bands_dft_node = bands_wc.outputs.band_structure
        else:
            raise ValueError(f"Unsupported node type {bands_wc.process_class}<{bands_wc.pk}>")

        if wan_wc.process_class == Wannier90Calculation:
            fermi_energy = wan_wc.inputs.parameters["fermi_energy"]
            bands_wannier_node = wan_wc.outputs.interpolated_bands
            try:
                exclude_list_dft = wan_wc.inputs.parameters["exclude_bands"]
            except KeyError:
                exclude_list_dft = []
        elif wan_wc.process_class in (
            Wannier90BandsWorkChain,
            Wannier90OptimizeWorkChain,
        ):
            try:
                fermi_energy, _fermi_source = _get_resolved_fermi_energy(wan_wc)
            except (KeyError, ValueError, AttributeError) as exc:
                print(
                    f"! Skip {wan_wc.process_label}<{wan_wc.pk}> of {formula}: "
                    f"cannot determine Fermi energy ({exc})"
                )
                continue
            exclude_list_dft = _get_resolved_exclude_bands(wan_wc)
            try:
                bands_wannier_node, _bands_source = _select_best_candidate_by_distance(
                    wan_wc,
                    bands_dft_node,
                    fermi_energy,
                    exclude_list_dft,
                )
            except (KeyError, ValueError, AttributeError) as exc:
                diagnosis = diagnose_bandsdist_workchain(wan_wc)
                print(
                    f"! Skip {wan_wc.process_label}<{wan_wc.pk}> of {formula}: "
                    f"{exc}; {_format_diagnosis_message(diagnosis)}"
                )
                continue

        print(bands_wc.pk, wan_wc.pk, _bands_source)
        dist = bands_distance(bands_dft_node, bands_wannier_node, fermi_energy, exclude_list_dft)

        res = [formula, wan_wc.pk, bands_wc.pk, float(fermi_energy)]
        # bands_dist_ef+{mu}
        res.extend([float(dist[_, 1]) for _ in range(len(mu_range))])
        # bands_maxdist_ef+{mu}
        res.extend([float(dist[_, 2]) for _ in range(len(mu_range))])
        # bands_maxdist2_ef+{mu}
        res.extend([float(dist[_, 3]) for _ in range(len(mu_range))])

        result.append(res)
        print(res)

    dataframe = pd.DataFrame(result, columns=columns)

    return dataframe


def save_distance(distance: pd.DataFrame, hdf_file: str):
    """Save bands distance to a HDF5 file.

    :param distance: [description]
    :type distance: pd.DataFrame
    :param hdf_file: [description]
    :type hdf_file: str
    """
    store = pd.HDFStore(hdf_file)

    # save it
    store["df"] = distance
    store.close()

    print(f"Saved to {hdf_file}")


def read_distance(hdf_file: str) -> pd.DataFrame:
    """Load bands distance stored in a HDF5 file."""
    import os.path

    if not os.path.exists(hdf_file):
        raise ValueError(f"File not existed: {hdf_file}")

    store = pd.HDFStore(hdf_file)

    # load it
    df = store["df"]
    store.close()

    return df


def plot_distance(  # pylint: disable=too-many-locals
    df: pd.DataFrame, max_dist: bool = False, show: bool = True
) -> None:  # pylint: disable=too-many-locals
    """Plot a histogram of bands distance."""
    import matplotlib.pyplot as plt

    # labels are the label for each column of distance
    mu_range = range(0, 6)
    labels = [f"Ef+{i}eV" for i in mu_range]
    # bands distance \eta
    distance = np.zeros((len(df), len(labels)))
    # pks are the wannier workchain PK of each row of eta
    pks = np.zeros(len(df), dtype=int)

    if max_dist:
        # I use the abs distance
        eta_index = "bands_maxdist2_ef+"
    else:
        eta_index = "bands_dist_ef+"

    # Some times the dataframe index is not continous, (after filtering the dataframe),
    # so we need a counter i to index distance[...].
    i = 0
    for _, row in df.iterrows():
        for j, mu in enumerate(mu_range):
            key = f"{eta_index}{mu}"
            distance[i, j] = row[key] * 1e3  # to meV
        pks[i] = row["wan_workchain"]
        i += 1

    fig, axs = plt.subplots(
        len(labels),
        1,
        sharex=True,
        sharey=True,
        # figsize=(10,8.21/6.47*10)
    )

    data_range = (distance.min(), distance.max())
    # print(f'data_range {data_range}')

    for i, lab in enumerate(labels):
        data = distance[:, i]
        label = f"{lab}, aver = {np.average(data):.3f}meV"
        # print(f'data ({data.min()}, {data.max()})')

        axs[i].hist(x=data, bins=100, range=data_range, label=label)
        axs[i].legend()
        # axs[i].grid(True)

    # Add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if max_dist:
        plt.xlabel("eta_max (meV)")
    else:
        plt.xlabel("eta (meV)")
    plt.ylabel("Count")
    plt.title(f"Histogram of {len(pks)} structures")
    # plt.tight_layout()
    # plt.savefig('distances.pdf')
    # plt.savefig('wf_center_xyz/' + 'distances.png')

    if show:
        plt.show()

    return fig
