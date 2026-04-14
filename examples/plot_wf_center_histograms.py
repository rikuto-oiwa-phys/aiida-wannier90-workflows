#!/usr/bin/env runaiida
"""Plot histograms of WF-center distances to nearest and next-nearest atoms.

This script generates one histogram figure per group. Each figure shows
the distances of the WF centers from the NN atom (red, dNN) and NNN atom
(green, dNNN). The inset shows the histogram of the ratio dNN/dNNN.

Examples
--------
    runaiida examples/plot_wf_center_histograms.py \
        --group SCDM=paper/wannier/scdm200/scdm_20260322
"""

import argparse
import typing as ty
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aiida import load_profile, orm
from aiida.common.links import LinkType
from aiida_wannier90.calculations import Wannier90Calculation
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from aiida_wannier90_workflows.utils.parser.center import (
    get_last_wan_calc,
    get_wf_center_distances,
)
from aiida_wannier90_workflows.utils.workflows import get_last_calcjob


plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 13,
    }
)


def _thousands_formatter(value, _pos):
    """Format ticks in units of 10^3."""
    return f"{value / 1000:g}"


def _load_group(group: ty.Union[int, str, orm.Group]) -> orm.Group:
    """Load a group from PK, UUID, or label."""
    if isinstance(group, orm.Group):
        return group
    try:
        return orm.load_group(int(group))
    except (TypeError, ValueError):
        return orm.load_group(group)


def sanitize_group_label(label: str) -> str:
    """Convert group label to a filesystem-friendly directory name."""
    return label.replace("/", "_")


def _is_wannier90_calc(node: orm.Node) -> bool:
    """Return whether a node corresponds to Wannier90Calculation."""
    if not isinstance(node, orm.CalcJobNode):
        return False

    process_class = getattr(node, "process_class", None)
    if process_class is Wannier90Calculation:
        return True

    process_label = getattr(node, "process_label", "")
    return process_label == "Wannier90Calculation"


def _is_usable_wannier_calc(calc: orm.Node) -> bool:
    """Return whether the calc has the outputs needed for WF-center distances."""
    if not _is_wannier90_calc(calc):
        return False

    try:
        output_parameters = calc.outputs.output_parameters
    except (AttributeError, KeyError, ValueError):
        return False

    try:
        wf_outputs = output_parameters["wannier_functions_output"]
    except (TypeError, KeyError, ValueError):
        return False

    try:
        if len(wf_outputs) == 0:
            return False
        return all("wf_centres" in wf for wf in wf_outputs)
    except (TypeError, KeyError, ValueError):
        return False


def _has_output_parameters(calc: orm.Node) -> bool:
    """Return whether `output_parameters` can be safely accessed."""
    try:
        _ = calc.outputs.output_parameters
    except (AttributeError, KeyError, ValueError):
        return False
    return True


def _has_usable_wf_center_distances(calc: orm.Node) -> bool:
    """Return whether NN/NNN distances can actually be computed for this calc."""
    if not _is_usable_wannier_calc(calc):
        return False

    try:
        get_wf_center_distances(calc, nth_neighbour=1)
        get_wf_center_distances(calc, nth_neighbour=2)
    except Exception:
        return False

    return True


def _iter_wannier_calc_candidates(node: orm.Node) -> ty.Iterable[orm.CalcJobNode]:
    """Yield Wannier90Calculation candidates in preference order."""
    seen = set()

    def add_candidate(candidate):
        if not _is_wannier90_calc(candidate):
            return
        if candidate.pk in seen:
            return
        seen.add(candidate.pk)
        return candidate

    if _is_wannier90_calc(node):
        candidate = add_candidate(node)
        if candidate is not None:
            yield candidate
        return

    if isinstance(node, orm.WorkChainNode):
        try:
            candidate = add_candidate(get_last_wan_calc(node))
            if candidate is not None:
                yield candidate
        except Exception:
            pass

        try:
            last_calc = get_last_calcjob(node)
            candidate = add_candidate(last_calc)
            if candidate is not None:
                yield candidate
        except Exception:
            pass

        descendants = []
        for child in node.called_descendants:
            if _is_wannier90_calc(child):
                descendants.append(child)

        descendants.sort(
            key=lambda calc: (
                _has_usable_wf_center_distances(calc),
                _is_usable_wannier_calc(calc),
                _has_output_parameters(calc),
                calc.pk,
            ),
        )

        for calc in reversed(descendants):
            candidate = add_candidate(calc)
            if candidate is not None:
                yield candidate

        for link in node.base.links.get_outgoing(
            link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK),
            link_label_filter="wannier90",
        ).all():
            child = link.node
            if isinstance(child, orm.WorkChainNode):
                try:
                    candidate = add_candidate(get_last_calcjob(child))
                    if candidate is not None:
                        yield candidate
                except Exception:
                    pass


def _resolve_last_wannier_calc(node: orm.Node) -> Wannier90Calculation:
    """Resolve a Wannier90Calculation carrying usable WF-center outputs.

    This is resilient to partially failed workchains such as
    ``Wannier90OptimizeWorkChain`` finishing with a plotting failure.
    """
    fallback = None
    for calc in _iter_wannier_calc_candidates(node):
        if fallback is None:
            fallback = calc
        if _has_usable_wf_center_distances(calc):
            return calc

    if fallback is not None:
        raise ValueError(f"Resolved Wannier90Calculation<{fallback.pk}> but it does not have usable WF-center outputs")

    raise ValueError(f"Cannot resolve a Wannier90Calculation from node<{node.pk}>")


def collect_distances_for_group(
    group: ty.Union[int, str, orm.Group],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float, float]]]:
    """Collect NN and NNN distances plus a per-WF table for all usable nodes in a group."""
    group = _load_group(group)

    dist_nn = []
    dist_nnn = []
    records: list[tuple[int, int, float, float]] = []

    for node in group.nodes:
        try:
            calc = _resolve_last_wannier_calc(node)
            d_nn, _, _, _ = get_wf_center_distances(calc, nth_neighbour=1)
            d_nnn, _, _, _ = get_wf_center_distances(calc, nth_neighbour=2)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Skip {node.process_label}<{node.pk}>: {exc}")
            continue

        nvals = min(len(d_nn), len(d_nnn))
        for index in range(nvals):
            nn_value = float(d_nn[index])
            nnn_value = float(d_nnn[index])
            dist_nn.append(nn_value)
            dist_nnn.append(nnn_value)
            records.append((node.pk, index + 1, nn_value, nnn_value))

    return np.array(dist_nn, dtype=float), np.array(dist_nnn, dtype=float), records


def _plot_panel(ax, dist_nn: np.ndarray, dist_nnn: np.ndarray, title: str, bins: int) -> None:
    """Plot one histogram panel with inset ratio histogram."""
    ratio = dist_nn / dist_nnn
    ratio = ratio[np.isfinite(ratio)]

    ax.hist(
        dist_nn,
        bins=bins,
        color="tab:red",
        alpha=0.55,
        label=r"$d_{\mathrm{NN}}$",
    )
    ax.hist(
        dist_nnn,
        bins=bins,
        color="tab:green",
        alpha=0.45,
        label=r"$d_{\mathrm{NNN}}$",
    )
    ax.set_title(title)
    ax.set_xlabel(r"$d$ (${\rm \AA}$)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 5000)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands_formatter))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.0, 1.02, r"$\times 10^3$", transform=ax.transAxes, ha="left", va="bottom")

    inset = inset_axes(
        ax,
        width="58%",
        height="58%",
        loc="upper right",
        bbox_to_anchor=(0.0, -0.06, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.8,
    )
    inset.hist(
        ratio,
        bins=max(20, bins // 2),
        color="tab:blue",
        alpha=0.85,
    )
    inset.set_xlim(0, 1)
    inset.set_xlabel(r"$d_{\mathrm{NN}}/d_{\mathrm{NNN}}$", fontsize=8)
    inset.set_ylabel("Count", fontsize=8)
    inset.set_ylim(0, 2000)
    inset.yaxis.set_major_formatter(FuncFormatter(_thousands_formatter))
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.2)
    inset.text(
        0.0,
        1.02,
        r"$\times 10^3$",
        transform=inset.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )

    summary = "\n".join(
        [
            f"$d_{{\\rm NN}}$ = {float(np.mean(dist_nn)):.3f}",
            f"$d_{{\\rm NNN}}$ = {float(np.mean(dist_nnn)):.3f}",
            f"$d_{{\\rm NN}}/d_{{\\rm NNN}}$ = {float(np.mean(ratio)):.3f}",
        ]
    )
    inset.text(
        0.98,
        0.5,
        summary,
        transform=inset.transAxes,
        ha="right",
        va="center",
        fontsize=8,
        multialignment="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )


def plot_histogram(
    name: str,
    dist_nn: np.ndarray,
    dist_nnn: np.ndarray,
    output_dir: Path,
    bins: int,
) -> None:
    """Generate and save a single histogram figure for one group."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    _plot_panel(ax, dist_nn, dist_nnn, name, bins)

    output = output_dir / "wf_center_distance_histogram.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output}")


def save_distances_table(records: list[tuple[int, int, float, float]], output_dir: Path) -> None:
    """Save per-WF NN/NNN distances as a plain-text table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / "wf_center_distances.txt"
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# parent_pk  wf_index  dNN  dNNN\n")
        for pk, wf_index, dnn, dnnn in records:
            handle.write(f"{pk:8d}  {wf_index:8d}  {dnn:16.8f}  {dnnn:16.8f}\n")

    print(f"Saved table to {output}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdwf-group",
        required=False,
        help="AiiDA group PK/UUID/label for PDWF results.",
    )
    parser.add_argument(
        "--scdm-group",
        required=False,
        help="AiiDA group PK/UUID/label for SCDM results.",
    )
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        default=[],
        metavar="NAME=GROUP",
        help="Group to process, e.g. --group SCDM=paper/wannier/scdm200/scdm_20260322",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def main():
    """Run the script."""
    load_profile()
    args = parse_args()

    groups: dict[str, str] = {}

    if args.pdwf_group:
        groups["PDWF"] = args.pdwf_group
    if args.scdm_group:
        groups["SCDM"] = args.scdm_group

    for item in args.groups:
        if "=" not in item:
            raise ValueError(f"Invalid --group value: {item!r}. Expected NAME=GROUP.")
        name, group_label = item.split("=", 1)
        groups[name.strip()] = group_label.strip()

    if not groups:
        raise ValueError("Specify at least one group with --pdwf-group, --scdm-group, or --group.")

    for name, group_label in groups.items():
        dist_nn, dist_nnn, records = collect_distances_for_group(group_label)
        if len(dist_nn) == 0:
            print(f"No WF-center distances found in group: {group_label}")
            continue

        output_dir = Path(f"{sanitize_group_label(group_label)}_distance_figures")
        plot_histogram(name, dist_nn, dist_nnn, output_dir, args.bins)
        save_distances_table(records, output_dir)

        ratio = dist_nn / dist_nnn
        ratio = ratio[np.isfinite(ratio)]

        print(f"Found {len(dist_nn)} WF centers for {name}")
        print(f"Mean dNN        = {float(np.mean(dist_nn)):.6f}")
        print(f"Mean dNNN       = {float(np.mean(dist_nnn)):.6f}")
        print(f"Mean dNN/dNNN   = {float(np.mean(ratio)):.6f}")


if __name__ == "__main__":
    main()
