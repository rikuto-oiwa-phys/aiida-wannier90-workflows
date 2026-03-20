#!/usr/bin/env runaiida
"""Plot histograms of WF-center distances to nearest and next-nearest atoms.

This script generates a two-panel figure similar to:

    Fig. 6 Histogram of the distances of the WF centers from the NN atom
    (red, dNN) and NNN atom (green, dNNN). The inset of each panel shows
    the histogram of the ratio dNN/dNNN.

Examples
--------
    runaiida examples/plot_wf_center_histograms.py \
        --pdwf-group 26 \
        --scdm-group 33 \
        --output fig6_histograms.png
"""

from __future__ import annotations

import argparse
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
from aiida import orm
from aiida.common.links import LinkType
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from aiida_wannier90.calculations import Wannier90Calculation
from aiida_wannier90_workflows.utils.parser.center import get_wf_center_distances
from aiida_wannier90_workflows.utils.workflows import get_last_calcjob
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain


def _load_group(group: ty.Union[int, str, orm.Group]) -> orm.Group:
    """Load a group from PK, UUID, or label."""
    if isinstance(group, orm.Group):
        return group
    try:
        return orm.load_group(int(group))
    except (TypeError, ValueError):
        return orm.load_group(group)


def _iter_called_children(node: orm.Node) -> ty.Iterable[tuple[str, orm.Node]]:
    """Yield direct called children only."""
    for link in node.base.links.get_outgoing().all():
        if link.link_type not in (LinkType.CALL_CALC, LinkType.CALL_WORK):
            continue
        yield link.link_label, link.node


def _resolve_last_wannier_calc(node: orm.Node) -> Wannier90Calculation:
    """Resolve the Wannier90Calculation carrying final WF centers.

    This is resilient to partially failed workchains such as
    ``Wannier90OptimizeWorkChain`` finishing with a plotting failure.
    """
    if isinstance(node, Wannier90Calculation):
        return node

    process_label = getattr(node, "process_label", "")

    if isinstance(node, orm.WorkChainNode):
        candidates: list[tuple[str, orm.Node]] = []

        for label, child in _iter_called_children(node):
            if label == "wannier90_plot":
                candidates.append((label, child))
        for label, child in _iter_called_children(node):
            if label.startswith("wannier90_optimize_iteration"):
                candidates.append((label, child))
        for label, child in _iter_called_children(node):
            if label == "wannier90":
                candidates.append((label, child))

        if "OptimizeWorkChain" in process_label:
            for _label, child in candidates:
                if isinstance(child, Wannier90BaseWorkChain):
                    calc = get_last_calcjob(child)
                    if isinstance(calc, Wannier90Calculation) and "output_parameters" in calc.outputs:
                        return calc

        for label, child in _iter_called_children(node):
            if label == "wannier90":
                if isinstance(child, Wannier90BaseWorkChain):
                    calc = get_last_calcjob(child)
                    if isinstance(calc, Wannier90Calculation):
                        return calc

    raise ValueError(f"Cannot resolve a Wannier90Calculation from node<{node.pk}>")


def collect_distances_for_group(group: ty.Union[int, str, orm.Group]) -> tuple[np.ndarray, np.ndarray]:
    """Collect NN and NNN distances for all usable nodes in a group."""
    group = _load_group(group)

    dist_nn = []
    dist_nnn = []

    for node in group.nodes:
        try:
            calc = _resolve_last_wannier_calc(node)
            d_nn, _, _, _ = get_wf_center_distances(calc, nth_neighbour=1)
            d_nnn, _, _, _ = get_wf_center_distances(calc, nth_neighbour=2)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Skip {node.process_label}<{node.pk}>: {exc}")
            continue

        dist_nn.extend(d_nn)
        dist_nnn.extend(d_nnn)

    return np.array(dist_nn, dtype=float), np.array(dist_nnn, dtype=float)


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
    ax.set_xlabel("Distance / Angstrom")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    inset = inset_axes(ax, width="42%", height="42%", loc="upper left")
    inset.hist(ratio, bins=max(20, bins // 2), color="0.35", alpha=0.85)
    inset.set_xlabel(r"$d_{\mathrm{NN}}/d_{\mathrm{NNN}}$", fontsize=8)
    inset.set_ylabel("Count", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.2)

    summary = "\n".join(
        [
            f"N = {len(dist_nn)}",
            f"<dNN> = {np.mean(dist_nn):.3f} A",
            f"<dNNN> = {np.mean(dist_nnn):.3f} A",
            f"<dNN/dNNN> = {np.mean(ratio):.3f}",
        ]
    )
    inset.text(
        0.98,
        0.02,
        summary,
        transform=inset.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )


def plot_histograms(
    pdwf_nn: np.ndarray,
    pdwf_nnn: np.ndarray,
    scdm_nn: np.ndarray,
    scdm_nnn: np.ndarray,
    output: str,
    bins: int,
) -> None:
    """Generate and save the two-panel comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    _plot_panel(axes[0], pdwf_nn, pdwf_nnn, "a  PDWF", bins)
    _plot_panel(axes[1], scdm_nn, scdm_nnn, "b  SCDM", bins)

    fig.suptitle("Histogram of WF-center distances to NN and NNN atoms", fontsize=14)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdwf-group", required=True, help="AiiDA group PK/UUID/label for PDWF results.")
    parser.add_argument("--scdm-group", required=True, help="AiiDA group PK/UUID/label for SCDM results.")
    parser.add_argument(
        "-o",
        "--output",
        default="wf_center_histograms.png",
        help="Output figure filename.",
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
    args = parse_args()

    pdwf_nn, pdwf_nnn = collect_distances_for_group(args.pdwf_group)
    scdm_nn, scdm_nnn = collect_distances_for_group(args.scdm_group)

    if len(pdwf_nn) == 0 or len(scdm_nn) == 0:
        raise RuntimeError("No WF-center distances were collected from one or both groups.")

    plot_histograms(
        pdwf_nn=pdwf_nn,
        pdwf_nnn=pdwf_nnn,
        scdm_nn=scdm_nn,
        scdm_nnn=scdm_nnn,
        output=args.output,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
