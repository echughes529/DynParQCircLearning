# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Shared plotting for backward-lightcone diagrams.

Deliberately not a real circuit diagram (no gate symbols/boxes with labels) -
just qubit "wires" (horizontal lines) with a marker/vertical-line per gate in
its application order, colored by whether that gate survived a given term's
lightcone reduction or was pruned away. Used by the custom tracer
(`lightcone_analysis.py`), which already knows exactly which of its recorded
tape entries were kept.
"""

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEPT_COLOR = "#d62728"    # red - this gate is in the term's backward lightcone
DROPPED_COLOR = "0.8"     # light grey - pruned away, doesn't affect this term


def _draw_wires_and_gates(ax, n_qubits: int, gate_qubits: Sequence[Sequence[int]]) -> None:
    """Shared setup: horizontal qubit wires, spanning the full gate-order axis."""
    n_gates = len(gate_qubits)
    for q in range(n_qubits):
        ax.hlines(q, -0.5, n_gates - 0.5, color="0.9", zorder=0, linewidth=1)


def plot_lightcone(
    n_qubits: int,
    gate_qubits: Sequence[Sequence[int]],
    kept_mask: Sequence[bool],
    target_qubits: Sequence[int],
    title: str = "",
    save_path: Optional[str] = None,
):
    """
    Draw one lightcone diagram.

    n_qubits: total qubit count (y-axis range).
    gate_qubits: one entry per gate, in circuit-application order - the list
        of qubit indices that gate touches (1 for single-qubit gates, 2 or 3
        for multi-qubit gates).
    kept_mask: same length as `gate_qubits` - True where that gate survived
        the lightcone reduction for the term being plotted.
    target_qubits: the Hamiltonian term's own qubits (the observable's
        support) - marked with a black star at the right edge, since these
        are the seed the backward walk started from.
    title: plot title (e.g. the term's label).
    save_path: if given, saves a PNG there (and closes the figure) instead of
        returning the live figure.
    """
    n_gates = len(gate_qubits)
    fig, ax = plt.subplots(figsize=(max(6.0, n_gates * 0.15), max(3.0, n_qubits * 0.4)))
    _draw_wires_and_gates(ax, n_qubits, gate_qubits)

    for i, (qubits, kept) in enumerate(zip(gate_qubits, kept_mask)):
        color = KEPT_COLOR if kept else DROPPED_COLOR
        zorder = 3 if kept else 1
        linewidth = 2.5 if kept else 1.0
        markersize = 6 if kept else 3
        if len(qubits) > 1:
            ax.vlines(i, min(qubits), max(qubits), color=color, linewidth=linewidth, zorder=zorder)
        for q in qubits:
            ax.plot(i, q, "o", color=color, markersize=markersize, zorder=zorder + 1)

    for q in target_qubits:
        ax.plot(n_gates - 0.5, q, marker="*", color="black", markersize=16, zorder=5)

    ax.set_yticks(range(n_qubits))
    ax.set_ylabel("qubit")
    ax.set_xlabel("gate order (circuit time →)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlim(-1, n_gates)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig
