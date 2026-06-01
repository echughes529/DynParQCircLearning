

"""
Compute the pure-state Quantum Fisher Information (QFI) directly with
TensorCircuit/JAX for a small unitary toric-code ansatz.

Run from the repository root with:

    python -m src.examples.compute_qfi_tc

This script deliberately starts with the simplest case:
    - 2x2 toric-code lattice
    - 1 ansatz layer
    - unitary ansatz only
    - no probabilistic resets

This does not need to be inside the training loop yet. QFI is expensive
because it is an nparams x nparams matrix. The sensible first step is to
compute it for one small ansatz instance, inspect the diagnostics, and only
later decide whether to track it occasionally during training.

For a pure parameterized state |psi(theta)>, the QFI is

    F_ij = 4 Re[ <d_i psi | d_j psi>
                 - <d_i psi | psi><psi | d_j psi> ].

This formula is directly implemented below using jax.jacobian.
"""

from __future__ import annotations

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax
import jax.numpy as jnp

import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.generate_toric_code_hamiltonian import ToricCode


# Use JAX backend if available, matching the rest of the DPQC codebase.
# If this fails on a particular environment, TensorCircuit will raise the
# relevant backend error immediately.
tc.set_backend("jax")

Lx=2
Ly=2
nlayers=2 
howoften_toreset=1
h=0.0
use_prob_resets=True

which_qubits_for_prob_reset=None
unitary=True
use_small_angle_initialization=False
range_initial_parameters=0.0
compute_qfi_eigenvalues=False

def build_small_unitary_ansatz() -> ToricCodeAnsatz:
    """Create the smallest useful toric-code ansatz for Qiskit export tests."""

    return ToricCodeAnsatz(
        Lx=Lx,
        Ly=Ly,
        nlayers=nlayers,
        howoften_toreset=howoften_toreset,
        h=h,
        use_prob_resets=use_prob_resets,
        which_qubits_for_prob_reset=which_qubits_for_prob_reset,
        unitary=unitary,
        use_small_angle_initialization=use_small_angle_initialization,
        range_initial_parameters=range_initial_parameters,
    )


def print_unitary_toric_parameter_breakdown(ansatz: ToricCodeAnsatz) -> None:
    """
    Print where the parameters come from for construct_unitary_circuit_toriccodelattice.

    This mirrors the counting logic in src.utilities.generate_ansatz:

        nparams = nplaquettes * 4 * 9 * nlayers + 3 * nq

    where each Cartan block consumes:
        - 3 parameters for a U3/Ry-Rz-Ry on the first qubit
        - 3 parameters for a U3/Ry-Rz-Ry on the second qubit
        - 3 parameters for RXX, RYY, RZZ
      so 9 parameters total per two-qubit block.
    """

    Lx = ansatz.Lx
    Ly = ansatz.Ly
    nlayers = ansatz.nlayers

    toriccode = ToricCode(Lx, Ly)
    nplaquettes = (Lx - 1) * (Ly - 1)
    nq = 2 * Lx * Ly - Lx - Ly

    raw_claws = toriccode.all_claws_unitaries()
    reordered_claws = [
        raw_claws[i::4][j]
        for i in range(4)
        for j in range((Lx - 1) * (Ly - 1))
    ]

    params_per_cartan_block = 9
    params_per_final_single_qubit_layer = 3 * nq
    cartan_blocks_per_layer = len(reordered_claws)
    cartan_params_per_layer = cartan_blocks_per_layer * params_per_cartan_block
    total_cartan_params = nlayers * cartan_params_per_layer
    expected_total = total_cartan_params + params_per_final_single_qubit_layer

    print("\nParameter-count diagnostic for unitary toric-code ansatz:")
    print(f"  Lx = {Lx}")
    print(f"  Ly = {Ly}")
    print(f"  nq = 2*Lx*Ly - Lx - Ly = {nq}")
    print(f"  nplaquettes = (Lx-1)*(Ly-1) = {nplaquettes}")
    print(f"  nlayers = {nlayers}")
    print(f"  len(raw all_claws_unitaries()) = {len(raw_claws)}")
    print(f"  len(reordered_claws actually used) = {len(reordered_claws)}")

    print("\n  Raw claws from toriccode.all_claws_unitaries():")
    for idx, claw in enumerate(raw_claws):
        print(f"    raw_claws[{idx}] = {claw}")

    print("\n  Reordered claws used by construct_unitary_circuit_toriccodelattice:")
    for idx, claw in enumerate(reordered_claws):
        print(f"    reordered_claws[{idx}] = {claw}")

    print("\n  Per Cartan block:")
    print("    first-qubit Ry/Rz/Ry params = 3")
    print("    second-qubit Ry/Rz/Ry params = 3")
    print("    RXX/RYY/RZZ params = 3")
    print(f"    total per Cartan block = {params_per_cartan_block}")

    print("\n  Count:")
    print(
        f"    Cartan block params = {nlayers} layer(s) "
        f"* {cartan_blocks_per_layer} block(s)/layer "
        f"* {params_per_cartan_block} params/block = {total_cartan_params}"
    )
    print(
        f"    final single-qubit layer params = {nq} qubit(s) "
        f"* 3 params/qubit = {params_per_final_single_qubit_layer}"
    )
    print(f"    expected total = {total_cartan_params} + {params_per_final_single_qubit_layer} = {expected_total}")
    print(f"    ansatz.nparams = {ansatz.nparams}")

    if expected_total != ansatz.nparams:
        print("\n  WARNING: diagnostic count does not match ansatz.nparams")
    else:
        print("\n  Diagnostic count matches ansatz.nparams.")




def get_tensorcircuit_from_ansatz(ansatz: ToricCodeAnsatz, params: jnp.ndarray) -> tc.Circuit:
    """
    Build the TensorCircuit circuit from the ansatz.

    Different versions of the codebase may expose the circuit builder under
    slightly different names. This helper tries the common internal method
    first, then falls back to a public method if one exists.
    """

    if hasattr(ansatz, "_circuit"):
        return ansatz._circuit(params)

    if hasattr(ansatz, "circuit"):
        return ansatz.circuit(params)

    raise AttributeError(
        "Could not find a circuit-building method on ToricCodeAnsatz. "
        "Expected either ansatz._circuit(params) or ansatz.circuit(params)."
    )


def state_fn(ansatz: ToricCodeAnsatz, params: jnp.ndarray) -> jnp.ndarray:
    """Return the statevector produced by the ansatz for the given parameters."""

    circuit = get_tensorcircuit_from_ansatz(ansatz, params)
    state = circuit.state()
    return jnp.reshape(state, (-1,)) # ensuring statevector is a flat 1d vector


def compute_qfi(ansatz: ToricCodeAnsatz, params: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the pure-state QFI matrix using JAX autodifferentiation.

    The output has shape (nparams, nparams).
    """

    def wrapped_state_fn(x: jnp.ndarray) -> jnp.ndarray:
        return state_fn(ansatz, x)

    psi = wrapped_state_fn(params)
    state_dim = psi.shape[0]

    # JAX's default jacobian rules require real-valued outputs. Since the
    # statevector is complex-valued but depends on real parameters, differentiate
    # the real and imaginary parts separately, then reconstruct d|psi>/dtheta.
    def wrapped_real_state_fn(x: jnp.ndarray) -> jnp.ndarray:
        psi_x = wrapped_state_fn(x)
        return jnp.concatenate([jnp.real(psi_x), jnp.imag(psi_x)])

    # Shape: (2 * state_dimension, nparams).
    jac_real = jax.jacobian(wrapped_real_state_fn)(params)

    # Convert to shape: (nparams, state_dimension), where
    # derivative_states[i, a] = d psi_a / d theta_i.
    derivative_states = jnp.transpose(
        jac_real[:state_dim, :] + 1.0j * jac_real[state_dim:, :],
        (1, 0),
    )

    # term1[i, j] = <d_i psi | d_j psi>
    term1 = jnp.einsum("ia,ja->ij", jnp.conj(derivative_states), derivative_states)

    # overlap[i] = <d_i psi | psi>
    overlap = jnp.einsum("ia,a->i", jnp.conj(derivative_states), psi)

    qgt = term1 - jnp.outer(overlap, jnp.conj(overlap))
    qfi = 4.0 * jnp.real(qgt)

    # Symmetrize to remove tiny numerical asymmetries.
    return 0.5 * (qfi + qfi.T)


def print_qfi_diagnostics(
    qfi: jnp.ndarray,
    eigval_tol: float = 1e-8,
    compute_eigenvalues: bool = True,
) -> None:
    """Print compact diagnostics for the QFI matrix."""
    trace = float(jnp.trace(qfi))

    print("\nQFI diagnostics:")
    print(f"  QFI shape = {qfi.shape}")
    print(f"  trace(QFI) = {trace:.12e}")

    if not compute_eigenvalues:
        print("  Eigenvalue diagnostics skipped")
        return

    eigvals = jnp.linalg.eigvalsh(qfi)
    positive_eigvals = eigvals[eigvals > eigval_tol]

    rank = int(jnp.sum(eigvals > eigval_tol))

    if positive_eigvals.size == 0:
        condition_number = np.inf
    else:
        condition_number = float(jnp.max(positive_eigvals) / jnp.min(positive_eigvals))

    print(f"  rank(QFI), tol={eigval_tol:g} = {rank}")
    print(f"  min eigenvalue = {float(jnp.min(eigvals)):.12e}")
    print(f"  max eigenvalue = {float(jnp.max(eigvals)):.12e}")
    print(f"  condition number over positive eigenvalues = {condition_number:.12e}")

    print("\nQFI eigenvalues:")
    print(np.asarray(eigvals))


# ---- Save QFI outputs ----
def save_qfi_outputs(
    qfi: jnp.ndarray,
    outdir: str,
    eigval_tol: float = 1e-12,
    compute_eigenvalues: bool = True,
) -> None:
    """Save the QFI matrix and optionally compute/save eigenvalues."""

    os.makedirs(outdir, exist_ok=True)

    qfi_np = np.asarray(qfi)

    qfi_path = os.path.join(outdir, "qfi_matrix.csv")
    eigvals_path = os.path.join(outdir, "qfi_eigenvalues.csv")
    plot_path = os.path.join(outdir, "qfi_eigenvalue_spectrum.png")

    np.savetxt(qfi_path, qfi_np, delimiter=",")

    print("\nSaved QFI outputs:")
    print(f"  QFI matrix: {qfi_path}")

    if not compute_eigenvalues:
        print("  QFI eigenvalue computation skipped")
        return

    eigvals = np.asarray(jnp.linalg.eigvalsh(qfi))
    np.savetxt(eigvals_path, eigvals, delimiter=",", header="qfi_eigenvalue", comments="")

    # QFI should be positive semidefinite, so tiny negative values are numerical
    # noise. Clip only for plotting on a log scale.
    eigvals_for_plot = np.clip(eigvals, eigval_tol, None)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(eigvals_for_plot) + 1), eigvals_for_plot, marker="o")
    plt.yscale("log")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("QFI eigenvalue")
    plt.title(f"QFI eigenvalue spectrum, {Lx}x{Ly}, Resets: {use_prob_resets}")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"  QFI eigenvalues: {eigvals_path}")
    print(f"  QFI eigenvalue spectrum plot: {plot_path}")


def save_qfi_heatmap(qfi: jnp.ndarray, outdir: str, eigval_tol: float = 1e-12) -> None:
    """Save a heatmap of the magnitudes of the QFI matrix entries."""

    os.makedirs(outdir, exist_ok=True)

    qfi_np = np.asarray(qfi)
    abs_qfi_np = np.abs(qfi_np)

    heatmap_path = os.path.join(outdir, "qfi_matrix_magnitude_heatmap.png")

    max_abs_entry = float(np.max(abs_qfi_np))
    if max_abs_entry == 0.0:
        max_abs_entry = eigval_tol

    # Magnitude log heatmap: useful for seeing the structure of QFI entries,
    # especially when many entries differ by orders of magnitude.
    abs_qfi_for_plot = np.clip(abs_qfi_np, eigval_tol, None)
    plt.figure(figsize=(7, 6))
    plt.imshow(
        abs_qfi_for_plot,
        aspect="auto",
        origin="lower",
        norm=LogNorm(vmin=eigval_tol, vmax=max_abs_entry),
    )
    plt.colorbar(label="|QFI matrix entry|")
    plt.xlabel("Parameter index")
    plt.ylabel("Parameter index")
    plt.title(f"QFI matrix magnitude heatmap, {Lx}x{Ly}, Resets: {use_prob_resets}")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    print("\nSaved QFI heatmap:")
    print(f"  QFI matrix magnitude heatmap: {heatmap_path}")


def main() -> None:
    ansatz = build_small_unitary_ansatz()

    print("Created ansatz:")
    print(f"  Lx = {ansatz.Lx}")
    print(f"  Ly = {ansatz.Ly}")
    print(f"  nlayers = {ansatz.nlayers}")
    print(f"  unitary = {ansatz.unitary}")
    print(f"  use_prob_resets = {ansatz.use_prob_resets}")
    print(f"  nparams = {ansatz.nparams}")
    print(f"  nancillas = {ansatz.nancillas}")
    print_unitary_toric_parameter_breakdown(ansatz)

    # Use fixed numerical parameters for this first QFI test.
    params_np = np.random.default_rng(seed=123).uniform(0.0, np.pi, size=ansatz.nparams)
    params = jnp.asarray(params_np)

    psi = state_fn(ansatz, params)
    print("\nBuilt TensorCircuit statevector.")
    print(f"  state dimension = {psi.shape[0]}")
    print(f"  state norm = {float(jnp.linalg.norm(psi)):.12e}")

    print("\nComputing QFI with JAX autodifferentiation...")
    qfi = compute_qfi(ansatz, params)
    print_qfi_diagnostics(qfi, compute_eigenvalues=compute_qfi_eigenvalues)

    outdir = os.environ.get("DPQC_OUTDIR", "logs/qfi_outputs")
    save_qfi_outputs(qfi, outdir=outdir, compute_eigenvalues=compute_qfi_eigenvalues)
    save_qfi_heatmap(qfi, outdir=outdir)


if __name__ == "__main__":
    main()