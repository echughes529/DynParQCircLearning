# NaN / gradient-corruption investigation

Reproduction scripts for `NAN_INVESTIGATION.md` (repo root). Every number quoted in that
report is produced by one of these; each script names what it proves in its docstring.

Run them from the repository root, e.g.

```bash
python tests/nan_investigation/t17_rzz_degeneracy_scan.py             # the main result
DT=complex128 python tests/nan_investigation/t17_rzz_degeneracy_scan.py
```

Start here:

| script | what it shows | runtime |
|---|---|---|
| `t17_rzz_degeneracy_scan.py` | gradient error vs distance to the SVD degeneracy, complex64 vs complex128 | ~1 min |
| `t24_epsilon_sweep.py` | that `_safe_reciprocal`'s `eps=1e-15` is the knob; instrumented backward pass | ~2 min |
| `t22_degeneracies_and_fixes.py` | both degeneracies, and that the proposed fixes work | ~2 min |
| `t21_train_track.py` | a real training run, gradient checked against the no-split reference every step | ~15 min (XLA compile dominates) |

`fastansatz.py` is an in-place rebuild of the repo's toric-code circuit builders that
avoids the O(M^2) `qc.append` rebuild; `t06_fast_equivalence.py` proves it is
bit-identical to `src/utilities/generate_ansatz.py`.

Environment knobs used by several scripts: `DT` / `MODE` (precision), `LX`, `LY`, `NL`,
`HOW`, `TRIALS`, `STEPS`, `LR`, `SEED`.

Tested with `tensorcircuit-ng 1.9.1`, `jax 0.10.2`, CPU.
