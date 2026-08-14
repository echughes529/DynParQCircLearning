"""Small, NumPy-only summaries for MPS numerical diagnostics."""

from typing import Any, Dict, Sequence

import numpy as np


def magnitude_summary(tensor: Any) -> Dict[str, Any]:
    """Summarize absolute entry magnitudes without printing a whole tensor.

    Quantiles are calculated over finite values (including exact zeros), while
    ``min_nonzero`` and the small-value fractions use finite nonzero entries.
    ``fraction_below_relative_epsilon`` flags entries that are tiny relative to
    the largest entry in the same tensor, which is usually more informative
    than the dtype's absolute underflow threshold alone.
    """
    magnitude = np.abs(np.asarray(tensor)).reshape(-1)
    finite_mask = np.isfinite(magnitude)
    finite = magnitude[finite_mask]
    nonzero = finite[finite > 0]

    # np.abs(complex64/128) gives float32/64, so this also selects the
    # appropriate real floating-point limits for complex tensors.
    real_dtype = magnitude.dtype
    if not np.issubdtype(real_dtype, np.floating):
        real_dtype = np.dtype(np.float64)
    finfo = np.finfo(real_dtype)

    summary: Dict[str, Any] = {
        "size": int(magnitude.size),
        "nonfinite_count": int((~finite_mask).sum()),
        "zero_count": int((finite == 0).sum()),
        "min_nonzero": np.nan,
        "q0_1_percent": np.nan,
        "q1_percent": np.nan,
        "median": np.nan,
        "max": np.nan,
        "dynamic_range": np.nan,
        "fraction_below_tiny": 0.0,
        "fraction_below_relative_epsilon": 0.0,
    }

    if finite.size:
        quantiles = np.quantile(finite, [0.001, 0.01, 0.5])
        summary.update(
            q0_1_percent=float(quantiles[0]),
            q1_percent=float(quantiles[1]),
            median=float(quantiles[2]),
            max=float(finite.max()),
        )

    if nonzero.size:
        min_nonzero = float(nonzero.min())
        max_value = float(finite.max())
        summary.update(
            min_nonzero=min_nonzero,
            dynamic_range=(max_value / min_nonzero if max_value > 0 else np.nan),
            fraction_below_tiny=float(np.mean(nonzero < finfo.tiny)),
            fraction_below_relative_epsilon=float(
                np.mean(nonzero < max_value * finfo.eps)
            ),
        )

    return summary


def singular_value_summary(spectra: Sequence[Any]) -> Dict[str, Any]:
    """Summarize the smallest scales and worst conditioning across MPS cuts."""
    min_nonzero = np.inf
    max_condition = 0.0
    zero_count = 0
    nonfinite_count = 0
    populated_cuts = 0

    for spectrum in spectra:
        values = np.abs(np.asarray(spectrum)).reshape(-1)
        nonfinite_count += int((~np.isfinite(values)).sum())
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue

        populated_cuts += 1
        zero_count += int((finite == 0).sum())
        nonzero = finite[finite > 0]
        if nonzero.size:
            min_nonzero = min(min_nonzero, float(nonzero.min()))

        largest = float(finite.max())
        if largest == 0:
            condition = np.inf
        elif zero_count and np.any(finite == 0):
            condition = np.inf
        else:
            condition = largest / float(nonzero.min())
        max_condition = max(max_condition, condition)

    return {
        "populated_cuts": populated_cuts,
        "min_nonzero": float(min_nonzero) if np.isfinite(min_nonzero) else np.nan,
        "max_condition": float(max_condition) if populated_cuts else np.nan,
        "zero_count": zero_count,
        "nonfinite_count": nonfinite_count,
    }


def format_magnitude_summary(summary: Dict[str, Any]) -> str:
    """Format ``magnitude_summary`` for compact training logs."""
    size = max(summary["size"], 1)
    return (
        f"min_nonzero={summary['min_nonzero']:.3e}  "
        f"q0.1%={summary['q0_1_percent']:.3e}  "
        f"q1%={summary['q1_percent']:.3e}  "
        f"median={summary['median']:.3e}  max={summary['max']:.3e}  "
        f"dynamic_range={summary['dynamic_range']:.3e}  "
        f"zeros={summary['zero_count'] / size:.2%}  "
        f"below_tiny={summary['fraction_below_tiny']:.2%}  "
        f"below_relative_eps={summary['fraction_below_relative_epsilon']:.2%}  "
        f"nonfinite={summary['nonfinite_count'] / size:.2%}"
    )
