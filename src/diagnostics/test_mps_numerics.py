"""Tests for the lightweight MPS numerical summaries."""

import numpy as np

from src.diagnostics.mps_numerics import magnitude_summary, singular_value_summary


def test_magnitude_summary_handles_complex_zeros_and_nonfinite_values():
    tensor = np.array([0.0, 1e-12, 2.0, np.nan + 0j], dtype=np.complex128)

    summary = magnitude_summary(tensor)

    assert summary["size"] == 4
    assert summary["zero_count"] == 1
    assert summary["nonfinite_count"] == 1
    assert summary["min_nonzero"] == 1e-12
    assert summary["max"] == 2.0
    assert summary["dynamic_range"] == 2e12


def test_magnitude_summary_detects_subnormal_values():
    subnormal = np.nextafter(np.float64(0), np.float64(1))

    summary = magnitude_summary(np.array([subnormal, 1.0]))

    assert summary["fraction_below_tiny"] == 0.5
    assert summary["fraction_below_relative_epsilon"] == 0.5


def test_singular_value_summary_reports_rank_deficient_cut():
    summary = singular_value_summary([np.array([4.0, 2.0]), np.array([3.0, 0.0])])

    assert summary == {
        "populated_cuts": 2,
        "min_nonzero": 2.0,
        "max_condition": np.inf,
        "zero_count": 1,
        "nonfinite_count": 0,
    }
