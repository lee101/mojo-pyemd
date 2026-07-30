"""PyEMD-compatible public API backed by a Mojo minimum-cost-flow solver."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from ._lib import solve

DEFAULT_EXTRA_MASS_PENALTY = -1.0
MAX_EXACT_FLOAT64_INTEGER = 2**53


def _reject_lossy_dtype(array: np.ndarray) -> None:
    if np.iscomplexobj(array):
        raise TypeError("Complex-valued inputs are not supported")
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        raise TypeError("Floating-point inputs wider than float64 are not supported")
    if array.dtype.kind in "iu" and np.any(
        array.astype(object) > MAX_EXACT_FLOAT64_INTEGER
    ):
        raise ValueError("Integer inputs must be exactly representable as float64")


def _inputs(
    first_histogram: ArrayLike,
    second_histogram: ArrayLike,
    distance_matrix: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_raw = np.asarray(first_histogram)
    second_raw = np.asarray(second_histogram)
    matrix_raw = np.asarray(distance_matrix)
    _reject_lossy_dtype(first_raw)
    _reject_lossy_dtype(second_raw)
    _reject_lossy_dtype(matrix_raw)
    if first_raw.ndim != 1 or second_raw.ndim != 1:
        raise ValueError("Histograms must be one-dimensional")
    if first_raw.shape[0] != second_raw.shape[0]:
        raise ValueError("Histogram lengths must be equal")
    n = first_raw.shape[0]
    if matrix_raw.ndim != 2 or matrix_raw.shape[0] < n or matrix_raw.shape[1] < n:
        raise ValueError(
            "Histogram lengths cannot be greater than the "
            "number of rows or columns of the distance matrix"
        )
    try:
        first = np.require(first_raw, dtype=np.float64, requirements=("C", "A"))
        second = np.require(second_raw, dtype=np.float64, requirements=("C", "A"))
        matrix = np.require(
            matrix_raw[:n, :n], dtype=np.float64, requirements=("C", "A")
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Histograms and distance matrix must be numeric") from exc
    if np.any(first < 0) or np.any(second < 0):
        raise ValueError("Histograms cannot contain negative values")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Histograms must contain finite values")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Distance matrix must contain finite non-negative values")
    return first, second, matrix


def _emd_and_flow(
    first_histogram: ArrayLike,
    second_histogram: ArrayLike,
    distance_matrix: ArrayLike,
    extra_mass_penalty: float,
) -> tuple[float, np.ndarray]:
    first, second, matrix = _inputs(
        first_histogram, second_histogram, distance_matrix
    )
    penalty = float(extra_mass_penalty)
    if penalty == DEFAULT_EXTRA_MASS_PENALTY:
        penalty = float(np.asarray(distance_matrix).max())
    if not np.isfinite(penalty):
        raise ValueError("extra_mass_penalty must be finite")
    transport_cost, flow = solve(first, second, matrix)
    extra_mass = abs(float(first.sum()) - float(second.sum()))
    return transport_cost + extra_mass * penalty, flow


def emd(
    first_histogram: np.ndarray,
    second_histogram: np.ndarray,
    distance_matrix: np.ndarray,
    extra_mass_penalty: float = DEFAULT_EXTRA_MASS_PENALTY,
) -> float:
    """Return the Earth Mover's Distance between two histograms."""
    value, _ = _emd_and_flow(
        first_histogram,
        second_histogram,
        distance_matrix,
        extra_mass_penalty,
    )
    return float(value)


def emd_with_flow(
    first_histogram: np.ndarray,
    second_histogram: np.ndarray,
    distance_matrix: np.ndarray,
    extra_mass_penalty: float = DEFAULT_EXTRA_MASS_PENALTY,
) -> tuple[float, list[list[float]]]:
    """Return the Earth Mover's Distance and its minimum-cost flow."""
    value, flow = _emd_and_flow(
        first_histogram,
        second_histogram,
        distance_matrix,
        extra_mass_penalty,
    )
    return float(value), flow.tolist()


def euclidean_pairwise_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Calculate the Euclidean pairwise distance matrix for a 1D array."""
    values = np.asarray(x)
    return np.abs(values[:, None] - values[None, :])


get_bins = np.histogram_bin_edges


def emd_samples(
    first_array: ArrayLike,
    second_array: ArrayLike,
    extra_mass_penalty: float = DEFAULT_EXTRA_MASS_PENALTY,
    distance: str | Callable[[np.ndarray], np.ndarray] = "euclidean",
    normalized: bool = True,
    bins: int | str = "auto",
    range: tuple[float, float] | None = None,
) -> float:
    """Return the EMD between histograms constructed from two sample arrays."""
    first_array = np.array(first_array)
    second_array = np.array(second_array)
    if not (first_array.size > 0 and second_array.size > 0):
        raise ValueError("Arrays of samples cannot be empty.")
    if range is None:
        range = (
            min(np.min(first_array), np.min(second_array)),
            max(np.max(first_array), np.max(second_array)),
        )
    bin_edges = get_bins(
        np.concatenate([first_array, second_array]).astype(np.float64),
        range=range,
        bins=bins,
    )
    first_histogram, bin_edges = np.histogram(
        first_array, range=range, bins=bin_edges
    )
    second_histogram, _ = np.histogram(
        second_array, range=range, bins=bin_edges
    )
    first_histogram = first_histogram.astype(np.float64)
    second_histogram = second_histogram.astype(np.float64)
    if normalized:
        first_histogram /= np.sum(first_histogram)
        second_histogram /= np.sum(second_histogram)
    bin_locations = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
    if distance == "euclidean":
        distance = euclidean_pairwise_distance_matrix
    distance_matrix = np.asarray(distance(bin_locations))
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(
            "Distance matrix must be square; check your `distance` function."
        )
    if (
        first_histogram.shape[0] > distance_matrix.shape[0]
        or second_histogram.shape[0] > distance_matrix.shape[1]
    ):
        raise ValueError(
            "Distance matrix must have at least as many rows/columns as there "
            "are bins in the histograms; check your `distance` function."
        )
    return emd(
        first_histogram,
        second_histogram,
        distance_matrix,
        extra_mass_penalty,
    )
