from __future__ import annotations

import inspect

import numpy as np
import pytest

import pyemd
from pyemd._lib import solve


PUBLISHED_CASES = [
    (
        [0.0, 1.0],
        [5.0, 3.0],
        [[0.0, 0.5], [0.5, 0.0]],
        -1.0,
        3.5,
    ),
    (
        [1.0, 1.0],
        [1.0, 1.0],
        [[0.0, 1.0], [1.0, 0.0]],
        -1.0,
        0.0,
    ),
    (
        [6.0, 1.0],
        [1.0, 7.0],
        [[0.0, 0.0], [0.0, 0.0]],
        -1.0,
        0.0,
    ),
    (
        [1.0, 2.0, 1.0, 2.0],
        [2.0, 1.0, 2.0, 1.0],
        [
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 0.0, 2.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
        ],
        -1.0,
        2.0,
    ),
    (
        [0.0, 2.0, 1.0, 2.0],
        [2.0, 1.0, 2.0, 1.0],
        [
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 0.0, 2.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
        ],
        2.5,
        4.5,
    ),
]


@pytest.mark.parametrize("first,second,matrix,penalty,expected", PUBLISHED_CASES)
def test_published_emd_vectors(
    upstream_pyemd, first, second, matrix, penalty, expected
):
    ours = pyemd.emd(first, second, matrix, penalty)
    theirs = upstream_pyemd.emd(first, second, matrix, penalty)
    assert ours == pytest.approx(expected, abs=1e-12)
    assert ours == pytest.approx(theirs, abs=1e-12)


@pytest.mark.parametrize("n", [2, 5, 12, 32])
@pytest.mark.parametrize("balanced", [True, False])
def test_random_metric_parity(upstream_pyemd, n, balanced):
    rng = np.random.default_rng(100 + n + balanced)
    points = rng.normal(size=(n, 3))
    matrix = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    first = rng.random(n)
    second = rng.random(n)
    if balanced:
        second *= first.sum() / second.sum()
    penalty = 3.25
    ours = pyemd.emd(first, second, matrix, penalty)
    theirs = upstream_pyemd.emd(first, second, matrix, penalty)
    assert ours == pytest.approx(theirs, rel=2e-12, abs=2e-12)


def test_simd_tail_parity(upstream_pyemd):
    rng = np.random.default_rng(701)
    n = 7
    points = rng.normal(size=(n, 3))
    matrix = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    first = rng.random(n)
    second = rng.random(n)
    second *= first.sum() / second.sum()
    assert pyemd.emd(first, second, matrix) == pytest.approx(
        upstream_pyemd.emd(first, second, matrix), rel=2e-12, abs=2e-12
    )


def test_sparse_reverse_threshold_parity(upstream_pyemd):
    rng = np.random.default_rng(971)
    n = 97
    points = rng.normal(size=(n, 3))
    matrix = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    first = rng.random(n)
    second = rng.random(n)
    second *= first.sum() / second.sum()
    assert pyemd.emd(first, second, matrix) == pytest.approx(
        upstream_pyemd.emd(first, second, matrix), rel=2e-12, abs=2e-12
    )


def test_parallel_flow_clear_threshold():
    n = 512
    first = np.linspace(1.0, 2.0, n)
    matrix = np.zeros((n, n), dtype=np.float64)
    value, flow_list = pyemd.emd_with_flow(first, first, matrix)
    flow = np.asarray(flow_list)
    assert value == 0.0
    assert np.array_equal(np.diag(flow), first)
    flow[np.diag_indices(n)] = 0.0
    assert not np.any(flow)


def test_flow_matches_value_and_marginals(upstream_pyemd):
    rng = np.random.default_rng(7)
    n = 18
    x = np.sort(rng.normal(size=n))
    matrix = np.abs(x[:, None] - x[None, :])
    first = rng.random(n)
    second = rng.random(n)
    second *= first.sum() / second.sum()
    value, flow_list = pyemd.emd_with_flow(first, second, matrix)
    upstream_value, _ = upstream_pyemd.emd_with_flow(first, second, matrix)
    flow = np.asarray(flow_list)
    assert value == pytest.approx(upstream_value, abs=2e-12)
    assert np.allclose(flow.sum(axis=1), first, atol=2e-12)
    assert np.allclose(flow.sum(axis=0), second, atol=2e-12)
    assert np.sum(flow * matrix) == pytest.approx(value, abs=2e-12)


@pytest.mark.parametrize(
    "first,second,matrix,expected_flow",
    [
        (
            [0.0, 1.0],
            [5.0, 3.0],
            [[0.0, 0.5], [0.5, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ),
        (
            [6.0, 1.0],
            [1.0, 7.0],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 5.0], [0.0, 1.0]],
        ),
        (
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 1.0, 2.0, 1.0],
            [
                [0.0, 1.0, 1.0, 2.0],
                [1.0, 0.0, 2.0, 1.0],
                [1.0, 2.0, 0.0, 1.0],
                [2.0, 1.0, 1.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
        ),
    ],
)
def test_published_flow_vectors(
    upstream_pyemd, first, second, matrix, expected_flow
):
    ours_value, ours_flow = pyemd.emd_with_flow(first, second, matrix)
    upstream_value, upstream_flow = upstream_pyemd.emd_with_flow(
        first, second, matrix
    )
    assert ours_value == pytest.approx(upstream_value, abs=1e-12)
    assert np.allclose(ours_flow, expected_flow, atol=1e-12)
    assert np.allclose(ours_flow, upstream_flow, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bins": 2},
        {"bins": 4},
        {"bins": 10, "range": (0, 10)},
        {"bins": 4, "normalized": False},
        {"bins": "auto"},
    ],
)
def test_samples_parity(upstream_pyemd, kwargs):
    first = [1, 2, 3, 4]
    second = [2, 3, 4, 5]
    ours = pyemd.emd_samples(first, second, **kwargs)
    theirs = upstream_pyemd.emd_samples(first, second, **kwargs)
    assert ours == pytest.approx(theirs, abs=1e-12)


def test_samples_custom_distance(upstream_pyemd):
    def distance(x):
        return np.not_equal.outer(x, x).astype(np.float64)

    first = [1, 2, 3, 4]
    second = [2, 3, 4, 5]
    ours = pyemd.emd_samples(first, second, bins=4, distance=distance)
    theirs = upstream_pyemd.emd_samples(first, second, bins=4, distance=distance)
    assert ours == pytest.approx(theirs, abs=1e-12)


def test_inputs_are_not_mutated():
    first = np.array([0.2, 0.8])
    second = np.array([0.7, 0.3])
    matrix = np.array([[0.0, 2.0], [2.0, 0.0]])
    copies = first.copy(), second.copy(), matrix.copy()
    pyemd.emd_with_flow(first, second, matrix)
    assert np.array_equal(first, copies[0])
    assert np.array_equal(second, copies[1])
    assert np.array_equal(matrix, copies[2])


def test_documented_larger_distance_matrix():
    first = [0.0, 1.0]
    second = [1.0, 0.0]
    matrix = np.array(
        [[0.0, 2.0, 7.0], [2.0, 0.0, 7.0], [7.0, 7.0, 0.0]]
    )
    assert pyemd.emd(first, second, matrix) == pytest.approx(2.0)


def test_array_like_and_noncontiguous_inputs(upstream_pyemd):
    base = np.array([0.1, 9.0, 0.2, 9.0, 0.7, 9.0])
    first = base[::2]
    second = [0.4, 0.4, 0.2]
    matrix = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    assert pyemd.emd(first, second, matrix) == pytest.approx(
        upstream_pyemd.emd(first, second, matrix)
    )


def test_non_native_and_misaligned_inputs_are_copied_safely(upstream_pyemd):
    first = np.array([0.2, 0.8], dtype=">f8")
    raw = np.zeros(17, dtype=np.uint8)
    second = raw[1:].view(np.float64)
    second[:] = [0.7, 0.3]
    matrix = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=">f8")
    assert not second.flags.aligned
    assert pyemd.emd(first, second, matrix) == pytest.approx(
        upstream_pyemd.emd(
            first.astype(np.float64),
            second.copy(),
            matrix.astype(np.float64),
        )
    )


@pytest.mark.parametrize(
    "first,second,matrix",
    [
        ([1.0 + 1.0j, 0.0], [0.0, 1.0], [[0.0, 1.0], [1.0, 0.0]]),
        ([1.0, 0.0], [0.0, 1.0], [[0.0, 1.0j], [1.0, 0.0]]),
    ],
)
def test_complex_inputs_are_not_silently_narrowed(first, second, matrix):
    with pytest.raises(TypeError, match="Complex"):
        pyemd.emd(first, second, matrix)


def test_wide_numeric_inputs_are_not_silently_narrowed():
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="exactly representable"):
        pyemd.emd(np.array([2**53 + 1, 0], dtype=np.uint64), [0, 1], matrix)
    if hasattr(np, "float128"):
        with pytest.raises(TypeError, match="wider than float64"):
            pyemd.emd(np.array([1, 0], dtype=np.float128), [0, 1], matrix)


def test_native_bridge_rejects_unsafe_buffer_contracts():
    good = np.array([0.25, 0.75], dtype=np.float64)
    costs = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(TypeError, match="float64"):
        solve(good.astype(np.float32), good, costs)
    with pytest.raises(ValueError, match="shape"):
        solve(good, good, np.zeros((3, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        solve(good, good, costs[:, ::-1])


@pytest.mark.parametrize(
    "first,second,matrix",
    [
        ([1, 2], [1], [[0, 1], [1, 0]]),
        ([1, 2, 3], [1, 2, 3], [[0, 1], [1, 0]]),
        ([-1, 2], [1, 1], [[0, 1], [1, 0]]),
        ([1, np.nan], [1, 1], [[0, 1], [1, 0]]),
    ],
)
def test_invalid_inputs_raise(first, second, matrix):
    with pytest.raises(ValueError):
        pyemd.emd(first, second, matrix)


def test_empty_samples_raise():
    with pytest.raises(ValueError, match="cannot be empty"):
        pyemd.emd_samples([], [1])


def test_public_names_and_signatures(upstream_pyemd):
    assert pyemd.__all__ == upstream_pyemd.__all__
    for name in pyemd.__all__:
        assert inspect.signature(getattr(pyemd, name)) == inspect.signature(
            getattr(upstream_pyemd, name)
        )
