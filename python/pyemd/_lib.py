"""ctypes bridge to the Mojo transport solver."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "pyemd.mojo")
LIB = os.environ.get("MOJO_PYEMD_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyemd.so"
)

I = ctypes.c_int64
F = ctypes.c_double


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_PYEMD_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run `pixi run build` first")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None
_library_lock = threading.Lock()
_workspace = threading.local()


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        with _library_lock:
            if _library is not None:
                return _library
            library = ctypes.CDLL(build())
            parallel_init = library.mpyemd_parallel_init
            parallel_init.argtypes = []
            parallel_init.restype = I
            if parallel_init() == 0:
                raise RuntimeError("Mojo CPU runtime initialization failed")
            fn = library.mpyemd_solve
            fn.argtypes = [I] * 13
            fn.restype = F
            _library = library
    assert _library is not None
    return _library


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a non-empty buffer")
    return address


def _check_input(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != np.dtype(np.float64) or not array.dtype.isnative:
        raise TypeError(f"{name} must have native float64 dtype")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError(f"{name} must be C-contiguous and aligned")


def solve(
    first: np.ndarray, second: np.ndarray, costs: np.ndarray
) -> tuple[float, np.ndarray]:
    if not isinstance(first, np.ndarray) or first.ndim != 1:
        raise TypeError("first must be a one-dimensional NumPy array")
    n = first.shape[0]
    _check_input(first, (n,), "first")
    _check_input(second, (n,), "second")
    _check_input(costs, (n, n), "costs")
    if n == 0:
        return 0.0, np.empty((0, 0), dtype=np.float64)
    if n > (sys.maxsize - 2) // 2 or n > sys.maxsize // n:
        raise OverflowError("histogram is too large for the native solver")
    node_count = 2 * n + 2
    if getattr(_workspace, "n", None) != n:
        _workspace.n = n
        _workspace.buffers = (
            np.empty((n, n), dtype=np.float64),
            np.empty(n, dtype=np.float64),
            np.empty(n, dtype=np.float64),
            np.empty(node_count, dtype=np.float64),
            np.empty(node_count, dtype=np.float64),
            np.empty(node_count, dtype=np.int64),
            np.empty(node_count, dtype=np.int64),
            np.empty((n, n), dtype=np.int64),
            np.empty(n, dtype=np.int64),
        )
    (
        flow,
        supply,
        demand,
        potential,
        distance,
        previous,
        visited,
        flow_sources,
        flow_counts,
    ) = _workspace.buffers
    transport_cost = lib().mpyemd_solve(
        addr(first),
        addr(second),
        addr(costs),
        addr(flow),
        addr(supply),
        addr(demand),
        addr(potential),
        addr(distance),
        addr(previous),
        addr(visited),
        addr(flow_sources),
        addr(flow_counts),
        n,
    )
    if not np.isfinite(transport_cost):
        raise RuntimeError("Mojo transport solver failed")
    return transport_cost, flow
