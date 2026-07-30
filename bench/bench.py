from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PYTHON = os.path.join(ROOT, "python")
sys.path.insert(0, LOCAL_PYTHON)

import pyemd  # noqa: E402


def load_upstream():
    local_init = os.path.realpath(os.path.join(LOCAL_PYTHON, "pyemd", "__init__.py"))
    for entry in sys.path:
        candidate = os.path.join(entry, "pyemd", "__init__.py")
        if os.path.isfile(candidate) and os.path.realpath(candidate) != local_init:
            spec = importlib.util.spec_from_file_location(
                "_benchmark_upstream_pyemd",
                candidate,
                submodule_search_locations=[os.path.dirname(candidate)],
            )
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module
    raise RuntimeError("upstream pyemd is not installed")


def best_time(fn, repeat=7):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def make_case(n, seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    matrix = np.ascontiguousarray(
        np.linalg.norm(points[:, None] - points[None, :], axis=2)
    )
    first = np.ascontiguousarray(rng.random(n))
    second = np.ascontiguousarray(rng.random(n))
    second *= first.sum() / second.sum()
    return first, second, matrix


def main():
    upstream = load_upstream()
    version = importlib.metadata.version("pyemd")
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print(f"| balanced dense EMD | mojo-pyemd | pyemd {version} | speedup |")
    print("|---:|---:|---:|---:|")
    for n in (8, 16, 32, 64, 128):
        first, second, matrix = make_case(n, n)
        ours = lambda: pyemd.emd(first, second, matrix)
        theirs = lambda: upstream.emd(first, second, matrix)
        ours()
        theirs()
        ours_time = best_time(ours)
        upstream_time = best_time(theirs)
        print(
            f"| {n} bins | {ours_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {upstream_time / ours_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
