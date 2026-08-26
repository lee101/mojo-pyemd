# mojo-pyemd

`mojo-pyemd` is a standalone Mojo port of the compute-bound core of
[PyEMD](https://github.com/wmayner/pyemd), the Earth Mover's Distance
(minimum-cost transport) library. It exposes the same `pyemd` Python import and
the same public function signatures, with the dense transport optimization
running in a compiled Mojo shared library.

## Supported API

The three public functions exported by PyEMD 2.0 are implemented and exercised
against the upstream package:

- `emd(first_histogram, second_histogram, distance_matrix,
  extra_mass_penalty=-1.0)`
- `emd_with_flow(first_histogram, second_histogram, distance_matrix,
  extra_mass_penalty=-1.0)`
- `emd_samples(first_array, second_array, extra_mass_penalty=-1.0,
  distance="euclidean", normalized=True, bins="auto", range=None)`

Balanced and unbalanced histograms, the default maximum-distance extra-mass
penalty, explicit penalties, flow output, NumPy histogram bin rules, and custom
Python distance functions are supported. The test suite compares against the
real `pyemd==2.0.0` package, including its published examples and randomized
metric transport problems.

This is API-compatible, not an implementation of every private PyEMD or POT
helper. It does not provide POT's wider optimal-transport API, sparse cost
matrices, batched EMD, GPU execution, or non-Linux builds. `emd_samples`
histogram construction and custom distance callbacks remain in Python; only
the minimum-cost-flow solve is native Mojo. As in upstream PyEMD, the caller
must supply a distance matrix representing a metric. When several optimal
flows exist, the returned flow can differ from upstream while having the same
cost and marginals. This port additionally accepts a distance matrix larger
than the histograms and uses its leading square submatrix.

## Install and build

This repository currently installs from source with
[Pixi](https://pixi.sh). From a clone, run:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-pyemd.so`. Run the parity suite with
`pixi run test`.

## Usage

With the Pixi environment active, or through `pixi run python`:

```python
import numpy as np
from pyemd import emd, emd_with_flow

first = np.array([0.0, 1.0])
second = np.array([5.0, 3.0])
distances = np.array([[0.0, 0.5], [0.5, 0.0]])

print(emd(first, second, distances))
print(emd_with_flow(first, second, distances))
```

This prints:

```text
3.5
(3.5, [[0.0, 0.0], [0.0, 1.0]])
```

## Benchmark

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of seven warmed calls using balanced random
histograms and a dense Euclidean ground-distance matrix.

| balanced dense EMD | mojo-pyemd | pyemd 2.0.0 | speedup |
|---:|---:|---:|---:|
| 8 bins | 0.076 ms | 0.295 ms | 3.87x |
| 16 bins | 0.104 ms | 0.368 ms | 3.55x |
| 32 bins | 0.270 ms | 0.429 ms | 1.59x |
| 64 bins | 1.822 ms | 0.598 ms | 0.33x |
| 128 bins | 7.316 ms | 1.219 ms | 0.17x |

Mojo wins through 32 bins in this run because it avoids much of POT's setup
cost. PyEMD/POT is faster from 64 bins upward: its mature network-simplex
implementation scales better than this port's successive-shortest-path
solver.

No GPU path is provided. The hot dense relaxations perform only a few
arithmetic operations while loading and conditionally storing several values
per edge, well below the roughly 2-flop-per-byte threshold where transfer and
launch costs could pay off. The residual-edge traversal is also irregular and
sequential between augmentations.

## How it works

The Python layer converts array-like inputs to aligned, C-contiguous, native
`float64` NumPy buffers. It validates shapes, values, layout, sizes, and
non-null addresses before passing integer addresses through `ctypes`. The
NumPy objects stay referenced for the whole call, which releases the Python
GIL while Mojo borrows their buffers.

The Mojo kernel first cancels mass in matching bins, matching PyEMD's zero-cost
preflow behavior. It then solves the remaining complete bipartite transport
problem with successive shortest augmenting paths. Reduced-cost node
potentials make each residual shortest-path search a dense Dijkstra pass;
reverse residual edges allow earlier assignments to be rerouted. Dense
minimum scans and forward relaxations use unaligned-safe SIMD loads and scalar
tails. Per-search scratch initialization is SIMD-vectorized as well. Problems
of 24 bins or more track sparse reverse flow edges instead of scanning strided
dense columns, and the search stops as soon as the sink is finalized.

All matrices are row-major, including the cost and returned flow matrices.
Already aligned native `float64` inputs cross the FFI boundary without a copy;
other valid array-like inputs are safely converted. Per-thread scratch buffers
are reused across calls, and flow buffers are cleared with unaligned-safe SIMD
stores and a scalar tail.

## License

MIT
