"""Dense Earth Mover's Distance over caller-owned row-major buffers."""

from std.ffi import external_call
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime INF = 1.0e300
comptime W = simdwidthof[DType.float64]()
comptime SPARSE_REVERSE_THRESHOLD = 96


def clear_flow(flow: FPtr, count: Int):
    var vector_end = count // W * W
    var zeros = SIMD[DType.float64, W](0.0)
    for k in range(0, vector_end, W):
        flow.store[alignment=1](k, zeros)
    for k in range(vector_end, count):
        flow[k] = 0.0


def solve_transport(
    a: FPtr,
    b: FPtr,
    costs: FPtr,
    flow: FPtr,
    supply: FPtr,
    demand: FPtr,
    potential: FPtr,
    distance: FPtr,
    previous: IPtr,
    visited: IPtr,
    flow_sources: IPtr,
    flow_counts: IPtr,
    n: Int,
) -> Float64:
    clear_flow(flow, n * n)
    if n >= SPARSE_REVERSE_THRESHOLD:
        for j in range(n):
            flow_counts[j] = 0
    var remaining_a_vec = SIMD[DType.float64, W](0.0)
    var remaining_b_vec = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var av = a.load[width=W, alignment=1](i)
        var bv = b.load[width=W, alignment=1](i)
        var shared = min(av, bv)
        var supply_vec = av - shared
        var demand_vec = bv - shared
        supply.store[alignment=1](i, supply_vec)
        demand.store[alignment=1](i, demand_vec)
        remaining_a_vec += supply_vec
        remaining_b_vec += demand_vec
        for lane in range(W):
            var diagonal = i + lane
            flow[diagonal * n + diagonal] = shared[lane]
            if n >= SPARSE_REVERSE_THRESHOLD and shared[lane] > 0.0:
                flow_sources[diagonal * n] = Int64(diagonal)
                flow_counts[diagonal] = 1
        i += W
    var remaining_a = remaining_a_vec.reduce_add()
    var remaining_b = remaining_b_vec.reduce_add()
    while i < n:
        var shared = min(a[i], b[i])
        flow[i * n + i] = shared
        supply[i] = a[i] - shared
        demand[i] = b[i] - shared
        remaining_a += supply[i]
        remaining_b += demand[i]
        if n >= SPARSE_REVERSE_THRESHOLD and shared > 0.0:
            flow_sources[i * n] = Int64(i)
            flow_counts[i] = 1
        i += 1

    var remaining = min(remaining_a, remaining_b)
    var transport_cost = 0.0
    var node_count = 2 * n + 2
    var source = 2 * n
    var sink = source + 1
    for v in range(node_count):
        potential[v] = 0.0

    while remaining > 0.0:
        for v in range(node_count):
            distance[v] = INF
            previous[v] = -1
            visited[v] = 0
        distance[source] = 0.0
        visited[source] = 1
        for i in range(n):
            if supply[i] > 0.0:
                distance[i] = max(
                    0.0, potential[source] - potential[i]
                )
                previous[i] = Int64(source)

        for _ in range(node_count - 1):
            var u = -1
            var best = INF
            var v = 0
            while n >= SPARSE_REVERSE_THRESHOLD and v + 4 * W <= node_count:
                var eligible0 = visited.load[
                    width=W, alignment=1
                ](v).eq(0).select(
                    distance.load[width=W, alignment=1](v),
                    SIMD[DType.float64, W](INF),
                )
                var eligible1 = visited.load[
                    width=W, alignment=1
                ](v + W).eq(0).select(
                    distance.load[width=W, alignment=1](v + W),
                    SIMD[DType.float64, W](INF),
                )
                var eligible2 = visited.load[
                    width=W, alignment=1
                ](v + 2 * W).eq(0).select(
                    distance.load[width=W, alignment=1](v + 2 * W),
                    SIMD[DType.float64, W](INF),
                )
                var eligible3 = visited.load[
                    width=W, alignment=1
                ](v + 3 * W).eq(0).select(
                    distance.load[width=W, alignment=1](v + 3 * W),
                    SIMD[DType.float64, W](INF),
                )
                var block_min = min(
                    min(eligible0, eligible1), min(eligible2, eligible3)
                )
                if block_min.reduce_min() < best:
                    for lane in range(W):
                        if eligible0[lane] < best:
                            best = eligible0[lane]
                            u = v + lane
                    for lane in range(W):
                        if eligible1[lane] < best:
                            best = eligible1[lane]
                            u = v + W + lane
                    for lane in range(W):
                        if eligible2[lane] < best:
                            best = eligible2[lane]
                            u = v + 2 * W + lane
                    for lane in range(W):
                        if eligible3[lane] < best:
                            best = eligible3[lane]
                            u = v + 3 * W + lane
                v += 4 * W
            while v + W <= node_count:
                var unused = visited.load[width=W, alignment=1](v).eq(0)
                var eligible = unused.select(
                    distance.load[width=W, alignment=1](v),
                    SIMD[DType.float64, W](INF),
                )
                if eligible.reduce_min() < best:
                    for lane in range(W):
                        if eligible[lane] < best:
                            best = eligible[lane]
                            u = v + lane
                v += W
            while v < node_count:
                if visited[v] == 0 and distance[v] < best:
                    best = distance[v]
                    u = v
                v += 1
            if u < 0:
                break
            visited[u] = 1
            if u == sink:
                break
            elif u < n:
                var j = 0
                while j + W <= n:
                    var right = n + j
                    var old_distance = distance.load[width=W, alignment=1](right)
                    var reduced = max(
                        SIMD[DType.float64, W](0.0),
                        costs.load[width=W, alignment=1](u * n + j)
                        + SIMD[DType.float64, W](potential[u])
                        - potential.load[width=W, alignment=1](right),
                    )
                    var candidate = SIMD[DType.float64, W](distance[u]) + reduced
                    var improved = candidate.lt(old_distance)
                    distance.store[alignment=1](
                        right, improved.select(candidate, old_distance)
                    )
                    previous.store[alignment=1](
                        right,
                        improved.select(
                            SIMD[DType.int64, W](Int64(u)),
                            previous.load[width=W, alignment=1](right),
                        ),
                    )
                    j += W
                while j < n:
                    v = n + j
                    var reduced = max(
                        0.0,
                        costs[u * n + j] + potential[u] - potential[v],
                    )
                    var candidate = distance[u] + reduced
                    if candidate < distance[v]:
                        distance[v] = candidate
                        previous[v] = Int64(u)
                    j += 1
            else:
                var j = u - n
                if j < n:
                    if n >= SPARSE_REVERSE_THRESHOLD:
                        for edge in range(Int(flow_counts[j])):
                            i = Int(flow_sources[j * n + edge])
                            if flow[i * n + j] > 0.0:
                                var reduced = max(
                                    0.0,
                                    -costs[i * n + j] + potential[u] - potential[i],
                                )
                                var candidate = distance[u] + reduced
                                if candidate < distance[i]:
                                    distance[i] = candidate
                                    previous[i] = Int64(u)
                    else:
                        i = 0
                        while i + W <= n:
                            var flow_vec = (
                                flow + i * n + j
                            ).strided_load[width=W](n)
                            var positive = flow_vec.gt(0.0)
                            var old_distance = distance.load[
                                width=W, alignment=1
                            ](i)
                            var reduced = max(
                                SIMD[DType.float64, W](0.0),
                                -(costs + i * n + j).strided_load[width=W](n)
                                + SIMD[DType.float64, W](potential[u])
                                - potential.load[width=W, alignment=1](i),
                            )
                            var candidate = (
                                SIMD[DType.float64, W](distance[u]) + reduced
                            )
                            var improved = positive & candidate.lt(old_distance)
                            distance.store[alignment=1](
                                i, improved.select(candidate, old_distance)
                            )
                            previous.store[alignment=1](
                                i,
                                improved.select(
                                    SIMD[DType.int64, W](Int64(u)),
                                    previous.load[width=W, alignment=1](i),
                                ),
                            )
                            i += W
                        while i < n:
                            if flow[i * n + j] > 0.0:
                                var reduced = max(
                                    0.0,
                                    -costs[i * n + j]
                                    + potential[u]
                                    - potential[i],
                                )
                                var candidate = distance[u] + reduced
                                if candidate < distance[i]:
                                    distance[i] = candidate
                                    previous[i] = Int64(u)
                            i += 1
                    if demand[j] > 0.0:
                        var reduced = max(
                            0.0, potential[u] - potential[sink]
                        )
                        var candidate = distance[u] + reduced
                        if candidate < distance[sink]:
                            distance[sink] = candidate
                            previous[sink] = Int64(u)

        if previous[sink] < 0:
            return transport_cost
        var sink_distance = SIMD[DType.float64, W](distance[sink])
        v = 0
        while v + W <= node_count:
            var distances = distance.load[width=W, alignment=1](v)
            var old_potential = potential.load[width=W, alignment=1](v)
            potential.store[alignment=1](
                v, old_potential + min(distances, sink_distance)
            )
            v += W
        while v < node_count:
            potential[v] += min(distance[v], distance[sink])
            v += 1

        var end_target = Int(previous[sink]) - n
        var cursor = Int(previous[sink])
        while Int(previous[cursor]) != source:
            cursor = Int(previous[cursor])
        var start_source = cursor
        var delta = min(supply[start_source], demand[end_target])

        cursor = Int(previous[sink])
        while Int(previous[cursor]) != source:
            var parent = Int(previous[cursor])
            if parent >= n and parent < 2 * n and cursor < n:
                delta = min(delta, flow[cursor * n + parent - n])
            cursor = parent

        cursor = Int(previous[sink])
        while Int(previous[cursor]) != source:
            var parent = Int(previous[cursor])
            if parent < n and cursor >= n and cursor < 2 * n:
                var j = cursor - n
                if (
                    n >= SPARSE_REVERSE_THRESHOLD
                    and flow[parent * n + j] == 0.0
                ):
                    var tracked = False
                    for edge in range(Int(flow_counts[j])):
                        if flow_sources[j * n + edge] == Int64(parent):
                            tracked = True
                            break
                    if not tracked:
                        flow_sources[j * n + Int(flow_counts[j])] = Int64(parent)
                        flow_counts[j] += 1
                flow[parent * n + j] += delta
                transport_cost += delta * costs[parent * n + j]
            elif parent >= n and parent < 2 * n and cursor < n:
                var j = parent - n
                if flow[cursor * n + j] == delta:
                    flow[cursor * n + j] = 0.0
                else:
                    flow[cursor * n + j] -= delta
                transport_cost -= delta * costs[cursor * n + j]
            cursor = parent

        if supply[start_source] == delta:
            supply[start_source] = 0.0
        else:
            supply[start_source] -= delta
        if demand[end_target] == delta:
            demand[end_target] = 0.0
        else:
            demand[end_target] -= delta
        if remaining == delta:
            remaining = 0.0
        else:
            remaining -= delta

    return transport_cost


@export("mpyemd_parallel_init")
def mpyemd_parallel_init() abi("C") -> Int:
    return external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()


@export("mpyemd_solve")
def mpyemd_solve(
    a_addr: Int,
    b_addr: Int,
    costs_addr: Int,
    flow_addr: Int,
    supply_addr: Int,
    demand_addr: Int,
    potential_addr: Int,
    distance_addr: Int,
    previous_addr: Int,
    visited_addr: Int,
    flow_sources_addr: Int,
    flow_counts_addr: Int,
    n: Int,
) abi("C") -> Float64:
    return solve_transport(
        FPtr(unsafe_from_address=a_addr),
        FPtr(unsafe_from_address=b_addr),
        FPtr(unsafe_from_address=costs_addr),
        FPtr(unsafe_from_address=flow_addr),
        FPtr(unsafe_from_address=supply_addr),
        FPtr(unsafe_from_address=demand_addr),
        FPtr(unsafe_from_address=potential_addr),
        FPtr(unsafe_from_address=distance_addr),
        IPtr(unsafe_from_address=previous_addr),
        IPtr(unsafe_from_address=visited_addr),
        IPtr(unsafe_from_address=flow_sources_addr),
        IPtr(unsafe_from_address=flow_counts_addr),
        n,
    )
