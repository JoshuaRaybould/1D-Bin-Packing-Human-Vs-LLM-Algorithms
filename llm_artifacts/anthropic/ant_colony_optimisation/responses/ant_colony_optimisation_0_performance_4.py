import random
import time
from collections import defaultdict

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    def elapsed():
        return time.time() - start_time

    def make_result(packing_bins):
        result_packing = []
        result_weights = []
        for b in packing_bins:
            if b:
                result_packing.append(list(b))
                result_weights.append(sum(weights[i] for i in b))
        return {"packing": result_packing, "bin_weights": result_weights}

    w_arr = weights
    total_weight = sum(w_arr)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity

    sorted_desc = sorted(range(n), key=lambda i: -w_arr[i])

    # Best Fit Decreasing
    def bfd(order=None):
        if order is None:
            sorted_items = sorted_desc
        else:
            sorted_items = order
        bins_rem = []
        bins_items = []
        for item in sorted_items:
            w = w_arr[item]
            best_idx = -1
            best_remaining = bin_capacity + 1
            for j in range(len(bins_rem)):
                rem = bins_rem[j]
                if rem >= w and rem < best_remaining:
                    best_remaining = rem
                    best_idx = j
            if best_idx >= 0:
                bins_rem[best_idx] -= w
                bins_items[best_idx].append(item)
            else:
                bins_rem.append(bin_capacity - w)
                bins_items.append([item])
        return bins_items

    # First Fit Decreasing
    def ffd(order=None):
        if order is None:
            sorted_items = sorted_desc
        else:
            sorted_items = order
        bins_rem = []
        bins_items = []
        for item in sorted_items:
            w = w_arr[item]
            placed = False
            for j in range(len(bins_rem)):
                if bins_rem[j] >= w:
                    bins_rem[j] -= w
                    bins_items[j].append(item)
                    placed = True
                    break
            if not placed:
                bins_rem.append(bin_capacity - w)
                bins_items.append([item])
        return bins_items

    # BFD with sorted remaining capacities (faster)
    def bfd_fast(order=None):
        from bisect import bisect_left, insort
        if order is None:
            sorted_items = sorted_desc
        else:
            sorted_items = order
        # Use sorted list of (remaining_cap, bin_index)
        bins_rem = []
        bins_items = []
        sorted_caps = []  # list of (cap, idx) sorted by cap
        for item in sorted_items:
            w = w_arr[item]
            # Find bin with smallest remaining capacity >= w (best fit)
            # Binary search in sorted_caps
            pos = bisect_left(sorted_caps, (w,))
            if pos < len(sorted_caps):
                cap, idx = sorted_caps[pos]
                # Remove and re-insert
                sorted_caps.pop(pos)
                new_cap = cap - w
                bins_items[idx].append(item)
                bins_rem[idx] = new_cap
                if new_cap > 0:
                    insort(sorted_caps, (new_cap, idx))
            else:
                idx = len(bins_items)
                new_cap = bin_capacity - w
                bins_rem.append(new_cap)
                bins_items.append([item])
                if new_cap > 0:
                    insort(sorted_caps, (new_cap, idx))
        return bins_items

    best_solution = bfd()
    best_num_bins = len(best_solution)

    if best_num_bins <= 1 or n <= 1:
        return make_result(best_solution)
    if best_num_bins <= lower_bound:
        return make_result(best_solution)

    sol2 = ffd()
    if len(sol2) < best_num_bins:
        best_num_bins = len(sol2)
        best_solution = sol2
        if best_num_bins <= lower_bound:
            return make_result(best_solution)

    sol3 = bfd_fast()
    if len(sol3) < best_num_bins:
        best_num_bins = len(sol3)
        best_solution = sol3
        if best_num_bins <= lower_bound:
            return make_result(best_solution)

    # Noisy initial heuristics
    noise_budget = min(3.0, time_limit * 0.15)
    while elapsed() < noise_budget:
        noise_level = random.uniform(0.01, 0.20)
        order = sorted(range(n), key=lambda i: -w_arr[i] + random.gauss(0, bin_capacity * noise_level))
        sol = bfd_fast(order)
        if len(sol) < best_num_bins:
            best_num_bins = len(sol)
            best_solution = sol
            if best_num_bins <= lower_bound:
                return make_result(best_solution)
        sol = ffd(order)
        if len(sol) < best_num_bins:
            best_num_bins = len(sol)
            best_solution = sol
            if best_num_bins <= lower_bound:
                return make_result(best_solution)

    if n <= 2:
        return make_result(best_solution)

    # ACO with sparse pheromone (dict-based)
    rho = 0.02
    q0 = 0.8  # ACS exploitation parameter
    beta = 2.0

    def compute_bounds(best_bins):
        t_max = 1.0 / (rho * max(best_bins, 1))
        t_min = t_max / (2.0 * n) if n > 0 else t_max / 2.0
        return t_max, t_min

    tau_max, tau_min = compute_bounds(best_num_bins)

    # Sparse pheromone storage
    # For small n, use flat array; for large n, use dict
    use_array = (n <= 2000)

    if use_array:
        tau_arr = bytearray(b'\x00' * (n * n * 8))  # not practical, use list
        tau_arr = [tau_min] * (n * n)
    else:
        tau_dict = {}

    def get_tau(i, j):
        if use_array:
            return tau_arr[i * n + j]
        else:
            if i > j:
                i, j = j, i
            return tau_dict.get(i * n + j, tau_min)

    def set_tau(i, j, val):
        if use_array:
            tau_arr[i * n + j] = val
            tau_arr[j * n + i] = val
        else:
            if i > j:
                i, j = j, i
            key = i * n + j
            if abs(val - tau_min) < 1e-15:
                tau_dict.pop(key, None)
            else:
                tau_dict[key] = val

    # Deposit for initial best
    dep_val = 1.0 / best_num_bins
    for b in best_solution:
        if len(b) <= 1:
            continue
        for ii in range(len(b)):
            for jj in range(ii + 1, len(b)):
                old = get_tau(b[ii], b[jj])
                nv = min(old + dep_val, tau_max)
                set_tau(b[ii], b[jj], nv)

    def construct_solution():
        """Bin-centric construction with ACS pseudo-random rule."""
        remaining = list(sorted_desc)
        in_rem = bytearray(n)
        for i in range(n):
            in_rem[i] = 1

        bins_result = []
        _w = w_arr
        _cap = bin_capacity
        _q0 = q0
        _beta = beta
        _tmin = tau_min
        _n = n

        while remaining:
            cap = _cap
            seed = remaining[0]
            in_rem[seed] = 0
            cap -= _w[seed]
            bin_items = [seed]

            if cap > 0 and len(remaining) > 1:
                # Collect candidates fitting in remaining capacity
                candidates = []
                for item in remaining:
                    if in_rem[item] == 0:
                        continue
                    if _w[item] <= cap:
                        candidates.append(item)
                        if len(candidates) >= 100:
                            break

                while cap > 0 and candidates:
                    nc = len(candidates)
                    if nc == 1:
                        chosen_item = candidates[0]
                    else:
                        inv_cap = 1.0 / cap
                        # Compute scores: sum of pheromone to all bin items * heuristic^beta
                        best_score = -1.0
                        best_ci = 0
                        scores = [0.0] * nc

                        if use_array:
                            for ci in range(nc):
                                item = candidates[ci]
                                h = _w[item] * inv_cap
                                h2 = h * h
                                # Sum pheromone from all bin items to this candidate
                                p_sum = 0.0
                                for bi_item in bin_items:
                                    p_sum += tau_arr[bi_item * _n + item]
                                s = p_sum * h2
                                scores[ci] = s
                                if s > best_score:
                                    best_score = s
                                    best_ci = ci
                        else:
                            for ci in range(nc):
                                item = candidates[ci]
                                h = _w[item] * inv_cap
                                h2 = h * h
                                p_sum = 0.0
                                for bi_item in bin_items:
                                    a, b_ = (bi_item, item) if bi_item < item else (item, bi_item)
                                    p_sum += tau_dict.get(a * _n + b_, _tmin)
                                s = p_sum * h2
                                scores[ci] = s
                                if s > best_score:
                                    best_score = s
                                    best_ci = ci

                        # ACS pseudo-random proportional rule
                        if random.random() < _q0:
                            chosen_idx = best_ci
                        else:
                            total_score = sum(scores)
                            if total_score <= 0:
                                chosen_idx = 0
                            else:
                                r = random.random() * total_score
                                cumsum = 0.0
                                chosen_idx = nc - 1
                                for pi in range(nc):
                                    cumsum += scores[pi]
                                    if cumsum >= r:
                                        chosen_idx = pi
                                        break

                        chosen_item = candidates[chosen_idx]

                    in_rem[chosen_item] = 0
                    cap -= _w[chosen_item]
                    bin_items.append(chosen_item)

                    if cap > 0:
                        new_cands = []
                        for item in candidates:
                            if in_rem[item] and _w[item] <= cap:
                                new_cands.append(item)
                        candidates = new_cands
                    else:
                        break

            # Rebuild remaining
            new_remaining = []
            for item in remaining:
                if in_rem[item]:
                    new_remaining.append(item)
            remaining = new_remaining
            bins_result.append(bin_items)

        return bins_result

    def deposit_pheromone(sol, amount):
        nonlocal tau_max
        _tm = tau_max
        if use_array:
            _n = n
            for b in sol:
                lb = len(b)
                if lb <= 1:
                    continue
                if lb <= 12:
                    for ii in range(lb):
                        a = b[ii]
                        a_off = a * _n
                        for jj in range(ii + 1, lb):
                            c = b[jj]
                            v = tau_arr[a_off + c] + amount
                            if v > _tm:
                                v = _tm
                            tau_arr[a_off + c] = v
                            tau_arr[c * _n + a] = v
                else:
                    # Only deposit seed to others
                    seed = b[0]
                    s_off = seed * _n
                    for k in range(1, lb):
                        item = b[k]
                        v = tau_arr[s_off + item] + amount
                        if v > _tm:
                            v = _tm
                        tau_arr[s_off + item] = v
                        tau_arr[item * _n + seed] = v
        else:
            _n = n
            _tmin = tau_min
            for b in sol:
                lb = len(b)
                if lb <= 1:
                    continue
                if lb <= 12:
                    for ii in range(lb):
                        for jj in range(ii + 1, lb):
                            a, c = b[ii], b[jj]
                            if a > c:
                                a, c = c, a
                            key = a * _n + c
                            old = tau_dict.get(key, _tmin)
                            nv = old + amount
                            if nv > _tm:
                                nv = _tm
                            tau_dict[key] = nv
                else:
                    seed = b[0]
                    for k in range(1, lb):
                        item = b[k]
                        a, c = (seed, item) if seed < item else (item, seed)
                        key = a * _n + c
                        old = tau_dict.get(key, _tmin)
                        nv = old + amount
                        if nv > _tm:
                            nv = _tm
                        tau_dict[key] = nv

    def evaporate():
        nonlocal tau_min
        factor = 1.0 - rho
        _tm = tau_min
        if use_array:
            nn = n * n
            for i in range(nn):
                v = tau_arr[i] * factor
                if v < _tm:
                    v = _tm
                tau_arr[i] = v
        else:
            to_del = []
            for key, val in tau_dict.items():
                nv = val * factor
                if nv <= _tm * 1.001:
                    to_del.append(key)
                else:
                    tau_dict[key] = nv
            for key in to_del:
                del tau_dict[key]

    # Adaptive ant count
    if n <= 30:
        num_ants = 40
    elif n <= 100:
        num_ants = 25
    elif n <= 300:
        num_ants = 15
    elif n <= 1000:
        num_ants = 10
    else:
        num_ants = 5

    iteration = 0
    stagnation_counter = 0
    time_limit_end = time_limit * 0.97

    while True:
        if elapsed() >= time_limit_end:
            break

        iter_start = time.time()

        iteration_best = None
        iteration_best_bins = float('inf')
        all_solutions = []

        for ant in range(num_ants):
            if elapsed() >= time_limit_end:
                break

            sol = construct_solution()
            num_b = len(sol)
            all_solutions.append((num_b, sol))

            if num_b < iteration_best_bins:
                iteration_best_bins = num_b
                iteration_best = sol

            if num_b < best_num_bins:
                best_num_bins = num_b
                best_solution = sol
                tau_max, tau_min = compute_bounds(best_num_bins)
                stagnation_counter = 0
                if best_num_bins <= lower_bound:
                    return make_result(best_solution)

        if elapsed() >= time_limit_end:
            break

        # Evaporate
        evaporate()

        # Rank-based deposit
        all_solutions.sort(key=lambda x: x[0])
        w_rank = min(5, len(all_solutions))
        for rank in range(w_rank):
            num_b, sol = all_solutions[rank]
            dep = (w_rank - rank) / (w_rank * num_b)
            deposit_pheromone(sol, dep)

        # Global best deposit with increasing weight
        progress = min(1.0, iteration / 100.0)
        global_weight = 1.0 + 4.0 * progress
        deposit_pheromone(best_solution, global_weight / best_num_bins)

        # Stagnation detection
        if iteration_best is None or iteration_best_bins >= best_num_bins:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= 30:
            # Restart: reset pheromone
            if use_array:
                for i in range(n * n):
                    tau_arr[i] = tau_min
            else:
                tau_dict.clear()
            deposit_pheromone(best_solution, 2.0 / best_num_bins)
            stagnation_counter = 0
            # Adjust q0 to encourage more exploration
            q0_new = max(0.5, q0 - 0.05)

        iteration += 1

        # Calibrate after first iteration
        if iteration == 1:
            iter_time = time.time() - iter_start
            remaining_time = time_limit_end - elapsed()
            if iter_time > 0 and remaining_time > 0:
                estimated_iters = remaining_time / iter_time
                if estimated_iters < 10:
                    num_ants = max(2, num_ants // 3)
                elif estimated_iters < 20:
                    num_ants = max(2, num_ants // 2)
                elif estimated_iters < 40:
                    num_ants = max(3, int(num_ants * 0.7))
                elif estimated_iters > 500:
                    num_ants = min(num_ants * 3, 80)
                elif estimated_iters > 200:
                    num_ants = min(num_ants * 2, 60)
                elif estimated_iters > 100:
                    num_ants = min(int(num_ants * 1.5), 50)

    return make_result(best_solution)