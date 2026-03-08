import random
import time
from bisect import bisect_right

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

    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity

    # Best Fit Decreasing
    def bfd(order=None):
        if order is None:
            sorted_items = sorted(range(n), key=lambda i: -weights[i])
        else:
            sorted_items = order
        bins_rem = []  # remaining capacities
        bins_items = []  # items in each bin
        for item in sorted_items:
            w = weights[item]
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
            sorted_items = sorted(range(n), key=lambda i: -weights[i])
        else:
            sorted_items = order
        bins_rem = []
        bins_items = []
        for item in sorted_items:
            w = weights[item]
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

    best_solution = bfd()
    best_num_bins = len(best_solution)

    if best_num_bins <= 1 or n <= 1:
        return make_result(best_solution)
    if best_num_bins <= lower_bound:
        return make_result(best_solution)

    # Try FFD too
    sol2 = ffd()
    if len(sol2) < best_num_bins:
        best_num_bins = len(sol2)
        best_solution = sol2
        if best_num_bins <= lower_bound:
            return make_result(best_solution)

    # Try noisy BFD variants
    noise_budget = min(1.0, time_limit * 0.08)
    w_arr = weights
    while elapsed() < noise_budget:
        noise_level = random.uniform(0.02, 0.15)
        order = sorted(range(n), key=lambda i: -w_arr[i] + random.gauss(0, bin_capacity * noise_level))
        sol = bfd(order)
        if len(sol) < best_num_bins:
            best_num_bins = len(sol)
            best_solution = sol
            if best_num_bins <= lower_bound:
                return make_result(best_solution)
        # Also try FFD with noise
        order2 = sorted(range(n), key=lambda i: -w_arr[i] + random.gauss(0, bin_capacity * noise_level))
        sol = ffd(order2)
        if len(sol) < best_num_bins:
            best_num_bins = len(sol)
            best_solution = sol
            if best_num_bins <= lower_bound:
                return make_result(best_solution)

    if n <= 3:
        return make_result(best_solution)

    # ACO Parameters
    alpha = 1.0
    beta = 2.0
    rho = 0.02

    # Use flat array for pheromone if n is manageable
    use_array = (n <= 4000)

    # MMAS bounds
    def compute_bounds(best_bins):
        t_max = 1.0 / (rho * max(best_bins, 1))
        t_min = t_max / (2.0 * n) if n > 0 else t_max / 2.0
        return t_max, t_min

    tau_max, tau_min = compute_bounds(best_num_bins)

    if use_array:
        # Pheromone stored in flat array, initialized to tau_min
        # tau[i*n+j] for i < j
        tau_arr = [tau_min] * (n * n)
    else:
        tau_dict = {}

    def get_tau(i, j):
        if i == j:
            return 0.0
        if use_array:
            return tau_arr[i * n + j]
        else:
            key = (i, j) if i < j else (j, i)
            return tau_dict.get(key, tau_min)

    def set_tau_val(i, j, val):
        if i == j:
            return
        if use_array:
            tau_arr[i * n + j] = val
            tau_arr[j * n + i] = val
        else:
            key = (i, j) if i < j else (j, i)
            tau_dict[key] = val

    # Deposit pheromone for the best solution to bootstrap
    deposit_val = 1.0 / best_num_bins
    for b in best_solution:
        if len(b) <= 1:
            continue
        seed = b[0]
        for k in range(1, len(b)):
            old = get_tau(seed, b[k])
            new_val = min(old + deposit_val, tau_max)
            set_tau_val(seed, b[k], new_val)

    # Precompute sorted indices by weight descending
    sorted_by_weight_desc = sorted(range(n), key=lambda i: -w_arr[i])
    weight_order = [0] * n  # rank of each item by weight (0 = heaviest)
    for rank, idx in enumerate(sorted_by_weight_desc):
        weight_order[idx] = rank

    def construct_solution():
        """Bin-centric construction using seed-based pheromone."""
        remaining = list(sorted_by_weight_desc)  # sorted by weight desc
        remaining_set = bytearray(n)  # 1 if remaining
        for i in range(n):
            remaining_set[i] = 1
        rem_weights = [w_arr[i] for i in remaining]  # parallel weights, sorted desc

        bins_result = []
        pos = 0  # current position tracking removed items

        while remaining:
            cap = bin_capacity
            # Pick heaviest remaining item as seed
            seed = remaining[0]
            seed_w = w_arr[seed]
            remaining_set[seed] = 0
            cap -= seed_w
            bin_items = [seed]

            if cap <= 0:
                # Remove seed from remaining
                remaining.pop(0)
                rem_weights.pop(0)
                bins_result.append(bin_items)
                continue

            # Build candidate list: items that fit (weight <= cap)
            # remaining is sorted by weight desc, so find cutoff
            # We need items with weight <= cap
            # Since remaining is sorted desc, items from some index onward have weight <= cap
            # Find first index where weight <= cap using linear scan from end or binary search

            # Actually remaining is sorted desc, so rem_weights is desc
            # Items with weight <= cap are at the tail
            # bisect: find leftmost position where weight <= cap
            # In a desc-sorted list, we want the first index where rem_weights[idx] <= cap
            
            # Fill the bin
            to_remove = [0]  # seed index 0
            
            while cap > 0 and len(remaining) > len(to_remove):
                # Gather candidates that fit
                candidates = []
                cand_indices = []  # indices in 'remaining' list
                max_cand = 60  # limit candidates for speed
                count = 0
                for ri in range(len(remaining)):
                    if ri in to_remove.__class__ and False:  # skip
                        pass
                    item = remaining[ri]
                    if remaining_set[item] == 0:
                        continue
                    wi = w_arr[item]
                    if wi > cap:
                        continue
                    candidates.append(item)
                    cand_indices.append(ri)
                    count += 1
                    if count >= max_cand:
                        break
                
                if not candidates:
                    break

                # Score candidates: pheromone(seed, item)^alpha * (weight/cap)^beta
                # Precompute
                if use_array:
                    seed_offset = seed * n
                    scores = []
                    for item in candidates:
                        p = tau_arr[seed_offset + item]
                        h = w_arr[item] / cap
                        scores.append(p * (h * h))  # alpha=1, beta=2
                else:
                    scores = []
                    for item in candidates:
                        p = get_tau(seed, item)
                        h = w_arr[item] / cap
                        scores.append(p * (h * h))

                # Roulette wheel selection
                total_score = sum(scores)
                if total_score <= 0:
                    # Pick heaviest (first candidate since remaining is sorted desc)
                    chosen_idx = 0
                else:
                    r = random.random() * total_score
                    cumsum = 0.0
                    chosen_idx = len(scores) - 1
                    for pi in range(len(scores)):
                        cumsum += scores[pi]
                        if cumsum >= r:
                            chosen_idx = pi
                            break

                chosen_item = candidates[chosen_idx]
                remaining_set[chosen_item] = 0
                cap -= w_arr[chosen_item]
                bin_items.append(chosen_item)

            # Rebuild remaining list (remove items placed in this bin)
            new_remaining = []
            new_rem_weights = []
            for i, item in enumerate(remaining):
                if remaining_set[item]:
                    new_remaining.append(item)
                    new_rem_weights.append(w_arr[item])
            remaining = new_remaining
            rem_weights = new_rem_weights

            bins_result.append(bin_items)

        return bins_result

    def construct_solution_fast():
        """Faster bin-centric construction."""
        remaining = list(sorted_by_weight_desc)
        in_remaining = bytearray(n)
        for i in range(n):
            in_remaining[i] = 1

        bins_result = []
        ridx = 0  # index into remaining

        while remaining:
            cap = bin_capacity
            seed = remaining[0]
            in_remaining[seed] = 0
            cap -= w_arr[seed]
            bin_items = [seed]

            if cap > 0 and len(remaining) > 1:
                # Collect fitting candidates
                # remaining is sorted desc by weight
                # Candidates are those with weight <= cap and still in_remaining
                candidates = []
                if use_array:
                    seed_off = seed * n
                    for item in remaining:
                        if in_remaining[item] == 0:
                            continue
                        if w_arr[item] <= cap:
                            candidates.append(item)
                            if len(candidates) >= 80:
                                break
                else:
                    for item in remaining:
                        if in_remaining[item] == 0:
                            continue
                        if w_arr[item] <= cap:
                            candidates.append(item)
                            if len(candidates) >= 80:
                                break

                while cap > 0 and candidates:
                    nc = len(candidates)
                    if nc == 1:
                        chosen_item = candidates[0]
                    else:
                        # Compute scores
                        if use_array:
                            seed_off = seed * n
                            scores = [0.0] * nc
                            inv_cap = 1.0 / cap
                            for ci in range(nc):
                                item = candidates[ci]
                                p = tau_arr[seed_off + item]
                                h = w_arr[item] * inv_cap
                                scores[ci] = p * h * h
                        else:
                            scores = [0.0] * nc
                            inv_cap = 1.0 / cap
                            for ci in range(nc):
                                item = candidates[ci]
                                p = get_tau(seed, item)
                                h = w_arr[item] * inv_cap
                                scores[ci] = p * h * h

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

                    in_remaining[chosen_item] = 0
                    cw = w_arr[chosen_item]
                    cap -= cw
                    bin_items.append(chosen_item)

                    # Update candidates: remove chosen and items that no longer fit
                    if cap > 0:
                        new_cands = []
                        for item in candidates:
                            if in_remaining[item] and w_arr[item] <= cap:
                                new_cands.append(item)
                        candidates = new_cands
                    else:
                        break

            # Rebuild remaining
            new_remaining = []
            for item in remaining:
                if in_remaining[item]:
                    new_remaining.append(item)
            remaining = new_remaining
            bins_result.append(bin_items)

        return bins_result

    # Deposit pheromone for a solution (seed-based: only between seed and other items)
    def deposit_pheromone(sol, amount):
        if use_array:
            for b in sol:
                if len(b) <= 1:
                    continue
                seed = b[0]  # heaviest item = seed
                seed_off = seed * n
                for k in range(1, len(b)):
                    item = b[k]
                    v = tau_arr[seed_off + item] + amount
                    if v > tau_max:
                        v = tau_max
                    tau_arr[seed_off + item] = v
                    tau_arr[item * n + seed] = v
                # Also deposit between all pairs for small bins
                if len(b) <= 8:
                    for ii in range(len(b)):
                        for jj in range(ii + 1, len(b)):
                            a, c = b[ii], b[jj]
                            v = tau_arr[a * n + c] + amount
                            if v > tau_max:
                                v = tau_max
                            tau_arr[a * n + c] = v
                            tau_arr[c * n + a] = v
        else:
            for b in sol:
                if len(b) <= 1:
                    continue
                seed = b[0]
                for k in range(1, len(b)):
                    old = get_tau(seed, b[k])
                    new_val = min(old + amount, tau_max)
                    set_tau_val(seed, b[k], new_val)

    # Evaporation
    def evaporate():
        if use_array:
            factor = 1.0 - rho
            tm = tau_min
            for i in range(n * n):
                v = tau_arr[i] * factor
                if v < tm:
                    v = tm
                tau_arr[i] = v
        else:
            keys_to_delete = []
            factor = 1.0 - rho
            for key in tau_dict:
                tau_dict[key] *= factor
                if tau_dict[key] < tau_min:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del tau_dict[key]

    # Adaptive ant count based on problem size and timing
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
    time_limit_95 = time_limit * 0.97

    while True:
        if elapsed() >= time_limit_95:
            break

        iter_start = time.time()

        iteration_best = None
        iteration_best_bins = float('inf')
        all_solutions = []

        for ant in range(num_ants):
            if elapsed() >= time_limit_95:
                break

            sol = construct_solution_fast()
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

        # Evaporate
        evaporate()

        # Rank-based deposit: top-w ants + global best
        all_solutions.sort(key=lambda x: x[0])
        w_rank = min(5, len(all_solutions))
        for rank in range(w_rank):
            num_b, sol = all_solutions[rank]
            dep = (w_rank - rank) / (w_rank * num_b)
            deposit_pheromone(sol, dep)

        # Global best deposit (stronger)
        progress = min(1.0, iteration / 150.0)
        global_weight = 1.0 + 3.0 * progress
        deposit_pheromone(best_solution, global_weight / best_num_bins)

        # Stagnation
        if iteration_best is None or iteration_best_bins >= best_num_bins:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= 40:
            # Soft restart: reset pheromone to tau_min
            if use_array:
                for i in range(n * n):
                    tau_arr[i] = tau_min
            else:
                tau_dict.clear()
            # Re-deposit from best solution
            deposit_pheromone(best_solution, 2.0 / best_num_bins)
            stagnation_counter = 0

        iteration += 1

        # After first iteration, calibrate ant count
        if iteration == 1:
            iter_time = time.time() - iter_start
            remaining_time = time_limit_95 - elapsed()
            if iter_time > 0 and remaining_time > 0:
                estimated_iters = remaining_time / iter_time
                if estimated_iters < 15:
                    num_ants = max(3, num_ants // 2)
                elif estimated_iters < 30:
                    num_ants = max(3, int(num_ants * 0.7))
                elif estimated_iters > 300:
                    num_ants = min(num_ants * 2, 60)
                elif estimated_iters > 150:
                    num_ants = min(int(num_ants * 1.5), 50)

    return make_result(best_solution)