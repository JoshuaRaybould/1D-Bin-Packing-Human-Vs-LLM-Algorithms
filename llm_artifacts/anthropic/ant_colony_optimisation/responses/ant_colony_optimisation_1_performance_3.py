import random
import time
from bisect import bisect_left, bisect_right, insort

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Sort items by decreasing weight
    indexed_weights = sorted(enumerate(weights), key=lambda x: -x[1])
    sorted_indices = [idx for idx, w in indexed_weights]
    sorted_weights = [w for idx, w in indexed_weights]

    total_weight = sum(weights)
    lower_bound_l1 = (total_weight + bin_capacity - 1) // bin_capacity

    # L2 lower bound
    half_cap = bin_capacity / 2.0
    large_count = 0
    remaining_space_in_large = 0
    small_total = 0
    for si in range(n):
        if sorted_weights[si] > half_cap:
            large_count += 1
            remaining_space_in_large += bin_capacity - sorted_weights[si]
        else:
            small_total += sorted_weights[si]

    leftover_small = max(0, small_total - remaining_space_in_large)
    l2_bound = large_count + ((leftover_small + bin_capacity - 1) // bin_capacity if leftover_small > 0 else 0)
    lower_bound = max(lower_bound_l1, l2_bound)

    # FFD
    def ffd_solution():
        bin_items = []
        bin_remaining = []
        # Use sorted remaining capacities for faster best-fit
        for si in range(n):
            w = sorted_weights[si]
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bin_remaining)):
                r = bin_remaining[b]
                if r >= w and r < best_remaining:
                    best_remaining = r
                    best_bin = b
            if best_bin == -1:
                bin_items.append([si])
                bin_remaining.append(bin_capacity - w)
            else:
                bin_items[best_bin].append(si)
                bin_remaining[best_bin] -= w
        return bin_items, len(bin_items)

    def bfd_solution():
        bin_items = []
        bin_remaining = []
        for si in range(n):
            w = sorted_weights[si]
            best_bin = -1
            best_fit = bin_capacity + 1
            for b in range(len(bin_remaining)):
                r = bin_remaining[b]
                if r >= w:
                    fit = r - w
                    if fit < best_fit:
                        best_fit = fit
                        best_bin = b
            if best_bin == -1:
                bin_items.append([si])
                bin_remaining.append(bin_capacity - w)
            else:
                bin_items[best_bin].append(si)
                bin_remaining[best_bin] -= w
        return bin_items, len(bin_items)

    ffd_bins, ffd_num = ffd_solution()
    bfd_bins, bfd_num = bfd_solution()

    if ffd_num <= bfd_num:
        best_solution = ffd_bins
        best_num_bins = ffd_num
    else:
        best_solution = bfd_bins
        best_num_bins = bfd_num

    def convert_solution(bin_items):
        packing = []
        bin_weights_list = []
        for b in bin_items:
            original_indices = [sorted_indices[si] for si in b]
            total_w = sum(sorted_weights[si] for si in b)
            packing.append(original_indices)
            bin_weights_list.append(total_w)
        return {"packing": packing, "bin_weights": bin_weights_list}

    if n <= 1 or best_num_bins <= 1 or best_num_bins <= lower_bound:
        return convert_solution(best_solution)

    # Pheromone storage: use flat array for speed
    use_array = n <= 2000
    if use_array:
        # Flat upper triangle: index(i,j) where i < j = i*n - i*(i+1)//2 + (j - i - 1)
        psize = n * (n - 1) // 2
        tau_max_init = 1.0 / (0.02 * best_num_bins)
        tau_min_init = max(tau_max_init / (2 * n), 0.01)
        tau_arr = [tau_max_init] * psize
        tau_max = tau_max_init
        tau_min = tau_min_init
        tau_init = tau_max

        def _idx(i, j):
            if i > j:
                i, j = j, i
            return i * n - (i * (i + 1) >> 1) + (j - i - 1)

        def get_tau(i, j):
            if i == j:
                return tau_init
            return tau_arr[_idx(i, j)]

        def set_tau_val(i, j, val):
            if i != j:
                tau_arr[_idx(i, j)] = val

        def evaporate_pheromone():
            factor = 1.0 - rho
            mn = tau_min
            for k in range(psize):
                v = tau_arr[k] * factor
                if v < mn:
                    v = mn
                tau_arr[k] = v

        def deposit_pheromone(solution, num_bins, multiplier=1.0):
            delta = multiplier / num_bins
            mx = tau_max
            for b in solution:
                lb = len(b)
                for ii in range(lb):
                    bi = b[ii]
                    for jj in range(ii + 1, lb):
                        bj = b[jj]
                        k = _idx(bi, bj)
                        v = tau_arr[k] + delta
                        if v > mx:
                            v = mx
                        tau_arr[k] = v

        def reset_pheromone():
            for k in range(psize):
                tau_arr[k] = tau_init
    else:
        tau_sparse = {}
        tau_max = 1.0 / (0.02 * best_num_bins)
        tau_min = max(tau_max / (2 * n), 0.01)
        tau_init = tau_max

        def get_tau(i, j):
            if i == j:
                return tau_init
            key = (i, j) if i < j else (j, i)
            return tau_sparse.get(key, tau_init)

        def evaporate_pheromone():
            nonlocal tau_sparse
            factor = 1.0 - rho
            mn = tau_min
            new_sparse = {}
            ti = tau_init
            for key, val in tau_sparse.items():
                new_val = val * factor
                if new_val < mn:
                    new_val = mn
                if abs(new_val - ti) > ti * 0.02:
                    new_sparse[key] = new_val
            tau_sparse = new_sparse

        def deposit_pheromone(solution, num_bins, multiplier=1.0):
            delta = multiplier / num_bins
            mx = tau_max
            ti = tau_init
            for b in solution:
                lb = len(b)
                for ii in range(lb):
                    bi = b[ii]
                    for jj in range(ii + 1, lb):
                        bj = b[jj]
                        key = (bi, bj) if bi < bj else (bj, bi)
                        old_val = tau_sparse.get(key, ti)
                        new_val = old_val + delta
                        if new_val > mx:
                            new_val = mx
                        tau_sparse[key] = new_val

        def reset_pheromone():
            tau_sparse.clear()

    # ACO parameters
    alpha = 1.0
    beta = 2.0
    rho = 0.02
    q0 = 0.3  # exploitation probability

    if n <= 100:
        num_ants = 20
    elif n <= 300:
        num_ants = 15
    elif n <= 600:
        num_ants = 10
    else:
        num_ants = 5

    def compute_mmas_bounds():
        nonlocal tau_max, tau_min, tau_init
        tau_max = 1.0 / (rho * best_num_bins)
        tau_min = max(tau_max / (2 * n), 0.01)
        tau_init = tau_max

    compute_mmas_bounds()

    # Precompute weights array for fast access
    sw = sorted_weights  # alias

    def construct_solution(use_q0=True, beta_val=2.0):
        # Bin-oriented construction
        # remaining items tracked via sorted list and set
        remaining = list(range(n))
        # Sort by weight ascending for binary search
        rem_by_weight = sorted(range(n), key=lambda x: sw[x])
        rem_weights_sorted = [sw[x] for x in rem_by_weight]
        # For O(1) removal tracking
        placed = [False] * n
        rem_count = n

        bins_result = []

        while rem_count > 0:
            # Find heaviest unplaced item
            while rem_by_weight and placed[rem_by_weight[-1]]:
                rem_by_weight.pop()
                rem_weights_sorted.pop()
            if not rem_by_weight:
                break

            seed_si = rem_by_weight.pop()
            seed_w = rem_weights_sorted.pop()
            placed[seed_si] = True
            rem_count -= 1

            current_bin = [seed_si]
            current_cap = bin_capacity - seed_w

            while current_cap > 0 and rem_count > 0:
                # Clean up trailing placed items
                while rem_by_weight and placed[rem_by_weight[-1]]:
                    rem_by_weight.pop()
                    rem_weights_sorted.pop()
                if not rem_by_weight:
                    break

                # Find feasible items: weight <= current_cap
                cutoff = bisect_right(rem_weights_sorted, current_cap)
                if cutoff == 0:
                    break

                # Build candidate list (limit size for speed)
                max_cands = 40
                candidates = []
                # Prefer tight-fitting items (near cutoff) and some random ones
                # Scan from cutoff-1 downward, skip placed
                count = 0
                tight_cands = []
                idx = cutoff - 1
                while idx >= 0 and count < max_cands // 2:
                    si = rem_by_weight[idx]
                    if not placed[si]:
                        tight_cands.append(si)
                        count += 1
                    idx -= 1

                # Also get some random candidates from the feasible set
                if cutoff > count * 2:
                    # sample random indices
                    num_rand = min(max_cands - count, max(5, max_cands // 2))
                    for _ in range(num_rand * 2):  # oversample to handle placed
                        if len(candidates) + len(tight_cands) >= max_cands:
                            break
                        ri = random.randint(0, cutoff - 1)
                        si = rem_by_weight[ri]
                        if not placed[si] and si not in tight_cands:
                            candidates.append(si)

                candidates = tight_cands + candidates
                if not candidates:
                    break

                # Score candidates
                cb = current_bin
                cb_len = len(cb)
                best_score = -1.0
                best_cand = -1

                if use_q0 and random.random() < q0:
                    # Exploitation: pick best deterministically
                    for csi in candidates:
                        cw = sw[csi]
                        remaining_after = current_cap - cw
                        eta = 1.0 / (remaining_after + 1.0)
                        pheromone_sum = 0.0
                        for item_in_bin in cb:
                            pheromone_sum += get_tau(csi, item_in_bin)
                        pheromone_avg = pheromone_sum / cb_len
                        score = (pheromone_avg ** alpha) * (eta ** beta_val)
                        if score > best_score:
                            best_score = score
                            best_cand = csi
                    chosen_si = best_cand
                else:
                    # Exploration: roulette wheel
                    scores = []
                    total_score = 0.0
                    for csi in candidates:
                        cw = sw[csi]
                        remaining_after = current_cap - cw
                        eta = 1.0 / (remaining_after + 1.0)
                        pheromone_sum = 0.0
                        for item_in_bin in cb:
                            pheromone_sum += get_tau(csi, item_in_bin)
                        pheromone_avg = pheromone_sum / cb_len
                        score = (pheromone_avg ** alpha) * (eta ** beta_val)
                        scores.append(score)
                        total_score += score

                    # Close bin option
                    fill_ratio = 1.0 - (current_cap / bin_capacity)
                    close_score = 0.05 * (fill_ratio ** 3)
                    total_score += close_score

                    if total_score <= 0:
                        break

                    r = random.random() * total_score
                    cumulative = 0.0
                    chosen_si = -1
                    for idx_c in range(len(scores)):
                        cumulative += scores[idx_c]
                        if cumulative >= r:
                            chosen_si = candidates[idx_c]
                            break

                    if chosen_si == -1:
                        break  # close bin

                placed[chosen_si] = True
                rem_count -= 1
                current_bin.append(chosen_si)
                current_cap -= sw[chosen_si]

            bins_result.append(current_bin)

        return bins_result, len(bins_result)

    # Seed pheromone from initial best solution
    deposit_pheromone(best_solution, best_num_bins, multiplier=3.0)

    # Main ACO loop
    iteration = 0
    no_improve_count = 0
    beta_val = 2.0
    beta_options = [1.5, 2.0, 2.5, 3.0, 3.5]
    beta_cycle = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break

        iteration_best = None
        iteration_best_bins = float('inf')

        for ant in range(num_ants):
            if time.time() - start_time >= time_limit * 0.95:
                break

            # Vary q0 per ant for diversity
            sol, num_bins = construct_solution(use_q0=True, beta_val=beta_val)

            if num_bins < iteration_best_bins:
                iteration_best_bins = num_bins
                iteration_best = sol

            if num_bins < best_num_bins:
                best_num_bins = num_bins
                best_solution = sol
                no_improve_count = 0
                compute_mmas_bounds()
                if use_array:
                    pass  # tau_init updated via compute_mmas_bounds

                if best_num_bins <= lower_bound:
                    break

        if best_num_bins <= lower_bound:
            break

        evaporate_pheromone()

        if iteration_best is not None:
            # Mix of global best and iteration best deposit
            if iteration % 3 == 0:
                deposit_pheromone(best_solution, best_num_bins, multiplier=1.0)
            else:
                deposit_pheromone(iteration_best, iteration_best_bins, multiplier=1.0)

        iteration += 1
        no_improve_count += 1

        # Stagnation: restart with different parameters
        if no_improve_count >= 80:
            no_improve_count = 0
            reset_pheromone()
            beta_cycle = (beta_cycle + 1) % len(beta_options)
            beta_val = beta_options[beta_cycle]
            compute_mmas_bounds()
            if use_array:
                for k in range(psize):
                    tau_arr[k] = tau_init
            deposit_pheromone(best_solution, best_num_bins, multiplier=3.0)

    return convert_solution(best_solution)
