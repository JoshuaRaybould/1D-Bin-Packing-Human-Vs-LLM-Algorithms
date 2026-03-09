import random
import time
from bisect import bisect_right, insort
from array import array

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Sort items by decreasing weight
    indexed_weights = sorted(enumerate(weights), key=lambda x: -x[1])
    sorted_indices = [idx for idx, w in indexed_weights]
    sw = [w for idx, w in indexed_weights]  # sorted weights descending

    total_weight = sum(weights)
    lower_bound_l1 = (total_weight + bin_capacity - 1) // bin_capacity

    # L2 lower bound
    half_cap = bin_capacity / 2.0
    large_count = 0
    remaining_space_in_large = 0
    small_total = 0
    for si in range(n):
        if sw[si] > half_cap:
            large_count += 1
            remaining_space_in_large += bin_capacity - sw[si]
        else:
            small_total += sw[si]

    leftover_small = max(0, small_total - remaining_space_in_large)
    l2_bound = large_count + ((leftover_small + bin_capacity - 1) // bin_capacity if leftover_small > 0 else 0)
    lower_bound = max(lower_bound_l1, l2_bound)

    # BFD solution using bisect for O(n log n)
    def bfd_sorted():
        bin_items = []
        # Maintain sorted list of (remaining_capacity, bin_index)
        # Use two parallel lists for bisect
        sorted_rem = []  # sorted remaining capacities
        sorted_bidx = []  # corresponding bin indices
        bin_rem = []
        
        for si in range(n):
            w = sw[si]
            # Find leftmost bin with remaining >= w using bisect
            pos = bisect_right(sorted_rem, w - 1)  # first >= w... no
            # We want best fit = smallest remaining >= w
            # sorted_rem is sorted ascending
            # bisect_left for w gives first position >= w
            from bisect import bisect_left
            pos = bisect_left(sorted_rem, w)
            if pos < len(sorted_rem):
                # Best fit: smallest remaining >= w
                bidx = sorted_bidx[pos]
                old_rem = sorted_rem[pos]
                # Remove old entry
                sorted_rem.pop(pos)
                sorted_bidx.pop(pos)
                # Insert new remaining
                new_rem = old_rem - w
                bin_items[bidx].append(si)
                bin_rem[bidx] = new_rem
                if new_rem > 0:
                    ins_pos = bisect_left(sorted_rem, new_rem)
                    sorted_rem.insert(ins_pos, new_rem)
                    sorted_bidx.insert(ins_pos, bidx)
            else:
                bidx = len(bin_items)
                bin_items.append([si])
                new_rem = bin_capacity - w
                bin_rem.append(new_rem)
                if new_rem > 0:
                    ins_pos = bisect_left(sorted_rem, new_rem)
                    sorted_rem.insert(ins_pos, new_rem)
                    sorted_bidx.insert(ins_pos, bidx)
        return bin_items, len(bin_items)

    # FFD solution using bisect (first fit decreasing = first bin that fits)
    def ffd_sorted():
        from bisect import bisect_left
        bin_items = []
        bin_rem = []
        # For FFD, we just need first bin that fits - use simple list
        # But with sorted remaining we can find largest remaining first
        # Actually FFD = first fit, so just scan bins in order
        # For speed with large n, just do simple FFD
        for si in range(n):
            w = sw[si]
            placed = False
            for b in range(len(bin_rem)):
                if bin_rem[b] >= w:
                    bin_items[b].append(si)
                    bin_rem[b] -= w
                    placed = True
                    break
            if not placed:
                bin_items.append([si])
                bin_rem.append(bin_capacity - w)
        return bin_items, len(bin_items)

    bfd_bins, bfd_num = bfd_sorted()
    
    best_solution = bfd_bins
    best_num_bins = bfd_num

    # Also try FFD for small instances
    if n <= 5000:
        ffd_bins, ffd_num = ffd_sorted()
        if ffd_num < best_num_bins:
            best_solution = ffd_bins
            best_num_bins = ffd_num

    def convert_solution(bin_items):
        packing = []
        bin_weights_list = []
        for b in bin_items:
            original_indices = [sorted_indices[si] for si in b]
            total_w = sum(sw[si] for si in b)
            packing.append(original_indices)
            bin_weights_list.append(total_w)
        return {"packing": packing, "bin_weights": bin_weights_list}

    if n <= 1 or best_num_bins <= 1 or best_num_bins <= lower_bound:
        return convert_solution(best_solution)

    # Check time
    if time.time() - start_time >= time_limit * 0.90:
        return convert_solution(best_solution)

    # Pheromone storage
    use_array = n <= 3000
    
    rho = 0.02
    
    if use_array:
        psize = n * (n - 1) // 2
        tau_max = 1.0 / (rho * best_num_bins)
        tau_min = max(tau_max / (2 * n), 0.001)
        tau_init = tau_max

        tau_arr = array('f', [tau_max] * psize)

        def _idx(i, j):
            if i > j:
                i, j = j, i
            return i * n - (i * (i + 1) >> 1) + (j - i - 1)

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

        def get_tau_sum_bin(csi, current_bin):
            s = 0.0
            for item in current_bin:
                if csi == item:
                    s += tau_init
                else:
                    s += tau_arr[_idx(csi, item)]
            return s
    else:
        tau_sparse = {}
        tau_max = 1.0 / (rho * best_num_bins)
        tau_min = max(tau_max / (2 * n), 0.001)
        tau_init = tau_max

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
                if abs(new_val - ti) > ti * 0.01:
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

        def get_tau_sum_bin(csi, current_bin):
            ti = tau_init
            s = 0.0
            for item in current_bin:
                if csi == item:
                    s += ti
                else:
                    key = (csi, item) if csi < item else (item, csi)
                    s += tau_sparse.get(key, ti)
            return s

    def compute_mmas_bounds():
        nonlocal tau_max, tau_min, tau_init
        tau_max = 1.0 / (rho * best_num_bins)
        tau_min = max(tau_max / (2 * n), 0.001)
        tau_init = tau_max

    compute_mmas_bounds()

    # ACO parameters
    alpha = 1.0
    q0 = 0.4

    if n <= 50:
        num_ants = 30
    elif n <= 150:
        num_ants = 20
    elif n <= 400:
        num_ants = 15
    elif n <= 800:
        num_ants = 10
    elif n <= 2000:
        num_ants = 5
    else:
        num_ants = 3

    # Precompute: items sorted ascending by weight for bisect
    asc_order = list(range(n - 1, -1, -1))  # indices in sw, ascending weight order
    asc_weights = [sw[i] for i in asc_order]  # ascending weights

    def construct_solution(beta_val, use_q0_flag):
        placed = bytearray(n)
        bins_result = []
        rem_count = n
        
        # Maintain a sorted list of remaining items by weight for efficient candidate finding
        # Use asc_weights structure: items are in ascending weight order
        # We'll maintain a list of (weight, si) sorted by weight ascending
        rem_sorted_w = list(asc_weights)  # copy
        rem_sorted_si = list(asc_order)   # copy
        
        while rem_count > 0:
            # Find heaviest unplaced item (last in sorted list)
            while rem_sorted_w and placed[rem_sorted_si[-1]]:
                rem_sorted_w.pop()
                rem_sorted_si.pop()
            if not rem_sorted_w:
                break
            
            seed_si = rem_sorted_si.pop()
            seed_w = rem_sorted_w.pop()
            placed[seed_si] = 1
            rem_count -= 1
            
            current_bin = [seed_si]
            cap = bin_capacity - seed_w
            
            while cap > 0 and rem_count > 0:
                # Clean trailing placed items
                while rem_sorted_w and placed[rem_sorted_si[-1]]:
                    rem_sorted_w.pop()
                    rem_sorted_si.pop()
                if not rem_sorted_w:
                    break
                
                # Find feasible items: weight <= cap
                cutoff = bisect_right(rem_sorted_w, cap)
                if cutoff == 0:
                    break
                
                # Build candidate list
                max_cands = 30
                candidates = []
                
                # Get tight-fit candidates (largest weight that fits) from top
                idx = cutoff - 1
                count = 0
                while idx >= 0 and count < max_cands:
                    si = rem_sorted_si[idx]
                    if not placed[si]:
                        candidates.append(si)
                        count += 1
                    idx -= 1
                
                if not candidates:
                    break
                
                # Score candidates
                cb = current_bin
                cb_len = len(cb)
                
                if use_q0_flag and random.random() < q0:
                    # Exploitation
                    best_score = -1.0
                    best_cand = candidates[0]
                    for csi in candidates:
                        cw = sw[csi]
                        remaining_after = cap - cw
                        eta = 1.0 / (remaining_after + 1.0)
                        phe = get_tau_sum_bin(csi, cb) / cb_len
                        score = phe * (eta ** beta_val)
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
                        remaining_after = cap - cw
                        eta = 1.0 / (remaining_after + 1.0)
                        phe = get_tau_sum_bin(csi, cb) / cb_len
                        score = phe * (eta ** beta_val)
                        scores.append(score)
                        total_score += score
                    
                    # Close bin option
                    fill_ratio = 1.0 - (cap / bin_capacity)
                    close_score = 0.02 * (fill_ratio ** 4)
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
                
                placed[chosen_si] = 1
                rem_count -= 1
                current_bin.append(chosen_si)
                cap -= sw[chosen_si]
            
            bins_result.append(current_bin)
        
        return bins_result, len(bins_result)

    # Seed pheromone from initial best solution
    deposit_pheromone(best_solution, best_num_bins, multiplier=3.0)

    # Main ACO loop
    iteration = 0
    no_improve_count = 0
    beta_val = 2.5
    beta_options = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    beta_cycle = 2  # start at 2.5
    
    stagnation_limit = 60 if n <= 300 else 40

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break

        iteration_best = None
        iteration_best_bins = float('inf')

        for ant in range(num_ants):
            if time.time() - start_time >= time_limit * 0.95:
                break

            # Vary parameters per ant
            use_q0_ant = True
            b_val = beta_val
            if ant == 0:
                # One pure greedy ant
                b_val = beta_val + 1.0
            elif ant == 1:
                # One more exploratory ant  
                b_val = max(1.0, beta_val - 0.5)
                use_q0_ant = False

            sol, num_bins = construct_solution(b_val, use_q0_ant)

            if num_bins < iteration_best_bins:
                iteration_best_bins = num_bins
                iteration_best = sol

            if num_bins < best_num_bins:
                best_num_bins = num_bins
                best_solution = sol
                no_improve_count = 0
                compute_mmas_bounds()

                if best_num_bins <= lower_bound:
                    break

        if best_num_bins <= lower_bound:
            break

        evaporate_pheromone()

        if iteration_best is not None:
            # Deposit: alternate between global best and iteration best
            if no_improve_count > 20 or iteration % 5 == 0:
                deposit_pheromone(best_solution, best_num_bins, multiplier=1.5)
            else:
                deposit_pheromone(iteration_best, iteration_best_bins, multiplier=1.0)
                if iteration_best_bins > best_num_bins:
                    deposit_pheromone(best_solution, best_num_bins, multiplier=0.5)

        iteration += 1
        no_improve_count += 1

        # Stagnation: restart
        if no_improve_count >= stagnation_limit:
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