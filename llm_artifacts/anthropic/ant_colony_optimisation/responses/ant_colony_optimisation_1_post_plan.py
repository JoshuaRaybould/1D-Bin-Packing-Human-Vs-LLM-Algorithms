import random
import time
from bisect import bisect_left, insort

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Sort items by decreasing weight, keep track of original indices
    indexed_weights = sorted(enumerate(weights), key=lambda x: -x[1])
    sorted_indices = [idx for idx, w in indexed_weights]
    sorted_weights = [w for idx, w in indexed_weights]
    
    # Precompute total weight and lower bounds
    total_weight = sum(weights)
    lower_bound_l1 = (total_weight + bin_capacity - 1) // bin_capacity
    
    # L2 lower bound
    half_cap = bin_capacity / 2.0
    large_items = []
    small_items = []
    for si in range(n):
        if sorted_weights[si] > half_cap:
            large_items.append(si)
        else:
            small_items.append(si)
    
    # Simple L2: each large item needs its own bin, small items fill remaining space
    remaining_space_in_large_bins = 0
    for si in large_items:
        remaining_space_in_large_bins += bin_capacity - sorted_weights[si]
    
    small_total = sum(sorted_weights[si] for si in small_items)
    leftover_small = max(0, small_total - remaining_space_in_large_bins)
    l2_bound = len(large_items) + (leftover_small + bin_capacity - 1) // bin_capacity if leftover_small > 0 else len(large_items)
    
    lower_bound = max(lower_bound_l1, l2_bound)
    
    # Helper: FFD solution
    def ffd_solution():
        bin_items = []
        bin_remaining = []
        for si in range(n):
            w = sorted_weights[si]
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w and bin_remaining[b] < best_remaining:
                    if bin_remaining[b] - w < best_remaining - w:
                        best_remaining = bin_remaining[b]
                        best_bin = b
            if best_bin == -1:
                bin_items.append([si])
                bin_remaining.append(bin_capacity - w)
            else:
                bin_items[best_bin].append(si)
                bin_remaining[best_bin] -= w
        return bin_items, len(bin_items)
    
    # Helper: BFD solution (best fit decreasing)
    def bfd_solution():
        bin_items = []
        bin_remaining = []
        for si in range(n):
            w = sorted_weights[si]
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w and bin_remaining[b] - w < best_remaining:
                    best_remaining = bin_remaining[b] - w
                    best_bin = b
            if best_bin == -1:
                bin_items.append([si])
                bin_remaining.append(bin_capacity - w)
            else:
                bin_items[best_bin].append(si)
                bin_remaining[best_bin] -= w
        return bin_items, len(bin_items)
    
    # Get initial solutions
    ffd_bins, ffd_num = ffd_solution()
    bfd_bins, bfd_num = bfd_solution()
    
    if ffd_num <= bfd_num:
        best_solution = ffd_bins
        best_num_bins = ffd_num
    else:
        best_solution = bfd_bins
        best_num_bins = bfd_num
    
    # Convert solution from sorted-order indices to original indices
    def convert_solution(bin_items):
        packing = []
        bin_weights_list = []
        for b in bin_items:
            original_indices = [sorted_indices[si] for si in b]
            total_w = sum(sorted_weights[si] for si in b)
            packing.append(original_indices)
            bin_weights_list.append(total_w)
        return {"packing": packing, "bin_weights": bin_weights_list}
    
    # For very small instances or trivial cases
    if n <= 1 or best_num_bins <= 1 or best_num_bins <= lower_bound:
        return convert_solution(best_solution)
    
    # ACO parameters
    alpha = 1.0
    beta = 2.5
    rho = 0.02
    close_bin_base = 0.1
    
    if n <= 200:
        num_ants = 15
    elif n <= 500:
        num_ants = 10
    else:
        num_ants = 5
    
    # MMAS pheromone
    tau_sparse = {}
    
    def compute_mmas_bounds():
        t_max = 1.0 / (rho * best_num_bins)
        t_min = max(t_max / (2 * n), 0.01)
        return t_max, t_min
    
    tau_max, tau_min = compute_mmas_bounds()
    tau_init = tau_max
    
    def get_tau(i, j):
        if i == j:
            return tau_init
        key = (i, j) if i < j else (j, i)
        return tau_sparse.get(key, tau_init)
    
    def set_tau(i, j, val):
        if i == j:
            return
        key = (i, j) if i < j else (j, i)
        val = max(tau_min, min(val, tau_max))
        # Only store if different from tau_init to save memory
        if abs(val - tau_init) > 0.001:
            tau_sparse[key] = val
        elif key in tau_sparse:
            del tau_sparse[key]
    
    def evaporate_pheromone():
        keys_to_delete = []
        for key, val in tau_sparse.items():
            new_val = (1 - rho) * val
            new_val = max(new_val, tau_min)
            if abs(new_val - tau_init) < tau_init * 0.05:
                keys_to_delete.append(key)
            else:
                tau_sparse[key] = new_val
        for key in keys_to_delete:
            del tau_sparse[key]
    
    def deposit_pheromone(solution, num_bins, multiplier=1.0):
        delta = multiplier / num_bins
        for b in solution:
            lb = len(b)
            for ii in range(lb):
                for jj in range(ii + 1, lb):
                    i, j = b[ii], b[jj]
                    key = (i, j) if i < j else (j, i)
                    old_val = tau_sparse.get(key, tau_init)
                    new_val = min(old_val + delta, tau_max)
                    new_val = max(new_val, tau_min)
                    if abs(new_val - tau_init) > 0.001:
                        tau_sparse[key] = new_val
                    elif key in tau_sparse:
                        del tau_sparse[key]
    
    def reset_pheromone():
        tau_sparse.clear()
    
    # Construct a solution using bin-oriented approach
    def construct_solution():
        # Maintain remaining items sorted by weight (ascending) for binary search
        # We store (weight, sorted_index) pairs
        remaining_sorted = []  # sorted by weight ascending
        for si in range(n):
            insort(remaining_sorted, (sorted_weights[si], si))
        remaining_set = set(range(n))
        
        bins_result = []
        
        while remaining_set:
            # Open a new bin; seed with the largest remaining item
            # largest is at the end of remaining_sorted
            seed_w, seed_si = remaining_sorted.pop()
            remaining_set.discard(seed_si)
            
            current_bin = [seed_si]
            current_remaining_cap = bin_capacity - seed_w
            
            while current_remaining_cap > 0 and remaining_set:
                # Find feasible candidates using binary search
                # Items with weight <= current_remaining_cap
                # remaining_sorted is sorted by weight ascending
                cutoff_idx = bisect_left(remaining_sorted, (current_remaining_cap + 1,)) 
                # All items at indices [0, cutoff_idx) are feasible
                
                if cutoff_idx == 0:
                    break
                
                feasible_count = cutoff_idx
                
                # If too many feasible, use candidate list (top 50 by heuristic)
                if feasible_count > 50:
                    # Pick candidates: best-fit heuristic - items closest to remaining cap
                    # Take top 25 by tightest fit (near end of feasible range)
                    # and 25 random from the rest
                    candidates = []
                    # Tightest fit: items near cutoff_idx - 1
                    tight_start = max(0, cutoff_idx - 25)
                    for idx in range(tight_start, cutoff_idx):
                        candidates.append(remaining_sorted[idx])
                    # Random sample from remaining feasible
                    if tight_start > 0:
                        num_random = min(25, tight_start)
                        random_indices = random.sample(range(tight_start), num_random)
                        for idx in random_indices:
                            candidates.append(remaining_sorted[idx])
                else:
                    candidates = [remaining_sorted[idx] for idx in range(feasible_count)]
                
                # Compute scores
                scores = []
                cb_len = len(current_bin)
                for cw, csi in candidates:
                    # Pheromone: average with items in current bin
                    pheromone_sum = 0.0
                    for item_in_bin in current_bin:
                        pheromone_sum += get_tau(csi, item_in_bin)
                    pheromone_avg = pheromone_sum / cb_len
                    
                    remaining_after = current_remaining_cap - cw
                    eta = 1.0 / (remaining_after + 1.0)
                    
                    score = (pheromone_avg ** alpha) * (eta ** beta)
                    scores.append(score)
                
                # Close bin pseudo-option
                fill_ratio = 1.0 - (current_remaining_cap / bin_capacity)
                close_score = close_bin_base * (fill_ratio ** 2)
                
                total = sum(scores) + close_score
                if total <= 0:
                    break
                
                r = random.random() * total
                cumulative = 0.0
                chosen_si = -1
                chosen_w = 0
                for idx_c, s in enumerate(scores):
                    cumulative += s
                    if cumulative >= r:
                        chosen_w, chosen_si = candidates[idx_c]
                        break
                
                if chosen_si == -1:
                    break  # close bin
                
                current_bin.append(chosen_si)
                current_remaining_cap -= chosen_w
                remaining_set.discard(chosen_si)
                # Remove from remaining_sorted
                rm_idx = bisect_left(remaining_sorted, (chosen_w, chosen_si))
                # Find exact match
                while rm_idx < len(remaining_sorted) and remaining_sorted[rm_idx] != (chosen_w, chosen_si):
                    rm_idx += 1
                if rm_idx < len(remaining_sorted):
                    remaining_sorted.pop(rm_idx)
            
            bins_result.append(current_bin)
        
        return bins_result, len(bins_result)
    
    # Seed pheromone from initial best solution
    deposit_pheromone(best_solution, best_num_bins, multiplier=2.0)
    
    # Main ACO loop
    max_iterations = 10000000
    iteration = 0
    no_improve_count = 0
    beta_options = [2.0, 2.5, 3.0]
    beta_idx = 0
    
    while iteration < max_iterations:
        if time.time() - start_time >= time_limit * 0.98:
            break
        
        iteration_best = None
        iteration_best_bins = float('inf')
        
        for ant in range(num_ants):
            if time.time() - start_time >= time_limit * 0.98:
                break
            
            sol, num_bins = construct_solution()
            
            if num_bins < iteration_best_bins:
                iteration_best_bins = num_bins
                iteration_best = sol
            
            if num_bins < best_num_bins:
                best_num_bins = num_bins
                best_solution = sol
                no_improve_count = 0
                # Recalculate MMAS bounds
                tau_max, tau_min = compute_mmas_bounds()
                tau_init = tau_max
                
                if best_num_bins <= lower_bound:
                    break
        
        if best_num_bins <= lower_bound:
            break
        
        # Evaporate
        evaporate_pheromone()
        
        # Deposit: alternate between iteration-best and global-best
        if iteration_best is not None:
            if iteration % 5 == 0:
                # Global best deposit
                deposit_pheromone(best_solution, best_num_bins)
            else:
                # Iteration best deposit
                deposit_pheromone(iteration_best, iteration_best_bins)
        
        iteration += 1
        no_improve_count += 1
        
        # Stagnation detection and restart
        if no_improve_count >= 100:
            no_improve_count = 0
            reset_pheromone()
            # Vary beta
            beta_idx = (beta_idx + 1) % len(beta_options)
            beta = beta_options[beta_idx]
            # Re-seed from global best
            tau_max, tau_min = compute_mmas_bounds()
            tau_init = tau_max
            deposit_pheromone(best_solution, best_num_bins, multiplier=2.0)
    
    return convert_solution(best_solution)
