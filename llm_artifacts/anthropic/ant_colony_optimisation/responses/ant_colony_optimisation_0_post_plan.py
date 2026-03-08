import random
import time
from bisect import bisect_left, insort

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    def make_result(packing_bins):
        result_packing = []
        result_weights = []
        for b in packing_bins:
            if b:
                result_packing.append(list(b))
                result_weights.append(sum(weights[i] for i in b))
        return {"packing": result_packing, "bin_weights": result_weights}
    
    # Best Fit Decreasing
    def bfd():
        sorted_items = sorted(range(n), key=lambda i: -weights[i])
        bins = []  # list of [remaining_cap, [items]]
        for item in sorted_items:
            w = weights[item]
            best_idx = -1
            best_remaining = bin_capacity + 1
            for j in range(len(bins)):
                rem = bins[j][0]
                if rem >= w and rem - w < best_remaining:
                    best_remaining = rem - w
                    best_idx = j
            if best_idx >= 0:
                bins[best_idx][0] -= w
                bins[best_idx][1].append(item)
            else:
                bins.append([bin_capacity - w, [item]])
        return [b[1] for b in bins]
    
    best_solution = bfd()
    best_num_bins = len(best_solution)
    
    if best_num_bins <= 1 or n <= 1:
        return make_result(best_solution)
    
    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity
    
    if best_num_bins <= lower_bound:
        return make_result(best_solution)
    
    # For very small n, just return BFD
    if n <= 5:
        return make_result(best_solution)
    
    # Try a few BFD variants with noise for potentially better starting solution
    elapsed = time.time() - start_time
    noise_budget = min(0.5, time_limit * 0.1)
    while time.time() - start_time - elapsed < noise_budget:
        sorted_items = sorted(range(n), key=lambda i: -weights[i] + random.gauss(0, bin_capacity * 0.05))
        bins = []
        for item in sorted_items:
            w = weights[item]
            best_idx = -1
            best_remaining = bin_capacity + 1
            for j in range(len(bins)):
                rem = bins[j][0]
                if rem >= w and rem - w < best_remaining:
                    best_remaining = rem - w
                    best_idx = j
            if best_idx >= 0:
                bins[best_idx][0] -= w
                bins[best_idx][1].append(item)
            else:
                bins.append([bin_capacity - w, [item]])
        sol = [b[1] for b in bins]
        if len(sol) < best_num_bins:
            best_num_bins = len(sol)
            best_solution = sol
            if best_num_bins <= lower_bound:
                return make_result(best_solution)
        if time.time() - start_time > noise_budget:
            break
    
    # ACO Parameters
    alpha = 1.0
    beta = 2.5
    rho = 0.02
    
    if n <= 50:
        num_ants = 30
    elif n <= 200:
        num_ants = 20
    elif n <= 500:
        num_ants = 15
    else:
        num_ants = 10
    
    # MMAS bounds
    def compute_bounds(best_bins):
        t_max = 1.0 / (rho * best_bins)
        t_min = t_max / (2.0 * n)
        return t_max, t_min
    
    tau_max, tau_min = compute_bounds(best_num_bins)
    tau_init = tau_max
    
    # Sparse pheromone dictionary
    # Key: (min(i,j), max(i,j)), Value: float
    tau = {}  # defaults to tau_init conceptually
    
    def get_tau(i, j):
        if i == j:
            return 0.0
        key = (i, j) if i < j else (j, i)
        return tau.get(key, tau_min)
    
    def set_tau(i, j, val):
        if i == j:
            return
        key = (i, j) if i < j else (j, i)
        tau[key] = val
    
    # Precompute sorted items by weight descending
    # For bin-centric construction, we need sorted remaining items
    # We'll use a list of (weight, index) sorted by weight ascending for bisect
    weight_arr = weights  # alias
    
    def construct_solution_bin_centric():
        """Bin-centric construction: fill bins one at a time."""
        # remaining items as a sorted list by weight ascending (for binary search)
        # We'll maintain a sorted list of (weight, item_index)
        remaining_sorted = sorted(range(n), key=lambda i: weight_arr[i])
        remaining_weights_sorted = [weight_arr[i] for i in remaining_sorted]
        remaining_set = set(range(n))
        
        bins_result = []
        
        while remaining_set:
            # Open a new bin
            cap = bin_capacity
            bin_items = []
            
            # Pick the heaviest remaining item as first item
            # The heaviest is at the end of remaining_sorted
            first_item = remaining_sorted[-1]
            w_first = weight_arr[first_item]
            
            # Remove from sorted list
            remaining_sorted.pop()
            remaining_weights_sorted.pop()
            remaining_set.remove(first_item)
            
            cap -= w_first
            bin_items.append(first_item)
            
            # Fill the bin
            while cap > 0 and remaining_set:
                # Find candidates: items with weight <= cap
                # Binary search for cutoff in remaining_weights_sorted
                cutoff_idx = bisect_left(remaining_weights_sorted, cap + 1)
                # All items at indices [0, cutoff_idx) fit
                
                if cutoff_idx == 0:
                    break  # No item fits
                
                num_candidates = cutoff_idx
                
                # If too many candidates, subsample top-50 by weight (heaviest)
                if num_candidates > 50:
                    # Take the heaviest 50 from the candidates
                    candidate_indices = list(range(max(0, cutoff_idx - 50), cutoff_idx))
                else:
                    candidate_indices = list(range(cutoff_idx))
                
                # Compute scores for each candidate
                scores = []
                
                # Items already in bin for pheromone computation
                # Limit to last K=5 items for speed
                pheromone_items = bin_items[-5:] if len(bin_items) > 5 else bin_items
                
                for ci in candidate_indices:
                    item_j = remaining_sorted[ci]
                    w_j = weight_arr[item_j]
                    
                    # Pheromone score: sum of pheromone between item_j and items in bin
                    pheromone_val = 0.0
                    for s in pheromone_items:
                        pheromone_val += get_tau(item_j, s)
                    if not pheromone_items:
                        pheromone_val = tau_min
                    
                    # Heuristic: (weight / remaining_capacity) ^ beta
                    heuristic = w_j / cap  # cap > 0 guaranteed
                    
                    score = (pheromone_val ** alpha) * (heuristic ** beta)
                    scores.append(score)
                
                # Roulette wheel selection
                total_score = sum(scores)
                if total_score <= 0:
                    chosen_ci_idx = len(candidate_indices) - 1  # heaviest
                else:
                    r = random.random() * total_score
                    cumsum = 0.0
                    chosen_ci_idx = len(scores) - 1
                    for pi, s in enumerate(scores):
                        cumsum += s
                        if cumsum >= r:
                            chosen_ci_idx = pi
                            break
                
                chosen_sorted_idx = candidate_indices[chosen_ci_idx]
                chosen_item = remaining_sorted[chosen_sorted_idx]
                chosen_weight = weight_arr[chosen_item]
                
                # Remove from sorted lists
                remaining_sorted.pop(chosen_sorted_idx)
                remaining_weights_sorted.pop(chosen_sorted_idx)
                remaining_set.remove(chosen_item)
                
                cap -= chosen_weight
                bin_items.append(chosen_item)
            
            bins_result.append(bin_items)
        
        return bins_result
    
    # Deposit pheromone for a solution
    def deposit_pheromone(sol, deposit_amount):
        for b in sol:
            nb = len(b)
            if nb <= 1:
                continue
            # For large bins, only deposit for pairs involving top-5 heaviest
            if nb > 10:
                # Find top-5 heaviest items in this bin
                sorted_bin = sorted(b, key=lambda i: -weight_arr[i])
                heavy = sorted_bin[:5]
                others = sorted_bin[5:]
                # Deposit between heavy items
                for ii in range(len(heavy)):
                    for jj in range(ii + 1, len(heavy)):
                        a, c = heavy[ii], heavy[jj]
                        key = (a, c) if a < c else (c, a)
                        old = tau.get(key, tau_min)
                        new_val = old + deposit_amount
                        if new_val > tau_max:
                            new_val = tau_max
                        tau[key] = new_val
                # Deposit between heavy and others
                for h in heavy:
                    for o in others:
                        key = (h, o) if h < o else (o, h)
                        old = tau.get(key, tau_min)
                        new_val = old + deposit_amount
                        if new_val > tau_max:
                            new_val = tau_max
                        tau[key] = new_val
            else:
                for ii in range(nb):
                    for jj in range(ii + 1, nb):
                        a, c = b[ii], b[jj]
                        key = (a, c) if a < c else (c, a)
                        old = tau.get(key, tau_min)
                        new_val = old + deposit_amount
                        if new_val > tau_max:
                            new_val = tau_max
                        tau[key] = new_val
    
    # Main ACO loop
    iteration = 0
    max_iterations = 10000
    stagnation_counter = 0
    
    while iteration < max_iterations:
        if time.time() - start_time >= time_limit * 0.95:
            break
        
        iter_start = time.time()
        
        iteration_best = None
        iteration_best_bins = float('inf')
        
        for ant in range(num_ants):
            if time.time() - start_time >= time_limit * 0.95:
                break
            
            sol = construct_solution_bin_centric()
            num_b = len(sol)
            
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
        
        # Evaporation: only over existing keys
        keys_to_delete = []
        for key in tau:
            tau[key] *= (1.0 - rho)
            if tau[key] < tau_min:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del tau[key]
        
        # Adaptive weighting: as iterations progress, global-best gets more weight
        # Start equal, end with global 3x
        progress = min(1.0, iteration / 200.0)
        global_weight = 1.0 + 2.0 * progress  # 1.0 -> 3.0
        iter_weight = 1.0
        
        # Deposit from iteration best
        if iteration_best is not None:
            deposit_iter = iter_weight * (1.0 / iteration_best_bins)
            deposit_pheromone(iteration_best, deposit_iter)
        
        # Deposit from global best
        deposit_global = global_weight * (1.0 / best_num_bins)
        deposit_pheromone(best_solution, deposit_global)
        
        # Stagnation detection
        if iteration_best is None or iteration_best_bins >= best_num_bins:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        if stagnation_counter >= 50:
            # Restart pheromone
            tau.clear()
            stagnation_counter = 0
        
        iteration += 1
        
        # Dynamic adjustment of num_ants after first iteration
        if iteration == 1:
            iter_time = time.time() - iter_start
            remaining_time = time_limit * 0.95 - (time.time() - start_time)
            if iter_time > 0 and remaining_time > 0:
                estimated_iters = remaining_time / iter_time
                if estimated_iters < 20:
                    # Reduce ants to get more iterations
                    num_ants = max(3, num_ants // 2)
                elif estimated_iters > 500:
                    # Can afford more ants
                    num_ants = min(num_ants * 2, 50)
        
        if time.time() - start_time >= time_limit * 0.95:
            break
    
    return make_result(best_solution)
