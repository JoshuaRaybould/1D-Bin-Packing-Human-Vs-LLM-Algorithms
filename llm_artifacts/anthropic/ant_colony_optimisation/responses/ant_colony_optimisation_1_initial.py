import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Sort items by decreasing weight, keep track of original indices
    indexed_weights = sorted(enumerate(weights), key=lambda x: -x[1])
    sorted_indices = [idx for idx, w in indexed_weights]
    sorted_weights = [w for idx, w in indexed_weights]
    
    # Helper: FFD solution
    def ffd_solution():
        bins = []  # list of (list of sorted-order indices, remaining capacity)
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
    
    # Get initial FFD solution
    ffd_bins, ffd_num_bins = ffd_solution()
    
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
    
    best_solution = ffd_bins
    best_num_bins = ffd_num_bins
    
    # For very small instances or trivial cases
    if n <= 1 or best_num_bins <= 1:
        return convert_solution(best_solution)
    
    # ACO parameters
    # Pheromone on pairs: tau[i][j] = desirability of items i,j being in same bin
    # For large n, use dictionary-based sparse pheromone
    
    use_sparse = n > 300
    
    tau_max = 10.0
    tau_min = 0.1
    tau_init = 1.0
    
    alpha = 1.0  # pheromone importance
    beta = 2.0   # heuristic importance
    rho = 0.05   # evaporation rate
    
    num_ants = 10 if n <= 500 else 5
    
    if not use_sparse:
        # Dense pheromone matrix (symmetric, only upper triangle needed)
        tau = [[tau_init] * n for _ in range(n)]
    else:
        # Sparse pheromone using dict
        tau_sparse = {}
        
    def get_tau(i, j):
        if not use_sparse:
            return tau[i][j]
        else:
            key = (min(i, j), max(i, j))
            return tau_sparse.get(key, tau_init)
    
    def set_tau(i, j, val):
        if not use_sparse:
            tau[i][j] = val
            tau[j][i] = val
        else:
            key = (min(i, j), max(i, j))
            tau_sparse[key] = val
    
    def evaporate_pheromone():
        if not use_sparse:
            for i in range(n):
                for j in range(i, n):
                    new_val = (1 - rho) * tau[i][j]
                    new_val = max(new_val, tau_min)
                    tau[i][j] = new_val
                    tau[j][i] = new_val
        else:
            keys_to_delete = []
            for key, val in tau_sparse.items():
                new_val = (1 - rho) * val
                if abs(new_val - tau_init) < 0.01:
                    keys_to_delete.append(key)
                else:
                    new_val = max(new_val, tau_min)
                    tau_sparse[key] = new_val
            for key in keys_to_delete:
                del tau_sparse[key]
    
    def deposit_pheromone(solution, num_bins):
        # Deposit pheromone on pairs of items in the same bin
        deposit = 1.0 / num_bins
        for b in solution:
            for ii in range(len(b)):
                for jj in range(ii + 1, len(b)):
                    i, j = b[ii], b[jj]
                    old_val = get_tau(i, j)
                    new_val = min(old_val + deposit, tau_max)
                    set_tau(i, j, new_val)
    
    # Construct a solution for one ant
    def construct_solution():
        # Each bin: list of sorted-order item indices, remaining capacity
        bin_items_local = []
        bin_remaining_local = []
        
        for si in range(n):
            w = sorted_weights[si]
            if w > bin_capacity:
                # Item doesn't fit anywhere, skip (shouldn't happen in valid input)
                continue
            
            # Evaluate each existing bin
            candidates = []  # (bin_index, score)
            
            for b in range(len(bin_remaining_local)):
                if bin_remaining_local[b] >= w:
                    # Pheromone component: sum of pheromone between si and items in bin b
                    pheromone_sum = 0.0
                    items_in_bin = bin_items_local[b]
                    for item_in_bin in items_in_bin:
                        pheromone_sum += get_tau(si, item_in_bin)
                    
                    # Average pheromone (avoid bias toward fuller bins just from count)
                    if len(items_in_bin) > 0:
                        pheromone_avg = pheromone_sum / len(items_in_bin)
                    else:
                        pheromone_avg = tau_init
                    
                    # Heuristic: best fit - prefer bins with less remaining space after placing
                    remaining_after = bin_remaining_local[b] - w
                    # Higher eta for tighter fit
                    eta = 1.0 / (remaining_after + 1.0)
                    
                    score = (pheromone_avg ** alpha) * (eta ** beta)
                    candidates.append((b, score))
            
            # Also consider opening a new bin
            new_bin_eta = 1.0 / (bin_capacity - w + 1.0)
            new_bin_score = (tau_init ** alpha) * (new_bin_eta ** beta)
            # Scale down new bin probability to discourage unnecessary new bins
            new_bin_score *= 0.3
            candidates.append((-1, new_bin_score))
            
            # Roulette wheel selection
            total_score = sum(s for _, s in candidates)
            if total_score <= 0:
                # Open new bin
                bin_items_local.append([si])
                bin_remaining_local.append(bin_capacity - w)
                continue
            
            r = random.random() * total_score
            cumulative = 0.0
            chosen = candidates[-1][0]  # default: new bin
            for b_idx, score in candidates:
                cumulative += score
                if cumulative >= r:
                    chosen = b_idx
                    break
            
            if chosen == -1:
                bin_items_local.append([si])
                bin_remaining_local.append(bin_capacity - w)
            else:
                bin_items_local[chosen].append(si)
                bin_remaining_local[chosen] -= w
        
        return bin_items_local, len(bin_items_local)
    
    # Initialize pheromone from FFD solution
    deposit_pheromone(best_solution, best_num_bins)
    
    # Main ACO loop
    max_iterations = 1000000  # will be limited by time
    iteration = 0
    
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break
        
        iteration_best = None
        iteration_best_bins = float('inf')
        
        for ant in range(num_ants):
            # Check time periodically
            if ant % 3 == 0:
                if time.time() - start_time >= time_limit * 0.95:
                    break
            
            sol, num_bins = construct_solution()
            
            if num_bins < iteration_best_bins:
                iteration_best_bins = num_bins
                iteration_best = sol
            
            if num_bins < best_num_bins:
                best_num_bins = num_bins
                best_solution = sol
        
        # Evaporate
        evaporate_pheromone()
        
        # Deposit pheromone from iteration best
        if iteration_best is not None:
            deposit_pheromone(iteration_best, iteration_best_bins)
        
        # Elitist: also deposit from global best
        deposit_pheromone(best_solution, best_num_bins)
        
        iteration += 1
        
        # Lower bound check: if we hit the theoretical lower bound, stop
        total_weight = sum(weights)
        lower_bound = (total_weight + bin_capacity - 1) // bin_capacity
        if best_num_bins <= lower_bound:
            break
    
    return convert_solution(best_solution)
