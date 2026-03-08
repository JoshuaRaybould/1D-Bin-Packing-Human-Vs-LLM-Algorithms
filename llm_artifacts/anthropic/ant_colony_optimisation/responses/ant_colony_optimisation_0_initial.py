import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Helper: build solution result dict
    def make_result(packing_bins):
        result_packing = []
        result_weights = []
        for b in packing_bins:
            if b:
                result_packing.append(list(b))
                result_weights.append(sum(weights[i] for i in b))
        return {"packing": result_packing, "bin_weights": result_weights}
    
    # First Fit Decreasing as baseline
    def ffd():
        sorted_items = sorted(range(n), key=lambda i: -weights[i])
        bins = []  # list of (remaining_cap, [items])
        for item in sorted_items:
            w = weights[item]
            best_idx = -1
            best_remaining = bin_capacity + 1
            for j, (rem, _) in enumerate(bins):
                if rem >= w and rem - w < best_remaining:
                    best_remaining = rem - w
                    best_idx = j
            if best_idx >= 0:
                bins[best_idx] = (bins[best_idx][0] - w, bins[best_idx][1])
                bins[best_idx][1].append(item)
            else:
                bins.append((bin_capacity - w, [item]))
        return [b[1] for b in bins]
    
    # Get FFD solution as initial best
    best_solution = ffd()
    best_num_bins = len(best_solution)
    
    if best_num_bins <= 1 or n <= 1:
        return make_result(best_solution)
    
    # Lower bound
    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity
    
    if best_num_bins <= lower_bound:
        return make_result(best_solution)
    
    # ACO parameters
    if n <= 50:
        num_ants = 20
    elif n <= 200:
        num_ants = 15
    elif n <= 500:
        num_ants = 10
    else:
        num_ants = 5
    
    alpha = 1.0  # pheromone importance
    beta = 2.0   # heuristic importance
    rho = 0.05   # evaporation rate
    
    # For large n, use a simplified pheromone model
    # Pheromone on item pairs: tau[i][j] = desirability of i,j in same bin
    # For efficiency with large n, we use a flat array approach
    
    use_pairwise = n <= 300
    
    if use_pairwise:
        # Initialize pheromone matrix (symmetric, only store i < j)
        tau_init = 1.0 / best_num_bins
        # Use dict for sparse or flat array
        tau = [[tau_init] * n for _ in range(n)]
    else:
        # For large n, use a group-based pheromone
        # Pheromone on (item, group_id) where group_id comes from best solution
        # We'll use a simpler approach: pheromone on item-to-template-bin
        tau_init = 1.0 / best_num_bins
        # We'll just track which items go well together via a simpler mechanism
        # Use item-bin pheromone from the best solution's bin assignments
        max_bins_track = best_num_bins + 10
        tau_item_bin = [[tau_init] * max_bins_track for _ in range(n)]
    
    def construct_solution_pairwise(item_order):
        """Construct a solution using pairwise pheromone."""
        bins_remaining = []  # remaining capacity
        bins_items = []      # items in each bin
        
        for item in item_order:
            w = weights[item]
            if w > bin_capacity:
                continue
            
            # Evaluate each existing bin + new bin option
            num_bins = len(bins_remaining)
            
            # Limit candidates for efficiency
            candidates = []
            for j in range(num_bins):
                if bins_remaining[j] >= w:
                    candidates.append(j)
            
            if not candidates:
                # Must open new bin
                bins_remaining.append(bin_capacity - w)
                bins_items.append([item])
                continue
            
            # Calculate probabilities
            probs = []
            for j in candidates:
                # Pheromone: average pheromone between item and items in bin j
                bin_items_j = bins_items[j]
                if bin_items_j:
                    pheromone_val = 0.0
                    for bi in bin_items_j:
                        pheromone_val += tau[item][bi]
                    pheromone_val /= len(bin_items_j)
                else:
                    pheromone_val = tau_init
                
                # Heuristic: prefer tighter fit
                remaining_after = bins_remaining[j] - w
                # Heuristic value: higher when remaining is smaller (tighter fit)
                heuristic = 1.0 / (1.0 + remaining_after)
                
                prob = (pheromone_val ** alpha) * (heuristic ** beta)
                probs.append(prob)
            
            # Add option to open new bin
            new_bin_pheromone = tau_init
            new_bin_heuristic = 1.0 / (1.0 + bin_capacity - w)
            new_bin_prob = (new_bin_pheromone ** alpha) * (new_bin_heuristic ** beta)
            candidates.append(-1)  # sentinel for new bin
            probs.append(new_bin_prob)
            
            # Roulette wheel selection
            total_prob = sum(probs)
            if total_prob <= 0:
                chosen_idx = 0
            else:
                r = random.random() * total_prob
                cumsum = 0.0
                chosen_idx = len(probs) - 1
                for pi, p in enumerate(probs):
                    cumsum += p
                    if cumsum >= r:
                        chosen_idx = pi
                        break
            
            chosen_bin = candidates[chosen_idx]
            if chosen_bin == -1:
                bins_remaining.append(bin_capacity - w)
                bins_items.append([item])
            else:
                bins_remaining[chosen_bin] -= w
                bins_items[chosen_bin].append(item)
        
        return bins_items
    
    def construct_solution_large(item_order):
        """Construct solution for large instances using simpler heuristic with randomization."""
        bins_remaining = []
        bins_items = []
        
        for item in item_order:
            w = weights[item]
            if w > bin_capacity:
                continue
            
            # Find feasible bins
            feasible = []
            for j in range(len(bins_remaining)):
                if bins_remaining[j] >= w:
                    feasible.append(j)
            
            if not feasible:
                bins_remaining.append(bin_capacity - w)
                bins_items.append([item])
                continue
            
            # Score each feasible bin
            scores = []
            for j in feasible:
                remaining_after = bins_remaining[j] - w
                heuristic = 1.0 / (1.0 + remaining_after)
                
                # Pheromone from item-bin matrix
                if j < len(tau_item_bin[item]):
                    pheromone_val = tau_item_bin[item][j]
                else:
                    pheromone_val = tau_init
                
                score = (pheromone_val ** alpha) * (heuristic ** beta)
                scores.append(score)
            
            # New bin option
            new_heuristic = 1.0 / (1.0 + bin_capacity - w)
            nb_idx = len(bins_remaining)
            if nb_idx < len(tau_item_bin[item]):
                new_pheromone = tau_item_bin[item][nb_idx]
            else:
                new_pheromone = tau_init
            new_score = (new_pheromone ** alpha) * (new_heuristic ** beta)
            
            feasible.append(-1)
            scores.append(new_score)
            
            total_score = sum(scores)
            if total_score <= 0:
                chosen_idx = 0
            else:
                r = random.random() * total_score
                cumsum = 0.0
                chosen_idx = len(scores) - 1
                for pi, s in enumerate(scores):
                    cumsum += s
                    if cumsum >= r:
                        chosen_idx = pi
                        break
            
            chosen_bin = feasible[chosen_idx]
            if chosen_bin == -1:
                bins_remaining.append(bin_capacity - w)
                bins_items.append([item])
            else:
                bins_remaining[chosen_bin] -= w
                bins_items[chosen_bin].append(item)
        
        return bins_items
    
    def update_pheromone_pairwise(solutions, num_bins_list):
        """Update pheromone matrix based on solutions."""
        # Evaporation
        for i in range(n):
            for j in range(n):
                tau[i][j] *= (1.0 - rho)
        
        # Deposit pheromone for best solutions
        # Use iteration best and global best
        for sol_idx, sol in enumerate(solutions):
            num_b = num_bins_list[sol_idx]
            deposit = 1.0 / num_b
            for b in sol:
                items_in_bin = b
                for ii in range(len(items_in_bin)):
                    for jj in range(ii + 1, len(items_in_bin)):
                        a, c = items_in_bin[ii], items_in_bin[jj]
                        tau[a][c] += deposit
                        tau[c][a] += deposit
    
    def update_pheromone_large(solutions, num_bins_list):
        """Update pheromone for large instances."""
        nonlocal tau_item_bin, max_bins_track
        
        # Check if we need to expand
        max_needed = max(len(sol) for sol in solutions) if solutions else max_bins_track
        if max_needed > max_bins_track:
            new_max = max_needed + 10
            for i in range(n):
                tau_item_bin[i].extend([tau_init] * (new_max - max_bins_track))
            max_bins_track = new_max
        
        # Evaporation
        for i in range(n):
            for j in range(max_bins_track):
                tau_item_bin[i][j] *= (1.0 - rho)
        
        # Deposit
        for sol_idx, sol in enumerate(solutions):
            num_b = num_bins_list[sol_idx]
            deposit = 1.0 / num_b
            for bin_idx, b in enumerate(sol):
                if bin_idx < max_bins_track:
                    for item in b:
                        tau_item_bin[item][bin_idx] += deposit
    
    # Sort items by decreasing weight as base order
    sorted_items = sorted(range(n), key=lambda i: -weights[i])
    
    iteration = 0
    max_iterations = 1000
    
    while iteration < max_iterations:
        if time.time() - start_time >= time_limit * 0.95:
            break
        
        iteration_best = None
        iteration_best_bins = float('inf')
        all_solutions = []
        all_num_bins = []
        
        for ant in range(num_ants):
            if time.time() - start_time >= time_limit * 0.95:
                break
            
            # Create item order: mostly sorted by weight but with some randomization
            # Shuffle slightly to explore different orderings
            if random.random() < 0.3:
                # Random permutation
                item_order = list(range(n))
                random.shuffle(item_order)
                # Sort by weight with some noise
                item_order.sort(key=lambda i: -weights[i] + random.gauss(0, bin_capacity * 0.05))
            else:
                item_order = list(sorted_items)
            
            if use_pairwise:
                sol = construct_solution_pairwise(item_order)
            else:
                sol = construct_solution_large(item_order)
            
            num_b = len(sol)
            
            if num_b < iteration_best_bins:
                iteration_best_bins = num_b
                iteration_best = sol
            
            if num_b < best_num_bins:
                best_num_bins = num_b
                best_solution = sol
                if best_num_bins <= lower_bound:
                    return make_result(best_solution)
        
        # Update pheromone with iteration best and global best
        solutions_to_update = []
        nums_to_update = []
        
        if iteration_best is not None:
            solutions_to_update.append(iteration_best)
            nums_to_update.append(iteration_best_bins)
        
        # Global best gets extra deposit (added twice)
        solutions_to_update.append(best_solution)
        nums_to_update.append(best_num_bins)
        solutions_to_update.append(best_solution)
        nums_to_update.append(best_num_bins)
        
        if use_pairwise:
            update_pheromone_pairwise(solutions_to_update, nums_to_update)
        else:
            update_pheromone_large(solutions_to_update, nums_to_update)
        
        iteration += 1
        
        if time.time() - start_time >= time_limit * 0.95:
            break
    
    return make_result(best_solution)
