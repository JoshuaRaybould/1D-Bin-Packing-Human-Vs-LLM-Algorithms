import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Helper: decode a priority vector into a packing using First Fit
    def decode_first_fit(priorities):
        # Sort items by priority (descending) to get order
        order = sorted(range(n), key=lambda i: -priorities[i])
        bins = []  # list of (remaining_capacity, [item_indices])
        bin_weights_list = []
        for idx in order:
            w = weights[idx]
            if w > bin_capacity:
                continue
            placed = False
            for b in range(len(bins)):
                if bin_weights_list[b] + w <= bin_capacity:
                    bins[b].append(idx)
                    bin_weights_list[b] += w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_weights_list.append(w)
        return bins, bin_weights_list
    
    # Decode with Best Fit for potentially better results
    def decode_best_fit(priorities):
        order = sorted(range(n), key=lambda i: -priorities[i])
        bins = []
        bin_weights_list = []
        for idx in order:
            w = weights[idx]
            if w > bin_capacity:
                continue
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bins)):
                remaining = bin_capacity - bin_weights_list[b]
                if w <= remaining and remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b
            if best_bin >= 0:
                bins[best_bin].append(idx)
                bin_weights_list[best_bin] += w
            else:
                bins.append([idx])
                bin_weights_list.append(w)
        return bins, bin_weights_list
    
    def decode(priorities):
        # Try both FF and BF, return the better one
        bins_ff, bw_ff = decode_first_fit(priorities)
        bins_bf, bw_bf = decode_best_fit(priorities)
        if len(bins_bf) <= len(bins_ff):
            return bins_bf, bw_bf
        return bins_ff, bw_ff
    
    def fitness(num_bins):
        # We want to minimize bins, so higher fitness = fewer bins
        # Use negative number of bins (or equivalently, return -num_bins)
        return -num_bins
    
    # Create FFD priority vector (items sorted by weight descending)
    ffd_priorities = [0.0] * n
    sorted_by_weight = sorted(range(n), key=lambda i: -weights[i])
    for rank, idx in enumerate(sorted_by_weight):
        ffd_priorities[idx] = float(n - rank)  # highest weight gets highest priority
    
    # Population size
    pop_size = min(50, max(10, n // 2))
    
    # Initialize population
    population = []
    fitnesses = []
    packings = []
    
    # Add FFD solution
    population.append(ffd_priorities[:])
    bins_result, bw_result = decode(ffd_priorities)
    packings.append((bins_result, bw_result))
    fitnesses.append(fitness(len(bins_result)))
    
    # Add random perturbations of FFD
    for i in range(pop_size - 1):
        p = [ffd_priorities[j] + random.gauss(0, n * 0.3) for j in range(n)]
        population.append(p)
        bins_result, bw_result = decode(p)
        packings.append((bins_result, bw_result))
        fitnesses.append(fitness(len(bins_result)))
    
    # Track global best
    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    global_best_fitness = fitnesses[best_idx]
    global_best_packing = packings[best_idx]
    global_best_priorities = population[best_idx][:]
    
    # Lower bound estimation
    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity
    
    # FDO main loop
    max_iterations = 10000
    iteration = 0
    
    # Weight factor for FDO - controls exploration vs exploitation
    wf_max = 1.0
    wf_min = 0.1
    
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break
        
        # Check if we've reached optimal
        if -global_best_fitness <= lower_bound:
            break
        
        iteration += 1
        
        # Find current best in population
        best_pop_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        best_pop_priorities = population[best_pop_idx]
        best_pop_fitness = fitnesses[best_pop_idx]
        
        # Adaptive weight factor (decreases over iterations for convergence)
        wf = wf_max - (wf_max - wf_min) * (iteration / max_iterations)
        
        for i in range(pop_size):
            if time.time() - start_time >= time_limit * 0.95:
                break
            
            if i == best_pop_idx:
                # Best agent: apply random walk
                new_p = [population[i][d] + random.gauss(0, n * 0.1) for d in range(n)]
            else:
                # FDO update rule
                # pace = wf * (fitness_i / fitness_best) if fitness_best != 0
                # Since fitnesses are negative, we need to handle this carefully
                # Use absolute values for ratio
                fi = abs(fitnesses[i]) if fitnesses[i] != 0 else 1
                fb = abs(best_pop_fitness) if best_pop_fitness != 0 else 1
                
                # In FDO: pace = wf * |fi / fb|
                # When fi is close to fb (good fitness), pace is large (closer to wf)
                # When fi is much worse, pace ratio changes
                # Actually in original FDO: if fitness_i > median, move toward best
                # if fitness_i <= median, random exploration
                
                median_fitness = sorted(fitnesses)[pop_size // 2]
                
                if fitnesses[i] >= median_fitness:
                    # Move toward best (exploitation)
                    # pace based on fitness ratio
                    # Original FDO: pace = wf * (f_i / f_best)
                    # For minimization with negative fitness:
                    # Better fitness means closer to 0 (less negative)
                    # ratio = fb / fi gives value in (0,1] for worse agents
                    if fi > 0:
                        pace = wf * (fb / fi)  # ratio < 1 for worse agents
                    else:
                        pace = wf
                    
                    new_p = [
                        population[i][d] + pace * (best_pop_priorities[d] - population[i][d])
                        + random.gauss(0, n * 0.02)
                        for d in range(n)
                    ]
                else:
                    # Exploration: move randomly (foraging behavior)
                    # Pick a random agent and move away or toward it
                    r = random.randint(0, pop_size - 1)
                    while r == i:
                        r = random.randint(0, pop_size - 1)
                    
                    if fi > 0:
                        pace = wf * (fb / fi)
                    else:
                        pace = wf
                    
                    # Random direction
                    sign = 1 if random.random() < 0.5 else -1
                    new_p = [
                        population[i][d] + sign * pace * (population[r][d] - population[i][d])
                        + random.gauss(0, n * 0.05)
                        for d in range(n)
                    ]
            
            # Evaluate new solution
            new_bins, new_bw = decode(new_p)
            new_fit = fitness(len(new_bins))
            
            # Greedy selection: keep better solution
            if new_fit >= fitnesses[i]:
                population[i] = new_p
                fitnesses[i] = new_fit
                packings[i] = (new_bins, new_bw)
                
                # Update global best
                if new_fit > global_best_fitness:
                    global_best_fitness = new_fit
                    global_best_packing = (new_bins, new_bw)
                    global_best_priorities = new_p[:]
        
        # Occasionally reinitialize worst solutions to maintain diversity
        if iteration % 20 == 0:
            worst_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])[:max(1, pop_size // 5)]
            for wi in worst_indices:
                if wi == best_pop_idx:
                    continue
                # Reinitialize around best with larger perturbation
                new_p = [global_best_priorities[d] + random.gauss(0, n * 0.3) for d in range(n)]
                new_bins, new_bw = decode(new_p)
                new_fit = fitness(len(new_bins))
                population[wi] = new_p
                fitnesses[wi] = new_fit
                packings[wi] = (new_bins, new_bw)
                if new_fit > global_best_fitness:
                    global_best_fitness = new_fit
                    global_best_packing = (new_bins, new_bw)
                    global_best_priorities = new_p[:]
    
    best_bins, best_bw = global_best_packing
    
    return {
        "packing": best_bins,
        "bin_weights": best_bw
    }
