import random
import time
import math

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    if n == 1:
        return {"packing": [[0]], "bin_weights": [weights[0]]}
    
    # Compute lower bound
    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity  # ceil division
    
    # Pre-compute weights as a tuple for faster access
    w = weights  # direct reference
    
    # Helper: decode a permutation into a packing using Best Fit
    def decode_bfd(perm):
        bins = []
        bin_loads = []
        
        for idx in perm:
            wi = w[idx]
            best_b = -1
            best_remaining = bin_capacity + 1  # initialize to larger than possible
            
            for b in range(len(bin_loads)):
                remaining = bin_capacity - bin_loads[b]
                if wi <= remaining and remaining < best_remaining:
                    best_remaining = remaining
                    best_b = b
                    # If perfect fit, no need to keep looking
                    if remaining == wi:
                        break
            
            if best_b >= 0:
                bins[best_b].append(idx)
                bin_loads[best_b] += wi
            else:
                bins.append([idx])
                bin_loads.append(wi)
        
        return bins, bin_loads
    
    # Fitness: lower is better
    # fitness = num_bins - sum((load/C)^3) / num_bins
    def fitness(perm):
        bins, bin_loads = decode_bfd(perm)
        num_bins = len(bins)
        if num_bins == 0:
            return 0.0, bins, bin_loads
        c3 = bin_capacity * bin_capacity * bin_capacity
        fill_score = sum(load * load * load for load in bin_loads) / (c3 * num_bins)
        return num_bins - fill_score, bins, bin_loads
    
    # Generate FFD permutation
    def generate_ffd_perm():
        return sorted(range(n), key=lambda i: -w[i])
    
    # Generate random permutation
    def generate_random_perm():
        perm = list(range(n))
        random.shuffle(perm)
        return perm
    
    # Generate semi-random permutation (sorted + some swaps)
    def generate_semi_random_perm():
        perm = sorted(range(n), key=lambda i: -w[i])
        num_swaps = random.randint(1, max(1, n // 4))
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    # Generate perturbation of best permutation
    def generate_perturbed_best(base_perm, num_swaps):
        perm = base_perm[:]
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    # Generate FFD with segment reversals
    def generate_ffd_perturbed():
        perm = sorted(range(n), key=lambda i: -w[i])
        num_reversals = random.randint(2, 5)
        for _ in range(num_reversals):
            if n < 3:
                break
            seg_len = random.randint(2, max(2, n // 5))
            start = random.randint(0, n - seg_len)
            perm[start:start+seg_len] = reversed(perm[start:start+seg_len])
        return perm
    
    # Population size
    pop_size = min(30, max(10, n // 2))
    
    # Initialize population
    population = []
    fitnesses = []
    all_bins = []
    all_bin_loads = []
    
    def add_agent(perm):
        f, b, bl = fitness(perm)
        population.append(perm)
        fitnesses.append(f)
        all_bins.append(b)
        all_bin_loads.append(bl)
        return f, b, bl
    
    # Agent 0: FFD
    ffd_perm = generate_ffd_perm()
    add_agent(ffd_perm)
    
    # Agent 1: same FFD (BFD decode handles it)
    add_agent(ffd_perm[:])
    
    # Agents 2-5: FFD with segment reversals
    for i in range(2, min(6, pop_size)):
        if time.time() - start_time > time_limit * 0.8:
            break
        add_agent(generate_ffd_perturbed())
    
    # Agents 6 to pop_size//3: semi-random
    third = max(6, pop_size // 3)
    for i in range(len(population), min(third, pop_size)):
        if time.time() - start_time > time_limit * 0.8:
            break
        add_agent(generate_semi_random_perm())
    
    # Remaining: random
    for i in range(len(population), pop_size):
        if time.time() - start_time > time_limit * 0.8:
            break
        add_agent(generate_random_perm())
    
    pop_size = len(population)
    
    # Track best
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_fitness = fitnesses[best_idx]
    best_perm = population[best_idx][:]
    best_bins = all_bins[best_idx]
    best_bin_loads = all_bin_loads[best_idx]
    
    # Check if already optimal
    if len(best_bins) <= lower_bound:
        return {"packing": best_bins, "bin_weights": best_bin_loads}
    
    # FDO main loop - time based
    iteration = 0
    last_improvement_iter = 0
    stagnation_threshold = 50
    
    while time.time() - start_time < time_limit * 0.95:
        iteration += 1
        
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_limit)  # 0 to 1
        
        # Find current best in population
        curr_best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        curr_best_perm = population[curr_best_idx]
        curr_best_fitness = fitnesses[curr_best_idx]
        
        # Worst fitness for normalization
        worst_fitness = max(fitnesses)
        fitness_range = worst_fitness - curr_best_fitness if worst_fitness != curr_best_fitness else 1.0
        
        for i in range(pop_size):
            if time.time() - start_time > time_limit * 0.95:
                break
            
            if i == curr_best_idx:
                continue
            
            fi = fitnesses[i]
            
            # Normalized fitness: 0 = best, 1 = worst
            if fitness_range > 0:
                normalized_fi = (fi - curr_best_fitness) / fitness_range
            else:
                normalized_fi = 0.0
            
            # FDO weight calculation
            if curr_best_fitness > 0:
                weight = fi / curr_best_fitness
            else:
                weight = 1.0 + normalized_fi
            
            new_perm = population[i][:]
            
            # Adaptive threshold based on progress
            r = random.random()
            threshold = 1.0 + (0.5 + progress) * r
            
            if weight > threshold:  # Scout behavior - explore
                # Structured perturbations
                rand_val = random.random()
                
                if rand_val < 0.5 and n >= 3:
                    # Segment reversal
                    seg_len = max(2, int(n * normalized_fi * 0.3))
                    seg_len = min(seg_len, n)
                    start_pos = random.randint(0, n - seg_len)
                    new_perm[start_pos:start_pos+seg_len] = reversed(new_perm[start_pos:start_pos+seg_len])
                elif rand_val < 0.8 and n >= 3:
                    # Block insertion
                    block_len = max(1, int(n * normalized_fi * 0.2))
                    block_len = min(block_len, n - 1)
                    start_pos = random.randint(0, n - block_len)
                    block = new_perm[start_pos:start_pos+block_len]
                    del new_perm[start_pos:start_pos+block_len]
                    insert_pos = random.randint(0, len(new_perm))
                    for k, item in enumerate(block):
                        new_perm.insert(insert_pos + k, item)
                else:
                    # Random swaps
                    num_changes = max(1, int(n * normalized_fi * 0.5))
                    for _ in range(num_changes):
                        a = random.randint(0, n - 1)
                        b_idx = random.randint(0, n - 1)
                        new_perm[a], new_perm[b_idx] = new_perm[b_idx], new_perm[a]
            else:  # Forager behavior - exploit toward best
                # Order Crossover (OX) with best permutation
                copy_rate = max(0.1, min(0.5, normalized_fi * 0.6))
                seg_len = max(1, int(n * copy_rate))
                
                # Select segment from best
                start_pos = random.randint(0, n - seg_len)
                end_pos = start_pos + seg_len
                
                # Items in the segment from best
                segment = curr_best_perm[start_pos:end_pos]
                segment_set = set(segment)
                
                # Build OX child
                result = [None] * n
                # Place segment from best
                for k in range(start_pos, end_pos):
                    result[k] = curr_best_perm[k]
                
                # Fill remaining positions with items from current in relative order
                remaining = [x for x in new_perm if x not in segment_set]
                ri = 0
                for j in range(n):
                    if result[j] is None:
                        result[j] = remaining[ri]
                        ri += 1
                
                new_perm = result
                
                # Micro-perturbation: 1-3 adjacent swaps
                num_micro = random.randint(1, 3)
                for _ in range(num_micro):
                    pos = random.randint(0, n - 2)
                    new_perm[pos], new_perm[pos+1] = new_perm[pos+1], new_perm[pos]
            
            # Evaluate new permutation
            new_f, new_b, new_bl = fitness(new_perm)
            
            # Accept if better (greedy selection)
            if new_f < fitnesses[i]:
                population[i] = new_perm
                fitnesses[i] = new_f
                all_bins[i] = new_b
                all_bin_loads[i] = new_bl
                
                # Update global best
                if new_f < best_fitness:
                    best_fitness = new_f
                    best_perm = new_perm[:]
                    best_bins = new_b
                    best_bin_loads = new_bl
                    last_improvement_iter = iteration
                    
                    # Check if optimal
                    if len(best_bins) <= lower_bound:
                        return {"packing": best_bins, "bin_weights": best_bin_loads}
            else:
                # Small probability of accepting worse solutions, decreasing over time
                accept_prob = 0.05 * (1.0 - progress)
                if random.random() < accept_prob:
                    population[i] = new_perm
                    fitnesses[i] = new_f
                    all_bins[i] = new_b
                    all_bin_loads[i] = new_bl
        
        # Stagnation-based reinitialization
        if iteration - last_improvement_iter > stagnation_threshold:
            last_improvement_iter = iteration  # reset counter
            
            # Sort by fitness, reinitialize worst 30%
            sorted_indices = sorted(range(pop_size), key=lambda idx: fitnesses[idx])
            num_reinit = max(1, int(pop_size * 0.3))
            reinit_indices = sorted_indices[-num_reinit:]
            
            for k, idx in enumerate(reinit_indices):
                if time.time() - start_time > time_limit * 0.95:
                    break
                
                if k < num_reinit // 2:
                    # Perturbation of global best
                    num_swaps = random.randint(5, min(15, n))
                    perm = generate_perturbed_best(best_perm, num_swaps)
                else:
                    # Semi-random
                    perm = generate_semi_random_perm()
                
                f, b, bl = fitness(perm)
                population[idx] = perm
                fitnesses[idx] = f
                all_bins[idx] = b
                all_bin_loads[idx] = bl
                
                if f < best_fitness:
                    best_fitness = f
                    best_perm = perm[:]
                    best_bins = b
                    best_bin_loads = bl
                    
                    if len(best_bins) <= lower_bound:
                        return {"packing": best_bins, "bin_weights": best_bin_loads}
            
            # Elitism: ensure best is in population
            # Replace one reinitialized agent with global best
            if reinit_indices:
                elitism_idx = reinit_indices[0]
                perm = best_perm[:]
                f, b, bl = fitness(perm)
                population[elitism_idx] = perm
                fitnesses[elitism_idx] = f
                all_bins[elitism_idx] = b
                all_bin_loads[elitism_idx] = bl
    
    return {"packing": best_bins, "bin_weights": best_bin_loads}