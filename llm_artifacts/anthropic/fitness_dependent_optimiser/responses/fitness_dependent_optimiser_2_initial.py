import random
import time
import math

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Helper: decode a permutation into a packing using First Fit Decreasing
    def decode_packing(perm):
        bins = []
        bin_loads = []
        item_to_bin = [0] * n
        
        for idx in perm:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if bin_loads[b] + w <= bin_capacity:
                    bins[b].append(idx)
                    bin_loads[b] += w
                    item_to_bin[idx] = b
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_loads.append(w)
                item_to_bin[idx] = len(bins) - 1
        
        return bins, bin_loads
    
    # Fitness: number of bins (lower is better)
    # We also use a secondary fitness based on how well bins are filled
    def evaluate(perm):
        bins, bin_loads = decode_packing(perm)
        num_bins = len(bins)
        # Falkenauer fitness: sum of (load/C)^2 / num_bins - higher is better for filled bins
        if num_bins == 0:
            return num_bins, 0.0, bins, bin_loads
        fill_score = sum((load / bin_capacity) ** 2 for load in bin_loads) / num_bins
        return num_bins, fill_score, bins, bin_loads
    
    # Combined fitness: primary is num_bins (minimize), secondary is fill_score (maximize)
    # For FDO, we want a single fitness value to minimize
    # fitness = num_bins - fill_score (since fill_score in [0,1])
    def fitness(perm):
        num_bins, fill_score, bins, bin_loads = evaluate(perm)
        return num_bins - fill_score, bins, bin_loads
    
    # Generate initial permutation using sorted-by-weight-descending (FFD ordering)
    def generate_ffd_perm():
        perm = sorted(range(n), key=lambda i: -weights[i])
        return perm
    
    # Generate a random permutation biased toward larger items first
    def generate_random_perm():
        perm = list(range(n))
        random.shuffle(perm)
        return perm
    
    # Generate semi-random permutation (partially sorted)
    def generate_semi_random_perm():
        perm = list(range(n))
        # Sort by weight descending, then apply some random swaps
        perm.sort(key=lambda i: -weights[i])
        num_swaps = random.randint(1, max(1, n // 4))
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    # Population size
    pop_size = min(50, max(10, n))
    
    # Initialize population
    population = []
    fitnesses = []
    all_bins = []
    all_bin_loads = []
    
    # First agent is FFD
    perm = generate_ffd_perm()
    f, b, bl = fitness(perm)
    population.append(perm)
    fitnesses.append(f)
    all_bins.append(b)
    all_bin_loads.append(bl)
    
    # Rest are semi-random or random
    for i in range(1, pop_size):
        if time.time() - start_time > time_limit * 0.9:
            break
        if i < pop_size // 3:
            perm = generate_semi_random_perm()
        else:
            perm = generate_random_perm()
        f, b, bl = fitness(perm)
        population.append(perm)
        fitnesses.append(f)
        all_bins.append(b)
        all_bin_loads.append(bl)
    
    pop_size = len(population)
    
    # Track best
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_fitness = fitnesses[best_idx]
    best_perm = population[best_idx][:]
    best_bins = all_bins[best_idx]
    best_bin_loads = all_bin_loads[best_idx]
    
    # FDO main loop
    max_iterations = 1000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit * 0.95:
            break
        
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
            
            # FDO weight calculation
            # Normalized fitness: 0 = best, 1 = worst
            if fitness_range > 0:
                normalized_fi = (fi - curr_best_fitness) / fitness_range
            else:
                normalized_fi = 0.0
            
            # In FDO: weight factor determines exploration vs exploitation
            # Higher weight (worse fitness) -> more exploration (more random changes)
            # Lower weight (better fitness) -> more exploitation (follow best)
            
            # The FDO formula: weight = fi / f_best
            # If weight > random.random() * pace_factor: scout (explore)
            # Else: forager (exploit toward best)
            
            if curr_best_fitness > 0:
                weight = fi / curr_best_fitness
            else:
                weight = 1.0 + normalized_fi
            
            new_perm = population[i][:]
            
            # FDO movement
            # The pace of movement depends on the weight
            # Scout bees (high weight ratio): random exploration
            # Forager bees (low weight ratio): move toward best
            
            r = random.random()
            
            if weight > 1.0 + r:  # Scout behavior - explore
                # Amount of randomization proportional to weight
                num_changes = max(1, int(n * normalized_fi * 0.5))
                for _ in range(num_changes):
                    a = random.randint(0, n - 1)
                    b_idx = random.randint(0, n - 1)
                    new_perm[a], new_perm[b_idx] = new_perm[b_idx], new_perm[a]
            else:  # Forager behavior - exploit toward best
                # Copy segments from best permutation
                # The amount copied is inversely proportional to the weight
                # Better agents copy less (fine-tune), worse agents copy more
                
                # Position-based crossover inspired movement
                # Select a fraction of positions from the best and keep rest from current
                copy_rate = max(0.05, min(0.6, normalized_fi * 0.8))
                num_copy = max(1, int(n * copy_rate))
                
                # Select positions to copy from best
                positions = random.sample(range(n), num_copy)
                
                # Get items at those positions in the best permutation
                items_from_best = [curr_best_perm[p] for p in positions]
                items_set = set(items_from_best)
                
                # Build new permutation: place items_from_best at their positions
                # Fill remaining positions with items from current perm in order
                remaining = [x for x in new_perm if x not in items_set]
                
                result = [None] * n
                for p in positions:
                    result[p] = curr_best_perm[p]
                
                ri = 0
                for j in range(n):
                    if result[j] is None:
                        result[j] = remaining[ri]
                        ri += 1
                
                new_perm = result
                
                # Small perturbation based on fitness
                num_swaps = max(1, int(n * (1 - normalized_fi) * 0.05))
                for _ in range(num_swaps):
                    a = random.randint(0, n - 1)
                    b_idx = random.randint(0, n - 1)
                    new_perm[a], new_perm[b_idx] = new_perm[b_idx], new_perm[a]
            
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
            else:
                # With some probability, accept worse solutions (for diversity)
                # This probability decreases over time
                accept_prob = 0.05 * (1 - iteration / max_iterations)
                if random.random() < accept_prob:
                    population[i] = new_perm
                    fitnesses[i] = new_f
                    all_bins[i] = new_b
                    all_bin_loads[i] = new_bl
        
        # Reinitialize worst agents periodically for diversity (part of FDO)
        if iteration % 20 == 0 and iteration > 0:
            # Sort by fitness, reinitialize worst 20%
            sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
            num_reinit = max(1, pop_size // 5)
            for idx in sorted_indices[-num_reinit:]:
                if time.time() - start_time > time_limit * 0.95:
                    break
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
    
    # Build result
    packing = best_bins
    bin_weights = best_bin_loads
    
    return {"packing": packing, "bin_weights": bin_weights}