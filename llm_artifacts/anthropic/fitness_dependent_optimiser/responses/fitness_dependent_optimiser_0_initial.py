import random
import time
import math

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Helper: decode a permutation into a packing using First Fit
    def decode_first_fit(perm):
        bin_remaining = []
        bin_contents = []
        for idx in perm:
            w = weights[idx]
            placed = False
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w:
                    bin_remaining[b] -= w
                    bin_contents[b].append(idx)
                    placed = True
                    break
            if not placed:
                bin_remaining.append(bin_capacity - w)
                bin_contents.append([idx])
        return bin_contents
    
    # Helper: decode using Best Fit
    def decode_best_fit(perm):
        bin_remaining = []
        bin_contents = []
        for idx in perm:
            w = weights[idx]
            best_b = -1
            best_rem = bin_capacity + 1
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w and bin_remaining[b] - w < best_rem:
                    best_rem = bin_remaining[b] - w
                    best_b = b
            if best_b >= 0:
                bin_remaining[best_b] -= w
                bin_contents[best_b].append(idx)
            else:
                bin_remaining.append(bin_capacity - w)
                bin_contents.append([idx])
        return bin_contents
    
    def evaluate(perm):
        # Use best fit for better quality
        bins = decode_best_fit(perm)
        num_bins = len(bins)
        # Secondary objective: minimize wasted space variance (pack tightly)
        # Primary: minimize number of bins
        return num_bins
    
    def make_packing_result(perm):
        bins = decode_best_fit(perm)
        bin_weights_list = []
        for b in bins:
            bin_weights_list.append(sum(weights[i] for i in b))
        return {"packing": bins, "bin_weights": bin_weights_list}
    
    # Generate initial permutation: sorted by weight descending (FFD-like)
    def ffd_perm():
        return sorted(range(n), key=lambda i: -weights[i])
    
    def random_perm():
        p = list(range(n))
        random.shuffle(p)
        return p
    
    # For small instances, just return FFD
    if n <= 1:
        perm = ffd_perm()
        return make_packing_result(perm)
    
    # Population size
    pop_size = min(50, max(10, n // 2))
    
    # Initialize population
    population = []
    fitnesses = []
    
    # Add FFD permutation
    ffd_p = ffd_perm()
    population.append(ffd_p)
    fitnesses.append(evaluate(ffd_p))
    
    # Add some variations of FFD (with small perturbations)
    for _ in range(min(5, pop_size - 1)):
        p = ffd_p[:]
        # Small random swaps
        num_swaps = max(1, n // 20)
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            p[i], p[j] = p[j], p[i]
        population.append(p)
        fitnesses.append(evaluate(p))
    
    # Fill rest with random permutations
    while len(population) < pop_size:
        p = random_perm()
        population.append(p)
        fitnesses.append(evaluate(p))
    
    # Track best
    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    best_perm = population[best_idx][:]
    
    # FDO main loop
    max_iterations = 10000
    
    # Precompute for time checking
    check_interval = max(1, pop_size)  # check time every generation
    
    iteration = 0
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break
        
        # Find current best (guide bee)
        guide_idx = fitnesses.index(min(fitnesses))
        guide = population[guide_idx]
        guide_fitness = fitnesses[guide_idx]
        
        # Find worst fitness for normalization
        max_fitness = max(fitnesses)
        fitness_range = max_fitness - guide_fitness
        if fitness_range == 0:
            fitness_range = 1.0
        
        new_population = []
        new_fitnesses = []
        
        for i in range(pop_size):
            if time.time() - start_time >= time_limit * 0.95:
                # Keep remaining as is
                new_population.append(population[i])
                new_fitnesses.append(fitnesses[i])
                continue
            
            # FDO: compute fitness weight
            # w_i = |f_i - f_best| / fitness_range
            # w_i close to 0 means good fitness (exploit near guide)
            # w_i close to 1 means bad fitness (explore more)
            w_i = abs(fitnesses[i] - guide_fitness) / fitness_range
            
            # Generate new solution
            # FDO movement: in permutation space
            # With probability (1 - w_i), copy segments from guide (exploitation)
            # With probability w_i, do random perturbation (exploration)
            
            new_perm = population[i][:]
            
            if random.random() < 0.5:
                # Strategy 1: Partial order crossover with guide
                # Copy a segment from guide, fill rest from current
                # Segment length depends on w_i: small w_i -> larger segment from guide
                seg_len = max(1, int((1 - w_i) * n * 0.5))
                start_pos = random.randint(0, n - 1)
                end_pos = min(start_pos + seg_len, n)
                
                # Order crossover (OX)
                child = [-1] * n
                guide_segment = guide[start_pos:end_pos]
                guide_set = set(guide_segment)
                
                for k in range(start_pos, end_pos):
                    child[k] = guide[k]
                
                # Fill remaining positions with items from current perm in order
                remaining = [x for x in new_perm if x not in guide_set]
                pos = 0
                for k in range(n):
                    if child[k] == -1:
                        child[k] = remaining[pos]
                        pos += 1
                
                new_perm = child
            else:
                # Strategy 2: Random perturbation scaled by w_i
                # More swaps for higher w_i (worse fitness -> more exploration)
                num_swaps = max(1, int(w_i * n * 0.3) + 1)
                for _ in range(num_swaps):
                    a = random.randint(0, n - 1)
                    b = random.randint(0, n - 1)
                    new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
            
            # Occasionally apply a guided move: move heavy items earlier
            if random.random() < 0.1:
                # Find a heavy item not near the front and move it forward
                heavy_items = sorted(range(n), key=lambda x: -weights[new_perm[x]])
                for hi in heavy_items[:3]:
                    pos_in_perm = hi
                    if pos_in_perm > n // 4:
                        new_pos = random.randint(0, n // 4)
                        item = new_perm.pop(pos_in_perm)
                        new_perm.insert(new_pos, item)
                        break
            
            new_fit = evaluate(new_perm)
            
            # Greedy selection: keep better solution
            if new_fit <= fitnesses[i]:
                new_population.append(new_perm)
                new_fitnesses.append(new_fit)
            else:
                new_population.append(population[i])
                new_fitnesses.append(fitnesses[i])
            
            # Update global best
            if new_fit < best_fitness:
                best_fitness = new_fit
                best_perm = new_perm[:]
            elif new_fit == best_fitness:
                # Tie-break by tightness of packing
                best_perm = new_perm[:]
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Ensure best is always in population (elitism)
        worst_idx = fitnesses.index(max(fitnesses))
        if best_fitness < fitnesses[worst_idx]:
            population[worst_idx] = best_perm[:]
            fitnesses[worst_idx] = best_fitness
        
        # Diversity injection: replace worst solutions occasionally
        if iteration % 50 == 49:
            sorted_indices = sorted(range(pop_size), key=lambda x: fitnesses[x])
            # Replace bottom 20% with new random solutions
            num_replace = max(1, pop_size // 5)
            for k in range(num_replace):
                idx_to_replace = sorted_indices[-(k+1)]
                if idx_to_replace == fitnesses.index(min(fitnesses)):
                    continue
                p = random_perm()
                population[idx_to_replace] = p
                fitnesses[idx_to_replace] = evaluate(p)
        
        iteration += 1
    
    return make_packing_result(best_perm)
