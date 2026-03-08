import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Pre-compute weights as tuple for faster indexing
    w_tuple = tuple(weights)
    C = bin_capacity
    
    # Local references for speed
    _random = random.random
    _randint = random.randint
    _shuffle = random.shuffle
    _time = time.time
    
    # Helper: decode a permutation into a packing using First Fit
    def decode_first_fit(perm):
        bin_remaining = []
        bin_contents = []
        bin_weights_list = []
        br_append = bin_remaining.append
        bc_append = bin_contents.append
        bw_append = bin_weights_list.append
        wt = w_tuple
        cap = C
        for idx in perm:
            w = wt[idx]
            placed = False
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w:
                    bin_remaining[b] -= w
                    bin_contents[b].append(idx)
                    bin_weights_list[b] += w
                    placed = True
                    break
            if not placed:
                br_append(cap - w)
                bc_append([idx])
                bw_append(w)
        return bin_contents, bin_weights_list
    
    # Helper: decode using Best Fit with bisect optimization
    def decode_best_fit(perm):
        # Maintain sorted list of (remaining_capacity, bin_index)
        # Use bisect to find best fit
        bin_remaining = []
        bin_contents = []
        bin_weights_list = []
        # sorted_remaining: list of remaining capacities sorted ascending
        # We also need to map back. Use a simple list approach.
        # For efficiency, we'll use a direct approach with bisect on a sorted list.
        sorted_rem = []  # sorted list of (remaining, bin_index)
        wt = w_tuple
        cap = C
        
        for idx in perm:
            w = wt[idx]
            # Find best fit: smallest remaining >= w
            pos = bisect.bisect_left(sorted_rem, (w,))
            if pos < len(sorted_rem):
                # Found a bin with remaining >= w
                rem, b = sorted_rem[pos]
                # Remove this entry
                sorted_rem.pop(pos)
                new_rem = rem - w
                bin_remaining[b] = new_rem
                bin_contents[b].append(idx)
                bin_weights_list[b] += w
                # Re-insert with new remaining
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, b))
            else:
                # Open new bin
                b = len(bin_remaining)
                new_rem = cap - w
                bin_remaining.append(new_rem)
                bin_contents.append([idx])
                bin_weights_list.append(w)
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, b))
        return bin_contents, bin_weights_list
    
    # Continuous Falkenauer-style fitness: lower is better
    # fitness = num_bins - sum((bin_weight_i / C)^2) / num_bins
    def evaluate(perm):
        _, bw = decode_best_fit(perm)
        num_bins = len(bw)
        if num_bins == 0:
            return 0.0
        cap = C
        sum_sq = 0.0
        for bwi in bw:
            ratio = bwi / cap
            sum_sq += ratio * ratio
        return num_bins - sum_sq / num_bins
    
    def evaluate_with_bins(perm):
        bins, bw = decode_best_fit(perm)
        num_bins = len(bw)
        if num_bins == 0:
            return 0.0, 0, bins, bw
        cap = C
        sum_sq = 0.0
        for bwi in bw:
            ratio = bwi / cap
            sum_sq += ratio * ratio
        return num_bins - sum_sq / num_bins, num_bins, bins, bw
    
    def make_packing_result(perm):
        # Try both FF and BF, pick better
        bins_bf, bw_bf = decode_best_fit(perm)
        bins_ff, bw_ff = decode_first_fit(perm)
        
        nb_bf = len(bins_bf)
        nb_ff = len(bins_ff)
        
        if nb_ff < nb_bf:
            return {"packing": bins_ff, "bin_weights": bw_ff}
        elif nb_bf < nb_ff:
            return {"packing": bins_bf, "bin_weights": bw_bf}
        else:
            # Same bin count, pick by Falkenauer fitness
            cap = C
            sum_sq_bf = sum((bwi / cap) ** 2 for bwi in bw_bf)
            sum_sq_ff = sum((bwi / cap) ** 2 for bwi in bw_ff)
            fit_bf = nb_bf - sum_sq_bf / nb_bf if nb_bf > 0 else 0
            fit_ff = nb_ff - sum_sq_ff / nb_ff if nb_ff > 0 else 0
            if fit_ff <= fit_bf:
                return {"packing": bins_ff, "bin_weights": bw_ff}
            else:
                return {"packing": bins_bf, "bin_weights": bw_bf}
    
    # Generate initial permutation: sorted by weight descending (FFD-like)
    def ffd_perm():
        return sorted(range(n), key=lambda i: -w_tuple[i])
    
    def random_perm():
        p = list(range(n))
        _shuffle(p)
        return p
    
    def weight_biased_random_perm():
        # Random permutation biased toward heavy items first
        items = list(range(n))
        result = []
        total_remaining = sum(w_tuple[i] for i in items)
        for _ in range(n):
            if not items:
                break
            # Weighted selection proportional to weight
            r = _random() * total_remaining
            cum = 0.0
            chosen_idx = len(items) - 1
            for k, it in enumerate(items):
                cum += w_tuple[it]
                if cum >= r:
                    chosen_idx = k
                    break
            chosen = items[chosen_idx]
            result.append(chosen)
            total_remaining -= w_tuple[chosen]
            items.pop(chosen_idx)
            if total_remaining <= 0:
                total_remaining = 1.0
        result.extend(items)  # any remaining
        return result
    
    def jittered_weight_sort_perm():
        # Sort by weight with random jitter
        jitter = C * 0.05
        return sorted(range(n), key=lambda i: -(w_tuple[i] + _random() * jitter))
    
    # For small instances, just return FFD
    if n <= 1:
        perm = ffd_perm()
        return make_packing_result(perm)
    
    # Population size
    pop_size = min(80, max(20, n))
    
    # Initialize population
    population = []
    fitnesses = []
    
    # 1. Pure FFD permutation
    ffd_p = ffd_perm()
    population.append(ffd_p)
    fitnesses.append(evaluate(ffd_p))
    
    # 2. Near-FFD permutations (5-10): swap pairs with small weight difference
    num_near_ffd = min(8, pop_size - 1)
    for _ in range(num_near_ffd):
        p = ffd_p[:]
        num_swaps = max(1, n // 20)
        for _ in range(num_swaps):
            i = _randint(0, n - 1)
            j = _randint(0, max(0, i - 3), min(n - 1, i + 3)) if n > 1 else 0
            p[i], p[j] = p[j], p[i]
        population.append(p)
        fitnesses.append(evaluate(p))
    
    # 3. Jittered weight sort permutations
    num_jittered = min(5, pop_size - len(population))
    for _ in range(num_jittered):
        p = jittered_weight_sort_perm()
        population.append(p)
        fitnesses.append(evaluate(p))
    
    # 4. Weight-biased random permutations
    num_biased = min(5, pop_size - len(population))
    if n <= 500:  # Only for smaller instances due to O(n^2) cost
        for _ in range(num_biased):
            p = weight_biased_random_perm()
            population.append(p)
            fitnesses.append(evaluate(p))
    
    # 5. Fill rest with random permutations
    while len(population) < pop_size:
        p = random_perm()
        population.append(p)
        fitnesses.append(evaluate(p))
    
    # Track best
    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    best_perm = population[best_idx][:]
    # Also track best bin count
    _, best_bin_count, _, _ = evaluate_with_bins(best_perm)
    
    # FDO main loop
    max_iterations = 100000
    generations_without_improvement = 0
    
    main_cutoff = time_limit * 0.92
    
    iteration = 0
    while iteration < max_iterations:
        elapsed = _time() - start_time
        if elapsed >= main_cutoff:
            break
        
        # Find current best and worst
        min_fit = float('inf')
        max_fit = float('-inf')
        guide_idx = 0
        for i in range(pop_size):
            if fitnesses[i] < min_fit:
                min_fit = fitnesses[i]
                guide_idx = i
            if fitnesses[i] > max_fit:
                max_fit = fitnesses[i]
        
        guide = population[guide_idx]
        guide_fitness = min_fit
        
        fitness_range = max_fit - min_fit
        if fitness_range < 1e-12:
            fitness_range = 1.0
        
        # Adaptive stagnation boost
        stagnation_boost = 0.0
        if generations_without_improvement > 100:
            stagnation_boost = 0.2
        
        new_population = []
        new_fitnesses = []
        
        # Elitism: preserve top 2
        sorted_by_fitness = sorted(range(pop_size), key=lambda x: fitnesses[x])
        elite_indices = set(sorted_by_fitness[:2])
        
        for i in range(pop_size):
            # Elitism: keep top 2 unchanged
            if i in elite_indices:
                new_population.append(population[i])
                new_fitnesses.append(fitnesses[i])
                continue
            
            # FDO: compute fitness weight
            w_i = (fitnesses[i] - guide_fitness) / fitness_range
            w_i = min(1.0, w_i + stagnation_boost)
            
            new_perm = population[i][:]
            
            if w_i < 0.3:
                # EXPLOITATION: small perturbation (insert moves)
                num_moves = _randint(1, 3)
                for _ in range(num_moves):
                    if _random() < 0.5:
                        # Insert move: pick random item, move 1-3 positions earlier
                        pos = _randint(1, n - 1)
                        new_pos = max(0, pos - _randint(1, 3))
                        item = new_perm.pop(pos)
                        new_perm.insert(new_pos, item)
                    else:
                        # Adjacent swap
                        pos = _randint(0, n - 2)
                        new_perm[pos], new_perm[pos + 1] = new_perm[pos + 1], new_perm[pos]
            
            elif w_i < 0.7:
                # BALANCED: Order crossover with guide
                seg_len = max(1, int((1.0 - w_i) * n * 0.6))
                start_pos = _randint(0, n - 1)
                end_pos = start_pos + seg_len
                
                if end_pos <= n:
                    # Non-wrapping OX
                    child = [-1] * n
                    guide_segment_set = set()
                    for k in range(start_pos, end_pos):
                        child[k] = guide[k]
                        guide_segment_set.add(guide[k])
                    remaining = [x for x in new_perm if x not in guide_segment_set]
                    pos = 0
                    for k in range(n):
                        if child[k] == -1:
                            child[k] = remaining[pos]
                            pos += 1
                    new_perm = child
                else:
                    # Wrapping OX
                    child = [-1] * n
                    guide_segment_set = set()
                    for k in range(start_pos, n):
                        child[k] = guide[k]
                        guide_segment_set.add(guide[k])
                    wrap_end = end_pos - n
                    for k in range(0, wrap_end):
                        child[k] = guide[k]
                        guide_segment_set.add(guide[k])
                    remaining = [x for x in new_perm if x not in guide_segment_set]
                    pos = 0
                    for k in range(n):
                        if child[k] == -1:
                            child[k] = remaining[pos]
                            pos += 1
                    new_perm = child
            
            else:
                # EXPLORATION: large movement
                if _random() < 0.5:
                    # New jittered weight-sorted permutation
                    new_perm = jittered_weight_sort_perm()
                else:
                    # Large segment OX from guide (70-90%)
                    seg_frac = 0.7 + _random() * 0.2
                    seg_len = max(1, int(seg_frac * n))
                    start_pos = _randint(0, n - 1)
                    end_pos = start_pos + seg_len
                    
                    if end_pos <= n:
                        child = [-1] * n
                        guide_segment_set = set()
                        for k in range(start_pos, end_pos):
                            child[k] = guide[k]
                            guide_segment_set.add(guide[k])
                        remaining = [x for x in new_perm if x not in guide_segment_set]
                        _shuffle(remaining)
                        pos = 0
                        for k in range(n):
                            if child[k] == -1:
                                child[k] = remaining[pos]
                                pos += 1
                        new_perm = child
                    else:
                        child = [-1] * n
                        guide_segment_set = set()
                        for k in range(start_pos, n):
                            child[k] = guide[k]
                            guide_segment_set.add(guide[k])
                        wrap_end = end_pos - n
                        for k in range(0, wrap_end):
                            child[k] = guide[k]
                            guide_segment_set.add(guide[k])
                        remaining = [x for x in new_perm if x not in guide_segment_set]
                        _shuffle(remaining)
                        pos = 0
                        for k in range(n):
                            if child[k] == -1:
                                child[k] = remaining[pos]
                                pos += 1
                        new_perm = child
            
            new_fit = evaluate(new_perm)
            
            # Greedy selection
            if new_fit <= fitnesses[i]:
                new_population.append(new_perm)
                new_fitnesses.append(new_fit)
            else:
                new_population.append(population[i])
                new_fitnesses.append(fitnesses[i])
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Update global best
        gen_best_fit = min(fitnesses)
        improved = False
        if gen_best_fit < best_fitness:
            best_fitness = gen_best_fit
            best_idx = fitnesses.index(best_fitness)
            best_perm = population[best_idx][:]
            _, best_bin_count, _, _ = evaluate_with_bins(best_perm)
            improved = True
        
        if improved:
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # Ensure best is always in population (elitism)
        worst_idx = 0
        worst_fit = fitnesses[0]
        for i in range(1, pop_size):
            if fitnesses[i] > worst_fit:
                worst_fit = fitnesses[i]
                worst_idx = i
        if best_fitness < worst_fit:
            population[worst_idx] = best_perm[:]
            fitnesses[worst_idx] = best_fitness
        
        # Diversity injection on stagnation
        if generations_without_improvement > 0 and generations_without_improvement % 300 == 0:
            sorted_indices = sorted(range(pop_size), key=lambda x: fitnesses[x])
            num_replace = max(1, int(pop_size * 0.3))
            for k in range(num_replace):
                idx_to_replace = sorted_indices[-(k + 1)]
                # Don't replace elites
                if idx_to_replace in elite_indices:
                    continue
                p = jittered_weight_sort_perm()
                population[idx_to_replace] = p
                fitnesses[idx_to_replace] = evaluate(p)
        
        iteration += 1
    
    # Final phase: use remaining time for small perturbations on best
    final_cutoff = time_limit * 0.98
    best_result_fitness = best_fitness
    
    while _time() - start_time < final_cutoff:
        # Try small perturbation on best permutation
        p = best_perm[:]
        move_type = _randint(0, 2)
        if move_type == 0:
            # Single swap
            a = _randint(0, n - 1)
            b = _randint(0, n - 1)
            p[a], p[b] = p[b], p[a]
        elif move_type == 1:
            # Single insert
            a = _randint(0, n - 1)
            b = _randint(0, n - 1)
            item = p.pop(a)
            p.insert(b, item)
        else:
            # Two adjacent swaps
            a = _randint(0, n - 2)
            p[a], p[a + 1] = p[a + 1], p[a]
            if n > 2:
                b = _randint(0, n - 2)
                p[b], p[b + 1] = p[b + 1], p[b]
        
        fit = evaluate(p)
        if fit < best_result_fitness:
            best_result_fitness = fit
            best_perm = p[:]
    
    return make_packing_result(best_perm)