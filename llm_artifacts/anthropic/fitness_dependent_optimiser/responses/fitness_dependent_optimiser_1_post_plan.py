import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Step 1: Pre-filter oversized items
    valid_indices = []
    oversized_indices = []
    for i in range(n):
        if weights[i] > bin_capacity:
            oversized_indices.append(i)
        else:
            valid_indices.append(i)
    
    valid_weights = [weights[i] for i in valid_indices]
    nv = len(valid_indices)  # number of valid items
    
    if nv == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    # Step 10: Tighter L2 lower bound
    total_weight = sum(valid_weights)
    L1 = (total_weight + C - 1) // C
    
    # L2 bound
    # Count items with weight > C/2
    half_C = C / 2.0
    large_items = [w for w in valid_weights if w > half_C]
    n_large = len(large_items)
    # Items with weight <= C/2
    small_items_weights = sorted([w for w in valid_weights if w <= half_C], reverse=True)
    
    # For each large item, it can pair with small items that fit
    # Simple L2: each large item needs its own bin. Remaining capacity in those bins
    # can absorb some small items.
    remaining_caps = sorted([C - w for w in large_items])  # remaining capacity in large-item bins
    unpaired_weight = 0
    small_idx = 0
    rc_idx = 0
    # Greedy matching: try to fit small items into remaining capacity of large-item bins
    # Sort remaining caps ascending, small items descending
    remaining_caps_sorted = sorted(remaining_caps)
    small_sorted = sorted(small_items_weights, reverse=True)
    used_cap = [0] * n_large
    
    # Simple approach: total small weight minus what can fit in large bins
    total_small = sum(small_items_weights)
    total_remaining_in_large = sum(remaining_caps)
    absorbable = min(total_small, total_remaining_in_large)
    leftover_small = total_small - absorbable
    L2 = n_large + (leftover_small + C - 1) // C if leftover_small > 0 else n_large
    
    lower_bound = max(L1, L2)
    
    # Step 3: BFD decoder with bisect optimization
    def decode_bfd(priorities):
        # Sort items by priority descending
        order = sorted(range(nv), key=lambda i: -priorities[i])
        
        # Maintain sorted list of remaining capacities and corresponding bin indices
        # sorted_caps: list of (remaining_cap, bin_id)
        # We'll use a list of remaining caps sorted ascending for bisect
        bins_content = []  # list of lists of item indices (in valid_indices space)
        bin_weights_list = []
        sorted_caps = []  # sorted list of remaining capacities
        cap_to_bins = {}  # remaining_cap -> list of bin indices with that cap
        
        for idx in order:
            w = valid_weights[idx]
            if w > C:
                continue
            
            # Find best fit: smallest remaining capacity >= w
            pos = bisect.bisect_left(sorted_caps, w)
            if pos < len(sorted_caps):
                # Best fit is the one at position pos (smallest remaining that fits)
                old_cap = sorted_caps[pos]
                # Remove this capacity from sorted list
                sorted_caps.pop(pos)
                
                # Find which bin has this capacity
                bin_list = cap_to_bins[old_cap]
                b = bin_list.pop()
                if not bin_list:
                    del cap_to_bins[old_cap]
                
                bins_content[b].append(idx)
                bin_weights_list[b] += w
                new_cap = old_cap - w
                
                # Insert new capacity
                ins_pos = bisect.bisect_left(sorted_caps, new_cap)
                sorted_caps.insert(ins_pos, new_cap)
                if new_cap in cap_to_bins:
                    cap_to_bins[new_cap].append(b)
                else:
                    cap_to_bins[new_cap] = [b]
            else:
                # Open new bin
                b = len(bins_content)
                bins_content.append([idx])
                bin_weights_list.append(w)
                new_cap = C - w
                ins_pos = bisect.bisect_left(sorted_caps, new_cap)
                sorted_caps.insert(ins_pos, new_cap)
                if new_cap in cap_to_bins:
                    cap_to_bins[new_cap].append(b)
                else:
                    cap_to_bins[new_cap] = [b]
        
        return bins_content, bin_weights_list
    
    # Step 2: Falkenauer-style fitness
    def fitness(bins_content, bin_weights_list):
        num_bins = len(bin_weights_list)
        if num_bins == 0:
            return 0.0
        fullness_sum = 0.0
        for bw in bin_weights_list:
            ratio = bw / C
            fullness_sum += ratio * ratio
        return -num_bins + fullness_sum / num_bins
    
    # Step 4: Rank normalization to [0, 1]
    def rank_normalize(priorities):
        if nv <= 1:
            return [0.5] * nv
        indexed = sorted(range(nv), key=lambda i: priorities[i])
        result = [0.0] * nv
        for rank, idx in enumerate(indexed):
            result[idx] = rank / (nv - 1)
        return result
    
    # Evaluate a priority vector
    def evaluate(priorities):
        bins_content, bin_weights_list = decode_bfd(priorities)
        fit = fitness(bins_content, bin_weights_list)
        return bins_content, bin_weights_list, fit
    
    # Step 7: Reduced population size
    pop_size = min(30, max(8, nv // 5))
    
    # Step 5: Diverse population initialization
    population = []
    fitnesses_list = []
    packings = []
    
    # FFD ordering (weight descending)
    ffd_priorities = [0.0] * nv
    sorted_by_weight_desc = sorted(range(nv), key=lambda i: -valid_weights[i])
    for rank, idx in enumerate(sorted_by_weight_desc):
        ffd_priorities[idx] = (nv - rank) / nv
    ffd_priorities = rank_normalize(ffd_priorities)
    population.append(ffd_priorities)
    bc, bw, f = evaluate(ffd_priorities)
    packings.append((bc, bw))
    fitnesses_list.append(f)
    
    # Weight ascending
    if pop_size > 1:
        asc_priorities = [0.0] * nv
        sorted_by_weight_asc = sorted(range(nv), key=lambda i: valid_weights[i])
        for rank, idx in enumerate(sorted_by_weight_asc):
            asc_priorities[idx] = (nv - rank) / nv
        asc_priorities = rank_normalize(asc_priorities)
        population.append(asc_priorities)
        bc, bw, f = evaluate(asc_priorities)
        packings.append((bc, bw))
        fitnesses_list.append(f)
    
    # Items > C/3 first, then rest by weight
    if pop_size > 2:
        third_C = C / 3.0
        big = [i for i in range(nv) if valid_weights[i] > third_C]
        small = [i for i in range(nv) if valid_weights[i] <= third_C]
        random.shuffle(big)
        small.sort(key=lambda i: -valid_weights[i])
        order3 = big + small
        p3 = [0.0] * nv
        for rank, idx in enumerate(order3):
            p3[idx] = (nv - rank) / nv
        p3 = rank_normalize(p3)
        population.append(p3)
        bc, bw, f = evaluate(p3)
        packings.append((bc, bw))
        fitnesses_list.append(f)
    
    # Group by size classes with different orderings (2 solutions)
    for variant in range(2):
        if len(population) >= pop_size:
            break
        large_g = [i for i in range(nv) if valid_weights[i] > C // 2]
        medium_g = [i for i in range(nv) if C // 3 < valid_weights[i] <= C // 2]
        small_g = [i for i in range(nv) if valid_weights[i] <= C // 3]
        if variant == 0:
            large_g.sort(key=lambda i: -valid_weights[i])
            medium_g.sort(key=lambda i: -valid_weights[i])
            small_g.sort(key=lambda i: valid_weights[i])
        else:
            random.shuffle(large_g)
            random.shuffle(medium_g)
            random.shuffle(small_g)
        order_v = large_g + medium_g + small_g
        pv = [0.0] * nv
        for rank, idx in enumerate(order_v):
            pv[idx] = (nv - rank) / nv
        pv = rank_normalize(pv)
        population.append(pv)
        bc, bw, f = evaluate(pv)
        packings.append((bc, bw))
        fitnesses_list.append(f)
    
    # Remaining: random permutations as random keys
    while len(population) < pop_size:
        p = [random.random() for _ in range(nv)]
        p = rank_normalize(p)
        population.append(p)
        bc, bw, f = evaluate(p)
        packings.append((bc, bw))
        fitnesses_list.append(f)
    
    # Track global best
    best_idx = max(range(pop_size), key=lambda i: fitnesses_list[i])
    global_best_fitness = fitnesses_list[best_idx]
    global_best_packing = packings[best_idx]
    global_best_priorities = population[best_idx][:]
    global_best_num_bins = len(global_best_packing[0])
    
    # Step 6 & 8: FDO main loop
    max_iterations = 100000
    iteration = 0
    
    wf_max = 0.8
    wf_min = 0.05
    
    # Step 11: Time check frequency
    time_check_interval = max(1, pop_size // 3)
    agent_counter = 0
    
    while iteration < max_iterations:
        # Check time at start of each generation
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break
        
        # Check optimality
        if global_best_num_bins <= lower_bound:
            break
        
        iteration += 1
        
        # Step 8: Precompute median once per generation
        sorted_fits = sorted(fitnesses_list)
        median_fitness = sorted_fits[pop_size // 2]
        
        best_pop_idx = max(range(pop_size), key=lambda i: fitnesses_list[i])
        best_pop_priorities = population[best_pop_idx]
        best_pop_fitness = fitnesses_list[best_pop_idx]
        worst_pop_fitness = sorted_fits[0]
        
        # Fitness range for pace normalization
        fit_range = abs(best_pop_fitness - worst_pop_fitness)
        if fit_range < 1e-10:
            fit_range = 1e-10
        
        # Adaptive weight factor
        wf = wf_max - (wf_max - wf_min) * (iteration / max_iterations)
        
        agent_counter = 0
        for i in range(pop_size):
            agent_counter += 1
            # Step 11: periodic time check
            if agent_counter % time_check_interval == 0:
                if time.time() - start_time >= time_limit * 0.95:
                    break
            
            if i == best_pop_idx:
                # Best agent: small random perturbation
                new_p = [population[i][d] + random.gauss(0, 0.05) for d in range(nv)]
            else:
                # Step 6: Proper FDO update
                # Normalized pace
                pace = wf * abs(fitnesses_list[i] - best_pop_fitness) / fit_range
                
                if fitnesses_list[i] >= median_fitness:
                    # Exploitation: move toward best
                    new_p = [
                        best_pop_priorities[d] + pace * random.uniform(-1, 1)
                        for d in range(nv)
                    ]
                else:
                    # Exploration: move toward random agent
                    r = random.randint(0, pop_size - 1)
                    while r == i:
                        r = random.randint(0, pop_size - 1)
                    new_p = [
                        population[r][d] + pace * random.uniform(-1, 1)
                        for d in range(nv)
                    ]
            
            # Step 4: rank normalize
            new_p = rank_normalize(new_p)
            
            # Evaluate
            new_bins, new_bw, new_fit = evaluate(new_p)
            
            # Greedy selection
            if new_fit >= fitnesses_list[i]:
                population[i] = new_p
                fitnesses_list[i] = new_fit
                packings[i] = (new_bins, new_bw)
                
                if new_fit > global_best_fitness:
                    global_best_fitness = new_fit
                    global_best_packing = (new_bins, new_bw)
                    global_best_priorities = new_p[:]
                    global_best_num_bins = len(new_bins)
                    if global_best_num_bins <= lower_bound:
                        break
        
        # Step 9: Stronger reinitialization
        if iteration % 30 == 0:
            n_replace = max(1, pop_size // 5)
            worst_indices = sorted(range(pop_size), key=lambda i: fitnesses_list[i])[:n_replace]
            for ki, wi in enumerate(worst_indices):
                if wi == best_pop_idx:
                    continue
                if ki < len(worst_indices) // 2:
                    # Perturbation of global best with varying noise
                    noise_scale = 0.1 + 0.4 * random.random()
                    new_p = [global_best_priorities[d] + random.gauss(0, noise_scale) for d in range(nv)]
                else:
                    # Completely random
                    new_p = [random.random() for _ in range(nv)]
                
                new_p = rank_normalize(new_p)
                new_bins, new_bw, new_fit = evaluate(new_p)
                population[wi] = new_p
                fitnesses_list[wi] = new_fit
                packings[wi] = (new_bins, new_bw)
                if new_fit > global_best_fitness:
                    global_best_fitness = new_fit
                    global_best_packing = (new_bins, new_bw)
                    global_best_priorities = new_p[:]
                    global_best_num_bins = len(new_bins)
    
    # Map back to original indices
    best_bins, best_bw = global_best_packing
    result_packing = []
    for bin_content in best_bins:
        result_packing.append([valid_indices[i] for i in bin_content])
    
    return {
        "packing": result_packing,
        "bin_weights": best_bw
    }