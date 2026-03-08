import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    # Pre-filter oversized items
    valid_indices = []
    for i in range(n):
        if weights[i] <= C and weights[i] > 0:
            valid_indices.append(i)
    
    nv = len(valid_indices)
    if nv == 0:
        return {"packing": [], "bin_weights": []}
    
    valid_weights = [weights[i] for i in valid_indices]
    
    # Lower bound computation
    total_weight = sum(valid_weights)
    L1 = (total_weight + C - 1) // C
    
    half_C = C / 2.0
    large_items = [w for w in valid_weights if w > half_C]
    n_large = len(large_items)
    total_small = sum(w for w in valid_weights if w <= half_C)
    total_remaining_in_large = sum(C - w for w in large_items)
    absorbable = min(total_small, total_remaining_in_large)
    leftover_small = total_small - absorbable
    L2 = n_large + ((leftover_small + C - 1) // C if leftover_small > 0 else 0)
    lower_bound = max(L1, L2)
    
    # Fast BFD decoder: takes a permutation (order of items to place)
    def decode_bfd_order(order):
        caps = []  # sorted remaining capacities (ascending)
        cap_bins = []  # corresponding bin indices
        bin_items = []
        bin_wts = []
        
        for idx in order:
            w = valid_weights[idx]
            pos = bisect.bisect_left(caps, w)
            if pos < len(caps):
                old_cap = caps[pos]
                bi = cap_bins[pos]
                del caps[pos]
                del cap_bins[pos]
                new_cap = old_cap - w
                bin_items[bi].append(idx)
                bin_wts[bi] += w
                npos = bisect.bisect_left(caps, new_cap)
                caps.insert(npos, new_cap)
                cap_bins.insert(npos, bi)
            else:
                bi = len(bin_items)
                bin_items.append([idx])
                bin_wts.append(w)
                new_cap = C - w
                npos = bisect.bisect_left(caps, new_cap)
                caps.insert(npos, new_cap)
                cap_bins.insert(npos, bi)
        
        return bin_items, bin_wts
    
    # Decode from priority vector: sort by priority descending, then apply BFD
    def decode_priorities(priorities):
        order = sorted(range(nv), key=lambda i: -priorities[i])
        return decode_bfd_order(order)
    
    # Falkenauer fitness (higher is better)
    def fitness(bin_wts):
        num_bins = len(bin_wts)
        if num_bins == 0:
            return 0.0
        fullness_sum = 0.0
        C_inv = 1.0 / C
        for bw in bin_wts:
            ratio = bw * C_inv
            fullness_sum += ratio * ratio
        return -num_bins + fullness_sum / num_bins
    
    def evaluate(priorities):
        bi, bw = decode_priorities(priorities)
        f = fitness(bw)
        return bi, bw, f
    
    # Population size
    pop_size = min(30, max(8, nv // 5))
    
    population = []
    fitnesses_list = []
    packings = []
    
    # Helper: create priority vector from ordering
    def order_to_priorities(order):
        p = [0.0] * nv
        for rank, idx in enumerate(order):
            p[idx] = (nv - rank) / nv
        return p
    
    # Initialization strategies
    init_orders = []
    
    # 1. FFD (weight descending)
    init_orders.append(sorted(range(nv), key=lambda i: -valid_weights[i]))
    
    # 2. Weight ascending
    init_orders.append(sorted(range(nv), key=lambda i: valid_weights[i]))
    
    # 3. Large first (>C/3), then rest by weight desc
    third_C = C / 3.0
    big = sorted([i for i in range(nv) if valid_weights[i] > third_C], key=lambda i: -valid_weights[i])
    sml = sorted([i for i in range(nv) if valid_weights[i] <= third_C], key=lambda i: -valid_weights[i])
    init_orders.append(big + sml)
    
    # 4. Large desc, medium desc, small asc
    large_g = sorted([i for i in range(nv) if valid_weights[i] > C // 2], key=lambda i: -valid_weights[i])
    medium_g = sorted([i for i in range(nv) if C // 3 < valid_weights[i] <= C // 2], key=lambda i: -valid_weights[i])
    small_g = sorted([i for i in range(nv) if valid_weights[i] <= C // 3], key=lambda i: valid_weights[i])
    init_orders.append(large_g + medium_g + small_g)
    
    # 5. Large desc, medium desc, small desc
    small_g2 = sorted([i for i in range(nv) if valid_weights[i] <= C // 3], key=lambda i: -valid_weights[i])
    init_orders.append(large_g + medium_g + small_g2)
    
    # 6-7. Random shuffles of large + sorted rest
    for _ in range(2):
        bg = [i for i in range(nv) if valid_weights[i] > third_C]
        random.shuffle(bg)
        sm = sorted([i for i in range(nv) if valid_weights[i] <= third_C], key=lambda i: -valid_weights[i])
        init_orders.append(bg + sm)
    
    for idx_o, order in enumerate(init_orders):
        if len(population) >= pop_size:
            break
        p = order_to_priorities(order)
        bi, bw, f = evaluate(p)
        population.append(p)
        fitnesses_list.append(f)
        packings.append((bi, bw))
    
    # Fill remaining with random
    while len(population) < pop_size:
        p = [random.random() for _ in range(nv)]
        bi, bw, f = evaluate(p)
        population.append(p)
        fitnesses_list.append(f)
        packings.append((bi, bw))
    
    # Track global best
    best_idx = max(range(pop_size), key=lambda i: fitnesses_list[i])
    global_best_fitness = fitnesses_list[best_idx]
    global_best_packing = packings[best_idx]
    global_best_priorities = population[best_idx][:]
    global_best_num_bins = len(global_best_packing[0])
    
    if global_best_num_bins <= lower_bound:
        best_bins, best_bw = global_best_packing
        result_packing = [[valid_indices[i] for i in bc] for bc in best_bins]
        return {"packing": result_packing, "bin_weights": best_bw}
    
    # FDO main loop
    wf_max = 0.9
    wf_min = 0.02
    max_iterations = 200000
    iteration = 0
    
    time_check_interval = max(1, pop_size // 2)
    stagnation = 0
    last_best_bins = global_best_num_bins
    
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break
        
        if global_best_num_bins <= lower_bound:
            break
        
        iteration += 1
        
        # Sort fitnesses for median
        sorted_fits = sorted(fitnesses_list)
        median_fitness = sorted_fits[pop_size // 2]
        
        best_pop_idx = max(range(pop_size), key=lambda i: fitnesses_list[i])
        best_pop_priorities = population[best_pop_idx]
        best_pop_fitness = fitnesses_list[best_pop_idx]
        worst_pop_fitness = sorted_fits[0]
        
        fit_range = abs(best_pop_fitness - worst_pop_fitness)
        if fit_range < 1e-12:
            fit_range = 1e-12
        
        # Adaptive weight factor with decay
        progress = min(1.0, iteration / max_iterations)
        wf = wf_max - (wf_max - wf_min) * progress
        
        # Track stagnation
        if global_best_num_bins < last_best_bins:
            last_best_bins = global_best_num_bins
            stagnation = 0
        else:
            stagnation += 1
        
        agent_counter = 0
        for i in range(pop_size):
            agent_counter += 1
            if agent_counter % time_check_interval == 0:
                if time.time() - start_time >= time_limit * 0.95:
                    break
            
            if i == best_pop_idx:
                # Best agent: small perturbation
                noise = 0.03 + 0.07 * random.random()
                new_p = [population[i][d] + random.gauss(0, noise) for d in range(nv)]
            else:
                pace = wf * abs(fitnesses_list[i] - best_pop_fitness) / fit_range
                
                if fitnesses_list[i] >= median_fitness:
                    # Exploitation: move toward best
                    new_p = [
                        best_pop_priorities[d] + pace * random.uniform(-1, 1)
                        for d in range(nv)
                    ]
                else:
                    # Exploration: move toward random agent or global best
                    if random.random() < 0.3:
                        target = global_best_priorities
                    else:
                        r = random.randint(0, pop_size - 1)
                        while r == i:
                            r = random.randint(0, pop_size - 1)
                        target = population[r]
                    new_p = [
                        target[d] + pace * random.uniform(-1, 1)
                        for d in range(nv)
                    ]
            
            # Evaluate directly (skip rank_normalize for speed - order is preserved)
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
                    stagnation = 0
                    if global_best_num_bins <= lower_bound:
                        break
        
        # Reinitialize worst agents periodically
        reinit_period = 20 if stagnation < 50 else 10
        if iteration % reinit_period == 0:
            n_replace = max(1, pop_size // 4)
            worst_indices = sorted(range(pop_size), key=lambda i: fitnesses_list[i])[:n_replace]
            for ki, wi in enumerate(worst_indices):
                if wi == best_pop_idx:
                    continue
                r = random.random()
                if r < 0.4:
                    # Perturbation of global best
                    noise_scale = 0.05 + 0.3 * random.random()
                    new_p = [global_best_priorities[d] + random.gauss(0, noise_scale) for d in range(nv)]
                elif r < 0.7:
                    # Crossover between global best and random member
                    partner = random.randint(0, pop_size - 1)
                    new_p = [
                        global_best_priorities[d] if random.random() < 0.7 else population[partner][d]
                        for d in range(nv)
                    ]
                else:
                    # Completely random
                    new_p = [random.random() for _ in range(nv)]
                
                new_bins, new_bw, new_fit = evaluate(new_p)
                population[wi] = new_p
                fitnesses_list[wi] = new_fit
                packings[wi] = (new_bins, new_bw)
                if new_fit > global_best_fitness:
                    global_best_fitness = new_fit
                    global_best_packing = (new_bins, new_bw)
                    global_best_priorities = new_p[:]
                    global_best_num_bins = len(new_bins)
                    stagnation = 0
        
        # Heavy restart on long stagnation
        if stagnation > 100:
            stagnation = 0
            for i in range(pop_size):
                if i == best_pop_idx:
                    continue
                if random.random() < 0.7:
                    noise_scale = 0.1 + 0.5 * random.random()
                    new_p = [global_best_priorities[d] + random.gauss(0, noise_scale) for d in range(nv)]
                else:
                    new_p = [random.random() for _ in range(nv)]
                new_bins, new_bw, new_fit = evaluate(new_p)
                population[i] = new_p
                fitnesses_list[i] = new_fit
                packings[i] = (new_bins, new_bw)
                if new_fit > global_best_fitness:
                    global_best_fitness = new_fit
                    global_best_packing = (new_bins, new_bw)
                    global_best_priorities = new_p[:]
                    global_best_num_bins = len(new_bins)
    
    # Map back to original indices
    best_bins, best_bw = global_best_packing
    result_packing = [[valid_indices[i] for i in bc] for bc in best_bins]
    
    return {
        "packing": result_packing,
        "bin_weights": best_bw
    }