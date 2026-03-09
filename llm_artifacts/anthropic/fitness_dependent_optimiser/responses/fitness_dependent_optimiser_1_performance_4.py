import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    # Pre-filter
    valid_indices = []
    for i in range(n):
        if 0 < weights[i] <= C:
            valid_indices.append(i)
    
    nv = len(valid_indices)
    if nv == 0:
        return {"packing": [], "bin_weights": []}
    
    valid_weights = [weights[i] for i in valid_indices]
    
    # Lower bound
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
    
    # L3 bound (Martello-Toth)
    lower_bound = max(L1, L2)
    
    # Try tighter bounds with different thresholds
    for k in range(1, C // 2 + 1):
        if k > 50 and k % 5 != 0:
            continue
        if k > 200:
            break
        n1 = sum(1 for w in valid_weights if w > C - k)
        n2 = sum(1 for w in valid_weights if C - k >= w > C // 2)
        n3_total = sum(1 for w in valid_weights if k <= w <= C // 2)
        s_n2 = sum(w for w in valid_weights if C - k >= w > C // 2)
        residual = n2 * C - s_n2
        s_j3 = sum(w for w in valid_weights if k <= w <= C // 2)
        n3_fit = 0
        if C > 0:
            n3_fit = min(n3_total, residual // k if k > 0 else n3_total)
        leftover_n3 = max(0, s_j3 - residual)
        lb_k = n1 + n2 + max(0, (leftover_n3 + C - 1) // C)
        if lb_k > lower_bound:
            lower_bound = lb_k
    
    # BFD decoder using sorted list with bisect
    def decode_bfd_order(order):
        caps = []  # sorted remaining capacities (ascending)
        cap_bins = []  # corresponding bin indices
        bin_items = []
        bin_wts = []
        
        for idx in order:
            w = valid_weights[idx]
            # Find leftmost bin with capacity >= w (best fit = smallest such capacity)
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
    
    # FFD decoder (first fit decreasing - simpler, sometimes different results)
    def decode_ffd_order(order):
        bin_items = []
        bin_wts = []
        bin_remaining = []
        
        for idx in order:
            w = valid_weights[idx]
            best_bi = -1
            best_rem = C + 1
            for bi in range(len(bin_remaining)):
                rem = bin_remaining[bi]
                if rem >= w and rem < best_rem:
                    best_rem = rem
                    best_bi = bi
            if best_bi >= 0:
                bin_items[best_bi].append(idx)
                bin_wts[best_bi] += w
                bin_remaining[best_bi] -= w
            else:
                bin_items.append([idx])
                bin_wts.append(w)
                bin_remaining.append(C - w)
        
        return bin_items, bin_wts
    
    def decode_priorities(priorities):
        order = sorted(range(nv), key=lambda i: -priorities[i])
        return decode_bfd_order(order)
    
    # Fitness: strongly penalize number of bins, use fullness as tiebreaker
    def fitness(bin_wts):
        num_bins = len(bin_wts)
        if num_bins == 0:
            return 0.0
        C_inv = 1.0 / C
        fullness_sum = 0.0
        for bw in bin_wts:
            ratio = bw * C_inv
            fullness_sum += ratio * ratio
        avg_fullness = fullness_sum / num_bins
        return -num_bins * 1000.0 + avg_fullness
    
    def evaluate(priorities):
        bi, bw = decode_priorities(priorities)
        f = fitness(bw)
        return bi, bw, f
    
    # Population size
    pop_size = min(40, max(10, nv // 4))
    
    population = []
    fitnesses_list = []
    packings = []
    
    def order_to_priorities(order):
        p = [0.0] * nv
        for rank, idx in enumerate(order):
            p[idx] = (nv - rank) / nv
        return p
    
    # Multiple initialization strategies
    init_orders = []
    
    # 1. FFD (weight descending)
    init_orders.append(sorted(range(nv), key=lambda i: -valid_weights[i]))
    
    # 2. Weight ascending
    init_orders.append(sorted(range(nv), key=lambda i: valid_weights[i]))
    
    # 3. Large first (>C/3), then rest desc
    third_C = C / 3.0
    big = sorted([i for i in range(nv) if valid_weights[i] > third_C], key=lambda i: -valid_weights[i])
    sml = sorted([i for i in range(nv) if valid_weights[i] <= third_C], key=lambda i: -valid_weights[i])
    init_orders.append(big + sml)
    
    # 4. Large desc, medium desc, small asc
    half = C // 2
    third = C // 3
    large_g = sorted([i for i in range(nv) if valid_weights[i] > half], key=lambda i: -valid_weights[i])
    medium_g = sorted([i for i in range(nv) if third < valid_weights[i] <= half], key=lambda i: -valid_weights[i])
    small_g = sorted([i for i in range(nv) if valid_weights[i] <= third], key=lambda i: valid_weights[i])
    init_orders.append(large_g + medium_g + small_g)
    
    # 5. Large desc, medium desc, small desc
    small_g2 = sorted([i for i in range(nv) if valid_weights[i] <= third], key=lambda i: -valid_weights[i])
    init_orders.append(large_g + medium_g + small_g2)
    
    # 6. Sorted by C - 2*w (tries to pair complementary items)
    init_orders.append(sorted(range(nv), key=lambda i: abs(valid_weights[i] - C / 2.0)))
    
    # 7. Sort by weight mod (C/2)
    init_orders.append(sorted(range(nv), key=lambda i: -(valid_weights[i] % max(1, C // 2))))
    
    # 8-12. Random permutations with large items first
    for _ in range(5):
        bg = [i for i in range(nv) if valid_weights[i] > third_C]
        random.shuffle(bg)
        sm = [i for i in range(nv) if valid_weights[i] <= third_C]
        random.shuffle(sm)
        init_orders.append(bg + sm)
    
    # 13-17. Fully random permutations
    for _ in range(5):
        perm = list(range(nv))
        random.shuffle(perm)
        init_orders.append(perm)
    
    # 18-22. Weight desc with noise
    base_order = sorted(range(nv), key=lambda i: -valid_weights[i])
    for noise_level in [0.05, 0.1, 0.2, 0.3, 0.5]:
        p = [0.0] * nv
        for rank, idx in enumerate(base_order):
            p[idx] = (nv - rank) / nv + random.gauss(0, noise_level)
        new_order = sorted(range(nv), key=lambda i: -p[i])
        init_orders.append(new_order)
    
    for order in init_orders:
        if len(population) >= pop_size:
            break
        p = order_to_priorities(order)
        bi, bw, f = evaluate(p)
        population.append(p)
        fitnesses_list.append(f)
        packings.append((bi, bw))
    
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
    wf_max = 0.8
    wf_min = 0.01
    iteration = 0
    max_iterations = 500000
    stagnation = 0
    last_best_bins = global_best_num_bins
    time_check_interval = max(1, pop_size)
    check_counter = 0
    
    while iteration < max_iterations:
        check_counter += 1
        if check_counter % time_check_interval == 0:
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
        
        progress = min(1.0, iteration / max_iterations)
        wf = wf_max - (wf_max - wf_min) * progress
        
        if global_best_num_bins < last_best_bins:
            last_best_bins = global_best_num_bins
            stagnation = 0
        else:
            stagnation += 1
        
        for i in range(pop_size):
            if i == best_pop_idx:
                # Best agent: small perturbation
                noise = 0.02 + 0.05 * random.random()
                new_p = [population[i][d] + random.gauss(0, noise) for d in range(nv)]
            else:
                pace = wf * abs(fitnesses_list[i] - best_pop_fitness) / fit_range
                
                if fitnesses_list[i] >= median_fitness:
                    # Exploitation: move toward best with crossover
                    alpha = random.random()
                    new_p = [
                        alpha * best_pop_priorities[d] + (1 - alpha) * population[i][d] + pace * random.gauss(0, 0.5)
                        for d in range(nv)
                    ]
                else:
                    # Exploration
                    if random.random() < 0.4:
                        target = global_best_priorities
                    elif random.random() < 0.5:
                        # Tournament selection
                        r1 = random.randint(0, pop_size - 1)
                        r2 = random.randint(0, pop_size - 1)
                        target = population[r1] if fitnesses_list[r1] > fitnesses_list[r2] else population[r2]
                    else:
                        r = random.randint(0, pop_size - 1)
                        target = population[r]
                    
                    new_p = [
                        target[d] + pace * random.gauss(0, 1.0)
                        for d in range(nv)
                    ]
            
            new_bins, new_bw, new_fit = evaluate(new_p)
            
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
        
        # Reinitialize worst agents
        reinit_period = 15 if stagnation < 30 else 8
        if iteration % reinit_period == 0:
            n_replace = max(2, pop_size // 3)
            worst_indices = sorted(range(pop_size), key=lambda i: fitnesses_list[i])[:n_replace]
            for wi in worst_indices:
                if wi == best_pop_idx:
                    continue
                r = random.random()
                if r < 0.35:
                    # Perturbation of global best
                    noise_scale = 0.03 + 0.2 * random.random()
                    new_p = [global_best_priorities[d] + random.gauss(0, noise_scale) for d in range(nv)]
                elif r < 0.55:
                    # Crossover global best and random
                    partner = random.randint(0, pop_size - 1)
                    alpha = 0.5 + 0.3 * random.random()
                    new_p = [
                        global_best_priorities[d] * alpha + population[partner][d] * (1 - alpha)
                        for d in range(nv)
                    ]
                elif r < 0.75:
                    # Segment swap from global best
                    new_p = global_best_priorities[:]
                    seg_len = max(1, nv // 10)
                    start_pos = random.randint(0, nv - 1)
                    for j in range(seg_len):
                        idx_j = (start_pos + j) % nv
                        new_p[idx_j] = random.random()
                elif r < 0.9:
                    # Smart init: weight-based with noise
                    base = sorted(range(nv), key=lambda i: -valid_weights[i])
                    ns = 0.1 + 0.4 * random.random()
                    new_p = [0.0] * nv
                    for rank, idx in enumerate(base):
                        new_p[idx] = (nv - rank) / nv + random.gauss(0, ns)
                else:
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
        if stagnation > 60:
            stagnation = 0
            for i in range(pop_size):
                if i == best_pop_idx:
                    continue
                r = random.random()
                if r < 0.5:
                    noise_scale = 0.05 + 0.4 * random.random()
                    new_p = [global_best_priorities[d] + random.gauss(0, noise_scale) for d in range(nv)]
                elif r < 0.8:
                    # New smart ordering
                    base = sorted(range(nv), key=lambda i: -valid_weights[i])
                    ns = random.random() * 0.5
                    new_p = [0.0] * nv
                    for rank, idx in enumerate(base):
                        new_p[idx] = (nv - rank) / nv + random.gauss(0, ns)
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
    
    best_bins, best_bw = global_best_packing
    result_packing = [[valid_indices[i] for i in bc] for bc in best_bins]
    
    return {
        "packing": result_packing,
        "bin_weights": best_bw
    }
