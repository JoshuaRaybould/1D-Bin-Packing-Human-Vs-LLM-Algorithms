import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    w_arr = weights  # list
    C = bin_capacity
    
    _random = random.random
    _randint = random.randint
    _shuffle = random.shuffle
    _time = time.time
    
    # Fast First Fit decode
    def decode_ff(perm):
        bin_rem = []
        bin_contents = []
        bin_wts = []
        for idx in perm:
            w = w_arr[idx]
            placed = False
            for b in range(len(bin_rem)):
                if bin_rem[b] >= w:
                    bin_rem[b] -= w
                    bin_contents[b].append(idx)
                    bin_wts[b] += w
                    placed = True
                    break
            if not placed:
                bin_rem.append(C - w)
                bin_contents.append([idx])
                bin_wts.append(w)
        return bin_contents, bin_wts
    
    # Fast Best Fit decode
    def decode_bf(perm):
        bin_rem = []
        bin_contents = []
        bin_wts = []
        sorted_rem = []  # (remaining, bin_index)
        
        for idx in perm:
            w = w_arr[idx]
            pos = bisect.bisect_left(sorted_rem, (w,))
            if pos < len(sorted_rem):
                rem, b = sorted_rem[pos]
                del sorted_rem[pos]
                new_rem = rem - w
                bin_rem[b] = new_rem
                bin_contents[b].append(idx)
                bin_wts[b] += w
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, b))
            else:
                b = len(bin_rem)
                new_rem = C - w
                bin_rem.append(new_rem)
                bin_contents.append([idx])
                bin_wts.append(w)
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, b))
        return bin_contents, bin_wts
    
    # Fast First Fit Decreasing - just count bins and return fitness
    def eval_ff_fast(perm):
        bin_rem = []
        bin_wts = []
        for idx in perm:
            w = w_arr[idx]
            placed = False
            for b in range(len(bin_rem)):
                if bin_rem[b] >= w:
                    bin_rem[b] -= w
                    bin_wts[b] += w
                    placed = True
                    break
            if not placed:
                bin_rem.append(C - w)
                bin_wts.append(w)
        nb = len(bin_wts)
        if nb == 0:
            return 0.0
        s = 0.0
        for bw in bin_wts:
            r = bw / C
            s += r * r
        return nb - s / nb
    
    # Best Fit eval - faster without building contents
    def eval_bf_fast(perm):
        bin_wts = []
        sorted_rem = []
        bin_rem = []
        
        for idx in perm:
            w = w_arr[idx]
            pos = bisect.bisect_left(sorted_rem, (w,))
            if pos < len(sorted_rem):
                rem, b = sorted_rem[pos]
                del sorted_rem[pos]
                new_rem = rem - w
                bin_rem[b] = new_rem
                bin_wts[b] += w
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, b))
            else:
                b = len(bin_rem)
                new_rem = C - w
                bin_rem.append(new_rem)
                bin_wts.append(w)
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, b))
        nb = len(bin_wts)
        if nb == 0:
            return 0.0
        s = 0.0
        for bw in bin_wts:
            r = bw / C
            s += r * r
        return nb - s / nb
    
    # Evaluate using best of FF and BF
    def evaluate(perm):
        f1 = eval_ff_fast(perm)
        f2 = eval_bf_fast(perm)
        return min(f1, f2)
    
    # For speed during search, use only FF (faster)
    def evaluate_fast(perm):
        return eval_bf_fast(perm)
    
    def make_packing_result(perm):
        bins_bf, bw_bf = decode_bf(perm)
        bins_ff, bw_ff = decode_ff(perm)
        nb_bf = len(bins_bf)
        nb_ff = len(bins_ff)
        if nb_ff < nb_bf:
            return {"packing": bins_ff, "bin_weights": bw_ff}
        elif nb_bf < nb_ff:
            return {"packing": bins_bf, "bin_weights": bw_bf}
        else:
            s_bf = sum((bw / C) ** 2 for bw in bw_bf)
            s_ff = sum((bw / C) ** 2 for bw in bw_ff)
            f_bf = nb_bf - s_bf / nb_bf if nb_bf > 0 else 0
            f_ff = nb_ff - s_ff / nb_ff if nb_ff > 0 else 0
            if f_ff <= f_bf:
                return {"packing": bins_ff, "bin_weights": bw_ff}
            else:
                return {"packing": bins_bf, "bin_weights": bw_bf}
    
    # Generate FFD permutation
    def ffd_perm():
        return sorted(range(n), key=lambda i: -w_arr[i])
    
    def random_perm():
        p = list(range(n))
        _shuffle(p)
        return p
    
    def jittered_perm(jitter_frac=0.05):
        jitter = C * jitter_frac
        return sorted(range(n), key=lambda i: -(w_arr[i] + _random() * jitter))
    
    # For tiny instances
    if n <= 1:
        perm = ffd_perm()
        return make_packing_result(perm)
    
    # Determine if we should use fast (FF-only) or full evaluation
    # For large n, use FF only during search for speed
    use_bf_in_search = n <= 2000
    
    if use_bf_in_search:
        eval_fn = evaluate_fast  # BF is generally better
    else:
        eval_fn = eval_ff_fast
    
    # Population
    pop_size = min(50, max(15, n // 5))
    
    population = []
    fitnesses = []
    
    # FFD
    ffd_p = ffd_perm()
    population.append(ffd_p)
    fitnesses.append(eval_fn(ffd_p))
    
    # Near-FFD variants with small perturbations
    num_near = min(10, pop_size - 1)
    for k in range(num_near):
        p = ffd_p[:]
        num_swaps = max(1, n // (20 - k))
        for _ in range(num_swaps):
            i = _randint(0, n - 1)
            d = _randint(1, min(5, n - 1))
            j = min(n - 1, max(0, i + (_randint(0, 1) * 2 - 1) * d))
            p[i], p[j] = p[j], p[i]
        population.append(p)
        fitnesses.append(eval_fn(p))
    
    # Jittered permutations
    num_jit = min(8, pop_size - len(population))
    for k in range(num_jit):
        jf = 0.02 + 0.1 * k / max(1, num_jit - 1)
        p = jittered_perm(jf)
        population.append(p)
        fitnesses.append(eval_fn(p))
    
    # Fill with random
    while len(population) < pop_size:
        p = random_perm()
        population.append(p)
        fitnesses.append(eval_fn(p))
    
    # Track best
    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    best_perm = population[best_idx][:]
    
    # Precompute: weight order for smarter mutations
    weight_order = ffd_p  # indices sorted by decreasing weight
    weight_rank = [0] * n
    for rank, idx in enumerate(weight_order):
        weight_rank[idx] = rank
    
    # FDO main loop
    generations_no_improve = 0
    main_cutoff = time_limit * 0.88
    
    # Adaptive: periodically check time
    check_interval = max(1, pop_size // 5)
    
    iteration = 0
    while True:
        if iteration % 5 == 0:
            if _time() - start_time >= main_cutoff:
                break
        
        # Find guide (best) and worst
        min_fit = float('inf')
        max_fit = float('-inf')
        guide_idx = 0
        worst_idx = 0
        for i in range(pop_size):
            if fitnesses[i] < min_fit:
                min_fit = fitnesses[i]
                guide_idx = i
            if fitnesses[i] > max_fit:
                max_fit = fitnesses[i]
                worst_idx = i
        
        guide = population[guide_idx]
        fitness_range = max_fit - min_fit
        if fitness_range < 1e-12:
            fitness_range = 1.0
        
        # Elitism: top 2
        sorted_by_fit = sorted(range(pop_size), key=lambda x: fitnesses[x])
        elite_set = set(sorted_by_fit[:2])
        
        improved_this_gen = False
        
        for i in range(pop_size):
            if i in elite_set:
                continue
            
            # FDO weight
            w_i = (fitnesses[i] - min_fit) / fitness_range
            
            # Stagnation boost
            if generations_no_improve > 200:
                w_i = min(1.0, w_i + 0.15)
            
            new_perm = population[i][:]
            
            if w_i < 0.25:
                # EXPLOITATION: very small moves
                num_moves = _randint(1, 2)
                for _ in range(num_moves):
                    r = _random()
                    if r < 0.4:
                        # Insert move: move item a few positions
                        pos = _randint(0, n - 1)
                        new_pos = max(0, min(n - 1, pos + _randint(-3, 3)))
                        if pos != new_pos:
                            item = new_perm.pop(pos)
                            new_perm.insert(new_pos, item)
                    elif r < 0.7:
                        # Swap two nearby items
                        a = _randint(0, n - 1)
                        d = _randint(1, min(4, n - 1))
                        b = min(n - 1, max(0, a + d if _random() < 0.5 else a - d))
                        new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
                    else:
                        # Or-opt: move segment of 2-3 items
                        seg_len = _randint(2, min(3, n))
                        a = _randint(0, n - seg_len)
                        seg = new_perm[a:a + seg_len]
                        del new_perm[a:a + seg_len]
                        b = _randint(0, len(new_perm))
                        for k, item in enumerate(seg):
                            new_perm.insert(b + k, item)
            
            elif w_i < 0.6:
                # BALANCED: crossover with guide
                seg_len = max(1, int((0.3 + 0.3 * w_i) * n))
                start_pos = _randint(0, n - seg_len)
                
                child = [-1] * n
                guide_set = set()
                for k in range(start_pos, start_pos + seg_len):
                    child[k] = guide[k]
                    guide_set.add(guide[k])
                remaining = [x for x in new_perm if x not in guide_set]
                pos = 0
                for k in range(n):
                    if child[k] == -1:
                        child[k] = remaining[pos]
                        pos += 1
                new_perm = child
                
                # Small mutation on top
                if _random() < 0.3:
                    a = _randint(0, n - 1)
                    b = _randint(0, n - 1)
                    new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
            
            else:
                # EXPLORATION
                r = _random()
                if r < 0.35:
                    new_perm = jittered_perm(0.02 + _random() * 0.15)
                elif r < 0.65:
                    # Large crossover with guide + shuffle remaining
                    seg_len = max(1, int((0.5 + _random() * 0.4) * n))
                    start_pos = _randint(0, n - seg_len)
                    child = [-1] * n
                    guide_set = set()
                    for k in range(start_pos, start_pos + seg_len):
                        child[k] = guide[k]
                        guide_set.add(guide[k])
                    remaining = [x for x in new_perm if x not in guide_set]
                    _shuffle(remaining)
                    pos = 0
                    for k in range(n):
                        if child[k] == -1:
                            child[k] = remaining[pos]
                            pos += 1
                    new_perm = child
                else:
                    # Random restart with weight bias
                    new_perm = random_perm()
                    # Sort first portion by weight
                    portion = max(1, int(n * 0.3))
                    head = new_perm[:portion]
                    head.sort(key=lambda x: -w_arr[x])
                    new_perm[:portion] = head
            
            new_fit = eval_fn(new_perm)
            
            if new_fit <= fitnesses[i]:
                population[i] = new_perm
                fitnesses[i] = new_fit
                if new_fit < best_fitness:
                    best_fitness = new_fit
                    best_perm = new_perm[:]
                    improved_this_gen = True
        
        if improved_this_gen:
            generations_no_improve = 0
        else:
            generations_no_improve += 1
        
        # Inject best back if lost
        max_fit_now = -1.0
        worst_now = 0
        for i in range(pop_size):
            if fitnesses[i] > max_fit_now:
                max_fit_now = fitnesses[i]
                worst_now = i
        if best_fitness < max_fit_now and worst_now not in elite_set:
            population[worst_now] = best_perm[:]
            fitnesses[worst_now] = best_fitness
        
        # Diversity injection
        if generations_no_improve > 0 and generations_no_improve % 150 == 0:
            sorted_idx = sorted(range(pop_size), key=lambda x: fitnesses[x])
            num_replace = max(2, int(pop_size * 0.4))
            for k in range(num_replace):
                ri = sorted_idx[-(k + 1)]
                if ri in elite_set:
                    continue
                if _random() < 0.5:
                    p = jittered_perm(0.01 + _random() * 0.2)
                else:
                    p = random_perm()
                population[ri] = p
                fitnesses[ri] = eval_fn(p)
        
        iteration += 1
    
    # Final local search phase on best permutation
    final_cutoff = time_limit * 0.98
    
    # More sophisticated local search
    current = best_perm[:]
    current_fit = best_fitness
    
    # Try systematic neighborhood
    step = 0
    while _time() - start_time < final_cutoff:
        p = current[:]
        step += 1
        move = step % 5
        
        if move == 0:
            # Swap
            a = _randint(0, n - 1)
            b = _randint(0, n - 1)
            p[a], p[b] = p[b], p[a]
        elif move == 1:
            # Insert
            a = _randint(0, n - 1)
            b = _randint(0, n - 1)
            item = p.pop(a)
            p.insert(b, item)
        elif move == 2:
            # Or-opt: move segment of 2
            if n >= 3:
                a = _randint(0, n - 2)
                seg = [p[a], p[a + 1]]
                del p[a:a + 2]
                b = _randint(0, len(p))
                p.insert(b, seg[0])
                p.insert(b + 1, seg[1])
        elif move == 3:
            # Reverse a small segment
            a = _randint(0, n - 1)
            seg_len = _randint(2, min(6, n))
            b = min(n, a + seg_len)
            p[a:b] = p[a:b][::-1]
        else:
            # Double swap
            a = _randint(0, n - 1)
            b = _randint(0, n - 1)
            p[a], p[b] = p[b], p[a]
            c = _randint(0, n - 1)
            d = _randint(0, n - 1)
            p[c], p[d] = p[d], p[c]
        
        # Evaluate with both decoders for final phase
        f = evaluate(p)
        if f < current_fit:
            current_fit = f
            current = p[:]
            best_perm = p[:]
            best_fitness = f
    
    return make_packing_result(best_perm)
