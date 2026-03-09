import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    w_arr = weights
    C = bin_capacity
    
    _random = random.random
    _randint = random.randint
    _shuffle = random.shuffle
    _time = time.time
    _bisect_left = bisect.bisect_left
    _insort = bisect.insort
    
    if n == 1:
        return {"packing": [[0]], "bin_weights": [weights[0]]}
    
    # Best Fit decode - returns (bin_contents, bin_weights)
    def decode_bf(perm):
        bin_rem = []
        bin_contents = []
        bin_wts = []
        sorted_rem = []
        
        for idx in perm:
            w = w_arr[idx]
            pos = _bisect_left(sorted_rem, (w,))
            if pos < len(sorted_rem):
                rem, b = sorted_rem[pos]
                del sorted_rem[pos]
                new_rem = rem - w
                bin_rem[b] = new_rem
                bin_contents[b].append(idx)
                bin_wts[b] += w
                if new_rem > 0:
                    _insort(sorted_rem, (new_rem, b))
            else:
                b = len(bin_rem)
                new_rem = C - w
                bin_rem.append(new_rem)
                bin_contents.append([idx])
                bin_wts.append(w)
                if new_rem > 0:
                    _insort(sorted_rem, (new_rem, b))
        return bin_contents, bin_wts
    
    # First Fit decode
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
    
    # Best Fit eval - faster without building contents
    def eval_bf_fast(perm):
        bin_wts = []
        sorted_rem = []
        bin_rem = []
        
        for idx in perm:
            w = w_arr[idx]
            pos = _bisect_left(sorted_rem, (w,))
            if pos < len(sorted_rem):
                rem, b = sorted_rem[pos]
                del sorted_rem[pos]
                new_rem = rem - w
                bin_rem[b] = new_rem
                bin_wts[b] += w
                if new_rem > 0:
                    _insort(sorted_rem, (new_rem, b))
            else:
                b = len(bin_rem)
                new_rem = C - w
                bin_rem.append(new_rem)
                bin_wts.append(w)
                if new_rem > 0:
                    _insort(sorted_rem, (new_rem, b))
        nb = len(bin_wts)
        if nb == 0:
            return 0.0
        s = 0.0
        Cinv = 1.0 / C
        for bw in bin_wts:
            r = bw * Cinv
            s += r * r
        return nb - s / nb
    
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
        Cinv = 1.0 / C
        for bw in bin_wts:
            r = bw * Cinv
            s += r * r
        return nb - s / nb
    
    # FF with sorted remaining capacity (FFD-like but using bisect for speed)
    def eval_ffs_fast(perm):
        bin_wts = []
        sorted_rem = []  # sorted list of (remaining, bin_index)
        bin_rem = []
        
        for idx in perm:
            w = w_arr[idx]
            # Find first bin that fits (smallest remaining >= w for first-fit feel)
            # Actually for FF we want the first opened bin that fits
            # Let's use sorted by bin index among those that fit
            pos = _bisect_left(sorted_rem, (w,))
            if pos < len(sorted_rem):
                # Pick the one with smallest remaining that still fits (best fit)
                rem, b = sorted_rem[pos]
                del sorted_rem[pos]
                new_rem = rem - w
                bin_rem[b] = new_rem
                bin_wts[b] += w
                if new_rem > 0:
                    _insort(sorted_rem, (new_rem, b))
            else:
                b = len(bin_rem)
                new_rem = C - w
                bin_rem.append(new_rem)
                bin_wts.append(w)
                if new_rem > 0:
                    _insort(sorted_rem, (new_rem, b))
        nb = len(bin_wts)
        if nb == 0:
            return 0.0
        s = 0.0
        Cinv = 1.0 / C
        for bw in bin_wts:
            r = bw * Cinv
            s += r * r
        return nb - s / nb
    
    # Choose eval function based on problem size
    if n <= 500:
        def eval_fn(perm):
            return min(eval_bf_fast(perm), eval_ff_fast(perm))
    elif n <= 2000:
        eval_fn = eval_bf_fast
    else:
        eval_fn = eval_bf_fast
    
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
            Cinv = 1.0 / C
            s_bf = sum((bw * Cinv) ** 2 for bw in bw_bf)
            s_ff = sum((bw * Cinv) ** 2 for bw in bw_ff)
            f_bf = nb_bf - s_bf / nb_bf if nb_bf > 0 else 0
            f_ff = nb_ff - s_ff / nb_ff if nb_ff > 0 else 0
            if f_ff <= f_bf:
                return {"packing": bins_ff, "bin_weights": bw_ff}
            else:
                return {"packing": bins_bf, "bin_weights": bw_bf}
    
    def ffd_perm():
        return sorted(range(n), key=lambda i: -w_arr[i])
    
    def random_perm():
        p = list(range(n))
        _shuffle(p)
        return p
    
    def jittered_perm(jitter_frac=0.05):
        jitter = C * jitter_frac
        return sorted(range(n), key=lambda i: -(w_arr[i] + _random() * jitter))
    
    # Order Crossover (OX)
    def ox_crossover(p1, p2):
        a = _randint(0, n - 1)
        b = _randint(0, n - 1)
        if a > b:
            a, b = b, a
        child = [-1] * n
        seg_set = set()
        for k in range(a, b + 1):
            child[k] = p1[k]
            seg_set.add(p1[k])
        pos = (b + 1) % n
        for k in range(n):
            idx2 = (b + 1 + k) % n
            item = p2[idx2]
            if item not in seg_set:
                child[pos] = item
                pos = (pos + 1) % n
        return child
    
    # Partially Mapped Crossover (PMX) - position-aware
    def pmx_crossover(p1, p2):
        a = _randint(0, n - 1)
        b = _randint(0, n - 1)
        if a > b:
            a, b = b, a
        child = p2[:]
        # Build position map for p2
        pos_in_child = [0] * n
        for i in range(n):
            pos_in_child[child[i]] = i
        for k in range(a, b + 1):
            if child[k] != p1[k]:
                # Swap in child
                old_item = child[k]
                new_item = p1[k]
                pos_new = pos_in_child[new_item]
                child[k] = new_item
                child[pos_new] = old_item
                pos_in_child[new_item] = k
                pos_in_child[old_item] = pos_new
        return child
    
    # Population
    pop_size = min(40, max(12, n // 8))
    
    population = []
    fitnesses = []
    
    # FFD
    ffd_p = ffd_perm()
    population.append(ffd_p)
    fitnesses.append(eval_fn(ffd_p))
    
    # Near-FFD variants
    num_near = min(8, pop_size - 1)
    for k in range(num_near):
        p = ffd_p[:]
        num_swaps = max(1, n // (20 - k * 2))
        for _ in range(num_swaps):
            i = _randint(0, n - 1)
            d = _randint(1, min(5 + k, n - 1))
            j = min(n - 1, max(0, i + (_randint(0, 1) * 2 - 1) * d))
            p[i], p[j] = p[j], p[i]
        population.append(p)
        fitnesses.append(eval_fn(p))
    
    # Jittered
    num_jit = min(6, pop_size - len(population))
    for k in range(num_jit):
        jf = 0.01 + 0.12 * k / max(1, num_jit - 1)
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
    
    # FDO main loop
    generations_no_improve = 0
    main_cutoff = time_limit * 0.92
    
    iteration = 0
    while True:
        if iteration % 3 == 0:
            if _time() - start_time >= main_cutoff:
                break
        
        # Find guide (best) and stats
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
        fitness_range = max_fit - min_fit
        if fitness_range < 1e-12:
            fitness_range = 1.0
        
        # Elitism: top 2
        sorted_by_fit = sorted(range(pop_size), key=lambda x: fitnesses[x])
        elite_indices = set(sorted_by_fit[:2])
        
        # Pick second-best as alternative guide
        guide2 = population[sorted_by_fit[1]] if pop_size > 1 else guide
        
        improved_this_gen = False
        
        for i in range(pop_size):
            if i in elite_indices:
                continue
            
            # FDO weight: fitness-dependent factor
            w_i = (fitnesses[i] - min_fit) / fitness_range
            
            # Stagnation boost
            if generations_no_improve > 100:
                w_i = min(1.0, w_i + 0.1 * (generations_no_improve / 500.0))
            
            new_perm = population[i][:]
            
            if w_i < 0.2:
                # STRONG EXPLOITATION: very small perturbations
                num_moves = _randint(1, 2)
                for _ in range(num_moves):
                    r = _random()
                    if r < 0.35:
                        pos = _randint(0, n - 1)
                        new_pos = max(0, min(n - 1, pos + _randint(-3, 3)))
                        if pos != new_pos:
                            item = new_perm.pop(pos)
                            new_perm.insert(new_pos, item)
                    elif r < 0.65:
                        a = _randint(0, n - 1)
                        d = _randint(1, min(3, n - 1))
                        b = min(n - 1, max(0, a + (_randint(0, 1) * 2 - 1) * d))
                        new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
                    else:
                        if n >= 3:
                            seg_len = _randint(2, min(3, n))
                            a = _randint(0, n - seg_len)
                            seg = new_perm[a:a + seg_len]
                            del new_perm[a:a + seg_len]
                            b = _randint(0, len(new_perm))
                            for k, item in enumerate(seg):
                                new_perm.insert(b + k, item)
            
            elif w_i < 0.5:
                # CROSSOVER with guide (exploitation-leaning)
                if _random() < 0.6:
                    new_perm = ox_crossover(guide, new_perm)
                else:
                    new_perm = pmx_crossover(guide, new_perm)
                
                # Small perturbation
                if _random() < 0.4:
                    a = _randint(0, n - 1)
                    d = _randint(1, min(5, n - 1))
                    b = min(n - 1, max(0, a + d))
                    new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
            
            elif w_i < 0.75:
                # CROSSOVER with guide (exploration-leaning)
                g = guide if _random() < 0.7 else guide2
                if _random() < 0.5:
                    new_perm = ox_crossover(g, new_perm)
                else:
                    new_perm = ox_crossover(new_perm, g)
                
                # Medium perturbation
                num_swaps = _randint(1, max(1, n // 20))
                for _ in range(num_swaps):
                    a = _randint(0, n - 1)
                    b = _randint(0, n - 1)
                    new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
            
            else:
                # EXPLORATION: large diversification
                r = _random()
                if r < 0.3:
                    new_perm = jittered_perm(0.02 + _random() * 0.2)
                elif r < 0.55:
                    # OX crossover with guide but heavy mutation
                    new_perm = ox_crossover(guide, random_perm())
                elif r < 0.8:
                    # Random with weight-biased head
                    new_perm = random_perm()
                    portion = max(1, int(n * (0.2 + _random() * 0.3)))
                    head = new_perm[:portion]
                    head.sort(key=lambda x: -w_arr[x])
                    new_perm[:portion] = head
                else:
                    new_perm = random_perm()
            
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
        if best_fitness < max_fit_now and worst_now not in elite_indices:
            population[worst_now] = best_perm[:]
            fitnesses[worst_now] = best_fitness
        
        # Diversity injection on stagnation
        if generations_no_improve > 0 and generations_no_improve % 100 == 0:
            sorted_idx = sorted(range(pop_size), key=lambda x: fitnesses[x])
            num_replace = max(2, int(pop_size * 0.5))
            for k in range(num_replace):
                ri = sorted_idx[-(k + 1)]
                if ri in elite_indices:
                    continue
                r = _random()
                if r < 0.3:
                    p = jittered_perm(0.01 + _random() * 0.15)
                elif r < 0.6:
                    # OX with best + jittered
                    p = ox_crossover(best_perm, jittered_perm(0.05 + _random() * 0.1))
                else:
                    p = random_perm()
                population[ri] = p
                fitnesses[ri] = eval_fn(p)
        
        iteration += 1
    
    # Final local search phase on best permutation
    final_cutoff = time_limit * 0.995
    
    current = best_perm[:]
    current_fit = best_fitness
    
    # More targeted local search
    # First decode to identify underfilled bins
    def get_underfilled_items(perm):
        """Decode and find items in underfilled bins for targeted mutations."""
        bins_c, bw_c = decode_bf(perm)
        # Find bins with significant waste
        underfilled = []
        for b_idx, bw in enumerate(bw_c):
            if bw < C * 0.85:
                underfilled.extend(bins_c[b_idx])
        return underfilled
    
    no_improve_ls = 0
    step = 0
    temp = 0.001  # Very small temperature for SA-like acceptance
    
    while _time() - start_time < final_cutoff:
        p = current[:]
        step += 1
        
        move = step % 7
        
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
            # Reverse small segment
            a = _randint(0, n - 1)
            seg_len = _randint(2, min(5, n))
            b = min(n, a + seg_len)
            p[a:b] = p[a:b][::-1]
        elif move == 4:
            # Double swap
            a = _randint(0, n - 1)
            b = _randint(0, n - 1)
            p[a], p[b] = p[b], p[a]
            c = _randint(0, n - 1)
            d = _randint(0, n - 1)
            p[c], p[d] = p[d], p[c]
        elif move == 5:
            # Move segment of 3
            if n >= 4:
                seg_len = min(3, n - 1)
                a = _randint(0, n - seg_len)
                seg = p[a:a + seg_len]
                del p[a:a + seg_len]
                b = _randint(0, len(p))
                for k, item in enumerate(seg):
                    p.insert(b + k, item)
        else:
            # Insert two items
            a = _randint(0, n - 1)
            item1 = p.pop(a)
            b = _randint(0, n - 2) if n > 2 else 0
            item2 = p.pop(b)
            c = _randint(0, len(p))
            p.insert(c, item1)
            d = _randint(0, len(p))
            p.insert(d, item2)
        
        # Evaluate with both decoders
        f1 = eval_bf_fast(p)
        f2 = eval_ff_fast(p)
        f = min(f1, f2)
        
        if f < current_fit:
            current_fit = f
            current = p[:]
            best_perm = p[:]
            best_fitness = f
            no_improve_ls = 0
        else:
            no_improve_ls += 1
    
    return make_packing_result(best_perm)
