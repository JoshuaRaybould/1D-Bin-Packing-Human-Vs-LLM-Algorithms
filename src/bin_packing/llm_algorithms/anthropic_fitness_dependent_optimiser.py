# anthropic
# fitness_dependent_optimiser_2_performance_4.py

import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    if n == 1:
        return {"packing": [[0]], "bin_weights": [weights[0]]}
    
    w = weights
    C = bin_capacity
    total_weight = sum(w)
    lower_bound = (total_weight + C - 1) // C
    
    # Enhanced lower bound
    half_C = C / 2.0
    count_large = sum(1 for wi in w if wi > half_C)
    lower_bound = max(lower_bound, count_large)
    
    # Fast BFD decode using bisect
    def decode_bfd(perm):
        bin_items = []
        bin_loads = []
        sorted_rem = []  # remaining capacities, sorted ascending
        sorted_bidx = []  # corresponding bin indices
        
        for idx in perm:
            wi = w[idx]
            if wi <= 0:
                continue
            pos = bisect.bisect_left(sorted_rem, wi)
            if pos < len(sorted_rem):
                rem = sorted_rem.pop(pos)
                b = sorted_bidx.pop(pos)
                new_rem = rem - wi
                bin_items[b].append(idx)
                bin_loads[b] += wi
                if new_rem > 0:
                    new_pos = bisect.bisect_left(sorted_rem, new_rem)
                    sorted_rem.insert(new_pos, new_rem)
                    sorted_bidx.insert(new_pos, b)
            else:
                b = len(bin_items)
                bin_items.append([idx])
                bin_loads.append(wi)
                new_rem = C - wi
                if new_rem > 0:
                    new_pos = bisect.bisect_left(sorted_rem, new_rem)
                    sorted_rem.insert(new_pos, new_rem)
                    sorted_bidx.insert(new_pos, b)
        
        return bin_items, bin_loads
    
    # Also FFD decode for comparison
    def decode_ffd(perm):
        bin_loads = []
        bin_items = []
        for idx in perm:
            wi = w[idx]
            if wi <= 0:
                continue
            placed = False
            for b in range(len(bin_loads)):
                if bin_loads[b] + wi <= C:
                    bin_items[b].append(idx)
                    bin_loads[b] += wi
                    placed = True
                    break
            if not placed:
                bin_items.append([idx])
                bin_loads.append(wi)
        return bin_items, bin_loads
    
    C3 = C * C * C
    
    def calc_fitness(bin_loads, num_bins):
        if num_bins == 0:
            return 0.0
        fill_score = 0
        for l in bin_loads:
            fill_score += l * l * l
        return num_bins - fill_score / (C3 * num_bins)
    
    def decode_best(perm):
        b1, l1 = decode_bfd(perm)
        nb1 = len(b1)
        # Only try FFD if instance is small enough
        if n <= 500:
            b2, l2 = decode_ffd(perm)
            nb2 = len(b2)
            if nb2 < nb1:
                return b2, l2
            elif nb2 == nb1:
                s1 = sum(x*x*x for x in l1)
                s2 = sum(x*x*x for x in l2)
                if s2 > s1:
                    return b2, l2
        return b1, l1
    
    def improve_packing(bins, bin_loads, max_attempts=8):
        for attempt in range(max_attempts):
            num_bins = len(bins)
            if num_bins <= lower_bound:
                break
            
            # Sort bins by load to find candidates to empty
            sorted_by_load = sorted(range(num_bins), key=lambda b: bin_loads[b])
            emptied = False
            
            for try_idx in range(min(3, num_bins)):
                min_b = sorted_by_load[try_idx]
                items_to_move = bins[min_b][:]
                
                # Build sorted remaining capacities for other bins
                other_rem = []
                other_idx = []
                for b in range(num_bins):
                    if b == min_b:
                        continue
                    rem = C - bin_loads[b]
                    if rem > 0:
                        pos = bisect.bisect_left(other_rem, rem)
                        other_rem.insert(pos, rem)
                        other_idx.insert(pos, b)
                
                can_move_all = True
                moves = []
                temp_rem = other_rem[:]
                temp_idx = other_idx[:]
                
                for item_idx in sorted(items_to_move, key=lambda x: -w[x]):
                    wi = w[item_idx]
                    pos = bisect.bisect_left(temp_rem, wi)
                    if pos < len(temp_rem):
                        rem = temp_rem.pop(pos)
                        target = temp_idx.pop(pos)
                        new_rem = rem - wi
                        moves.append((item_idx, target))
                        if new_rem > 0:
                            new_pos = bisect.bisect_left(temp_rem, new_rem)
                            temp_rem.insert(new_pos, new_rem)
                            temp_idx.insert(new_pos, target)
                    else:
                        can_move_all = False
                        break
                
                if can_move_all:
                    for item_idx, target_b in moves:
                        bins[target_b].append(item_idx)
                        bin_loads[target_b] += w[item_idx]
                    bins.pop(min_b)
                    bin_loads.pop(min_b)
                    emptied = True
                    break
            
            if not emptied:
                break
        
        return bins, bin_loads
    
    def evaluate_and_improve(perm):
        bins, loads = decode_best(perm)
        bins, loads = improve_packing(bins, loads)
        num_bins = len(bins)
        f = calc_fitness(loads, num_bins)
        return f, bins, loads
    
    # Generators
    def gen_ffd():
        return sorted(range(n), key=lambda i: -w[i])
    
    def gen_random():
        perm = list(range(n))
        random.shuffle(perm)
        return perm
    
    def gen_semi_random():
        perm = sorted(range(n), key=lambda i: -w[i])
        num_swaps = random.randint(1, max(1, n // 4))
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def gen_perturbed(base, num_swaps):
        perm = base[:]
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def gen_ffd_perturbed():
        perm = sorted(range(n), key=lambda i: -w[i])
        num_reversals = random.randint(2, 6)
        for _ in range(num_reversals):
            if n < 3:
                break
            seg_len = random.randint(2, max(2, n // 4))
            start = random.randint(0, n - seg_len)
            perm[start:start+seg_len] = reversed(perm[start:start+seg_len])
        return perm
    
    def gen_group():
        items = sorted(range(n), key=lambda i: -w[i])
        result = []
        used = [False] * n
        for idx in items:
            if used[idx]:
                continue
            used[idx] = True
            result.append(idx)
            remaining = C - w[idx]
            for idx2 in items:
                if not used[idx2] and w[idx2] <= remaining:
                    used[idx2] = True
                    result.append(idx2)
                    remaining -= w[idx2]
                    if remaining == 0:
                        break
        return result
    
    def gen_group_random():
        """Group perm with randomized item selection."""
        items = sorted(range(n), key=lambda i: -w[i])
        result = []
        used = [False] * n
        for idx in items:
            if used[idx]:
                continue
            used[idx] = True
            result.append(idx)
            remaining = C - w[idx]
            # Collect candidates
            candidates = []
            for idx2 in items:
                if not used[idx2] and w[idx2] <= remaining:
                    candidates.append(idx2)
            # Randomly choose among good fits
            random.shuffle(candidates)
            for idx2 in candidates:
                if not used[idx2] and w[idx2] <= remaining:
                    used[idx2] = True
                    result.append(idx2)
                    remaining -= w[idx2]
                    if remaining == 0:
                        break
        return result
    
    def gen_weight_class():
        """Sort by weight with random perturbation within weight classes."""
        # Group items by weight ranges
        perm = sorted(range(n), key=lambda i: (-w[i], random.random()))
        return perm
    
    # Population
    pop_size = min(30, max(8, n // 3))
    
    population = []
    fitnesses = []
    pop_bins = []
    pop_loads = []
    
    def add_agent(perm):
        f, b, bl = evaluate_and_improve(perm)
        population.append(perm)
        fitnesses.append(f)
        pop_bins.append(b)
        pop_loads.append(bl)
        return f
    
    # Initialize population
    add_agent(gen_ffd())
    if time.time() - start_time < time_limit * 0.4:
        add_agent(gen_group())
    
    for _ in range(min(6, pop_size) - len(population)):
        if time.time() - start_time > time_limit * 0.4:
            break
        if random.random() < 0.5:
            add_agent(gen_ffd_perturbed())
        else:
            add_agent(gen_group_random())
    
    while len(population) < pop_size:
        if time.time() - start_time > time_limit * 0.4:
            break
        r = random.random()
        if r < 0.3:
            add_agent(gen_semi_random())
        elif r < 0.6:
            add_agent(gen_weight_class())
        elif r < 0.8:
            add_agent(gen_ffd_perturbed())
        else:
            add_agent(gen_random())
    
    pop_size = len(population)
    if pop_size == 0:
        # Fallback
        f, b, bl = evaluate_and_improve(gen_ffd())
        return {"packing": b, "bin_weights": bl}
    
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_fitness = fitnesses[best_idx]
    best_perm = population[best_idx][:]
    best_bins = [b[:] for b in pop_bins[best_idx]]
    best_loads = pop_loads[best_idx][:]
    
    if len(best_bins) <= lower_bound:
        return {"packing": best_bins, "bin_weights": best_loads}
    
    iteration = 0
    last_improvement = 0
    stagnation_limit = 40
    
    time_frac = 0.98
    
    while time.time() - start_time < time_limit * time_frac:
        iteration += 1
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_limit)
        
        ci = min(range(pop_size), key=lambda i: fitnesses[i])
        cb_perm = population[ci]
        cb_fit = fitnesses[ci]
        
        worst_fit = max(fitnesses)
        fit_range = worst_fit - cb_fit if worst_fit != cb_fit else 1.0
        
        for i in range(pop_size):
            if time.time() - start_time > time_limit * time_frac:
                break
            if i == ci:
                continue
            
            fi = fitnesses[i]
            normalized = (fi - cb_fit) / fit_range if fit_range > 0 else 0.0
            
            weight = fi / cb_fit if cb_fit > 0 else 1.0 + normalized
            
            new_perm = population[i][:]
            r = random.random()
            threshold = 1.0 + (0.5 + progress) * r
            
            if weight > threshold:  # Scout - explore
                rv = random.random()
                if rv < 0.2 and n >= 3:
                    # Segment reversal
                    seg_len = max(2, int(n * (0.1 + normalized * 0.3)))
                    seg_len = min(seg_len, n)
                    sp = random.randint(0, n - seg_len)
                    new_perm[sp:sp+seg_len] = reversed(new_perm[sp:sp+seg_len])
                elif rv < 0.4 and n >= 3:
                    # Block insertion (or-opt)
                    bl = max(1, int(n * (0.05 + normalized * 0.15)))
                    bl = min(bl, n - 1)
                    sp = random.randint(0, n - bl)
                    block = new_perm[sp:sp+bl]
                    del new_perm[sp:sp+bl]
                    ip = random.randint(0, len(new_perm))
                    new_perm[ip:ip] = block
                elif rv < 0.6:
                    # Random swaps
                    ns = max(1, int(n * normalized * 0.4))
                    for _ in range(ns):
                        a = random.randint(0, n - 1)
                        b2 = random.randint(0, n - 1)
                        new_perm[a], new_perm[b2] = new_perm[b2], new_perm[a]
                elif rv < 0.75:
                    new_perm = gen_ffd_perturbed()
                elif rv < 0.85:
                    new_perm = gen_group_random()
                elif rv < 0.95:
                    new_perm = gen_semi_random()
                else:
                    new_perm = gen_weight_class()
            else:  # Forager - exploit
                rv = random.random()
                if rv < 0.45:
                    # OX crossover with current best
                    seg_len = max(1, int(n * max(0.1, min(0.6, normalized * 0.7))))
                    sp = random.randint(0, n - seg_len)
                    ep = sp + seg_len
                    segment_set = set(cb_perm[sp:ep])
                    result = [None] * n
                    for k in range(sp, ep):
                        result[k] = cb_perm[k]
                    remaining = [x for x in new_perm if x not in segment_set]
                    ri = 0
                    for j in range(n):
                        if result[j] is None:
                            result[j] = remaining[ri]
                            ri += 1
                    new_perm = result
                elif rv < 0.7:
                    # OX with global best
                    seg_len = max(1, int(n * max(0.1, min(0.6, normalized * 0.7))))
                    sp = random.randint(0, n - seg_len)
                    ep = sp + seg_len
                    segment_set = set(best_perm[sp:ep])
                    result = [None] * n
                    for k in range(sp, ep):
                        result[k] = best_perm[k]
                    remaining = [x for x in new_perm if x not in segment_set]
                    ri = 0
                    for j in range(n):
                        if result[j] is None:
                            result[j] = remaining[ri]
                            ri += 1
                    new_perm = result
                elif rv < 0.85:
                    # Position swap toward best
                    num_fixes = max(1, int(n * 0.2))
                    pos_map = {v: k for k, v in enumerate(new_perm)}
                    target = best_perm if random.random() < 0.5 else cb_perm
                    for _ in range(num_fixes):
                        tp = random.randint(0, n - 1)
                        tv = target[tp]
                        cp = pos_map[tv]
                        if cp != tp:
                            ov = new_perm[tp]
                            new_perm[tp], new_perm[cp] = new_perm[cp], new_perm[tp]
                            pos_map[tv] = tp
                            pos_map[ov] = cp
                else:
                    # Partially mapped crossover style
                    seg_len = max(1, int(n * 0.3))
                    sp = random.randint(0, n - seg_len)
                    ep = sp + seg_len
                    target = best_perm if random.random() < 0.6 else cb_perm
                    segment_set = set(target[sp:ep])
                    result = [None] * n
                    for k in range(sp, ep):
                        result[k] = target[k]
                    remaining = [x for x in new_perm if x not in segment_set]
                    ri = 0
                    for j in range(n):
                        if result[j] is None:
                            result[j] = remaining[ri]
                            ri += 1
                    new_perm = result
                
                # Small perturbation
                nm = random.randint(1, max(1, min(3, n // 10)))
                for _ in range(nm):
                    if n >= 2:
                        pos = random.randint(0, n - 2)
                        new_perm[pos], new_perm[pos+1] = new_perm[pos+1], new_perm[pos]
            
            new_f, new_b, new_bl = evaluate_and_improve(new_perm)
            
            if new_f < fitnesses[i]:
                population[i] = new_perm
                fitnesses[i] = new_f
                pop_bins[i] = new_b
                pop_loads[i] = new_bl
                
                if new_f < best_fitness:
                    best_fitness = new_f
                    best_perm = new_perm[:]
                    best_bins = [bb[:] for bb in new_b]
                    best_loads = new_bl[:]
                    last_improvement = iteration
                    if len(best_bins) <= lower_bound:
                        return {"packing": best_bins, "bin_weights": best_loads}
            else:
                ap = 0.04 * (1.0 - progress)
                if random.random() < ap:
                    population[i] = new_perm
                    fitnesses[i] = new_f
                    pop_bins[i] = new_b
                    pop_loads[i] = new_bl
        
        # Stagnation handling
        if iteration - last_improvement > stagnation_limit:
            last_improvement = iteration
            sorted_idx = sorted(range(pop_size), key=lambda x: fitnesses[x])
            num_reinit = max(1, int(pop_size * 0.5))
            reinit_idx = sorted_idx[-num_reinit:]
            
            for idx in reinit_idx:
                if time.time() - start_time > time_limit * time_frac:
                    break
                r = random.random()
                if r < 0.3:
                    perm = gen_perturbed(best_perm, random.randint(2, min(30, n)))
                elif r < 0.5:
                    perm = gen_ffd_perturbed()
                elif r < 0.65:
                    perm = gen_group()
                    ns = random.randint(1, max(1, n // 6))
                    for _ in range(ns):
                        a = random.randint(0, n-1)
                        b2 = random.randint(0, n-1)
                        perm[a], perm[b2] = perm[b2], perm[a]
                elif r < 0.8:
                    perm = gen_group_random()
                elif r < 0.9:
                    perm = gen_semi_random()
                else:
                    perm = gen_weight_class()
                
                f, b, bl = evaluate_and_improve(perm)
                population[idx] = perm
                fitnesses[idx] = f
                pop_bins[idx] = b
                pop_loads[idx] = bl
                
                if f < best_fitness:
                    best_fitness = f
                    best_perm = perm[:]
                    best_bins = [bb[:] for bb in b]
                    best_loads = bl[:]
                    if len(best_bins) <= lower_bound:
                        return {"packing": best_bins, "bin_weights": best_loads}
            
            # Ensure best is in population
            worst_in_pop = max(range(pop_size), key=lambda x: fitnesses[x])
            if fitnesses[worst_in_pop] > best_fitness:
                perm = best_perm[:]
                f, b, bl = evaluate_and_improve(perm)
                population[worst_in_pop] = perm
                fitnesses[worst_in_pop] = f
                pop_bins[worst_in_pop] = b
                pop_loads[worst_in_pop] = bl
    
    return {"packing": best_bins, "bin_weights": best_loads}