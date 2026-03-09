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
    
    total_weight = sum(weights)
    C = bin_capacity
    w_arr = weights
    
    # Basic lower bound
    lower_bound = (total_weight + C - 1) // C
    
    # L2 lower bound (Martello-Toth style)
    sorted_weights = sorted(weights, reverse=True)
    half_C = C // 2
    
    # Compute L2 bound
    if n <= 20000:
        # Build sorted ascending for binary search
        sorted_asc = sorted(weights)
        
        for alpha in range(1, half_C + 1):
            threshold_large = C - alpha
            
            n_large = 0
            n_medium = 0
            sum_medium = 0
            sum_small = 0
            
            for w in sorted_weights:
                if w > threshold_large:
                    n_large += 1
                elif w > half_C:
                    n_medium += 1
                    sum_medium += w
                elif w >= alpha:
                    sum_small += w
                else:
                    break
            
            sum_medium_remaining = n_medium * C - sum_medium
            leftover = max(0, sum_small - sum_medium_remaining)
            extra_bins = (leftover + C - 1) // C if leftover > 0 else 0
            
            lb = n_large + n_medium + extra_bins
            if lb > lower_bound:
                lower_bound = lb
            
            if alpha > 20 and alpha % 2 != 0:
                continue
            if alpha > 100 and alpha % 5 != 0:
                continue
            if alpha > 500 and alpha % 20 != 0:
                continue
    
    # Best Fit Decreasing decode
    def decode_bf(perm):
        sorted_remaining = []  # (remaining_cap, bin_idx)
        bin_items = []
        
        for idx in perm:
            w = w_arr[idx]
            if w > C:
                continue
            if w == 0:
                if bin_items:
                    bin_items[0].append(idx)
                else:
                    bin_items.append([idx])
                    bisect.insort(sorted_remaining, (C, 0))
                continue
            
            pos = bisect.bisect_left(sorted_remaining, (w,))
            if pos < len(sorted_remaining):
                rem, b_idx = sorted_remaining[pos]
                del sorted_remaining[pos]
                new_rem = rem - w
                bin_items[b_idx].append(idx)
                if new_rem > 0:
                    bisect.insort(sorted_remaining, (new_rem, b_idx))
            else:
                b_idx = len(bin_items)
                bin_items.append([idx])
                new_rem = C - w
                if new_rem > 0:
                    bisect.insort(sorted_remaining, (new_rem, b_idx))
        
        return bin_items
    
    # Faster BFD decode that returns (num_bins, bin_items)
    def decode_bf_count(perm):
        sorted_remaining = []
        count = 0
        
        for idx in perm:
            w = w_arr[idx]
            if w <= 0 or w > C:
                continue
            
            pos = bisect.bisect_left(sorted_remaining, (w,))
            if pos < len(sorted_remaining):
                rem, b_idx = sorted_remaining[pos]
                del sorted_remaining[pos]
                new_rem = rem - w
                if new_rem > 0:
                    bisect.insort(sorted_remaining, (new_rem, b_idx))
            else:
                count += 1
                new_rem = C - w
                if new_rem > 0:
                    bisect.insort(sorted_remaining, (new_rem, count - 1))
        
        return count
    
    def evaluate(perm):
        return decode_bf_count(perm)
    
    def format_solution(packing):
        bin_weights = [sum(w_arr[i] for i in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}
    
    # FFD permutation
    ffd_perm = sorted(range(n), key=lambda i: -w_arr[i])
    
    best_packing = decode_bf(ffd_perm)
    best_fitness = len(best_packing)
    best_perm = list(ffd_perm)
    
    if best_fitness <= lower_bound:
        return format_solution(best_packing)
    
    # Weight classes
    large_items = [i for i in range(n) if w_arr[i] > C * 2 // 3]
    medium_items = [i for i in range(n) if C // 3 < w_arr[i] <= C * 2 // 3]
    small_items = [i for i in range(n) if w_arr[i] <= C // 3]
    
    def generate_weight_class_perm():
        l = list(large_items)
        m = list(medium_items)
        s = list(small_items)
        random.shuffle(l)
        random.shuffle(m)
        random.shuffle(s)
        return l + m + s
    
    def generate_paired_perm():
        result = []
        used = set()
        l = list(large_items)
        random.shuffle(l)
        s_sorted = sorted(small_items, key=lambda i: -w_arr[i])
        s_avail = list(s_sorted)
        
        for li in l:
            result.append(li)
            used.add(li)
            remaining = C - w_arr[li]
            new_avail = []
            for si in s_avail:
                if si not in used and w_arr[si] <= remaining:
                    result.append(si)
                    used.add(si)
                    remaining -= w_arr[si]
                elif si not in used:
                    new_avail.append(si)
            s_avail = new_avail
        
        m = list(medium_items)
        random.shuffle(m)
        for mi in m:
            if mi not in used:
                result.append(mi)
                used.add(mi)
        
        for si in s_avail:
            if si not in used:
                result.append(si)
                used.add(si)
        
        for i in range(n):
            if i not in used:
                result.append(i)
        
        return result
    
    # Population size
    if n < 50:
        pop_size = 80
    elif n < 200:
        pop_size = 60
    elif n < 500:
        pop_size = 40
    else:
        pop_size = 30
    
    population = []
    fitness = []
    
    population.append(list(ffd_perm))
    fitness.append(best_fitness)
    
    # Generate initial population with diverse strategies
    init_strategies = [
        lambda: list(reversed(ffd_perm)),
        generate_weight_class_perm,
        generate_paired_perm,
    ]
    
    for strat in init_strategies:
        if time.time() - start_time > time_limit * 0.15:
            break
        perm = strat()
        f = evaluate(perm)
        population.append(perm)
        fitness.append(f)
        if f < best_fitness:
            best_fitness = f
            best_perm = list(perm)
            if best_fitness <= lower_bound:
                return format_solution(decode_bf(best_perm))
    
    while len(population) < pop_size:
        if time.time() - start_time > time_limit * 0.15:
            break
        
        r = random.random()
        if r < 0.3:
            perm = list(ffd_perm)
            num_swaps = random.randint(1, max(1, n // 6))
            for _ in range(num_swaps):
                a = random.randrange(n)
                delta = random.randint(-max(5, n // 10), max(5, n // 10))
                b = min(n - 1, max(0, a + delta))
                perm[a], perm[b] = perm[b], perm[a]
        elif r < 0.5:
            perm = generate_weight_class_perm()
        elif r < 0.7:
            perm = generate_paired_perm()
        else:
            perm = list(range(n))
            random.shuffle(perm)
        
        f = evaluate(perm)
        population.append(perm)
        fitness.append(f)
        if f < best_fitness:
            best_fitness = f
            best_perm = list(perm)
            if best_fitness <= lower_bound:
                return format_solution(decode_bf(best_perm))
    
    pop_size = len(population)
    
    # BBX crossover: take fullest bins from parent, remaining in other parent order
    def bbx_crossover(p1, p2):
        bins1 = decode_bf(p1)
        scored = []
        for b in bins1:
            bw = sum(w_arr[i] for i in b)
            scored.append((bw, b))
        scored.sort(key=lambda x: -x[0])
        
        threshold = int(C * 0.75)
        good_bins = [b for bw, b in scored if bw >= threshold]
        
        if len(good_bins) < max(1, len(scored) // 4):
            count = max(1, len(scored) // 3)
            good_bins = [b for bw, b in scored[:count]]
        
        if len(good_bins) > 1:
            k = random.randint(max(1, len(good_bins) // 2), len(good_bins))
            selected = random.sample(good_bins, k)
        else:
            selected = good_bins
        
        placed = set()
        prefix = []
        for b in selected:
            for item in b:
                prefix.append(item)
                placed.add(item)
        
        suffix = [item for item in p2 if item not in placed]
        return prefix + suffix
    
    # BBX hybrid: take best non-conflicting bins from both parents
    def bbx_hybrid(p1, p2):
        bins1 = decode_bf(p1)
        bins2 = decode_bf(p2)
        
        all_bins = []
        for b in bins1:
            bw = sum(w_arr[i] for i in b)
            all_bins.append((bw, frozenset(b), b))
        for b in bins2:
            bw = sum(w_arr[i] for i in b)
            all_bins.append((bw, frozenset(b), b))
        
        all_bins.sort(key=lambda x: -x[0])
        
        placed = set()
        prefix = []
        
        for bw, bset, b in all_bins:
            if bset.isdisjoint(placed):
                prefix.extend(b)
                placed.update(bset)
        
        remaining = [i for i in ffd_perm if i not in placed]
        return prefix + remaining
    
    # Enhanced BBX: take best bins, then randomly order remaining
    def bbx_enhanced(p1, p2):
        bins1 = decode_bf(p1)
        bins2 = decode_bf(p2)
        
        all_bins = []
        for b in bins1:
            bw = sum(w_arr[i] for i in b)
            # Prioritize fuller bins and full bins extra
            score = bw + (C * 10 if bw == C else 0)
            all_bins.append((score, frozenset(b), b))
        for b in bins2:
            bw = sum(w_arr[i] for i in b)
            score = bw + (C * 10 if bw == C else 0)
            all_bins.append((score, frozenset(b), b))
        
        all_bins.sort(key=lambda x: -x[0])
        
        placed = set()
        prefix = []
        
        for score, bset, b in all_bins:
            if bset.isdisjoint(placed):
                prefix.extend(b)
                placed.update(bset)
        
        # Remaining items sorted by weight descending with slight randomization
        remaining = [(w_arr[i] + random.random() * 0.1, i) for i in range(n) if i not in placed]
        remaining.sort(key=lambda x: -x[0])
        suffix = [i for _, i in remaining]
        return prefix + suffix
    
    # Order Crossover
    def order_crossover(p1, p2):
        size = len(p1)
        if size <= 1:
            return list(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b+1] = p1[a:b+1]
        segment_set = set(child[a:b+1])
        pos = (b + 1) % size
        for item in p2:
            if item not in segment_set:
                child[pos] = item
                pos = (pos + 1) % size
        return child
    
    # Mutations
    def mutate_swap(perm, num_swaps=1):
        perm = list(perm)
        for _ in range(num_swaps):
            a, b = random.randrange(n), random.randrange(n)
            perm[a], perm[b] = perm[b], perm[a]
        return perm
    
    def mutate_inversion(perm):
        perm = list(perm)
        if n <= 1:
            return perm
        a, b = sorted(random.sample(range(n), 2))
        perm[a:b+1] = reversed(perm[a:b+1])
        return perm
    
    def mutate_scramble(perm):
        perm = list(perm)
        if n <= 1:
            return perm
        a, b = sorted(random.sample(range(n), 2))
        sub = perm[a:b+1]
        random.shuffle(sub)
        perm[a:b+1] = sub
        return perm
    
    def mutate_block_move(perm):
        perm = list(perm)
        if n <= 2:
            return perm
        a = random.randrange(n)
        length = random.randint(1, max(1, min(n // 5, n - a)))
        b = min(n, a + length)
        block = perm[a:b]
        del perm[a:b]
        insert_pos = random.randrange(len(perm) + 1)
        perm[insert_pos:insert_pos] = block
        return perm
    
    def mutate_bin_shake(perm):
        """Take items from worst bins and reinsert."""
        bins_decoded = decode_bf(perm)
        if len(bins_decoded) <= 1:
            return list(perm)
        
        scored = [(sum(w_arr[i] for i in b), idx) for idx, b in enumerate(bins_decoded)]
        scored.sort()
        
        num_worst = max(1, len(scored) // 5)
        shake_items = set()
        for _, bidx in scored[:num_worst]:
            for item in bins_decoded[bidx]:
                shake_items.add(item)
        
        kept = [item for item in perm if item not in shake_items]
        shaken = sorted(shake_items, key=lambda i: -w_arr[i])
        # Add some randomness
        for i in range(len(shaken) - 1):
            if random.random() < 0.3:
                j = random.randint(i, len(shaken) - 1)
                shaken[i], shaken[j] = shaken[j], shaken[i]
        
        # Insert shaken items at positions that might help
        result = list(kept)
        for item in shaken:
            pos = random.randrange(len(result) + 1)
            result.insert(pos, item)
        
        return result
    
    def mutate_bin_repack(perm):
        """Take items from worst bins, sort by weight, prepend to FFD-ordered remaining."""
        bins_decoded = decode_bf(perm)
        if len(bins_decoded) <= 1:
            return list(perm)
        
        scored = [(sum(w_arr[i] for i in b), idx) for idx, b in enumerate(bins_decoded)]
        scored.sort()
        
        # Take worst 1-3 bins
        num_worst = random.randint(1, max(1, min(3, len(scored) - 1)))
        shake_items = set()
        for _, bidx in scored[:num_worst]:
            for item in bins_decoded[bidx]:
                shake_items.add(item)
        
        # Keep good bins' items in order, append shaken items sorted by weight desc
        kept = [item for item in perm if item not in shake_items]
        shaken = sorted(shake_items, key=lambda i: -w_arr[i])
        
        # Try inserting at beginning (so they get packed first with BFD)
        return shaken + kept
    
    # Tournament selection
    def tournament_select(k=3):
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best_c = min(candidates, key=lambda c: fitness[c])
        return best_c
    
    # GA parameters
    elite_count = max(2, pop_size // 6)
    stagnation_counter = 0
    last_best = best_fitness
    
    generation = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.97:
            break
        
        if best_fitness <= lower_bound:
            break
        
        generation += 1
        
        # Sort population
        paired = sorted(zip(fitness, population), key=lambda x: x[0])
        fitness = [p[0] for p in paired]
        population = [p[1] for p in paired]
        
        # Track stagnation
        if fitness[0] < last_best:
            last_best = fitness[0]
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Adaptive parameters
        unique_f = len(set(fitness))
        diversity = unique_f / max(1, len(fitness))
        
        high_stagnation = stagnation_counter > 15
        
        if high_stagnation or diversity < 0.15:
            mutation_rate = 0.8
            swap_max = max(2, n // 3)
            tournament_k = 2
        elif diversity < 0.3:
            mutation_rate = 0.6
            swap_max = max(1, n // 6)
            tournament_k = 3
        else:
            mutation_rate = 0.35
            swap_max = max(1, n // 10)
            tournament_k = 4
        
        new_population = []
        new_fitness = []
        
        # Elitism
        for i in range(min(elite_count, len(population))):
            new_population.append(list(population[i]))
            new_fitness.append(fitness[i])
        
        # Restart injection on high stagnation
        inject = 0
        if stagnation_counter > 0 and stagnation_counter % 20 == 0:
            inject = max(3, pop_size // 4)
            stagnation_counter = 0
        
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit * 0.97:
                break
            
            if inject > 0:
                inject -= 1
                r = random.random()
                if r < 0.25:
                    child = generate_weight_class_perm()
                elif r < 0.5:
                    child = generate_paired_perm()
                elif r < 0.75:
                    # Perturbed best
                    child = list(best_perm)
                    num_swaps = random.randint(n // 4, max(n // 4 + 1, n // 2))
                    for _ in range(num_swaps):
                        a, b = random.randrange(n), random.randrange(n)
                        child[a], child[b] = child[b], child[a]
                else:
                    child = list(ffd_perm)
                    num_swaps = random.randint(n // 5, max(n // 5 + 1, n // 2))
                    for _ in range(num_swaps):
                        a, b = random.randrange(n), random.randrange(n)
                        child[a], child[b] = child[b], child[a]
                f = evaluate(child)
                new_population.append(child)
                new_fitness.append(f)
                if f < best_fitness:
                    best_fitness = f
                    best_perm = list(child)
                    if best_fitness <= lower_bound:
                        break
                continue
            
            p1_idx = tournament_select(tournament_k)
            p2_idx = tournament_select(tournament_k)
            while p2_idx == p1_idx and len(population) > 1:
                p2_idx = tournament_select(tournament_k)
            
            # Crossover selection
            r = random.random()
            if r < 0.30:
                child = bbx_hybrid(population[p1_idx], population[p2_idx])
            elif r < 0.55:
                child = bbx_enhanced(population[p1_idx], population[p2_idx])
            elif r < 0.75:
                child = bbx_crossover(population[p1_idx], population[p2_idx])
            elif r < 0.88:
                child = order_crossover(population[p1_idx], population[p2_idx])
            else:
                child = list(population[p1_idx])
            
            # Mutation
            if random.random() < mutation_rate:
                r = random.random()
                if r < 0.15:
                    child = mutate_swap(child, random.randint(1, swap_max))
                elif r < 0.30:
                    child = mutate_inversion(child)
                elif r < 0.45:
                    child = mutate_block_move(child)
                elif r < 0.60:
                    child = mutate_bin_shake(child)
                elif r < 0.80:
                    child = mutate_bin_repack(child)
                else:
                    child = mutate_scramble(child)
            
            f = evaluate(child)
            new_population.append(child)
            new_fitness.append(f)
            
            if f < best_fitness:
                best_fitness = f
                best_perm = list(child)
                if best_fitness <= lower_bound:
                    break
        
        population = new_population
        fitness = new_fitness
        
        if best_fitness <= lower_bound:
            break
    
    return format_solution(decode_bf(best_perm))