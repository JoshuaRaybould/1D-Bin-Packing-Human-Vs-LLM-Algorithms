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
    
    # Basic lower bound
    lower_bound = (total_weight + C - 1) // C
    
    # L2 lower bound (Martello-Toth style)
    sorted_weights = sorted(weights, reverse=True)
    half_C = C // 2
    
    if n <= 10000:
        # Precompute prefix info for L2
        # Sort weights descending, compute cumulative for categories
        for alpha in range(1, half_C + 1):
            n_large = 0
            n_medium = 0
            sum_medium = 0
            sum_small_med = 0
            
            threshold_large = C - alpha
            
            for w in sorted_weights:
                if w > threshold_large:
                    n_large += 1
                elif w > half_C:
                    n_medium += 1
                    sum_medium += w
                elif w >= alpha:
                    sum_small_med += w
                else:
                    break  # sorted descending, rest are smaller
            
            sum_medium_remaining = n_medium * C - sum_medium
            leftover = max(0, sum_small_med - sum_medium_remaining)
            extra_bins = (leftover + C - 1) // C if leftover > 0 else 0
            
            lb = n_large + n_medium + extra_bins
            if lb > lower_bound:
                lower_bound = lb
            
            # Skip some alpha values for speed
            if alpha > 50 and alpha % 3 != 0:
                continue
            if alpha > 200 and alpha % 10 != 0:
                continue
    
    # Precompute weights array for fast access
    w_arr = weights
    
    # Best Fit Decreasing decode using bisect on sorted remaining capacities
    def decode_bf(perm):
        sorted_remaining = []  # list of (remaining, bin_idx), sorted by remaining
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
    
    # First Fit Decreasing decode
    def decode_ff(perm):
        bin_remaining = []
        bin_items = []
        
        for idx in perm:
            w = w_arr[idx]
            placed = False
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w:
                    bin_remaining[b] -= w
                    bin_items[b].append(idx)
                    placed = True
                    break
            if not placed:
                bin_items.append([idx])
                bin_remaining.append(C - w)
        
        return bin_items
    
    def evaluate(perm):
        return len(decode_bf(perm))
    
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
    
    # Population size
    if n < 50:
        pop_size = 60
    elif n < 200:
        pop_size = 50
    elif n < 500:
        pop_size = 40
    else:
        pop_size = 30
    
    population = []
    fitness = []
    
    population.append(list(ffd_perm))
    fitness.append(best_fitness)
    
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
        """Try to pair large items with small items that fill bins well."""
        result = []
        used = set()
        
        # Sort large items randomly
        l = list(large_items)
        random.shuffle(l)
        s_sorted = sorted(small_items, key=lambda i: -w_arr[i])
        s_avail = list(s_sorted)
        
        for li in l:
            result.append(li)
            used.add(li)
            remaining = C - w_arr[li]
            # Greedily pick small items
            new_avail = []
            for si in s_avail:
                if si not in used and w_arr[si] <= remaining:
                    result.append(si)
                    used.add(si)
                    remaining -= w_arr[si]
                    if remaining <= 0:
                        # Add rest to new_avail
                        pass
                    else:
                        continue
                if si not in used:
                    new_avail.append(si)
            s_avail = new_avail
        
        # Add medium items
        m = list(medium_items)
        random.shuffle(m)
        for mi in m:
            if mi not in used:
                result.append(mi)
                used.add(mi)
        
        # Add remaining small items
        for si in s_avail:
            if si not in used:
                result.append(si)
                used.add(si)
        
        # Add any missed items
        for i in range(n):
            if i not in used:
                result.append(i)
        
        return result
    
    # Generate initial population
    init_strategies = [
        lambda: list(ffd_perm),
        lambda: list(reversed(ffd_perm)),
        generate_weight_class_perm,
        generate_paired_perm,
    ]
    
    for strat in init_strategies:
        if time.time() - start_time > time_limit * 0.3:
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
        if time.time() - start_time > time_limit * 0.3:
            break
        
        r = random.random()
        if r < 0.3:
            perm = list(ffd_perm)
            num_swaps = random.randint(1, max(1, n // 8))
            for _ in range(num_swaps):
                a = random.randrange(n)
                delta = random.randint(-max(5, n // 15), max(5, n // 15))
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
    
    # BBX crossover: take fullest bins from parent1, remaining in parent2 order
    def bbx_crossover(p1, p2):
        bins1 = decode_bf(p1)
        
        # Score by fullness
        scored = []
        for b in bins1:
            bw = sum(w_arr[i] for i in b)
            scored.append((bw, b))
        
        scored.sort(key=lambda x: -x[0])
        
        # Take bins that are >= 80% full, or top fraction
        threshold = int(C * 0.8)
        good_bins = [b for bw, b in scored if bw >= threshold]
        
        if len(good_bins) < max(1, len(scored) // 4):
            count = max(1, len(scored) // 3)
            good_bins = [b for bw, b in scored[:count]]
        
        # Randomly select subset
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
    
    # Hybrid BBX: take best bins from BOTH parents
    def bbx_hybrid(p1, p2):
        bins1 = decode_bf(p1)
        bins2 = decode_bf(p2)
        
        all_bins = []
        for b in bins1:
            bw = sum(w_arr[i] for i in b)
            all_bins.append((bw, b, 1))
        for b in bins2:
            bw = sum(w_arr[i] for i in b)
            all_bins.append((bw, b, 2))
        
        all_bins.sort(key=lambda x: -x[0])
        
        placed = set()
        prefix = []
        
        for bw, b, src in all_bins:
            # Check if all items in this bin are still available
            if all(item not in placed for item in b):
                for item in b:
                    prefix.append(item)
                    placed.add(item)
        
        # Remaining items in FFD order (by weight descending)
        remaining = [i for i in ffd_perm if i not in placed]
        return prefix + remaining
    
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
        """Decode, find worst bins, shuffle their items into random positions."""
        bins_decoded = decode_bf(perm)
        if len(bins_decoded) <= 1:
            return list(perm)
        
        # Find bins with lowest weight
        scored = [(sum(w_arr[i] for i in b), idx) for idx, b in enumerate(bins_decoded)]
        scored.sort()
        
        # Take items from worst bins (bottom 20%)
        num_worst = max(1, len(scored) // 5)
        shake_items = set()
        for _, bidx in scored[:num_worst]:
            for item in bins_decoded[bidx]:
                shake_items.add(item)
        
        # Build new perm: items not in shake_items keep their relative order from perm,
        # then shake_items are shuffled and appended strategically
        kept = [item for item in perm if item not in shake_items]
        shaken = list(shake_items)
        random.shuffle(shaken)
        
        # Insert shaken items at random positions
        result = list(kept)
        for item in shaken:
            pos = random.randrange(len(result) + 1)
            result.insert(pos, item)
        
        return result
    
    # Tournament selection
    def tournament_select(k=3):
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best_c = min(candidates, key=lambda c: fitness[c])
        return best_c
    
    # GA parameters
    elite_count = max(2, pop_size // 8)
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
        
        high_stagnation = stagnation_counter > 20
        
        if high_stagnation or diversity < 0.2:
            mutation_rate = 0.7
            swap_max = max(2, n // 4)
            tournament_k = 2
        elif diversity < 0.4:
            mutation_rate = 0.5
            swap_max = max(1, n // 8)
            tournament_k = 3
        else:
            mutation_rate = 0.3
            swap_max = max(1, n // 10)
            tournament_k = 4
        
        new_population = []
        new_fitness = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(list(population[i]))
            new_fitness.append(fitness[i])
        
        # Restart injection on high stagnation
        inject = 0
        if stagnation_counter > 0 and stagnation_counter % 30 == 0:
            inject = max(2, pop_size // 5)
            stagnation_counter = 0
        
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit * 0.97:
                break
            
            if inject > 0:
                inject -= 1
                r = random.random()
                if r < 0.3:
                    child = generate_weight_class_perm()
                elif r < 0.6:
                    child = generate_paired_perm()
                else:
                    child = list(ffd_perm)
                    num_swaps = random.randint(n // 5, n // 2)
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
            
            # Crossover
            r = random.random()
            if r < 0.45:
                child = bbx_crossover(population[p1_idx], population[p2_idx])
            elif r < 0.70:
                child = bbx_hybrid(population[p1_idx], population[p2_idx])
            elif r < 0.85:
                child = order_crossover(population[p1_idx], population[p2_idx])
            else:
                child = list(population[p1_idx])
            
            # Mutation
            if random.random() < mutation_rate:
                r = random.random()
                if r < 0.25:
                    child = mutate_swap(child, random.randint(1, swap_max))
                elif r < 0.45:
                    child = mutate_inversion(child)
                elif r < 0.65:
                    child = mutate_block_move(child)
                elif r < 0.80:
                    child = mutate_bin_shake(child)
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