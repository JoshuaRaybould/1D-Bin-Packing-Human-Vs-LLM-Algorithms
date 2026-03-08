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
    
    # Basic lower bound
    lower_bound_basic = (total_weight + bin_capacity - 1) // bin_capacity
    
    # L2 lower bound (Martello-Toth style)
    def compute_l2_bound():
        C = bin_capacity
        best_lb = lower_bound_basic
        # Try several values of alpha
        half_C = C // 2
        # Precompute weight categories for various alpha
        sorted_w = sorted(weights, reverse=True)
        
        for alpha in range(1, half_C + 1):
            # Items with weight > C - alpha (large items, each needs ~own bin)
            n_large = 0
            sum_large_remaining = 0  # remaining space in bins with large items
            # Items with weight in (C/2, C - alpha] (medium items)
            n_medium = 0
            sum_medium = 0
            # Items with weight in [alpha, C/2] (small-medium items)
            n_small_med = 0
            sum_small_med = 0
            
            for w in sorted_w:
                if w > C - alpha:
                    n_large += 1
                    sum_large_remaining += C - w
                elif w > half_C:
                    n_medium += 1
                    sum_medium += w
                elif w >= alpha:
                    n_small_med += 1
                    sum_small_med += w
            
            # Each large item gets its own bin
            # Medium items can pair with small items but not with each other (since both > C/2 won't fit... wait medium is <= C - alpha and > C/2)
            # Actually medium items (> C/2) each need their own bin (can't pair two)
            # But they can share with small items
            # Remaining space after large bins: sum_large_remaining
            # Medium items each need own bin, remaining space: C - w for each medium
            
            # Number of bins needed at minimum:
            # n_large bins (for large items)
            # n_medium bins (for medium items, since each > C/2, can't pair)
            # Then small-med items need to fill remaining space or get new bins
            
            # Space available in large bins for small-med items: sum_large_remaining
            # But small-med items have weight >= alpha, and large bins have remaining < alpha
            # So actually no small-med items fit in large bins!
            # Space available in medium bins: sum over medium items of (C - w_medium)
            # Each medium item has w > C/2, so remaining < C/2, and small-med items have w <= C/2
            # So some small-med items might fit
            
            sum_medium_remaining = n_medium * C - sum_medium  # total remaining in medium bins
            
            # Small-med items that don't fit in medium bins need new bins
            leftover = max(0, sum_small_med - sum_medium_remaining)
            extra_bins = (leftover + C - 1) // C if leftover > 0 else 0
            
            lb = n_large + n_medium + extra_bins
            if lb > best_lb:
                best_lb = lb
            
            # Early exit if we've checked enough
            if alpha > 20 and alpha % 5 != 0:
                continue
        
        return best_lb
    
    # Only compute L2 for manageable sizes
    if n <= 5000 and bin_capacity <= 10000:
        lower_bound = compute_l2_bound()
    else:
        lower_bound = lower_bound_basic
    
    # Optimized Best Fit decode using bisect
    def decode_best_fit(perm):
        # sorted_remaining: sorted list of (remaining_capacity, bin_index)
        # We want to find the bin with smallest remaining >= w (best fit)
        sorted_remaining = []  # sorted by remaining capacity
        bin_items = []
        
        for idx in perm:
            w = weights[idx]
            # Find first bin with remaining >= w using bisect
            pos = bisect.bisect_left(sorted_remaining, (w,))
            if pos < len(sorted_remaining):
                # Best fit: this is the tightest fit (smallest remaining >= w)
                rem, b_idx = sorted_remaining[pos]
                # Remove this entry
                sorted_remaining.pop(pos)
                new_rem = rem - w
                bin_items[b_idx].append(idx)
                if new_rem > 0:
                    bisect.insort(sorted_remaining, (new_rem, b_idx))
            else:
                # New bin
                b_idx = len(bin_items)
                bin_items.append([idx])
                new_rem = bin_capacity - w
                if new_rem > 0:
                    bisect.insort(sorted_remaining, (new_rem, b_idx))
        
        return bin_items
    
    def evaluate(perm):
        return len(decode_best_fit(perm))
    
    def format_solution(perm):
        packing = decode_best_fit(perm)
        bin_weights = [sum(weights[i] for i in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}
    
    # FFD permutation
    ffd_perm = sorted(range(n), key=lambda i: -weights[i])
    
    # Population size based on instance size
    if n < 50:
        pop_size = 50
    elif n < 200:
        pop_size = 80
    else:
        pop_size = 40
    
    population = []
    fitness = []
    
    best_perm = list(ffd_perm)
    best_fitness = evaluate(ffd_perm)
    
    if best_fitness <= lower_bound:
        return format_solution(best_perm)
    
    population.append(list(ffd_perm))
    fitness.append(best_fitness)
    
    # Also try reverse of FFD
    rev_ffd = list(reversed(ffd_perm))
    rev_f = evaluate(rev_ffd)
    population.append(rev_ffd)
    fitness.append(rev_f)
    if rev_f < best_fitness:
        best_fitness = rev_f
        best_perm = list(rev_ffd)
        if best_fitness <= lower_bound:
            return format_solution(best_perm)
    
    # Weight classes for generating diverse initial solutions
    half_cap = bin_capacity // 2
    third_cap = bin_capacity // 3
    large_items = [i for i in range(n) if weights[i] > half_cap]
    medium_items = [i for i in range(n) if third_cap < weights[i] <= half_cap]
    small_items = [i for i in range(n) if weights[i] <= third_cap]
    
    def generate_weight_class_perm():
        """Shuffle within weight classes, keep large first."""
        l = list(large_items)
        m = list(medium_items)
        s = list(small_items)
        random.shuffle(l)
        random.shuffle(m)
        random.shuffle(s)
        return l + m + s
    
    # Generate initial population
    while len(population) < pop_size:
        if time.time() - start_time > time_limit * 0.5:
            break
        
        idx = len(population)
        
        if idx < 2 + 3:  # Perturbed FFD with small swaps among similar-weight items
            perm = list(ffd_perm)
            num_swaps = random.randint(1, max(1, n // 10))
            for _ in range(num_swaps):
                a = random.randrange(n)
                # Swap with nearby position (similar weight)
                b = min(n - 1, max(0, a + random.randint(-max(3, n // 20), max(3, n // 20))))
                perm[a], perm[b] = perm[b], perm[a]
        elif idx < 2 + 3 + 3:  # Weight-class shuffled
            perm = generate_weight_class_perm()
        else:
            # Mix of random and partially sorted
            if random.random() < 0.5:
                perm = list(range(n))
                random.shuffle(perm)
            else:
                perm = generate_weight_class_perm()
        
        f = evaluate(perm)
        
        # Also try reverse, keep better
        rev_perm = list(reversed(perm))
        rev_f = evaluate(rev_perm)
        if rev_f < f:
            perm = rev_perm
            f = rev_f
        
        population.append(perm)
        fitness.append(f)
        if f < best_fitness:
            best_fitness = f
            best_perm = list(perm)
            if best_fitness <= lower_bound:
                return format_solution(best_perm)
    
    # Tournament selection
    def tournament_select(k=3):
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best_c = candidates[0]
        for c in candidates[1:]:
            if fitness[c] < fitness[best_c]:
                best_c = c
        return best_c
    
    # Order Crossover (OX)
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
    
    # Bin-Based Crossover (BBX)
    def bbx_crossover(p1, p2):
        # Decode parent1 into bins
        bins1 = decode_best_fit(p1)
        
        # Score bins by fullness
        scored_bins = []
        for b in bins1:
            bw = sum(weights[i] for i in b)
            fullness = bw / bin_capacity
            scored_bins.append((fullness, b))
        
        # Select bins with fullness > threshold (0.8) or top 50%
        threshold = 0.8
        good_bins = [b for f, b in scored_bins if f >= threshold]
        
        # If too few good bins, take top 50%
        if len(good_bins) < len(scored_bins) // 3:
            scored_bins.sort(key=lambda x: -x[0])
            count = max(1, len(scored_bins) // 2)
            good_bins = [b for f, b in scored_bins[:count]]
        
        # Randomly select a subset of good bins to preserve
        if len(good_bins) > 1:
            k = random.randint(1, len(good_bins))
            selected_bins = random.sample(good_bins, k)
        else:
            selected_bins = good_bins
        
        placed = set()
        child_prefix = []
        for b in selected_bins:
            for item in b:
                child_prefix.append(item)
                placed.add(item)
        
        # Remaining items in order of p2
        child_suffix = [item for item in p2 if item not in placed]
        
        child = child_prefix + child_suffix
        return child
    
    # Mutation: swap
    def mutate_swap(perm, num_swaps=1):
        perm = list(perm)
        for _ in range(num_swaps):
            a, b = random.randrange(n), random.randrange(n)
            perm[a], perm[b] = perm[b], perm[a]
        return perm
    
    # Mutation: inversion
    def mutate_inversion(perm):
        perm = list(perm)
        if n <= 1:
            return perm
        a, b = sorted(random.sample(range(n), 2))
        perm[a:b+1] = reversed(perm[a:b+1])
        return perm
    
    # Mutation: scramble
    def mutate_scramble(perm):
        perm = list(perm)
        if n <= 1:
            return perm
        a, b = sorted(random.sample(range(n), 2))
        sub = perm[a:b+1]
        random.shuffle(sub)
        perm[a:b+1] = sub
        return perm
    
    # Mutation: block move
    def mutate_block_move(perm):
        perm = list(perm)
        if n <= 2:
            return perm
        a, b = sorted(random.sample(range(n), 2))
        block = perm[a:b+1]
        del perm[a:b+1]
        insert_pos = random.randrange(len(perm) + 1)
        perm[insert_pos:insert_pos] = block
        return perm
    
    # GA parameters
    elite_count = max(2, pop_size // 10)
    mutation_rate = 0.3
    crossover_rate = 0.85
    
    generation = 0
    max_generations = 100000
    
    offspring_counter = 0
    
    while generation < max_generations:
        # Time check every 5 generations
        if generation % 5 == 0:
            if time.time() - start_time > time_limit * 0.97:
                break
        
        if best_fitness <= lower_bound:
            break
        
        generation += 1
        
        # Sort population by fitness
        paired = list(zip(fitness, population))
        paired.sort(key=lambda x: x[0])
        fitness = [p[0] for p in paired]
        population = [p[1] for p in paired]
        
        # Adaptive parameters based on diversity
        unique_fitnesses = len(set(fitness))
        diversity_ratio = unique_fitnesses / len(fitness)
        
        if diversity_ratio < 0.3:
            current_mutation_rate = 0.6
            current_tournament_k = 3
            swap_count_max = max(1, n // 5)
        else:
            current_mutation_rate = 0.3
            current_tournament_k = 5 if diversity_ratio > 0.6 else 3
            swap_count_max = max(1, n // 10)
        
        new_population = []
        new_fitness = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(list(population[i]))
            new_fitness.append(fitness[i])
        
        elite_fitness_set = set(fitness[:elite_count])
        
        # Periodically inject fresh random individuals
        inject_count = 0
        if generation % 50 == 0:
            inject_count = max(1, pop_size // 10)
        
        offspring_count = 0
        
        while len(new_population) < pop_size:
            offspring_count += 1
            
            # Time check every 5 offspring
            if offspring_count % 5 == 0:
                if time.time() - start_time > time_limit * 0.97:
                    break
            
            # Injection of fresh individuals
            if inject_count > 0:
                inject_count -= 1
                if random.random() < 0.5:
                    child = list(range(n))
                    random.shuffle(child)
                else:
                    child = generate_weight_class_perm()
                f = evaluate(child)
                new_population.append(child)
                new_fitness.append(f)
                if f < best_fitness:
                    best_fitness = f
                    best_perm = list(child)
                    if best_fitness <= lower_bound:
                        break
                continue
            
            # Selection
            p1_idx = tournament_select(current_tournament_k)
            p2_idx = tournament_select(current_tournament_k)
            
            # Crossover
            if random.random() < crossover_rate:
                if random.random() < 0.7:
                    # BBX crossover
                    child = bbx_crossover(population[p1_idx], population[p2_idx])
                else:
                    # OX crossover
                    if random.random() < 0.5:
                        child = order_crossover(population[p1_idx], population[p2_idx])
                    else:
                        child = order_crossover(population[p2_idx], population[p1_idx])
            else:
                child = list(population[p1_idx])
            
            # Mutation
            if random.random() < current_mutation_rate:
                r = random.random()
                if r < 0.30:
                    child = mutate_swap(child, random.randint(1, swap_count_max))
                elif r < 0.55:
                    child = mutate_inversion(child)
                elif r < 0.80:
                    child = mutate_block_move(child)
                else:
                    child = mutate_scramble(child)
            
            f = evaluate(child)
            
            # Duplicate elimination: if fitness matches an elite, re-mutate with 50% chance
            if f in elite_fitness_set and random.random() < 0.5:
                r = random.random()
                if r < 0.5:
                    child = mutate_swap(child, random.randint(1, max(1, n // 5)))
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
    
    return format_solution(best_perm)