import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    # Decode a permutation into a packing using Best Fit with bisect optimization
    def decode_best_fit(perm):
        # sorted_rem: list of (remaining_capacity, bin_id)
        # We use bisect to find the bin with smallest remaining >= w
        sorted_rem = []  # sorted by remaining capacity
        bin_items_dict = {}
        bin_weights_dict = {}
        next_bin_id = 0
        
        for idx in perm:
            w = weights[idx]
            # Find first bin with remaining >= w
            pos = bisect.bisect_left(sorted_rem, (w,))
            if pos < len(sorted_rem):
                # Best fit: smallest remaining >= w
                rem, bid = sorted_rem[pos]
                # Remove this entry
                sorted_rem.pop(pos)
                new_rem = rem - w
                bin_weights_dict[bid] += w
                bin_items_dict[bid].append(idx)
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, bid))
            else:
                # Open new bin
                bid = next_bin_id
                next_bin_id += 1
                bin_weights_dict[bid] = w
                bin_items_dict[bid] = [idx]
                new_rem = C - w
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, bid))
        
        # Convert to lists
        bin_items = []
        bin_wts = []
        for bid in range(next_bin_id):
            if bid in bin_items_dict:
                bin_items.append(bin_items_dict[bid])
                bin_wts.append(bin_weights_dict[bid])
        return bin_items, bin_wts
    
    # Decode using First Fit
    def decode_first_fit(perm):
        bin_weights = []
        bin_items = []
        for idx in perm:
            w = weights[idx]
            placed = False
            for b in range(len(bin_weights)):
                if bin_capacity - bin_weights[b] >= w:
                    bin_weights[b] += w
                    bin_items[b].append(idx)
                    placed = True
                    break
            if not placed:
                bin_weights.append(w)
                bin_items.append([idx])
        return bin_items, bin_weights
    
    def decode(perm):
        return decode_best_fit(perm)
    
    # Granular fitness: returns (num_bins, -fill_score)
    # fill_score = sum((bw/C)^2) / num_bins  -- higher is better (more tightly packed)
    # We minimize (num_bins, -fill_score)
    def fitness(perm):
        _, bw = decode(perm)
        num_bins = len(bw)
        if num_bins == 0:
            return (0, 0.0)
        c_sq = C * C
        fill_score = 0.0
        for w in bw:
            fill_score += (w * w) / c_sq
        fill_score /= num_bins
        return (num_bins, -fill_score)
    
    # Generate initial solutions
    def make_ffd_perm():
        return sorted(range(n), key=lambda i: -weights[i])
    
    def make_random_perm():
        p = list(range(n))
        random.shuffle(p)
        return p
    
    def make_ffd_random_perm():
        return sorted(range(n), key=lambda i: -weights[i] + random.uniform(-0.5, 0.5) * weights[i])
    
    def make_ffd_swap_perm(num_swaps):
        p = make_ffd_perm()
        for _ in range(num_swaps):
            if len(p) > 1:
                i, j = random.sample(range(len(p)), 2)
                p[i], p[j] = p[j], p[i]
        return p
    
    def make_ascending_perm():
        return sorted(range(n), key=lambda i: weights[i])
    
    def make_weight_class_perm():
        # Group into large/medium/small, shuffle within groups, large first
        items = list(range(n))
        large = [i for i in items if weights[i] > C * 2 // 3]
        medium = [i for i in items if C // 3 < weights[i] <= C * 2 // 3]
        small = [i for i in items if weights[i] <= C // 3]
        random.shuffle(large)
        random.shuffle(medium)
        random.shuffle(small)
        return large + medium + small
    
    # Order Crossover (OX)
    def order_crossover(p1, p2):
        size = len(p1)
        if size <= 2:
            return p1[:], p2[:]
        
        c1, c2 = random.sample(range(size), 2)
        if c1 > c2:
            c1, c2 = c2, c1
        
        child1 = [-1] * size
        child1[c1:c2+1] = p1[c1:c2+1]
        segment_set = set(p1[c1:c2+1])
        
        pos = (c2 + 1) % size
        for i in range(size):
            idx = (c2 + 1 + i) % size
            gene = p2[idx]
            if gene not in segment_set:
                child1[pos] = gene
                pos = (pos + 1) % size
        
        child2 = [-1] * size
        child2[c1:c2+1] = p2[c1:c2+1]
        segment_set2 = set(p2[c1:c2+1])
        
        pos = (c2 + 1) % size
        for i in range(size):
            idx = (c2 + 1 + i) % size
            gene = p1[idx]
            if gene not in segment_set2:
                child2[pos] = gene
                pos = (pos + 1) % size
        
        return child1, child2
    
    # Partially Mapped Crossover (PMX)
    def pmx_crossover(p1, p2):
        size = len(p1)
        if size <= 2:
            return p1[:], p2[:]
        
        c1, c2 = sorted(random.sample(range(size), 2))
        
        child1 = p1[:]
        child2 = p2[:]
        
        pos1 = [0] * size
        pos2 = [0] * size
        for i in range(size):
            pos1[child1[i]] = i
            pos2[child2[i]] = i
        
        for i in range(c1, c2 + 1):
            val1 = child1[i]
            val2 = child2[i]
            if val1 != val2:
                p1_pos = pos1[val2]
                child1[i], child1[p1_pos] = child1[p1_pos], child1[i]
                pos1[val1] = p1_pos
                pos1[val2] = i
                
                p2_pos = pos2[val1]
                child2[i], child2[p2_pos] = child2[p2_pos], child2[i]
                pos2[val2] = p2_pos
                pos2[val1] = i
        
        return child1, child2
    
    # Group-Preserving Crossover
    def group_crossover(p1, p2):
        # Decode parent1 to get bin assignments
        bins1, bw1 = decode(p1)
        if len(bins1) == 0:
            return p1[:], p2[:]
        
        # Select ~50% of bins, preferring well-filled ones
        num_select = max(1, len(bins1) // 2)
        # Weight selection by fill ratio
        bin_indices = list(range(len(bins1)))
        # Sort by fill ratio descending, pick top ones with some randomness
        bin_indices.sort(key=lambda b: -bw1[b])
        # Take the top num_select with some randomness
        selected = set()
        for b in bin_indices:
            if len(selected) >= num_select:
                break
            if random.random() < 0.7 or len(selected) < num_select // 2:
                selected.add(b)
        
        # Items from selected bins
        selected_items = set()
        front_items = []
        for b in selected:
            for item in bins1[b]:
                selected_items.add(item)
                front_items.append(item)
        
        # Remaining items in p2's order
        remaining = [item for item in p2 if item not in selected_items]
        
        child1 = front_items + remaining
        
        # For child2, reverse roles
        bins2, bw2 = decode(p2)
        if len(bins2) == 0:
            return child1, p2[:]
        
        num_select2 = max(1, len(bins2) // 2)
        bin_indices2 = list(range(len(bins2)))
        bin_indices2.sort(key=lambda b: -bw2[b])
        selected2 = set()
        for b in bin_indices2:
            if len(selected2) >= num_select2:
                break
            if random.random() < 0.7 or len(selected2) < num_select2 // 2:
                selected2.add(b)
        
        selected_items2 = set()
        front_items2 = []
        for b in selected2:
            for item in bins2[b]:
                selected_items2.add(item)
                front_items2.append(item)
        
        remaining2 = [item for item in p1 if item not in selected_items2]
        child2 = front_items2 + remaining2
        
        return child1, child2
    
    # Mutation operators
    def swap_mutation(perm):
        p = perm[:]
        if len(p) <= 1:
            return p
        i, j = random.sample(range(len(p)), 2)
        p[i], p[j] = p[j], p[i]
        return p
    
    def insert_mutation(perm):
        p = perm[:]
        if len(p) <= 1:
            return p
        i = random.randrange(len(p))
        gene = p.pop(i)
        j = random.randrange(len(p) + 1)
        p.insert(j, gene)
        return p
    
    def inversion_mutation(perm):
        p = perm[:]
        if len(p) <= 2:
            return p
        i, j = sorted(random.sample(range(len(p)), 2))
        p[i:j+1] = reversed(p[i:j+1])
        return p
    
    def scramble_mutation(perm):
        p = perm[:]
        if len(p) <= 2:
            return p
        seg_len = random.randint(2, min(8, len(p)))
        start = random.randint(0, len(p) - seg_len)
        segment = p[start:start+seg_len]
        random.shuffle(segment)
        p[start:start+seg_len] = segment
        return p
    
    # Bin-emptying mutation
    def bin_emptying_mutation(perm):
        bins_list, bw = decode(perm)
        if len(bins_list) <= 1:
            return perm[:]
        
        # Find the least-filled bin
        min_bin = min(range(len(bins_list)), key=lambda b: bw[b])
        items_to_move = set(bins_list[min_bin])
        
        # Remove these items from perm and insert them at random earlier positions
        p = [x for x in perm if x not in items_to_move]
        items_list = list(items_to_move)
        random.shuffle(items_list)
        for item in items_list:
            # Insert at a random position in first half
            pos = random.randint(0, max(0, len(p) // 2))
            p.insert(pos, item)
        return p
    
    def mutate(perm, mutation_rate_current):
        r = random.random()
        if r < 0.25:
            return bin_emptying_mutation(perm)
        elif r < 0.45:
            return swap_mutation(perm)
        elif r < 0.65:
            return insert_mutation(perm)
        elif r < 0.85:
            return inversion_mutation(perm)
        else:
            return scramble_mutation(perm)
    
    # Tournament selection
    def tournament_select(population, fitnesses, k=3):
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best = min(candidates, key=lambda i: fitnesses[i])
        return population[best]
    
    # Parameters
    if n <= 20:
        pop_size = 30
    elif n <= 100:
        pop_size = 60
    elif n <= 500:
        pop_size = 80
    else:
        pop_size = 60
    
    # Scale with time
    if time_limit >= 60:
        pop_size = max(pop_size, 80)
    
    crossover_rate = 0.85
    mutation_rate = 0.3
    elite_count = max(2, pop_size // 10)
    tournament_size = 3
    
    # Hall of fame
    hall_of_fame = []  # list of (fitness, perm)
    hof_max = 5
    
    def update_hof(fit, perm):
        # Add to hall of fame if it's good enough
        for f, p in hall_of_fame:
            if f == fit and p == perm:
                return
        hall_of_fame.append((fit, perm[:]))
        hall_of_fame.sort(key=lambda x: x[0])
        while len(hall_of_fame) > hof_max:
            hall_of_fame.pop()
    
    # Initialize population
    population = []
    
    # FFD solution
    ffd_perm = make_ffd_perm()
    population.append(ffd_perm)
    
    # Ascending sort
    population.append(make_ascending_perm())
    
    # Weight class permutations
    for _ in range(min(3, pop_size // 10)):
        population.append(make_weight_class_perm())
    
    # FFD with noise
    for _ in range(pop_size // 3):
        population.append(make_ffd_random_perm())
    
    # FFD with swaps
    for _ in range(min(5, pop_size // 10)):
        population.append(make_ffd_swap_perm(random.randint(3, 10)))
    
    # Fill rest with random
    while len(population) < pop_size:
        population.append(make_random_perm())
    
    # Trim to pop_size
    population = population[:pop_size]
    
    # Evaluate initial population
    fitnesses = [fitness(p) for p in population]
    
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_perm = population[best_idx][:]
    best_fitness = fitnesses[best_idx]
    update_hof(best_fitness, best_perm)
    
    # Also try first fit decoding for the FFD perm
    ffd_items_ff, ffd_weights_ff = decode_first_fit(ffd_perm)
    ff_num = len(ffd_weights_ff)
    if ff_num < best_fitness[0]:
        best_fitness = (ff_num, -sum((w/C)**2 for w in ffd_weights_ff)/ff_num if ff_num > 0 else 0.0)
        best_perm = ffd_perm[:]
        # Store this special result
    
    # Track the best known packing directly too
    best_packing, best_bin_weights = decode(best_perm)
    best_num_bins = len(best_bin_weights)
    
    # Check FF on ffd_perm
    if ff_num < best_num_bins:
        best_num_bins = ff_num
        best_packing = ffd_items_ff
        best_bin_weights = ffd_weights_ff
    
    generation = 0
    no_improve_count = 0
    
    time_budget = time_limit * 0.95
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break
        
        generation += 1
        
        # Adaptive mutation rate
        current_mutation_rate = min(0.8, mutation_rate + 0.05 * (no_improve_count // 15))
        
        # Adaptive tournament size
        current_tournament = tournament_size if no_improve_count < 30 else 5
        
        # Sort population by fitness for elitism
        combined = list(zip(fitnesses, population))
        combined.sort(key=lambda x: x[0])
        fitnesses = [c[0] for c in combined]
        population = [c[1] for c in combined]
        
        new_population = []
        new_fitnesses = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(population[i][:])
            new_fitnesses.append(fitnesses[i])
        
        # Dual decode for elites: try FF too
        for i in range(min(elite_count, len(population))):
            packing_ff, bw_ff = decode_first_fit(population[i])
            if len(bw_ff) < best_num_bins:
                best_num_bins = len(bw_ff)
                best_packing = packing_ff
                best_bin_weights = bw_ff
                best_perm = population[i][:]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time >= time_budget:
                break
            
            parent1 = tournament_select(population, fitnesses, current_tournament)
            parent2 = tournament_select(population, fitnesses, current_tournament)
            
            if random.random() < crossover_rate:
                r = random.random()
                if r < 0.4:
                    child1, child2 = group_crossover(parent1, parent2)
                elif r < 0.7:
                    child1, child2 = order_crossover(parent1, parent2)
                else:
                    child1, child2 = pmx_crossover(parent1, parent2)
            else:
                child1 = parent1[:]
                child2 = parent2[:]
            
            if random.random() < current_mutation_rate:
                child1 = mutate(child1, current_mutation_rate)
            if random.random() < current_mutation_rate:
                child2 = mutate(child2, current_mutation_rate)
            
            f1 = fitness(child1)
            new_population.append(child1)
            new_fitnesses.append(f1)
            
            if len(new_population) < pop_size:
                f2 = fitness(child2)
                new_population.append(child2)
                new_fitnesses.append(f2)
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Update best
        gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_perm = population[gen_best_idx][:]
            no_improve_count = 0
            mutation_rate = 0.3  # reset
            
            # Update best packing
            bp, bbw = decode(best_perm)
            if len(bbw) < best_num_bins:
                best_num_bins = len(bbw)
                best_packing = bp
                best_bin_weights = bbw
            
            update_hof(best_fitness, best_perm)
        else:
            no_improve_count += 1
        
        # Duplicate elimination: if >30% share same fitness tuple, mutate excess
        if generation % 5 == 0:
            from collections import Counter
            fit_counts = Counter(fitnesses)
            threshold = len(population) * 0.3
            for fit_val, count in fit_counts.items():
                if count > threshold:
                    excess = 0
                    for i in range(elite_count, len(population)):
                        if fitnesses[i] == fit_val and excess < count - max(2, int(threshold // 2)):
                            # Heavy mutation
                            for _ in range(random.randint(3, 5)):
                                population[i] = mutate(population[i], 1.0)
                            fitnesses[i] = fitness(population[i])
                            excess += 1
        
        # Restart if stagnant
        if no_improve_count > 30:
            combined = list(zip(fitnesses, population))
            combined.sort(key=lambda x: x[0])
            keep = pop_size // 4
            population = [c[1] for c in combined[:keep]]
            fitnesses = [c[0] for c in combined[:keep]]
            
            # Add hall of fame
            for f, p in hall_of_fame:
                if len(population) < pop_size:
                    population.append(p[:])
                    fitnesses.append(f)
            
            # Ensure best is in there
            population.insert(0, best_perm[:])
            fitnesses.insert(0, best_fitness)
            
            # Heavy mutations of best
            num_mutated = pop_size // 4
            for _ in range(num_mutated):
                if len(population) >= pop_size:
                    break
                p = best_perm[:]
                num_ops = random.randint(5, 15)
                for __ in range(num_ops):
                    r = random.random()
                    if r < 0.5:
                        if len(p) > 1:
                            i, j = random.sample(range(len(p)), 2)
                            p[i], p[j] = p[j], p[i]
                    else:
                        if len(p) > 2:
                            i, j = sorted(random.sample(range(len(p)), 2))
                            p[i:j+1] = reversed(p[i:j+1])
                population.append(p)
                fitnesses.append(fitness(p))
            
            # FFD noise
            num_ffd = pop_size // 4
            for _ in range(num_ffd):
                if len(population) >= pop_size:
                    break
                p = make_ffd_random_perm()
                population.append(p)
                fitnesses.append(fitness(p))
            
            # Random
            while len(population) < pop_size:
                p = make_random_perm()
                population.append(p)
                fitnesses.append(fitness(p))
            
            population = population[:pop_size]
            fitnesses = fitnesses[:pop_size]
            no_improve_count = 0
    
    # Build final solution - try both decodings on best perm
    packing_bf, bw_bf = decode_best_fit(best_perm)
    packing_ff, bw_ff = decode_first_fit(best_perm)
    
    if len(bw_ff) < len(bw_bf):
        final_packing = packing_ff
        final_bw = bw_ff
    else:
        final_packing = packing_bf
        final_bw = bw_bf
    
    # Also compare with our tracked best
    if best_num_bins < len(final_bw):
        final_packing = best_packing
        final_bw = best_bin_weights
    
    # Try hall of fame solutions too
    for f, p in hall_of_fame:
        pb, wb = decode_best_fit(p)
        if len(wb) < len(final_bw):
            final_packing = pb
            final_bw = wb
        pf, wf = decode_first_fit(p)
        if len(wf) < len(final_bw):
            final_packing = pf
            final_bw = wf
    
    return {"packing": final_packing, "bin_weights": final_bw}
