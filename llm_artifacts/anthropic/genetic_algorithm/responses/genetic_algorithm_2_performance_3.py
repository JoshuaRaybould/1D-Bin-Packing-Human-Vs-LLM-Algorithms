import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    w_arr = weights  # reference
    
    # Fast Best Fit decode
    def decode_bf(perm):
        sorted_rem = []  # sorted (remaining, bin_id)
        bin_items = []
        bin_wts = []
        
        for idx in perm:
            w = w_arr[idx]
            pos = bisect.bisect_left(sorted_rem, (w,))
            if pos < len(sorted_rem):
                rem, bid = sorted_rem[pos]
                del sorted_rem[pos]
                new_rem = rem - w
                bin_wts[bid] += w
                bin_items[bid].append(idx)
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, bid))
            else:
                bid = len(bin_items)
                bin_wts.append(w)
                bin_items.append([idx])
                new_rem = C - w
                if new_rem > 0:
                    bisect.insort(sorted_rem, (new_rem, bid))
        
        return bin_items, bin_wts
    
    # Fast First Fit Decreasing decode
    def decode_ffd(perm):
        # Sort by weight descending, then apply first fit
        order = sorted(perm, key=lambda i: -w_arr[i])
        remainders = []
        bin_items = []
        bin_wts = []
        for idx in order:
            w = w_arr[idx]
            placed = False
            # Find best fit
            best_b = -1
            best_rem = C + 1
            for b in range(len(remainders)):
                r = remainders[b]
                if r >= w and r < best_rem:
                    best_rem = r
                    best_b = b
                    if r == w:
                        break
            if best_b >= 0:
                remainders[best_b] -= w
                bin_wts[best_b] += w
                bin_items[best_b].append(idx)
            else:
                remainders.append(C - w)
                bin_wts.append(w)
                bin_items.append([idx])
        return bin_items, bin_wts
    
    # First Fit decode (order-dependent)
    def decode_ff(perm):
        sorted_rem = []  # sorted (remaining, bin_id)
        bin_items = []
        bin_wts = []
        
        for idx in perm:
            w = w_arr[idx]
            # First fit: find any bin with enough space
            # For speed, just use linear scan with early exit
            placed = False
            for b in range(len(bin_wts)):
                if C - bin_wts[b] >= w:
                    bin_wts[b] += w
                    bin_items[b].append(idx)
                    placed = True
                    break
            if not placed:
                bin_wts.append(w)
                bin_items.append([idx])
        
        return bin_items, bin_wts
    
    def decode(perm):
        return decode_bf(perm)
    
    # Fitness: (num_bins, -fill_score)
    def fitness(perm):
        _, bw = decode(perm)
        num_bins = len(bw)
        if num_bins == 0:
            return (0, 0.0)
        c_sq = C * C
        fill_score = 0.0
        for w in bw:
            fill_score += w * w
        fill_score /= (c_sq * num_bins)
        return (num_bins, -fill_score)
    
    def fitness_from_bw(bw):
        num_bins = len(bw)
        if num_bins == 0:
            return (0, 0.0)
        c_sq = C * C
        fill_score = 0.0
        for w in bw:
            fill_score += w * w
        fill_score /= (c_sq * num_bins)
        return (num_bins, -fill_score)
    
    # Generate initial solutions
    def make_ffd_perm():
        return sorted(range(n), key=lambda i: -w_arr[i])
    
    def make_random_perm():
        p = list(range(n))
        random.shuffle(p)
        return p
    
    def make_ffd_noise_perm(noise=0.3):
        return sorted(range(n), key=lambda i: -w_arr[i] + random.uniform(-noise, noise) * w_arr[i])
    
    def make_weight_class_perm():
        items = list(range(n))
        large = [i for i in items if w_arr[i] > C * 2 // 3]
        medium = [i for i in items if C // 3 < w_arr[i] <= C * 2 // 3]
        small = [i for i in items if w_arr[i] <= C // 3]
        random.shuffle(large)
        random.shuffle(medium)
        random.shuffle(small)
        return large + medium + small
    
    # Group-Preserving Crossover - most important for bin packing
    def group_crossover(p1, p2):
        bins1, bw1 = decode(p1)
        bins2, bw2 = decode(p2)
        if len(bins1) == 0 or len(bins2) == 0:
            return p1[:], p2[:]
        
        # Child 1: select well-filled bins from p1, rest from p2 order
        # Sort bins by fill ratio desc
        indices1 = sorted(range(len(bins1)), key=lambda b: -bw1[b])
        num_select = max(1, len(bins1) // 2)
        
        selected_items = set()
        front_items = []
        count = 0
        for b in indices1:
            if count >= num_select:
                break
            if random.random() < 0.75 or count < num_select // 3:
                for item in bins1[b]:
                    selected_items.add(item)
                    front_items.append(item)
                count += 1
        
        remaining = [item for item in p2 if item not in selected_items]
        child1 = front_items + remaining
        
        # Child 2: select well-filled bins from p2, rest from p1 order
        indices2 = sorted(range(len(bins2)), key=lambda b: -bw2[b])
        num_select2 = max(1, len(bins2) // 2)
        
        selected_items2 = set()
        front_items2 = []
        count = 0
        for b in indices2:
            if count >= num_select2:
                break
            if random.random() < 0.75 or count < num_select2 // 3:
                for item in bins2[b]:
                    selected_items2.add(item)
                    front_items2.append(item)
                count += 1
        
        remaining2 = [item for item in p1 if item not in selected_items2]
        child2 = front_items2 + remaining2
        
        return child1, child2
    
    # Order Crossover (OX)
    def order_crossover(p1, p2):
        size = len(p1)
        if size <= 2:
            return p1[:], p2[:]
        
        c1, c2 = sorted(random.sample(range(size), 2))
        
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
    
    # PMX Crossover
    def pmx_crossover(p1, p2):
        size = len(p1)
        if size <= 2:
            return p1[:], p2[:]
        
        c1, c2 = sorted(random.sample(range(size), 2))
        
        child1 = p1[:]
        child2 = p2[:]
        
        pos1 = [0] * n
        pos2 = [0] * n
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
    
    # Bin-emptying mutation: redistribute items from least-filled bins
    def bin_emptying_mutation(perm):
        bins_list, bw = decode(perm)
        if len(bins_list) <= 1:
            return perm[:]
        
        # Find k least-filled bins
        k = random.randint(1, max(1, min(3, len(bins_list) // 5)))
        sorted_bins = sorted(range(len(bins_list)), key=lambda b: bw[b])
        
        items_to_move = set()
        for i in range(min(k, len(sorted_bins))):
            for item in bins_list[sorted_bins[i]]:
                items_to_move.add(item)
        
        p = [x for x in perm if x not in items_to_move]
        items_list = sorted(items_to_move, key=lambda i: -w_arr[i])
        
        for item in items_list:
            pos = random.randint(0, max(0, len(p) // 3))
            p.insert(pos, item)
        return p
    
    # Multi-swap mutation
    def multi_swap_mutation(perm):
        p = perm[:]
        if len(p) <= 1:
            return p
        num_swaps = random.randint(2, min(5, len(p) // 2))
        for _ in range(num_swaps):
            i, j = random.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
        return p
    
    def mutate(perm):
        r = random.random()
        if r < 0.35:
            return bin_emptying_mutation(perm)
        elif r < 0.50:
            return insert_mutation(perm)
        elif r < 0.65:
            return swap_mutation(perm)
        elif r < 0.75:
            return multi_swap_mutation(perm)
        elif r < 0.88:
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
        pop_size = 40
    elif n <= 60:
        pop_size = 60
    elif n <= 200:
        pop_size = 80
    elif n <= 500:
        pop_size = 100
    else:
        pop_size = 80
    
    crossover_rate = 0.9
    mutation_rate = 0.4
    elite_count = max(2, pop_size // 8)
    tournament_size = 3
    
    # Best solution tracking
    best_num_bins = n + 1
    best_packing = None
    best_bin_weights = None
    
    def update_best(packing, bw):
        nonlocal best_num_bins, best_packing, best_bin_weights
        nb = len(bw)
        if nb < best_num_bins:
            best_num_bins = nb
            best_packing = [b[:] for b in packing]
            best_bin_weights = bw[:]
            return True
        return False
    
    def try_perm(perm):
        p1, w1 = decode_bf(perm)
        update_best(p1, w1)
        return fitness_from_bw(w1)
    
    # Hall of fame
    hall_of_fame = []
    hof_max = 10
    
    def update_hof(fit, perm):
        for f, p in hall_of_fame:
            if f == fit:
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
    
    # Also try FFD with the dedicated decoder
    ffd_items, ffd_wts = decode_ffd(list(range(n)))
    update_best(ffd_items, ffd_wts)
    
    # Weight class permutations
    for _ in range(min(5, pop_size // 8)):
        population.append(make_weight_class_perm())
    
    # FFD with various noise levels
    for noise in [0.1, 0.2, 0.3, 0.5, 0.7]:
        for _ in range(max(1, pop_size // 10)):
            population.append(make_ffd_noise_perm(noise))
    
    # Fill rest with random
    while len(population) < pop_size:
        population.append(make_random_perm())
    
    population = population[:pop_size]
    
    # Evaluate initial population
    fitnesses = []
    for p in population:
        bi, bw = decode(p)
        update_best(bi, bw)
        fitnesses.append(fitness_from_bw(bw))
    
    best_fitness = min(fitnesses)
    best_fit_idx = fitnesses.index(best_fitness)
    best_perm = population[best_fit_idx][:]
    update_hof(best_fitness, best_perm)
    
    generation = 0
    no_improve_count = 0
    
    time_budget = time_limit * 0.95
    
    # Precompute: check time every few offspring
    check_interval = max(1, pop_size // 4)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break
        
        generation += 1
        
        # Adaptive mutation rate
        current_mutation_rate = min(0.9, mutation_rate + 0.03 * (no_improve_count // 10))
        
        # Adaptive tournament size
        current_tournament = tournament_size if no_improve_count < 20 else min(6, tournament_size + no_improve_count // 20)
        
        # Sort population by fitness for elitism
        combined = sorted(zip(fitnesses, population), key=lambda x: x[0])
        fitnesses = [c[0] for c in combined]
        population = [c[1] for c in combined]
        
        new_population = []
        new_fitnesses = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(population[i][:])
            new_fitnesses.append(fitnesses[i])
        
        # Generate offspring
        offspring_count = 0
        while len(new_population) < pop_size:
            offspring_count += 1
            if offspring_count % check_interval == 0:
                if time.time() - start_time >= time_budget:
                    break
            
            parent1 = tournament_select(population, fitnesses, current_tournament)
            parent2 = tournament_select(population, fitnesses, current_tournament)
            
            if random.random() < crossover_rate:
                r = random.random()
                if r < 0.55:
                    child1, child2 = group_crossover(parent1, parent2)
                elif r < 0.80:
                    child1, child2 = order_crossover(parent1, parent2)
                else:
                    child1, child2 = pmx_crossover(parent1, parent2)
            else:
                child1 = parent1[:]
                child2 = parent2[:]
            
            if random.random() < current_mutation_rate:
                child1 = mutate(child1)
            if random.random() < current_mutation_rate:
                child2 = mutate(child2)
            
            bi1, bw1 = decode(child1)
            update_best(bi1, bw1)
            f1 = fitness_from_bw(bw1)
            new_population.append(child1)
            new_fitnesses.append(f1)
            
            if len(new_population) < pop_size:
                bi2, bw2 = decode(child2)
                update_best(bi2, bw2)
                f2 = fitness_from_bw(bw2)
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
            update_hof(best_fitness, best_perm)
        else:
            no_improve_count += 1
        
        # Diversity: duplicate elimination every few generations
        if generation % 8 == 0:
            from collections import Counter
            fit_counts = Counter()
            for f in fitnesses:
                fit_counts[f] += 1
            threshold = len(population) * 0.3
            for fit_val, count in fit_counts.items():
                if count > threshold:
                    excess = 0
                    for i in range(elite_count, len(population)):
                        if fitnesses[i] == fit_val and excess < count - max(2, int(threshold // 2)):
                            for _ in range(random.randint(2, 5)):
                                population[i] = mutate(population[i])
                            bi, bw = decode(population[i])
                            update_best(bi, bw)
                            fitnesses[i] = fitness_from_bw(bw)
                            excess += 1
        
        # Restart if stagnant
        if no_improve_count >= 25:
            combined = sorted(zip(fitnesses, population), key=lambda x: x[0])
            keep = pop_size // 5
            population = [c[1] for c in combined[:keep]]
            fitnesses = [c[0] for c in combined[:keep]]
            
            # Add hall of fame
            for f, p in hall_of_fame:
                if len(population) < pop_size:
                    population.append(p[:])
                    fitnesses.append(f)
            
            # Ensure best perm is present
            population.insert(0, best_perm[:])
            fitnesses.insert(0, best_fitness)
            
            # Heavy mutations of best
            num_mutated = pop_size // 3
            for _ in range(num_mutated):
                if len(population) >= pop_size:
                    break
                p = best_perm[:]
                num_ops = random.randint(3, 12)
                for __ in range(num_ops):
                    p = mutate(p)
                bi, bw = decode(p)
                update_best(bi, bw)
                population.append(p)
                fitnesses.append(fitness_from_bw(bw))
            
            # FFD noise variants
            for _ in range(pop_size // 4):
                if len(population) >= pop_size:
                    break
                noise = random.uniform(0.1, 0.8)
                p = make_ffd_noise_perm(noise)
                bi, bw = decode(p)
                update_best(bi, bw)
                population.append(p)
                fitnesses.append(fitness_from_bw(bw))
            
            # Random
            while len(population) < pop_size:
                p = make_random_perm()
                bi, bw = decode(p)
                update_best(bi, bw)
                population.append(p)
                fitnesses.append(fitness_from_bw(bw))
            
            population = population[:pop_size]
            fitnesses = fitnesses[:pop_size]
            no_improve_count = 0
    
    # Final: try all hall of fame with multiple decoders
    for f, p in hall_of_fame:
        pb, wb = decode_bf(p)
        update_best(pb, wb)
        pf, wf = decode_ff(p)
        update_best(pf, wf)
    
    # Try best perm with FF too
    pf, wf = decode_ff(best_perm)
    update_best(pf, wf)
    
    # Also try FFD decode on best perm
    pffd, wffd = decode_ffd(best_perm)
    update_best(pffd, wffd)
    
    return {"packing": best_packing, "bin_weights": best_bin_weights}
