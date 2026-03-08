# anthropic
# genetic_algorithm_2_performance_4.py

import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    w_arr = weights
    
    # Compute lower bound (L1: ceiling of sum/C)
    total_weight = sum(w_arr)
    L1 = (total_weight + C - 1) // C
    
    # Best Fit decode from permutation
    def decode_bf(perm):
        sorted_rem = []
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
    
    # Best Fit Decreasing
    def decode_bfd():
        order = sorted(range(n), key=lambda i: -w_arr[i])
        return decode_bf(order)
    
    # First Fit Decreasing
    def decode_ffd():
        order = sorted(range(n), key=lambda i: -w_arr[i])
        remainders = []
        bin_items = []
        bin_wts = []
        for idx in order:
            w = w_arr[idx]
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
    
    # Best solution tracking
    best_num_bins = n + 1
    best_packing = None
    best_bin_weights = None
    
    def update_best(packing, bw):
        nonlocal best_num_bins, best_packing, best_bin_weights
        nb = len(bw)
        if nb < best_num_bins or (nb == best_num_bins and best_bin_weights is not None):
            if nb < best_num_bins:
                best_num_bins = nb
                best_packing = [b[:] for b in packing]
                best_bin_weights = bw[:]
                return True
        return False
    
    # Solution representation: list of bins, each bin is a list of item indices
    # We work directly with bin groupings for crossover, decode back to permutations when needed
    
    def bins_to_perm(bins_list):
        """Convert bin grouping to a permutation (items from fuller bins first)"""
        # Sort bins by fill ratio descending, within each bin sort items by weight descending
        bin_info = [(sum(w_arr[i] for i in b), b) for b in bins_list]
        bin_info.sort(key=lambda x: -x[0])
        perm = []
        for _, b in bin_info:
            items = sorted(b, key=lambda i: -w_arr[i])
            perm.extend(items)
        return perm
    
    def perm_to_solution(perm):
        return decode_bf(perm)
    
    # Generate initial permutations
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
    
    # ---- Crossover Operators ----
    
    # Bin-Preserving Crossover (BPX) - works on bin groupings
    def bpx_crossover(p1, p2):
        bins1, bw1 = decode_bf(p1)
        bins2, bw2 = decode_bf(p2)
        
        if not bins1 or not bins2:
            return p1[:], p2[:]
        
        # Child 1: alternately pick best bins from parent 1 and parent 2
        used_items = set()
        child_bins = []
        
        # Sort bins by fullness descending
        idx1 = sorted(range(len(bins1)), key=lambda b: -bw1[b])
        idx2 = sorted(range(len(bins2)), key=lambda b: -bw2[b])
        
        i1, i2 = 0, 0
        turn = 0  # 0 = parent1, 1 = parent2
        
        while i1 < len(idx1) or i2 < len(idx2):
            if turn == 0 and i1 < len(idx1):
                b = idx1[i1]
                i1 += 1
                items = [item for item in bins1[b] if item not in used_items]
                if items:
                    child_bins.append(items)
                    for item in items:
                        used_items.add(item)
                turn = 1
            elif turn == 1 and i2 < len(idx2):
                b = idx2[i2]
                i2 += 1
                items = [item for item in bins2[b] if item not in used_items]
                if items:
                    child_bins.append(items)
                    for item in items:
                        used_items.add(item)
                turn = 0
            else:
                turn = 1 - turn
                if i1 >= len(idx1) and i2 >= len(idx2):
                    break
        
        # Any remaining items
        remaining = [i for i in range(n) if i not in used_items]
        if remaining:
            random.shuffle(remaining)
            child_bins.append(remaining)
        
        child1_perm = bins_to_perm(child_bins)
        
        # Child 2: reverse order (parent2 first)
        used_items2 = set()
        child_bins2 = []
        i1, i2 = 0, 0
        turn = 1
        
        while i1 < len(idx1) or i2 < len(idx2):
            if turn == 1 and i2 < len(idx2):
                b = idx2[i2]
                i2 += 1
                items = [item for item in bins2[b] if item not in used_items2]
                if items:
                    child_bins2.append(items)
                    for item in items:
                        used_items2.add(item)
                turn = 0
            elif turn == 0 and i1 < len(idx1):
                b = idx1[i1]
                i1 += 1
                items = [item for item in bins1[b] if item not in used_items2]
                if items:
                    child_bins2.append(items)
                    for item in items:
                        used_items2.add(item)
                turn = 1
            else:
                turn = 1 - turn
                if i1 >= len(idx1) and i2 >= len(idx2):
                    break
        
        remaining2 = [i for i in range(n) if i not in used_items2]
        if remaining2:
            random.shuffle(remaining2)
            child_bins2.append(remaining2)
        
        child2_perm = bins_to_perm(child_bins2)
        
        return child1_perm, child2_perm
    
    # Group crossover - preserves well-filled bins from one parent
    def group_crossover(p1, p2):
        bins1, bw1 = decode_bf(p1)
        bins2, bw2 = decode_bf(p2)
        if len(bins1) == 0 or len(bins2) == 0:
            return p1[:], p2[:]
        
        # Child 1: take fullest bins from p1, rest from p2 order
        indices1 = sorted(range(len(bins1)), key=lambda b: -bw1[b])
        # Take bins that are at least 80% full, or top half
        threshold = C * 0.8
        selected_items = set()
        front_items = []
        for b in indices1:
            if bw1[b] >= threshold or len(front_items) == 0:
                for item in bins1[b]:
                    selected_items.add(item)
                    front_items.append(item)
            elif random.random() < 0.3:
                for item in bins1[b]:
                    selected_items.add(item)
                    front_items.append(item)
        
        remaining = [item for item in p2 if item not in selected_items]
        child1 = front_items + remaining
        
        # Child 2: take fullest bins from p2, rest from p1 order
        indices2 = sorted(range(len(bins2)), key=lambda b: -bw2[b])
        selected_items2 = set()
        front_items2 = []
        for b in indices2:
            if bw2[b] >= threshold or len(front_items2) == 0:
                for item in bins2[b]:
                    selected_items2.add(item)
                    front_items2.append(item)
            elif random.random() < 0.3:
                for item in bins2[b]:
                    selected_items2.add(item)
                    front_items2.append(item)
        
        remaining2 = [item for item in p1 if item not in selected_items2]
        child2 = front_items2 + remaining2
        
        return child1, child2
    
    # Order crossover
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
    
    # ---- Mutation Operators ----
    
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
    
    # Bin-emptying mutation: tries to eliminate least-filled bins
    def bin_emptying_mutation(perm):
        bins_list, bw = decode_bf(perm)
        if len(bins_list) <= 1:
            return perm[:]
        
        # Find the least-filled bins
        k = random.randint(1, max(1, min(3, len(bins_list) // 4)))
        sorted_bins = sorted(range(len(bins_list)), key=lambda b: bw[b])
        
        items_to_move = []
        items_set = set()
        for i in range(min(k, len(sorted_bins))):
            for item in bins_list[sorted_bins[i]]:
                items_to_move.append(item)
                items_set.add(item)
        
        # Remove these items and sort them by weight descending
        p = [x for x in perm if x not in items_set]
        items_to_move.sort(key=lambda i: -w_arr[i])
        
        # Try to insert each item near bins that have space
        # Reconstruct remaining bins to find good positions
        for item in items_to_move:
            # Insert at a random position in the first third
            pos = random.randint(0, max(0, len(p) // 2))
            p.insert(pos, item)
        return p
    
    # Smart bin-emptying: try to fit items from smallest bin into other bins
    def smart_bin_emptying(perm):
        bins_list, bw = decode_bf(perm)
        nb = len(bins_list)
        if nb <= 1:
            return perm[:]
        
        # Pick the least filled bin
        min_b = min(range(nb), key=lambda b: bw[b])
        items_to_redistribute = bins_list[min_b][:]
        items_set = set(items_to_redistribute)
        
        # Sort items by weight descending for redistribution
        items_to_redistribute.sort(key=lambda i: -w_arr[i])
        
        # Build remaining perm without these items
        remaining_perm = [x for x in perm if x not in items_set]
        
        # For each item, find the best insertion position
        # by trying to place it right after an item in a bin with enough remaining capacity
        remaining_bins, remaining_bw = decode_bf(remaining_perm)
        rem_capacity = [C - bwt for bwt in remaining_bw]
        
        # Create a mapping: item -> bin_id in remaining solution
        item_to_bin = {}
        for bid, b in enumerate(remaining_bins):
            for item in b:
                item_to_bin[item] = bid
        
        # For each item to redistribute, find a bin with enough capacity
        new_perm = remaining_perm[:]
        for item in items_to_redistribute:
            w = w_arr[item]
            # Find best-fit bin
            best_bid = -1
            best_rem = C + 1
            for bid in range(len(rem_capacity)):
                if rem_capacity[bid] >= w and rem_capacity[bid] < best_rem:
                    best_rem = rem_capacity[bid]
                    best_bid = bid
            
            if best_bid >= 0:
                rem_capacity[best_bid] -= w
                # Insert item right after the last item of that bin in the perm
                last_item = remaining_bins[best_bid][-1]
                try:
                    pos = new_perm.index(last_item) + 1
                except ValueError:
                    pos = 0
                new_perm.insert(pos, item)
                remaining_bins[best_bid].append(item)
            else:
                # No bin has space, insert at end
                new_perm.append(item)
        
        return new_perm
    
    def multi_swap_mutation(perm):
        p = perm[:]
        if len(p) <= 1:
            return p
        num_swaps = random.randint(2, min(5, len(p) // 2))
        for _ in range(num_swaps):
            i, j = random.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
        return p
    
    # Segment shift mutation
    def segment_shift_mutation(perm):
        p = perm[:]
        if len(p) <= 3:
            return p
        seg_len = random.randint(1, min(5, len(p) // 3))
        start = random.randint(0, len(p) - seg_len)
        segment = p[start:start+seg_len]
        del p[start:start+seg_len]
        new_pos = random.randint(0, len(p))
        for i, item in enumerate(segment):
            p.insert(new_pos + i, item)
        return p
    
    def mutate(perm):
        r = random.random()
        if r < 0.25:
            return smart_bin_emptying(perm)
        elif r < 0.42:
            return bin_emptying_mutation(perm)
        elif r < 0.54:
            return insert_mutation(perm)
        elif r < 0.64:
            return segment_shift_mutation(perm)
        elif r < 0.74:
            return swap_mutation(perm)
        elif r < 0.82:
            return multi_swap_mutation(perm)
        elif r < 0.92:
            return inversion_mutation(perm)
        else:
            return scramble_mutation(perm)
    
    # Tournament selection
    def tournament_select(population, fitnesses, k=3):
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best = min(candidates, key=lambda i: fitnesses[i])
        return population[best]
    
    # Parameters - adaptive based on problem size
    if n <= 20:
        pop_size = 50
    elif n <= 60:
        pop_size = 80
    elif n <= 200:
        pop_size = 100
    elif n <= 500:
        pop_size = 120
    else:
        pop_size = 100
    
    crossover_rate = 0.92
    mutation_rate = 0.5
    elite_count = max(2, pop_size // 6)
    tournament_size = 3
    
    # Initialize best with BFD and FFD
    bfd_items, bfd_wts = decode_bfd()
    update_best(bfd_items, bfd_wts)
    
    ffd_items, ffd_wts = decode_ffd()
    update_best(ffd_items, ffd_wts)
    
    # Hall of fame
    hall_of_fame = []
    hof_max = 15
    
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
    
    # FFD-based permutation
    ffd_perm = make_ffd_perm()
    population.append(ffd_perm)
    
    # Weight class permutations
    for _ in range(min(8, pop_size // 6)):
        population.append(make_weight_class_perm())
    
    # FFD with various noise levels
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for noise in noise_levels:
        for _ in range(max(1, pop_size // 12)):
            population.append(make_ffd_noise_perm(noise))
    
    # Fill rest with random
    while len(population) < pop_size:
        population.append(make_random_perm())
    
    population = population[:pop_size]
    
    # Evaluate initial population
    fitnesses = []
    for p in population:
        bi, bw = decode_bf(p)
        update_best(bi, bw)
        fitnesses.append(fitness_from_bw(bw))
    
    best_fitness = min(fitnesses)
    best_fit_idx = fitnesses.index(best_fitness)
    best_perm = population[best_fit_idx][:]
    update_hof(best_fitness, best_perm)
    
    generation = 0
    no_improve_count = 0
    
    time_budget = time_limit * 0.95
    
    check_interval = max(1, pop_size // 4)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break
        
        # Early termination if optimal
        if best_num_bins <= L1:
            break
        
        generation += 1
        
        # Adaptive mutation rate
        current_mutation_rate = min(0.95, mutation_rate + 0.02 * (no_improve_count // 5))
        
        # Adaptive tournament size
        current_tournament = tournament_size if no_improve_count < 15 else min(7, tournament_size + no_improve_count // 15)
        
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
                if r < 0.40:
                    child1, child2 = bpx_crossover(parent1, parent2)
                elif r < 0.75:
                    child1, child2 = group_crossover(parent1, parent2)
                else:
                    child1, child2 = order_crossover(parent1, parent2)
            else:
                child1 = parent1[:]
                child2 = parent2[:]
            
            if random.random() < current_mutation_rate:
                child1 = mutate(child1)
            if random.random() < current_mutation_rate:
                child2 = mutate(child2)
            
            bi1, bw1 = decode_bf(child1)
            update_best(bi1, bw1)
            f1 = fitness_from_bw(bw1)
            new_population.append(child1)
            new_fitnesses.append(f1)
            
            if len(new_population) < pop_size:
                bi2, bw2 = decode_bf(child2)
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
        
        # Diversity injection
        if generation % 10 == 0:
            fit_set = {}
            for i, f in enumerate(fitnesses):
                if f not in fit_set:
                    fit_set[f] = []
                fit_set[f].append(i)
            for fit_val, indices in fit_set.items():
                if len(indices) > max(3, len(population) * 0.25):
                    for idx in indices[3:]:
                        if idx >= elite_count:
                            for _ in range(random.randint(2, 6)):
                                population[idx] = mutate(population[idx])
                            bi, bw = decode_bf(population[idx])
                            update_best(bi, bw)
                            fitnesses[idx] = fitness_from_bw(bw)
        
        # Restart if stagnant
        if no_improve_count >= 20:
            combined = sorted(zip(fitnesses, population), key=lambda x: x[0])
            keep = pop_size // 4
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
                # Apply smart bin emptying multiple times
                num_ops = random.randint(2, 8)
                for __ in range(num_ops):
                    p = mutate(p)
                bi, bw = decode_bf(p)
                update_best(bi, bw)
                population.append(p)
                fitnesses.append(fitness_from_bw(bw))
            
            # FFD noise variants
            for _ in range(pop_size // 4):
                if len(population) >= pop_size:
                    break
                noise = random.uniform(0.05, 1.0)
                p = make_ffd_noise_perm(noise)
                bi, bw = decode_bf(p)
                update_best(bi, bw)
                population.append(p)
                fitnesses.append(fitness_from_bw(bw))
            
            # Weight class
            for _ in range(pop_size // 8):
                if len(population) >= pop_size:
                    break
                p = make_weight_class_perm()
                bi, bw = decode_bf(p)
                update_best(bi, bw)
                population.append(p)
                fitnesses.append(fitness_from_bw(bw))
            
            # Random
            while len(population) < pop_size:
                p = make_random_perm()
                bi, bw = decode_bf(p)
                update_best(bi, bw)
                population.append(p)
                fitnesses.append(fitness_from_bw(bw))
            
            population = population[:pop_size]
            fitnesses = fitnesses[:pop_size]
            no_improve_count = 0
    
    # Final: try all hall of fame solutions with different decoders
    for f, p in hall_of_fame:
        pb, wb = decode_bf(p)
        update_best(pb, wb)
    
    # Also try best perm with BFD ordering
    bfd_perm = sorted(best_perm, key=lambda i: -w_arr[i])
    pb, wb = decode_bf(bfd_perm)
    update_best(pb, wb)
    
    return {"packing": best_packing, "bin_weights": best_bin_weights}
