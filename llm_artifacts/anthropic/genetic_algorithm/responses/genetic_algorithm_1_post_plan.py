import random

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    if n == 1:
        return {"packing": [[0]], "bin_weights": [weights[0]]}
    
    def make_result(individual):
        bins = individual['bins']
        bin_weights = [sum(weights[i] for i in b) for b in bins]
        return {"packing": bins, "bin_weights": bin_weights}
    
    # BFD insert: insert items into existing bins using best-fit, creating new bins as needed
    def bfd_insert(bins, remaining, items):
        # Sort items by weight descending
        sorted_items = sorted(items, key=lambda i: -weights[i])
        for item in sorted_items:
            w = weights[item]
            best_b = -1
            best_rem = bin_capacity + 1
            for b in range(len(bins)):
                if remaining[b] >= w and remaining[b] < best_rem:
                    best_rem = remaining[b]
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(item)
                remaining[best_b] -= w
            else:
                bins.append([item])
                remaining.append(bin_capacity - w)
    
    # Create individual from a permutation using a decode strategy
    def decode_bfd(perm):
        bins = []
        remaining = []
        for idx in perm:
            w = weights[idx]
            best_b = -1
            best_rem = bin_capacity + 1
            for b in range(len(bins)):
                if remaining[b] >= w and remaining[b] < best_rem:
                    best_rem = remaining[b]
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(idx)
                remaining[best_b] -= w
            else:
                bins.append([idx])
                remaining.append(bin_capacity - w)
        return {'bins': bins, 'remaining': remaining, 'fitness': len(bins)}
    
    def decode_ffd(perm):
        bins = []
        remaining = []
        for idx in perm:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if remaining[b] >= w:
                    bins[b].append(idx)
                    remaining[b] -= w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                remaining.append(bin_capacity - w)
        return {'bins': bins, 'remaining': remaining, 'fitness': len(bins)}
    
    def copy_individual(ind):
        return {
            'bins': [b[:] for b in ind['bins']],
            'remaining': ind['remaining'][:],
            'fitness': ind['fitness']
        }
    
    # Bin elimination mutation: remove least-full bin, reinsert items
    def mutation_bin_elimination(ind):
        if len(ind['bins']) <= 1:
            return ind
        ind = copy_individual(ind)
        # Find least-full bin (max remaining capacity)
        worst_b = -1
        worst_rem = -1
        for b in range(len(ind['bins'])):
            if ind['remaining'][b] > worst_rem:
                worst_rem = ind['remaining'][b]
                worst_b = b
        # Remove this bin
        items = ind['bins'].pop(worst_b)
        ind['remaining'].pop(worst_b)
        old_count = ind['fitness']
        # Reinsert items using BFD
        bfd_insert(ind['bins'], ind['remaining'], items)
        ind['fitness'] = len(ind['bins'])
        # If worse, revert is expensive, so we just accept (for diversity)
        # But if strictly worse by more than 1, revert
        if ind['fitness'] > old_count:
            return None  # signal to revert
        return ind
    
    # Item move mutation
    def mutation_item_move(ind):
        if len(ind['bins']) <= 1:
            return ind
        ind = copy_individual(ind)
        # Pick random bin and random item
        b1 = random.randint(0, len(ind['bins']) - 1)
        item_idx_in_bin = random.randint(0, len(ind['bins'][b1]) - 1)
        item = ind['bins'][b1][item_idx_in_bin]
        w = weights[item]
        # Remove item from bin
        ind['bins'][b1].pop(item_idx_in_bin)
        ind['remaining'][b1] += w
        # If bin empty, remove it
        if len(ind['bins'][b1]) == 0:
            ind['bins'].pop(b1)
            ind['remaining'].pop(b1)
            b1 = -1  # mark as removed
        # Find best-fit bin (different from original if it still exists)
        best_b = -1
        best_rem = bin_capacity + 1
        for b in range(len(ind['bins'])):
            if ind['remaining'][b] >= w and ind['remaining'][b] < best_rem:
                best_rem = ind['remaining'][b]
                best_b = b
        if best_b >= 0:
            ind['bins'][best_b].append(item)
            ind['remaining'][best_b] -= w
        else:
            ind['bins'].append([item])
            ind['remaining'].append(bin_capacity - w)
        ind['fitness'] = len(ind['bins'])
        return ind
    
    # Item swap mutation
    def mutation_item_swap(ind):
        if len(ind['bins']) < 2:
            return ind
        ind = copy_individual(ind)
        b1 = random.randint(0, len(ind['bins']) - 1)
        b2 = random.randint(0, len(ind['bins']) - 1)
        while b2 == b1:
            b2 = random.randint(0, len(ind['bins']) - 1)
        i1 = random.randint(0, len(ind['bins'][b1]) - 1)
        i2 = random.randint(0, len(ind['bins'][b2]) - 1)
        item1 = ind['bins'][b1][i1]
        item2 = ind['bins'][b2][i2]
        w1 = weights[item1]
        w2 = weights[item2]
        # Check feasibility
        new_rem_b1 = ind['remaining'][b1] + w1 - w2
        new_rem_b2 = ind['remaining'][b2] + w2 - w1
        if new_rem_b1 >= 0 and new_rem_b2 >= 0:
            ind['bins'][b1][i1] = item2
            ind['bins'][b2][i2] = item1
            ind['remaining'][b1] = new_rem_b1
            ind['remaining'][b2] = new_rem_b2
        return ind
    
    # Bin merge attempt mutation
    def mutation_bin_merge(ind):
        if len(ind['bins']) < 2:
            return ind
        ind = copy_individual(ind)
        # Sort bins by remaining capacity descending (least full first)
        bin_order = sorted(range(len(ind['bins'])), key=lambda b: -ind['remaining'][b])
        # Try to merge the two least-full bins
        b1 = bin_order[0]
        b2 = bin_order[1]
        total_weight = (bin_capacity - ind['remaining'][b1]) + (bin_capacity - ind['remaining'][b2])
        if total_weight <= bin_capacity:
            # Direct merge
            ind['bins'][b1].extend(ind['bins'][b2])
            ind['remaining'][b1] = bin_capacity - total_weight
            ind['bins'].pop(b2)
            ind['remaining'].pop(b2)
            ind['fitness'] = len(ind['bins'])
        else:
            # Try BFD reinsertion of smaller bin's items into other bins
            smaller = b1 if (bin_capacity - ind['remaining'][b1]) <= (bin_capacity - ind['remaining'][b2]) else b2
            items = ind['bins'].pop(smaller)
            ind['remaining'].pop(smaller)
            bfd_insert(ind['bins'], ind['remaining'], items)
            ind['fitness'] = len(ind['bins'])
        return ind
    
    def mutate(ind):
        r = random.random()
        if r < 0.40:
            result = mutation_bin_elimination(ind)
            if result is None:
                return copy_individual(ind)
            return result
        elif r < 0.65:
            return mutation_item_move(ind)
        elif r < 0.85:
            return mutation_item_swap(ind)
        else:
            return mutation_bin_merge(ind)
    
    # Falkenauer-style bin-based crossover
    def crossover(p1, p2):
        nb1 = len(p1['bins'])
        nb2 = len(p2['bins'])
        
        if nb1 == 0 or nb2 == 0:
            return copy_individual(p1), copy_individual(p2)
        
        # Select donor bins from P1
        k = random.randint(1, max(1, nb1 // 3))
        # Prefer fuller bins: sort by fullness descending
        fullness_order = sorted(range(nb1), key=lambda b: p1['remaining'][b])
        # Select from top half with higher probability
        top_half = max(1, nb1 // 2)
        selected_bins = []
        selected_set = set()
        for _ in range(k):
            if random.random() < 0.7 and top_half > 0:
                idx = fullness_order[random.randint(0, top_half - 1)]
            else:
                idx = random.randint(0, nb1 - 1)
            if idx not in selected_set:
                selected_set.add(idx)
                selected_bins.append(idx)
        
        # Child 1: selected P1 bins + cleaned P2 bins
        transferred_items = set()
        child_bins = []
        child_remaining = []
        for b in selected_bins:
            child_bins.append(p1['bins'][b][:])
            child_remaining.append(p1['remaining'][b])
            for item in p1['bins'][b]:
                transferred_items.add(item)
        
        # Process P2's bins: remove transferred items
        for b in range(nb2):
            new_bin = [item for item in p2['bins'][b] if item not in transferred_items]
            if new_bin:
                rem = bin_capacity - sum(weights[i] for i in new_bin)
                child_bins.append(new_bin)
                child_remaining.append(rem)
                for item in new_bin:
                    transferred_items.add(item)
        
        # Check for orphans (items not in any bin)
        all_placed = set()
        for b in child_bins:
            for item in b:
                all_placed.add(item)
        orphans = [i for i in range(n) if i not in all_placed]
        if orphans:
            bfd_insert(child_bins, child_remaining, orphans)
        
        c1 = {'bins': child_bins, 'remaining': child_remaining, 'fitness': len(child_bins)}
        
        # Child 2: swap roles
        k2 = random.randint(1, max(1, nb2 // 3))
        fullness_order2 = sorted(range(nb2), key=lambda b: p2['remaining'][b])
        top_half2 = max(1, nb2 // 2)
        selected_bins2 = []
        selected_set2 = set()
        for _ in range(k2):
            if random.random() < 0.7 and top_half2 > 0:
                idx = fullness_order2[random.randint(0, top_half2 - 1)]
            else:
                idx = random.randint(0, nb2 - 1)
            if idx not in selected_set2:
                selected_set2.add(idx)
                selected_bins2.append(idx)
        
        transferred_items2 = set()
        child_bins2 = []
        child_remaining2 = []
        for b in selected_bins2:
            child_bins2.append(p2['bins'][b][:])
            child_remaining2.append(p2['remaining'][b])
            for item in p2['bins'][b]:
                transferred_items2.add(item)
        
        for b in range(nb1):
            new_bin = [item for item in p1['bins'][b] if item not in transferred_items2]
            if new_bin:
                rem = bin_capacity - sum(weights[i] for i in new_bin)
                child_bins2.append(new_bin)
                child_remaining2.append(rem)
                for item in new_bin:
                    transferred_items2.add(item)
        
        all_placed2 = set()
        for b in child_bins2:
            for item in b:
                all_placed2.add(item)
        orphans2 = [i for i in range(n) if i not in all_placed2]
        if orphans2:
            bfd_insert(child_bins2, child_remaining2, orphans2)
        
        c2 = {'bins': child_bins2, 'remaining': child_remaining2, 'fitness': len(child_bins2)}
        
        return c1, c2
    
    def tournament_select(pop, k):
        best = None
        best_f = float('inf')
        for _ in range(k):
            i = random.randint(0, len(pop) - 1)
            if pop[i]['fitness'] < best_f:
                best_f = pop[i]['fitness']
                best = i
        return best
    
    # Apply bin elimination cleanup to an individual
    def cleanup_bin_elimination(ind):
        improved = True
        while improved:
            improved = False
            if len(ind['bins']) <= 1:
                break
            # Find least-full bin
            worst_b = -1
            worst_rem = -1
            for b in range(len(ind['bins'])):
                if ind['remaining'][b] > worst_rem:
                    worst_rem = ind['remaining'][b]
                    worst_b = b
            items = ind['bins'][worst_b][:]
            old_fitness = ind['fitness']
            # Try removing and reinserting
            test_bins = [b[:] for i, b in enumerate(ind['bins']) if i != worst_b]
            test_remaining = [r for i, r in enumerate(ind['remaining']) if i != worst_b]
            bfd_insert(test_bins, test_remaining, items)
            if len(test_bins) < old_fitness:
                ind['bins'] = test_bins
                ind['remaining'] = test_remaining
                ind['fitness'] = len(test_bins)
                improved = True
            else:
                break
        return ind
    
    # Population initialization
    if n <= 100:
        pop_size = 60
    elif n <= 500:
        pop_size = 40
    else:
        pop_size = 30
    
    tournament_size = 4
    crossover_rate = 0.85
    mutation_rate = 0.5
    elite_count = max(2, pop_size * 15 // 100)
    
    population = []
    
    # FFD individual
    ffd_perm = sorted(range(n), key=lambda i: -weights[i])
    ind_ffd = decode_ffd(ffd_perm)
    cleanup_bin_elimination(ind_ffd)
    population.append(ind_ffd)
    
    # BFD individual
    ind_bfd = decode_bfd(ffd_perm)
    cleanup_bin_elimination(ind_bfd)
    population.append(ind_bfd)
    
    # Perturbed FFD individuals
    for _ in range(min(8, pop_size // 5)):
        perm = ffd_perm[:]
        for __ in range(max(1, n // 10)):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            perm[a], perm[b] = perm[b], perm[a]
        ind = decode_bfd(perm)
        cleanup_bin_elimination(ind)
        population.append(ind)
    
    # Fill rest with random-order BFD
    while len(population) < pop_size:
        perm = list(range(n))
        random.shuffle(perm)
        ind = decode_bfd(perm)
        cleanup_bin_elimination(ind)
        population.append(ind)
    
    # Track best
    best = None
    best_fitness = float('inf')
    for ind in population:
        if ind['fitness'] < best_fitness:
            best_fitness = ind['fitness']
            best = copy_individual(ind)
    
    # Main GA loop
    generation = 0
    stagnation = 0
    max_stagnation = 30
    total_stagnation = 0
    max_total_stagnation = 100
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break
        
        generation += 1
        prev_best = best_fitness
        
        # Sort population by fitness
        population.sort(key=lambda x: x['fitness'])
        
        new_population = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(copy_individual(population[i]))
        
        # Generate offspring
        while len(new_population) < pop_size:
            if generation % 5 == 0 and time.time() - start_time >= time_limit * 0.95:
                break
            
            p1_idx = tournament_select(population, tournament_size)
            p2_idx = tournament_select(population, tournament_size)
            
            if random.random() < crossover_rate:
                c1, c2 = crossover(population[p1_idx], population[p2_idx])
            else:
                c1 = copy_individual(population[p1_idx])
                c2 = copy_individual(population[p2_idx])
            
            if random.random() < mutation_rate:
                c1 = mutate(c1)
            if random.random() < mutation_rate:
                c2 = mutate(c2)
            
            new_population.append(c1)
            if c1['fitness'] < best_fitness:
                best_fitness = c1['fitness']
                best = copy_individual(c1)
            
            if len(new_population) < pop_size:
                new_population.append(c2)
                if c2['fitness'] < best_fitness:
                    best_fitness = c2['fitness']
                    best = copy_individual(c2)
        
        population = new_population
        
        # Aggressive bin elimination on best individual
        for _ in range(3):
            trial = mutation_bin_elimination(best)
            if trial is not None and trial['fitness'] < best_fitness:
                best_fitness = trial['fitness']
                best = trial
        
        # Ensure best is in population
        population[0] = copy_individual(best)
        
        # Track stagnation
        if best_fitness < prev_best:
            stagnation = 0
            total_stagnation = 0
        else:
            stagnation += 1
            total_stagnation += 1
        
        # Stagnation restart
        if stagnation >= max_stagnation:
            stagnation = 0
            population.sort(key=lambda x: x['fitness'])
            keep = 5
            for i in range(keep, len(population)):
                if random.random() < 0.5:
                    perm = list(range(n))
                    random.shuffle(perm)
                    ind = decode_bfd(perm)
                else:
                    # Heavy perturbation of best
                    ind = copy_individual(best)
                    num_moves = max(1, n // 3)
                    for _ in range(num_moves):
                        ind = mutation_item_move(ind)
                cleanup_bin_elimination(ind)
                population[i] = ind
                if ind['fitness'] < best_fitness:
                    best_fitness = ind['fitness']
                    best = copy_individual(ind)
        
        if total_stagnation >= max_total_stagnation:
            total_stagnation = 0
            population.sort(key=lambda x: x['fitness'])
            keep = 3
            for i in range(keep, len(population)):
                perm = list(range(n))
                random.shuffle(perm)
                ind = decode_bfd(perm)
                cleanup_bin_elimination(ind)
                population[i] = ind
                if ind['fitness'] < best_fitness:
                    best_fitness = ind['fitness']
                    best = copy_individual(ind)
        
        if generation % 10 == 0:
            if time.time() - start_time >= time_limit * 0.95:
                break
    
    # Final post-processing: attempt bin elimination on every bin from least full to most full
    improved = True
    while improved:
        improved = False
        if time.time() - start_time >= time_limit * 0.98:
            break
        if len(best['bins']) <= 1:
            break
        # Find least-full bin
        worst_b = -1
        worst_rem = -1
        for b in range(len(best['bins'])):
            if best['remaining'][b] > worst_rem:
                worst_rem = best['remaining'][b]
                worst_b = b
        items = best['bins'][worst_b][:]
        test_bins = [b[:] for i, b in enumerate(best['bins']) if i != worst_b]
        test_remaining = [r for i, r in enumerate(best['remaining']) if i != worst_b]
        bfd_insert(test_bins, test_remaining, items)
        if len(test_bins) < best['fitness']:
            best['bins'] = test_bins
            best['remaining'] = test_remaining
            best['fitness'] = len(test_bins)
            improved = True
        else:
            break
    
    # Also try re-decoding all items via BFD in weight-descending order
    all_items_sorted = sorted(range(n), key=lambda i: -weights[i])
    baseline = decode_bfd(all_items_sorted)
    if baseline['fitness'] < best['fitness']:
        best = baseline
    
    return make_result(best)
