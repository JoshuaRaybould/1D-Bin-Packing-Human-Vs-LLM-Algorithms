import random

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    from bisect import insort, bisect_left
    
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
    
    # BFD insert: insert items into existing bins using best-fit
    def bfd_insert(bins, remaining, items):
        sorted_items = sorted(items, key=lambda i: -weights[i])
        for item in sorted_items:
            w = weights[item]
            best_b = -1
            best_rem = bin_capacity + 1
            for b in range(len(bins)):
                r = remaining[b]
                if r >= w and r < best_rem:
                    best_rem = r
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(item)
                remaining[best_b] -= w
            else:
                bins.append([item])
                remaining.append(bin_capacity - w)
    
    # FFD insert
    def ffd_insert(bins, remaining, items):
        sorted_items = sorted(items, key=lambda i: -weights[i])
        for item in sorted_items:
            w = weights[item]
            placed = False
            for b in range(len(bins)):
                if remaining[b] >= w:
                    bins[b].append(item)
                    remaining[b] -= w
                    placed = True
                    break
            if not placed:
                bins.append([item])
                remaining.append(bin_capacity - w)
    
    def decode_bfd(perm):
        bins = []
        remaining = []
        for idx in perm:
            w = weights[idx]
            best_b = -1
            best_rem = bin_capacity + 1
            for b in range(len(bins)):
                r = remaining[b]
                if r >= w and r < best_rem:
                    best_rem = r
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
    
    # Worst-fit decreasing
    def decode_wfd(perm):
        bins = []
        remaining = []
        for idx in perm:
            w = weights[idx]
            best_b = -1
            best_rem = -1
            for b in range(len(bins)):
                r = remaining[b]
                if r >= w and r > best_rem:
                    best_rem = r
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(idx)
                remaining[best_b] -= w
            else:
                bins.append([idx])
                remaining.append(bin_capacity - w)
        return {'bins': bins, 'remaining': remaining, 'fitness': len(bins)}
    
    def copy_individual(ind):
        return {
            'bins': [b[:] for b in ind['bins']],
            'remaining': ind['remaining'][:],
            'fitness': ind['fitness']
        }
    
    def mutation_bin_elimination(ind):
        if len(ind['bins']) <= 1:
            return ind
        ind = copy_individual(ind)
        worst_b = -1
        worst_rem = -1
        for b in range(len(ind['bins'])):
            if ind['remaining'][b] > worst_rem:
                worst_rem = ind['remaining'][b]
                worst_b = b
        items = ind['bins'].pop(worst_b)
        ind['remaining'].pop(worst_b)
        old_count = ind['fitness']
        bfd_insert(ind['bins'], ind['remaining'], items)
        ind['fitness'] = len(ind['bins'])
        if ind['fitness'] > old_count:
            return None
        return ind
    
    def mutation_bin_elimination_random(ind):
        """Remove a random bin (biased toward least full) and reinsert."""
        if len(ind['bins']) <= 1:
            return ind
        ind = copy_individual(ind)
        nb = len(ind['bins'])
        # Weight selection toward emptier bins
        rem_list = ind['remaining'][:]
        total_rem = sum(rem_list)
        if total_rem == 0:
            target_b = random.randint(0, nb - 1)
        else:
            r = random.random() * total_rem
            cumul = 0
            target_b = nb - 1
            for b in range(nb):
                cumul += rem_list[b]
                if cumul >= r:
                    target_b = b
                    break
        items = ind['bins'].pop(target_b)
        ind['remaining'].pop(target_b)
        old_count = ind['fitness']
        # Shuffle items before reinserting for diversity
        random.shuffle(items)
        bfd_insert(ind['bins'], ind['remaining'], items)
        ind['fitness'] = len(ind['bins'])
        if ind['fitness'] > old_count:
            return None
        return ind
    
    def mutation_item_move(ind):
        if len(ind['bins']) <= 1:
            return ind
        ind = copy_individual(ind)
        b1 = random.randint(0, len(ind['bins']) - 1)
        item_idx_in_bin = random.randint(0, len(ind['bins'][b1]) - 1)
        item = ind['bins'][b1][item_idx_in_bin]
        w = weights[item]
        ind['bins'][b1].pop(item_idx_in_bin)
        ind['remaining'][b1] += w
        if len(ind['bins'][b1]) == 0:
            ind['bins'].pop(b1)
            ind['remaining'].pop(b1)
        best_b = -1
        best_rem = bin_capacity + 1
        for b in range(len(ind['bins'])):
            r = ind['remaining'][b]
            if r >= w and r < best_rem:
                best_rem = r
                best_b = b
        if best_b >= 0:
            ind['bins'][best_b].append(item)
            ind['remaining'][best_b] -= w
        else:
            ind['bins'].append([item])
            ind['remaining'].append(bin_capacity - w)
        ind['fitness'] = len(ind['bins'])
        return ind
    
    def mutation_item_swap(ind):
        if len(ind['bins']) < 2:
            return ind
        ind = copy_individual(ind)
        b1 = random.randint(0, len(ind['bins']) - 1)
        b2 = random.randint(0, len(ind['bins']) - 1)
        attempts = 0
        while b2 == b1 and attempts < 5:
            b2 = random.randint(0, len(ind['bins']) - 1)
            attempts += 1
        if b2 == b1:
            return ind
        i1 = random.randint(0, len(ind['bins'][b1]) - 1)
        i2 = random.randint(0, len(ind['bins'][b2]) - 1)
        item1 = ind['bins'][b1][i1]
        item2 = ind['bins'][b2][i2]
        w1 = weights[item1]
        w2 = weights[item2]
        new_rem_b1 = ind['remaining'][b1] + w1 - w2
        new_rem_b2 = ind['remaining'][b2] + w2 - w1
        if new_rem_b1 >= 0 and new_rem_b2 >= 0:
            ind['bins'][b1][i1] = item2
            ind['bins'][b2][i2] = item1
            ind['remaining'][b1] = new_rem_b1
            ind['remaining'][b2] = new_rem_b2
        return ind
    
    def mutation_multi_swap(ind):
        """Try swapping 1 item from one bin with 2 items from another."""
        if len(ind['bins']) < 2:
            return ind
        ind = copy_individual(ind)
        nb = len(ind['bins'])
        b1 = random.randint(0, nb - 1)
        b2 = random.randint(0, nb - 1)
        if b2 == b1:
            b2 = (b1 + 1) % nb
        if len(ind['bins'][b1]) < 1 or len(ind['bins'][b2]) < 2:
            return ind
        i1 = random.randint(0, len(ind['bins'][b1]) - 1)
        item1 = ind['bins'][b1][i1]
        w1 = weights[item1]
        # Pick 2 items from b2
        indices = random.sample(range(len(ind['bins'][b2])), 2)
        items2 = [ind['bins'][b2][j] for j in indices]
        w2_total = sum(weights[it] for it in items2)
        new_rem_b1 = ind['remaining'][b1] + w1 - w2_total
        new_rem_b2 = ind['remaining'][b2] + w2_total - w1
        if new_rem_b1 >= 0 and new_rem_b2 >= 0:
            # Remove item1 from b1
            ind['bins'][b1].pop(i1)
            ind['remaining'][b1] += w1
            # Remove items2 from b2 (remove in reverse sorted order)
            for j in sorted(indices, reverse=True):
                ind['bins'][b2].pop(j)
            ind['remaining'][b2] += w2_total
            # Add items2 to b1
            ind['bins'][b1].extend(items2)
            ind['remaining'][b1] -= w2_total
            # Add item1 to b2
            ind['bins'][b2].append(item1)
            ind['remaining'][b2] -= w1
        return ind
    
    def mutation_bin_merge(ind):
        if len(ind['bins']) < 2:
            return ind
        ind = copy_individual(ind)
        bin_order = sorted(range(len(ind['bins'])), key=lambda b: -ind['remaining'][b])
        b1 = bin_order[0]
        b2 = bin_order[1]
        total_weight = (bin_capacity - ind['remaining'][b1]) + (bin_capacity - ind['remaining'][b2])
        if total_weight <= bin_capacity:
            ind['bins'][b1].extend(ind['bins'][b2])
            ind['remaining'][b1] = bin_capacity - total_weight
            ind['bins'].pop(b2)
            ind['remaining'].pop(b2)
            ind['fitness'] = len(ind['bins'])
        else:
            smaller = b1 if (bin_capacity - ind['remaining'][b1]) <= (bin_capacity - ind['remaining'][b2]) else b2
            items = ind['bins'].pop(smaller)
            ind['remaining'].pop(smaller)
            bfd_insert(ind['bins'], ind['remaining'], items)
            ind['fitness'] = len(ind['bins'])
        return ind
    
    def mutate(ind):
        r = random.random()
        if r < 0.30:
            result = mutation_bin_elimination(ind)
            if result is None:
                return copy_individual(ind)
            return result
        elif r < 0.45:
            result = mutation_bin_elimination_random(ind)
            if result is None:
                return copy_individual(ind)
            return result
        elif r < 0.60:
            return mutation_item_move(ind)
        elif r < 0.75:
            return mutation_item_swap(ind)
        elif r < 0.88:
            return mutation_multi_swap(ind)
        else:
            return mutation_bin_merge(ind)
    
    def crossover(p1, p2):
        nb1 = len(p1['bins'])
        nb2 = len(p2['bins'])
        
        if nb1 == 0 or nb2 == 0:
            return copy_individual(p1), copy_individual(p2)
        
        # Select donor bins from P1 - prefer fuller bins
        k = random.randint(1, max(1, nb1 // 3))
        fullness_order = sorted(range(nb1), key=lambda b: p1['remaining'][b])
        top_count = max(1, nb1 * 2 // 3)
        selected_set = set()
        for _ in range(k):
            if random.random() < 0.8 and top_count > 0:
                idx = fullness_order[random.randint(0, top_count - 1)]
            else:
                idx = random.randint(0, nb1 - 1)
            selected_set.add(idx)
        
        transferred_items = set()
        child_bins = []
        child_remaining = []
        for b in selected_set:
            child_bins.append(p1['bins'][b][:])
            child_remaining.append(p1['remaining'][b])
            for item in p1['bins'][b]:
                transferred_items.add(item)
        
        for b in range(nb2):
            new_bin = [item for item in p2['bins'][b] if item not in transferred_items]
            if new_bin:
                rem = bin_capacity - sum(weights[i] for i in new_bin)
                child_bins.append(new_bin)
                child_remaining.append(rem)
                for item in new_bin:
                    transferred_items.add(item)
        
        all_placed = set()
        for b in child_bins:
            for item in b:
                all_placed.add(item)
        orphans = [i for i in range(n) if i not in all_placed]
        if orphans:
            bfd_insert(child_bins, child_remaining, orphans)
        
        # Remove empty bins
        non_empty = [(child_bins[i], child_remaining[i]) for i in range(len(child_bins)) if child_bins[i]]
        if non_empty:
            child_bins, child_remaining = zip(*non_empty)
            child_bins = [list(b) for b in child_bins]
            child_remaining = list(child_remaining)
        
        c1 = {'bins': child_bins, 'remaining': child_remaining, 'fitness': len(child_bins)}
        
        # Child 2
        k2 = random.randint(1, max(1, nb2 // 3))
        fullness_order2 = sorted(range(nb2), key=lambda b: p2['remaining'][b])
        top_count2 = max(1, nb2 * 2 // 3)
        selected_set2 = set()
        for _ in range(k2):
            if random.random() < 0.8 and top_count2 > 0:
                idx = fullness_order2[random.randint(0, top_count2 - 1)]
            else:
                idx = random.randint(0, nb2 - 1)
            selected_set2.add(idx)
        
        transferred_items2 = set()
        child_bins2 = []
        child_remaining2 = []
        for b in selected_set2:
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
        
        non_empty2 = [(child_bins2[i], child_remaining2[i]) for i in range(len(child_bins2)) if child_bins2[i]]
        if non_empty2:
            child_bins2, child_remaining2 = zip(*non_empty2)
            child_bins2 = [list(b) for b in child_bins2]
            child_remaining2 = list(child_remaining2)
        
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
    
    def cleanup_bin_elimination(ind):
        improved = True
        max_iters = 50
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if len(ind['bins']) <= 1:
                break
            worst_b = -1
            worst_rem = -1
            for b in range(len(ind['bins'])):
                if ind['remaining'][b] > worst_rem:
                    worst_rem = ind['remaining'][b]
                    worst_b = b
            items = ind['bins'][worst_b][:]
            old_fitness = ind['fitness']
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
    
    def deep_cleanup(ind, time_limit_abs):
        """More aggressive cleanup: try eliminating each bin, not just the worst."""
        improved = True
        while improved:
            improved = False
            if time.time() >= time_limit_abs:
                break
            if len(ind['bins']) <= 1:
                break
            # Try all bins sorted by remaining capacity (descending = least full first)
            bin_order = sorted(range(len(ind['bins'])), key=lambda b: -ind['remaining'][b])
            for worst_b in bin_order:
                if time.time() >= time_limit_abs:
                    break
                items = ind['bins'][worst_b][:]
                test_bins = [b[:] for i, b in enumerate(ind['bins']) if i != worst_b]
                test_remaining = [r for i, r in enumerate(ind['remaining']) if i != worst_b]
                bfd_insert(test_bins, test_remaining, items)
                if len(test_bins) < ind['fitness']:
                    ind['bins'] = test_bins
                    ind['remaining'] = test_remaining
                    ind['fitness'] = len(test_bins)
                    improved = True
                    break
        return ind
    
    # Population initialization
    if n <= 50:
        pop_size = 80
    elif n <= 200:
        pop_size = 60
    elif n <= 500:
        pop_size = 50
    else:
        pop_size = 40
    
    tournament_size = 4
    crossover_rate = 0.85
    mutation_rate = 0.6
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
    
    # WFD individual
    ind_wfd = decode_wfd(ffd_perm)
    cleanup_bin_elimination(ind_wfd)
    population.append(ind_wfd)
    
    # Perturbed FFD individuals with varying perturbation levels
    for pct in [5, 10, 15, 20, 30]:
        perm = ffd_perm[:]
        num_swaps = max(1, n * pct // 100)
        for __ in range(num_swaps):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            perm[a], perm[b] = perm[b], perm[a]
        ind = decode_bfd(perm)
        cleanup_bin_elimination(ind)
        population.append(ind)
    
    # Perturbed FFD with FFD decode
    for pct in [5, 10, 20]:
        perm = ffd_perm[:]
        num_swaps = max(1, n * pct // 100)
        for __ in range(num_swaps):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            perm[a], perm[b] = perm[b], perm[a]
        ind = decode_ffd(perm)
        cleanup_bin_elimination(ind)
        population.append(ind)
    
    # Fill rest with random-order BFD
    while len(population) < pop_size:
        perm = list(range(n))
        random.shuffle(perm)
        if random.random() < 0.5:
            ind = decode_bfd(perm)
        else:
            ind = decode_ffd(perm)
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
    max_stagnation = 25
    total_stagnation = 0
    max_total_stagnation = 80
    time_limit_abs = start_time + time_limit * 0.93
    
    while True:
        if time.time() >= time_limit_abs:
            break
        
        generation += 1
        prev_best = best_fitness
        
        population.sort(key=lambda x: x['fitness'])
        
        new_population = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(copy_individual(population[i]))
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() >= time_limit_abs:
                break
            
            p1_idx = tournament_select(population, tournament_size)
            p2_idx = tournament_select(population, tournament_size)
            
            if random.random() < crossover_rate:
                c1, c2 = crossover(population[p1_idx], population[p2_idx])
            else:
                c1 = copy_individual(population[p1_idx])
                c2 = copy_individual(population[p2_idx])
            
            # Apply multiple mutations sometimes
            num_mut = 1 if random.random() < 0.7 else 2
            for _ in range(num_mut):
                if random.random() < mutation_rate:
                    c1 = mutate(c1)
            for _ in range(num_mut):
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
        
        # Periodic cleanup on best individuals
        if generation % 5 == 0:
            for i in range(min(3, len(population))):
                cleanup_bin_elimination(population[i])
                if population[i]['fitness'] < best_fitness:
                    best_fitness = population[i]['fitness']
                    best = copy_individual(population[i])
        
        # Aggressive bin elimination on best
        for _ in range(5):
            trial = mutation_bin_elimination(best)
            if trial is not None and trial['fitness'] < best_fitness:
                best_fitness = trial['fitness']
                best = trial
        
        population[0] = copy_individual(best)
        
        if best_fitness < prev_best:
            stagnation = 0
            total_stagnation = 0
        else:
            stagnation += 1
            total_stagnation += 1
        
        # Stagnation: partial restart with diversity
        if stagnation >= max_stagnation:
            stagnation = 0
            population.sort(key=lambda x: x['fitness'])
            keep = max(3, elite_count)
            for i in range(keep, len(population)):
                r = random.random()
                if r < 0.4:
                    perm = list(range(n))
                    random.shuffle(perm)
                    ind = decode_bfd(perm)
                elif r < 0.7:
                    # Heavy perturbation of best
                    ind = copy_individual(best)
                    num_moves = max(1, n // 3)
                    for _ in range(num_moves):
                        ind = mutation_item_move(ind)
                else:
                    # Reconstruct from best's items in shuffled order
                    all_items = []
                    for b in best['bins']:
                        all_items.extend(b)
                    random.shuffle(all_items)
                    ind = decode_bfd(all_items)
                cleanup_bin_elimination(ind)
                population[i] = ind
                if ind['fitness'] < best_fitness:
                    best_fitness = ind['fitness']
                    best = copy_individual(ind)
        
        if total_stagnation >= max_total_stagnation:
            total_stagnation = 0
            population.sort(key=lambda x: x['fitness'])
            keep = 2
            for i in range(keep, len(population)):
                perm = list(range(n))
                random.shuffle(perm)
                if random.random() < 0.5:
                    ind = decode_bfd(perm)
                else:
                    ind = decode_ffd(perm)
                cleanup_bin_elimination(ind)
                population[i] = ind
                if ind['fitness'] < best_fitness:
                    best_fitness = ind['fitness']
                    best = copy_individual(ind)
    
    # Final post-processing with remaining time
    time_limit_final = start_time + time_limit * 0.99
    
    # Deep cleanup on best
    deep_cleanup(best, time_limit_final)
    
    # Try full re-decode with BFD in weight-descending order
    all_items_sorted = sorted(range(n), key=lambda i: -weights[i])
    baseline = decode_bfd(all_items_sorted)
    cleanup_bin_elimination(baseline)
    if baseline['fitness'] < best['fitness']:
        best = baseline
    
    # Try multiple random reinsertions of the least-full bins
    while time.time() < time_limit_final:
        if len(best['bins']) <= 1:
            break
        # Find the k least-full bins
        nb = len(best['bins'])
        bin_order = sorted(range(nb), key=lambda b: -best['remaining'][b])
        # Try removing 1-3 least-full bins and reinserting
        k = min(3, nb - 1)
        for try_k in range(1, k + 1):
            if time.time() >= time_limit_final:
                break
            test_ind = copy_individual(best)
            bins_to_remove = set(bin_order[:try_k])
            all_items_to_reinsert = []
            for b in bins_to_remove:
                all_items_to_reinsert.extend(test_ind['bins'][b])
            new_bins = []
            new_remaining = []
            for b in range(nb):
                if b not in bins_to_remove:
                    new_bins.append(test_ind['bins'][b])
                    new_remaining.append(test_ind['remaining'][b])
            # Try different orderings of reinsertion
            for _ in range(min(20, max(1, 100 // (try_k + 1)))):
                if time.time() >= time_limit_final:
                    break
                tb = [b[:] for b in new_bins]
                tr = new_remaining[:]
                random.shuffle(all_items_to_reinsert)
                bfd_insert(tb, tr, all_items_to_reinsert)
                if len(tb) < best['fitness']:
                    best['bins'] = tb
                    best['remaining'] = tr
                    best['fitness'] = len(tb)
                    break
        else:
            break
        if len(best['bins']) <= best['fitness']:
            break
        # If no improvement found, stop
        break
    
    return make_result(best)
