import random

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    if n == 1:
        return {"packing": [[0]], "bin_weights": [weights[0]]}
    
    # Compute lower bound (L1)
    total_weight = sum(weights)
    lb = (total_weight + bin_capacity - 1) // bin_capacity
    
    def make_result(ind):
        bins = ind['bins']
        bin_weights = [sum(weights[i] for i in b) for b in bins]
        return {"packing": bins, "bin_weights": bin_weights}
    
    def calc_fitness2(ind):
        """Secondary fitness: sum of (load/capacity)^2, higher is better."""
        s = 0.0
        cap2 = bin_capacity * bin_capacity
        for b_idx in range(len(ind['bins'])):
            load = bin_capacity - ind['remaining'][b_idx]
            s += load * load
        return s / (cap2 * len(ind['bins'])) if ind['bins'] else 0
    
    def is_better(a, b):
        if a['fitness'] != b['fitness']:
            return a['fitness'] < b['fitness']
        return a.get('fitness2', 0) > b.get('fitness2', 0)
    
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
        ind = {'bins': bins, 'remaining': remaining, 'fitness': len(bins)}
        ind['fitness2'] = calc_fitness2(ind)
        return ind
    
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
        ind = {'bins': bins, 'remaining': remaining, 'fitness': len(bins)}
        ind['fitness2'] = calc_fitness2(ind)
        return ind
    
    def copy_individual(ind):
        return {
            'bins': [b[:] for b in ind['bins']],
            'remaining': ind['remaining'][:],
            'fitness': ind['fitness'],
            'fitness2': ind.get('fitness2', 0)
        }
    
    def update_fitness2(ind):
        ind['fitness2'] = calc_fitness2(ind)
        return ind
    
    def cleanup_bin_elimination(ind):
        improved = True
        max_iters = 100
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
        update_fitness2(ind)
        return ind
    
    def deep_cleanup(ind, time_limit_abs):
        improved = True
        while improved:
            improved = False
            if time.time() >= time_limit_abs:
                break
            if len(ind['bins']) <= 1:
                break
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
        update_fitness2(ind)
        return ind
    
    def try_eliminate_bin(ind, target_b):
        """Try to eliminate a specific bin by distributing its items."""
        items = ind['bins'][target_b][:]
        test_bins = [b[:] for i, b in enumerate(ind['bins']) if i != target_b]
        test_remaining = [r for i, r in enumerate(ind['remaining']) if i != target_b]
        
        # Try all permutations of items (for small bins) or BFD
        sorted_items = sorted(items, key=lambda i: -weights[i])
        
        # Check if total weight can fit in remaining capacity
        items_weight = sum(weights[i] for i in items)
        total_remaining = sum(test_remaining)
        if items_weight > total_remaining:
            return None
        
        # Try BFD insertion
        bfd_insert(test_bins, test_remaining, sorted_items)
        if len(test_bins) < ind['fitness']:
            new_ind = {'bins': test_bins, 'remaining': test_remaining, 'fitness': len(test_bins)}
            update_fitness2(new_ind)
            return new_ind
        return None
    
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
        update_fitness2(ind)
        return ind
    
    def mutation_bin_elimination_random(ind):
        if len(ind['bins']) <= 1:
            return ind
        ind = copy_individual(ind)
        nb = len(ind['bins'])
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
        random.shuffle(items)
        bfd_insert(ind['bins'], ind['remaining'], items)
        ind['fitness'] = len(ind['bins'])
        if ind['fitness'] > old_count:
            return None
        update_fitness2(ind)
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
        update_fitness2(ind)
        return ind
    
    def mutation_item_swap(ind):
        if len(ind['bins']) < 2:
            return ind
        ind = copy_individual(ind)
        nb = len(ind['bins'])
        b1 = random.randint(0, nb - 1)
        b2 = random.randint(0, nb - 2)
        if b2 >= b1:
            b2 += 1
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
            update_fitness2(ind)
        return ind
    
    def mutation_multi_swap(ind):
        if len(ind['bins']) < 2:
            return ind
        ind = copy_individual(ind)
        nb = len(ind['bins'])
        b1 = random.randint(0, nb - 1)
        b2 = random.randint(0, nb - 2)
        if b2 >= b1:
            b2 += 1
        if len(ind['bins'][b1]) < 1 or len(ind['bins'][b2]) < 2:
            return ind
        i1 = random.randint(0, len(ind['bins'][b1]) - 1)
        item1 = ind['bins'][b1][i1]
        w1 = weights[item1]
        indices = random.sample(range(len(ind['bins'][b2])), 2)
        items2 = [ind['bins'][b2][j] for j in indices]
        w2_total = sum(weights[it] for it in items2)
        new_rem_b1 = ind['remaining'][b1] + w1 - w2_total
        new_rem_b2 = ind['remaining'][b2] + w2_total - w1
        if new_rem_b1 >= 0 and new_rem_b2 >= 0:
            ind['bins'][b1].pop(i1)
            ind['remaining'][b1] += w1
            for j in sorted(indices, reverse=True):
                ind['bins'][b2].pop(j)
            ind['remaining'][b2] += w2_total
            ind['bins'][b1].extend(items2)
            ind['remaining'][b1] -= w2_total
            ind['bins'][b2].append(item1)
            ind['remaining'][b2] -= w1
            update_fitness2(ind)
        return ind
    
    def mutation_bin_merge(ind):
        if len(ind['bins']) < 2:
            return ind
        ind = copy_individual(ind)
        bin_order = sorted(range(len(ind['bins'])), key=lambda b: -ind['remaining'][b])
        b1 = bin_order[0]
        b2 = bin_order[1]
        total_w = (bin_capacity - ind['remaining'][b1]) + (bin_capacity - ind['remaining'][b2])
        if total_w <= bin_capacity:
            ind['bins'][b1].extend(ind['bins'][b2])
            ind['remaining'][b1] = bin_capacity - total_w
            ind['bins'].pop(b2)
            ind['remaining'].pop(b2)
            ind['fitness'] = len(ind['bins'])
            update_fitness2(ind)
        else:
            smaller = b1 if (bin_capacity - ind['remaining'][b1]) <= (bin_capacity - ind['remaining'][b2]) else b2
            items = ind['bins'].pop(smaller)
            ind['remaining'].pop(smaller)
            bfd_insert(ind['bins'], ind['remaining'], items)
            ind['fitness'] = len(ind['bins'])
            update_fitness2(ind)
        return ind
    
    def mutation_redistribute_two(ind):
        """Take two least-full bins, merge items, redistribute via BFD."""
        if len(ind['bins']) < 3:
            return ind
        ind = copy_individual(ind)
        nb = len(ind['bins'])
        bin_order = sorted(range(nb), key=lambda b: -ind['remaining'][b])
        b1, b2 = bin_order[0], bin_order[1]
        items = ind['bins'][b1][:] + ind['bins'][b2][:]
        # Remove both bins (remove larger index first)
        for b in sorted([b1, b2], reverse=True):
            ind['bins'].pop(b)
            ind['remaining'].pop(b)
        old_fitness = ind['fitness']
        random.shuffle(items)
        bfd_insert(ind['bins'], ind['remaining'], items)
        ind['fitness'] = len(ind['bins'])
        if ind['fitness'] > old_fitness:
            return None
        update_fitness2(ind)
        return ind
    
    def mutate(ind):
        r = random.random()
        if r < 0.25:
            result = mutation_bin_elimination(ind)
            if result is None:
                return copy_individual(ind)
            return result
        elif r < 0.38:
            result = mutation_bin_elimination_random(ind)
            if result is None:
                return copy_individual(ind)
            return result
        elif r < 0.48:
            result = mutation_redistribute_two(ind)
            if result is None:
                return copy_individual(ind)
            return result
        elif r < 0.62:
            return mutation_item_move(ind)
        elif r < 0.76:
            return mutation_item_swap(ind)
        elif r < 0.88:
            return mutation_multi_swap(ind)
        else:
            return mutation_bin_merge(ind)
    
    def crossover_gga(p1, p2):
        """Grouping GA crossover: take some bins from p1, fill rest from p2."""
        nb1 = len(p1['bins'])
        nb2 = len(p2['bins'])
        
        if nb1 == 0 or nb2 == 0:
            return copy_individual(p1)
        
        # Select subset of bins from p1 (prefer fuller bins)
        k = random.randint(1, max(1, nb1 // 2))
        fullness = [(bin_capacity - p1['remaining'][b], b) for b in range(nb1)]
        fullness.sort(reverse=True)
        top_count = max(1, nb1 * 3 // 4)
        selected = set()
        for _ in range(k):
            if random.random() < 0.85 and top_count > 0:
                idx = fullness[random.randint(0, top_count - 1)][1]
            else:
                idx = random.randint(0, nb1 - 1)
            selected.add(idx)
        
        transferred = set()
        child_bins = []
        child_remaining = []
        for b in selected:
            child_bins.append(p1['bins'][b][:])
            child_remaining.append(p1['remaining'][b])
            for item in p1['bins'][b]:
                transferred.add(item)
        
        # Add non-conflicting bins from p2
        for b in range(nb2):
            new_bin = [item for item in p2['bins'][b] if item not in transferred]
            if new_bin:
                rem = bin_capacity - sum(weights[i] for i in new_bin)
                child_bins.append(new_bin)
                child_remaining.append(rem)
                for item in new_bin:
                    transferred.add(item)
        
        # Place orphans
        orphans = [i for i in range(n) if i not in transferred]
        if orphans:
            bfd_insert(child_bins, child_remaining, orphans)
        
        # Remove empty bins
        non_empty = [(child_bins[i], child_remaining[i]) for i in range(len(child_bins)) if child_bins[i]]
        if non_empty:
            child_bins = [list(b) for b, _ in non_empty]
            child_remaining = [r for _, r in non_empty]
        
        c = {'bins': child_bins, 'remaining': child_remaining, 'fitness': len(child_bins)}
        update_fitness2(c)
        return c
    
    def tournament_select(pop, k):
        best = None
        best_f = float('inf')
        best_f2 = -1
        for _ in range(k):
            i = random.randint(0, len(pop) - 1)
            f = pop[i]['fitness']
            f2 = pop[i].get('fitness2', 0)
            if f < best_f or (f == best_f and f2 > best_f2):
                best_f = f
                best_f2 = f2
                best = i
        return best
    
    # Population initialization
    if n <= 50:
        pop_size = 100
    elif n <= 200:
        pop_size = 70
    elif n <= 500:
        pop_size = 50
    else:
        pop_size = 40
    
    tournament_size = 4
    crossover_rate = 0.9
    mutation_rate = 0.7
    elite_count = max(2, pop_size * 15 // 100)
    
    population = []
    
    ffd_perm = sorted(range(n), key=lambda i: -weights[i])
    
    # FFD
    ind_ffd = decode_ffd(ffd_perm)
    cleanup_bin_elimination(ind_ffd)
    population.append(ind_ffd)
    
    # BFD
    ind_bfd = decode_bfd(ffd_perm)
    cleanup_bin_elimination(ind_bfd)
    population.append(ind_bfd)
    
    # Perturbed variants
    for pct in [3, 5, 8, 10, 15, 20, 25, 30]:
        perm = ffd_perm[:]
        num_swaps = max(1, n * pct // 100)
        for __ in range(num_swaps):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            perm[a], perm[b] = perm[b], perm[a]
        if random.random() < 0.5:
            ind = decode_bfd(perm)
        else:
            ind = decode_ffd(perm)
        cleanup_bin_elimination(ind)
        population.append(ind)
    
    # Fill rest with random-order BFD/FFD
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
        if ind['fitness'] < best_fitness or (ind['fitness'] == best_fitness and ind.get('fitness2', 0) > best.get('fitness2', 0) if best else True):
            best_fitness = ind['fitness']
            best = copy_individual(ind)
    
    if best_fitness <= lb:
        return make_result(best)
    
    # Main GA loop
    generation = 0
    stagnation = 0
    max_stagnation = 30
    total_stagnation = 0
    max_total_stagnation = 100
    time_limit_abs = start_time + time_limit * 0.92
    
    while True:
        if time.time() >= time_limit_abs:
            break
        
        generation += 1
        prev_best = best_fitness
        
        population.sort(key=lambda x: (x['fitness'], -x.get('fitness2', 0)))
        
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
                c1 = crossover_gga(population[p1_idx], population[p2_idx])
                c2 = crossover_gga(population[p2_idx], population[p1_idx])
            else:
                c1 = copy_individual(population[p1_idx])
                c2 = copy_individual(population[p2_idx])
            
            num_mut = 1 if random.random() < 0.6 else (2 if random.random() < 0.7 else 3)
            for _ in range(num_mut):
                if random.random() < mutation_rate:
                    c1 = mutate(c1)
            for _ in range(num_mut):
                if random.random() < mutation_rate:
                    c2 = mutate(c2)
            
            new_population.append(c1)
            if c1['fitness'] < best_fitness or (c1['fitness'] == best_fitness and c1.get('fitness2', 0) > best.get('fitness2', 0)):
                if c1['fitness'] < best_fitness:
                    best_fitness = c1['fitness']
                best = copy_individual(c1)
                if best_fitness <= lb:
                    return make_result(best)
            
            if len(new_population) < pop_size:
                new_population.append(c2)
                if c2['fitness'] < best_fitness or (c2['fitness'] == best_fitness and c2.get('fitness2', 0) > best.get('fitness2', 0)):
                    if c2['fitness'] < best_fitness:
                        best_fitness = c2['fitness']
                    best = copy_individual(c2)
                    if best_fitness <= lb:
                        return make_result(best)
        
        population = new_population
        
        # Periodic cleanup on top individuals
        if generation % 4 == 0:
            for i in range(min(5, len(population))):
                cleanup_bin_elimination(population[i])
                if population[i]['fitness'] < best_fitness:
                    best_fitness = population[i]['fitness']
                    best = copy_individual(population[i])
                    if best_fitness <= lb:
                        return make_result(best)
        
        # Aggressive attempts on best
        for _ in range(8):
            trial = mutation_bin_elimination(best)
            if trial is not None and trial['fitness'] < best_fitness:
                best_fitness = trial['fitness']
                best = trial
                if best_fitness <= lb:
                    return make_result(best)
        
        # Also try random bin elimination
        for _ in range(4):
            trial = mutation_bin_elimination_random(best)
            if trial is not None and trial['fitness'] < best_fitness:
                best_fitness = trial['fitness']
                best = trial
                if best_fitness <= lb:
                    return make_result(best)
        
        population[0] = copy_individual(best)
        
        if best_fitness < prev_best:
            stagnation = 0
            total_stagnation = 0
        else:
            stagnation += 1
            total_stagnation += 1
        
        if stagnation >= max_stagnation:
            stagnation = 0
            population.sort(key=lambda x: (x['fitness'], -x.get('fitness2', 0)))
            keep = max(3, elite_count)
            for i in range(keep, len(population)):
                r = random.random()
                if r < 0.35:
                    perm = list(range(n))
                    random.shuffle(perm)
                    ind = decode_bfd(perm)
                elif r < 0.6:
                    ind = copy_individual(best)
                    num_moves = max(1, n // 3)
                    for _ in range(num_moves):
                        ind = mutation_item_move(ind)
                elif r < 0.8:
                    all_items = []
                    for b in best['bins']:
                        all_items.extend(b)
                    random.shuffle(all_items)
                    ind = decode_bfd(all_items)
                else:
                    perm = list(range(n))
                    random.shuffle(perm)
                    ind = decode_ffd(perm)
                cleanup_bin_elimination(ind)
                population[i] = ind
                if ind['fitness'] < best_fitness:
                    best_fitness = ind['fitness']
                    best = copy_individual(ind)
                    if best_fitness <= lb:
                        return make_result(best)
        
        if total_stagnation >= max_total_stagnation:
            total_stagnation = 0
            population.sort(key=lambda x: (x['fitness'], -x.get('fitness2', 0)))
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
                    if best_fitness <= lb:
                        return make_result(best)
    
    # Final post-processing
    time_limit_final = start_time + time_limit * 0.99
    
    # Deep cleanup on best
    deep_cleanup(best, time_limit_final)
    if best['fitness'] <= lb:
        return make_result(best)
    
    # Try exhaustive bin elimination: for each bin, try to eliminate it
    while time.time() < time_limit_final:
        if len(best['bins']) <= 1:
            break
        improved = False
        nb = len(best['bins'])
        bin_order = sorted(range(nb), key=lambda b: -best['remaining'][b])
        
        for target_b in bin_order:
            if time.time() >= time_limit_final:
                break
            items = best['bins'][target_b][:]
            items_weight = sum(weights[i] for i in items)
            test_bins = [b[:] for i, b in enumerate(best['bins']) if i != target_b]
            test_remaining = [r for i, r in enumerate(best['remaining']) if i != target_b]
            total_rem = sum(test_remaining)
            if items_weight > total_rem:
                continue
            
            # Try multiple orderings
            for attempt in range(min(50, max(5, 200 // nb))):
                if time.time() >= time_limit_final:
                    break
                tb = [b[:] for b in test_bins]
                tr = test_remaining[:]
                if attempt == 0:
                    sorted_items = sorted(items, key=lambda i: -weights[i])
                else:
                    sorted_items = items[:]
                    random.shuffle(sorted_items)
                bfd_insert(tb, tr, sorted_items)
                if len(tb) < best['fitness']:
                    best['bins'] = tb
                    best['remaining'] = tr
                    best['fitness'] = len(tb)
                    update_fitness2(best)
                    improved = True
                    break
        
        if not improved:
            break
    
    # Try removing pairs of least-full bins and reinserting
    while time.time() < time_limit_final:
        if len(best['bins']) <= 2:
            break
        nb = len(best['bins'])
        bin_order = sorted(range(nb), key=lambda b: -best['remaining'][b])
        improved = False
        for k in range(2, min(4, nb)):
            if time.time() >= time_limit_final:
                break
            bins_to_remove = set(bin_order[:k])
            all_items_to_reinsert = []
            for b in bins_to_remove:
                all_items_to_reinsert.extend(best['bins'][b])
            items_weight = sum(weights[i] for i in all_items_to_reinsert)
            new_bins = []
            new_remaining = []
            for b in range(nb):
                if b not in bins_to_remove:
                    new_bins.append(best['bins'][b][:])
                    new_remaining.append(best['remaining'][b])
            total_rem = sum(new_remaining)
            if items_weight > total_rem:
                continue
            for _ in range(min(30, max(5, 200 // k))):
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
                    update_fitness2(best)
                    improved = True
                    break
        if not improved:
            break
    
    return make_result(best)