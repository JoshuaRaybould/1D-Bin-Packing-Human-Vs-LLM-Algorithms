import random
import time
import math

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    if n == 1:
        return {"packing": [[0]], "bin_weights": [weights[0]]}
    
    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity
    
    w = weights
    C = bin_capacity
    
    # Optimized Best Fit Decreasing decode
    def decode_bfd(perm):
        bin_loads = []
        bin_items = []
        
        for idx in perm:
            wi = w[idx]
            best_b = -1
            best_remaining = C + 1
            
            for b in range(len(bin_loads)):
                rem = C - bin_loads[b]
                if wi <= rem < best_remaining:
                    best_remaining = rem
                    best_b = b
                    if rem == wi:
                        break
            
            if best_b >= 0:
                bin_items[best_b].append(idx)
                bin_loads[best_b] += wi
            else:
                bin_items.append([idx])
                bin_loads.append(wi)
        
        return bin_items, bin_loads
    
    # First Fit Decreasing decode
    def decode_ffd(perm):
        bin_loads = []
        bin_items = []
        
        for idx in perm:
            wi = w[idx]
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
    
    # Decode with both BFD and FFD, return best
    def decode_best(perm):
        b1, l1 = decode_bfd(perm)
        b2, l2 = decode_ffd(perm)
        if len(b2) < len(b1):
            return b2, l2
        elif len(b2) == len(b1):
            # prefer higher fill variance (more full bins)
            s1 = sum(x*x*x for x in l1)
            s2 = sum(x*x*x for x in l2)
            if s2 > s1:
                return b2, l2
        return b1, l1
    
    C3 = C * C * C
    
    def fitness(perm):
        bins, bin_loads = decode_best(perm)
        num_bins = len(bins)
        if num_bins == 0:
            return 0.0, bins, bin_loads
        fill_score = sum(load * load * load for load in bin_loads) / (C3 * num_bins)
        return num_bins - fill_score, bins, bin_loads
    
    def generate_ffd_perm():
        return sorted(range(n), key=lambda i: -w[i])
    
    def generate_random_perm():
        perm = list(range(n))
        random.shuffle(perm)
        return perm
    
    def generate_semi_random_perm():
        perm = sorted(range(n), key=lambda i: -w[i])
        num_swaps = random.randint(1, max(1, n // 4))
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def generate_perturbed_best(base_perm, num_swaps):
        perm = base_perm[:]
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def generate_ffd_perturbed():
        perm = sorted(range(n), key=lambda i: -w[i])
        num_reversals = random.randint(2, 5)
        for _ in range(num_reversals):
            if n < 3:
                break
            seg_len = random.randint(2, max(2, n // 5))
            start = random.randint(0, n - seg_len)
            perm[start:start+seg_len] = reversed(perm[start:start+seg_len])
        return perm
    
    # Generate permutation by grouping items that fill bins well
    def generate_group_perm():
        # Sort items by weight descending
        items = sorted(range(n), key=lambda i: -w[i])
        result = []
        used = [False] * n
        
        for idx in items:
            if used[idx]:
                continue
            used[idx] = True
            result.append(idx)
            remaining = C - w[idx]
            
            # Try to find items that fill remaining capacity
            for idx2 in items:
                if not used[idx2] and w[idx2] <= remaining:
                    used[idx2] = True
                    result.append(idx2)
                    remaining -= w[idx2]
                    if remaining == 0:
                        break
        
        return result
    
    # Try to improve a packing directly by moving items from small bins
    def improve_packing(bins, bin_loads):
        improved = True
        while improved:
            improved = False
            num_bins = len(bins)
            if num_bins <= lower_bound:
                break
            
            # Find the bin with smallest load
            min_b = min(range(num_bins), key=lambda b: bin_loads[b])
            items_to_move = bins[min_b][:]
            
            # Try to redistribute all items from this bin to others
            can_move_all = True
            moves = []  # (item_idx, target_bin)
            temp_loads = bin_loads[:]
            
            for item_idx in sorted(items_to_move, key=lambda x: -w[x]):
                wi = w[item_idx]
                best_target = -1
                best_rem = C + 1
                
                for b in range(num_bins):
                    if b == min_b:
                        continue
                    rem = C - temp_loads[b]
                    if wi <= rem < best_rem:
                        best_rem = rem
                        best_target = b
                        if rem == wi:
                            break
                
                if best_target >= 0:
                    moves.append((item_idx, best_target))
                    temp_loads[best_target] += wi
                else:
                    can_move_all = False
                    break
            
            if can_move_all:
                # Apply moves
                for item_idx, target_b in moves:
                    bins[target_b].append(item_idx)
                    bin_loads[target_b] += w[item_idx]
                
                # Remove the emptied bin
                bins.pop(min_b)
                bin_loads.pop(min_b)
                improved = True
            else:
                break
        
        return bins, bin_loads
    
    # Population size - smaller for faster iterations
    pop_size = min(20, max(8, n // 3))
    
    population = []
    fitnesses_list = []
    all_bins = []
    all_bin_loads = []
    
    def add_agent(perm):
        f, b, bl = fitness(perm)
        # Try improving packing directly
        b, bl = improve_packing(b, bl)
        num_bins = len(b)
        if num_bins > 0:
            fill_score = sum(load * load * load for load in bl) / (C3 * num_bins)
            f = num_bins - fill_score
        population.append(perm)
        fitnesses_list.append(f)
        all_bins.append(b)
        all_bin_loads.append(bl)
        return f, b, bl
    
    # Agent 0: FFD
    ffd_perm = generate_ffd_perm()
    add_agent(ffd_perm)
    
    # Agent 1: group perm
    add_agent(generate_group_perm())
    
    # Agents 2-4: FFD with segment reversals
    for i in range(2, min(5, pop_size)):
        if time.time() - start_time > time_limit * 0.7:
            break
        add_agent(generate_ffd_perturbed())
    
    # Semi-random agents
    third = max(5, pop_size // 3)
    for i in range(len(population), min(third, pop_size)):
        if time.time() - start_time > time_limit * 0.7:
            break
        add_agent(generate_semi_random_perm())
    
    # Random agents
    for i in range(len(population), pop_size):
        if time.time() - start_time > time_limit * 0.7:
            break
        add_agent(generate_random_perm())
    
    pop_size = len(population)
    
    best_idx = min(range(pop_size), key=lambda i: fitnesses_list[i])
    best_fitness = fitnesses_list[best_idx]
    best_perm = population[best_idx][:]
    best_bins = [b[:] for b in all_bins[best_idx]]
    best_bin_loads = all_bin_loads[best_idx][:]
    
    if len(best_bins) <= lower_bound:
        return {"packing": best_bins, "bin_weights": best_bin_loads}
    
    iteration = 0
    last_improvement_iter = 0
    stagnation_threshold = 30
    
    def evaluate_and_improve(perm):
        f, b, bl = fitness(perm)
        b, bl = improve_packing(b, bl)
        num_bins = len(b)
        if num_bins > 0:
            fill_score = sum(load * load * load for load in bl) / (C3 * num_bins)
            f = num_bins - fill_score
        return f, b, bl
    
    while time.time() - start_time < time_limit * 0.95:
        iteration += 1
        
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_limit)
        
        curr_best_idx = min(range(pop_size), key=lambda i: fitnesses_list[i])
        curr_best_perm = population[curr_best_idx]
        curr_best_fitness = fitnesses_list[curr_best_idx]
        
        worst_fitness = max(fitnesses_list)
        fitness_range = worst_fitness - curr_best_fitness if worst_fitness != curr_best_fitness else 1.0
        
        for i in range(pop_size):
            if time.time() - start_time > time_limit * 0.95:
                break
            
            if i == curr_best_idx:
                continue
            
            fi = fitnesses_list[i]
            
            if fitness_range > 0:
                normalized_fi = (fi - curr_best_fitness) / fitness_range
            else:
                normalized_fi = 0.0
            
            if curr_best_fitness > 0:
                weight = fi / curr_best_fitness
            else:
                weight = 1.0 + normalized_fi
            
            new_perm = population[i][:]
            
            r = random.random()
            threshold = 1.0 + (0.5 + progress) * r
            
            if weight > threshold:  # Scout - explore
                rand_val = random.random()
                
                if rand_val < 0.3 and n >= 3:
                    # Segment reversal
                    seg_len = max(2, int(n * (0.1 + normalized_fi * 0.3)))
                    seg_len = min(seg_len, n)
                    start_pos = random.randint(0, n - seg_len)
                    new_perm[start_pos:start_pos+seg_len] = reversed(new_perm[start_pos:start_pos+seg_len])
                elif rand_val < 0.6 and n >= 3:
                    # Block insertion (or-opt style)
                    block_len = max(1, int(n * (0.05 + normalized_fi * 0.15)))
                    block_len = min(block_len, n - 1)
                    start_pos = random.randint(0, n - block_len)
                    block = new_perm[start_pos:start_pos+block_len]
                    del new_perm[start_pos:start_pos+block_len]
                    insert_pos = random.randint(0, len(new_perm))
                    for k, item in enumerate(block):
                        new_perm.insert(insert_pos + k, item)
                elif rand_val < 0.8:
                    # Random swaps
                    num_changes = max(1, int(n * normalized_fi * 0.4))
                    for _ in range(num_changes):
                        a = random.randint(0, n - 1)
                        b_idx = random.randint(0, n - 1)
                        new_perm[a], new_perm[b_idx] = new_perm[b_idx], new_perm[a]
                else:
                    # Generate fresh from FFD perturbed or semi-random
                    if random.random() < 0.5:
                        new_perm = generate_ffd_perturbed()
                    else:
                        new_perm = generate_semi_random_perm()
            else:  # Forager - exploit toward best
                rand_val = random.random()
                
                if rand_val < 0.6:
                    # Order Crossover (OX) with best
                    copy_rate = max(0.1, min(0.6, normalized_fi * 0.7))
                    seg_len = max(1, int(n * copy_rate))
                    
                    start_pos = random.randint(0, n - seg_len)
                    end_pos = start_pos + seg_len
                    
                    segment = curr_best_perm[start_pos:end_pos]
                    segment_set = set(segment)
                    
                    result = [None] * n
                    for k in range(start_pos, end_pos):
                        result[k] = curr_best_perm[k]
                    
                    remaining = [x for x in new_perm if x not in segment_set]
                    ri = 0
                    for j in range(n):
                        if result[j] is None:
                            result[j] = remaining[ri]
                            ri += 1
                    
                    new_perm = result
                elif rand_val < 0.85:
                    # Position-based crossover with global best
                    positions = random.sample(range(n), max(1, int(n * 0.3)))
                    pos_set = set(positions)
                    result = [None] * n
                    taken = set()
                    for p in positions:
                        result[p] = best_perm[p]
                        taken.add(best_perm[p])
                    remaining = [x for x in new_perm if x not in taken]
                    ri = 0
                    for j in range(n):
                        if result[j] is None:
                            result[j] = remaining[ri]
                            ri += 1
                    new_perm = result
                else:
                    # Move toward best with adjacent swaps to match positions
                    num_fixes = max(1, int(n * 0.15))
                    pos_map = {v: k for k, v in enumerate(new_perm)}
                    for _ in range(num_fixes):
                        target_pos = random.randint(0, n - 1)
                        target_val = curr_best_perm[target_pos]
                        curr_pos = pos_map[target_val]
                        if curr_pos != target_pos:
                            other_val = new_perm[target_pos]
                            new_perm[target_pos], new_perm[curr_pos] = new_perm[curr_pos], new_perm[target_pos]
                            pos_map[target_val] = target_pos
                            pos_map[other_val] = curr_pos
                
                # Small perturbation
                num_micro = random.randint(1, max(1, min(3, n // 10)))
                for _ in range(num_micro):
                    pos = random.randint(0, n - 2)
                    new_perm[pos], new_perm[pos+1] = new_perm[pos+1], new_perm[pos]
            
            new_f, new_b, new_bl = evaluate_and_improve(new_perm)
            
            if new_f < fitnesses_list[i]:
                population[i] = new_perm
                fitnesses_list[i] = new_f
                all_bins[i] = new_b
                all_bin_loads[i] = new_bl
                
                if new_f < best_fitness:
                    best_fitness = new_f
                    best_perm = new_perm[:]
                    best_bins = [b[:] for b in new_b]
                    best_bin_loads = new_bl[:]
                    last_improvement_iter = iteration
                    
                    if len(best_bins) <= lower_bound:
                        return {"packing": best_bins, "bin_weights": best_bin_loads}
            else:
                accept_prob = 0.03 * (1.0 - progress)
                if random.random() < accept_prob:
                    population[i] = new_perm
                    fitnesses_list[i] = new_f
                    all_bins[i] = new_b
                    all_bin_loads[i] = new_bl
        
        # Stagnation handling
        if iteration - last_improvement_iter > stagnation_threshold:
            last_improvement_iter = iteration
            
            sorted_indices = sorted(range(pop_size), key=lambda idx: fitnesses_list[idx])
            num_reinit = max(1, int(pop_size * 0.4))
            reinit_indices = sorted_indices[-num_reinit:]
            
            for k, idx in enumerate(reinit_indices):
                if time.time() - start_time > time_limit * 0.95:
                    break
                
                r = random.random()
                if r < 0.4:
                    num_swaps = random.randint(3, min(20, n))
                    perm = generate_perturbed_best(best_perm, num_swaps)
                elif r < 0.7:
                    perm = generate_ffd_perturbed()
                elif r < 0.85:
                    perm = generate_group_perm()
                    # perturb it slightly
                    ns = random.randint(1, max(1, n // 8))
                    for _ in range(ns):
                        a = random.randint(0, n-1)
                        b2 = random.randint(0, n-1)
                        perm[a], perm[b2] = perm[b2], perm[a]
                else:
                    perm = generate_semi_random_perm()
                
                f, b, bl = evaluate_and_improve(perm)
                population[idx] = perm
                fitnesses_list[idx] = f
                all_bins[idx] = b
                all_bin_loads[idx] = bl
                
                if f < best_fitness:
                    best_fitness = f
                    best_perm = perm[:]
                    best_bins = [bb[:] for bb in b]
                    best_bin_loads = bl[:]
                    
                    if len(best_bins) <= lower_bound:
                        return {"packing": best_bins, "bin_weights": best_bin_loads}
            
            # Elitism
            if reinit_indices:
                elitism_idx = reinit_indices[0]
                perm = best_perm[:]
                f, b, bl = evaluate_and_improve(perm)
                population[elitism_idx] = perm
                fitnesses_list[elitism_idx] = f
                all_bins[elitism_idx] = b
                all_bin_loads[elitism_idx] = bl
    
    return {"packing": best_bins, "bin_weights": best_bin_loads}