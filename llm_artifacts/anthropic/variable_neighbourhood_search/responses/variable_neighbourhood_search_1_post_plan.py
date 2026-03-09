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
    
    C = bin_capacity
    W = weights  # local reference for speed
    
    # --- Lower Bound ---
    total_weight = sum(W)
    L1 = math.ceil(total_weight / C)
    
    # L2 bound (Martello-Toth style)
    def compute_L2():
        best_lb = L1
        # Sort weights for efficient counting
        sw = sorted(W, reverse=True)
        ns = len(sw)
        
        # For each alpha from 1 to C//2
        # Items > C - alpha are "large"
        # Items in (alpha, C - alpha] are "medium"
        # Items <= alpha are "small"
        # We iterate over distinct thresholds
        
        for alpha in range(1, C // 2 + 1):
            threshold_large = C - alpha
            n_large = 0
            n_medium = 0
            sum_medium = 0
            sum_small = 0
            
            for w in sw:
                if w > threshold_large:
                    n_large += 1
                elif w > alpha:
                    n_medium += 1
                    sum_medium += w
                else:
                    sum_small += w
            
            # Each large item needs its own bin at minimum
            # Medium items can share with large if they fit
            # Remaining medium items pair up
            # Small items fill remaining space
            
            # Residual capacity in large-item bins
            # This is a simplified L2: n_large + ceil((sum_medium + sum_small - residual_in_large_bins) / C)
            # But the standard L2 is:
            # L2(alpha) = n_large + n_medium + max(0, ceil((sum_small - (n_large*C - n_large*(C-alpha+1)... )) / C))
            # Let me use the simpler but effective version:
            # L2(alpha) = n_large + ceil((sum_medium + sum_small) / C)
            
            lb = n_large + max(0, math.ceil((sum_medium + sum_small) / C))
            if lb > best_lb:
                best_lb = lb
        
        return best_lb
    
    lower_bound = compute_L2()
    
    # --- Helper functions ---
    def validate_and_clean(bins, bin_rem):
        new_bins = []
        new_rem = []
        for b in range(len(bins)):
            if len(bins[b]) > 0:
                new_bins.append(bins[b])
                new_rem.append(bin_rem[b])
        return new_bins, new_rem
    
    def copy_solution(bins, bin_rem):
        return [lst[:] for lst in bins], bin_rem[:]
    
    def solution_cost(bins):
        return len(bins)
    
    def sum_of_squares(bins, bin_rem):
        s = 0
        for b in range(len(bins)):
            load = C - bin_rem[b]
            s += load * load
        return s
    
    def is_better(cost1, sos1, cost2, sos2):
        """Returns True if solution 1 is strictly better than solution 2."""
        if cost1 < cost2:
            return True
        if cost1 == cost2 and sos1 > sos2:
            return True
        return False
    
    def is_better_or_equal(cost1, sos1, cost2, sos2):
        if cost1 < cost2:
            return True
        if cost1 == cost2 and sos1 >= sos2:
            return True
        return False
    
    # --- Initial Solutions ---
    def first_fit_decreasing(items_indices, cap):
        sorted_items = sorted(items_indices, key=lambda i: W[i], reverse=True)
        bins = []
        bin_rem = []
        for idx in sorted_items:
            w = W[idx]
            placed = False
            for b in range(len(bins)):
                if bin_rem[b] >= w:
                    bins[b].append(idx)
                    bin_rem[b] -= w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_rem.append(cap - w)
        return bins, bin_rem
    
    def best_fit_decreasing(items_indices, cap):
        sorted_items = sorted(items_indices, key=lambda i: W[i], reverse=True)
        bins = []
        bin_rem = []
        for idx in sorted_items:
            w = W[idx]
            best_b = -1
            best_r = cap + 1
            for b in range(len(bins)):
                if bin_rem[b] >= w and bin_rem[b] < best_r:
                    best_r = bin_rem[b]
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(idx)
                bin_rem[best_b] -= w
            else:
                bins.append([idx])
                bin_rem.append(cap - w)
        return bins, bin_rem
    
    all_items = list(range(n))
    
    bins_ffd, rem_ffd = first_fit_decreasing(all_items, C)
    bins_bfd, rem_bfd = best_fit_decreasing(all_items, C)
    
    # Pick the better initial solution
    cost_ffd = len(bins_ffd)
    cost_bfd = len(bins_bfd)
    sos_ffd = sum_of_squares(bins_ffd, rem_ffd)
    sos_bfd = sum_of_squares(bins_bfd, rem_bfd)
    
    if is_better(cost_bfd, sos_bfd, cost_ffd, sos_ffd):
        bins, bin_rem = bins_bfd, rem_bfd
    else:
        bins, bin_rem = bins_ffd, rem_ffd
    
    bins, bin_rem = validate_and_clean(bins, bin_rem)
    
    best_bins, best_rem = copy_solution(bins, bin_rem)
    best_cost = solution_cost(best_bins)
    best_sos = sum_of_squares(best_bins, best_rem)
    
    if best_cost <= lower_bound:
        packing = best_bins
        bin_weights = [sum(W[idx] for idx in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}
    
    # --- Enhanced Local Search ---
    def local_search(bins, bin_rem):
        deadline = start_time + time_limit * 0.98
        op_count = 0
        
        # Phase 1: Bin emptying
        improved = True
        while improved:
            improved = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            
            # Sort bins by load ascending
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            
            # Pre-compute total available capacity
            total_rem = sum(bin_rem)
            
            for src_b in bin_order:
                if src_b >= len(bins) or len(bins[src_b]) == 0:
                    continue
                
                src_load = C - bin_rem[src_b]
                # Quick feasibility: total remaining in other bins must >= src_load
                other_rem = total_rem - bin_rem[src_b]
                if other_rem < src_load:
                    continue
                
                items_to_move = sorted(bins[src_b], key=lambda i: W[i], reverse=True)
                temp_rem = bin_rem[:]
                assignments = {}
                can_empty = True
                
                for idx in items_to_move:
                    w = W[idx]
                    best_b = -1
                    best_r = C + 1
                    for b in range(len(bins)):
                        if b == src_b:
                            continue
                        if temp_rem[b] >= w and temp_rem[b] < best_r:
                            best_r = temp_rem[b]
                            best_b = b
                    if best_b >= 0:
                        assignments[idx] = best_b
                        temp_rem[best_b] -= w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    for idx, tgt_b in assignments.items():
                        bins[tgt_b].append(idx)
                        bin_rem[tgt_b] -= W[idx]
                    bins[src_b] = []
                    bin_rem[src_b] = C
                    bins, bin_rem = validate_and_clean(bins, bin_rem)
                    improved = True
                    break
                
                op_count += 1
                if op_count % 50 == 0 and time.time() > deadline:
                    return bins, bin_rem
        
        # Phase 2: Transfer moves (improve sum of squares)
        for _pass in range(2):
            did_improve = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            
            for src_b in range(num_bins):
                if len(bins[src_b]) == 0:
                    continue
                for item_pos in range(len(bins[src_b]) - 1, -1, -1):
                    idx = bins[src_b][item_pos]
                    w = W[idx]
                    src_load = C - bin_rem[src_b]
                    
                    # Current contribution to SOS from src_b
                    old_src_sq = src_load * src_load
                    new_src_load = src_load - w
                    new_src_sq = new_src_load * new_src_load
                    
                    best_gain = 0
                    best_tgt = -1
                    
                    for tgt_b in range(num_bins):
                        if tgt_b == src_b:
                            continue
                        if bin_rem[tgt_b] < w:
                            continue
                        tgt_load = C - bin_rem[tgt_b]
                        old_tgt_sq = tgt_load * tgt_load
                        new_tgt_load = tgt_load + w
                        new_tgt_sq = new_tgt_load * new_tgt_load
                        
                        gain = (new_src_sq + new_tgt_sq) - (old_src_sq + old_tgt_sq)
                        if gain > best_gain:
                            best_gain = gain
                            best_tgt = tgt_b
                    
                    if best_tgt >= 0:
                        bins[src_b].pop(item_pos)
                        bin_rem[src_b] += w
                        bins[best_tgt].append(idx)
                        bin_rem[best_tgt] -= w
                        did_improve = True
                    
                    op_count += 1
                    if op_count % 100 == 0 and time.time() > deadline:
                        bins, bin_rem = validate_and_clean(bins, bin_rem)
                        return bins, bin_rem
            
            if not did_improve:
                break
        
        bins, bin_rem = validate_and_clean(bins, bin_rem)
        
        # Phase 3: Swap moves
        for _pass in range(2):
            did_improve = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            
            for b1 in range(num_bins):
                found = False
                for b2 in range(b1 + 1, num_bins):
                    for i1_pos in range(len(bins[b1])):
                        item1 = bins[b1][i1_pos]
                        w1 = W[item1]
                        for i2_pos in range(len(bins[b2])):
                            item2 = bins[b2][i2_pos]
                            w2 = W[item2]
                            diff = w1 - w2
                            if bin_rem[b1] + diff >= 0 and bin_rem[b2] - diff >= 0:
                                load1 = C - bin_rem[b1]
                                load2 = C - bin_rem[b2]
                                old_sq = load1 * load1 + load2 * load2
                                new_load1 = load1 - w1 + w2
                                new_load2 = load2 - w2 + w1
                                new_sq = new_load1 * new_load1 + new_load2 * new_load2
                                if new_sq > old_sq:
                                    bins[b1][i1_pos] = item2
                                    bins[b2][i2_pos] = item1
                                    bin_rem[b1] += diff
                                    bin_rem[b2] -= diff
                                    did_improve = True
                                    found = True
                                    break
                            
                            op_count += 1
                            if op_count % 200 == 0 and time.time() > deadline:
                                return bins, bin_rem
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            
            if not did_improve:
                break
        
        # Phase 4: Try bin emptying again after transfers/swaps
        improved = True
        while improved:
            improved = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            total_rem = sum(bin_rem)
            
            for src_b in bin_order:
                if src_b >= len(bins) or len(bins[src_b]) == 0:
                    continue
                
                src_load = C - bin_rem[src_b]
                other_rem = total_rem - bin_rem[src_b]
                if other_rem < src_load:
                    continue
                
                items_to_move = sorted(bins[src_b], key=lambda i: W[i], reverse=True)
                temp_rem = bin_rem[:]
                assignments = {}
                can_empty = True
                
                for idx in items_to_move:
                    w = W[idx]
                    best_b = -1
                    best_r = C + 1
                    for b in range(len(bins)):
                        if b == src_b:
                            continue
                        if temp_rem[b] >= w and temp_rem[b] < best_r:
                            best_r = temp_rem[b]
                            best_b = b
                    if best_b >= 0:
                        assignments[idx] = best_b
                        temp_rem[best_b] -= w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    for idx, tgt_b in assignments.items():
                        bins[tgt_b].append(idx)
                        bin_rem[tgt_b] -= W[idx]
                    bins[src_b] = []
                    bin_rem[src_b] = C
                    bins, bin_rem = validate_and_clean(bins, bin_rem)
                    improved = True
                    break
                
                op_count += 1
                if op_count % 50 == 0 and time.time() > deadline:
                    return bins, bin_rem
        
        return bins, bin_rem
    
    # Apply local search to initial solution
    bins, bin_rem = local_search(bins, bin_rem)
    bins, bin_rem = validate_and_clean(bins, bin_rem)
    cost = solution_cost(bins)
    sos = sum_of_squares(bins, bin_rem)
    if is_better(cost, sos, best_cost, best_sos):
        best_bins, best_rem = copy_solution(bins, bin_rem)
        best_cost = cost
        best_sos = sos
    
    if best_cost <= lower_bound:
        packing = best_bins
        bin_weights = [sum(W[idx] for idx in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}
    
    # --- Randomized Repair ---
    def randomized_best_fit(freed_items, bins, bin_rem):
        """Repack freed_items into bins using randomized best-fit."""
        freed_items_sorted = sorted(freed_items, key=lambda i: W[i], reverse=True)
        for idx in freed_items_sorted:
            w = W[idx]
            # Find all feasible bins and their remaining capacities
            feasible = []
            for b in range(len(bins)):
                if bin_rem[b] >= w:
                    feasible.append((bin_rem[b], b))
            
            if not feasible:
                bins.append([idx])
                bin_rem.append(C - w)
                continue
            
            feasible.sort()
            
            # With probability 0.3, pick randomly among top-3 best fitting
            if random.random() < 0.3 and len(feasible) > 1:
                top_k = min(3, len(feasible))
                _, chosen_b = random.choice(feasible[:top_k])
            else:
                _, chosen_b = feasible[0]
            
            bins[chosen_b].append(idx)
            bin_rem[chosen_b] -= w
        
        return bins, bin_rem
    
    # --- Shaking ---
    k_max = 7
    
    def shaking(bins, bin_rem, k):
        bins = [lst[:] for lst in bins]
        bin_rem = bin_rem[:]
        num_bins = len(bins)
        
        if num_bins <= 1:
            return bins, bin_rem
        
        if k == 1:
            # Single random transfer
            src_b = random.randint(0, num_bins - 1)
            if len(bins[src_b]) == 0:
                return bins, bin_rem
            item_pos = random.randint(0, len(bins[src_b]) - 1)
            item = bins[src_b][item_pos]
            w = W[item]
            candidates = [b for b in range(num_bins) if b != src_b and bin_rem[b] >= w]
            if candidates:
                tgt_b = random.choice(candidates)
                bins[src_b].pop(item_pos)
                bin_rem[src_b] += w
                bins[tgt_b].append(item)
                bin_rem[tgt_b] -= w
        
        elif k == 2:
            # Random swap
            b1 = random.randint(0, num_bins - 1)
            b2 = random.randint(0, num_bins - 1)
            if b1 != b2 and len(bins[b1]) > 0 and len(bins[b2]) > 0:
                i1_pos = random.randint(0, len(bins[b1]) - 1)
                i2_pos = random.randint(0, len(bins[b2]) - 1)
                item1 = bins[b1][i1_pos]
                item2 = bins[b2][i2_pos]
                w1, w2 = W[item1], W[item2]
                diff = w1 - w2
                if bin_rem[b1] + diff >= 0 and bin_rem[b2] - diff >= 0:
                    bins[b1][i1_pos] = item2
                    bins[b2][i2_pos] = item1
                    bin_rem[b1] += diff
                    bin_rem[b2] -= diff
        
        elif k == 3:
            # Chain move: 2-4 random items
            num_moves = random.randint(2, 4)
            for _ in range(num_moves):
                if len(bins) <= 1:
                    break
                src_b = random.randint(0, len(bins) - 1)
                if len(bins[src_b]) == 0:
                    continue
                item_pos = random.randint(0, len(bins[src_b]) - 1)
                item = bins[src_b][item_pos]
                w = W[item]
                candidates = [b for b in range(len(bins)) if b != src_b and bin_rem[b] >= w]
                if candidates:
                    tgt_b = random.choice(candidates)
                    bins[src_b].pop(item_pos)
                    bin_rem[src_b] += w
                    bins[tgt_b].append(item)
                    bin_rem[tgt_b] -= w
        
        elif k == 4:
            # Destroy lightest 1-2 bins, repair with randomized BFD
            num_to_empty = min(random.randint(1, 2), num_bins - 1)
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            bins_to_empty = bin_order[:num_to_empty]
            freed_items = []
            for b in bins_to_empty:
                for item in bins[b]:
                    freed_items.append(item)
                    bin_rem[b] += W[item]
                bins[b] = []
            bins, bin_rem = randomized_best_fit(freed_items, bins, bin_rem)
        
        elif k == 5:
            # Destroy 2-4 random bins, repair with randomized BFD
            num_to_empty = min(random.randint(2, 4), num_bins - 1)
            if num_to_empty <= 0:
                return bins, bin_rem
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                for item in bins[b]:
                    freed_items.append(item)
                    bin_rem[b] += W[item]
                bins[b] = []
            bins, bin_rem = randomized_best_fit(freed_items, bins, bin_rem)
        
        elif k == 6:
            # Destroy 20-30% lightest bins, repair
            pct = random.uniform(0.2, 0.3)
            num_to_empty = max(1, min(int(num_bins * pct), num_bins - 1))
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            bins_to_empty = bin_order[:num_to_empty]
            freed_items = []
            for b in bins_to_empty:
                for item in bins[b]:
                    freed_items.append(item)
                    bin_rem[b] += W[item]
                bins[b] = []
            bins, bin_rem = randomized_best_fit(freed_items, bins, bin_rem)
        
        elif k == 7:
            # Large random perturbation: 40-50% random bins, first-fit repair
            pct = random.uniform(0.4, 0.5)
            num_to_empty = max(1, min(int(num_bins * pct), num_bins - 1))
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                for item in bins[b]:
                    freed_items.append(item)
                    bin_rem[b] += W[item]
                bins[b] = []
            # First-fit in random order for diversity
            random.shuffle(freed_items)
            for idx in freed_items:
                w = W[idx]
                placed = False
                for b in range(len(bins)):
                    if bin_rem[b] >= w:
                        bins[b].append(idx)
                        bin_rem[b] -= w
                        placed = True
                        break
                if not placed:
                    bins.append([idx])
                    bin_rem.append(C - w)
        
        bins, bin_rem = validate_and_clean(bins, bin_rem)
        return bins, bin_rem
    
    # --- VNS Main Loop ---
    k = 1
    max_iterations = 1000000
    no_improve_count = 0
    cur_cost = solution_cost(bins)
    cur_sos = sum_of_squares(bins, bin_rem)
    deadline = start_time + time_limit * 0.98
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        if best_cost <= lower_bound:
            break
        
        # Shaking
        new_bins, new_rem = shaking(bins, bin_rem, k)
        
        # Local search
        new_bins, new_rem = local_search(new_bins, new_rem)
        new_bins, new_rem = validate_and_clean(new_bins, new_rem)
        new_cost = solution_cost(new_bins)
        new_sos = sum_of_squares(new_bins, new_rem)
        
        # Move or not
        if is_better_or_equal(new_cost, new_sos, cur_cost, cur_sos):
            bins, bin_rem = new_bins, new_rem
            cur_cost = new_cost
            cur_sos = new_sos
            k = 1
            no_improve_count = 0
            
            if is_better(new_cost, new_sos, best_cost, best_sos):
                best_bins, best_rem = copy_solution(bins, bin_rem)
                best_cost = new_cost
                best_sos = new_sos
                
                if best_cost <= lower_bound:
                    break
        else:
            k += 1
            no_improve_count += 1
            if k > k_max:
                k = 1
        
        # Restart if stuck
        if no_improve_count > k_max * 15:
            bins, bin_rem = copy_solution(best_bins, best_rem)
            num_bins = len(bins)
            if num_bins > 2:
                # Large perturbation: empty 30-50% of bins randomly
                pct = random.uniform(0.3, 0.5)
                num_to_empty = max(1, min(int(num_bins * pct), num_bins - 1))
                bins_to_empty = random.sample(range(num_bins), num_to_empty)
                freed_items = []
                for b in bins_to_empty:
                    for item in bins[b]:
                        freed_items.append(item)
                        bin_rem[b] += W[item]
                    bins[b] = []
                
                random.shuffle(freed_items)
                freed_items.sort(key=lambda i: W[i], reverse=True)
                # Randomized best fit
                for idx in freed_items:
                    w = W[idx]
                    feasible = []
                    for b in range(len(bins)):
                        if bin_rem[b] >= w:
                            feasible.append((bin_rem[b], b))
                    if not feasible:
                        bins.append([idx])
                        bin_rem.append(C - w)
                    else:
                        feasible.sort()
                        if random.random() < 0.3 and len(feasible) > 1:
                            top_k = min(3, len(feasible))
                            _, chosen_b = random.choice(feasible[:top_k])
                        else:
                            _, chosen_b = feasible[0]
                        bins[chosen_b].append(idx)
                        bin_rem[chosen_b] -= w
                
                bins, bin_rem = validate_and_clean(bins, bin_rem)
            
            cur_cost = solution_cost(bins)
            cur_sos = sum_of_squares(bins, bin_rem)
            k = 1
            no_improve_count = 0
    
    # Build output
    packing = best_bins
    bin_weights = [sum(W[idx] for idx in b) for b in packing]
    
    return {"packing": packing, "bin_weights": bin_weights}