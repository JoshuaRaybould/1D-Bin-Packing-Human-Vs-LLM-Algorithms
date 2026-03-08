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
    W = weights
    
    total_weight = sum(W)
    L1 = math.ceil(total_weight / C)
    
    # Better lower bound
    def compute_lower_bound():
        best_lb = L1
        sw = sorted(W, reverse=True)
        for alpha in range(1, C // 2 + 1):
            threshold_large = C - alpha
            n_large = 0
            sum_medium = 0
            sum_small = 0
            for w in sw:
                if w > threshold_large:
                    n_large += 1
                elif w > alpha:
                    sum_medium += w
                else:
                    sum_small += w
            lb = n_large + max(0, math.ceil((sum_medium + sum_small) / C))
            if lb > best_lb:
                best_lb = lb
        return best_lb
    
    lower_bound = compute_lower_bound()
    
    def validate_and_clean(bins, bin_rem):
        new_bins = []
        new_rem = []
        for b in range(len(bins)):
            if bins[b]:
                new_bins.append(bins[b])
                new_rem.append(bin_rem[b])
        return new_bins, new_rem
    
    def copy_solution(bins, bin_rem):
        return [lst[:] for lst in bins], bin_rem[:]
    
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
    
    if len(bins_bfd) < len(bins_ffd) or (len(bins_bfd) == len(bins_ffd)):
        bins, bin_rem = bins_bfd, rem_bfd
        if len(bins_ffd) < len(bins_bfd):
            bins, bin_rem = bins_ffd, rem_ffd
    else:
        bins, bin_rem = bins_ffd, rem_ffd
    
    bins, bin_rem = validate_and_clean(bins, bin_rem)
    best_bins, best_rem = copy_solution(bins, bin_rem)
    best_cost = len(best_bins)
    
    if best_cost <= lower_bound:
        packing = best_bins
        bin_weights = [sum(W[idx] for idx in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}
    
    # --- Focused bin emptying with smarter redistribution ---
    def try_empty_bin(bins, bin_rem, src_b):
        """Try to empty bin src_b by redistributing its items to other bins.
        Uses best-fit with backtracking for small bins."""
        if not bins[src_b]:
            return False
        
        items = sorted(bins[src_b], key=lambda i: W[i], reverse=True)
        src_load = sum(W[i] for i in items)
        other_rem = sum(bin_rem[b] for b in range(len(bins)) if b != src_b)
        if other_rem < src_load:
            return False
        
        temp_rem = bin_rem[:]
        assignments = {}
        
        for idx in items:
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
                return False
        
        # Apply
        for idx, tgt_b in assignments.items():
            bins[tgt_b].append(idx)
            bin_rem[tgt_b] -= W[idx]
        bins[src_b] = []
        bin_rem[src_b] = C
        return True
    
    def try_empty_bin_with_swaps(bins, bin_rem, src_b, deadline):
        """Try to empty bin src_b, allowing swaps to make room."""
        if not bins[src_b]:
            return False
        
        items = sorted(bins[src_b], key=lambda i: W[i], reverse=True)
        src_load = sum(W[i] for i in items)
        other_rem = sum(bin_rem[b] for b in range(len(bins)) if b != src_b)
        if other_rem < src_load:
            return False
        
        num_bins = len(bins)
        # For each item that doesn't fit directly, try swap-and-place
        temp_rem = bin_rem[:]
        # Track items added to each bin during this process
        temp_additions = [[] for _ in range(num_bins)]
        temp_removals = [[] for _ in range(num_bins)]
        placed_items = set()
        
        for idx in items:
            w = W[idx]
            # Try direct placement
            best_b = -1
            best_r = C + 1
            for b in range(num_bins):
                if b == src_b:
                    continue
                if temp_rem[b] >= w and temp_rem[b] < best_r:
                    best_r = temp_rem[b]
                    best_b = b
            if best_b >= 0:
                temp_additions[best_b].append(idx)
                temp_rem[best_b] -= w
                placed_items.add(idx)
                continue
            
            # Try swap: find item j in bin b such that w - W[j] <= temp_rem[b]
            # and W[j] can fit somewhere else
            found_swap = False
            candidates = []
            for b in range(num_bins):
                if b == src_b:
                    continue
                # Items in this bin (original minus removed plus added)
                current_items = [x for x in bins[b] if x not in set(temp_removals[b])] + temp_additions[b]
                for j in current_items:
                    if j in placed_items:
                        continue
                    wj = W[j]
                    if temp_rem[b] + wj >= w:
                        # j would be freed, can it fit elsewhere?
                        # Check if there's a bin with room for wj
                        for b2 in range(num_bins):
                            if b2 == src_b or b2 == b:
                                continue
                            if temp_rem[b2] >= wj:
                                candidates.append((temp_rem[b] + wj - w, b, j, b2))
                                break
            
            if candidates:
                candidates.sort()
                _, b, j, b2 = candidates[0]
                wj = W[j]
                temp_rem[b] += wj
                temp_removals[b].append(j)
                temp_rem[b] -= w
                temp_additions[b].append(idx)
                temp_rem[b2] -= wj
                temp_additions[b2].append(j)
                placed_items.add(idx)
                found_swap = True
            
            if not found_swap:
                return False
            
            if time.time() > deadline:
                return False
        
        # Apply all changes
        for b in range(num_bins):
            if b == src_b:
                continue
            for j in temp_removals[b]:
                bins[b].remove(j)
                bin_rem[b] += W[j]
            for j in temp_additions[b]:
                bins[b].append(j)
                bin_rem[b] -= W[j]
        bins[src_b] = []
        bin_rem[src_b] = C
        return True
    
    def local_search(bins, bin_rem, deadline, thorough=True):
        # Phase 1: Try to empty bins (lightest first)
        improved = True
        while improved:
            improved = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            for src_b in bin_order:
                if try_empty_bin(bins, bin_rem, src_b):
                    bins, bin_rem = validate_and_clean(bins, bin_rem)
                    improved = True
                    break
                if time.time() > deadline:
                    return validate_and_clean(bins, bin_rem)
        
        if not thorough:
            return validate_and_clean(bins, bin_rem)
        
        # Phase 2: Transfer moves to improve packing tightness
        for _pass in range(3):
            did_improve = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            for src_b in range(num_bins):
                if not bins[src_b]:
                    continue
                for item_pos in range(len(bins[src_b]) - 1, -1, -1):
                    idx = bins[src_b][item_pos]
                    w = W[idx]
                    src_load = C - bin_rem[src_b]
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
                        new_tgt_sq = (tgt_load + w) * (tgt_load + w)
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
                if time.time() > deadline:
                    return validate_and_clean(bins, bin_rem)
            if not did_improve:
                break
        
        bins, bin_rem = validate_and_clean(bins, bin_rem)
        
        # Phase 3: Swap moves
        for _pass in range(3):
            did_improve = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            for b1 in range(num_bins):
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
                    if time.time() > deadline:
                        return validate_and_clean(bins, bin_rem)
            if not did_improve:
                break
        
        # Phase 4: Try emptying again after transfers/swaps
        improved = True
        while improved:
            improved = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            for src_b in bin_order:
                if try_empty_bin(bins, bin_rem, src_b):
                    bins, bin_rem = validate_and_clean(bins, bin_rem)
                    improved = True
                    break
                if time.time() > deadline:
                    return validate_and_clean(bins, bin_rem)
        
        # Phase 5: Try emptying with swaps for the lightest bins
        num_bins = len(bins)
        if num_bins > 1:
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            for src_b in bin_order[:min(3, num_bins)]:
                if try_empty_bin_with_swaps(bins, bin_rem, src_b, deadline):
                    bins, bin_rem = validate_and_clean(bins, bin_rem)
                    # After success, try simple emptying again
                    cont = True
                    while cont:
                        cont = False
                        num_bins2 = len(bins)
                        bo2 = sorted(range(num_bins2), key=lambda b: C - bin_rem[b])
                        for sb in bo2:
                            if try_empty_bin(bins, bin_rem, sb):
                                bins, bin_rem = validate_and_clean(bins, bin_rem)
                                cont = True
                                break
                    break
                if time.time() > deadline:
                    break
        
        return validate_and_clean(bins, bin_rem)
    
    # Apply local search to initial solution
    deadline = start_time + time_limit * 0.15
    bins, bin_rem = local_search(bins, bin_rem, deadline, thorough=True)
    best_bins, best_rem = copy_solution(bins, bin_rem)
    best_cost = len(best_bins)
    
    if best_cost <= lower_bound:
        packing = best_bins
        bin_weights = [sum(W[idx] for idx in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}
    
    # --- Shaking ---
    def randomized_best_fit(freed_items, bins, bin_rem):
        freed_items_sorted = sorted(freed_items, key=lambda i: W[i], reverse=True)
        for idx in freed_items_sorted:
            w = W[idx]
            feasible = []
            for b in range(len(bins)):
                if bin_rem[b] >= w:
                    feasible.append((bin_rem[b], b))
            if not feasible:
                bins.append([idx])
                bin_rem.append(C - w)
                continue
            feasible.sort()
            if random.random() < 0.3 and len(feasible) > 1:
                top_k = min(3, len(feasible))
                _, chosen_b = random.choice(feasible[:top_k])
            else:
                _, chosen_b = feasible[0]
            bins[chosen_b].append(idx)
            bin_rem[chosen_b] -= w
        return bins, bin_rem
    
    k_max = 8
    
    def shaking(bins, bin_rem, k):
        bins = [lst[:] for lst in bins]
        bin_rem = bin_rem[:]
        num_bins = len(bins)
        if num_bins <= 1:
            return bins, bin_rem
        
        if k == 1:
            # Single random transfer
            src_b = random.randint(0, num_bins - 1)
            if not bins[src_b]:
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
            if b1 != b2 and bins[b1] and bins[b2]:
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
            # Chain move
            num_moves = random.randint(2, 5)
            for _ in range(num_moves):
                if len(bins) <= 1:
                    break
                src_b = random.randint(0, len(bins) - 1)
                if not bins[src_b]:
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
            # Destroy lightest 1-2 bins
            num_to_empty = min(random.randint(1, 2), num_bins - 1)
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            bins_to_empty = bin_order[:num_to_empty]
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] += sum(W[i] for i in bins[b])
                bins[b] = []
            bins, bin_rem = randomized_best_fit(freed_items, bins, bin_rem)
        
        elif k == 5:
            # Destroy 2-4 random bins
            num_to_empty = min(random.randint(2, 4), num_bins - 1)
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] += sum(W[i] for i in bins[b])
                bins[b] = []
            bins, bin_rem = randomized_best_fit(freed_items, bins, bin_rem)
        
        elif k == 6:
            # Destroy 15-25% lightest bins
            pct = random.uniform(0.15, 0.25)
            num_to_empty = max(1, min(int(num_bins * pct), num_bins - 1))
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            bins_to_empty = bin_order[:num_to_empty]
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] += sum(W[i] for i in bins[b])
                bins[b] = []
            bins, bin_rem = randomized_best_fit(freed_items, bins, bin_rem)
        
        elif k == 7:
            # Destroy 30-50% random bins with first-fit random repair
            pct = random.uniform(0.3, 0.5)
            num_to_empty = max(1, min(int(num_bins * pct), num_bins - 1))
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] += sum(W[i] for i in bins[b])
                bins[b] = []
            random.shuffle(freed_items)
            freed_items.sort(key=lambda i: W[i], reverse=True)
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
        
        elif k == 8:
            # Targeted: destroy the two least-loaded bins plus a random bin
            num_to_empty = min(3, num_bins - 1)
            bin_order = sorted(range(num_bins), key=lambda b: C - bin_rem[b])
            bins_to_empty = set(bin_order[:min(2, num_to_empty)])
            # Add a random bin
            remaining = [b for b in range(num_bins) if b not in bins_to_empty]
            if remaining:
                bins_to_empty.add(random.choice(remaining))
            bins_to_empty = list(bins_to_empty)
            if len(bins_to_empty) > num_bins - 1:
                bins_to_empty = bins_to_empty[:num_bins - 1]
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] += sum(W[i] for i in bins[b])
                bins[b] = []
            bins, bin_rem = randomized_best_fit(freed_items, bins, bin_rem)
        
        bins, bin_rem = validate_and_clean(bins, bin_rem)
        return bins, bin_rem
    
    # --- VNS Main Loop ---
    k = 1
    no_improve_count = 0
    cur_cost = len(bins)
    main_deadline = start_time + time_limit * 0.98
    
    # Adaptive time for local search
    iteration = 0
    
    while True:
        now = time.time()
        if now > main_deadline:
            break
        if best_cost <= lower_bound:
            break
        
        iteration += 1
        
        # Determine how thorough local search should be based on remaining time
        remaining_time = main_deadline - now
        ls_deadline = now + min(remaining_time * 0.5, remaining_time)
        
        # Shaking
        new_bins, new_rem = shaking(bins, bin_rem, k)
        
        # Local search
        thorough = (k >= 4)  # More thorough for larger perturbations
        new_bins, new_rem = local_search(new_bins, new_rem, ls_deadline, thorough=thorough)
        new_cost = len(new_bins)
        
        # Accept if better or equal cost
        cur_cost_now = len(bins)
        if new_cost <= cur_cost_now:
            bins, bin_rem = new_bins, new_rem
            cur_cost = new_cost
            k = 1
            no_improve_count = 0
            
            if new_cost < best_cost:
                best_bins, best_rem = copy_solution(bins, bin_rem)
                best_cost = new_cost
                if best_cost <= lower_bound:
                    break
        else:
            k += 1
            no_improve_count += 1
            if k > k_max:
                k = 1
        
        # Restart if stuck
        if no_improve_count > k_max * 10:
            bins, bin_rem = copy_solution(best_bins, best_rem)
            num_bins = len(bins)
            if num_bins > 2:
                pct = random.uniform(0.3, 0.6)
                num_to_empty = max(1, min(int(num_bins * pct), num_bins - 1))
                bins_to_empty = random.sample(range(num_bins), num_to_empty)
                freed_items = []
                for b in bins_to_empty:
                    freed_items.extend(bins[b])
                    bin_rem[b] += sum(W[i] for i in bins[b])
                    bins[b] = []
                freed_items.sort(key=lambda i: W[i], reverse=True)
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
                        if random.random() < 0.4 and len(feasible) > 1:
                            top_k = min(3, len(feasible))
                            _, chosen_b = random.choice(feasible[:top_k])
                        else:
                            _, chosen_b = feasible[0]
                        bins[chosen_b].append(idx)
                        bin_rem[chosen_b] -= w
                bins, bin_rem = validate_and_clean(bins, bin_rem)
            cur_cost = len(bins)
            k = 1
            no_improve_count = 0
    
    packing = best_bins
    bin_weights = [sum(W[idx] for idx in b) for b in packing]
    return {"packing": packing, "bin_weights": bin_weights}
