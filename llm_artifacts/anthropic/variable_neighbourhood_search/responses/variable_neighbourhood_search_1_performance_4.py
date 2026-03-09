import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok(fraction=0.95):
        return elapsed() < time_limit * fraction
    
    # --- Heuristic constructors ---
    def best_fit_decreasing(items_indices, cap):
        sorted_items = sorted(items_indices, key=lambda i: weights[i], reverse=True)
        bins = []
        bin_rem = []
        for idx in sorted_items:
            w = weights[idx]
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
    
    def first_fit_decreasing(items_indices, cap):
        sorted_items = sorted(items_indices, key=lambda i: weights[i], reverse=True)
        bins = []
        bin_rem = []
        for idx in sorted_items:
            w = weights[idx]
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
    
    # Build initial solution - try both BFD and FFD
    all_items = list(range(n))
    bins_bfd, rem_bfd = best_fit_decreasing(all_items, bin_capacity)
    bins_ffd, rem_ffd = first_fit_decreasing(all_items, bin_capacity)
    
    if len(bins_bfd) <= len(bins_ffd):
        bins, bin_rem = bins_bfd, rem_bfd
    else:
        bins, bin_rem = bins_ffd, rem_ffd
    
    bins, bin_rem = validate_and_clean(bins, bin_rem)
    
    # --- Local Search: Try to empty bins with swap-enhanced moves ---
    def try_empty_bin(bins, bin_rem, src_b):
        """Try to empty bin src_b by redistributing its items.
        Returns True and modifies bins/bin_rem in place if successful."""
        if src_b >= len(bins) or len(bins[src_b]) == 0:
            return False
        
        items = bins[src_b][:]
        items.sort(key=lambda i: weights[i], reverse=True)
        
        temp_rem = bin_rem[:]
        assignments = {}
        can_empty = True
        
        for idx in items:
            w = weights[idx]
            best_b = -1
            best_r = bin_capacity + 1
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
                bin_rem[tgt_b] -= weights[idx]
            bins[src_b] = []
            bin_rem[src_b] = bin_capacity
            return True
        return False
    
    def try_empty_bin_with_swaps(bins, bin_rem, src_b):
        """Try to empty bin src_b using direct moves + (1,1) swaps."""
        if src_b >= len(bins) or len(bins[src_b]) == 0:
            return False
        
        items = bins[src_b][:]
        items.sort(key=lambda i: weights[i], reverse=True)
        num_bins = len(bins)
        
        # For each item, find where it can go directly
        temp_rem = bin_rem[:]
        assignments = {}  # item -> (target_bin, swap_item or None)
        remaining_items = []
        
        for idx in items:
            w = weights[idx]
            best_b = -1
            best_r = bin_capacity + 1
            for b in range(num_bins):
                if b == src_b:
                    continue
                if temp_rem[b] >= w and temp_rem[b] < best_r:
                    best_r = temp_rem[b]
                    best_b = b
            if best_b >= 0:
                assignments[idx] = (best_b, None)
                temp_rem[best_b] -= w
            else:
                remaining_items.append(idx)
        
        if not remaining_items:
            # All items can be placed directly
            for idx, (tgt_b, _) in assignments.items():
                bins[tgt_b].append(idx)
                bin_rem[tgt_b] -= weights[idx]
            bins[src_b] = []
            bin_rem[src_b] = bin_capacity
            return True
        
        # Try swap-based placement for remaining items
        # For each remaining item r, find a bin b and an item j in b such that:
        #   - w[r] - w[j] <= temp_rem[b] (r fits if j is removed)
        #   - j can fit somewhere else (using temp_rem after swap)
        # Limit complexity
        if len(remaining_items) > 3:
            return False
        
        # Try to resolve remaining items one by one with swaps
        for r_idx in remaining_items:
            w_r = weights[r_idx]
            placed = False
            for b in range(num_bins):
                if b == src_b:
                    continue
                for j_pos, j_idx in enumerate(bins[b]):
                    # Skip items already assigned away
                    if j_idx in assignments:
                        continue
                    w_j = weights[j_idx]
                    # Can r replace j in bin b?
                    needed = w_r - w_j
                    if needed > temp_rem[b]:
                        continue
                    # Can j fit somewhere else?
                    new_rem_b = temp_rem[b] - needed  # remaining after swap
                    best_target = -1
                    best_target_r = bin_capacity + 1
                    for b2 in range(num_bins):
                        if b2 == src_b or b2 == b:
                            r2 = temp_rem[b2]
                        else:
                            r2 = temp_rem[b2]
                        if b2 == src_b:
                            continue
                        if b2 == b:
                            continue
                        if r2 >= w_j and r2 < best_target_r:
                            best_target_r = r2
                            best_target = b2
                    if best_target >= 0:
                        # Do the swap
                        assignments[r_idx] = (b, j_idx)  # r goes to b, displacing j
                        assignments[j_idx] = (best_target, None)  # j goes to best_target
                        temp_rem[b] -= needed
                        temp_rem[best_target] -= w_j
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                return False
        
        # All items can be placed - apply moves
        # First, handle swaps: remove swapped-out items from their bins
        swapped_out = set()
        for idx, (tgt_b, swap_item) in assignments.items():
            if swap_item is not None:
                swapped_out.add(swap_item)
        
        # Remove swapped items from their original bins
        for b in range(num_bins):
            if b == src_b:
                continue
            bins[b] = [x for x in bins[b] if x not in swapped_out]
            # Recalculate remaining
        
        # Recalculate bin_rem for affected bins
        for b in range(num_bins):
            if b == src_b:
                continue
            bin_rem[b] = bin_capacity - sum(weights[x] for x in bins[b])
        
        # Now place all items from assignments
        for idx, (tgt_b, _) in assignments.items():
            bins[tgt_b].append(idx)
            bin_rem[tgt_b] -= weights[idx]
        
        bins[src_b] = []
        bin_rem[src_b] = bin_capacity
        return True
    
    def local_search(bins, bin_rem, use_swaps=True):
        improved = True
        while improved:
            improved = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            
            # Sort bins by load ascending - try to empty lightest first
            bin_loads = [(bin_capacity - bin_rem[b], b) for b in range(num_bins)]
            bin_loads.sort()
            
            for load, src_b in bin_loads:
                if src_b >= len(bins) or len(bins[src_b]) == 0:
                    continue
                
                success = try_empty_bin(bins, bin_rem, src_b)
                if not success and use_swaps:
                    success = try_empty_bin_with_swaps(bins, bin_rem, src_b)
                
                if success:
                    bins, bin_rem = validate_and_clean(bins, bin_rem)
                    improved = True
                    break
            
            if not time_ok(0.95):
                break
        
        return bins, bin_rem
    
    def local_search_moves(bins, bin_rem):
        """Try (1,1) swaps and (1,0) moves that improve packing tightness."""
        num_bins = len(bins)
        if num_bins <= 1:
            return bins, bin_rem
        
        improved = True
        iters = 0
        while improved and iters < 2:
            improved = False
            iters += 1
            
            if not time_ok(0.93):
                break
            
            # Try (1,1) swaps that reduce the number of bins or improve balance
            # Focus on making lightest bins lighter (easier to empty)
            bin_loads = [(bin_capacity - bin_rem[b], b) for b in range(num_bins)]
            bin_loads.sort()
            
            # For lightest bins, try swapping their items with lighter items in heavier bins
            light_bins = [b for _, b in bin_loads[:max(1, num_bins // 4)]]
            
            for src_b in light_bins:
                if not time_ok(0.93):
                    break
                best_swap = None
                best_delta = 0  # we want to minimize load of src_b
                
                for i_pos, i_idx in enumerate(bins[src_b]):
                    w_i = weights[i_idx]
                    for tgt_b in range(num_bins):
                        if tgt_b == src_b:
                            continue
                        for j_pos, j_idx in enumerate(bins[tgt_b]):
                            w_j = weights[j_idx]
                            # Swap i and j: src_b loses w_i, gains w_j
                            # We want w_j < w_i (reduce src_b load)
                            delta = w_i - w_j
                            if delta <= 0:
                                continue
                            # Check feasibility
                            if bin_rem[tgt_b] + w_j - w_i >= 0:  # tgt_b can accommodate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_swap = (src_b, i_pos, i_idx, tgt_b, j_pos, j_idx)
                
                if best_swap:
                    sb, ip, ii, tb, jp, ji = best_swap
                    w_i = weights[ii]
                    w_j = weights[ji]
                    bins[sb][ip] = ji
                    bins[tb][jp] = ii
                    bin_rem[sb] += w_i - w_j
                    bin_rem[tb] += w_j - w_i
                    improved = True
            
            # Also try (1,0) moves: move item from light bin to another bin
            for src_b in light_bins:
                if not time_ok(0.93):
                    break
                for i_pos in range(len(bins[src_b]) - 1, -1, -1):
                    i_idx = bins[src_b][i_pos]
                    w_i = weights[i_idx]
                    # Find best-fit target
                    best_b = -1
                    best_r = bin_capacity + 1
                    for b in range(num_bins):
                        if b == src_b:
                            continue
                        if bin_rem[b] >= w_i and bin_rem[b] < best_r:
                            best_r = bin_rem[b]
                            best_b = b
                    if best_b >= 0 and best_r < bin_rem[src_b]:
                        # Move only if target is tighter fit than leaving in src
                        bins[best_b].append(i_idx)
                        bin_rem[best_b] -= w_i
                        bins[src_b].pop(i_pos)
                        bin_rem[src_b] += w_i
                        improved = True
            
            num_bins = len(bins)
        
        return bins, bin_rem
    
    # Apply initial local search
    bins, bin_rem = local_search(bins, bin_rem, use_swaps=True)
    bins, bin_rem = validate_and_clean(bins, bin_rem)
    
    best_bins, best_rem = copy_solution(bins, bin_rem)
    best_cost = len(best_bins)
    
    # --- VNS Shaking ---
    def repack_items_bf(freed_items, bins, bin_rem):
        """Repack freed items into existing bins using best-fit."""
        freed_items.sort(key=lambda i: weights[i], reverse=True)
        for idx in freed_items:
            w = weights[idx]
            best_b = -1
            best_r = bin_capacity + 1
            for b in range(len(bins)):
                if bin_rem[b] >= w and bin_rem[b] < best_r:
                    best_r = bin_rem[b]
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(idx)
                bin_rem[best_b] -= w
            else:
                bins.append([idx])
                bin_rem.append(bin_capacity - w)
    
    def shaking(bins, bin_rem, k):
        bins = [lst[:] for lst in bins]
        bin_rem = bin_rem[:]
        num_bins = len(bins)
        
        if num_bins <= 1:
            return bins, bin_rem
        
        if k == 1:
            # Move one random item to a random different bin
            src_b = random.randint(0, num_bins - 1)
            if len(bins[src_b]) == 0:
                return bins, bin_rem
            item_pos = random.randint(0, len(bins[src_b]) - 1)
            item = bins[src_b][item_pos]
            w = weights[item]
            candidates = [b for b in range(num_bins) if b != src_b and bin_rem[b] >= w]
            if candidates:
                tgt_b = random.choice(candidates)
                bins[src_b].pop(item_pos)
                bin_rem[src_b] += w
                bins[tgt_b].append(item)
                bin_rem[tgt_b] -= w
            
        elif k == 2:
            # Swap two items between two bins
            b1 = random.randint(0, num_bins - 1)
            b2 = random.randint(0, num_bins - 1)
            if b1 != b2 and len(bins[b1]) > 0 and len(bins[b2]) > 0:
                i1_pos = random.randint(0, len(bins[b1]) - 1)
                i2_pos = random.randint(0, len(bins[b2]) - 1)
                item1 = bins[b1][i1_pos]
                item2 = bins[b2][i2_pos]
                w1, w2 = weights[item1], weights[item2]
                diff = w1 - w2
                if bin_rem[b1] + diff >= 0 and bin_rem[b2] - diff >= 0:
                    bins[b1][i1_pos] = item2
                    bins[b2][i2_pos] = item1
                    bin_rem[b1] += diff
                    bin_rem[b2] -= diff
            
        elif k == 3:
            # Move 2-3 random items
            num_moves = random.randint(2, 3)
            for _ in range(num_moves):
                if len(bins) <= 1:
                    break
                src_b = random.randint(0, len(bins) - 1)
                if len(bins[src_b]) == 0:
                    continue
                item_pos = random.randint(0, len(bins[src_b]) - 1)
                item = bins[src_b][item_pos]
                w = weights[item]
                candidates = [b for b in range(len(bins)) if b != src_b and bin_rem[b] >= w]
                if candidates:
                    tgt_b = random.choice(candidates)
                    bins[src_b].pop(item_pos)
                    bin_rem[src_b] += w
                    bins[tgt_b].append(item)
                    bin_rem[tgt_b] -= w
            
        elif k == 4:
            # Destroy 1-2 bins and repair
            num_to_empty = min(random.randint(1, 2), num_bins - 1)
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] = bin_capacity
                bins[b] = []
            random.shuffle(freed_items)
            repack_items_bf(freed_items, bins, bin_rem)
            
        elif k == 5:
            # Destroy 3-5 bins and repair
            num_to_empty = min(random.randint(3, 5), num_bins - 1)
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] = bin_capacity
                bins[b] = []
            random.shuffle(freed_items)
            repack_items_bf(freed_items, bins, bin_rem)
            
        elif k == 6:
            # Destroy lightest bins and repair
            num_to_empty = min(random.randint(2, max(3, num_bins // 5)), num_bins - 1)
            bin_order = sorted(range(num_bins), key=lambda b: bin_capacity - bin_rem[b])
            bins_to_empty = bin_order[:num_to_empty]
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] = bin_capacity
                bins[b] = []
            random.shuffle(freed_items)
            repack_items_bf(freed_items, bins, bin_rem)
            
        elif k == 7:
            # Large perturbation: destroy 20-40% of bins
            num_to_empty = min(max(2, num_bins // 3), num_bins - 1)
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                freed_items.extend(bins[b])
                bin_rem[b] = bin_capacity
                bins[b] = []
            random.shuffle(freed_items)
            repack_items_bf(freed_items, bins, bin_rem)
        
        bins, bin_rem = validate_and_clean(bins, bin_rem)
        return bins, bin_rem
    
    # Main VNS loop
    k_max = 7
    k = 1
    no_improve_count = 0
    
    while time_ok(0.90):
        # Shaking
        new_bins, new_rem = shaking(bins, bin_rem, k)
        
        # Local search: first try moves to consolidate, then try emptying
        if time_ok(0.85):
            new_bins, new_rem = local_search_moves(new_bins, new_rem)
        new_bins, new_rem = validate_and_clean(new_bins, new_rem)
        
        if time_ok(0.80):
            new_bins, new_rem = local_search(new_bins, new_rem, use_swaps=(len(new_bins) < 100))
            new_bins, new_rem = validate_and_clean(new_bins, new_rem)
        
        new_cost = len(new_bins)
        
        if new_cost <= len(bins):
            bins, bin_rem = new_bins, new_rem
            k = 1
            no_improve_count = 0
            
            if new_cost < best_cost:
                best_bins, best_rem = copy_solution(bins, bin_rem)
                best_cost = new_cost
        else:
            k += 1
            no_improve_count += 1
            if k > k_max:
                k = 1
        
        # Restart from best if stuck
        if no_improve_count > k_max * 8:
            bins, bin_rem = copy_solution(best_bins, best_rem)
            num_bins = len(bins)
            if num_bins > 2:
                # Large perturbation from best
                num_to_empty = min(max(num_bins // 3, 2), num_bins - 1)
                bin_order = sorted(range(num_bins), key=lambda b: bin_capacity - bin_rem[b])
                # Mix: some lightest, some random
                light_count = min(num_to_empty // 2 + 1, len(bin_order))
                light_bins = bin_order[:light_count]
                remaining_for_random = [b for b in range(num_bins) if b not in light_bins]
                random_count = min(num_to_empty - light_count, len(remaining_for_random))
                if random_count > 0:
                    random_bins = random.sample(remaining_for_random, random_count)
                else:
                    random_bins = []
                bins_to_empty = light_bins + random_bins
                
                freed_items = []
                for b in bins_to_empty:
                    freed_items.extend(bins[b])
                    bin_rem[b] = bin_capacity
                    bins[b] = []
                
                random.shuffle(freed_items)
                repack_items_bf(freed_items, bins, bin_rem)
                bins, bin_rem = validate_and_clean(bins, bin_rem)
            
            k = 1
            no_improve_count = 0
    
    # Build output
    packing = best_bins
    bin_weights = [sum(weights[idx] for idx in b) for b in packing]
    
    return {"packing": packing, "bin_weights": bin_weights}