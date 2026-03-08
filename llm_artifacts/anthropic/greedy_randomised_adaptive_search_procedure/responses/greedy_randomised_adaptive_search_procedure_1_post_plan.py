import random
import time
import math
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    W = tuple(weights)
    
    def elapsed():
        return time.time() - start_time
    
    # Lower bounds
    total_weight = sum(W)
    lower_bound_l1 = math.ceil(total_weight / bin_capacity)
    
    # L2 lower bound
    def compute_l2():
        best_lb = lower_bound_l1
        # Try multiple thresholds
        for denom in range(3, 10):
            t = bin_capacity // denom
            if t == 0:
                continue
            # Items larger than bin_capacity - t need their own bin essentially
            # Items larger than t but <= bin_capacity - t can potentially share
            large = []  # items > bin_capacity - t
            medium = []  # items > t and <= bin_capacity - t  
            small_total = 0  # items <= t
            
            for w in W:
                if w > bin_capacity - t:
                    large.append(w)
                elif w > t:
                    medium.append(w)
                else:
                    small_total += w
            
            n_large = len(large)
            # Each large item needs its own bin. How much space is left in large bins?
            space_in_large = sum(bin_capacity - w for w in large)
            
            # Medium items: try to pair with large items
            medium.sort()
            medium_remaining = []
            space_used = 0
            for mw in medium:
                if space_used + mw <= space_in_large:
                    space_used += mw
                else:
                    medium_remaining.append(mw)
            
            # Remaining medium items need bins, pair them up
            n_medium_bins = math.ceil(len(medium_remaining) / 2) if medium_remaining else 0
            medium_remaining_weight = sum(medium_remaining)
            
            # Small items fill remaining space
            space_after_large = space_in_large - space_used
            space_in_medium_bins = n_medium_bins * bin_capacity - medium_remaining_weight
            total_space_for_small = space_after_large + space_in_medium_bins
            
            remaining_small = max(0, small_total - total_space_for_small)
            extra_bins = math.ceil(remaining_small / bin_capacity) if remaining_small > 0 else 0
            
            lb = n_large + n_medium_bins + extra_bins
            best_lb = max(best_lb, lb)
        
        return best_lb
    
    lower_bound = compute_l2()
    
    # Best-fit using bisect on sorted remaining capacities
    def bestfit_place(item_weight, bins_list, bin_remaining, sorted_rem):
        """Place item using best-fit with bisect. sorted_rem is list of (remaining, bin_idx) sorted by remaining."""
        # Find smallest remaining >= item_weight
        pos = bisect.bisect_left(sorted_rem, (item_weight, -1))
        if pos < len(sorted_rem):
            rem, bidx = sorted_rem[pos]
            # Remove old entry
            sorted_rem.pop(pos)
            # Update
            new_rem = rem - item_weight
            bins_list[bidx].append(item_weight)  # will fix to use indices
            bin_remaining[bidx] = new_rem
            # Insert new entry
            bisect.insort(sorted_rem, (new_rem, bidx))
            return bidx
        else:
            # New bin
            bidx = len(bins_list)
            bins_list.append([])
            new_rem = bin_capacity - item_weight
            bin_remaining.append(new_rem)
            bisect.insort(sorted_rem, (new_rem, bidx))
            return bidx
    
    def construct_fast(alpha, variant=0):
        """Fast construction with bisect-based best-fit."""
        # Build order of items to place
        if variant == 0:
            # RCL on item selection (unsorted), best-fit placement
            sorted_desc = sorted(range(n), key=lambda i: -W[i])
            remaining = list(sorted_desc)  # will remove by swap-to-end
            order = []
            while remaining:
                # max is first, min is last (sorted desc)
                w_max = W[remaining[0]]
                w_min = W[remaining[-1]]
                threshold = w_max - alpha * (w_max - w_min)
                # Find RCL: items with weight >= threshold
                # Since sorted desc, find cutoff
                # Binary search for threshold in the sorted (desc) list
                lo, hi = 0, len(remaining)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if W[remaining[mid]] >= threshold:
                        lo = mid + 1
                    else:
                        hi = mid
                rcl_end = lo  # remaining[0:rcl_end] have weight >= threshold
                if rcl_end == 0:
                    rcl_end = 1
                pick = random.randint(0, rcl_end - 1)
                item = remaining[pick]
                order.append(item)
                # Swap-to-end removal
                remaining[pick] = remaining[-1]
                remaining.pop()
                # Re-sort is needed... actually swap breaks the sort order
                # We need to maintain sorted order. Let's just pop from a list
                # Actually for efficiency, let's not maintain sorted order
                # Instead, just scan for max/min
                pass
            # Hmm, swap-to-end breaks sorted order. Let me use a different approach.
            pass
        
        if variant == 1:
            # Sorted items with RCL perturbation + best-fit
            sorted_desc = sorted(range(n), key=lambda i: -W[i])
            remaining = list(sorted_desc)
            order = []
            while remaining:
                w_max = W[remaining[0]]
                w_min = W[remaining[-1]]
                threshold = w_max - alpha * (w_max - w_min)
                lo, hi = 0, len(remaining)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if W[remaining[mid]] >= threshold:
                        lo = mid + 1
                    else:
                        hi = mid
                rcl_end = max(1, lo)
                pick = random.randint(0, rcl_end - 1)
                item = remaining[pick]
                order.append(item)
                remaining.pop(pick)
            pass
        elif variant == 2:
            # RCL on item selection + first-fit placement (handled below)
            sorted_desc = sorted(range(n), key=lambda i: -W[i])
            remaining = list(sorted_desc)
            order = []
            while remaining:
                w_max = W[remaining[0]]
                w_min = W[remaining[-1]]
                threshold = w_max - alpha * (w_max - w_min)
                lo, hi = 0, len(remaining)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if W[remaining[mid]] >= threshold:
                        lo = mid + 1
                    else:
                        hi = mid
                rcl_end = max(1, lo)
                pick = random.randint(0, rcl_end - 1)
                item = remaining[pick]
                order.append(item)
                remaining.pop(pick)
        else:
            # variant 0: same as 1 for now (we fix this below)
            sorted_desc = sorted(range(n), key=lambda i: -W[i])
            remaining = list(sorted_desc)
            order = []
            while remaining:
                w_max = W[remaining[0]]
                w_min = W[remaining[-1]]
                threshold = w_max - alpha * (w_max - w_min)
                lo, hi = 0, len(remaining)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if W[remaining[mid]] >= threshold:
                        lo = mid + 1
                    else:
                        hi = mid
                rcl_end = max(1, lo)
                pick = random.randint(0, rcl_end - 1)
                item = remaining[pick]
                order.append(item)
                remaining.pop(pick)
        
        # Now place items using best-fit (or first-fit for variant 2)
        bins = []  # list of lists of item indices
        bin_rem = []  # remaining capacity
        sorted_rem = []  # sorted list of (remaining, bin_idx)
        
        if variant == 2:
            # First-fit
            for item in order:
                w = W[item]
                placed = False
                for b in range(len(bins)):
                    if bin_rem[b] >= w:
                        bins[b].append(item)
                        bin_rem[b] -= w
                        placed = True
                        break
                if not placed:
                    bins.append([item])
                    bin_rem.append(bin_capacity - w)
        else:
            # Best-fit using bisect
            for item in order:
                w = W[item]
                pos = bisect.bisect_left(sorted_rem, (w, -1))
                if pos < len(sorted_rem):
                    rem, bidx = sorted_rem[pos]
                    sorted_rem.pop(pos)
                    new_rem = rem - w
                    bins[bidx].append(item)
                    bin_rem[bidx] = new_rem
                    bisect.insort(sorted_rem, (new_rem, bidx))
                else:
                    bidx = len(bins)
                    bins.append([item])
                    new_rem = bin_capacity - w
                    bin_rem.append(new_rem)
                    bisect.insort(sorted_rem, (new_rem, bidx))
        
        return bins, bin_rem
    
    def local_search_full(bins, bin_rem, deadline):
        """Full local search with swap-assisted bin emptying."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            num_bins = len(bins)
            
            # Phase 1: Try to empty bins (sorted by ascending load)
            loads = [bin_capacity - bin_rem[b] for b in range(num_bins)]
            bin_order = sorted(range(num_bins), key=lambda b: loads[b])
            
            for src_idx in bin_order:
                if time.time() >= deadline:
                    break
                if not bins[src_idx]:
                    continue
                
                src_load = loads[src_idx]
                if src_load == 0:
                    continue
                
                # Quick feasibility check
                total_free = sum(bin_rem[b] for b in range(num_bins) if b != src_idx and bins[b])
                if total_free < src_load:
                    continue
                
                # Try to redistribute items from src using best-fit decreasing
                src_items = sorted(bins[src_idx], key=lambda i: -W[i])
                can_empty = True
                moves = []
                temp_rem = bin_rem[:]
                
                for item in src_items:
                    w = W[item]
                    best_bin = -1
                    best_r = bin_capacity + 1
                    for b in range(num_bins):
                        if b == src_idx or not bins[b]:
                            continue
                        if temp_rem[b] >= w:
                            r = temp_rem[b] - w
                            if r < best_r:
                                best_r = r
                                best_bin = b
                    if best_bin >= 0:
                        moves.append((item, best_bin))
                        temp_rem[best_bin] -= w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    for item, target in moves:
                        bins[src_idx].remove(item)
                        bins[target].append(item)
                        bin_rem[target] -= W[item]
                    bin_rem[src_idx] = bin_capacity
                    improved = True
                    continue
                
                # Swap-assisted bin emptying
                # Try again but when an item fails to place, try 1-1 swap to create space
                if time.time() >= deadline:
                    break
                
                temp_rem2 = bin_rem[:]
                temp_bins_items = [list(b) for b in bins]  # copy
                success = True
                moves2 = []  # list of operations
                
                for item in src_items:
                    w = W[item]
                    # Try direct placement first
                    best_bin = -1
                    best_r = bin_capacity + 1
                    for b in range(num_bins):
                        if b == src_idx:
                            continue
                        if not temp_bins_items[b]:
                            continue
                        if temp_rem2[b] >= w:
                            r = temp_rem2[b] - w
                            if r < best_r:
                                best_r = r
                                best_bin = b
                    
                    if best_bin >= 0:
                        moves2.append(('move', item, src_idx, best_bin))
                        temp_rem2[best_bin] -= w
                        temp_bins_items[src_idx].remove(item)
                        temp_bins_items[best_bin].append(item)
                    else:
                        # Try 1-1 swap: find item_b in some bin b such that
                        # W[item_b] < w and temp_rem2[b] + W[item_b] >= w
                        # i.e., W[item_b] >= w - temp_rem2[b]
                        # And we need to place item_b somewhere else (or in src)
                        found_swap = False
                        # Collect candidates
                        swap_candidates = []
                        for b in range(num_bins):
                            if b == src_idx or not temp_bins_items[b]:
                                continue
                            needed = w - temp_rem2[b]
                            if needed <= 0:
                                continue  # already handled above
                            for item_b in temp_bins_items[b]:
                                wb = W[item_b]
                                if wb >= needed and wb < w:
                                    # Swap: remove item_b from b, place item in b
                                    # item_b needs to go somewhere (src is being emptied, so to another bin)
                                    # Check if item_b can go to src (will be emptied later)
                                    # Actually place item_b in src for now - it will be handled in subsequent iterations
                                    # No, we're trying to empty src. item_b must go to some other bin.
                                    new_rem_b = temp_rem2[b] + wb - w
                                    # Find a place for item_b
                                    best_for_b = -1
                                    best_r_b = bin_capacity + 1
                                    for c in range(num_bins):
                                        if c == src_idx or c == b:
                                            continue
                                        if not temp_bins_items[c] and c != b:
                                            continue
                                        if temp_rem2[c] >= wb:
                                            r_c = temp_rem2[c] - wb
                                            if r_c < best_r_b:
                                                best_r_b = r_c
                                                best_for_b = c
                                    if best_for_b >= 0:
                                        swap_candidates.append((wb - needed, item_b, b, best_for_b, new_rem_b))
                        
                        if swap_candidates:
                            # Pick the best swap (smallest waste)
                            swap_candidates.sort(key=lambda x: x[0])
                            _, item_b, b, dest_b, new_rem_b = swap_candidates[0]
                            wb = W[item_b]
                            # Perform swap
                            moves2.append(('swap', item, src_idx, b, item_b, dest_b))
                            temp_bins_items[src_idx].remove(item)
                            temp_bins_items[b].remove(item_b)
                            temp_bins_items[b].append(item)
                            temp_bins_items[dest_b].append(item_b)
                            temp_rem2[b] = new_rem_b
                            temp_rem2[dest_b] -= wb
                            found_swap = True
                        
                        if not found_swap:
                            success = False
                            break
                
                if success:
                    # Apply all moves to actual bins
                    # Rebuild from temp state
                    bins_backup = bins
                    bin_rem_backup = bin_rem
                    for op in moves2:
                        if op[0] == 'move':
                            _, item, src, dst = op
                            if item in bins[src]:
                                bins[src].remove(item)
                                bins[dst].append(item)
                                bin_rem[src] += W[item]
                                bin_rem[dst] -= W[item]
                        elif op[0] == 'swap':
                            _, item, src, b, item_b, dest_b = op
                            if item in bins[src]:
                                bins[src].remove(item)
                                bin_rem[src] += W[item]
                            if item_b in bins[b]:
                                bins[b].remove(item_b)
                                bin_rem[b] += W[item_b]
                            bins[b].append(item)
                            bin_rem[b] -= W[item]
                            bins[dest_b].append(item_b)
                            bin_rem[dest_b] -= W[item_b]
                    improved = True
            
            # Remove empty bins
            new_bins = []
            new_rem = []
            for b in range(len(bins)):
                if bins[b]:
                    new_bins.append(bins[b])
                    new_rem.append(bin_rem[b])
            bins = new_bins
            bin_rem = new_rem
            
            if not improved and time.time() < deadline:
                # Phase 2: Single item relocations
                num_bins = len(bins)
                for src in range(num_bins):
                    if time.time() >= deadline:
                        break
                    for item in bins[src][:]:
                        w = W[item]
                        best_bin = -1
                        best_r = bin_rem[src]  # only move if tighter fit at destination
                        for dst in range(num_bins):
                            if dst == src:
                                continue
                            if bin_rem[dst] >= w:
                                r = bin_rem[dst] - w
                                if r < best_r:
                                    best_r = r
                                    best_bin = dst
                        if best_bin >= 0:
                            bins[src].remove(item)
                            bin_rem[src] += w
                            bins[best_bin].append(item)
                            bin_rem[best_bin] -= w
                            improved = True
            
            if not improved and time.time() < deadline:
                # Phase 3: 1-1 swaps to concentrate free space
                num_bins = len(bins)
                for a in range(num_bins):
                    if time.time() >= deadline or improved:
                        break
                    for b in range(a + 1, num_bins):
                        if time.time() >= deadline or improved:
                            break
                        for item_a in bins[a]:
                            if improved:
                                break
                            wa = W[item_a]
                            for item_b in bins[b]:
                                wb = W[item_b]
                                diff = wa - wb
                                new_rem_a = bin_rem[a] + diff
                                new_rem_b = bin_rem[b] - diff
                                if new_rem_a >= 0 and new_rem_b >= 0 and diff != 0:
                                    old_max = max(bin_rem[a], bin_rem[b])
                                    new_max = max(new_rem_a, new_rem_b)
                                    if new_max > old_max:
                                        bins[a].remove(item_a)
                                        bins[b].remove(item_b)
                                        bins[a].append(item_b)
                                        bins[b].append(item_a)
                                        bin_rem[a] = new_rem_a
                                        bin_rem[b] = new_rem_b
                                        improved = True
                                        break
            
            if not improved and time.time() < deadline:
                # Phase 4: (2,1) moves
                num_bins = len(bins)
                for a in range(num_bins):
                    if time.time() >= deadline or improved:
                        break
                    for b in range(num_bins):
                        if a == b or time.time() >= deadline or improved:
                            continue
                        # Move items i,j from a to b, move item k from b to a
                        for ii in range(len(bins[a])):
                            if improved:
                                break
                            item_i = bins[a][ii]
                            wi = W[item_i]
                            for jj in range(ii + 1, len(bins[a])):
                                if improved:
                                    break
                                item_j = bins[a][jj]
                                wj = W[item_j]
                                # Need: bin_rem[b] + wk >= wi + wj, bin_rem[a] + wi + wj >= wk
                                needed_in_b = wi + wj - bin_rem[b]
                                if needed_in_b <= 0:
                                    continue  # direct move would work, handled elsewhere
                                space_in_a = bin_rem[a] + wi + wj
                                for item_k in bins[b]:
                                    wk = W[item_k]
                                    if wk >= needed_in_b and wk <= space_in_a:
                                        new_rem_a = bin_rem[a] + wi + wj - wk
                                        new_rem_b = bin_rem[b] + wk - wi - wj
                                        if new_rem_a >= 0 and new_rem_b >= 0:
                                            old_max = max(bin_rem[a], bin_rem[b])
                                            new_max = max(new_rem_a, new_rem_b)
                                            if new_max > old_max:
                                                bins[a].remove(item_i)
                                                bins[a].remove(item_j)
                                                bins[b].remove(item_k)
                                                bins[a].append(item_k)
                                                bins[b].append(item_i)
                                                bins[b].append(item_j)
                                                bin_rem[a] = new_rem_a
                                                bin_rem[b] = new_rem_b
                                                improved = True
                                                break
        
        # Final cleanup
        final_bins = [b for b in bins if b]
        final_rem = [bin_rem[i] for i in range(len(bins)) if bins[i]]
        return final_bins, final_rem
    
    # FFD baseline with best-fit
    sorted_items = sorted(range(n), key=lambda i: -W[i])
    best_bins = []
    best_rem = []
    sorted_rem_aux = []  # (remaining, bin_idx)
    
    for item in sorted_items:
        w = W[item]
        pos = bisect.bisect_left(sorted_rem_aux, (w, -1))
        if pos < len(sorted_rem_aux):
            rem, bidx = sorted_rem_aux[pos]
            sorted_rem_aux.pop(pos)
            new_rem = rem - w
            best_bins[bidx].append(item)
            best_rem[bidx] = new_rem
            bisect.insort(sorted_rem_aux, (new_rem, bidx))
        else:
            bidx = len(best_bins)
            best_bins.append([item])
            new_rem = bin_capacity - w
            best_rem.append(new_rem)
            bisect.insort(sorted_rem_aux, (new_rem, bidx))
    
    best_num_bins = len(best_bins)
    
    if best_num_bins <= lower_bound:
        packing = best_bins
        bin_weights_out = [bin_capacity - best_rem[b] for b in range(len(best_bins))]
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Initial local search on FFD (5% of time)
    ls_deadline = min(start_time + time_limit * 0.05, start_time + time_limit - 0.05)
    best_bins, best_rem = local_search_full(best_bins, best_rem, ls_deadline)
    best_num_bins = len(best_bins)
    
    if best_num_bins <= lower_bound:
        packing = best_bins
        bin_weights_out = [bin_capacity - best_rem[b] for b in range(len(best_bins))]
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Reactive GRASP
    alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    num_alphas = len(alpha_values)
    alpha_scores = [1.0] * num_alphas
    alpha_counts = [0] * num_alphas
    alpha_total_scores = [0.0] * num_alphas
    alpha_probs = [1.0 / num_alphas] * num_alphas
    
    def select_alpha():
        r = random.random()
        cumsum = 0.0
        for i in range(num_alphas):
            cumsum += alpha_probs[i]
            if r <= cumsum:
                return i
        return num_alphas - 1
    
    iteration = 0
    update_interval = 10
    
    final_ls_time = min(time_limit * 0.05, 2.0)
    grasp_deadline = start_time + time_limit - final_ls_time - 0.05
    
    while time.time() < grasp_deadline:
        iteration += 1
        
        # Select alpha using reactive probabilities
        alpha_idx = select_alpha()
        alpha = alpha_values[alpha_idx]
        
        # Construction phase - choose variant
        variant = random.randint(0, 2)
        bins, bin_rem = construct_fast(alpha, variant)
        
        # Local search phase
        remaining_time = grasp_deadline - time.time()
        if remaining_time < 0.01:
            if len(bins) < best_num_bins:
                best_bins = [b[:] for b in bins]
                best_rem = bin_rem[:]
                best_num_bins = len(bins)
            break
        
        ls_time = min(remaining_time * 0.1, 1.0)
        ls_dead = time.time() + ls_time
        
        bins, bin_rem = local_search_full(bins, bin_rem, ls_dead)
        
        num_result_bins = len(bins)
        
        # Update reactive scores
        if num_result_bins > 0:
            score = best_num_bins / num_result_bins
        else:
            score = 1.0
        alpha_counts[alpha_idx] += 1
        alpha_total_scores[alpha_idx] += score
        
        # Update probabilities every update_interval iterations
        if iteration % update_interval == 0:
            avg_scores = []
            for i in range(num_alphas):
                if alpha_counts[i] > 0:
                    avg_scores.append(alpha_total_scores[i] / alpha_counts[i])
                else:
                    avg_scores.append(1.0)
            total_score = sum(avg_scores)
            if total_score > 0:
                alpha_probs = [s / total_score for s in avg_scores]
        
        # Update best
        if num_result_bins < best_num_bins:
            best_bins = [b[:] for b in bins]
            best_rem = bin_rem[:]
            best_num_bins = num_result_bins
            
            if best_num_bins <= lower_bound:
                break
    
    # Final intensive local search
    final_deadline = start_time + time_limit - 0.05
    if time.time() < final_deadline:
        best_bins, best_rem = local_search_full(best_bins, best_rem, final_deadline)
        best_num_bins = len(best_bins)
    
    # Build output
    packing = best_bins
    bin_weights_out = [sum(W[i] for i in b) for b in packing]
    
    return {"packing": packing, "bin_weights": bin_weights_out}