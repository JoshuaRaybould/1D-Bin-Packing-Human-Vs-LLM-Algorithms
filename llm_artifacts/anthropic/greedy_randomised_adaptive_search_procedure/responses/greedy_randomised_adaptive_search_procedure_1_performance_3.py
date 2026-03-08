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
    
    # Lower bounds
    total_weight = sum(W)
    lower_bound_l1 = math.ceil(total_weight / bin_capacity)
    
    # L2 lower bound
    def compute_l2():
        best_lb = lower_bound_l1
        for t in range(1, bin_capacity // 2 + 1):
            if t > 100 and t % 5 != 0:
                continue
            if t > 500 and t % 20 != 0:
                continue
            n1 = 0  # items > C - t
            n2 = 0  # items in (t, C-t]
            s2 = 0
            s3 = 0  # items <= t
            half = bin_capacity / 2.0
            ct = bin_capacity - t
            for w in W:
                if w > ct:
                    n1 += 1
                elif w > t:
                    n2 += 1
                    s2 += w
                else:
                    s3 += w
            # Each n1 item needs own bin. Remaining capacity in n1 bins = sum(C - w) for large items
            # But simplified L2: lb = n1 + max(n2, ceil((s2+s3)/C))
            # Actually use the standard L2 formula
            lb = n1 + n2 + max(0, math.ceil((s3 - (n1 + n2) * bin_capacity + sum(w for w in W if w > t)) / bin_capacity))
            # Simpler: Fekete-Schepers style
            # L2(t) = n1 + n2 + max(0, ceil((s3 - sum_{large}(C - w_i) - (n2*C - s2) + s2) / C))
            # Let me just use a cleaner formulation
            pass
            best_lb = max(best_lb, lb)
        
        # Simpler but effective: try specific thresholds
        best_lb2 = lower_bound_l1
        for denom in range(2, 20):
            t = bin_capacity // denom
            if t == 0:
                continue
            ct = bin_capacity - t
            large = []
            medium = []
            small_total = 0
            for w in W:
                if w > ct:
                    large.append(w)
                elif w > t:
                    medium.append(w)
                else:
                    small_total += w
            n_large = len(large)
            space_in_large = sum(bin_capacity - w for w in large)
            medium.sort()
            used = 0
            medium_remaining = []
            for mw in medium:
                if used + mw <= space_in_large:
                    used += mw
                else:
                    medium_remaining.append(mw)
            n_medium_bins = math.ceil(len(medium_remaining) / 2) if medium_remaining else 0
            medium_remaining_weight = sum(medium_remaining)
            space_after_large = space_in_large - used
            space_in_medium_bins = n_medium_bins * bin_capacity - medium_remaining_weight if n_medium_bins > 0 else 0
            total_space_for_small = space_after_large + space_in_medium_bins
            remaining_small = max(0, small_total - total_space_for_small)
            extra_bins = math.ceil(remaining_small / bin_capacity) if remaining_small > 0 else 0
            lb = n_large + n_medium_bins + extra_bins
            best_lb2 = max(best_lb2, lb)
        
        return max(best_lb, best_lb2)
    
    lower_bound = compute_l2()
    
    def bfd_construct(order):
        """Best-fit decreasing placement given an item order."""
        bins = []
        bin_rem = []
        sorted_rem = []  # (remaining, bin_idx)
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
    
    def construct_grasp(alpha):
        """GRASP construction: RCL on item selection, best-fit placement."""
        sorted_desc = sorted(range(n), key=lambda i: -W[i])
        remaining = list(sorted_desc)
        order = []
        while remaining:
            w_max = W[remaining[0]]
            w_min = W[remaining[-1]]
            threshold = w_max - alpha * (w_max - w_min)
            # Binary search for cutoff in descending-sorted list
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
        return bfd_construct(order)
    
    def construct_grasp_bin_choice(alpha):
        """GRASP construction: items sorted descending, RCL on bin selection."""
        sorted_desc = sorted(range(n), key=lambda i: -W[i])
        bins = []
        bin_rem = []
        
        for item in sorted_desc:
            w = W[item]
            # Find all feasible bins
            feasible = []
            for b in range(len(bins)):
                if bin_rem[b] >= w:
                    feasible.append((bin_rem[b] - w, b))  # (waste, bin_idx)
            
            if not feasible:
                bidx = len(bins)
                bins.append([item])
                bin_rem.append(bin_capacity - w)
            else:
                feasible.sort()
                waste_min = feasible[0][0]
                waste_max = feasible[-1][0]
                threshold = waste_min + alpha * (waste_max - waste_min)
                rcl = [f for f in feasible if f[0] <= threshold]
                if not rcl:
                    rcl = [feasible[0]]
                _, chosen_bin = random.choice(rcl)
                bins[chosen_bin].append(item)
                bin_rem[chosen_bin] -= w
        
        return bins, bin_rem
    
    def local_search(bins, bin_rem, deadline):
        """Focused local search: try to empty bins."""
        improved = True
        passes = 0
        while improved and time.time() < deadline:
            improved = False
            passes += 1
            num_bins = len(bins)
            if num_bins <= lower_bound:
                break
            
            # Sort bins by ascending load (try to empty lightest first)
            loads = [bin_capacity - bin_rem[b] for b in range(num_bins)]
            bin_order = sorted(range(num_bins), key=lambda b: loads[b])
            
            emptied = set()
            
            for src_idx in bin_order:
                if time.time() >= deadline:
                    break
                if src_idx in emptied or not bins[src_idx]:
                    continue
                
                src_load = loads[src_idx]
                if src_load == 0:
                    emptied.add(src_idx)
                    continue
                
                # Try to redistribute items from src using best-fit
                src_items = sorted(bins[src_idx], key=lambda i: -W[i])
                can_empty = True
                moves = []
                temp_rem = {}
                
                for item in src_items:
                    w = W[item]
                    best_bin = -1
                    best_waste = bin_capacity + 1
                    for b in range(num_bins):
                        if b == src_idx or b in emptied or not bins[b]:
                            continue
                        r = temp_rem.get(b, bin_rem[b])
                        if r >= w:
                            waste = r - w
                            if waste < best_waste:
                                best_waste = waste
                                best_bin = b
                    if best_bin >= 0:
                        moves.append((item, best_bin))
                        temp_rem[best_bin] = temp_rem.get(best_bin, bin_rem[best_bin]) - w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    for item, target in moves:
                        bins[target].append(item)
                        bin_rem[target] -= W[item]
                    bins[src_idx] = []
                    bin_rem[src_idx] = bin_capacity
                    emptied.add(src_idx)
                    improved = True
                    continue
                
                # Swap-assisted: for each failing item, try 1-1 swap
                if time.time() >= deadline:
                    break
                
                temp_rem2 = dict()
                temp_additions = {}  # bin -> list of items added
                temp_removals = {}  # bin -> list of items removed
                success = True
                
                for item in src_items:
                    w = W[item]
                    # Direct placement
                    best_bin = -1
                    best_waste = bin_capacity + 1
                    for b in range(num_bins):
                        if b == src_idx or b in emptied:
                            continue
                        if not bins[b] and b not in temp_additions:
                            continue
                        r = temp_rem2.get(b, bin_rem[b])
                        if r >= w:
                            waste = r - w
                            if waste < best_waste:
                                best_waste = waste
                                best_bin = b
                    
                    if best_bin >= 0:
                        temp_rem2[best_bin] = temp_rem2.get(best_bin, bin_rem[best_bin]) - w
                        if best_bin not in temp_additions:
                            temp_additions[best_bin] = []
                        temp_additions[best_bin].append(item)
                    else:
                        # Try 1-1 swap
                        found = False
                        best_swap = None
                        best_swap_score = float('inf')
                        
                        for b in range(num_bins):
                            if b == src_idx or b in emptied:
                                continue
                            rb = temp_rem2.get(b, bin_rem[b])
                            needed = w - rb
                            if needed <= 0:
                                continue
                            
                            # Get current items in bin b
                            removals_b = set(temp_removals.get(b, []))
                            candidates = [it for it in bins[b] if it not in removals_b]
                            if b in temp_additions:
                                candidates.extend(temp_additions[b])
                            
                            for item_b in candidates:
                                wb = W[item_b]
                                if wb < needed or wb >= w:
                                    continue
                                # item_b freed, item placed in b
                                new_rb = rb + wb - w
                                if new_rb < 0:
                                    continue
                                # item_b needs placement elsewhere
                                best_c = -1
                                best_cw = bin_capacity + 1
                                for c in range(num_bins):
                                    if c == src_idx or c == b or c in emptied:
                                        continue
                                    if not bins[c] and c not in temp_additions:
                                        continue
                                    rc = temp_rem2.get(c, bin_rem[c])
                                    if rc >= wb:
                                        cw = rc - wb
                                        if cw < best_cw:
                                            best_cw = cw
                                            best_c = c
                                
                                if best_c >= 0:
                                    score = best_cw  # prefer tight fit
                                    if score < best_swap_score:
                                        best_swap_score = score
                                        best_swap = (item_b, b, best_c, wb, new_rb)
                        
                        if best_swap is not None:
                            item_b, b, c, wb, new_rb = best_swap
                            temp_rem2[b] = new_rb
                            temp_rem2[c] = temp_rem2.get(c, bin_rem[c]) - wb
                            if b not in temp_additions:
                                temp_additions[b] = []
                            temp_additions[b].append(item)
                            if b not in temp_removals:
                                temp_removals[b] = []
                            temp_removals[b].append(item_b)
                            if c not in temp_additions:
                                temp_additions[c] = []
                            temp_additions[c].append(item_b)
                            found = True
                        
                        if not found:
                            success = False
                            break
                
                if success:
                    # Apply changes
                    for b, items_to_add in temp_additions.items():
                        for it in items_to_add:
                            bins[b].append(it)
                            bin_rem[b] -= W[it]
                    for b, items_to_remove in temp_removals.items():
                        for it in items_to_remove:
                            bins[b].remove(it)
                            bin_rem[b] += W[it]
                    bins[src_idx] = []
                    bin_rem[src_idx] = bin_capacity
                    emptied.add(src_idx)
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
                # Phase 2: relocations to consolidate
                num_bins = len(bins)
                for src in range(num_bins):
                    if time.time() >= deadline:
                        break
                    for item in bins[src][:]:
                        w = W[item]
                        best_bin = -1
                        best_r = bin_rem[src]  # only move if tighter
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
                
                # Try 1-1 swaps to concentrate free space
                if not improved and time.time() < deadline:
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
                                    if diff == 0:
                                        continue
                                    new_rem_a = bin_rem[a] + diff
                                    new_rem_b = bin_rem[b] - diff
                                    if new_rem_a >= 0 and new_rem_b >= 0:
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
        
        return bins, bin_rem
    
    # BFD baseline
    sorted_items = sorted(range(n), key=lambda i: -W[i])
    best_bins, best_rem = bfd_construct(sorted_items)
    best_num_bins = len(best_bins)
    
    if best_num_bins <= lower_bound:
        packing = best_bins
        bin_weights_out = [bin_capacity - best_rem[b] for b in range(len(best_bins))]
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Initial local search on BFD
    ls_deadline = min(start_time + time_limit * 0.03, start_time + time_limit - 0.1)
    best_bins, best_rem = local_search(best_bins, best_rem, ls_deadline)
    best_num_bins = len(best_bins)
    
    if best_num_bins <= lower_bound:
        packing = best_bins
        bin_weights_out = [bin_capacity - best_rem[b] for b in range(len(best_bins))]
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Reactive GRASP
    alpha_values = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    num_alphas = len(alpha_values)
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
    update_interval = 20
    
    grasp_deadline = start_time + time_limit - 0.1
    
    # Adaptive time allocation for local search
    while time.time() < grasp_deadline:
        iteration += 1
        
        alpha_idx = select_alpha()
        alpha = alpha_values[alpha_idx]
        
        # Alternate construction strategies
        if iteration % 3 == 0:
            bins, bin_rem = construct_grasp_bin_choice(alpha)
        else:
            bins, bin_rem = construct_grasp(alpha)
        
        current_bins = len(bins)
        
        # Only do local search if solution is promising
        remaining_time = grasp_deadline - time.time()
        if remaining_time < 0.005:
            if current_bins < best_num_bins:
                best_bins = bins
                best_rem = bin_rem
                best_num_bins = current_bins
            break
        
        # Adaptive LS time: more time for promising solutions
        if current_bins <= best_num_bins + 1:
            ls_time = min(remaining_time * 0.15, 2.0)
        elif current_bins <= best_num_bins + 3:
            ls_time = min(remaining_time * 0.05, 0.5)
        else:
            ls_time = min(remaining_time * 0.01, 0.1)
        
        ls_dead = time.time() + ls_time
        bins, bin_rem = local_search(bins, bin_rem, ls_dead)
        num_result_bins = len(bins)
        
        # Update reactive scores
        score = best_num_bins / max(num_result_bins, 1)
        alpha_counts[alpha_idx] += 1
        alpha_total_scores[alpha_idx] += score
        
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
        
        if num_result_bins < best_num_bins:
            best_bins = [b[:] for b in bins]
            best_rem = bin_rem[:]
            best_num_bins = num_result_bins
            
            if best_num_bins <= lower_bound:
                break
    
    # Final intensive local search with remaining time
    final_deadline = start_time + time_limit - 0.05
    if time.time() < final_deadline:
        best_bins, best_rem = local_search(best_bins, best_rem, final_deadline)
        best_num_bins = len(best_bins)
    
    packing = best_bins
    bin_weights_out = [sum(W[i] for i in b) for b in packing]
    
    return {"packing": packing, "bin_weights": bin_weights_out}
