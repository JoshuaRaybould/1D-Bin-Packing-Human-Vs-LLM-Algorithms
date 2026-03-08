import random
import time
from bisect import insort, bisect_left

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    w = weights  # alias
    
    def elapsed():
        return time.time() - start_time
    
    # Lower bound
    total_weight = sum(w)
    lb = (total_weight + bin_capacity - 1) // bin_capacity
    
    # ---- Efficient Best Fit Decreasing ----
    def bfd_solution(order):
        """Place items in given order using best-fit."""
        assign = [0] * n
        bin_rem = []
        bin_count = 0
        
        for item in order:
            wi = w[item]
            best_b = -1
            best_r = bin_capacity + 1
            for b in range(bin_count):
                r = bin_rem[b]
                if r >= wi:
                    leftover = r - wi
                    if leftover < best_r:
                        best_r = leftover
                        best_b = b
            if best_b >= 0:
                assign[item] = best_b
                bin_rem[best_b] -= wi
            else:
                assign[item] = bin_count
                bin_rem.append(bin_capacity - wi)
                bin_count += 1
        
        return assign, bin_rem, bin_count
    
    def solution_from_assign(assign, bin_rem, bin_count):
        """Convert assignment array to bins structure."""
        bins = [[] for _ in range(bin_count)]
        for i in range(n):
            bins[assign[i]].append(i)
        return bins, list(bin_rem)
    
    def compact_solution(bins, bin_rem):
        """Remove empty bins and reindex."""
        new_bins = []
        new_rem = []
        for i in range(len(bins)):
            if bins[i]:
                new_bins.append(bins[i])
                new_rem.append(bin_rem[i])
        return new_bins, new_rem
    
    # ---- GRASP Construction ----
    def construct_grasp(alpha):
        """GRASP construction: RCL on item weights (prefer heavy), best-fit placement."""
        remaining = list(range(n))
        # Sort by weight descending for efficiency
        remaining.sort(key=lambda i: -w[i])
        
        assign = [0] * n
        bin_rem = []
        bin_count = 0
        
        while remaining:
            # RCL based on weights
            w_max = w[remaining[0]]
            w_min = w[remaining[-1]]
            threshold = w_max - alpha * (w_max - w_min)
            
            # Find RCL candidates (those with weight >= threshold)
            # Since remaining is sorted descending, find cutoff
            rcl_end = len(remaining)
            for k in range(len(remaining)):
                if w[remaining[k]] < threshold:
                    rcl_end = k
                    break
            if rcl_end == 0:
                rcl_end = 1
            
            # Pick random from RCL
            idx = random.randint(0, rcl_end - 1)
            item = remaining[idx]
            # Remove efficiently
            remaining[idx] = remaining[-1]
            remaining.pop()
            # Re-sort periodically or maintain sorted order
            # Actually, after swap the order is broken. Let's just keep it simple.
            # We'll re-sort the remaining list partially
            # For speed, just don't re-sort - the RCL will still work approximately
            
            wi = w[item]
            # Best fit
            best_b = -1
            best_r = bin_capacity + 1
            for b in range(bin_count):
                r = bin_rem[b]
                if r >= wi:
                    leftover = r - wi
                    if leftover < best_r:
                        best_r = leftover
                        best_b = b
            if best_b >= 0:
                assign[item] = best_b
                bin_rem[best_b] -= wi
            else:
                assign[item] = bin_count
                bin_rem.append(bin_capacity - wi)
                bin_count += 1
        
        return assign, bin_rem, bin_count
    
    def construct_grasp_v2(alpha):
        """GRASP construction v2: sorted by weight desc with RCL, best-fit."""
        sorted_items = sorted(range(n), key=lambda i: -w[i])
        order = []
        remaining = list(sorted_items)
        
        while remaining:
            w_max = w[remaining[0]]
            w_min = w[remaining[-1]]
            threshold = w_max - alpha * (w_max - w_min)
            
            rcl_end = len(remaining)
            for k in range(len(remaining)):
                if w[remaining[k]] < threshold:
                    rcl_end = k
                    break
            if rcl_end == 0:
                rcl_end = 1
            
            idx = random.randint(0, rcl_end - 1)
            order.append(remaining[idx])
            remaining.pop(idx)
        
        return bfd_solution(order)
    
    # ---- Local Search ----
    def local_search(bins, bin_rem, deadline):
        """Local search: try to empty bins by redistributing items."""
        num_bins = len(bins)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            num_bins = len(bins)
            
            # Calculate loads
            loads = [bin_capacity - bin_rem[b] for b in range(num_bins)]
            
            # Sort bins by load ascending (try to empty lightest first)
            bin_order = sorted(range(num_bins), key=lambda b: loads[b])
            
            for src_idx in bin_order:
                if time.time() >= deadline:
                    break
                if not bins[src_idx]:
                    continue
                
                src_load = loads[src_idx]
                # Quick check: enough total remaining capacity?
                total_rem_others = sum(bin_rem[b] for b in range(num_bins) if b != src_idx and bins[b])
                if total_rem_others < src_load:
                    continue
                
                # Try to redistribute all items from src
                src_items = sorted(bins[src_idx], key=lambda i: -w[i])
                can_empty = True
                moves = []
                temp_rem = list(bin_rem)
                
                for item in src_items:
                    wi = w[item]
                    best_b = -1
                    best_r = bin_capacity + 1
                    for b in range(num_bins):
                        if b == src_idx or not bins[b]:
                            continue
                        if temp_rem[b] >= wi:
                            leftover = temp_rem[b] - wi
                            if leftover < best_r:
                                best_r = leftover
                                best_b = b
                    if best_b >= 0:
                        moves.append((item, best_b))
                        temp_rem[best_b] -= wi
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    # Apply moves
                    for item, target in moves:
                        bins[target].append(item)
                        bin_rem[target] -= w[item]
                    bins[src_idx] = []
                    bin_rem[src_idx] = bin_capacity
                    loads[src_idx] = 0
                    improved = True
            
            # Compact
            bins, bin_rem = compact_solution(bins, bin_rem)
            num_bins = len(bins)
            
            if not improved:
                # Try swap-assisted bin emptying
                # For each light bin, try swapping items to make redistribution possible
                loads = [bin_capacity - bin_rem[b] for b in range(num_bins)]
                bin_order = sorted(range(num_bins), key=lambda b: loads[b])
                
                for src_idx in bin_order[:max(1, num_bins // 4)]:
                    if time.time() >= deadline:
                        break
                    if improved:
                        break
                    if not bins[src_idx]:
                        continue
                    
                    src_load = loads[src_idx]
                    total_rem_others = sum(bin_rem[b] for b in range(num_bins) if b != src_idx)
                    if total_rem_others < src_load:
                        continue
                    
                    # Try different orderings of items to redistribute
                    src_items = list(bins[src_idx])
                    for attempt in range(min(10, max(1, len(src_items) * 2))):
                        if time.time() >= deadline:
                            break
                        if attempt == 0:
                            order = sorted(src_items, key=lambda i: -w[i])
                        else:
                            order = list(src_items)
                            random.shuffle(order)
                        
                        can_empty = True
                        moves = []
                        temp_rem = list(bin_rem)
                        
                        for item in order:
                            wi = w[item]
                            best_b = -1
                            best_r = bin_capacity + 1
                            for b in range(num_bins):
                                if b == src_idx:
                                    continue
                                if temp_rem[b] >= wi:
                                    leftover = temp_rem[b] - wi
                                    if leftover < best_r:
                                        best_r = leftover
                                        best_b = b
                            if best_b >= 0:
                                moves.append((item, best_b))
                                temp_rem[best_b] -= wi
                            else:
                                can_empty = False
                                break
                        
                        if can_empty:
                            for item, target in moves:
                                bins[target].append(item)
                                bin_rem[target] -= w[item]
                            bins[src_idx] = []
                            bin_rem[src_idx] = bin_capacity
                            improved = True
                            break
                
                if improved:
                    bins, bin_rem = compact_solution(bins, bin_rem)
                    continue
                
                # Try 1-1 swaps to improve packing balance
                # This can enable future bin emptying
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
                            wa = w[item_a]
                            for item_b in bins[b]:
                                wb = w[item_b]
                                diff = wa - wb
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
                
                if not improved:
                    # Try 1-0 relocations that increase max remaining
                    for src in range(num_bins):
                        if time.time() >= deadline or improved:
                            break
                        for item in list(bins[src]):
                            wi = w[item]
                            for dst in range(num_bins):
                                if dst == src:
                                    continue
                                if bin_rem[dst] >= wi:
                                    new_rem_dst = bin_rem[dst] - wi
                                    new_rem_src = bin_rem[src] + wi
                                    # Only if it creates more free space in src
                                    if new_rem_src > bin_rem[src] and new_rem_dst < bin_rem[src]:
                                        bins[src].remove(item)
                                        bins[dst].append(item)
                                        bin_rem[src] = new_rem_src
                                        bin_rem[dst] = new_rem_dst
                                        improved = True
                                        break
                            if improved:
                                break
        
        bins, bin_rem = compact_solution(bins, bin_rem)
        return bins, bin_rem
    
    def local_search_deep(bins, bin_rem, deadline):
        """Deep local search with swap-assisted emptying using item-to-bin mapping."""
        # Build item-to-bin mapping
        num_bins = len(bins)
        item_bin = [0] * n
        bin_sets = [set(b) for b in bins]
        bin_rem_arr = list(bin_rem)
        
        for b in range(num_bins):
            for item in bins[b]:
                item_bin[item] = b
        
        def try_empty_bin(src):
            """Try to empty bin src by redistributing its items."""
            if not bin_sets[src]:
                return False
            
            items = sorted(bin_sets[src], key=lambda i: -w[i])
            src_load = sum(w[i] for i in items)
            
            # Quick feasibility check
            total_rem = sum(bin_rem_arr[b] for b in range(len(bin_sets)) if b != src and bin_sets[b])
            if total_rem < src_load:
                return False
            
            # Try multiple orderings
            for attempt in range(min(20, max(1, len(items) * 3))):
                if time.time() >= deadline:
                    return False
                
                if attempt == 0:
                    order = sorted(items, key=lambda i: -w[i])
                elif attempt == 1:
                    order = sorted(items, key=lambda i: w[i])
                else:
                    order = list(items)
                    random.shuffle(order)
                
                moves = []
                temp_rem = list(bin_rem_arr)
                ok = True
                
                for item in order:
                    wi = w[item]
                    best_b = -1
                    best_r = bin_capacity + 1
                    for b in range(len(bin_sets)):
                        if b == src or not bin_sets[b]:
                            continue
                        if temp_rem[b] >= wi:
                            leftover = temp_rem[b] - wi
                            if leftover < best_r:
                                best_r = leftover
                                best_b = b
                    if best_b >= 0:
                        moves.append((item, best_b))
                        temp_rem[best_b] -= wi
                    else:
                        ok = False
                        break
                
                if ok:
                    for item, target in moves:
                        bin_sets[src].discard(item)
                        bin_sets[target].add(item)
                        bin_rem_arr[target] -= w[item]
                        item_bin[item] = target
                    bin_rem_arr[src] = bin_capacity
                    return True
            
            return False
        
        def do_swap(item_a, bin_a, item_b, bin_b):
            """Perform a 1-1 swap."""
            wa_val = w[item_a]
            wb_val = w[item_b]
            bin_sets[bin_a].discard(item_a)
            bin_sets[bin_b].discard(item_b)
            bin_sets[bin_a].add(item_b)
            bin_sets[bin_b].add(item_a)
            bin_rem_arr[bin_a] += wa_val - wb_val
            bin_rem_arr[bin_b] += wb_val - wa_val
            item_bin[item_a] = bin_b
            item_bin[item_b] = bin_a
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            active_bins = [b for b in range(len(bin_sets)) if bin_sets[b]]
            
            # Sort by load ascending
            active_bins.sort(key=lambda b: bin_capacity - bin_rem_arr[b])
            
            for src in active_bins:
                if time.time() >= deadline:
                    break
                if not bin_sets[src]:
                    continue
                if try_empty_bin(src):
                    improved = True
            
            if not improved:
                # Try swaps that increase max remaining capacity
                active_bins = [b for b in range(len(bin_sets)) if bin_sets[b]]
                num_active = len(active_bins)
                
                for ai in range(num_active):
                    if time.time() >= deadline or improved:
                        break
                    a = active_bins[ai]
                    for bi in range(ai + 1, num_active):
                        if time.time() >= deadline or improved:
                            break
                        b = active_bins[bi]
                        items_a = list(bin_sets[a])
                        items_b = list(bin_sets[b])
                        for item_a in items_a:
                            if improved:
                                break
                            wa_val = w[item_a]
                            for item_b in items_b:
                                wb_val = w[item_b]
                                diff = wa_val - wb_val
                                new_rem_a = bin_rem_arr[a] + diff
                                new_rem_b = bin_rem_arr[b] - diff
                                if new_rem_a >= 0 and new_rem_b >= 0:
                                    old_max = max(bin_rem_arr[a], bin_rem_arr[b])
                                    new_max = max(new_rem_a, new_rem_b)
                                    if new_max > old_max:
                                        do_swap(item_a, a, item_b, b)
                                        improved = True
                                        break
                
                if not improved:
                    # Try relocations
                    active_bins = [b for b in range(len(bin_sets)) if bin_sets[b]]
                    for src in active_bins:
                        if time.time() >= deadline or improved:
                            break
                        for item in list(bin_sets[src]):
                            wi = w[item]
                            best_b = -1
                            best_r = bin_rem_arr[src]  # must be tighter fit than current
                            for dst in active_bins:
                                if dst == src:
                                    continue
                                if bin_rem_arr[dst] >= wi:
                                    leftover = bin_rem_arr[dst] - wi
                                    if leftover < best_r:
                                        best_r = leftover
                                        best_b = dst
                            if best_b >= 0 and best_r < bin_rem_arr[src]:
                                bin_sets[src].discard(item)
                                bin_sets[best_b].add(item)
                                bin_rem_arr[src] += wi
                                bin_rem_arr[best_b] -= wi
                                item_bin[item] = best_b
                                improved = True
                                break
        
        # Convert back
        result_bins = []
        result_rem = []
        for b in range(len(bin_sets)):
            if bin_sets[b]:
                result_bins.append(list(bin_sets[b]))
                result_rem.append(bin_rem_arr[b])
        
        return result_bins, result_rem
    
    # ---- Initial FFD + BFD solution ----
    sorted_desc = sorted(range(n), key=lambda i: -w[i])
    assign, bin_rem, bin_count = bfd_solution(sorted_desc)
    best_bins, best_rem = solution_from_assign(assign, bin_rem, bin_count)
    best_num_bins = len(best_bins)
    
    # Apply deep local search to initial solution
    ls_deadline = start_time + min(time_limit * 0.2, time_limit - 0.1)
    best_bins, best_rem = local_search_deep(list(best_bins), list(best_rem), ls_deadline)
    best_num_bins = len(best_bins)
    
    if best_num_bins <= lb:
        bin_weights = [sum(w[i] for i in b) for b in best_bins]
        return {"packing": best_bins, "bin_weights": bin_weights}
    
    # ---- GRASP iterations ----
    alpha_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Track which alphas produce best results (reactive GRASP)
    alpha_scores = {a: 0.0 for a in alpha_values}
    alpha_counts = {a: 1 for a in alpha_values}
    
    iteration = 0
    while True:
        if time.time() - start_time >= time_limit - 0.1:
            break
        
        if best_num_bins <= lb:
            break
        
        iteration += 1
        
        # Choose alpha using reactive selection
        if iteration <= len(alpha_values) * 2:
            alpha = alpha_values[iteration % len(alpha_values)]
        else:
            # Weighted random selection based on scores
            avg_scores = [alpha_scores[a] / alpha_counts[a] for a in alpha_values]
            total = sum(avg_scores)
            if total > 0:
                probs = [s / total for s in avg_scores]
                r = random.random()
                cumsum = 0
                alpha = alpha_values[-1]
                for i, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum:
                        alpha = alpha_values[i]
                        break
            else:
                alpha = random.choice(alpha_values)
        
        # Construction
        remaining_time = time_limit - elapsed()
        if remaining_time < 0.1:
            break
        
        if random.random() < 0.5:
            assign, bin_rem, bin_count = construct_grasp(alpha)
        else:
            assign, bin_rem, bin_count = construct_grasp_v2(alpha)
        
        bins, b_rem = solution_from_assign(assign, bin_rem, bin_count)
        constructed_bins = len(bins)
        
        # Local search
        remaining_time = time_limit - elapsed()
        if remaining_time < 0.1:
            if constructed_bins < best_num_bins:
                best_bins = bins
                best_rem = b_rem
                best_num_bins = constructed_bins
            break
        
        # Allocate time: more time for better solutions
        if constructed_bins <= best_num_bins:
            ls_time = min(remaining_time * 0.5, 2.0)
        else:
            ls_time = min(remaining_time * 0.15, 0.5)
        
        ls_deadline = time.time() + ls_time
        bins, b_rem = local_search_deep(bins, b_rem, ls_deadline)
        
        current_bins = len(bins)
        
        # Update reactive scores
        score = max(0, best_num_bins - current_bins + 1) if current_bins <= best_num_bins else 0.1
        alpha_scores[alpha] += score
        alpha_counts[alpha] += 1
        
        # Update best
        if current_bins < best_num_bins:
            best_bins = [b[:] for b in bins]
            best_rem = list(b_rem)
            best_num_bins = current_bins
            if best_num_bins <= lb:
                break
    
    # Final intensive local search with remaining time
    remaining_time = time_limit - elapsed()
    if remaining_time > 0.2 and best_num_bins > lb:
        final_deadline = start_time + time_limit - 0.05
        best_bins, best_rem = local_search_deep(best_bins, best_rem, final_deadline)
        best_num_bins = len(best_bins)
    
    bin_weights = [sum(w[i] for i in b) for b in best_bins]
    return {"packing": best_bins, "bin_weights": bin_weights}
