import random
import time
import math

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    # --- Phase 1: Lower Bounds ---
    total_weight = sum(weights)
    L1 = math.ceil(total_weight / C)
    
    # L2 lower bound
    def compute_L2():
        best_lb = L1
        # Try several threshold values
        for t in range(1, C // 2 + 1):
            # n1: items with weight > C - t (large items)
            # n2: items with weight in (C/2, C-t] (medium items)
            # n3_items: items with weight <= t (small items that could pair)
            # But also items in (t, C/2] don't fit in any category for pairing
            
            n1 = 0
            n2 = 0
            sum_large = 0  # sum of weights of n2 items
            sum_small = 0
            n_small = 0
            
            for w in weights:
                if w > C - t:
                    n1 += 1
                elif w > C // 2:  # Actually w > C/2 and w <= C-t
                    if C % 2 == 0:
                        if w > C // 2 and w <= C - t:
                            n2 += 1
                            sum_large += w
                    else:
                        if w > C / 2 and w <= C - t:
                            n2 += 1
                            sum_large += w
                elif w <= t:  # small items
                    # Actually we need items with w in (0, t] but also w <= C/2
                    sum_small += w
                    n_small += 1
            
            # Actually let me use the standard Martello-Toth L2
            # For threshold t: items split into 3 groups
            # J1: w > C - t (need own bin, can share with J3)
            # J2: C - t >= w > C/2 (each needs own bin basically, pairs can't fit two J2)
            # J3: w <= t (small, can fill remaining space)
            
            # Residual capacity in J1 bins: sum over J1 items of (C - w_i)
            # But only J3 items (w <= t) can fit in J1 bins
            # Residual in J2 bins: sum over J2 items of (C - w_i), only J3 fit
            
            # L2(t) = n1 + n2 + max(0, ceil((sum_J3 - residual_J1_for_J3 - residual_J2) / C))
            # But we need to be more careful. Let me just use the simpler formula:
            # Total bins needed >= n1 + n2 + max(0, ceil((sum_J3 - (n1*C - sum_J1) - (n2*C - sum_J2)) / C))
            # Wait, not all J3 items fit in J1 residual (only those with w <= t fit since J1 has C - w > t... wait no)
            # J1 items have w > C-t, so residual < t. J3 items have w <= t.
            # Actually J1 residual = C - w_J1 < t. So only items with weight <= residual fit.
            # The standard L2 bound just uses aggregate sums.
            pass
            
            # Simpler approach: just use aggregate
            # Actually let me just recompute properly
            break
        
        # Let me implement a simpler but correct L2
        sorted_w = sorted(weights, reverse=True)
        
        for t in range(1, max(2, C // 2 + 1)):
            n1 = 0
            sum_n1 = 0
            n2 = 0 
            sum_n2 = 0
            sum_n3 = 0
            
            half_c = C / 2.0
            threshold_high = C - t
            
            for w in sorted_w:
                if w > threshold_high:
                    n1 += 1
                    sum_n1 += w
                elif w > half_c:
                    n2 += 1
                    sum_n2 += w
                elif w > 0 and w <= t:
                    sum_n3 += w
                # items with t < w <= C/2 are "middle" items not counted in simple L2
            
            # Residual capacity in n1 bins that J3 items can use
            residual_n1 = n1 * C - sum_n1  # each J1 bin has capacity C, used sum_n1 total
            # But each J1 item is in its own bin, residual per bin = C - w_i < t
            # So J3 items (w <= t) could potentially fill this
            
            # Residual in n2 bins
            residual_n2 = n2 * C - sum_n2
            
            # J3 items that can't fit in residuals need new bins
            leftover = sum_n3 - residual_n1 - residual_n2
            extra = max(0, math.ceil(leftover / C)) if leftover > 0 else 0
            
            lb = n1 + n2 + extra
            if lb > best_lb:
                best_lb = lb
        
        return best_lb
    
    lower_bound = compute_L2()
    
    # --- Phase 1.1: Initial Solutions ---
    indices_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    def first_fit_decreasing():
        bin_items = []
        bin_weights_list = []
        for idx in indices_sorted:
            w = weights[idx]
            placed = False
            for b in range(len(bin_items)):
                if bin_weights_list[b] + w <= C:
                    bin_items[b].append(idx)
                    bin_weights_list[b] += w
                    placed = True
                    break
            if not placed:
                bin_items.append([idx])
                bin_weights_list.append(w)
        return bin_items, bin_weights_list
    
    def best_fit_decreasing():
        bin_items = []
        bin_weights_list = []
        for idx in indices_sorted:
            w = weights[idx]
            best_bin = -1
            best_remaining = C + 1
            for b in range(len(bin_items)):
                remaining = C - bin_weights_list[b]
                if remaining >= w and remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b
            if best_bin >= 0:
                bin_items[best_bin].append(idx)
                bin_weights_list[best_bin] += w
            else:
                bin_items.append([idx])
                bin_weights_list.append(w)
        return bin_items, bin_weights_list
    
    def worst_fit_decreasing():
        bin_items = []
        bin_weights_list = []
        for idx in indices_sorted:
            w = weights[idx]
            best_bin = -1
            best_remaining = -1
            for b in range(len(bin_items)):
                remaining = C - bin_weights_list[b]
                if remaining >= w and remaining > best_remaining:
                    best_remaining = remaining
                    best_bin = b
            if best_bin >= 0:
                bin_items[best_bin].append(idx)
                bin_weights_list[best_bin] += w
            else:
                bin_items.append([idx])
                bin_weights_list.append(w)
        return bin_items, bin_weights_list
    
    ffd_items, ffd_weights = first_fit_decreasing()
    bfd_items, bfd_weights = best_fit_decreasing()
    wfd_items, wfd_weights = worst_fit_decreasing()
    
    candidates = [(ffd_items, ffd_weights), (bfd_items, bfd_weights), (wfd_items, wfd_weights)]
    best_init = min(candidates, key=lambda x: len(x[0]))
    
    current_bins = best_init[0]
    current_weights = best_init[1]
    
    num_bins = len(current_bins)
    
    # Save best feasible solution
    best_num_bins = num_bins
    best_packing = [list(b) for b in current_bins]
    best_bin_weights = list(current_weights)
    
    if best_num_bins <= lower_bound:
        return {"packing": best_packing, "bin_weights": best_bin_weights}
    
    # --- Phase 2: Penalty-Based Tabu Search ---
    # We'll try to reduce bins one at a time
    # For each target: eliminate a bin, redistribute (allowing overfill),
    # then tabu search to minimize total overflow
    
    def elapsed():
        return time.time() - start_time
    
    target_bins = best_num_bins - 1
    
    while target_bins >= lower_bound and elapsed() < time_limit * 0.95:
        # Try eliminating different bins
        # Build current solution state from best known
        
        # Try up to 5 different bins to eliminate
        tried_eliminate = set()
        success = False
        
        for attempt in range(5):
            if elapsed() >= time_limit * 0.95:
                break
            
            # Rebuild solution from best known
            K = best_num_bins  # current number of bins
            if target_bins >= K:
                break
            
            # item_to_bin, bin_weight, bins_items
            item_to_bin = [0] * n
            bins_items = [[] for _ in range(K)]
            bin_weight = [0] * K
            
            for b in range(K):
                for idx in best_packing[b]:
                    item_to_bin[idx] = b
                    bins_items[b].append(idx)
                    bin_weight[b] += weights[idx]
            
            # Choose which bin to eliminate: lightest first
            bin_order = sorted(range(K), key=lambda b: bin_weight[b])
            
            elim_bin = -1
            for b in bin_order:
                if b not in tried_eliminate:
                    elim_bin = b
                    break
            
            if elim_bin == -1:
                # Try random
                elim_bin = random.choice(bin_order[:max(1, K // 3)])
            
            tried_eliminate.add(elim_bin)
            
            # Redistribute items from elim_bin to other bins (allow overfill)
            items_to_redistribute = list(bins_items[elim_bin])
            items_to_redistribute.sort(key=lambda i: weights[i], reverse=True)
            
            # Remove elim_bin by swapping with last bin
            # Easier: just create new arrays with K-1 bins
            new_K = K - 1
            new_bins_items = []
            new_bin_weight = []
            old_to_new = {}
            idx_counter = 0
            for b in range(K):
                if b == elim_bin:
                    continue
                old_to_new[b] = idx_counter
                new_bins_items.append(list(bins_items[b]))
                new_bin_weight.append(bin_weight[b])
                idx_counter += 1
            
            new_item_to_bin = [0] * n
            for b in range(new_K):
                for idx in new_bins_items[b]:
                    new_item_to_bin[idx] = b
            
            # Place redistributed items using best-fit (into least overfilled bin)
            for idx in items_to_redistribute:
                w = weights[idx]
                # Best fit: find bin with least remaining capacity that still fits, or least overflow increase
                best_b = -1
                best_score = float('inf')
                for b in range(new_K):
                    remaining = C - new_bin_weight[b]
                    # If remaining >= w, no overflow increase - pick tightest fit
                    if remaining >= w:
                        score = remaining - w  # lower = tighter = better
                        if score < best_score:
                            best_score = score
                            best_b = b
                
                if best_b == -1:
                    # All bins would overflow; pick the one with most remaining capacity
                    best_b = 0
                    best_remaining = C - new_bin_weight[0]
                    for b in range(1, new_K):
                        remaining = C - new_bin_weight[b]
                        if remaining > best_remaining:
                            best_remaining = remaining
                            best_b = b
                
                new_bins_items[best_b].append(idx)
                new_bin_weight[best_b] += w
                new_item_to_bin[idx] = best_b
            
            # Compute overflow
            overflow = [max(0, new_bin_weight[b] - C) for b in range(new_K)]
            total_overflow = sum(overflow)
            
            if total_overflow == 0:
                # Already feasible!
                best_num_bins = new_K
                best_packing = [list(b) for b in new_bins_items]
                best_bin_weights = list(new_bin_weight)
                success = True
                break
            
            # --- Tabu search to minimize total_overflow ---
            tabu_dict = {}  # (item, bin) -> expiry iteration
            tabu_tenure = max(7, n // 10)
            best_overflow = total_overflow
            best_state_bins = [list(b) for b in new_bins_items]
            best_state_weights = list(new_bin_weight)
            
            iteration = 0
            no_improve_count = 0
            max_no_improve = max(500, n * 2)
            
            # For large instances, limit neighborhood sampling
            large_instance = n > 500
            
            # Overflow tracking with sets for efficiency
            overfilled = set(b for b in range(new_K) if overflow[b] > 0)
            
            # Convert bins_items to sets for O(1) operations
            bins_sets = [set(b) for b in new_bins_items]
            
            while total_overflow > 0 and elapsed() < time_limit * 0.95:
                iteration += 1
                
                if no_improve_count > max_no_improve:
                    break
                
                # Find best move
                best_move = None
                best_delta = float('inf')  # we want to minimize overflow, so negative delta is good
                best_is_tabu = False
                
                # Also track best tabu move for aspiration
                best_tabu_move = None
                best_tabu_delta = float('inf')
                best_tabu_resulting_overflow = float('inf')
                
                # Move neighborhood: move items FROM overfilled bins
                overfilled_list = list(overfilled)
                
                for src_bin in overfilled_list:
                    items_in_src = list(bins_sets[src_bin])
                    if large_instance and len(items_in_src) > 200:
                        items_in_src = random.sample(items_in_src, 200)
                    
                    for item in items_in_src:
                        w = weights[item]
                        # Overflow reduction at source
                        old_src_overflow = overflow[src_bin]
                        new_src_weight = new_bin_weight[src_bin] - w
                        new_src_overflow = max(0, new_src_weight - C)
                        delta_src = new_src_overflow - old_src_overflow
                        
                        for dst_bin in range(new_K):
                            if dst_bin == src_bin:
                                continue
                            
                            old_dst_overflow = overflow[dst_bin]
                            new_dst_weight = new_bin_weight[dst_bin] + w
                            new_dst_overflow = max(0, new_dst_weight - C)
                            delta_dst = new_dst_overflow - old_dst_overflow
                            
                            delta = delta_src + delta_dst
                            
                            tabu_key = (item, dst_bin)
                            is_tabu = tabu_key in tabu_dict and tabu_dict[tabu_key] > iteration
                            
                            if is_tabu:
                                # Aspiration: accept if results in best overflow ever
                                resulting = total_overflow + delta
                                if resulting < best_overflow:
                                    if delta < best_delta:
                                        best_delta = delta
                                        best_move = ('move', item, src_bin, dst_bin)
                                        best_is_tabu = False  # aspiration overrides
                                # Track best tabu move
                                if delta < best_tabu_delta:
                                    best_tabu_delta = delta
                                    best_tabu_move = ('move', item, src_bin, dst_bin)
                                    best_tabu_resulting_overflow = total_overflow + delta
                            else:
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = ('move', item, src_bin, dst_bin)
                
                # Swap neighborhood: swap item from overfilled bin with item from another bin
                for src_bin in overfilled_list:
                    items_in_src = list(bins_sets[src_bin])
                    if large_instance and len(items_in_src) > 50:
                        # Pick heaviest items from overfilled bin
                        items_in_src.sort(key=lambda i: weights[i], reverse=True)
                        items_in_src = items_in_src[:50]
                    
                    for item_a in items_in_src:
                        w_a = weights[item_a]
                        
                        for dst_bin in range(new_K):
                            if dst_bin == src_bin:
                                continue
                            
                            items_in_dst = list(bins_sets[dst_bin])
                            if large_instance and len(items_in_dst) > 50:
                                items_in_dst = random.sample(items_in_dst, 50)
                            
                            for item_b in items_in_dst:
                                w_b = weights[item_b]
                                if w_b >= w_a:
                                    continue  # Only swap if w_a > w_b (net reduction in src)
                                
                                # Compute delta
                                new_src_w = new_bin_weight[src_bin] - w_a + w_b
                                new_dst_w = new_bin_weight[dst_bin] - w_b + w_a
                                
                                new_src_of = max(0, new_src_w - C)
                                new_dst_of = max(0, new_dst_w - C)
                                
                                delta = (new_src_of - overflow[src_bin]) + (new_dst_of - overflow[dst_bin])
                                
                                tabu_a = (item_a, dst_bin) in tabu_dict and tabu_dict[(item_a, dst_bin)] > iteration
                                tabu_b = (item_b, src_bin) in tabu_dict and tabu_dict[(item_b, src_bin)] > iteration
                                is_tabu = tabu_a or tabu_b
                                
                                if is_tabu:
                                    resulting = total_overflow + delta
                                    if resulting < best_overflow:
                                        if delta < best_delta:
                                            best_delta = delta
                                            best_move = ('swap', item_a, src_bin, item_b, dst_bin)
                                    if delta < best_tabu_delta:
                                        best_tabu_delta = delta
                                        best_tabu_move = ('swap', item_a, src_bin, item_b, dst_bin)
                                        best_tabu_resulting_overflow = total_overflow + delta
                                else:
                                    if delta < best_delta:
                                        best_delta = delta
                                        best_move = ('swap', item_a, src_bin, item_b, dst_bin)
                
                # If no non-tabu move found, use best tabu move
                if best_move is None:
                    if best_tabu_move is not None:
                        best_move = best_tabu_move
                        best_delta = best_tabu_delta
                    else:
                        break  # No moves at all
                
                # Execute the move
                if best_move[0] == 'move':
                    _, item, src, dst = best_move
                    w = weights[item]
                    
                    bins_sets[src].discard(item)
                    bins_sets[dst].add(item)
                    new_bin_weight[src] -= w
                    new_bin_weight[dst] += w
                    new_item_to_bin[item] = dst
                    
                    overflow[src] = max(0, new_bin_weight[src] - C)
                    overflow[dst] = max(0, new_bin_weight[dst] - C)
                    
                    total_overflow += best_delta
                    
                    # Update overfilled set
                    if overflow[src] > 0:
                        overfilled.add(src)
                    else:
                        overfilled.discard(src)
                    if overflow[dst] > 0:
                        overfilled.add(dst)
                    else:
                        overfilled.discard(dst)
                    
                    # Tabu: item cannot go back to src
                    tabu_dict[(item, src)] = iteration + tabu_tenure
                    
                else:  # swap
                    _, item_a, src, item_b, dst = best_move
                    w_a = weights[item_a]
                    w_b = weights[item_b]
                    
                    bins_sets[src].discard(item_a)
                    bins_sets[src].add(item_b)
                    bins_sets[dst].discard(item_b)
                    bins_sets[dst].add(item_a)
                    
                    new_bin_weight[src] = new_bin_weight[src] - w_a + w_b
                    new_bin_weight[dst] = new_bin_weight[dst] - w_b + w_a
                    new_item_to_bin[item_a] = dst
                    new_item_to_bin[item_b] = src
                    
                    overflow[src] = max(0, new_bin_weight[src] - C)
                    overflow[dst] = max(0, new_bin_weight[dst] - C)
                    
                    total_overflow += best_delta
                    
                    if overflow[src] > 0:
                        overfilled.add(src)
                    else:
                        overfilled.discard(src)
                    if overflow[dst] > 0:
                        overfilled.add(dst)
                    else:
                        overfilled.discard(dst)
                    
                    tabu_dict[(item_a, src)] = iteration + tabu_tenure
                    tabu_dict[(item_b, dst)] = iteration + tabu_tenure
                
                # Check improvement
                if total_overflow < best_overflow:
                    best_overflow = total_overflow
                    best_state_bins = [list(b) for b in bins_sets]
                    best_state_weights = list(new_bin_weight)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if total_overflow == 0:
                    break
                
                # Perturbation if stuck
                if no_improve_count == 200:
                    # Small perturbation
                    ovf_list = list(overfilled)
                    if ovf_list:
                        for _ in range(min(5, len(ovf_list) * 2)):
                            if not overfilled:
                                break
                            sb = random.choice(list(overfilled))
                            if not bins_sets[sb]:
                                continue
                            item = random.choice(list(bins_sets[sb]))
                            w = weights[item]
                            db = random.randint(0, new_K - 1)
                            if db == sb:
                                continue
                            bins_sets[sb].discard(item)
                            bins_sets[db].add(item)
                            new_bin_weight[sb] -= w
                            new_bin_weight[db] += w
                            new_item_to_bin[item] = db
                            overflow[sb] = max(0, new_bin_weight[sb] - C)
                            overflow[db] = max(0, new_bin_weight[db] - C)
                            if overflow[sb] > 0:
                                overfilled.add(sb)
                            else:
                                overfilled.discard(sb)
                            if overflow[db] > 0:
                                overfilled.add(db)
                            else:
                                overfilled.discard(db)
                            tabu_dict[(item, sb)] = iteration + tabu_tenure
                        total_overflow = sum(overflow)
                    tabu_tenure = random.randint(max(5, n // 20), max(10, n // 5))
                
                if no_improve_count == 500:
                    # Reset tabu
                    tabu_dict.clear()
                    tabu_tenure = max(7, n // 10)
            
            if total_overflow == 0:
                # Feasible solution found!
                best_num_bins = new_K
                best_packing = [list(bins_sets[b]) for b in range(new_K) if bins_sets[b]]
                best_bin_weights = [new_bin_weight[b] for b in range(new_K) if bins_sets[b]]
                # Sanity: all bins should be non-empty if overflow is 0 and we placed all items
                success = True
                break
        
        if success:
            if best_num_bins <= lower_bound:
                break
            target_bins = best_num_bins - 1
        else:
            break
    
    # --- Phase 4: Output ---
    # Filter out any empty bins (shouldn't happen but just in case)
    final_packing = []
    final_weights = []
    for i, b in enumerate(best_packing):
        if b:
            final_packing.append(b)
            final_weights.append(best_bin_weights[i])
    
    return {
        "packing": final_packing,
        "bin_weights": final_weights
    }