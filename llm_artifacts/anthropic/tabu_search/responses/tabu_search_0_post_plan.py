import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    w_tuple = tuple(weights)
    _max = max
    _time = time.time
    
    # First Fit Decreasing for initial solution
    def ffd(cap, wts, indices):
        sorted_idx = sorted(indices, key=lambda i: wts[i], reverse=True)
        bins = []  # list of (list of indices, total weight)
        for idx in sorted_idx:
            w = wts[idx]
            placed = False
            for b in range(len(bins)):
                if bins[b][1] + w <= cap:
                    bins[b][0].append(idx)
                    bins[b] = (bins[b][0], bins[b][1] + w)
                    placed = True
                    break
            if not placed:
                bins.append(([idx], w))
        packing = [b[0] for b in bins]
        bin_weights = [b[1] for b in bins]
        return packing, bin_weights
    
    all_indices = list(range(n))
    best_packing, best_bin_weights = ffd(bin_capacity, w_tuple, all_indices)
    best_num_bins = len(best_packing)
    
    if best_num_bins <= 1:
        return {"packing": best_packing, "bin_weights": best_bin_weights}
    
    # L1 lower bound
    total_weight = sum(weights)
    lb1 = (total_weight + bin_capacity - 1) // bin_capacity
    
    # L2 lower bound (Martello-Toth)
    C = bin_capacity
    half_C = C / 2.0
    
    # Sort weights for efficient computation
    sorted_weights = sorted(weights, reverse=True)
    
    def compute_l2():
        best_l2 = lb1
        # Collect unique alpha values worth trying
        # alpha ranges over possible values; we try alpha = C - w for items with w > C/2
        # and also some discrete values
        alphas = set()
        for w in sorted_weights:
            if w > half_C:
                a = C - w
                if a >= 0:
                    alphas.add(a)
            else:
                break
        alphas.add(0)
        # Also try a few more values
        for w in sorted_weights:
            if w <= half_C:
                a = C - w
                if a >= 0:
                    alphas.add(a)
        
        for alpha in alphas:
            # J1: items with weight > C - alpha (large items, each needs own bin essentially)
            # J2: items with weight in (C/2, C - alpha]
            # J3: items with weight in [alpha, C/2]
            # Wait, standard L2:
            # For parameter alpha in [0, C/2]:
            # J1 = {j : w_j > C - alpha}  (very large)
            # J2 = {j : C - alpha >= w_j > C/2}  (large)
            # J3 = {j : C/2 >= w_j >= alpha}  (medium)
            
            if alpha < 0 or alpha > half_C:
                continue
            
            n_j1 = 0
            n_j2 = 0
            sum_j3 = 0
            n_j3 = 0
            
            threshold1 = C - alpha  # w > C - alpha for J1
            # threshold2: w > C/2 for J2, and w <= C - alpha
            
            for w in sorted_weights:
                if w > threshold1:
                    n_j1 += 1
                elif w > half_C:
                    n_j2 += 1
                elif w >= alpha:
                    sum_j3 += w
                    n_j3 += 1
                else:
                    break  # sorted descending, rest are smaller
            
            # Each J1 item needs its own bin. Each J2 item needs its own bin (can't pair two J2 items).
            # Remaining capacity from J2 bins: each J2 bin has capacity C - w_j2, 
            # but since w_j2 > C/2, remaining < C/2, so J3 items (which are <= C/2) could fit.
            # But J2 items have weight <= C - alpha, so remaining >= alpha.
            # J3 items have weight >= alpha.
            # Actually the formula is:
            # L2(alpha) = n_j1 + n_j2 + max(0, ceil((sum_j3 - (n_j2 * C - sum_j2) ... )))
            # Simpler standard formula:
            # Free space in J2 bins = n_j2 * C - sum(J2 weights)
            # But we don't track sum_j2 easily. Let me recompute.
            
            # Actually, simpler approach:
            # L2 = n_j1 + n_j2 + max(0, ceil((n_j3 - n_j2_free_slots) / 1))
            # No, the standard is:
            # L2(alpha) = n_j1 + n_j2 + max(0, ceil((sum_j3 - (n_j2*C - sum_j2)) / C))
            # But we need sum_j2
            pass
        
        # Let me redo this more carefully
        # Actually let me use a simpler but correct formulation
        best_l2 = lb1
        
        for alpha in alphas:
            if alpha < 0 or alpha > half_C:
                continue
            
            n_j1 = 0
            n_j2 = 0
            sum_j2 = 0
            sum_j3 = 0
            n_j3 = 0
            threshold1 = C - alpha
            
            for w in sorted_weights:
                if w > threshold1:
                    n_j1 += 1
                elif w > half_C:
                    n_j2 += 1
                    sum_j2 += w
                elif w >= alpha:
                    sum_j3 += w
                    n_j3 += 1
            
            # Free space in J2 bins (each J2 item is in its own bin)
            free_in_j2 = n_j2 * C - sum_j2
            # J3 items that don't fit in J2 free space
            remaining_j3 = _max(0, sum_j3 - free_in_j2)
            extra_bins = (remaining_j3 + C - 1) // C if remaining_j3 > 0 else 0
            
            l2_val = n_j1 + n_j2 + extra_bins
            if l2_val > best_l2:
                best_l2 = l2_val
        
        return best_l2
    
    lb = compute_l2()
    
    if best_num_bins <= lb:
        return {"packing": best_packing, "bin_weights": best_bin_weights}
    
    # Tabu search to try packing into fewer bins
    def tabu_search_pack(target_bins, deadline, init_variant=0):
        """Try to pack all items into target_bins bins using tabu search."""
        
        # Initialize bins
        bin_items = [[] for _ in range(target_bins)]  # list of items per bin
        bin_wt = [0] * target_bins
        item_bin = [0] * n
        # Position tracking for O(1) removal
        item_pos = [0] * n  # position of item in its bin's list
        
        if init_variant == 0:
            # FFD best-fit into target bins
            sorted_idx = sorted(range(n), key=lambda i: w_tuple[i], reverse=True)
            for idx in sorted_idx:
                w = w_tuple[idx]
                best_b = -1
                best_remain = -1
                for b in range(target_bins):
                    remain = C - bin_wt[b]
                    if remain >= w and (best_b == -1 or remain < best_remain):
                        best_b = b
                        best_remain = remain
                if best_b == -1:
                    # Place in bin with most remaining capacity
                    best_b = 0
                    best_remain = C - bin_wt[0]
                    for b in range(1, target_bins):
                        remain = C - bin_wt[b]
                        if remain > best_remain:
                            best_b = b
                            best_remain = remain
                item_pos[idx] = len(bin_items[best_b])
                bin_items[best_b].append(idx)
                bin_wt[best_b] += w
                item_bin[idx] = best_b
        elif init_variant == 1:
            # Random order, first-fit
            order = list(range(n))
            random.shuffle(order)
            for idx in order:
                w = w_tuple[idx]
                placed = False
                for b in range(target_bins):
                    if bin_wt[b] + w <= C:
                        item_pos[idx] = len(bin_items[b])
                        bin_items[b].append(idx)
                        bin_wt[b] += w
                        item_bin[idx] = b
                        placed = True
                        break
                if not placed:
                    # Place in bin with most remaining capacity
                    best_b = 0
                    best_remain = C - bin_wt[0]
                    for b in range(1, target_bins):
                        remain = C - bin_wt[b]
                        if remain > best_remain:
                            best_b = b
                            best_remain = remain
                    item_pos[idx] = len(bin_items[best_b])
                    bin_items[best_b].append(idx)
                    bin_wt[best_b] += w
                    item_bin[idx] = best_b
        elif init_variant == 2:
            # Ascending order, best-fit
            sorted_idx = sorted(range(n), key=lambda i: w_tuple[i])
            for idx in sorted_idx:
                w = w_tuple[idx]
                best_b = -1
                best_remain = C + 1
                for b in range(target_bins):
                    remain = C - bin_wt[b]
                    if remain >= w and remain < best_remain:
                        best_b = b
                        best_remain = remain
                if best_b == -1:
                    best_b = 0
                    best_remain = C - bin_wt[0]
                    for b in range(1, target_bins):
                        remain = C - bin_wt[b]
                        if remain > best_remain:
                            best_b = b
                            best_remain = remain
                item_pos[idx] = len(bin_items[best_b])
                bin_items[best_b].append(idx)
                bin_wt[best_b] += w
                item_bin[idx] = best_b
        elif init_variant == 3:
            # Load-balancing: distribute items to equalize bin weights
            sorted_idx = sorted(range(n), key=lambda i: w_tuple[i], reverse=True)
            for idx in sorted_idx:
                w = w_tuple[idx]
                # Find bin with minimum weight
                best_b = 0
                best_w = bin_wt[0]
                for b in range(1, target_bins):
                    if bin_wt[b] < best_w:
                        best_b = b
                        best_w = bin_wt[b]
                item_pos[idx] = len(bin_items[best_b])
                bin_items[best_b].append(idx)
                bin_wt[best_b] += w
                item_bin[idx] = best_b
        else:
            # Variant 4: FFD with random perturbation
            sorted_idx = sorted(range(n), key=lambda i: w_tuple[i], reverse=True)
            for idx in sorted_idx:
                w = w_tuple[idx]
                best_b = -1
                best_remain = -1
                for b in range(target_bins):
                    remain = C - bin_wt[b]
                    if remain >= w and (best_b == -1 or remain < best_remain):
                        best_b = b
                        best_remain = remain
                if best_b == -1:
                    best_b = 0
                    best_remain = C - bin_wt[0]
                    for b in range(1, target_bins):
                        remain = C - bin_wt[b]
                        if remain > best_remain:
                            best_b = b
                            best_remain = remain
                item_pos[idx] = len(bin_items[best_b])
                bin_items[best_b].append(idx)
                bin_wt[best_b] += w
                item_bin[idx] = best_b
            # Now randomly move 20% of items
            items_to_move = random.sample(range(n), _max(1, n // 5))
            for idx in items_to_move:
                old_b = item_bin[idx]
                new_b = random.randint(0, target_bins - 2)
                if new_b >= old_b:
                    new_b += 1
                # Remove from old bin (swap and pop)
                pos = item_pos[idx]
                last_idx = bin_items[old_b][-1]
                bin_items[old_b][pos] = last_idx
                item_pos[last_idx] = pos
                bin_items[old_b].pop()
                bin_wt[old_b] -= w_tuple[idx]
                # Add to new bin
                item_pos[idx] = len(bin_items[new_b])
                bin_items[new_b].append(idx)
                bin_wt[new_b] += w_tuple[idx]
                item_bin[idx] = new_b
        
        # Compute squared overflow
        def compute_overflow_sq():
            total = 0
            for b in range(target_bins):
                ov = bin_wt[b] - C
                if ov > 0:
                    total += ov * ov
            return total
        
        def is_feasible():
            for b in range(target_bins):
                if bin_wt[b] > C:
                    return False
            return True
        
        if is_feasible():
            return bin_items, bin_wt
        
        overflow_sq = compute_overflow_sq()
        
        tabu = {}
        base_tenure = _max(7, int(n ** 0.5))
        if base_tenure > n:
            base_tenure = n
        tenure = base_tenure
        max_tenure = base_tenure * 2
        
        best_overflow_sq = overflow_sq
        
        # Move frequency tracking
        move_freq = [0] * n
        
        iteration = 0
        max_iterations = 1000000
        no_improve_limit = _max(3000, n * 30)
        no_improve_count = 0
        diversify_count = 0
        
        check_freq = 200
        
        while iteration < max_iterations:
            if iteration % check_freq == 0:
                if _time() >= deadline:
                    break
            
            iteration += 1
            no_improve_count += 1
            
            # Adaptive tenure
            if no_improve_count > 500 and no_improve_count % 500 == 0:
                tenure = min(max_tenure, int(tenure * 1.2))
            
            # Diversification
            if no_improve_count > no_improve_limit:
                diversify_count += 1
                perturb_frac = min(0.5, 0.2 + 0.05 * diversify_count)
                
                for b in range(target_bins):
                    if bin_wt[b] > C:
                        items_in_b = list(bin_items[b])
                        random.shuffle(items_in_b)
                        num_to_move = _max(1, int(len(items_in_b) * perturb_frac))
                        # Prefer items moved least frequently
                        items_in_b.sort(key=lambda i: move_freq[i])
                        for idx in items_in_b[:num_to_move]:
                            new_b = random.randint(0, target_bins - 2)
                            if new_b >= b:
                                new_b += 1
                            # Remove idx from b using swap-and-pop
                            pos = item_pos[idx]
                            last = bin_items[b][-1]
                            bin_items[b][pos] = last
                            item_pos[last] = pos
                            bin_items[b].pop()
                            bin_wt[b] -= w_tuple[idx]
                            # Add to new_b
                            item_pos[idx] = len(bin_items[new_b])
                            bin_items[new_b].append(idx)
                            bin_wt[new_b] += w_tuple[idx]
                            item_bin[idx] = new_b
                            move_freq[idx] += 1
                            tabu[(idx, b)] = iteration + tenure
                
                overflow_sq = compute_overflow_sq()
                no_improve_count = 0
                tenure = base_tenure  # Reset tenure after diversification
                
                if is_feasible():
                    return bin_items, bin_wt
                continue
            
            # Find overflowing bins
            overflow_bins = []
            max_ov = 0
            max_ov_bin = -1
            for b in range(target_bins):
                ov = bin_wt[b] - C
                if ov > 0:
                    overflow_bins.append(b)
                    if ov > max_ov:
                        max_ov = ov
                        max_ov_bin = b
            
            if not overflow_bins:
                return bin_items, bin_wt
            
            # Source selection: 80% highest overflow, 20% random
            if random.random() < 0.8 and max_ov_bin >= 0:
                source_bin = max_ov_bin
            else:
                source_bin = random.choice(overflow_bins)
            
            best_move = None
            best_move_delta = 0
            best_move_is_tabu = False
            
            items_in_source = bin_items[source_bin]
            src_len = len(items_in_source)
            
            cand_limit = min(50, src_len)
            if src_len > cand_limit:
                candidates = random.sample(items_in_source, cand_limit)
            else:
                candidates = list(items_in_source)
            
            src_wt = bin_wt[source_bin]
            old_src_ov = src_wt - C
            old_src_sq = old_src_ov * old_src_ov if old_src_ov > 0 else 0
            
            # Evaluate moves
            for idx in candidates:
                w = w_tuple[idx]
                new_src_wt = src_wt - w
                new_src_ov = new_src_wt - C
                new_src_sq = new_src_ov * new_src_ov if new_src_ov > 0 else 0
                src_delta = new_src_sq - old_src_sq
                
                for dest_bin in range(target_bins):
                    if dest_bin == source_bin:
                        continue
                    
                    dst_wt = bin_wt[dest_bin]
                    old_dst_ov = dst_wt - C
                    old_dst_sq = old_dst_ov * old_dst_ov if old_dst_ov > 0 else 0
                    new_dst_wt = dst_wt + w
                    new_dst_ov = new_dst_wt - C
                    new_dst_sq = new_dst_ov * new_dst_ov if new_dst_ov > 0 else 0
                    
                    delta = src_delta + (new_dst_sq - old_dst_sq)
                    
                    is_tabu = tabu.get((idx, dest_bin), 0) > iteration
                    new_ov_sq = overflow_sq + delta
                    
                    # Aspiration: accept if better than best known
                    if new_ov_sq < best_overflow_sq:
                        if best_move is None or delta < best_move_delta:
                            best_move = ('m', idx, source_bin, dest_bin)
                            best_move_delta = delta
                            best_move_is_tabu = False
                    elif not is_tabu:
                        if best_move is None or delta < best_move_delta or (delta == best_move_delta and best_move_is_tabu):
                            best_move = ('m', idx, source_bin, dest_bin)
                            best_move_delta = delta
                            best_move_is_tabu = False
                    elif best_move is None:
                        best_move = ('m', idx, source_bin, dest_bin)
                        best_move_delta = delta
                        best_move_is_tabu = True
            
            # Try swaps - focus on non-overflowing destination bins
            non_overflow_bins = [b for b in range(target_bins) if b != source_bin and bin_wt[b] <= C]
            if not non_overflow_bins:
                non_overflow_bins = [b for b in range(target_bins) if b != source_bin]
            
            swap_limit = min(8, len(non_overflow_bins))
            if len(non_overflow_bins) > swap_limit:
                swap_target_bins = random.sample(non_overflow_bins, swap_limit)
            else:
                swap_target_bins = non_overflow_bins
            
            for idx1 in candidates:
                w1 = w_tuple[idx1]
                new_src_wt_base = src_wt - w1
                
                for dest_bin in swap_target_bins:
                    dst_wt = bin_wt[dest_bin]
                    old_dst_ov = dst_wt - C
                    old_dst_sq = old_dst_ov * old_dst_ov if old_dst_ov > 0 else 0
                    
                    items_in_dest = bin_items[dest_bin]
                    dest_len = len(items_in_dest)
                    swap_cand_limit = min(20, dest_len)
                    if dest_len > swap_cand_limit:
                        swap_cands = random.sample(items_in_dest, swap_cand_limit)
                    else:
                        swap_cands = items_in_dest  # iterate directly, no copy needed for read
                    
                    for idx2 in swap_cands:
                        w2 = w_tuple[idx2]
                        if w1 == w2:
                            continue
                        
                        new_src_wt2 = new_src_wt_base + w2
                        new_src_ov2 = new_src_wt2 - C
                        new_src_sq2 = new_src_ov2 * new_src_ov2 if new_src_ov2 > 0 else 0
                        
                        new_dst_wt2 = dst_wt - w2 + w1
                        new_dst_ov2 = new_dst_wt2 - C
                        new_dst_sq2 = new_dst_ov2 * new_dst_ov2 if new_dst_ov2 > 0 else 0
                        
                        delta = (new_src_sq2 - old_src_sq) + (new_dst_sq2 - old_dst_sq)
                        new_ov_sq = overflow_sq + delta
                        
                        is_tabu = (tabu.get((idx1, dest_bin), 0) > iteration or
                                   tabu.get((idx2, source_bin), 0) > iteration)
                        
                        if new_ov_sq < best_overflow_sq:
                            if best_move is None or delta < best_move_delta:
                                best_move = ('s', idx1, source_bin, idx2, dest_bin)
                                best_move_delta = delta
                                best_move_is_tabu = False
                        elif not is_tabu:
                            if best_move is None or delta < best_move_delta or (delta == best_move_delta and best_move_is_tabu):
                                best_move = ('s', idx1, source_bin, idx2, dest_bin)
                                best_move_delta = delta
                                best_move_is_tabu = False
                        elif best_move is None:
                            best_move = ('s', idx1, source_bin, idx2, dest_bin)
                            best_move_delta = delta
                            best_move_is_tabu = True
            
            if best_move is None:
                continue
            
            # Apply move
            if best_move[0] == 'm':
                _, idx, src, dst = best_move
                w = w_tuple[idx]
                # Remove from src using swap-and-pop
                pos = item_pos[idx]
                last = bin_items[src][-1]
                bin_items[src][pos] = last
                item_pos[last] = pos
                bin_items[src].pop()
                bin_wt[src] -= w
                # Add to dst
                item_pos[idx] = len(bin_items[dst])
                bin_items[dst].append(idx)
                bin_wt[dst] += w
                item_bin[idx] = dst
                move_freq[idx] += 1
                rand_extra = random.randint(0, tenure // 3)
                tabu[(idx, src)] = iteration + tenure + rand_extra
            else:  # swap
                _, idx1, src, idx2, dst = best_move
                w1 = w_tuple[idx1]
                w2 = w_tuple[idx2]
                # Remove idx1 from src
                pos1 = item_pos[idx1]
                last1 = bin_items[src][-1]
                bin_items[src][pos1] = last1
                item_pos[last1] = pos1
                bin_items[src].pop()
                bin_wt[src] -= w1
                # Remove idx2 from dst
                pos2 = item_pos[idx2]
                last2 = bin_items[dst][-1]
                bin_items[dst][pos2] = last2
                item_pos[last2] = pos2
                bin_items[dst].pop()
                bin_wt[dst] -= w2
                # Add idx2 to src
                item_pos[idx2] = len(bin_items[src])
                bin_items[src].append(idx2)
                bin_wt[src] += w2
                # Add idx1 to dst
                item_pos[idx1] = len(bin_items[dst])
                bin_items[dst].append(idx1)
                bin_wt[dst] += w1
                item_bin[idx1] = dst
                item_bin[idx2] = src
                move_freq[idx1] += 1
                move_freq[idx2] += 1
                rand_extra1 = random.randint(0, tenure // 3)
                rand_extra2 = random.randint(0, tenure // 3)
                tabu[(idx1, src)] = iteration + tenure + rand_extra1
                tabu[(idx2, dst)] = iteration + tenure + rand_extra2
            
            overflow_sq = overflow_sq + best_move_delta
            
            if overflow_sq < best_overflow_sq:
                best_overflow_sq = overflow_sq
                no_improve_count = 0
                tenure = base_tenure  # Reset tenure on improvement
                if overflow_sq == 0:
                    return bin_items, bin_wt
                # Check actual feasibility (in case of rounding)
                if is_feasible():
                    return bin_items, bin_wt
            
            # Periodic cleanup
            if iteration % 5000 == 0:
                tabu = {k: v for k, v in tabu.items() if v > iteration}
        
        return None
    
    # Main loop: try to reduce number of bins
    current_num_bins = best_num_bins
    hard_deadline = start_time + time_limit * 0.98
    
    init_variants = [0, 1, 2, 3, 4]
    
    while current_num_bins > lb:
        if _time() >= hard_deadline:
            break
        
        target = current_num_bins - 1
        found = False
        restart_idx = 0
        
        while True:
            now = _time()
            remaining = hard_deadline - now
            if remaining <= 0.01:
                break
            
            # First attempt gets 50% of remaining, subsequent get 50% of what's left
            phase_deadline = now + remaining * 0.5
            
            variant = init_variants[restart_idx % len(init_variants)]
            restart_idx += 1
            
            result = tabu_search_pack(target, phase_deadline, variant)
            
            if result is not None:
                bin_items, bin_wt = result
                new_packing = []
                new_weights = []
                for b in range(target):
                    if bin_items[b]:
                        new_packing.append(list(bin_items[b]))
                        new_weights.append(bin_wt[b])
                best_packing = new_packing
                best_bin_weights = new_weights
                current_num_bins = len(new_packing)
                found = True
                break
        
        if not found:
            break
    
    return {"packing": best_packing, "bin_weights": best_bin_weights}