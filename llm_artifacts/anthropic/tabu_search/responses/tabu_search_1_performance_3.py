import random
import time
import math

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    # --- Lower Bound ---
    total_weight = sum(weights)
    L1 = math.ceil(total_weight / C)
    
    # L2 lower bound (Martello-Toth)
    def compute_L2():
        best_lb = L1
        half_c = C / 2.0
        for t in range(1, C // 2 + 1):
            n1 = 0; sum_n1 = 0
            n2 = 0; sum_n2 = 0
            sum_n3 = 0
            threshold_high = C - t
            for w in weights:
                if w > threshold_high:
                    n1 += 1; sum_n1 += w
                elif w > half_c:
                    n2 += 1; sum_n2 += w
                elif w <= t:
                    sum_n3 += w
            residual = (n1 * C - sum_n1) + (n2 * C - sum_n2)
            leftover = sum_n3 - residual
            extra = max(0, math.ceil(leftover / C)) if leftover > 0 else 0
            lb = n1 + n2 + extra
            if lb > best_lb:
                best_lb = lb
        return best_lb
    
    lower_bound = compute_L2()
    
    # --- Initial Solutions ---
    indices_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    def first_fit_decreasing():
        bins = []; bw = []
        for idx in indices_sorted:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if bw[b] + w <= C:
                    bins[b].append(idx); bw[b] += w
                    placed = True; break
            if not placed:
                bins.append([idx]); bw.append(w)
        return bins, bw
    
    def best_fit_decreasing():
        bins = []; bw = []
        for idx in indices_sorted:
            w = weights[idx]
            bb = -1; br = C + 1
            for b in range(len(bins)):
                r = C - bw[b]
                if r >= w and r < br:
                    br = r; bb = b
            if bb >= 0:
                bins[bb].append(idx); bw[bb] += w
            else:
                bins.append([idx]); bw.append(w)
        return bins, bw
    
    ffd_bins, ffd_bw = first_fit_decreasing()
    bfd_bins, bfd_bw = best_fit_decreasing()
    
    if len(bfd_bins) <= len(ffd_bins):
        current_bins, current_bw = bfd_bins, bfd_bw
    else:
        current_bins, current_bw = ffd_bins, ffd_bw
    
    best_num_bins = len(current_bins)
    best_packing = [list(b) for b in current_bins]
    best_bin_weights = list(current_bw)
    
    if best_num_bins <= lower_bound:
        return {"packing": best_packing, "bin_weights": best_bin_weights}
    
    def elapsed():
        return time.time() - start_time
    
    target_bins = best_num_bins - 1
    
    while target_bins >= lower_bound and elapsed() < time_limit * 0.95:
        K = best_num_bins
        if target_bins >= K:
            break
        
        success = False
        tried_eliminate = set()
        
        # Compute bin weights for elimination ordering
        bin_weight_list = list(best_bin_weights)
        
        # Try eliminating bins, lightest first
        bin_order = sorted(range(K), key=lambda b: bin_weight_list[b])
        
        for elim_idx in range(min(K, 10)):
            if elapsed() >= time_limit * 0.95:
                break
            if success:
                break
            
            if elim_idx < len(bin_order):
                elim_bin = bin_order[elim_idx]
            else:
                break
            
            if elim_bin in tried_eliminate:
                continue
            tried_eliminate.add(elim_bin)
            
            new_K = K - 1
            
            # Build new state without elim_bin
            new_bins = []
            new_bw = []
            for b in range(K):
                if b == elim_bin:
                    continue
                new_bins.append(set(best_packing[b]))
                new_bw.append(bin_weight_list[b])
            
            # Redistribute items from eliminated bin
            items_to_place = sorted(best_packing[elim_bin], key=lambda i: weights[i], reverse=True)
            
            item_to_bin = [0] * n
            for b in range(new_K):
                for idx in new_bins[b]:
                    item_to_bin[idx] = b
            
            for idx in items_to_place:
                w = weights[idx]
                # Best fit
                bb = -1; br = C + 1
                for b in range(new_K):
                    r = C - new_bw[b]
                    if r >= w and r < br:
                        br = r; bb = b
                if bb == -1:
                    # Most remaining capacity
                    bb = 0; br2 = C - new_bw[0]
                    for b in range(1, new_K):
                        r = C - new_bw[b]
                        if r > br2:
                            br2 = r; bb = b
                new_bins[bb].add(idx)
                new_bw[bb] += w
                item_to_bin[idx] = bb
            
            overflow = [max(0, new_bw[b] - C) for b in range(new_K)]
            total_overflow = sum(overflow)
            
            if total_overflow == 0:
                best_num_bins = new_K
                best_packing = [list(b) for b in new_bins]
                best_bin_weights = list(new_bw)
                success = True
                break
            
            # --- Tabu Search to eliminate overflow ---
            tabu_move = {}  # (item, bin) -> iteration expiry
            base_tenure = max(7, int(n * 0.1))
            tabu_tenure = base_tenure
            
            best_overflow = total_overflow
            best_state_bins = [set(b) for b in new_bins]
            best_state_bw = list(new_bw)
            best_state_itb = list(item_to_bin)
            
            overfilled = set(b for b in range(new_K) if overflow[b] > 0)
            
            iteration = 0
            no_improve = 0
            max_no_improve = max(1000, n * 5)
            
            # Precompute weight array for fast access
            w_arr = weights
            
            while total_overflow > 0 and elapsed() < time_limit * 0.95:
                iteration += 1
                
                if no_improve > max_no_improve:
                    break
                
                best_move = None
                best_delta = float('inf')
                
                ovf_list = list(overfilled)
                
                # === MOVE neighborhood ===
                for src in ovf_list:
                    src_items = list(new_bins[src])
                    # Sort by weight descending, try heaviest first
                    src_items.sort(key=lambda i: w_arr[i], reverse=True)
                    limit_items = min(len(src_items), 30 if n > 300 else len(src_items))
                    
                    for item_idx in range(limit_items):
                        item = src_items[item_idx]
                        w = w_arr[item]
                        
                        old_src_of = overflow[src]
                        new_src_w = new_bw[src] - w
                        new_src_of = max(0, new_src_w - C)
                        d_src = new_src_of - old_src_of
                        
                        # Try all destination bins
                        for dst in range(new_K):
                            if dst == src:
                                continue
                            
                            old_dst_of = overflow[dst]
                            new_dst_w = new_bw[dst] + w
                            new_dst_of = max(0, new_dst_w - C)
                            d_dst = new_dst_of - old_dst_of
                            
                            delta = d_src + d_dst
                            
                            is_tabu = (item, dst) in tabu_move and tabu_move[(item, dst)] > iteration
                            
                            if is_tabu:
                                if total_overflow + delta < best_overflow:
                                    if delta < best_delta:
                                        best_delta = delta
                                        best_move = ('m', item, src, dst)
                            else:
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = ('m', item, src, dst)
                
                # === SWAP neighborhood ===
                for src in ovf_list:
                    src_items = list(new_bins[src])
                    src_items.sort(key=lambda i: w_arr[i], reverse=True)
                    limit_a = min(len(src_items), 15 if n > 300 else 30)
                    
                    for ai in range(limit_a):
                        item_a = src_items[ai]
                        w_a = w_arr[item_a]
                        
                        for dst in range(new_K):
                            if dst == src:
                                continue
                            
                            dst_items = list(new_bins[dst])
                            # Sort by weight ascending to find lighter items to swap with
                            dst_items.sort(key=lambda i: w_arr[i])
                            limit_b = min(len(dst_items), 15 if n > 300 else 30)
                            
                            for bi in range(limit_b):
                                item_b = dst_items[bi]
                                w_b = w_arr[item_b]
                                if w_b >= w_a:
                                    break  # sorted ascending, no point continuing
                                
                                new_src_w = new_bw[src] - w_a + w_b
                                new_dst_w = new_bw[dst] - w_b + w_a
                                
                                new_src_of = max(0, new_src_w - C)
                                new_dst_of = max(0, new_dst_w - C)
                                
                                delta = (new_src_of - overflow[src]) + (new_dst_of - overflow[dst])
                                
                                ta = (item_a, dst) in tabu_move and tabu_move[(item_a, dst)] > iteration
                                tb = (item_b, src) in tabu_move and tabu_move[(item_b, src)] > iteration
                                is_tabu = ta or tb
                                
                                if is_tabu:
                                    if total_overflow + delta < best_overflow:
                                        if delta < best_delta:
                                            best_delta = delta
                                            best_move = ('s', item_a, src, item_b, dst)
                                else:
                                    if delta < best_delta:
                                        best_delta = delta
                                        best_move = ('s', item_a, src, item_b, dst)
                
                if best_move is None:
                    # Try random move to escape
                    if ovf_list:
                        src = random.choice(ovf_list)
                        if new_bins[src]:
                            item = random.choice(list(new_bins[src]))
                            dst = random.randint(0, new_K - 2)
                            if dst >= src:
                                dst += 1
                            best_move = ('m', item, src, dst)
                            w = w_arr[item]
                            old_src_of = overflow[src]
                            new_src_of = max(0, new_bw[src] - w - C)
                            old_dst_of = overflow[dst]
                            new_dst_of = max(0, new_bw[dst] + w - C)
                            best_delta = (new_src_of - old_src_of) + (new_dst_of - old_dst_of)
                        else:
                            break
                    else:
                        break
                
                # Execute move
                if best_move[0] == 'm':
                    _, item, src, dst = best_move
                    w = w_arr[item]
                    
                    new_bins[src].discard(item)
                    new_bins[dst].add(item)
                    new_bw[src] -= w
                    new_bw[dst] += w
                    item_to_bin[item] = dst
                    
                    overflow[src] = max(0, new_bw[src] - C)
                    overflow[dst] = max(0, new_bw[dst] - C)
                    
                    total_overflow += best_delta
                    
                    if overflow[src] > 0: overfilled.add(src)
                    else: overfilled.discard(src)
                    if overflow[dst] > 0: overfilled.add(dst)
                    else: overfilled.discard(dst)
                    
                    tabu_move[(item, src)] = iteration + tabu_tenure + random.randint(0, 3)
                    
                else:  # swap
                    _, item_a, src, item_b, dst = best_move
                    w_a = w_arr[item_a]
                    w_b = w_arr[item_b]
                    
                    new_bins[src].discard(item_a)
                    new_bins[src].add(item_b)
                    new_bins[dst].discard(item_b)
                    new_bins[dst].add(item_a)
                    
                    new_bw[src] = new_bw[src] - w_a + w_b
                    new_bw[dst] = new_bw[dst] - w_b + w_a
                    item_to_bin[item_a] = dst
                    item_to_bin[item_b] = src
                    
                    overflow[src] = max(0, new_bw[src] - C)
                    overflow[dst] = max(0, new_bw[dst] - C)
                    
                    total_overflow += best_delta
                    
                    if overflow[src] > 0: overfilled.add(src)
                    else: overfilled.discard(src)
                    if overflow[dst] > 0: overfilled.add(dst)
                    else: overfilled.discard(dst)
                    
                    tabu_move[(item_a, src)] = iteration + tabu_tenure + random.randint(0, 3)
                    tabu_move[(item_b, dst)] = iteration + tabu_tenure + random.randint(0, 3)
                
                if total_overflow < best_overflow:
                    best_overflow = total_overflow
                    best_state_bins = [set(b) for b in new_bins]
                    best_state_bw = list(new_bw)
                    best_state_itb = list(item_to_bin)
                    no_improve = 0
                else:
                    no_improve += 1
                
                if total_overflow == 0:
                    break
                
                # Perturbation
                if no_improve > 0 and no_improve % 300 == 0:
                    # Random perturbation
                    for _ in range(min(3, len(overfilled))):
                        if not overfilled:
                            break
                        sb = random.choice(list(overfilled))
                        if not new_bins[sb]:
                            continue
                        item = random.choice(list(new_bins[sb]))
                        w = w_arr[item]
                        db = random.randint(0, new_K - 1)
                        if db == sb:
                            continue
                        new_bins[sb].discard(item)
                        new_bins[db].add(item)
                        new_bw[sb] -= w
                        new_bw[db] += w
                        item_to_bin[item] = db
                        overflow[sb] = max(0, new_bw[sb] - C)
                        overflow[db] = max(0, new_bw[db] - C)
                        if overflow[sb] > 0: overfilled.add(sb)
                        else: overfilled.discard(sb)
                        if overflow[db] > 0: overfilled.add(db)
                        else: overfilled.discard(db)
                        tabu_move[(item, sb)] = iteration + tabu_tenure
                    total_overflow = sum(overflow)
                    tabu_tenure = random.randint(max(5, n // 20), max(10, n // 5))
                
                if no_improve > 0 and no_improve % 800 == 0:
                    # Restore best state
                    new_bins = [set(b) for b in best_state_bins]
                    new_bw = list(best_state_bw)
                    item_to_bin = list(best_state_itb)
                    overflow = [max(0, new_bw[b] - C) for b in range(new_K)]
                    total_overflow = sum(overflow)
                    overfilled = set(b for b in range(new_K) if overflow[b] > 0)
                    tabu_move.clear()
                    tabu_tenure = base_tenure + random.randint(-2, 5)
                
                # Clean tabu dict periodically
                if iteration % 5000 == 0:
                    keys_to_del = [k for k, v in tabu_move.items() if v <= iteration]
                    for k in keys_to_del:
                        del tabu_move[k]
            
            if total_overflow == 0:
                best_num_bins = new_K
                best_packing = [list(b) for b in new_bins if b]
                best_bin_weights = [new_bw[b] for b in range(new_K) if new_bins[b]]
                success = True
                break
        
        if success:
            if best_num_bins <= lower_bound:
                break
            target_bins = best_num_bins - 1
        else:
            break
    
    # Filter empty bins
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
