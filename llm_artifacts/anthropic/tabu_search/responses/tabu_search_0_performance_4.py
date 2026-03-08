import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    w_arr = weights  # direct reference
    _time = time.time
    _randint = random.randint
    _random = random.random
    _shuffle = random.shuffle
    _sample = random.sample
    
    # First Fit Decreasing for initial solution
    sorted_idx = sorted(range(n), key=lambda i: w_arr[i], reverse=True)
    
    def ffd_into_open(order):
        bins = []  # list of [items_list, total_weight]
        for idx in order:
            w = w_arr[idx]
            best_b = -1
            best_remain = C + 1
            for b_idx in range(len(bins)):
                remain = C - bins[b_idx][1]
                if remain >= w and remain < best_remain:
                    best_b = b_idx
                    best_remain = remain
            if best_b >= 0:
                bins[best_b][0].append(idx)
                bins[best_b][1] += w
            else:
                bins.append([[idx], w])
        return bins
    
    init_bins = ffd_into_open(sorted_idx)
    best_packing = [b[0] for b in init_bins]
    best_bin_weights = [b[1] for b in init_bins]
    best_num_bins = len(best_packing)
    
    if best_num_bins <= 1:
        return {"packing": best_packing, "bin_weights": best_bin_weights}
    
    # Lower bound
    total_weight = sum(weights)
    lb1 = (total_weight + C - 1) // C
    
    half_C = C / 2.0
    sw = sorted(weights, reverse=True)
    
    def compute_l2():
        best_l2 = lb1
        alphas = set()
        alphas.add(0)
        for w in sw:
            if w > half_C:
                a = C - w
                if 0 <= a <= half_C:
                    alphas.add(a)
            # also try alpha = w if w <= half_C
            if w <= half_C:
                alphas.add(w)
                a2 = C - w
                if 0 <= a2 <= half_C:
                    alphas.add(a2)
        
        for alpha in alphas:
            if alpha < 0 or alpha > half_C:
                continue
            n_j1 = 0
            n_j2 = 0
            sum_j2 = 0
            sum_j3 = 0
            threshold1 = C - alpha
            for w in sw:
                if w > threshold1:
                    n_j1 += 1
                elif w > half_C:
                    n_j2 += 1
                    sum_j2 += w
                elif w >= alpha:
                    sum_j3 += w
            free_in_j2 = n_j2 * C - sum_j2
            remaining_j3 = sum_j3 - free_in_j2
            if remaining_j3 > 0:
                extra_bins = (remaining_j3 + C - 1) // C
            else:
                extra_bins = 0
            l2_val = n_j1 + n_j2 + extra_bins
            if l2_val > best_l2:
                best_l2 = l2_val
        return best_l2
    
    lb = compute_l2()
    
    if best_num_bins <= lb:
        return {"packing": best_packing, "bin_weights": best_bin_weights}
    
    def tabu_search_pack(target_bins, deadline, init_variant=0):
        T = target_bins
        bin_items = [[] for _ in range(T)]
        bin_wt = [0] * T
        item_bin = [0] * n
        item_pos = [0] * n
        
        # Initialization
        if init_variant == 0:
            # Best-fit decreasing
            for idx in sorted_idx:
                w = w_arr[idx]
                best_b = -1
                best_remain = C + 1
                for b in range(T):
                    remain = C - bin_wt[b]
                    if remain >= w and remain < best_remain:
                        best_b = b
                        best_remain = remain
                if best_b == -1:
                    best_b = 0
                    br = C - bin_wt[0]
                    for b in range(1, T):
                        r = C - bin_wt[b]
                        if r > br:
                            best_b = b
                            br = r
                item_pos[idx] = len(bin_items[best_b])
                bin_items[best_b].append(idx)
                bin_wt[best_b] += w
                item_bin[idx] = best_b
        elif init_variant == 1:
            order = list(range(n))
            _shuffle(order)
            for idx in order:
                w = w_arr[idx]
                best_b = -1
                best_remain = C + 1
                for b in range(T):
                    remain = C - bin_wt[b]
                    if remain >= w and remain < best_remain:
                        best_b = b
                        best_remain = remain
                if best_b == -1:
                    best_b = 0
                    br = C - bin_wt[0]
                    for b in range(1, T):
                        r = C - bin_wt[b]
                        if r > br:
                            best_b = b
                            br = r
                item_pos[idx] = len(bin_items[best_b])
                bin_items[best_b].append(idx)
                bin_wt[best_b] += w
                item_bin[idx] = best_b
        elif init_variant == 2:
            # First-fit decreasing
            for idx in sorted_idx:
                w = w_arr[idx]
                placed = False
                for b in range(T):
                    if bin_wt[b] + w <= C:
                        item_pos[idx] = len(bin_items[b])
                        bin_items[b].append(idx)
                        bin_wt[b] += w
                        item_bin[idx] = b
                        placed = True
                        break
                if not placed:
                    best_b = 0
                    br = C - bin_wt[0]
                    for b in range(1, T):
                        r = C - bin_wt[b]
                        if r > br:
                            best_b = b
                            br = r
                    item_pos[idx] = len(bin_items[best_b])
                    bin_items[best_b].append(idx)
                    bin_wt[best_b] += w
                    item_bin[idx] = best_b
        elif init_variant == 3:
            # Load balancing
            for idx in sorted_idx:
                w = w_arr[idx]
                best_b = 0
                bw = bin_wt[0]
                for b in range(1, T):
                    if bin_wt[b] < bw:
                        best_b = b
                        bw = bin_wt[b]
                item_pos[idx] = len(bin_items[best_b])
                bin_items[best_b].append(idx)
                bin_wt[best_b] += w
                item_bin[idx] = best_b
        else:
            # Random best-fit with perturbation
            order = list(sorted_idx)
            # Slightly shuffle: swap adjacent pairs randomly
            for i in range(len(order) - 1):
                if _random() < 0.3:
                    order[i], order[i+1] = order[i+1], order[i]
            for idx in order:
                w = w_arr[idx]
                best_b = -1
                best_remain = C + 1
                for b in range(T):
                    remain = C - bin_wt[b]
                    if remain >= w and remain < best_remain:
                        best_b = b
                        best_remain = remain
                if best_b == -1:
                    best_b = 0
                    br = C - bin_wt[0]
                    for b in range(1, T):
                        r = C - bin_wt[b]
                        if r > br:
                            best_b = b
                            br = r
                item_pos[idx] = len(bin_items[best_b])
                bin_items[best_b].append(idx)
                bin_wt[best_b] += w
                item_bin[idx] = best_b
        
        # Check feasibility
        overflow_sq = 0
        for b in range(T):
            ov = bin_wt[b] - C
            if ov > 0:
                overflow_sq += ov * ov
        
        if overflow_sq == 0:
            return bin_items, bin_wt
        
        # Tabu array: tabu_until[idx * T + dest] = iteration when tabu expires
        tabu_until = [0] * (n * T)
        
        base_tenure = max(7, int(n ** 0.5))
        if base_tenure > n:
            base_tenure = n
        tenure = base_tenure
        max_tenure = base_tenure * 3
        
        best_overflow_sq = overflow_sq
        best_state_bins = None  # save best state
        best_state_wt = None
        
        iteration = 0
        no_improve_count = 0
        no_improve_limit = max(2000, n * 20)
        
        check_interval = 100 if n > 500 else 200
        
        while True:
            if iteration % check_interval == 0:
                if _time() >= deadline:
                    break
            
            iteration += 1
            no_improve_count += 1
            
            # Adaptive tenure
            if no_improve_count > 300 and no_improve_count % 300 == 0:
                tenure = min(max_tenure, tenure + 1)
            
            # Diversification
            if no_improve_count > no_improve_limit:
                # Strong perturbation of overflowing bins
                for b in range(T):
                    if bin_wt[b] > C:
                        items_copy = list(bin_items[b])
                        _shuffle(items_copy)
                        num_to_move = max(1, len(items_copy) // 3)
                        for idx in items_copy[:num_to_move]:
                            w = w_arr[idx]
                            new_b = _randint(0, T - 2)
                            if new_b >= b:
                                new_b += 1
                            pos = item_pos[idx]
                            last = bin_items[b][-1]
                            bin_items[b][pos] = last
                            item_pos[last] = pos
                            bin_items[b].pop()
                            bin_wt[b] -= w
                            item_pos[idx] = len(bin_items[new_b])
                            bin_items[new_b].append(idx)
                            bin_wt[new_b] += w
                            item_bin[idx] = new_b
                            tabu_until[idx * T + b] = iteration + tenure
                
                overflow_sq = 0
                for b in range(T):
                    ov = bin_wt[b] - C
                    if ov > 0:
                        overflow_sq += ov * ov
                no_improve_count = 0
                tenure = base_tenure
                if overflow_sq == 0:
                    return bin_items, bin_wt
                continue
            
            # Find the most overflowing bin
            max_ov = 0
            max_ov_bin = -1
            for b in range(T):
                ov = bin_wt[b] - C
                if ov > max_ov:
                    max_ov = ov
                    max_ov_bin = b
            
            if max_ov_bin == -1:
                return bin_items, bin_wt
            
            # Sometimes pick a random overflowing bin
            if _random() < 0.2:
                overflow_bins = [b for b in range(T) if bin_wt[b] > C]
                source_bin = overflow_bins[_randint(0, len(overflow_bins) - 1)] if overflow_bins else max_ov_bin
            else:
                source_bin = max_ov_bin
            
            src_items = bin_items[source_bin]
            src_wt = bin_wt[source_bin]
            old_src_ov = src_wt - C
            old_src_sq = old_src_ov * old_src_ov if old_src_ov > 0 else 0
            
            src_len = len(src_items)
            cand_limit = min(40, src_len)
            if src_len > cand_limit:
                candidates = _sample(src_items, cand_limit)
            else:
                candidates = list(src_items)
            
            best_delta = 1 << 60
            best_move = None
            best_is_tabu = True
            
            iter_val = iteration
            
            # Evaluate moves
            for idx in candidates:
                w = w_arr[idx]
                new_src_wt = src_wt - w
                new_src_ov = new_src_wt - C
                src_delta = (new_src_ov * new_src_ov if new_src_ov > 0 else 0) - old_src_sq
                idx_T = idx * T
                
                for dest_bin in range(T):
                    if dest_bin == source_bin:
                        continue
                    
                    dst_wt = bin_wt[dest_bin]
                    old_dst_ov = dst_wt - C
                    new_dst_wt = dst_wt + w
                    new_dst_ov = new_dst_wt - C
                    
                    delta = src_delta + (new_dst_ov * new_dst_ov if new_dst_ov > 0 else 0) - (old_dst_ov * old_dst_ov if old_dst_ov > 0 else 0)
                    
                    is_tabu = tabu_until[idx_T + dest_bin] > iter_val
                    new_ov = overflow_sq + delta
                    
                    # Aspiration criterion
                    if new_ov < best_overflow_sq:
                        if delta < best_delta or best_is_tabu:
                            best_delta = delta
                            best_move = (0, idx, source_bin, dest_bin, 0)
                            best_is_tabu = False
                    elif not is_tabu:
                        if delta < best_delta or (delta == best_delta and best_is_tabu):
                            best_delta = delta
                            best_move = (0, idx, source_bin, dest_bin, 0)
                            best_is_tabu = False
                    elif best_move is None:
                        best_delta = delta
                        best_move = (0, idx, source_bin, dest_bin, 0)
                        best_is_tabu = True
            
            # Evaluate swaps with non-overflowing bins (or least overflowing)
            # Only do swaps periodically or when moves don't help
            do_swaps = (iteration % 3 == 0) or (best_move is not None and best_delta >= 0)
            
            if do_swaps:
                # Pick destination bins for swaps
                swap_bins = []
                for b in range(T):
                    if b != source_bin and bin_wt[b] <= C:
                        swap_bins.append(b)
                if not swap_bins:
                    swap_bins = [b for b in range(T) if b != source_bin]
                
                swap_bin_limit = min(6, len(swap_bins))
                if len(swap_bins) > swap_bin_limit:
                    swap_bins = _sample(swap_bins, swap_bin_limit)
                
                swap_cand_limit_src = min(20, len(candidates))
                src_cands = candidates[:swap_cand_limit_src]
                
                for idx1 in src_cands:
                    w1 = w_arr[idx1]
                    new_src_base = src_wt - w1
                    idx1_T = idx1 * T
                    
                    for dest_bin in swap_bins:
                        dst_wt = bin_wt[dest_bin]
                        old_dst_ov = dst_wt - C
                        old_dst_sq = old_dst_ov * old_dst_ov if old_dst_ov > 0 else 0
                        
                        d_items = bin_items[dest_bin]
                        d_len = len(d_items)
                        sc_limit = min(15, d_len)
                        if d_len > sc_limit:
                            swap_cands = _sample(d_items, sc_limit)
                        else:
                            swap_cands = d_items
                        
                        for idx2 in swap_cands:
                            w2 = w_arr[idx2]
                            diff = w1 - w2
                            if diff == 0:
                                continue
                            
                            new_src_wt2 = new_src_base + w2
                            ns_ov = new_src_wt2 - C
                            new_dst_wt2 = dst_wt + diff
                            nd_ov = new_dst_wt2 - C
                            
                            delta = (ns_ov * ns_ov if ns_ov > 0 else 0) - old_src_sq + (nd_ov * nd_ov if nd_ov > 0 else 0) - old_dst_sq
                            
                            new_ov = overflow_sq + delta
                            is_tabu = tabu_until[idx1_T + dest_bin] > iter_val or tabu_until[idx2 * T + source_bin] > iter_val
                            
                            if new_ov < best_overflow_sq:
                                if delta < best_delta or best_is_tabu:
                                    best_delta = delta
                                    best_move = (1, idx1, source_bin, idx2, dest_bin)
                                    best_is_tabu = False
                            elif not is_tabu:
                                if delta < best_delta or (delta == best_delta and best_is_tabu):
                                    best_delta = delta
                                    best_move = (1, idx1, source_bin, idx2, dest_bin)
                                    best_is_tabu = False
                            elif best_move is None:
                                best_delta = delta
                                best_move = (1, idx1, source_bin, idx2, dest_bin)
                                best_is_tabu = True
            
            if best_move is None:
                continue
            
            # Apply move
            mv_type = best_move[0]
            if mv_type == 0:
                _, idx, src, dst, _ = best_move
                w = w_arr[idx]
                pos = item_pos[idx]
                last = bin_items[src][-1]
                bin_items[src][pos] = last
                item_pos[last] = pos
                bin_items[src].pop()
                bin_wt[src] -= w
                item_pos[idx] = len(bin_items[dst])
                bin_items[dst].append(idx)
                bin_wt[dst] += w
                item_bin[idx] = dst
                tabu_until[idx * T + src] = iter_val + tenure + _randint(0, max(1, tenure // 3))
            else:
                _, idx1, src, idx2, dst = best_move
                w1 = w_arr[idx1]
                w2 = w_arr[idx2]
                pos1 = item_pos[idx1]
                last1 = bin_items[src][-1]
                bin_items[src][pos1] = last1
                item_pos[last1] = pos1
                bin_items[src].pop()
                bin_wt[src] -= w1
                pos2 = item_pos[idx2]
                last2 = bin_items[dst][-1]
                bin_items[dst][pos2] = last2
                item_pos[last2] = pos2
                bin_items[dst].pop()
                bin_wt[dst] -= w2
                item_pos[idx2] = len(bin_items[src])
                bin_items[src].append(idx2)
                bin_wt[src] += w2
                item_pos[idx1] = len(bin_items[dst])
                bin_items[dst].append(idx1)
                bin_wt[dst] += w1
                item_bin[idx1] = dst
                item_bin[idx2] = src
                tabu_until[idx1 * T + src] = iter_val + tenure + _randint(0, max(1, tenure // 3))
                tabu_until[idx2 * T + dst] = iter_val + tenure + _randint(0, max(1, tenure // 3))
            
            overflow_sq += best_delta
            
            if overflow_sq < best_overflow_sq:
                best_overflow_sq = overflow_sq
                no_improve_count = 0
                tenure = base_tenure
                if overflow_sq == 0:
                    return bin_items, bin_wt
        
        return None
    
    current_num_bins = best_num_bins
    hard_deadline = start_time + time_limit * 0.98
    
    init_variants = [0, 1, 2, 3, 4, 1, 4, 1, 4, 1]
    
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
            
            # Give each attempt a fraction of remaining time, with minimum
            phase_time = max(0.5, remaining * 0.4)
            phase_deadline = now + phase_time
            if phase_deadline > hard_deadline:
                phase_deadline = hard_deadline
            
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