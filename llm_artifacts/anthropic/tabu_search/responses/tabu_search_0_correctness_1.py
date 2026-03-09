import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
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
    best_packing, best_bin_weights = ffd(bin_capacity, weights, all_indices)
    best_num_bins = len(best_packing)
    
    if best_num_bins <= 1:
        return {"packing": best_packing, "bin_weights": best_bin_weights}
    
    # Lower bound
    total_weight = sum(weights)
    lb = (total_weight + bin_capacity - 1) // bin_capacity
    
    if best_num_bins <= lb:
        return {"packing": best_packing, "bin_weights": best_bin_weights}
    
    # Tabu search to try packing into fewer bins
    def tabu_search_pack(target_bins, deadline):
        """Try to pack all items into target_bins bins using tabu search.
        deadline is an absolute time (time.time() value).
        Returns packing if feasible found, else None."""
        
        # Initialize: use FFD-like assignment forced into target_bins bins
        sorted_idx = sorted(range(n), key=lambda i: weights[i], reverse=True)
        
        bin_items = [[] for _ in range(target_bins)]
        bin_wt = [0] * target_bins
        item_bin = [0] * n
        
        for idx in sorted_idx:
            w = weights[idx]
            best_b = -1
            best_remain = -1
            for b in range(target_bins):
                remain = bin_capacity - bin_wt[b]
                if remain >= w and (best_b == -1 or remain < best_remain):
                    best_b = b
                    best_remain = remain
            if best_b == -1:
                best_b = 0
                best_remain = bin_capacity - bin_wt[0]
                for b in range(1, target_bins):
                    remain = bin_capacity - bin_wt[b]
                    if remain > best_remain:
                        best_b = b
                        best_remain = remain
            bin_items[best_b].append(idx)
            bin_wt[best_b] += w
            item_bin[idx] = best_b
        
        def compute_overflow():
            return sum(max(0, bin_wt[b] - bin_capacity) for b in range(target_bins))
        
        overflow = compute_overflow()
        if overflow == 0:
            return bin_items, bin_wt
        
        tabu = {}
        tenure = max(5, int(n ** 0.5))
        if tenure > n:
            tenure = n
        
        best_overflow = overflow
        
        iteration = 0
        max_iterations = 500000
        no_improve_limit = max(5000, n * 50)
        no_improve_count = 0
        
        check_freq = 100  # check time every this many iterations
        
        while iteration < max_iterations:
            if iteration % check_freq == 0:
                if time.time() >= deadline:
                    break
            
            iteration += 1
            no_improve_count += 1
            
            # Diversification if no improvement for a while
            if no_improve_count > no_improve_limit:
                for b in range(target_bins):
                    if bin_wt[b] > bin_capacity:
                        items_in_b = list(bin_items[b])
                        random.shuffle(items_in_b)
                        for idx in items_in_b[:max(1, len(items_in_b)//3)]:
                            new_b = random.randint(0, target_bins - 2)
                            if new_b >= b:
                                new_b += 1
                            bin_items[b].remove(idx)
                            bin_wt[b] -= weights[idx]
                            bin_items[new_b].append(idx)
                            bin_wt[new_b] += weights[idx]
                            item_bin[idx] = new_b
                            tabu[(idx, b)] = iteration + tenure
                overflow = compute_overflow()
                no_improve_count = 0
                if overflow == 0:
                    return bin_items, bin_wt
                continue
            
            # Find overflowing bins
            overflow_bins = [b for b in range(target_bins) if bin_wt[b] > bin_capacity]
            
            if not overflow_bins:
                return bin_items, bin_wt
            
            source_bin = random.choice(overflow_bins)
            
            best_move = None
            best_move_overflow = overflow
            best_move_is_tabu = False
            
            items_in_source = bin_items[source_bin]
            
            if len(items_in_source) > 30:
                candidates = random.sample(items_in_source, 30)
            else:
                candidates = list(items_in_source)
            
            for idx in candidates:
                w = weights[idx]
                old_source_overflow = max(0, bin_wt[source_bin] - bin_capacity)
                new_source_overflow = max(0, bin_wt[source_bin] - w - bin_capacity)
                
                for dest_bin in range(target_bins):
                    if dest_bin == source_bin:
                        continue
                    
                    old_dest_overflow = max(0, bin_wt[dest_bin] - bin_capacity)
                    new_dest_overflow = max(0, bin_wt[dest_bin] + w - bin_capacity)
                    
                    delta = (new_source_overflow - old_source_overflow) + (new_dest_overflow - old_dest_overflow)
                    new_overflow = overflow + delta
                    
                    is_tabu = tabu.get((idx, dest_bin), 0) > iteration
                    
                    if new_overflow < best_overflow:
                        if best_move is None or new_overflow < best_move_overflow:
                            best_move = ('move', idx, source_bin, dest_bin)
                            best_move_overflow = new_overflow
                            best_move_is_tabu = False
                    elif not is_tabu:
                        if best_move is None or new_overflow < best_move_overflow or (new_overflow == best_move_overflow and best_move_is_tabu):
                            best_move = ('move', idx, source_bin, dest_bin)
                            best_move_overflow = new_overflow
                            best_move_is_tabu = False
                    else:
                        if best_move is None:
                            best_move = ('move', idx, source_bin, dest_bin)
                            best_move_overflow = new_overflow
                            best_move_is_tabu = True
            
            # Try swaps
            other_bins = [b for b in range(target_bins) if b != source_bin]
            if len(other_bins) > 5:
                swap_target_bins = random.sample(other_bins, 5)
            else:
                swap_target_bins = other_bins
            
            for idx1 in candidates:
                w1 = weights[idx1]
                for dest_bin in swap_target_bins:
                    items_in_dest = bin_items[dest_bin]
                    if len(items_in_dest) > 15:
                        swap_candidates = random.sample(items_in_dest, 15)
                    else:
                        swap_candidates = list(items_in_dest)
                    
                    for idx2 in swap_candidates:
                        w2 = weights[idx2]
                        if w1 == w2:
                            continue
                        
                        old_source_overflow = max(0, bin_wt[source_bin] - bin_capacity)
                        new_source_overflow = max(0, bin_wt[source_bin] - w1 + w2 - bin_capacity)
                        old_dest_overflow = max(0, bin_wt[dest_bin] - bin_capacity)
                        new_dest_overflow = max(0, bin_wt[dest_bin] - w2 + w1 - bin_capacity)
                        
                        delta = (new_source_overflow - old_source_overflow) + (new_dest_overflow - old_dest_overflow)
                        new_overflow = overflow + delta
                        
                        is_tabu = (tabu.get((idx1, dest_bin), 0) > iteration or 
                                   tabu.get((idx2, source_bin), 0) > iteration)
                        
                        if new_overflow < best_overflow:
                            if best_move is None or new_overflow < best_move_overflow:
                                best_move = ('swap', idx1, source_bin, idx2, dest_bin)
                                best_move_overflow = new_overflow
                                best_move_is_tabu = False
                        elif not is_tabu:
                            if best_move is None or new_overflow < best_move_overflow or (new_overflow == best_move_overflow and best_move_is_tabu):
                                best_move = ('swap', idx1, source_bin, idx2, dest_bin)
                                best_move_overflow = new_overflow
                                best_move_is_tabu = False
                        else:
                            if best_move is None:
                                best_move = ('swap', idx1, source_bin, idx2, dest_bin)
                                best_move_overflow = new_overflow
                                best_move_is_tabu = True
            
            if best_move is None:
                continue
            
            if best_move[0] == 'move':
                _, idx, src, dst = best_move
                bin_items[src].remove(idx)
                bin_wt[src] -= weights[idx]
                bin_items[dst].append(idx)
                bin_wt[dst] += weights[idx]
                item_bin[idx] = dst
                tabu[(idx, src)] = iteration + tenure + random.randint(0, tenure // 2)
            else:  # swap
                _, idx1, src, idx2, dst = best_move
                bin_items[src].remove(idx1)
                bin_wt[src] -= weights[idx1]
                bin_items[dst].remove(idx2)
                bin_wt[dst] -= weights[idx2]
                bin_items[src].append(idx2)
                bin_wt[src] += weights[idx2]
                bin_items[dst].append(idx1)
                bin_wt[dst] += weights[idx1]
                item_bin[idx1] = dst
                item_bin[idx2] = src
                tabu[(idx1, src)] = iteration + tenure + random.randint(0, tenure // 2)
                tabu[(idx2, dst)] = iteration + tenure + random.randint(0, tenure // 2)
            
            overflow = best_move_overflow
            
            if overflow < best_overflow:
                best_overflow = overflow
                no_improve_count = 0
                if overflow == 0:
                    return bin_items, bin_wt
            
            if iteration % 10000 == 0:
                tabu = {k: v for k, v in tabu.items() if v > iteration}
        
        return None
    
    # Main loop: try to reduce number of bins
    current_num_bins = best_num_bins
    hard_deadline = start_time + time_limit * 0.95
    
    while current_num_bins > lb:
        if time.time() >= hard_deadline:
            break
        
        target = current_num_bins - 1
        now = time.time()
        remaining = hard_deadline - now
        if remaining <= 0.01:
            break
        
        # Use 70% of remaining time for this attempt
        phase_deadline = now + remaining * 0.7
        
        result = tabu_search_pack(target, phase_deadline)
        
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
        else:
            break
    
    return {"packing": best_packing, "bin_weights": best_bin_weights}