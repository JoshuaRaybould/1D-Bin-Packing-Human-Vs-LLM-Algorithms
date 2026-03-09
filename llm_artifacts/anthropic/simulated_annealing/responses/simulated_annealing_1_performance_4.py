import random
import time
import math

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Initial solution: First Fit Decreasing
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    bins_list = []  # list of sets of item indices
    bin_wts = []  # bin weights
    item_bin = [0] * n  # which bin each item is in
    
    for idx in sorted_indices:
        w = weights[idx]
        # Best fit: find the fullest bin that can still fit this item
        best_b = -1
        best_bw = -1
        for b in range(len(bins_list)):
            remaining = bin_capacity - bin_wts[b]
            if remaining >= w and bin_wts[b] > best_bw:
                best_bw = bin_wts[b]
                best_b = b
        if best_b >= 0:
            bins_list[best_b].add(idx)
            bin_wts[best_b] += w
            item_bin[idx] = best_b
        else:
            item_bin[idx] = len(bins_list)
            bins_list.append({idx})
            bin_wts.append(w)
    
    num_bins = len(bins_list)
    
    # Fitness: -num_bins * PENALTY + sum(bin_weight^2)
    PENALTY = bin_capacity * bin_capacity * (n + 1)
    
    def compute_fitness(bw, nb):
        s = 0
        for x in bw:
            s += x * x
        return -nb * PENALTY + s
    
    current_fitness = compute_fitness(bin_wts, num_bins)
    
    # Best solution tracking
    best_num_bins = num_bins
    best_fitness = current_fitness
    best_packing = [list(b) for b in bins_list]
    best_bin_wts = list(bin_wts)
    
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    if remaining_time <= 0.05:
        return {"packing": best_packing, "bin_weights": best_bin_wts}
    
    # Local variable caching for hot loop
    _random = random.random
    _randrange = random.randrange
    _randint = random.randint
    _choice = random.choice
    _time = time.time
    _weights = weights
    _bin_capacity = bin_capacity
    _exp = math.exp
    _log = math.log
    
    # SA parameters - use higher starting temperature for better exploration
    T_start = bin_capacity * bin_capacity * 0.5
    T_min = 1.0
    T = T_start
    
    iteration = 0
    check_interval = 3000
    
    # Find lightest bin
    def find_lightest_bin():
        if num_bins == 0:
            return 0
        min_w = bin_wts[0]
        min_idx = 0
        for i in range(1, num_bins):
            if bin_wts[i] < min_w:
                min_w = bin_wts[i]
                min_idx = i
        return min_idx
    
    def find_two_lightest():
        if num_bins <= 1:
            return 0, -1
        if bin_wts[0] <= bin_wts[1]:
            l1, l2 = 0, 1
        else:
            l1, l2 = 1, 0
        for i in range(2, num_bins):
            if bin_wts[i] < bin_wts[l1]:
                l2 = l1
                l1 = i
            elif bin_wts[i] < bin_wts[l2]:
                l2 = i
        return l1, l2
    
    lightest_bin = find_lightest_bin()
    
    def remove_empty_bin(src):
        nonlocal num_bins
        last = num_bins - 1
        if src != last:
            bins_list[src] = bins_list[last]
            bin_wts[src] = bin_wts[last]
            for it in bins_list[src]:
                item_bin[it] = src
        bins_list.pop()
        bin_wts.pop()
        num_bins -= 1
    
    def try_empty_bin(lb):
        """Try to empty bin lb by redistributing all its items."""
        nonlocal num_bins, current_fitness, lightest_bin
        if lb >= num_bins or len(bins_list[lb]) == 0:
            return False
        
        items_in_lb = sorted(bins_list[lb], key=lambda i: _weights[i], reverse=True)
        placements = []
        temp_wts = list(bin_wts)
        
        for it in items_in_lb:
            w = _weights[it]
            best_b = -1
            best_remaining = _bin_capacity + 1
            for b in range(num_bins):
                if b == lb:
                    continue
                rem = _bin_capacity - temp_wts[b]
                if rem >= w and rem < best_remaining:
                    best_remaining = rem
                    best_b = b
            if best_b >= 0:
                placements.append((it, best_b))
                temp_wts[best_b] += w
            else:
                return False
        
        # Apply all placements
        for it, tgt in placements:
            w = _weights[it]
            bins_list[lb].discard(it)
            bins_list[tgt].add(it)
            bin_wts[tgt] += w
            item_bin[it] = tgt
        bin_wts[lb] = 0
        remove_empty_bin(lb)
        current_fitness = compute_fitness(bin_wts, num_bins)
        lightest_bin = find_lightest_bin() if num_bins > 0 else 0
        return True
    
    def save_best():
        nonlocal best_num_bins, best_fitness, best_packing, best_bin_wts
        best_num_bins = num_bins
        best_fitness = current_fitness
        best_packing = [list(b) for b in bins_list]
        best_bin_wts = list(bin_wts)
    
    def restore_best():
        nonlocal num_bins, current_fitness, lightest_bin
        nonlocal bins_list, bin_wts
        bins_list_new = [set(b) for b in best_packing]
        bin_wts_new = list(best_bin_wts)
        # We need to reassign
        bins_list.clear()
        bins_list.extend(bins_list_new)
        bin_wts.clear()
        bin_wts.extend(bin_wts_new)
        num_bins = len(bins_list)
        for b_idx in range(num_bins):
            for it in bins_list[b_idx]:
                item_bin[it] = b_idx
        current_fitness = compute_fitness(bin_wts, num_bins)
        lightest_bin = find_lightest_bin() if num_bins > 0 else 0
    
    # Try emptying lightest bins at the start
    for _ in range(num_bins):
        lb = find_lightest_bin()
        if not try_empty_bin(lb):
            break
    if num_bins < best_num_bins or (num_bins == best_num_bins and current_fitness > best_fitness):
        save_best()
    
    last_improvement_iter = 0
    phase_iters = 0
    
    while True:
        if iteration % check_interval == 0:
            elapsed = _time() - start_time
            if elapsed >= time_limit - 0.02:
                break
            # Temperature: exponential cooling
            fraction_remaining = max(0.001, (time_limit - elapsed) / time_limit)
            T = T_start * fraction_remaining
            if T < T_min:
                T = T_min
            
            # Restart if too far from best
            if num_bins > best_num_bins + 2 and iteration - last_improvement_iter >= 10000:
                restore_best()
                last_improvement_iter = iteration
        
        iteration += 1
        
        if num_bins <= 1:
            break
        
        # Periodically try to empty lightest bins
        if iteration % 500 == 0:
            lb = find_lightest_bin()
            lightest_bin = lb
            if try_empty_bin(lb):
                if num_bins < best_num_bins or (num_bins == best_num_bins and current_fitness > best_fitness):
                    save_best()
                    last_improvement_iter = iteration
                continue
        
        # Also try emptying second lightest periodically
        if iteration % 2000 == 0 and num_bins >= 2:
            l1, l2 = find_two_lightest()
            if try_empty_bin(l2):
                if num_bins < best_num_bins or (num_bins == best_num_bins and current_fitness > best_fitness):
                    save_best()
                    last_improvement_iter = iteration
                lightest_bin = find_lightest_bin()
                continue
        
        r = _random()
        
        if r < 0.35:
            # MOVE from lightest bin to random or best-fit target
            lb = lightest_bin
            if lb >= num_bins or len(bins_list[lb]) == 0:
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                lb = lightest_bin
            
            lb_items = bins_list[lb]
            if len(lb_items) == 0:
                continue
            
            # Pick random item from lightest bin
            item_list = list(lb_items)
            item = item_list[_randrange(len(item_list))]
            w = _weights[item]
            src = lb
            
            # Best fit target
            best_b = -1
            best_bw = -1
            for b in range(num_bins):
                if b == src:
                    continue
                if bin_wts[b] + w <= _bin_capacity and bin_wts[b] > best_bw:
                    best_bw = bin_wts[b]
                    best_b = b
            
            if best_b < 0:
                continue
            
            tgt = best_b
            old_src_w = bin_wts[src]
            old_tgt_w = bin_wts[tgt]
            new_src_w = old_src_w - w
            new_tgt_w = old_tgt_w + w
            
            # Always accept moves from lightest bin to best-fit
            bins_list[src].remove(item)
            bins_list[tgt].add(item)
            bin_wts[src] = new_src_w
            bin_wts[tgt] = new_tgt_w
            item_bin[item] = tgt
            
            delta = (new_src_w * new_src_w + new_tgt_w * new_tgt_w) - \
                    (old_src_w * old_src_w + old_tgt_w * old_tgt_w)
            
            src_becomes_empty = (new_src_w == 0)
            if src_becomes_empty:
                delta += PENALTY
            current_fitness += delta
            
            if src_becomes_empty:
                remove_empty_bin(src)
            
            lightest_bin = find_lightest_bin() if num_bins > 0 else 0
            
            if num_bins < best_num_bins or (num_bins == best_num_bins and current_fitness > best_fitness):
                save_best()
                last_improvement_iter = iteration
        
        elif r < 0.60:
            # Standard random MOVE with SA acceptance
            item = _randrange(n)
            w = _weights[item]
            src = item_bin[item]
            
            if num_bins <= 1:
                continue
            
            tgt = _randrange(num_bins - 1)
            if tgt >= src:
                tgt += 1
            
            if bin_wts[tgt] + w > _bin_capacity:
                continue
            
            old_src_w = bin_wts[src]
            old_tgt_w = bin_wts[tgt]
            new_src_w = old_src_w - w
            new_tgt_w = old_tgt_w + w
            
            src_becomes_empty = (new_src_w == 0)
            
            delta = (new_src_w * new_src_w + new_tgt_w * new_tgt_w) - \
                    (old_src_w * old_src_w + old_tgt_w * old_tgt_w)
            if src_becomes_empty:
                delta += PENALTY
            
            accept = False
            if delta >= 0 or src_becomes_empty:
                accept = True
            elif T > 0:
                try:
                    ratio = delta / T
                    if ratio > -500:
                        if _random() < _exp(ratio):
                            accept = True
                except (OverflowError, ZeroDivisionError):
                    pass
            
            if accept:
                bins_list[src].remove(item)
                bins_list[tgt].add(item)
                bin_wts[src] = new_src_w
                bin_wts[tgt] = new_tgt_w
                item_bin[item] = tgt
                current_fitness += delta
                
                if src_becomes_empty:
                    remove_empty_bin(src)
                
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                
                if num_bins < best_num_bins or (num_bins == best_num_bins and current_fitness > best_fitness):
                    save_best()
                    last_improvement_iter = iteration
        
        elif r < 0.85:
            # SWAP two items from different bins
            item1 = _randrange(n)
            item2 = _randrange(n)
            
            b1 = item_bin[item1]
            b2 = item_bin[item2]
            
            if b1 == b2:
                continue
            
            w1 = _weights[item1]
            w2 = _weights[item2]
            
            if w1 == w2:
                continue
            
            new_b1_w = bin_wts[b1] - w1 + w2
            new_b2_w = bin_wts[b2] - w2 + w1
            
            if new_b1_w > _bin_capacity or new_b2_w > _bin_capacity:
                continue
            
            if new_b1_w < 0 or new_b2_w < 0:
                continue
            
            old_contrib = bin_wts[b1] * bin_wts[b1] + bin_wts[b2] * bin_wts[b2]
            new_contrib = new_b1_w * new_b1_w + new_b2_w * new_b2_w
            delta = new_contrib - old_contrib
            
            accept = False
            if delta > 0:
                accept = True
            elif delta == 0:
                continue
            else:
                if T > 0:
                    try:
                        ratio = delta / T
                        if ratio > -500:
                            if _random() < _exp(ratio):
                                accept = True
                    except (OverflowError, ZeroDivisionError):
                        pass
            
            if accept:
                bins_list[b1].remove(item1)
                bins_list[b1].add(item2)
                bins_list[b2].remove(item2)
                bins_list[b2].add(item1)
                bin_wts[b1] = new_b1_w
                bin_wts[b2] = new_b2_w
                item_bin[item1] = b2
                item_bin[item2] = b1
                current_fitness += delta
                
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                
                if num_bins < best_num_bins or (num_bins == best_num_bins and current_fitness > best_fitness):
                    save_best()
                    last_improvement_iter = iteration
        
        else:
            # Targeted swap: pick item from lightest bin and swap with smaller item from another bin
            lb = lightest_bin
            if lb >= num_bins or len(bins_list[lb]) == 0:
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                lb = lightest_bin
            
            lb_items = bins_list[lb]
            if len(lb_items) == 0:
                continue
            
            item_list = list(lb_items)
            item1 = item_list[_randrange(len(item_list))]
            w1 = _weights[item1]
            b1 = lb
            
            # Pick a random other bin
            b2 = _randrange(num_bins - 1)
            if b2 >= b1:
                b2 += 1
            
            # Pick random item from b2
            b2_items = bins_list[b2]
            if len(b2_items) == 0:
                continue
            b2_list = list(b2_items)
            item2 = b2_list[_randrange(len(b2_list))]
            w2 = _weights[item2]
            
            if w1 == w2:
                continue
            
            # We want to make lightest bin lighter, so swap w1 (from lightest) with w2 where w2 < w1
            new_b1_w = bin_wts[b1] - w1 + w2
            new_b2_w = bin_wts[b2] - w2 + w1
            
            if new_b1_w > _bin_capacity or new_b2_w > _bin_capacity:
                continue
            if new_b1_w < 0 or new_b2_w < 0:
                continue
            
            old_contrib = bin_wts[b1] * bin_wts[b1] + bin_wts[b2] * bin_wts[b2]
            new_contrib = new_b1_w * new_b1_w + new_b2_w * new_b2_w
            delta = new_contrib - old_contrib
            
            # Bias toward accepting if lightest bin gets lighter
            accept = False
            if new_b1_w < bin_wts[b1]:
                # Lightest bin got lighter - good for emptying
                accept = True
            elif delta > 0:
                accept = True
            elif delta == 0:
                if new_b1_w < bin_wts[b1]:
                    accept = True
                else:
                    continue
            else:
                if T > 0:
                    try:
                        ratio = delta / T
                        if ratio > -500:
                            if _random() < _exp(ratio):
                                accept = True
                    except (OverflowError, ZeroDivisionError):
                        pass
            
            if accept:
                bins_list[b1].remove(item1)
                bins_list[b1].add(item2)
                bins_list[b2].remove(item2)
                bins_list[b2].add(item1)
                bin_wts[b1] = new_b1_w
                bin_wts[b2] = new_b2_w
                item_bin[item1] = b2
                item_bin[item2] = b1
                current_fitness += delta
                
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                
                if num_bins < best_num_bins or (num_bins == best_num_bins and current_fitness > best_fitness):
                    save_best()
                    last_improvement_iter = iteration
    
    return {
        "packing": best_packing,
        "bin_weights": best_bin_wts
    }