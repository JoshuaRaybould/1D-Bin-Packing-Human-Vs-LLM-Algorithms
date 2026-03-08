import random
import time

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
        placed = False
        for b in range(len(bins_list)):
            if bin_wts[b] + w <= bin_capacity:
                bins_list[b].add(idx)
                bin_wts[b] += w
                item_bin[idx] = b
                placed = True
                break
        if not placed:
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
    
    # SA parameters
    T_start = bin_capacity * bin_capacity * 0.3
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining <= 0.05:
        return {
            "packing": best_packing,
            "bin_weights": best_bin_wts
        }
    
    # Local variable caching for hot loop
    _random = random.random
    _randrange = random.randrange
    _time = time.time
    _weights = weights
    _bin_capacity = bin_capacity
    _exp = 2.718281828459045
    
    iteration = 0
    check_interval = 2000
    last_restart_iter = 0
    last_best_save_iter = 0
    
    # Find lightest bin
    def find_lightest_bin():
        min_w = bin_wts[0]
        min_idx = 0
        for i in range(1, len(bin_wts)):
            if bin_wts[i] < min_w:
                min_w = bin_wts[i]
                min_idx = i
        return min_idx
    
    lightest_bin = find_lightest_bin()
    
    # Compute initial T based on time
    T = T_start
    
    while True:
        if iteration % check_interval == 0:
            elapsed = _time() - start_time
            if elapsed >= time_limit - 0.02:
                break
            # Time-based temperature
            fraction_remaining = max(0.0, (time_limit - elapsed) / time_limit)
            T = T_start * fraction_remaining
            
            # Periodic restart if diverged
            if num_bins > best_num_bins + 3 and iteration - last_restart_iter >= 5000:
                last_restart_iter = iteration
                bins_list = [set(b) for b in best_packing]
                bin_wts = list(best_bin_wts)
                num_bins = len(bins_list)
                for b_idx in range(num_bins):
                    for it in bins_list[b_idx]:
                        item_bin[it] = b_idx
                current_fitness = compute_fitness(bin_wts, num_bins)
                lightest_bin = find_lightest_bin()
        
        iteration += 1
        
        if num_bins <= 1:
            break
        
        # Periodic bin-emptying attempt
        if iteration % 10000 == 0:
            lb = lightest_bin
            if lb < num_bins:
                items_in_lb = sorted(bins_list[lb], key=lambda i: _weights[i], reverse=True)
                placements = []  # (item, target_bin)
                success = True
                temp_wts = list(bin_wts)
                for it in items_in_lb:
                    w = _weights[it]
                    best_b = -1
                    best_bw = -1
                    for b in range(num_bins):
                        if b == lb:
                            continue
                        if temp_wts[b] + w <= _bin_capacity and temp_wts[b] > best_bw:
                            best_bw = temp_wts[b]
                            best_b = b
                    if best_b >= 0:
                        placements.append((it, best_b))
                        temp_wts[best_b] += w
                    else:
                        success = False
                        break
                
                if success and len(placements) > 0:
                    # Apply all placements
                    for it, tgt in placements:
                        w = _weights[it]
                        bins_list[lb].discard(it)
                        bins_list[tgt].add(it)
                        bin_wts[tgt] += w
                        item_bin[it] = tgt
                    bin_wts[lb] = 0
                    # Remove empty bin
                    last = num_bins - 1
                    if lb != last:
                        bins_list[lb] = bins_list[last]
                        bin_wts[lb] = bin_wts[last]
                        for it in bins_list[lb]:
                            item_bin[it] = lb
                    bins_list.pop()
                    bin_wts.pop()
                    num_bins -= 1
                    current_fitness = compute_fitness(bin_wts, num_bins)
                    lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                    
                    if num_bins < best_num_bins or (num_bins == best_num_bins and current_fitness > best_fitness):
                        best_num_bins = num_bins
                        best_fitness = current_fitness
                        best_packing = [list(b) for b in bins_list]
                        best_bin_wts = list(bin_wts)
                    continue
        
        # Choose move type
        r = _random()
        
        if r < 0.40:
            # MOVE from lightest bin to random target
            lb = lightest_bin
            if lb >= num_bins or len(bins_list[lb]) == 0:
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                lb = lightest_bin
            
            # Pick random item from lightest bin
            lb_items = bins_list[lb]
            if len(lb_items) == 0:
                continue
            item = random.choice(list(lb_items))
            w = _weights[item]
            src = lb
            
            # Pick target bin
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
                delta += PENALTY  # removing a bin
            
            accept = False
            if delta >= 0 or src_becomes_empty:
                accept = True
            elif T > 0:
                try:
                    prob = _exp ** (delta / T)
                    if _random() < prob:
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
                    last = num_bins - 1
                    if src != last:
                        bins_list[src] = bins_list[last]
                        bin_wts[src] = bin_wts[last]
                        for it in bins_list[src]:
                            item_bin[it] = src
                    bins_list.pop()
                    bin_wts.pop()
                    num_bins -= 1
                
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                
                cur_num_bins = num_bins
                if cur_num_bins < best_num_bins:
                    best_num_bins = cur_num_bins
                    best_fitness = current_fitness
                    best_packing = [list(b) for b in bins_list]
                    best_bin_wts = list(bin_wts)
                elif cur_num_bins == best_num_bins and current_fitness > best_fitness:
                    if iteration - last_best_save_iter >= 1000:
                        last_best_save_iter = iteration
                        best_fitness = current_fitness
                        best_packing = [list(b) for b in bins_list]
                        best_bin_wts = list(bin_wts)
        
        elif r < 0.50:
            # MOVE from lightest bin to best-fit target (greedy)
            lb = lightest_bin
            if lb >= num_bins or len(bins_list[lb]) == 0:
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                lb = lightest_bin
            
            lb_items = bins_list[lb]
            if len(lb_items) == 0:
                continue
            item = random.choice(list(lb_items))
            w = _weights[item]
            src = lb
            
            # Find best-fit target (fullest bin where item fits)
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
            
            src_becomes_empty = (new_src_w == 0)
            
            # Always accept (greedy)
            bins_list[src].remove(item)
            bins_list[tgt].add(item)
            bin_wts[src] = new_src_w
            bin_wts[tgt] = new_tgt_w
            item_bin[item] = tgt
            
            delta = (new_src_w * new_src_w + new_tgt_w * new_tgt_w) - \
                    (old_src_w * old_src_w + old_tgt_w * old_tgt_w)
            if src_becomes_empty:
                delta += PENALTY
            current_fitness += delta
            
            if src_becomes_empty:
                last = num_bins - 1
                if src != last:
                    bins_list[src] = bins_list[last]
                    bin_wts[src] = bin_wts[last]
                    for it in bins_list[src]:
                        item_bin[it] = src
                bins_list.pop()
                bin_wts.pop()
                num_bins -= 1
            
            lightest_bin = find_lightest_bin() if num_bins > 0 else 0
            
            cur_num_bins = num_bins
            if cur_num_bins < best_num_bins:
                best_num_bins = cur_num_bins
                best_fitness = current_fitness
                best_packing = [list(b) for b in bins_list]
                best_bin_wts = list(bin_wts)
            elif cur_num_bins == best_num_bins and current_fitness > best_fitness:
                if iteration - last_best_save_iter >= 1000:
                    last_best_save_iter = iteration
                    best_fitness = current_fitness
                    best_packing = [list(b) for b in bins_list]
                    best_bin_wts = list(bin_wts)
        
        elif r < 0.75:
            # Standard random MOVE
            item = _randrange(n)
            w = _weights[item]
            src = item_bin[item]
            
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
                    prob = _exp ** (delta / T)
                    if _random() < prob:
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
                    last = num_bins - 1
                    if src != last:
                        bins_list[src] = bins_list[last]
                        bin_wts[src] = bin_wts[last]
                        for it in bins_list[src]:
                            item_bin[it] = src
                    bins_list.pop()
                    bin_wts.pop()
                    num_bins -= 1
                
                lightest_bin = find_lightest_bin() if num_bins > 0 else 0
                
                cur_num_bins = num_bins
                if cur_num_bins < best_num_bins:
                    best_num_bins = cur_num_bins
                    best_fitness = current_fitness
                    best_packing = [list(b) for b in bins_list]
                    best_bin_wts = list(bin_wts)
                elif cur_num_bins == best_num_bins and current_fitness > best_fitness:
                    if iteration - last_best_save_iter >= 1000:
                        last_best_save_iter = iteration
                        best_fitness = current_fitness
                        best_packing = [list(b) for b in bins_list]
                        best_bin_wts = list(bin_wts)
        
        else:
            # SWAP
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
                        prob = _exp ** (delta / T)
                        if _random() < prob:
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
                
                cur_num_bins = num_bins
                if cur_num_bins < best_num_bins:
                    best_num_bins = cur_num_bins
                    best_fitness = current_fitness
                    best_packing = [list(b) for b in bins_list]
                    best_bin_wts = list(bin_wts)
                elif cur_num_bins == best_num_bins and current_fitness > best_fitness:
                    if iteration - last_best_save_iter >= 1000:
                        last_best_save_iter = iteration
                        best_fitness = current_fitness
                        best_packing = [list(b) for b in bins_list]
                        best_bin_wts = list(bin_wts)
    
    return {
        "packing": best_packing,
        "bin_weights": best_bin_wts
    }