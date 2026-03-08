import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Initial solution: First Fit Decreasing
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    bins = []  # list of lists of item indices
    bin_wts = []  # bin weights
    item_bin = [0] * n  # which bin each item is in
    
    for idx in sorted_indices:
        w = weights[idx]
        placed = False
        for b in range(len(bins)):
            if bin_wts[b] + w <= bin_capacity:
                bins[b].append(idx)
                bin_wts[b] += w
                item_bin[idx] = b
                placed = True
                break
        if not placed:
            item_bin[idx] = len(bins)
            bins.append([idx])
            bin_wts.append(w)
    
    # Fitness: sum of bin_weight^2 (maximize)
    def compute_fitness(bw):
        return sum(x * x for x in bw)
    
    current_fitness = compute_fitness(bin_wts)
    num_bins = len(bins)
    
    # Best solution tracking
    best_num_bins = num_bins
    best_fitness = current_fitness
    best_packing = [list(b) for b in bins]
    best_bin_wts = list(bin_wts)
    
    # Convert bins to sets for faster operations
    bin_sets = [set(b) for b in bins]
    
    # SA parameters
    # We'll use a geometric cooling schedule
    T_start = bin_capacity * bin_capacity * 0.5
    T_min = 1.0
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining <= 0.05:
        return {
            "packing": best_packing,
            "bin_weights": best_bin_wts
        }
    
    # Estimate iterations
    # We'll do iterations in batches and check time
    alpha = 0.99995
    T = T_start
    
    iteration = 0
    check_interval = 500
    
    while True:
        if iteration % check_interval == 0:
            elapsed = time.time() - start_time
            if elapsed >= time_limit - 0.02:
                break
            # Adaptive: adjust temperature based on remaining time
            # fraction of time remaining
            frac_remaining = max(0, (time_limit - elapsed) / time_limit)
            # We want T to reach T_min at the end
            # Recompute alpha periodically isn't needed if we just cool
        
        iteration += 1
        num_bins = len(bins)
        
        if num_bins <= 1:
            # Can't improve further
            break
        
        # Choose move type
        # 70% move, 30% swap
        r = random.random()
        
        if r < 0.7:
            # MOVE: pick a random item, move to a different bin
            item = random.randrange(n)
            w = weights[item]
            src = item_bin[item]
            
            # Pick target bin (different from src)
            # With some probability, pick the bin with most weight that can still fit
            tgt = random.randrange(num_bins - 1)
            if tgt >= src:
                tgt += 1
            
            # Check feasibility
            if bin_wts[tgt] + w > bin_capacity:
                T *= alpha
                continue
            
            # Compute fitness delta
            old_src_w = bin_wts[src]
            old_tgt_w = bin_wts[tgt]
            new_src_w = old_src_w - w
            new_tgt_w = old_tgt_w + w
            
            delta = (new_src_w * new_src_w + new_tgt_w * new_tgt_w) - \
                    (old_src_w * old_src_w + old_tgt_w * old_tgt_w)
            
            # If src becomes empty, that's a bonus (one fewer bin)
            src_becomes_empty = (new_src_w == 0)
            
            # Accept?
            accept = False
            if delta > 0 or src_becomes_empty:
                accept = True
            elif T > 0:
                # delta <= 0, accept with probability
                try:
                    prob = 2.718281828 ** (delta / T)
                    if random.random() < prob:
                        accept = True
                except (OverflowError, ZeroDivisionError):
                    pass
            
            if accept:
                # Execute move
                bin_sets[src].remove(item)
                bin_sets[tgt].add(item)
                bin_wts[src] = new_src_w
                bin_wts[tgt] = new_tgt_w
                item_bin[item] = tgt
                current_fitness += delta
                
                # If src is empty, remove it
                if src_becomes_empty:
                    # Remove bin src
                    last = num_bins - 1
                    if src != last:
                        # Swap with last bin
                        bin_sets[src] = bin_sets[last]
                        bin_wts[src] = bin_wts[last]
                        # Update item_bin for items in the moved bin
                        for it in bin_sets[src]:
                            item_bin[it] = src
                    bin_sets.pop()
                    bin_wts.pop()
                    num_bins -= 1
                    # Recompute fitness since we removed a bin (0^2 contribution gone)
                    # Actually delta already accounts for new_src_w=0, so fitness is correct
                    # But we removed the bin entry, so current_fitness should remain the same
                    # since 0^2 = 0 contributes nothing
                
                # Check if this is best
                cur_num_bins = len(bin_sets)
                if cur_num_bins < best_num_bins or \
                   (cur_num_bins == best_num_bins and current_fitness > best_fitness):
                    best_num_bins = cur_num_bins
                    best_fitness = current_fitness
                    best_packing = [list(b) for b in bin_sets]
                    best_bin_wts = list(bin_wts)
        
        else:
            # SWAP: pick two items from different bins, swap them
            item1 = random.randrange(n)
            item2 = random.randrange(n)
            
            b1 = item_bin[item1]
            b2 = item_bin[item2]
            
            if b1 == b2:
                T *= alpha
                continue
            
            w1 = weights[item1]
            w2 = weights[item2]
            
            if w1 == w2:
                T *= alpha
                continue
            
            # Check feasibility
            new_b1_w = bin_wts[b1] - w1 + w2
            new_b2_w = bin_wts[b2] - w2 + w1
            
            if new_b1_w > bin_capacity or new_b2_w > bin_capacity:
                T *= alpha
                continue
            
            if new_b1_w < 0 or new_b2_w < 0:
                T *= alpha
                continue
            
            # Compute fitness delta
            old_contrib = bin_wts[b1] * bin_wts[b1] + bin_wts[b2] * bin_wts[b2]
            new_contrib = new_b1_w * new_b1_w + new_b2_w * new_b2_w
            delta = new_contrib - old_contrib
            
            accept = False
            if delta > 0:
                accept = True
            elif delta == 0:
                T *= alpha
                continue
            else:
                if T > 0:
                    try:
                        prob = 2.718281828 ** (delta / T)
                        if random.random() < prob:
                            accept = True
                    except (OverflowError, ZeroDivisionError):
                        pass
            
            if accept:
                bin_sets[b1].remove(item1)
                bin_sets[b1].add(item2)
                bin_sets[b2].remove(item2)
                bin_sets[b2].add(item1)
                bin_wts[b1] = new_b1_w
                bin_wts[b2] = new_b2_w
                item_bin[item1] = b2
                item_bin[item2] = b1
                current_fitness += delta
                
                cur_num_bins = len(bin_sets)
                if cur_num_bins < best_num_bins or \
                   (cur_num_bins == best_num_bins and current_fitness > best_fitness):
                    best_num_bins = cur_num_bins
                    best_fitness = current_fitness
                    best_packing = [list(b) for b in bin_sets]
                    best_bin_wts = list(bin_wts)
        
        # Cool
        T *= alpha
        if T < T_min:
            # Reheat
            T = T_start * 0.3
    
    return {
        "packing": best_packing,
        "bin_weights": best_bin_wts
    }