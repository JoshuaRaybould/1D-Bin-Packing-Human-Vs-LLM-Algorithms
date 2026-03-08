def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import random
    import time as time_module

    start_time = time_module.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Handle single item
    if n == 1:
        return {"packing": [[0]], "bin_weights": [weights[0]]}

    C = bin_capacity
    W = weights

    # --- FFD initial solution ---
    indices_sorted = sorted(range(n), key=lambda i: W[i], reverse=True)
    bins = []       # list of list of item indices
    bin_loads = []   # load of each bin
    item_bin = [0] * n  # bin assignment for each item

    for idx in indices_sorted:
        w = W[idx]
        placed = False
        for b in range(len(bins)):
            if bin_loads[b] + w <= C:
                bins[b].append(idx)
                bin_loads[b] += w
                item_bin[idx] = b
                placed = True
                break
        if not placed:
            item_bin[idx] = len(bins)
            bins.append([idx])
            bin_loads.append(w)

    # Remove empty bins and reindex
    def compact():
        nonlocal bins, bin_loads, item_bin
        new_bins = []
        new_loads = []
        mapping = {}
        for i, b in enumerate(bins):
            if len(b) > 0:
                mapping[i] = len(new_bins)
                new_bins.append(b)
                new_loads.append(bin_loads[i])
        bins = new_bins
        bin_loads = new_loads
        for item in range(n):
            item_bin[item] = mapping[item_bin[item]]

    compact()

    # Compute cost: we want to minimize bins, and as secondary maximize sum of squares
    # Cost = num_bins * C*C - sum(load^2)
    # Lower cost = better
    def compute_cost():
        nb = len(bins)
        ss = sum(l * l for l in bin_loads)
        return nb * C * C - ss

    current_cost = compute_cost()
    best_bins = [b[:] for b in bins]
    best_loads = bin_loads[:]
    best_num_bins = len(bins)
    best_cost = current_cost

    # SA parameters
    elapsed = time_module.time() - start_time
    remaining = time_limit - elapsed
    if remaining <= 0.01:
        packing = [b[:] for b in best_bins]
        return {"packing": packing, "bin_weights": best_loads[:]}

    # Temperature schedule
    T_start = max(C * C * 0.5, 1.0)
    T_end = 0.1
    
    # Pre-compute some things
    C2 = C * C
    
    iteration = 0
    check_interval = 500
    
    # We'll run SA
    deadline = start_time + time_limit * 0.98  # leave some buffer
    
    # Estimate iterations based on time
    # Do a small calibration
    cal_start = time_module.time()
    for _ in range(1000):
        pass
    cal_time = time_module.time() - cal_start
    # Rough estimate: each SA iteration is ~10x a pass iteration
    # We'll just set max_iterations and check time
    
    total_remaining = deadline - time_module.time()
    if total_remaining <= 0:
        packing = [b[:] for b in best_bins]
        return {"packing": packing, "bin_weights": best_loads[:]}

    import math
    log = math.log
    exp = math.exp
    randint = random.randint
    rand = random.random
    
    alpha = 0.99999  # Will be adjusted
    T = T_start
    
    # Adaptive: compute cooling rate based on expected iterations
    # Estimate ~50000 iterations per second for moderate n
    est_iters = max(int(total_remaining * 30000), 1000)
    if est_iters > 1:
        alpha = (T_end / T_start) ** (1.0 / est_iters)
    else:
        alpha = 0.999
    
    no_improve_count = 0
    
    while True:
        iteration += 1
        
        if iteration % check_interval == 0:
            now = time_module.time()
            if now >= deadline:
                break
            # Adjust temperature based on progress
            progress = (now - start_time) / (deadline - start_time)
            T = T_start * ((T_end / T_start) ** progress)
        else:
            T *= alpha
            if T < T_end:
                T = T_end
        
        num_bins = len(bins)
        
        # Choose move type
        # 0: move item to different bin
        # 1: swap two items
        # 2: move item from smallest bin
        r = rand()
        
        if r < 0.5:
            # Move a random item to a random different bin
            item = randint(0, n - 1)
            src = item_bin[item]
            w = W[item]
            
            if num_bins <= 1:
                continue
            
            # Pick destination
            dst = randint(0, num_bins - 1)
            if dst == src:
                dst = (dst + 1) % num_bins
            
            # Check feasibility
            if bin_loads[dst] + w > C:
                continue
            
            # Compute cost delta
            old_src_load = bin_loads[src]
            old_dst_load = bin_loads[dst]
            new_src_load = old_src_load - w
            new_dst_load = old_dst_load + w
            
            # Cost change from loads
            delta = -(new_src_load * new_src_load - old_src_load * old_src_load) - (new_dst_load * new_dst_load - old_dst_load * old_dst_load)
            
            bin_removed = False
            if new_src_load == 0 and len(bins[src]) == 1:
                # This bin will become empty -> one fewer bin
                delta -= C2  # removing a bin reduces cost by C^2
                bin_removed = True
            
            if delta <= 0 or rand() < exp(-delta / T):
                # Accept move
                bins[src].remove(item)
                bins[dst].append(item)
                bin_loads[src] = new_src_load
                bin_loads[dst] = new_dst_load
                item_bin[item] = dst
                current_cost += delta
                
                if bin_removed:
                    # Remove empty bin and reindex
                    removed_idx = src
                    last_idx = len(bins) - 1
                    if removed_idx != last_idx:
                        # Swap with last bin
                        bins[removed_idx] = bins[last_idx]
                        bin_loads[removed_idx] = bin_loads[last_idx]
                        for it in bins[removed_idx]:
                            item_bin[it] = removed_idx
                    bins.pop()
                    bin_loads.pop()
                
                if len(bins) < best_num_bins or (len(bins) == best_num_bins and current_cost < best_cost):
                    best_bins = [b[:] for b in bins]
                    best_loads = bin_loads[:]
                    best_num_bins = len(bins)
                    best_cost = current_cost
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        elif r < 0.85:
            # Swap two items from different bins
            if num_bins <= 1:
                continue
            
            item1 = randint(0, n - 1)
            item2 = randint(0, n - 1)
            if item1 == item2:
                continue
            
            b1 = item_bin[item1]
            b2 = item_bin[item2]
            if b1 == b2:
                continue
            
            w1 = W[item1]
            w2 = W[item2]
            
            if w1 == w2:
                continue  # No effect
            
            # Check feasibility
            new_load_b1 = bin_loads[b1] - w1 + w2
            new_load_b2 = bin_loads[b2] - w2 + w1
            
            if new_load_b1 > C or new_load_b2 > C:
                continue
            
            old_load_b1 = bin_loads[b1]
            old_load_b2 = bin_loads[b2]
            
            delta = -(new_load_b1 * new_load_b1 - old_load_b1 * old_load_b1) - (new_load_b2 * new_load_b2 - old_load_b2 * old_load_b2)
            
            if delta <= 0 or rand() < exp(-delta / T):
                bins[b1].remove(item1)
                bins[b1].append(item2)
                bins[b2].remove(item2)
                bins[b2].append(item1)
                bin_loads[b1] = new_load_b1
                bin_loads[b2] = new_load_b2
                item_bin[item1] = b2
                item_bin[item2] = b1
                current_cost += delta
                
                if len(bins) < best_num_bins or (len(bins) == best_num_bins and current_cost < best_cost):
                    best_bins = [b[:] for b in bins]
                    best_loads = bin_loads[:]
                    best_num_bins = len(bins)
                    best_cost = current_cost
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        else:
            # Try to move an item from the least loaded bin
            if num_bins <= 1:
                continue
            
            # Find the bin with smallest load
            min_load = bin_loads[0]
            min_bin = 0
            for b in range(1, num_bins):
                if bin_loads[b] < min_load:
                    min_load = bin_loads[b]
                    min_bin = b
            
            if len(bins[min_bin]) == 0:
                continue
            
            # Pick a random item from this bin
            item_idx_in_bin = randint(0, len(bins[min_bin]) - 1)
            item = bins[min_bin][item_idx_in_bin]
            w = W[item]
            src = min_bin
            
            # Find best destination (most full that still fits)
            best_dst = -1
            best_dst_load = -1
            for b in range(num_bins):
                if b == src:
                    continue
                if bin_loads[b] + w <= C and bin_loads[b] > best_dst_load:
                    best_dst = b
                    best_dst_load = bin_loads[b]
            
            if best_dst == -1:
                continue
            
            dst = best_dst
            old_src_load = bin_loads[src]
            old_dst_load = bin_loads[dst]
            new_src_load = old_src_load - w
            new_dst_load = old_dst_load + w
            
            delta = -(new_src_load * new_src_load - old_src_load * old_src_load) - (new_dst_load * new_dst_load - old_dst_load * old_dst_load)
            
            bin_removed = False
            if new_src_load == 0 and len(bins[src]) == 1:
                delta -= C2
                bin_removed = True
            
            if delta <= 0 or rand() < exp(-delta / T):
                bins[src].remove(item)
                bins[dst].append(item)
                bin_loads[src] = new_src_load
                bin_loads[dst] = new_dst_load
                item_bin[item] = dst
                current_cost += delta
                
                if bin_removed:
                    removed_idx = src
                    last_idx = len(bins) - 1
                    if removed_idx != last_idx:
                        bins[removed_idx] = bins[last_idx]
                        bin_loads[removed_idx] = bin_loads[last_idx]
                        for it in bins[removed_idx]:
                            item_bin[it] = removed_idx
                    bins.pop()
                    bin_loads.pop()
                
                if len(bins) < best_num_bins or (len(bins) == best_num_bins and current_cost < best_cost):
                    best_bins = [b[:] for b in bins]
                    best_loads = bin_loads[:]
                    best_num_bins = len(bins)
                    best_cost = current_cost
                    no_improve_count = 0
                else:
                    no_improve_count += 1
    
    # Return best solution
    return {"packing": best_bins, "bin_weights": best_loads}