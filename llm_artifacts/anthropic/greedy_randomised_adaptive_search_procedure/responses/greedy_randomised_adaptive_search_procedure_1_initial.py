import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return time_limit - elapsed()
    
    # Greedy construction with RCL
    def construct_solution(alpha):
        # Create a list of item indices sorted by weight descending
        remaining = list(range(n))
        # Randomized greedy: process items using RCL based on weight
        bins = []  # list of lists of item indices
        bin_remaining = []  # remaining capacity for each bin
        
        while remaining:
            # Build RCL based on item weights (prefer heavier items)
            w_max = max(weights[i] for i in remaining)
            w_min = min(weights[i] for i in remaining)
            threshold = w_max - alpha * (w_max - w_min)
            rcl = [i for i in remaining if weights[i] >= threshold]
            
            # Pick a random item from RCL
            item = random.choice(rcl)
            remaining.remove(item)
            
            # Try to place in existing bin - best fit
            best_bin = -1
            best_remaining = bin_capacity + 1
            
            for b in range(len(bins)):
                if bin_remaining[b] >= weights[item]:
                    if bin_remaining[b] - weights[item] < best_remaining:
                        best_remaining = bin_remaining[b] - weights[item]
                        best_bin = b
            
            if best_bin >= 0:
                bins[best_bin].append(item)
                bin_remaining[best_bin] -= weights[item]
            else:
                bins.append([item])
                bin_remaining.append(bin_capacity - weights[item])
        
        return bins, bin_remaining
    
    def construct_solution_v2(alpha):
        """Alternative construction: randomized FFD with randomized bin selection."""
        # Sort items by weight descending, then apply RCL-based ordering
        sorted_items = sorted(range(n), key=lambda i: -weights[i])
        
        # Introduce randomness by shuffling within groups of similar weight
        # Using RCL on the ordering: process items roughly in decreasing weight
        # but with some randomization
        order = []
        remaining = list(sorted_items)
        
        while remaining:
            w_max = weights[remaining[0]]  # already sorted descending
            w_min = weights[remaining[-1]]
            threshold = w_max - alpha * (w_max - w_min)
            rcl = [i for i in remaining if weights[i] >= threshold]
            item = random.choice(rcl)
            order.append(item)
            remaining.remove(item)
        
        bins = []
        bin_remaining = []
        
        for item in order:
            # Best fit
            best_bin = -1
            best_rem = bin_capacity + 1
            for b in range(len(bins)):
                if bin_remaining[b] >= weights[item]:
                    rem = bin_remaining[b] - weights[item]
                    if rem < best_rem:
                        best_rem = rem
                        best_bin = b
            if best_bin >= 0:
                bins[best_bin].append(item)
                bin_remaining[best_bin] -= weights[item]
            else:
                bins.append([item])
                bin_remaining.append(bin_capacity - weights[item])
        
        return bins, bin_remaining
    
    def local_search(bins, bin_remaining, deadline):
        """Local search to improve the solution by reducing number of bins."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            num_bins = len(bins)
            
            # Sort bins by load (ascending) - try to empty least loaded bins
            bin_order = sorted(range(num_bins), key=lambda b: sum(weights[i] for i in bins[b]))
            
            for src_idx in bin_order:
                if time.time() >= deadline:
                    break
                if not bins[src_idx]:  # already empty
                    continue
                
                src_items = bins[src_idx][:]
                # Try to move all items from src to other bins
                can_empty = True
                moves = []  # (item, target_bin)
                temp_remaining = bin_remaining[:]
                
                # Sort items in src by weight descending (place large items first)
                src_items_sorted = sorted(src_items, key=lambda i: -weights[i])
                
                for item in src_items_sorted:
                    # Find best fit in another bin
                    best_bin = -1
                    best_rem = bin_capacity + 1
                    for b in range(num_bins):
                        if b == src_idx:
                            continue
                        if temp_remaining[b] >= weights[item]:
                            rem = temp_remaining[b] - weights[item]
                            if rem < best_rem:
                                best_rem = rem
                                best_bin = b
                    if best_bin >= 0:
                        moves.append((item, best_bin))
                        temp_remaining[best_bin] -= weights[item]
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    # Apply moves
                    for item, target in moves:
                        bins[src_idx].remove(item)
                        bins[target].append(item)
                        bin_remaining[target] -= weights[item]
                    bin_remaining[src_idx] = bin_capacity
                    improved = True
            
            # Remove empty bins
            new_bins = []
            new_remaining = []
            for b in range(len(bins)):
                if bins[b]:
                    new_bins.append(bins[b])
                    new_remaining.append(bin_remaining[b])
            bins = new_bins
            bin_remaining = new_remaining
            
            if not improved:
                # Try single item moves
                for src in range(len(bins)):
                    if time.time() >= deadline:
                        break
                    for item in bins[src][:]:
                        best_bin = -1
                        best_rem = bin_remaining[src]  # only move if it improves (tighter fit)
                        # Actually we want to move to a tighter bin to potentially free up src
                        for dst in range(len(bins)):
                            if dst == src:
                                continue
                            if bin_remaining[dst] >= weights[item]:
                                rem = bin_remaining[dst] - weights[item]
                                if rem < best_rem:
                                    best_rem = rem
                                    best_bin = dst
                        if best_bin >= 0 and best_rem < bin_remaining[src]:
                            bins[src].remove(item)
                            bin_remaining[src] += weights[item]
                            bins[best_bin].append(item)
                            bin_remaining[best_bin] -= weights[item]
                            improved = True
            
            if not improved:
                # Try swap moves: swap item i from bin a with item j from bin b
                # if it leads to better packing
                for a in range(len(bins)):
                    if time.time() >= deadline:
                        break
                    if improved:
                        break
                    for b in range(a + 1, len(bins)):
                        if time.time() >= deadline:
                            break
                        if improved:
                            break
                        for i_idx, item_a in enumerate(bins[a]):
                            if improved:
                                break
                            for j_idx, item_b in enumerate(bins[b]):
                                diff = weights[item_a] - weights[item_b]
                                # Check if swap is feasible
                                if diff > 0:
                                    if bin_remaining[b] >= diff:
                                        # Swap reduces remaining in b, increases in a
                                        # Only do if it helps (e.g., might allow future moves)
                                        new_rem_a = bin_remaining[a] + diff
                                        new_rem_b = bin_remaining[b] - diff
                                        # Accept if it makes bin utilization more balanced
                                        # or specifically if it helps empty a bin
                                        pass
                                elif diff < 0:
                                    if bin_remaining[a] >= -diff:
                                        pass
                                # Skip swaps for now in basic local search
                                # They're expensive and the bin-emptying heuristic is more effective
                                break
        
        # Final cleanup: remove empty bins
        final_bins = []
        final_remaining = []
        for b in range(len(bins)):
            if bins[b]:
                final_bins.append(bins[b])
                final_remaining.append(bin_remaining[b])
        
        return final_bins, final_remaining
    
    def local_search_advanced(bins, bin_remaining, deadline):
        """More advanced local search with swaps to enable bin emptying."""
        improved = True
        iteration = 0
        while improved and time.time() < deadline:
            improved = False
            iteration += 1
            num_bins = len(bins)
            
            # Try to empty each bin (sorted by ascending load)
            loads = [sum(weights[i] for i in bins[b]) for b in range(num_bins)]
            bin_order = sorted(range(num_bins), key=lambda b: loads[b])
            
            for src_idx in bin_order:
                if time.time() >= deadline:
                    break
                if not bins[src_idx]:
                    continue
                
                src_load = loads[src_idx]
                # Quick check: is there enough total remaining capacity elsewhere?
                total_remaining_others = sum(bin_remaining[b] for b in range(num_bins) if b != src_idx)
                if total_remaining_others < src_load:
                    continue
                
                src_items = sorted(bins[src_idx], key=lambda i: -weights[i])
                can_empty = True
                moves = []
                temp_remaining = bin_remaining[:]
                
                for item in src_items:
                    best_bin = -1
                    best_rem = bin_capacity + 1
                    for b in range(num_bins):
                        if b == src_idx:
                            continue
                        if not bins[b] and not moves:  # don't open new empty bins
                            continue
                        if temp_remaining[b] >= weights[item]:
                            rem = temp_remaining[b] - weights[item]
                            if rem < best_rem:
                                best_rem = rem
                                best_bin = b
                    if best_bin >= 0:
                        moves.append((item, best_bin))
                        temp_remaining[best_bin] -= weights[item]
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    for item, target in moves:
                        bins[src_idx].remove(item)
                        bins[target].append(item)
                        bin_remaining[target] -= weights[item]
                    bin_remaining[src_idx] = bin_capacity
                    loads[src_idx] = 0
                    for item, target in moves:
                        loads[target] += weights[item]
                    improved = True
            
            # Remove empty bins
            new_bins = [b for b in bins if b]
            new_remaining = [bin_remaining[i] for i in range(len(bins)) if bins[i]]
            bins = new_bins
            bin_remaining = new_remaining
            
            if not improved:
                # Try 1-1 swaps that reduce the number of bins
                # For each pair of bins, try swapping items to free up space
                # then try emptying bins again
                num_bins = len(bins)
                
                # Try to find swaps that could help
                for a in range(num_bins):
                    if time.time() >= deadline:
                        break
                    if improved:
                        break
                    for b in range(a + 1, num_bins):
                        if time.time() >= deadline:
                            break
                        if improved:
                            break
                        for item_a in bins[a]:
                            if improved:
                                break
                            for item_b in bins[b]:
                                diff = weights[item_a] - weights[item_b]
                                new_rem_a = bin_remaining[a] + diff
                                new_rem_b = bin_remaining[b] - diff
                                if new_rem_a >= 0 and new_rem_b >= 0:
                                    if new_rem_a != bin_remaining[a]:  # actual change
                                        # Check if this swap makes some bin more empty
                                        # Simple criterion: accept if it increases max remaining
                                        old_max_rem = max(bin_remaining[a], bin_remaining[b])
                                        new_max_rem = max(new_rem_a, new_rem_b)
                                        if new_max_rem > old_max_rem:
                                            # Perform swap
                                            bins[a].remove(item_a)
                                            bins[b].remove(item_b)
                                            bins[a].append(item_b)
                                            bins[b].append(item_a)
                                            bin_remaining[a] = new_rem_a
                                            bin_remaining[b] = new_rem_b
                                            improved = True
                                            break
        
        final_bins = [b for b in bins if b]
        final_remaining = [bin_remaining[i] for i in range(len(bins)) if bins[i]]
        return final_bins, final_remaining
    
    # Initial deterministic FFD solution as baseline
    sorted_items = sorted(range(n), key=lambda i: -weights[i])
    best_bins = []
    best_remaining = []
    for item in sorted_items:
        placed = False
        best_b = -1
        best_r = bin_capacity + 1
        for b in range(len(best_bins)):
            if best_remaining[b] >= weights[item]:
                r = best_remaining[b] - weights[item]
                if r < best_r:
                    best_r = r
                    best_b = b
        if best_b >= 0:
            best_bins[best_b].append(item)
            best_remaining[best_b] -= weights[item]
        else:
            best_bins.append([item])
            best_remaining.append(bin_capacity - weights[item])
    
    best_num_bins = len(best_bins)
    
    # Apply local search to FFD solution
    ls_deadline = min(start_time + time_limit * 0.15, start_time + time_limit - 0.05)
    best_bins, best_remaining = local_search_advanced(best_bins, best_remaining, ls_deadline)
    best_num_bins = len(best_bins)
    
    # GRASP iterations
    max_iterations = 10000
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for iteration in range(max_iterations):
        if time.time() - start_time >= time_limit - 0.05:
            break
        
        # Choose alpha
        alpha = random.choice(alpha_values)
        
        # Construction phase
        if random.random() < 0.5:
            bins, bin_rem = construct_solution(alpha)
        else:
            bins, bin_rem = construct_solution_v2(alpha)
        
        # Local search phase (essential part of GRASP)
        # Allocate time for local search
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.05:
            # Just check if constructed solution is already better
            if len(bins) < best_num_bins:
                best_bins = bins
                best_remaining = bin_rem
                best_num_bins = len(bins)
            break
        
        # Give local search a portion of remaining time
        ls_time = min(remaining_time * 0.3, 0.5)  # at most 0.5 seconds per LS
        ls_deadline = time.time() + ls_time
        
        bins, bin_rem = local_search_advanced(bins, bin_rem, ls_deadline)
        
        # Update best
        if len(bins) < best_num_bins:
            best_bins = [b[:] for b in bins]
            best_remaining = bin_rem[:]
            best_num_bins = len(bins)
    
    # Compute bin weights
    packing = best_bins
    bin_weights = [sum(weights[i] for i in b) for b in packing]
    
    return {"packing": packing, "bin_weights": bin_weights}