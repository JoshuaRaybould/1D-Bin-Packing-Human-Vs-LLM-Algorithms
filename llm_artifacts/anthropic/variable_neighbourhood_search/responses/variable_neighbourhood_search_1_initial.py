import random
import time
import copy

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # --- Initial solution: First Fit Decreasing ---
    def first_fit_decreasing(items_indices, cap):
        sorted_items = sorted(items_indices, key=lambda i: weights[i], reverse=True)
        bins = []  # list of lists of item indices
        bin_rem = []  # remaining capacity
        for idx in sorted_items:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if bin_rem[b] >= w:
                    bins[b].append(idx)
                    bin_rem[b] -= w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_rem.append(cap - w)
        return bins, bin_rem
    
    def best_fit_decreasing(items_indices, cap):
        sorted_items = sorted(items_indices, key=lambda i: weights[i], reverse=True)
        bins = []
        bin_rem = []
        for idx in sorted_items:
            w = weights[idx]
            best_b = -1
            best_rem = cap + 1
            for b in range(len(bins)):
                if bin_rem[b] >= w and bin_rem[b] < best_rem:
                    best_rem = bin_rem[b]
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(idx)
                bin_rem[best_b] -= w
            else:
                bins.append([idx])
                bin_rem.append(cap - w)
        return bins, bin_rem
    
    # Build initial solution
    all_items = list(range(n))
    bins, bin_rem = best_fit_decreasing(all_items, bin_capacity)
    
    def make_item_to_bin(bins):
        itb = [0] * n
        for b in range(len(bins)):
            for idx in bins[b]:
                itb[idx] = b
        return itb
    
    def solution_cost(bins):
        return len(bins)
    
    def copy_solution(bins, bin_rem):
        return [lst[:] for lst in bins], bin_rem[:]
    
    def validate_and_clean(bins, bin_rem):
        # Remove empty bins
        new_bins = []
        new_rem = []
        for b in range(len(bins)):
            if len(bins[b]) > 0:
                new_bins.append(bins[b])
                new_rem.append(bin_rem[b])
        return new_bins, new_rem
    
    bins, bin_rem = validate_and_clean(bins, bin_rem)
    
    best_bins, best_rem = copy_solution(bins, bin_rem)
    best_cost = solution_cost(best_bins)
    
    # --- Local Search: Try to empty bins ---
    def local_search(bins, bin_rem):
        improved = True
        while improved:
            improved = False
            num_bins = len(bins)
            if num_bins <= 1:
                break
            
            # Sort bins by total weight (ascending) - try to empty least loaded first
            bin_loads = [(bin_capacity - bin_rem[b], b) for b in range(num_bins)]
            bin_loads.sort()
            
            for load, src_b in bin_loads:
                if src_b >= len(bins) or len(bins[src_b]) == 0:
                    continue
                # Try to move all items from src_b to other bins
                items_to_move = bins[src_b][:]
                # Sort items by weight descending for better packing
                items_to_move.sort(key=lambda i: weights[i], reverse=True)
                
                # Check if all items can fit somewhere
                # Use a greedy assignment: for each item, find the best-fit bin
                temp_rem = bin_rem[:]
                assignments = {}  # item -> target bin
                can_empty = True
                
                for idx in items_to_move:
                    w = weights[idx]
                    best_b = -1
                    best_r = bin_capacity + 1
                    for b in range(len(bins)):
                        if b == src_b:
                            continue
                        if temp_rem[b] >= w and temp_rem[b] < best_r:
                            best_r = temp_rem[b]
                            best_b = b
                    if best_b >= 0:
                        assignments[idx] = best_b
                        temp_rem[best_b] -= w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    # Apply the moves
                    for idx, tgt_b in assignments.items():
                        bins[tgt_b].append(idx)
                        bin_rem[tgt_b] -= weights[idx]
                    bins[src_b] = []
                    bin_rem[src_b] = bin_capacity
                    # Clean up
                    bins, bin_rem = validate_and_clean(bins, bin_rem)
                    improved = True
                    break
            
            if time.time() - start_time > time_limit * 0.95:
                break
        
        return bins, bin_rem
    
    # Apply local search to initial solution
    bins, bin_rem = local_search(bins, bin_rem)
    bins, bin_rem = validate_and_clean(bins, bin_rem)
    cost = solution_cost(bins)
    if cost < best_cost:
        best_bins, best_rem = copy_solution(bins, bin_rem)
        best_cost = cost
    
    # --- VNS ---
    k_max = 5
    
    def shaking(bins, bin_rem, k):
        """Perturb the solution in neighbourhood k."""
        bins = [lst[:] for lst in bins]
        bin_rem = bin_rem[:]
        num_bins = len(bins)
        
        if num_bins <= 1:
            return bins, bin_rem
        
        if k == 1:
            # Move one random item to a random different bin (even if it means creating a new bin)
            src_b = random.randint(0, num_bins - 1)
            if len(bins[src_b]) == 0:
                return bins, bin_rem
            item_idx_in_bin = random.randint(0, len(bins[src_b]) - 1)
            item = bins[src_b][item_idx_in_bin]
            w = weights[item]
            
            # Try to find a feasible different bin
            candidates = [b for b in range(num_bins) if b != src_b and bin_rem[b] >= w]
            if candidates:
                tgt_b = random.choice(candidates)
                bins[src_b].pop(item_idx_in_bin)
                bin_rem[src_b] += w
                bins[tgt_b].append(item)
                bin_rem[tgt_b] -= w
            
        elif k == 2:
            # Swap two items between two bins
            b1 = random.randint(0, num_bins - 1)
            b2 = random.randint(0, num_bins - 1)
            if b1 == b2 or len(bins[b1]) == 0 or len(bins[b2]) == 0:
                return bins, bin_rem
            i1_pos = random.randint(0, len(bins[b1]) - 1)
            i2_pos = random.randint(0, len(bins[b2]) - 1)
            item1 = bins[b1][i1_pos]
            item2 = bins[b2][i2_pos]
            w1, w2 = weights[item1], weights[item2]
            diff = w1 - w2
            if bin_rem[b1] + diff >= 0 and bin_rem[b2] - diff >= 0:
                bins[b1][i1_pos] = item2
                bins[b2][i2_pos] = item1
                bin_rem[b1] += diff
                bin_rem[b2] -= diff
            
        elif k == 3:
            # Move 2-3 random items to random feasible bins
            num_moves = random.randint(2, 3)
            for _ in range(num_moves):
                if len(bins) <= 1:
                    break
                src_b = random.randint(0, len(bins) - 1)
                if len(bins[src_b]) == 0:
                    continue
                item_pos = random.randint(0, len(bins[src_b]) - 1)
                item = bins[src_b][item_pos]
                w = weights[item]
                candidates = [b for b in range(len(bins)) if b != src_b and bin_rem[b] >= w]
                if candidates:
                    tgt_b = random.choice(candidates)
                    bins[src_b].pop(item_pos)
                    bin_rem[src_b] += w
                    bins[tgt_b].append(item)
                    bin_rem[tgt_b] -= w
            
        elif k == 4:
            # Destroy and repair: empty 1-2 random bins, repack their items using BFD into remaining bins
            num_to_empty = min(random.randint(1, 2), num_bins - 1)
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                for item in bins[b]:
                    freed_items.append(item)
                    bin_rem[b] += weights[item]
                bins[b] = []
            
            # Shuffle freed items for randomness, then try best-fit
            random.shuffle(freed_items)
            freed_items.sort(key=lambda i: weights[i], reverse=True)
            for idx in freed_items:
                w = weights[idx]
                best_b = -1
                best_r = bin_capacity + 1
                for b in range(len(bins)):
                    if bin_rem[b] >= w and bin_rem[b] < best_r:
                        best_r = bin_rem[b]
                        best_b = b
                if best_b >= 0:
                    bins[best_b].append(idx)
                    bin_rem[best_b] -= w
                else:
                    bins.append([idx])
                    bin_rem.append(bin_capacity - w)
            
        elif k == 5:
            # Larger perturbation: empty 3-5 bins and repack
            num_to_empty = min(random.randint(3, 5), num_bins - 1)
            if num_to_empty <= 0:
                return bins, bin_rem
            bins_to_empty = random.sample(range(num_bins), num_to_empty)
            freed_items = []
            for b in bins_to_empty:
                for item in bins[b]:
                    freed_items.append(item)
                    bin_rem[b] += weights[item]
                bins[b] = []
            
            # Random order then best-fit
            random.shuffle(freed_items)
            freed_items.sort(key=lambda i: weights[i], reverse=True)
            for idx in freed_items:
                w = weights[idx]
                best_b = -1
                best_r = bin_capacity + 1
                for b in range(len(bins)):
                    if bin_rem[b] >= w and bin_rem[b] < best_r:
                        best_r = bin_rem[b]
                        best_b = b
                if best_b >= 0:
                    bins[best_b].append(idx)
                    bin_rem[best_b] -= w
                else:
                    bins.append([idx])
                    bin_rem.append(bin_capacity - w)
        
        bins, bin_rem = validate_and_clean(bins, bin_rem)
        return bins, bin_rem
    
    # Enhanced local search: try moving items between bins to free up bins
    def local_search_enhanced(bins, bin_rem):
        # First try the bin-emptying local search
        bins, bin_rem = local_search(bins, bin_rem)
        
        # Then try pair-wise item moves that reduce waste in target bins
        # This helps set up future bin emptying
        num_bins = len(bins)
        if num_bins <= 1:
            return bins, bin_rem
        
        # Try to move items to make bins more full
        improved = True
        max_iters = 3
        iteration = 0
        while improved and iteration < max_iters:
            improved = False
            iteration += 1
            
            # For each pair of bins, try swapping items that reduce total waste variance
            # Actually, let's try: for each item in a bin, see if moving it to another bin
            # and then checking if the source bin can be emptied
            
            # Sort bins by load ascending
            bin_order = sorted(range(len(bins)), key=lambda b: bin_capacity - bin_rem[b])
            
            for src_b in bin_order:
                if src_b >= len(bins) or len(bins[src_b]) == 0:
                    continue
                items = bins[src_b][:]
                items.sort(key=lambda i: weights[i], reverse=True)
                
                # Try to redistribute all items
                temp_rem = bin_rem[:]
                assignments = {}
                can_empty = True
                
                for idx in items:
                    w = weights[idx]
                    best_b = -1
                    best_r = bin_capacity + 1
                    for b in range(len(bins)):
                        if b == src_b:
                            continue
                        if temp_rem[b] >= w and temp_rem[b] < best_r:
                            best_r = temp_rem[b]
                            best_b = b
                    if best_b >= 0:
                        assignments[idx] = best_b
                        temp_rem[best_b] -= w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    for idx, tgt_b in assignments.items():
                        bins[tgt_b].append(idx)
                        bin_rem[tgt_b] -= weights[idx]
                    bins[src_b] = []
                    bin_rem[src_b] = bin_capacity
                    bins, bin_rem = validate_and_clean(bins, bin_rem)
                    improved = True
                    break
            
            if time.time() - start_time > time_limit * 0.95:
                break
        
        return bins, bin_rem
    
    # Main VNS loop
    k = 1
    max_iterations = 100000
    iteration = 0
    no_improve_count = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        if time.time() - start_time > time_limit * 0.90:
            break
        
        # Shaking
        new_bins, new_rem = shaking(bins, bin_rem, k)
        
        # Local search
        new_bins, new_rem = local_search_enhanced(new_bins, new_rem)
        new_bins, new_rem = validate_and_clean(new_bins, new_rem)
        new_cost = solution_cost(new_bins)
        
        # Move or not
        if new_cost <= solution_cost(bins):
            bins, bin_rem = new_bins, new_rem
            k = 1
            no_improve_count = 0
            
            if new_cost < best_cost:
                best_bins, best_rem = copy_solution(bins, bin_rem)
                best_cost = new_cost
        else:
            k += 1
            no_improve_count += 1
            if k > k_max:
                k = 1
        
        # If stuck for too long, do a larger restart-like perturbation
        if no_improve_count > k_max * 10:
            # Partial restart: take best solution and apply large perturbation
            bins, bin_rem = copy_solution(best_bins, best_rem)
            num_bins = len(bins)
            if num_bins > 2:
                # Empty a significant portion and repack
                num_to_empty = min(max(num_bins // 3, 2), num_bins - 1)
                # Pick the least loaded bins
                bin_order = sorted(range(num_bins), key=lambda b: bin_capacity - bin_rem[b])
                bins_to_empty = bin_order[:num_to_empty]
                freed_items = []
                for b in bins_to_empty:
                    for item in bins[b]:
                        freed_items.append(item)
                        bin_rem[b] += weights[item]
                    bins[b] = []
                
                random.shuffle(freed_items)
                freed_items.sort(key=lambda i: weights[i], reverse=True)
                for idx in freed_items:
                    w = weights[idx]
                    best_b = -1
                    best_r = bin_capacity + 1
                    for b in range(len(bins)):
                        if bin_rem[b] >= w and bin_rem[b] < best_r:
                            best_r = bin_rem[b]
                            best_b = b
                    if best_b >= 0:
                        bins[best_b].append(idx)
                        bin_rem[best_b] -= w
                    else:
                        bins.append([idx])
                        bin_rem.append(bin_capacity - w)
                
                bins, bin_rem = validate_and_clean(bins, bin_rem)
            
            k = 1
            no_improve_count = 0
    
    # Build output from best solution
    packing = best_bins
    bin_weights = []
    for b in range(len(packing)):
        bw = sum(weights[idx] for idx in packing[b])
        bin_weights.append(bw)
    
    return {"packing": packing, "bin_weights": bin_weights}