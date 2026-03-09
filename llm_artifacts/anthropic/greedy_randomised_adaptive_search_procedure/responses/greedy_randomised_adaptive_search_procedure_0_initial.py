import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Sort indices by weight descending for construction
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    best_packing = None
    best_num_bins = n + 1
    
    def elapsed():
        return time.time() - start_time
    
    def construct_solution(alpha):
        """Greedy randomized construction using Best Fit with RCL."""
        bins = []  # list of (items_list, remaining_capacity)
        
        for idx in sorted_indices:
            w = weights[idx]
            if w > bin_capacity:
                # Item can't fit in any bin, skip or put in own bin
                bins.append(([idx], bin_capacity - w))
                continue
            
            # Find all bins that can fit this item
            candidates = []
            for b_idx, (items, rem) in enumerate(bins):
                if rem >= w:
                    # Score: prefer bins with less remaining (best fit)
                    # Lower remaining after placing = better fit
                    candidates.append((b_idx, rem - w))  # remaining after placing
            
            if not candidates:
                # Open new bin
                bins.append(([idx], bin_capacity - w))
            else:
                # Build RCL based on best fit criterion
                # Best = minimum remaining after placing (tightest fit)
                # Worst = maximum remaining after placing
                best_val = min(c[1] for c in candidates)
                worst_val = max(c[1] for c in candidates)
                
                threshold = best_val + alpha * (worst_val - best_val)
                
                rcl = [c for c in candidates if c[1] <= threshold]
                
                if not rcl:
                    rcl = [candidates[0]]  # fallback
                
                chosen_b_idx, _ = random.choice(rcl)
                bins[chosen_b_idx][0].append(idx)
                bins[chosen_b_idx] = (bins[chosen_b_idx][0], bins[chosen_b_idx][1] - w)
        
        return bins
    
    def local_search(bins):
        """Local search: try to reduce number of bins."""
        # Convert to mutable structure
        # bins: list of (items_list, remaining_capacity)
        # Sort bins by total weight (ascending) - try to empty lightest bins first
        
        improved = True
        while improved:
            improved = False
            
            # Sort bins by weight (lightest first) for emptying attempts
            bin_weights_list = [(bin_capacity - rem, i) for i, (items, rem) in enumerate(bins)]
            bin_weights_list.sort()
            
            for bw, src_idx in bin_weights_list:
                if not bins[src_idx][0]:
                    continue
                if bw == 0:
                    continue
                
                src_items = bins[src_idx][0][:]
                # Try to move all items from src to other bins
                moves = {}  # item -> target bin
                temp_remaining = {}
                for i, (items, rem) in enumerate(bins):
                    if i != src_idx:
                        temp_remaining[i] = rem
                
                success = True
                for item_idx in sorted(src_items, key=lambda x: weights[x], reverse=True):
                    w = weights[item_idx]
                    # Find best fit among other bins
                    best_target = -1
                    best_rem = bin_capacity + 1
                    for t_idx, t_rem in temp_remaining.items():
                        if t_rem >= w and t_rem - w < best_rem:
                            best_rem = t_rem - w
                            best_target = t_idx
                    
                    if best_target == -1:
                        success = False
                        break
                    
                    moves[item_idx] = best_target
                    temp_remaining[best_target] -= w
                
                if success:
                    # Apply moves
                    for item_idx, target in moves.items():
                        bins[target][0].append(item_idx)
                        bins[target] = (bins[target][0], bins[target][1] - weights[item_idx])
                    bins[src_idx] = ([], bin_capacity)
                    improved = True
            
            # Remove empty bins
            bins = [(items, rem) for items, rem in bins if items]
            
            if not improved:
                break
        
        return bins
    
    def local_search_swap(bins):
        """Swap-based local search: try 1-1 swaps to create room for bin elimination."""
        improved = True
        max_rounds = 3
        round_count = 0
        
        while improved and round_count < max_rounds:
            improved = False
            round_count += 1
            
            if elapsed() > time_limit * 0.95:
                break
            
            num_bins = len(bins)
            # Try to empty the lightest bin via swaps + moves
            bin_weights_list = [(bin_capacity - rem, i) for i, (items, rem) in enumerate(bins)]
            bin_weights_list.sort()
            
            for bw, src_idx in bin_weights_list[:max(1, num_bins // 4)]:
                if elapsed() > time_limit * 0.95:
                    break
                if not bins[src_idx][0]:
                    continue
                
                # Try pair-swap + redistribute approach
                # For each item in src, try to find a spot in another bin
                # possibly by swapping with a smaller item
                src_items = list(bins[src_idx][0])
                remaining_items = list(src_items)
                temp_bins_rem = {i: rem for i, (items, rem) in enumerate(bins) if i != src_idx}
                temp_bins_items = {i: list(items) for i, (items, rem) in enumerate(bins) if i != src_idx}
                
                moved = True
                while moved and remaining_items:
                    moved = False
                    for item_idx in remaining_items[:]:
                        w = weights[item_idx]
                        # Direct move
                        best_target = -1
                        best_rem = bin_capacity + 1
                        for t_idx, t_rem in temp_bins_rem.items():
                            if t_rem >= w and t_rem - w < best_rem:
                                best_rem = t_rem - w
                                best_target = t_idx
                        
                        if best_target != -1:
                            temp_bins_rem[best_target] -= w
                            temp_bins_items[best_target].append(item_idx)
                            remaining_items.remove(item_idx)
                            moved = True
                            continue
                        
                        # Try 1-1 swap: replace item_idx with a smaller item from some bin
                        best_swap = None
                        best_gain = -1  # net gain in remaining of target
                        for t_idx, t_items in temp_bins_items.items():
                            for swap_item in t_items:
                                sw = weights[swap_item]
                                if sw < w and temp_bins_rem[t_idx] + sw >= w:
                                    # This swap would work
                                    gain = sw  # the item we get back is smaller
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_swap = (t_idx, swap_item)
                        
                        if best_swap is not None:
                            t_idx, swap_item = best_swap
                            sw = weights[swap_item]
                            temp_bins_rem[t_idx] = temp_bins_rem[t_idx] + sw - w
                            temp_bins_items[t_idx].remove(swap_item)
                            temp_bins_items[t_idx].append(item_idx)
                            remaining_items.remove(item_idx)
                            remaining_items.append(swap_item)
                            moved = True
                
                if not remaining_items:
                    # Successfully emptied this bin!
                    # Apply changes
                    new_bins = []
                    for i, (items, rem) in enumerate(bins):
                        if i == src_idx:
                            continue
                        elif i in temp_bins_items:
                            new_items = temp_bins_items[i]
                            new_rem = temp_bins_rem[i]
                            new_bins.append((new_items, new_rem))
                        else:
                            new_bins.append((items, rem))
                    bins = new_bins
                    improved = True
                    break
        
        return bins
    
    # GRASP iterations
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    iteration = 0
    
    while elapsed() < time_limit * 0.95:
        alpha = random.choice(alphas)
        
        # Construction phase
        bins = construct_solution(alpha)
        
        # Local search phase (essential part of GRASP)
        bins = local_search(bins)
        
        num_bins = len(bins)
        
        # If we have time and solution is close to best, do deeper local search
        if num_bins <= best_num_bins + 2 and elapsed() < time_limit * 0.85:
            bins = local_search_swap(bins)
            num_bins = len(bins)
        
        if num_bins < best_num_bins:
            best_num_bins = num_bins
            best_packing = bins[:]
        
        iteration += 1
        
        # For very large instances, limit iterations
        if iteration > 1000:
            break
    
    # Convert best_packing to output format
    if best_packing is None:
        # Shouldn't happen, but fallback
        bins = construct_solution(0.0)
        best_packing = bins
    
    packing = [items for items, rem in best_packing]
    bin_weights_out = [bin_capacity - rem for items, rem in best_packing]
    
    return {"packing": packing, "bin_weights": bin_weights_out}