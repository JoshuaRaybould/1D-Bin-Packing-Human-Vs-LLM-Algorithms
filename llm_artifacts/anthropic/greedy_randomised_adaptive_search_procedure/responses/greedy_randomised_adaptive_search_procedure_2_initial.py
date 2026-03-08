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
        """Greedy randomized construction phase."""
        bins = []  # list of (list_of_indices, current_weight)
        
        for idx in sorted_indices:
            w = weights[idx]
            if w > bin_capacity:
                # Item can't fit anywhere, skip or put in own bin
                bins.append(([idx], w))
                continue
            
            # Evaluate all open bins that can fit this item
            candidates = []
            for b_idx, (b_items, b_weight) in enumerate(bins):
                residual = bin_capacity - b_weight - w
                if residual >= 0:
                    candidates.append((b_idx, residual))
            
            if not candidates:
                # Must open a new bin
                bins.append(([idx], w))
            else:
                # Build RCL
                min_res = min(c[1] for c in candidates)
                max_res = max(c[1] for c in candidates)
                threshold = min_res + alpha * (max_res - min_res)
                rcl = [c for c in candidates if c[1] <= threshold]
                
                # Randomly select from RCL
                chosen_bin_idx = random.choice(rcl)[0]
                bins[chosen_bin_idx][0].append(idx)
                bins[chosen_bin_idx] = (bins[chosen_bin_idx][0], bins[chosen_bin_idx][1] + w)
        
        return bins
    
    def local_search(bins):
        """Local search to improve solution by reducing number of bins."""
        # Convert to mutable structure
        bin_items = [list(b[0]) for b in bins]
        bin_weights = [b[1] for b in bins]
        
        improved = True
        max_ls_iterations = 50
        iteration = 0
        
        while improved and iteration < max_ls_iterations:
            improved = False
            iteration += 1
            
            if elapsed() > time_limit * 0.95:
                break
            
            # Sort bins by weight (try to empty lightest bins first)
            order = sorted(range(len(bin_items)), key=lambda i: bin_weights[i])
            
            for src_idx_pos in range(len(order)):
                src = order[src_idx_pos]
                if not bin_items[src]:
                    continue
                
                # Try to move all items from this bin to other bins
                items_to_move = list(bin_items[src])
                moved_all = True
                moves = []  # (item, target_bin)
                
                # Sort items from this bin by weight descending (harder to place first)
                items_to_move.sort(key=lambda i: weights[i], reverse=True)
                
                temp_remaining = {}
                for b in range(len(bin_items)):
                    if b != src and bin_items[b]:
                        temp_remaining[b] = bin_capacity - bin_weights[b]
                
                for item in items_to_move:
                    w = weights[item]
                    # Find best fit among other bins
                    best_target = -1
                    best_residual = bin_capacity + 1
                    for b, rem in temp_remaining.items():
                        if rem >= w and rem - w < best_residual:
                            best_residual = rem - w
                            best_target = b
                    
                    if best_target >= 0:
                        moves.append((item, best_target))
                        temp_remaining[best_target] -= w
                    else:
                        moved_all = False
                        break
                
                if moved_all and moves:
                    # Execute moves
                    for item, target in moves:
                        bin_items[target].append(item)
                        bin_weights[target] += weights[item]
                    bin_items[src] = []
                    bin_weights[src] = 0
                    improved = True
            
            # Try single item moves to reduce fragmentation
            for i in range(len(bin_items)):
                if not bin_items[i]:
                    continue
                if elapsed() > time_limit * 0.95:
                    break
                for item_pos in range(len(bin_items[i]) - 1, -1, -1):
                    item = bin_items[i][item_pos]
                    w = weights[item]
                    rem_i = bin_capacity - bin_weights[i]
                    
                    # Try to move to a bin where it fits better (smaller residual)
                    current_residual = rem_i + w  # residual of source after removing
                    best_target = -1
                    best_score = 0  # improvement
                    
                    for j in range(len(bin_items)):
                        if j == i or not bin_items[j]:
                            continue
                        rem_j = bin_capacity - bin_weights[j]
                        if rem_j >= w:
                            # Moving item from i to j
                            # Score: we want to make bin i emptier (easier to empty later)
                            # and bin j fuller (less waste)
                            new_rem_j = rem_j - w
                            if new_rem_j < rem_j and new_rem_j >= 0:
                                score = rem_j - new_rem_j  # always w, not useful
                                # Prefer tighter fit
                                if best_target == -1 or new_rem_j < (bin_capacity - bin_weights[best_target] - w):
                                    best_target = j
                    
                    # Only move if target bin becomes very full (residual < w)
                    if best_target >= 0:
                        target_rem = bin_capacity - bin_weights[best_target]
                        if target_rem - w < rem_i:  # Better fit in target
                            pass  # Don't do random single moves, they might not help
            
            # Try swap moves: swap item from bin i with item from bin j
            # if it allows better packing
            if not improved and iteration <= 3:
                for i in range(len(bin_items)):
                    if not bin_items[i] or elapsed() > time_limit * 0.95:
                        break
                    for j in range(i + 1, len(bin_items)):
                        if not bin_items[j]:
                            continue
                        for ai_pos, ai in enumerate(bin_items[i]):
                            for aj_pos, aj in enumerate(bin_items[j]):
                                wi = weights[ai]
                                wj = weights[aj]
                                diff = wi - wj
                                # After swap: bin_i_weight changes by -wi+wj, bin_j by -wj+wi
                                new_bw_i = bin_weights[i] - diff
                                new_bw_j = bin_weights[j] + diff
                                if new_bw_i <= bin_capacity and new_bw_j <= bin_capacity:
                                    if new_bw_i >= 0 and new_bw_j >= 0:
                                        # Check if this swap helps
                                        pass  # Swaps alone don't reduce bins
        
        # Remove empty bins
        result_bins = []
        for i in range(len(bin_items)):
            if bin_items[i]:
                result_bins.append((bin_items[i], bin_weights[i]))
        
        return result_bins
    
    def advanced_local_search(bins):
        """More aggressive local search with swap + redistribute."""
        bin_items = [list(b[0]) for b in bins]
        bin_weights = [b[1] for b in bins]
        num_bins = len(bin_items)
        
        if num_bins <= 1:
            return bins
        
        improved = True
        while improved:
            improved = False
            if elapsed() > time_limit * 0.95:
                break
            
            # Try to empty each bin using swaps + moves
            order = sorted(range(len(bin_items)), key=lambda i: bin_weights[i] if bin_items[i] else float('inf'))
            
            for src in order:
                if not bin_items[src]:
                    continue
                if elapsed() > time_limit * 0.95:
                    break
                if bin_weights[src] == 0:
                    continue
                
                # For small bins (few items), try to redistribute with swaps
                if len(bin_items[src]) <= 4:
                    success = try_empty_bin_with_swaps(bin_items, bin_weights, src)
                    if success:
                        improved = True
        
        result_bins = []
        for i in range(len(bin_items)):
            if bin_items[i]:
                result_bins.append((bin_items[i], bin_weights[i]))
        return result_bins
    
    def try_empty_bin_with_swaps(bin_items, bin_weights, src):
        """Try to empty bin src by moving its items, possibly swapping to make room."""
        items = list(bin_items[src])
        items.sort(key=lambda i: weights[i], reverse=True)
        
        # Simple: for each item in src, try direct placement first
        remaining = list(items)
        placements = []  # (item, target)
        swap_actions = []  # (item_from_src, target_bin, item_swapped_out, swap_dest_bin)
        
        temp_space = {}
        for b in range(len(bin_items)):
            if b != src and bin_items[b]:
                temp_space[b] = bin_capacity - bin_weights[b]
        
        unplaced = []
        for item in remaining:
            w = weights[item]
            # Best fit
            best_b = -1
            best_rem = bin_capacity + 1
            for b, space in temp_space.items():
                if space >= w and space - w < best_rem:
                    best_rem = space - w
                    best_b = b
            if best_b >= 0:
                placements.append((item, best_b))
                temp_space[best_b] -= w
            else:
                unplaced.append(item)
        
        if not unplaced:
            # Execute placements
            for item, target in placements:
                bin_items[target].append(item)
                bin_weights[target] += weights[item]
            bin_items[src] = []
            bin_weights[src] = 0
            return True
        
        # Try swap-based approach for unplaced items
        # For each unplaced item, find a bin where swapping out a smaller item makes room
        if len(unplaced) == 1:
            item = unplaced[0]
            w = weights[item]
            for b in range(len(bin_items)):
                if b == src or not bin_items[b]:
                    continue
                space = temp_space.get(b, bin_capacity - bin_weights[b])
                # Need to swap out an item from b that is smaller than item
                for swap_item in bin_items[b]:
                    ws = weights[swap_item]
                    if ws < w and space + ws >= w:
                        # After swap: bin b has space + ws - w remaining
                        new_space = space + ws - w
                        # Now need to place swap_item somewhere else
                        for b2, sp2 in temp_space.items():
                            if b2 == b:
                                actual_sp = new_space
                            else:
                                actual_sp = sp2
                            if b2 != src and actual_sp >= ws:
                                # Execute the whole thing
                                for it, tgt in placements:
                                    bin_items[tgt].append(it)
                                    bin_weights[tgt] += weights[it]
                                # Swap
                                bin_items[b].remove(swap_item)
                                bin_weights[b] -= ws
                                bin_items[b].append(item)
                                bin_weights[b] += w
                                if b2 == b:
                                    bin_items[b].append(swap_item)
                                    bin_weights[b] += ws
                                else:
                                    bin_items[b2].append(swap_item)
                                    bin_weights[b2] += ws
                                bin_items[src] = []
                                bin_weights[src] = 0
                                return True
        
        return False
    
    # GRASP main loop
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    iteration = 0
    max_iterations = 10000
    
    while iteration < max_iterations and elapsed() < time_limit * 0.90:
        iteration += 1
        
        # Reactive GRASP: randomly choose alpha
        alpha = random.choice(alphas)
        
        # Construction phase
        bins = construct_solution(alpha)
        
        # Local search phase (essential part of GRASP)
        bins = local_search(bins)
        
        if len(bins) < best_num_bins:
            # Try advanced local search on promising solutions
            bins = advanced_local_search(bins)
        
        num_bins = len(bins)
        
        if num_bins < best_num_bins:
            best_num_bins = num_bins
            best_packing = bins
        
        if elapsed() > time_limit * 0.90:
            break
    
    # Build result
    packing = [b[0] for b in best_packing]
    bin_weights_result = [b[1] for b in best_packing]
    
    return {"packing": packing, "bin_weights": bin_weights_result}
