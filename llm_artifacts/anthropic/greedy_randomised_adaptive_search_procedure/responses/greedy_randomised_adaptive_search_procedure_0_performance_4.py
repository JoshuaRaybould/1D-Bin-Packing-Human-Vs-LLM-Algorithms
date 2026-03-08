import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    def elapsed():
        return time.time() - start_time
    
    # Compute lower bound (L1: sum/capacity ceiling)
    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity
    
    # Sort indices by weight descending
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    w_sorted = [weights[i] for i in sorted_indices]  # weights in sorted order
    
    # Map from sorted position to original index
    # sorted_indices[pos] = original_index
    
    best_assignment = None  # assignment[original_index] = bin_number
    best_num_bins = n + 1
    
    half_cap = bin_capacity / 2.0
    
    def construct_solution(alpha):
        """Greedy randomized construction using Best Fit Decreasing with RCL."""
        # bin_rem[b] = remaining capacity of bin b
        bin_rem = []
        assignment = [0] * n  # assignment[original_idx] = bin_idx
        
        for pos in range(len(sorted_indices)):
            orig_idx = sorted_indices[pos]
            w = w_sorted[pos]
            
            if w > bin_capacity:
                b = len(bin_rem)
                bin_rem.append(bin_capacity - w)
                assignment[orig_idx] = b
                continue
            
            # Find candidates (bins that can fit this item)
            # Score = remaining - w (lower is better = tighter fit)
            best_score = bin_capacity + 1
            worst_score = -1
            candidates = []
            
            for b in range(len(bin_rem)):
                rem = bin_rem[b]
                if rem >= w:
                    score = rem - w
                    candidates.append((b, score))
                    if score < best_score:
                        best_score = score
                    if score > worst_score:
                        worst_score = score
            
            if not candidates:
                b = len(bin_rem)
                bin_rem.append(bin_capacity - w)
                assignment[orig_idx] = b
            else:
                if alpha == 0.0 or best_score == worst_score:
                    # Pure greedy: pick best fit
                    chosen_b = candidates[0][0]
                    for b, s in candidates:
                        if s < best_score + 1:  # == best_score effectively
                            if s == best_score:
                                chosen_b = b
                                break
                    # Actually find the one with best_score
                    for b, s in candidates:
                        if s == best_score:
                            chosen_b = b
                            break
                else:
                    threshold = best_score + alpha * (worst_score - best_score)
                    rcl = [b for b, s in candidates if s <= threshold]
                    chosen_b = random.choice(rcl)
                
                bin_rem[chosen_b] -= w
                assignment[orig_idx] = chosen_b
        
        return assignment, bin_rem
    
    def construct_solution_ff(alpha):
        """First Fit Decreasing with RCL."""
        bin_rem = []
        assignment = [0] * n
        
        for pos in range(len(sorted_indices)):
            orig_idx = sorted_indices[pos]
            w = w_sorted[pos]
            
            if w > bin_capacity:
                b = len(bin_rem)
                bin_rem.append(bin_capacity - w)
                assignment[orig_idx] = b
                continue
            
            # First fit: find first bin that fits
            placed = False
            if alpha == 0.0:
                # Pure first fit decreasing
                for b in range(len(bin_rem)):
                    if bin_rem[b] >= w:
                        bin_rem[b] -= w
                        assignment[orig_idx] = b
                        placed = True
                        break
            else:
                # Collect candidates and use RCL
                candidates = []
                for b in range(len(bin_rem)):
                    if bin_rem[b] >= w:
                        score = bin_rem[b] - w
                        candidates.append((b, score))
                
                if candidates:
                    best_score = min(s for _, s in candidates)
                    worst_score = max(s for _, s in candidates)
                    threshold = best_score + alpha * (worst_score - best_score)
                    rcl = [b for b, s in candidates if s <= threshold]
                    if not rcl:
                        rcl = [candidates[0][0]]
                    chosen_b = random.choice(rcl)
                    bin_rem[chosen_b] -= w
                    assignment[orig_idx] = chosen_b
                    placed = True
            
            if not placed:
                b = len(bin_rem)
                bin_rem.append(bin_capacity - w)
                assignment[orig_idx] = b
        
        return assignment, bin_rem
    
    def local_search_eliminate(assignment, bin_rem):
        """Try to eliminate bins by redistributing their items."""
        num_bins = len(bin_rem)
        
        # Build bin contents
        bin_items = [[] for _ in range(num_bins)]
        for i in range(n):
            bin_items[assignment[i]].append(i)
        
        # Sort bins by total weight (ascending) - try to empty lightest first
        bin_weight = [bin_capacity - bin_rem[b] for b in range(num_bins)]
        order = sorted(range(num_bins), key=lambda b: bin_weight[b])
        
        eliminated = set()
        
        for src in order:
            if src in eliminated:
                continue
            if not bin_items[src]:
                eliminated.add(src)
                continue
            if bin_weight[src] == 0:
                eliminated.add(src)
                continue
            
            # Try to move all items from src to other bins
            src_items_sorted = sorted(bin_items[src], key=lambda i: weights[i], reverse=True)
            
            # Temporary remaining capacities
            temp_rem = dict()
            for b in range(num_bins):
                if b != src and b not in eliminated:
                    temp_rem[b] = bin_rem[b]
            
            moves = {}  # item -> target_bin
            success = True
            
            for item in src_items_sorted:
                w = weights[item]
                # Best fit
                best_target = -1
                best_left = bin_capacity + 1
                for b, rem in temp_rem.items():
                    if rem >= w:
                        left = rem - w
                        if left < best_left:
                            best_left = left
                            best_target = b
                
                if best_target == -1:
                    success = False
                    break
                
                moves[item] = best_target
                temp_rem[best_target] -= w
            
            if success:
                # Apply moves
                for item, target in moves.items():
                    assignment[item] = target
                    bin_rem[target] = temp_rem[target]  # might be updated multiple times but that's ok
                    bin_items[target].append(item)
                bin_items[src] = []
                bin_rem[src] = bin_capacity
                bin_weight[src] = 0
                eliminated.add(src)
                # Update temp_rem for future iterations
                for b in temp_rem:
                    bin_rem[b] = temp_rem[b]
        
        return assignment, bin_rem, bin_items, eliminated
    
    def local_search_swap_eliminate(assignment, bin_rem, bin_items, eliminated, time_frac=0.95):
        """Try swap-based elimination of bins."""
        num_bins_total = len(bin_rem)
        active_bins = [b for b in range(num_bins_total) if b not in eliminated and bin_items[b]]
        
        if not active_bins:
            return assignment, bin_rem, bin_items, eliminated
        
        bin_weight = {b: bin_capacity - bin_rem[b] for b in active_bins}
        order = sorted(active_bins, key=lambda b: bin_weight[b])
        
        max_attempts = min(len(order), 5)  # try to empty up to 5 lightest bins
        
        for attempt_idx in range(max_attempts):
            if elapsed() > time_limit * time_frac:
                break
            
            src = order[attempt_idx]
            if src in eliminated or not bin_items[src]:
                continue
            
            # Build temp structures
            other_bins = [b for b in active_bins if b != src and b not in eliminated]
            temp_rem = {b: bin_rem[b] for b in other_bins}
            temp_items = {b: list(bin_items[b]) for b in other_bins}
            
            remaining_items = sorted(bin_items[src], key=lambda i: weights[i], reverse=True)
            
            max_rounds = 20
            round_num = 0
            while remaining_items and round_num < max_rounds:
                round_num += 1
                moved_any = False
                
                for item in remaining_items[:]:
                    w = weights[item]
                    
                    # Try direct move (best fit)
                    best_target = -1
                    best_left = bin_capacity + 1
                    for b in other_bins:
                        if b in eliminated:
                            continue
                        rem = temp_rem[b]
                        if rem >= w:
                            left = rem - w
                            if left < best_left:
                                best_left = left
                                best_target = b
                    
                    if best_target != -1:
                        temp_rem[best_target] -= w
                        temp_items[best_target].append(item)
                        remaining_items.remove(item)
                        moved_any = True
                        continue
                    
                    # Try 1-1 swap: find item in another bin that is smaller,
                    # and swapping would allow our item to fit
                    best_swap = None
                    best_swap_gain = -1  # prefer swap that frees most space (smallest returned item)
                    
                    for b in other_bins:
                        if b in eliminated:
                            continue
                        rem_b = temp_rem[b]
                        for swap_item in temp_items[b]:
                            sw = weights[swap_item]
                            if sw < w and rem_b + sw >= w:
                                # After swap: rem_b + sw - w >= 0
                                # We want the returned item to be as small as possible
                                # so it's easier to place elsewhere
                                if best_swap is None or sw < best_swap_gain:
                                    best_swap_gain = sw
                                    best_swap = (b, swap_item)
                    
                    if best_swap is not None:
                        b, swap_item = best_swap
                        sw = weights[swap_item]
                        temp_rem[b] = temp_rem[b] + sw - w
                        temp_items[b].remove(swap_item)
                        temp_items[b].append(item)
                        remaining_items.remove(item)
                        remaining_items.append(swap_item)
                        # Re-sort: try largest first
                        remaining_items.sort(key=lambda i: weights[i], reverse=True)
                        moved_any = True
                        break  # restart the for loop
                
                if not moved_any:
                    break
            
            if not remaining_items:
                # Successfully emptied this bin!
                for b in other_bins:
                    if b not in eliminated:
                        bin_rem[b] = temp_rem[b]
                        bin_items[b] = temp_items[b]
                        for item in temp_items[b]:
                            assignment[item] = b
                bin_items[src] = []
                bin_rem[src] = bin_capacity
                eliminated.add(src)
                active_bins = [b for b in active_bins if b != src]
        
        return assignment, bin_rem, bin_items, eliminated
    
    def compact_solution(assignment, bin_rem, bin_items, eliminated):
        """Compact bins: renumber and create final structures."""
        active_bins = [b for b in range(len(bin_rem)) if b not in eliminated and bin_items[b]]
        num_bins = len(active_bins)
        return num_bins, active_bins
    
    def get_packing(assignment, bin_rem, bin_items, eliminated):
        active_bins = [b for b in range(len(bin_rem)) if b not in eliminated and bin_items[b]]
        packing = [bin_items[b] for b in active_bins]
        bin_weights_out = [bin_capacity - bin_rem[b] for b in active_bins]
        return packing, bin_weights_out
    
    # GRASP main loop
    alphas = [0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    iteration = 0
    
    # First, do a pure greedy solution
    assignment, bin_rem = construct_solution(0.0)
    assignment, bin_rem, bin_items, eliminated = local_search_eliminate(assignment, bin_rem)
    assignment, bin_rem, bin_items, eliminated = local_search_swap_eliminate(
        assignment, bin_rem, bin_items, eliminated, time_frac=0.3
    )
    num_bins, active = compact_solution(assignment, bin_rem, bin_items, eliminated)
    
    if num_bins < best_num_bins:
        best_num_bins = num_bins
        best_packing_data = (assignment[:], list(bin_rem), [list(bi) for bi in bin_items], set(eliminated))
    
    if best_num_bins <= lower_bound:
        packing, bin_weights_out = get_packing(*best_packing_data)
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Also try FFD
    assignment2, bin_rem2 = construct_solution_ff(0.0)
    assignment2, bin_rem2, bin_items2, eliminated2 = local_search_eliminate(assignment2, bin_rem2)
    assignment2, bin_rem2, bin_items2, eliminated2 = local_search_swap_eliminate(
        assignment2, bin_rem2, bin_items2, eliminated2, time_frac=0.3
    )
    num_bins2, _ = compact_solution(assignment2, bin_rem2, bin_items2, eliminated2)
    
    if num_bins2 < best_num_bins:
        best_num_bins = num_bins2
        best_packing_data = (assignment2[:], list(bin_rem2), [list(bi) for bi in bin_items2], set(eliminated2))
    
    if best_num_bins <= lower_bound:
        packing, bin_weights_out = get_packing(*best_packing_data)
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    iteration = 2
    
    while elapsed() < time_limit * 0.95:
        if best_num_bins <= lower_bound:
            break
        
        alpha = random.choice(alphas)
        
        # Alternate between BF and FF construction
        if iteration % 2 == 0:
            assignment, bin_rem = construct_solution(alpha)
        else:
            assignment, bin_rem = construct_solution_ff(alpha)
        
        # Local search: eliminate bins
        assignment, bin_rem, bin_items, eliminated = local_search_eliminate(assignment, bin_rem)
        
        # Count current bins
        num_bins = sum(1 for b in range(len(bin_rem)) if b not in eliminated and bin_items[b])
        
        # Only do expensive swap search if solution is promising
        if num_bins <= best_num_bins + 1 and elapsed() < time_limit * 0.90:
            assignment, bin_rem, bin_items, eliminated = local_search_swap_eliminate(
                assignment, bin_rem, bin_items, eliminated, time_frac=0.93
            )
            num_bins = sum(1 for b in range(len(bin_rem)) if b not in eliminated and bin_items[b])
        
        if num_bins < best_num_bins:
            best_num_bins = num_bins
            best_packing_data = (assignment[:], list(bin_rem), [list(bi) for bi in bin_items], set(eliminated))
            
            if best_num_bins <= lower_bound:
                break
        
        iteration += 1
        
        if iteration > 5000:
            break
    
    packing, bin_weights_out = get_packing(*best_packing_data)
    return {"packing": packing, "bin_weights": bin_weights_out}
