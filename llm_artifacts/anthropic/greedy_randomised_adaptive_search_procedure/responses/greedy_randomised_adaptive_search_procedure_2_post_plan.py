import random
import time
import math
from bisect import bisect_left, insort

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    def elapsed():
        return time.time() - start_time
    
    # Precompute
    total_weight = sum(weights)
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    # --- Step 3: Lower bound computation (L2 Martello-Toth) ---
    L1 = math.ceil(total_weight / C)
    
    # Compute L2
    L2 = L1
    half_C = C / 2.0
    # Sort weights for threshold computation
    sorted_w = sorted(weights, reverse=True)
    
    # For each threshold t from 1 to floor(C/2)
    # Items > C - t go into "large" category (need own bin or share with small)
    # Items in (t, C-t] are "medium" - but we use the standard formulation
    # L2(t) = |{i: w_i > C/2}| + max(0, ceil((sum of w_i where w_i <= C/2 and w_i > t - free_space_from_large_bins) / C))
    # Actually let's use a simpler but effective L2:
    # For threshold t: count items > C-t (they each need a bin), 
    # compute free space in those bins, subtract items <= t that fit,
    # remaining items need ceil(sum/C) additional bins
    
    try:
        # More efficient L2 computation
        for t in range(1, C // 2 + 1):
            # Items with weight > C - t (large items, each needs own bin)
            n_large = 0
            free_space = 0
            sum_small = 0
            n_small = 0
            sum_medium = 0
            n_medium = 0
            
            for w in weights:
                if w > C - t:
                    n_large += 1
                    free_space += C - w
                elif w > t:
                    # medium items: in range (t, C-t]
                    sum_medium += w
                    n_medium += 1
                else:
                    # small items: <= t
                    sum_small += w
                    n_small += 1
            
            # Medium items need their own bins (paired among themselves)
            bins_for_medium = math.ceil(sum_medium / C) if sum_medium > 0 else 0
            # But actually L2 formula: n_large + n_medium + max(0, ceil((sum_small - free_from_large - free_from_medium)/C))
            # Let's use: L2(t) = n_large + ceil((sum_medium + max(0, sum_small - free_space)) / C)
            
            remaining_small = max(0, sum_small - free_space)
            needed = n_large + math.ceil((sum_medium + remaining_small) / C) if (sum_medium + remaining_small) > 0 else n_large
            L2 = max(L2, needed)
            
            if elapsed() > time_limit * 0.01:  # Don't spend too long on LB
                break
    except:
        pass
    
    lower_bound = L2
    
    # --- Data structures for solutions ---
    # Solution: parallel lists bin_items[i], bin_remaining[i]
    
    best_packing = None
    best_num_bins = n + 1
    
    # --- Step 2: Efficient construction with bisect-based best-fit ---
    def construct_solution(alpha, strategy=0):
        """Greedy randomized construction phase."""
        bin_items_list = []
        bin_remaining_list = []
        # For fast best-fit: maintain sorted list of (remaining, bin_index)
        sorted_bins = []  # sorted by remaining capacity
        
        if strategy == 0 or strategy == 1:
            order = sorted_indices
        else:
            # Strategy 2: slightly shuffled
            order = list(sorted_indices)
            for i in range(len(order) - 1):
                if random.random() < 0.1:
                    order[i], order[i+1] = order[i+1], order[i]
        
        for idx in order:
            w = weights[idx]
            if w > C:
                bin_items_list.append([idx])
                bin_remaining_list.append(C - w)
                # Don't add to sorted_bins since remaining < 0
                continue
            
            if w == 0:
                # Put zero-weight items in first bin or create one
                if bin_items_list:
                    bin_items_list[0].append(idx)
                else:
                    bin_items_list.append([idx])
                    bin_remaining_list.append(C)
                    insort(sorted_bins, (C, 0))
                continue
            
            # Find feasible bins using bisect
            pos = bisect_left(sorted_bins, (w, -1))
            
            if pos >= len(sorted_bins):
                # No bin can fit this item
                b_idx = len(bin_items_list)
                bin_items_list.append([idx])
                bin_remaining_list.append(C - w)
                insort(sorted_bins, (C - w, b_idx))
                continue
            
            if alpha == 0.0:
                # Pure best-fit: smallest remaining that fits
                rem, b_idx = sorted_bins[pos]
                # Remove from sorted list
                sorted_bins.pop(pos)
                bin_items_list[b_idx].append(idx)
                new_rem = rem - w
                bin_remaining_list[b_idx] = new_rem
                if new_rem > 0:
                    insort(sorted_bins, (new_rem, b_idx))
            elif strategy == 1:
                # First-fit style with RCL
                # Candidates are all bins from pos onwards
                num_feasible = len(sorted_bins) - pos
                if num_feasible == 1 or alpha == 0.0:
                    rem, b_idx = sorted_bins[pos]
                    sorted_bins.pop(pos)
                    bin_items_list[b_idx].append(idx)
                    new_rem = rem - w
                    bin_remaining_list[b_idx] = new_rem
                    if new_rem > 0:
                        insort(sorted_bins, (new_rem, b_idx))
                else:
                    min_res = sorted_bins[pos][0] - w
                    max_res = sorted_bins[-1][0] - w
                    threshold = min_res + alpha * (max_res - min_res)
                    # RCL: bins with residual_after <= threshold
                    # residual_after = remaining - w, so remaining <= threshold + w
                    rcl_end = bisect_left(sorted_bins, (threshold + w + 1, -1))
                    if rcl_end <= pos:
                        rcl_end = pos + 1
                    chosen_pos = random.randint(pos, min(rcl_end - 1, len(sorted_bins) - 1))
                    rem, b_idx = sorted_bins[chosen_pos]
                    sorted_bins.pop(chosen_pos)
                    bin_items_list[b_idx].append(idx)
                    new_rem = rem - w
                    bin_remaining_list[b_idx] = new_rem
                    if new_rem > 0:
                        insort(sorted_bins, (new_rem, b_idx))
            else:
                # Best-fit with RCL (strategy 0 or 2)
                num_feasible = len(sorted_bins) - pos
                if num_feasible == 1:
                    rem, b_idx = sorted_bins[pos]
                    sorted_bins.pop(pos)
                    bin_items_list[b_idx].append(idx)
                    new_rem = rem - w
                    bin_remaining_list[b_idx] = new_rem
                    if new_rem > 0:
                        insort(sorted_bins, (new_rem, b_idx))
                else:
                    min_res = sorted_bins[pos][0] - w
                    max_res = sorted_bins[-1][0] - w
                    threshold = min_res + alpha * (max_res - min_res)
                    rcl_end = bisect_left(sorted_bins, (threshold + w + 1, -1))
                    if rcl_end <= pos:
                        rcl_end = pos + 1
                    chosen_pos = random.randint(pos, min(rcl_end - 1, len(sorted_bins) - 1))
                    rem, b_idx = sorted_bins[chosen_pos]
                    sorted_bins.pop(chosen_pos)
                    bin_items_list[b_idx].append(idx)
                    new_rem = rem - w
                    bin_remaining_list[b_idx] = new_rem
                    if new_rem > 0:
                        insort(sorted_bins, (new_rem, b_idx))
        
        return bin_items_list, bin_remaining_list
    
    # --- Steps 1,4,6: Rewritten local search ---
    def local_search(bin_items, bin_remaining, time_frac=0.95):
        """Local search: bin emptying with swap-assisted redistribution + consolidation."""
        time_check_counter = 0
        
        # Phase 1 + Phase 2 loop
        improved = True
        ls_rounds = 0
        while improved and ls_rounds < 20:
            improved = False
            ls_rounds += 1
            
            time_check_counter += 1
            if time_check_counter % 5 == 0 and elapsed() > time_limit * time_frac:
                break
            
            # Get active bins sorted by weight ascending (lightest first to try emptying)
            active = []
            for i in range(len(bin_items)):
                if bin_items[i]:
                    active.append((C - bin_remaining[i], i))  # (weight, index)
            active.sort()
            
            # Build sorted remaining for best-fit queries
            # We'll rebuild this as needed
            
            for weight_val, src in active:
                if not bin_items[src]:
                    continue
                if weight_val == 0:
                    continue
                
                time_check_counter += 1
                if time_check_counter % 10 == 0 and elapsed() > time_limit * time_frac:
                    break
                
                # Try to empty bin src
                items_to_place = sorted(bin_items[src], key=lambda i: weights[i], reverse=True)
                
                # Simulate placement
                temp_remaining = {}
                for i in range(len(bin_items)):
                    if i != src and bin_items[i]:
                        temp_remaining[i] = bin_remaining[i]
                
                # Build sorted structure for best-fit
                sr = sorted([(rem, idx) for idx, rem in temp_remaining.items()])
                
                placements = []
                unplaced = []
                
                for item in items_to_place:
                    w = weights[item]
                    if w == 0:
                        # Place in any bin
                        if sr:
                            placements.append((item, sr[0][1]))
                        continue
                    # Best fit using bisect
                    pos = bisect_left(sr, (w, -1))
                    if pos < len(sr):
                        rem, b_idx = sr[pos]
                        sr.pop(pos)
                        new_rem = rem - w
                        if new_rem > 0:
                            insort(sr, (new_rem, b_idx))
                        placements.append((item, b_idx))
                    else:
                        unplaced.append(item)
                
                if not unplaced:
                    # Successfully empty the bin!
                    for item, target in placements:
                        bin_items[target].append(item)
                        bin_remaining[target] -= weights[item]
                    bin_items[src] = []
                    bin_remaining[src] = 0
                    improved = True
                    continue
                
                # Phase 2: Swap-assisted emptying for unplaced items
                if len(unplaced) <= 2:
                    # Try to swap items to make room
                    success = try_swap_empty(bin_items, bin_remaining, src, placements, unplaced, sr, temp_remaining)
                    if success:
                        improved = True
                        continue
            
            # Phase 6: Consolidation moves
            if not improved and ls_rounds <= 5:
                did_consolidate = consolidation(bin_items, bin_remaining, time_frac)
                if did_consolidate:
                    improved = True
        
        # Remove empty bins and return
        new_items = []
        new_remaining = []
        for i in range(len(bin_items)):
            if bin_items[i]:
                new_items.append(bin_items[i])
                new_remaining.append(bin_remaining[i])
        return new_items, new_remaining
    
    def try_swap_empty(bin_items, bin_remaining, src, placements, unplaced, sr, temp_remaining):
        """Try swap-assisted emptying for unplaced items."""
        if len(unplaced) > 2:
            return False
        
        # Rebuild temp_remaining accounting for placements
        placed_adjustments = {}
        for item, target in placements:
            placed_adjustments[target] = placed_adjustments.get(target, 0) + weights[item]
        
        adjusted_remaining = {}
        for idx, rem in temp_remaining.items():
            adjusted_remaining[idx] = rem - placed_adjustments.get(idx, 0)
        
        for item in unplaced:
            w = weights[item]
            found = False
            # Try each bin, swap out a smaller item
            for b_idx in list(adjusted_remaining.keys()):
                if adjusted_remaining[b_idx] >= w:
                    # Can place directly (shouldn't happen since it was unplaced, but just in case)
                    adjusted_remaining[b_idx] -= w
                    placements.append((item, b_idx))
                    found = True
                    break
                
                # Try swapping out an item from b_idx
                space = adjusted_remaining[b_idx]
                for swap_item in bin_items[b_idx]:
                    # Don't swap out items that were just placed
                    ws = weights[swap_item]
                    if ws >= w:
                        continue  # Need smaller item to swap out
                    if space + ws >= w:
                        # Can place item after removing swap_item
                        # Now need to place swap_item somewhere
                        new_space_b = space + ws - w
                        # Find a place for swap_item
                        placed_swap = False
                        for b2_idx, rem2 in adjusted_remaining.items():
                            if b2_idx == b_idx:
                                check_rem = new_space_b
                            else:
                                check_rem = rem2
                            if check_rem >= ws:
                                # Execute
                                for it, tgt in placements:
                                    bin_items[tgt].append(it)
                                    bin_remaining[tgt] -= weights[it]
                                # Swap
                                bin_items[b_idx].remove(swap_item)
                                bin_remaining[b_idx] += ws
                                bin_items[b_idx].append(item)
                                bin_remaining[b_idx] -= w
                                if b2_idx == b_idx:
                                    bin_items[b_idx].append(swap_item)
                                    bin_remaining[b_idx] -= ws
                                else:
                                    bin_items[b2_idx].append(swap_item)
                                    bin_remaining[b2_idx] -= ws
                                # Handle remaining unplaced
                                remaining_unplaced = [u for u in unplaced if u != item]
                                if remaining_unplaced:
                                    # Try to place remaining
                                    all_placed = True
                                    for u in remaining_unplaced:
                                        wu = weights[u]
                                        placed_u = False
                                        for bi in range(len(bin_items)):
                                            if bi != src and bin_items[bi] and bin_remaining[bi] >= wu:
                                                bin_items[bi].append(u)
                                                bin_remaining[bi] -= wu
                                                placed_u = True
                                                break
                                        if not placed_u:
                                            all_placed = False
                                            break
                                    if not all_placed:
                                        # Rollback - too complex, just fail
                                        return False
                                bin_items[src] = []
                                bin_remaining[src] = 0
                                return True
                        # If we get here, couldn't place swap_item
                        continue
            if not found:
                return False
        
        # All unplaced items were placed directly
        for it, tgt in placements:
            bin_items[tgt].append(it)
            bin_remaining[tgt] -= weights[it]
        bin_items[src] = []
        bin_remaining[src] = 0
        return True
    
    def consolidation(bin_items, bin_remaining, time_frac):
        """Consolidation: move items to create tighter fits, enabling future bin emptying."""
        did_something = False
        
        # Pairwise swaps to make bins more unbalanced (one fuller, one emptier)
        active_bins = [i for i in range(len(bin_items)) if bin_items[i]]
        
        if len(active_bins) < 2:
            return False
        
        tc = 0
        for ai in range(len(active_bins)):
            i = active_bins[ai]
            tc += 1
            if tc % 20 == 0 and elapsed() > time_limit * time_frac:
                break
            for aj in range(ai + 1, len(active_bins)):
                j = active_bins[aj]
                bw_i = C - bin_remaining[i]
                bw_j = C - bin_remaining[j]
                
                # Try swapping items
                best_swap = None
                best_score = min(bw_i, bw_j)  # current min weight (we want to minimize it to empty a bin)
                
                for item_a in bin_items[i]:
                    wa = weights[item_a]
                    for item_b in bin_items[j]:
                        wb = weights[item_b]
                        diff = wa - wb
                        new_bw_i = bw_i - diff
                        new_bw_j = bw_j + diff
                        if new_bw_i > C or new_bw_j > C:
                            continue
                        if new_bw_i < 0 or new_bw_j < 0:
                            continue
                        new_min = min(new_bw_i, new_bw_j)
                        if new_min < best_score:
                            best_score = new_min
                            best_swap = (item_a, item_b)
                
                # Also try moving item from i to j (no swap back)
                for item_a in bin_items[i]:
                    wa = weights[item_a]
                    if bin_remaining[j] >= wa:
                        new_bw_i = bw_i - wa
                        new_min = new_bw_i  # This makes bin i lighter
                        if new_min < best_score:
                            best_score = new_min
                            best_swap = (item_a, None)  # move from i to j
                
                # Try moving from j to i
                for item_b in bin_items[j]:
                    wb = weights[item_b]
                    if bin_remaining[i] >= wb:
                        new_bw_j = bw_j - wb
                        new_min = new_bw_j
                        if new_min < best_score:
                            best_score = new_min
                            best_swap = (None, item_b)  # move from j to i
                
                if best_swap is not None:
                    item_a, item_b = best_swap
                    if item_a is not None and item_b is not None:
                        # Swap
                        wa = weights[item_a]
                        wb = weights[item_b]
                        bin_items[i].remove(item_a)
                        bin_items[j].remove(item_b)
                        bin_items[i].append(item_b)
                        bin_items[j].append(item_a)
                        bin_remaining[i] += wa - wb
                        bin_remaining[j] += wb - wa
                        did_something = True
                    elif item_a is not None:
                        # Move item_a from i to j
                        wa = weights[item_a]
                        bin_items[i].remove(item_a)
                        bin_items[j].append(item_a)
                        bin_remaining[i] += wa
                        bin_remaining[j] -= wa
                        did_something = True
                    elif item_b is not None:
                        # Move item_b from j to i
                        wb = weights[item_b]
                        bin_items[j].remove(item_b)
                        bin_items[i].append(item_b)
                        bin_remaining[j] += wb
                        bin_remaining[i] -= wb
                        did_something = True
        
        return did_something
    
    def perturb_solution(bin_items, bin_remaining):
        """Perturb solution by removing 2-3 bins and re-inserting items."""
        active = [i for i in range(len(bin_items)) if bin_items[i]]
        if len(active) <= 2:
            return bin_items, bin_remaining
        
        num_remove = min(random.randint(2, 3), len(active) - 1)
        to_remove = random.sample(active, num_remove)
        
        # Collect items from removed bins
        freed_items = []
        for b in to_remove:
            freed_items.extend(bin_items[b])
            bin_items[b] = []
            bin_remaining[b] = 0
        
        # Shuffle freed items and re-insert with best-fit
        random.shuffle(freed_items)
        freed_items.sort(key=lambda i: weights[i], reverse=True)
        
        # Build sorted remaining for active bins
        sr = []
        for i in range(len(bin_items)):
            if bin_items[i]:
                sr.append((bin_remaining[i], i))
        sr.sort()
        
        for item in freed_items:
            w = weights[item]
            if w == 0:
                if sr:
                    bin_items[sr[0][1]].append(item)
                else:
                    b_idx = len(bin_items)
                    bin_items.append([item])
                    bin_remaining.append(C)
                    insort(sr, (C, b_idx))
                continue
            pos = bisect_left(sr, (w, -1))
            if pos < len(sr):
                rem, b_idx = sr[pos]
                sr.pop(pos)
                new_rem = rem - w
                bin_items[b_idx].append(item)
                bin_remaining[b_idx] = new_rem
                if new_rem > 0:
                    insort(sr, (new_rem, b_idx))
            else:
                b_idx = len(bin_items)
                bin_items.append([item])
                bin_remaining.append(C - w)
                insort(sr, (C - w, b_idx))
        
        return bin_items, bin_remaining
    
    def count_active(bin_items):
        return sum(1 for b in bin_items if b)
    
    def solution_to_result(bin_items, bin_remaining):
        packing = []
        bw = []
        for i in range(len(bin_items)):
            if bin_items[i]:
                packing.append(list(bin_items[i]))
                bw.append(C - bin_remaining[i])
        return packing, bw
    
    # --- Step 5: Reactive GRASP alpha selection ---
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_scores = [1.0] * len(alphas)
    
    def select_alpha():
        total = sum(alpha_scores)
        r = random.random() * total
        cumsum = 0.0
        for i, s in enumerate(alpha_scores):
            cumsum += s
            if r <= cumsum:
                return i, alphas[i]
        return len(alphas) - 1, alphas[-1]
    
    # --- Initial solution: BFD (alpha=0) ---
    bi, br = construct_solution(0.0, strategy=0)
    bi, br = local_search(bi, br, time_frac=0.95)
    num_bins = count_active(bi)
    
    best_num_bins = num_bins
    best_bi = [list(b) for b in bi]
    best_br = list(br)
    
    if best_num_bins <= lower_bound:
        packing, bw = solution_to_result(best_bi, best_br)
        return {"packing": packing, "bin_weights": bw}
    
    # --- GRASP main loop ---
    iteration = 0
    max_iterations = 100000
    no_improve_count = 0
    
    while iteration < max_iterations and elapsed() < time_limit * 0.98:
        iteration += 1
        
        # Step 5: Select alpha reactively
        alpha_idx, alpha = select_alpha()
        
        # Step 9: Select construction strategy
        r = random.random()
        if r < 0.70:
            strategy = 0  # BFD-style
        elif r < 0.85:
            strategy = 1  # FFD-style
        else:
            strategy = 2  # Shuffled
        
        # Construction phase
        bi, br = construct_solution(alpha, strategy)
        
        # Local search phase
        num_before_ls = count_active(bi)
        bi, br = local_search(bi, br, time_frac=0.98)
        num_bins = count_active(bi)
        
        # Update alpha scores
        if num_bins <= best_num_bins and best_num_bins > 0:
            reward = (best_num_bins / num_bins) ** 2
            alpha_scores[alpha_idx] += reward
        
        # Decay scores every 50 iterations
        if iteration % 50 == 0:
            alpha_scores = [s * 0.9 for s in alpha_scores]
            # Ensure minimum score
            alpha_scores = [max(s, 0.1) for s in alpha_scores]
        
        if num_bins < best_num_bins:
            best_num_bins = num_bins
            best_bi = [list(b) for b in bi]
            best_br = list(br)
            no_improve_count = 0
            
            if best_num_bins <= lower_bound:
                break
        else:
            no_improve_count += 1
        
        # Step 8: Perturbation-based diversification
        if no_improve_count >= 20:
            # Perturb best solution
            p_bi = [list(b) for b in best_bi]
            p_br = list(best_br)
            p_bi, p_br = perturb_solution(p_bi, p_br)
            p_bi, p_br = local_search(p_bi, p_br, time_frac=0.98)
            p_num = count_active(p_bi)
            
            if p_num < best_num_bins:
                best_num_bins = p_num
                best_bi = [list(b) for b in p_bi]
                best_br = list(p_br)
                no_improve_count = 0
                
                if best_num_bins <= lower_bound:
                    break
            else:
                no_improve_count = 0  # Reset to avoid constant perturbation
        
        if elapsed() > time_limit * 0.98:
            break
    
    # Build result
    packing, bw = solution_to_result(best_bi, best_br)
    return {"packing": packing, "bin_weights": bw}
