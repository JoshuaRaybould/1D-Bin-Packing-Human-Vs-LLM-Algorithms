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
    
    # Build initial solution using First Fit Decreasing
    def ffd():
        indices_sorted = sorted(range(n), key=lambda i: -weights[i])
        bins = []  # list of lists of item indices
        bin_wts = []  # corresponding weights
        for idx in indices_sorted:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if bin_wts[b] + w <= bin_capacity:
                    bins[b].append(idx)
                    bin_wts[b] += w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_wts.append(w)
        return bins, bin_wts
    
    # Best Fit Decreasing
    def bfd():
        indices_sorted = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        for idx in indices_sorted:
            w = weights[idx]
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bins)):
                remaining = bin_capacity - bin_wts[b]
                if w <= remaining and remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b
            if best_bin >= 0:
                bins[best_bin].append(idx)
                bin_wts[best_bin] += w
            else:
                bins.append([idx])
                bin_wts.append(w)
        return bins, bin_wts
    
    def copy_solution(bins, bin_wts):
        return [list(b) for b in bins], list(bin_wts)
    
    def validate(bins, bin_wts):
        # Quick validation (for debugging, can be removed)
        for i, b in enumerate(bins):
            s = sum(weights[idx] for idx in b)
            if s != bin_wts[i]:
                return False
            if s > bin_capacity:
                return False
        return True
    
    def solution_score(bins, bin_wts):
        # Primary: minimize number of bins
        # Secondary: maximize sum of squared fill ratios (promotes consolidation)
        num_bins = len(bins)
        if num_bins == 0:
            return (0, 0)
        fill_score = sum(w * w for w in bin_wts)
        return (-num_bins, fill_score)  # higher is better
    
    def remove_empty_bins(bins, bin_wts):
        new_bins = []
        new_wts = []
        for i in range(len(bins)):
            if len(bins[i]) > 0:
                new_bins.append(bins[i])
                new_wts.append(bin_wts[i])
        return new_bins, new_wts
    
    # Build item-to-bin mapping
    def build_item_bin_map(bins):
        mapping = [0] * n
        for b_idx, b in enumerate(bins):
            for item in b:
                mapping[item] = b_idx
        return mapping
    
    # Local search: try to reduce bins by moving items from least-filled bins
    def local_search(bins, bin_wts, max_iter=None):
        improved = True
        iteration = 0
        while improved:
            if elapsed() > time_limit * 0.98:
                break
            improved = False
            iteration += 1
            if max_iter and iteration > max_iter:
                break
            
            # Sort bins by weight (ascending) - try to empty lightest bins
            order = sorted(range(len(bins)), key=lambda i: bin_wts[i])
            
            for src_idx in order:
                if elapsed() > time_limit * 0.98:
                    return bins, bin_wts
                if len(bins[src_idx]) == 0:
                    continue
                
                # Try to move all items from this bin to other bins
                items = list(bins[src_idx])
                # Sort items by weight descending (harder to place first)
                items.sort(key=lambda i: -weights[i])
                
                # Check if all items can be placed elsewhere
                # Use a greedy best-fit approach
                placements = {}  # item -> target bin
                temp_wts = list(bin_wts)
                can_empty = True
                
                for item in items:
                    w = weights[item]
                    best_bin = -1
                    best_remaining = bin_capacity + 1
                    for b in range(len(bins)):
                        if b == src_idx:
                            continue
                        remaining = bin_capacity - temp_wts[b]
                        if w <= remaining and remaining - w < best_remaining:
                            best_remaining = remaining - w
                            best_bin = b
                    if best_bin >= 0:
                        placements[item] = best_bin
                        temp_wts[best_bin] += w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    # Execute the moves
                    for item, target in placements.items():
                        bins[target].append(item)
                        bin_wts[target] += weights[item]
                    bins[src_idx] = []
                    bin_wts[src_idx] = 0
                    improved = True
            
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
        
        # Additional: try single item moves to improve fill score
        # Try moving items to create better packing
        move_improved = True
        move_rounds = 0
        while move_improved and move_rounds < 3:
            if elapsed() > time_limit * 0.98:
                break
            move_improved = False
            move_rounds += 1
            
            for src in range(len(bins)):
                if elapsed() > time_limit * 0.98:
                    return bins, bin_wts
                for item_pos in range(len(bins[src]) - 1, -1, -1):
                    item = bins[src][item_pos]
                    w = weights[item]
                    # Try to swap with items in other bins to free up space
                    # or move to a bin where it fits better
                    best_target = -1
                    best_score_gain = 0
                    
                    old_src_wt = bin_wts[src]
                    new_src_wt = old_src_wt - w
                    
                    for tgt in range(len(bins)):
                        if tgt == src:
                            continue
                        if bin_wts[tgt] + w > bin_capacity:
                            continue
                        new_tgt_wt = bin_wts[tgt] + w
                        # Score change: new_src^2 + new_tgt^2 - old_src^2 - old_tgt^2
                        gain = (new_src_wt * new_src_wt + new_tgt_wt * new_tgt_wt 
                                - old_src_wt * old_src_wt - bin_wts[tgt] * bin_wts[tgt])
                        if gain > best_score_gain:
                            best_score_gain = gain
                            best_target = tgt
                    
                    if best_target >= 0:
                        bins[best_target].append(item)
                        bin_wts[best_target] += w
                        bins[src].pop(item_pos)
                        bin_wts[src] -= w
                        move_improved = True
            
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
        
        return bins, bin_wts
    
    # Shaking procedures for different neighborhoods
    def shake(bins, bin_wts, k):
        num_bins = len(bins)
        if num_bins <= 1:
            return bins, bin_wts
        
        bins = [list(b) for b in bins]
        bin_wts = list(bin_wts)
        
        if k == 1:
            # Move a random item from a random bin to another (if feasible), or force it
            src = random.randint(0, num_bins - 1)
            if len(bins[src]) == 0:
                return remove_empty_bins(bins, bin_wts)
            item_pos = random.randint(0, len(bins[src]) - 1)
            item = bins[src][item_pos]
            w = weights[item]
            
            # Find feasible targets
            feasible = [b for b in range(num_bins) if b != src and bin_wts[b] + w <= bin_capacity]
            if feasible:
                tgt = random.choice(feasible)
                bins[src].pop(item_pos)
                bin_wts[src] -= w
                bins[tgt].append(item)
                bin_wts[tgt] += w
            
        elif k == 2:
            # Swap two items between two different bins
            b1, b2 = random.sample(range(num_bins), 2)
            if len(bins[b1]) == 0 or len(bins[b2]) == 0:
                return remove_empty_bins(bins, bin_wts)
            i1 = random.randint(0, len(bins[b1]) - 1)
            i2 = random.randint(0, len(bins[b2]) - 1)
            item1, item2 = bins[b1][i1], bins[b2][i2]
            w1, w2 = weights[item1], weights[item2]
            
            new_wt1 = bin_wts[b1] - w1 + w2
            new_wt2 = bin_wts[b2] - w2 + w1
            
            if new_wt1 <= bin_capacity and new_wt2 <= bin_capacity:
                bins[b1][i1] = item2
                bins[b2][i2] = item1
                bin_wts[b1] = new_wt1
                bin_wts[b2] = new_wt2
            
        elif k == 3:
            # Move 2 random items from random bins to other bins
            for _ in range(2):
                src = random.randint(0, len(bins) - 1)
                if len(bins[src]) == 0:
                    continue
                item_pos = random.randint(0, len(bins[src]) - 1)
                item = bins[src][item_pos]
                w = weights[item]
                feasible = [b for b in range(len(bins)) if b != src and bin_wts[b] + w <= bin_capacity]
                if feasible:
                    tgt = random.choice(feasible)
                    bins[src].pop(item_pos)
                    bin_wts[src] -= w
                    bins[tgt].append(item)
                    bin_wts[tgt] += w
            
        elif k == 4:
            # Empty the least-filled bin and redistribute randomly
            min_idx = min(range(len(bins)), key=lambda i: bin_wts[i] if len(bins[i]) > 0 else float('inf'))
            items_to_place = list(bins[min_idx])
            random.shuffle(items_to_place)
            bins[min_idx] = []
            bin_wts[min_idx] = 0
            
            for item in items_to_place:
                w = weights[item]
                feasible = [b for b in range(len(bins)) if bin_wts[b] + w <= bin_capacity]
                if feasible:
                    # Use random among feasible
                    tgt = random.choice(feasible)
                    bins[tgt].append(item)
                    bin_wts[tgt] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
            
        elif k == 5:
            # Empty two least-filled bins and redistribute
            if num_bins < 3:
                return bins, bin_wts
            sorted_indices = sorted(range(num_bins), key=lambda i: bin_wts[i] if len(bins[i]) > 0 else float('inf'))
            items_to_place = []
            empties = set()
            for idx in sorted_indices[:2]:
                if len(bins[idx]) > 0:
                    items_to_place.extend(bins[idx])
                    bins[idx] = []
                    bin_wts[idx] = 0
                    empties.add(idx)
            
            random.shuffle(items_to_place)
            # Sort by weight descending for better packing
            items_to_place.sort(key=lambda i: -weights[i])
            
            for item in items_to_place:
                w = weights[item]
                # Best fit
                best_b = -1
                best_remaining = bin_capacity + 1
                for b in range(len(bins)):
                    rem = bin_capacity - bin_wts[b]
                    if w <= rem and rem - w < best_remaining:
                        best_remaining = rem - w
                        best_b = b
                if best_b >= 0:
                    bins[best_b].append(item)
                    bin_wts[best_b] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
            
        elif k == 6:
            # Empty three least-filled bins and redistribute
            if num_bins < 4:
                return shake(bins, bin_wts, 5)
            sorted_indices = sorted(range(num_bins), key=lambda i: bin_wts[i] if len(bins[i]) > 0 else float('inf'))
            items_to_place = []
            for idx in sorted_indices[:3]:
                if len(bins[idx]) > 0:
                    items_to_place.extend(bins[idx])
                    bins[idx] = []
                    bin_wts[idx] = 0
            
            items_to_place.sort(key=lambda i: -weights[i])
            
            for item in items_to_place:
                w = weights[item]
                best_b = -1
                best_remaining = bin_capacity + 1
                for b in range(len(bins)):
                    rem = bin_capacity - bin_wts[b]
                    if w <= rem and rem - w < best_remaining:
                        best_remaining = rem - w
                        best_b = b
                if best_b >= 0:
                    bins[best_b].append(item)
                    bin_wts[best_b] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
            
        elif k >= 7:
            # Larger perturbation: shuffle items from k-3 least-filled bins
            num_to_empty = min(k - 3, num_bins - 1)
            if num_to_empty < 1:
                num_to_empty = 1
            sorted_indices = sorted(range(num_bins), key=lambda i: bin_wts[i] if len(bins[i]) > 0 else float('inf'))
            items_to_place = []
            for idx in sorted_indices[:num_to_empty]:
                if len(bins[idx]) > 0:
                    items_to_place.extend(bins[idx])
                    bins[idx] = []
                    bin_wts[idx] = 0
            
            items_to_place.sort(key=lambda i: -weights[i])
            
            for item in items_to_place:
                w = weights[item]
                best_b = -1
                best_remaining = bin_capacity + 1
                for b in range(len(bins)):
                    rem = bin_capacity - bin_wts[b]
                    if w <= rem and rem - w < best_remaining:
                        best_remaining = rem - w
                        best_b = b
                if best_b >= 0:
                    bins[best_b].append(item)
                    bin_wts[best_b] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
        
        bins, bin_wts = remove_empty_bins(bins, bin_wts)
        return bins, bin_wts
    
    # Compute lower bound
    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity
    
    # Generate initial solution
    bins_ffd, wts_ffd = ffd()
    bins_bfd, wts_bfd = bfd()
    
    if len(bins_bfd) < len(bins_ffd):
        best_bins, best_wts = bins_bfd, wts_bfd
    else:
        best_bins, best_wts = bins_ffd, wts_ffd
    
    # Apply local search to initial solution
    best_bins, best_wts = local_search(best_bins, best_wts)
    best_score = solution_score(best_bins, best_wts)
    
    if len(best_bins) <= lower_bound:
        return {
            "packing": best_bins,
            "bin_weights": best_wts
        }
    
    # VNS main loop
    k_max = 10
    current_bins, current_wts = copy_solution(best_bins, best_wts)
    current_score = best_score
    
    iteration = 0
    while elapsed() < time_limit * 0.95:
        k = 1
        while k <= k_max and elapsed() < time_limit * 0.95:
            iteration += 1
            
            # Shaking
            shaken_bins, shaken_wts = shake(
                [list(b) for b in current_bins], 
                list(current_wts), 
                k
            )
            
            # Local search
            ls_bins, ls_wts = local_search(shaken_bins, shaken_wts)
            ls_score = solution_score(ls_bins, ls_wts)
            
            # Move or not
            if ls_score > current_score:
                current_bins, current_wts = ls_bins, ls_wts
                current_score = ls_score
                k = 1  # Reset to first neighborhood
                
                if current_score > best_score:
                    best_bins, best_wts = copy_solution(current_bins, current_wts)
                    best_score = current_score
                    
                    if len(best_bins) <= lower_bound:
                        return {
                            "packing": best_bins,
                            "bin_weights": best_wts
                        }
            else:
                k += 1
        
        # If we exhausted all neighborhoods, do a larger perturbation and restart
        # Restart from best solution with a random perturbation
        current_bins, current_wts = copy_solution(best_bins, best_wts)
        current_score = best_score
        
        # Apply a significant random perturbation
        num_bins_curr = len(current_bins)
        if num_bins_curr > 2:
            # Randomly redistribute items from several bins
            num_destroy = max(2, num_bins_curr // 4)
            destroy_indices = random.sample(range(num_bins_curr), min(num_destroy, num_bins_curr))
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(current_bins[idx])
                current_bins[idx] = []
                current_wts[idx] = 0
            
            current_bins, current_wts = remove_empty_bins(current_bins, current_wts)
            
            # Reinsert with some randomization
            random.shuffle(items_to_place)
            items_to_place.sort(key=lambda i: -weights[i])
            
            for item in items_to_place:
                w = weights[item]
                # With some probability use first fit vs best fit
                if random.random() < 0.5:
                    # Best fit
                    best_b = -1
                    best_remaining = bin_capacity + 1
                    for b in range(len(current_bins)):
                        rem = bin_capacity - current_wts[b]
                        if w <= rem and rem - w < best_remaining:
                            best_remaining = rem - w
                            best_b = b
                    if best_b >= 0:
                        current_bins[best_b].append(item)
                        current_wts[best_b] += w
                    else:
                        current_bins.append([item])
                        current_wts.append(w)
                else:
                    # First fit
                    placed = False
                    for b in range(len(current_bins)):
                        if current_wts[b] + w <= bin_capacity:
                            current_bins[b].append(item)
                            current_wts[b] += w
                            placed = True
                            break
                    if not placed:
                        current_bins.append([item])
                        current_wts.append(w)
            
            current_score = solution_score(current_bins, current_wts)
    
    return {
        "packing": best_bins,
        "bin_weights": best_wts
    }