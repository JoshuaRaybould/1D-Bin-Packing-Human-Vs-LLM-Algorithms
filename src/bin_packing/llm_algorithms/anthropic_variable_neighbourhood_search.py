# anthropic
# variable_neighbourhood_search_0_initial.py

import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Helper: build solution representation
    def make_solution(packing_indices):
        """Create solution dict from list of lists of indices."""
        bins = [list(b) for b in packing_indices if len(b) > 0]
        bin_wts = [sum(weights[i] for i in b) for b in bins]
        return bins, bin_wts
    
    def solution_cost(bins):
        return len(bins)
    
    # FFD initial solution
    def ffd():
        order = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        for idx in order:
            w = weights[idx]
            placed = False
            # Find first bin that fits
            best_j = -1
            for j in range(len(bins)):
                if bin_wts[j] + w <= bin_capacity:
                    best_j = j
                    break
            if best_j >= 0:
                bins[best_j].append(idx)
                bin_wts[best_j] += w
            else:
                bins.append([idx])
                bin_wts.append(w)
        return bins, bin_wts
    
    # Best Fit Decreasing
    def bfd():
        order = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        for idx in order:
            w = weights[idx]
            best_j = -1
            best_remaining = bin_capacity + 1
            for j in range(len(bins)):
                remaining = bin_capacity - bin_wts[j]
                if w <= remaining and remaining < best_remaining:
                    best_j = j
                    best_remaining = remaining
            if best_j >= 0:
                bins[best_j].append(idx)
                bin_wts[best_j] += w
            else:
                bins.append([idx])
                bin_wts.append(w)
        return bins, bin_wts
    
    # Start with better of FFD and BFD
    bins_ffd, wts_ffd = ffd()
    bins_bfd, wts_bfd = bfd()
    if len(bins_bfd) < len(bins_ffd):
        current_bins, current_wts = bins_bfd, wts_bfd
    else:
        current_bins, current_wts = bins_ffd, wts_ffd
    
    best_bins = [list(b) for b in current_bins]
    best_wts = list(current_wts)
    best_cost = len(best_bins)
    
    def elapsed():
        return time.time() - start_time
    
    def copy_solution(bins, wts):
        return [list(b) for b in bins], list(wts)
    
    def clean_empty(bins, wts):
        """Remove empty bins."""
        new_bins = []
        new_wts = []
        for i in range(len(bins)):
            if len(bins[i]) > 0:
                new_bins.append(bins[i])
                new_wts.append(wts[i])
        return new_bins, new_wts
    
    def validate(bins, wts):
        for i in range(len(bins)):
            assert wts[i] == sum(weights[idx] for idx in bins[i])
            assert wts[i] <= bin_capacity
    
    # Local search: try to reduce number of bins
    def local_search(bins, wts, max_iter=None):
        improved = True
        iteration = 0
        while improved:
            if elapsed() > time_limit * 0.98:
                break
            improved = False
            nb = len(bins)
            if nb <= 1:
                break
            
            # Sort bins by weight (try to empty lightest bins)
            order = sorted(range(nb), key=lambda j: wts[j])
            
            for src_idx in order:
                if elapsed() > time_limit * 0.98:
                    return bins, wts
                src = src_idx
                if len(bins[src]) == 0:
                    continue
                
                # Try to move each item from src to another bin
                items_moved = 0
                items_to_move = list(bins[src])
                for item in items_to_move:
                    w = weights[item]
                    # Find best fit target
                    best_target = -1
                    best_remaining = bin_capacity + 1
                    for t in range(nb):
                        if t == src:
                            continue
                        if len(bins[t]) == 0:
                            continue
                        rem = bin_capacity - wts[t]
                        if w <= rem and rem < best_remaining:
                            best_target = t
                            best_remaining = rem
                    if best_target >= 0:
                        bins[src].remove(item)
                        wts[src] -= w
                        bins[best_target].append(item)
                        wts[best_target] += w
                        items_moved += 1
                
                if items_moved == len(items_to_move):
                    # Emptied the bin
                    improved = True
            
            bins, wts = clean_empty(bins, wts)
            
            # Try swap + move: swap two items between bins to create space
            nb = len(bins)
            swap_improved = False
            for _ in range(min(nb * nb, 500)):
                if elapsed() > time_limit * 0.98:
                    return bins, wts
                if nb <= 1:
                    break
                b1 = random.randint(0, nb - 1)
                b2 = random.randint(0, nb - 1)
                if b1 == b2:
                    continue
                if len(bins[b1]) == 0 or len(bins[b2]) == 0:
                    continue
                i1 = random.randint(0, len(bins[b1]) - 1)
                i2 = random.randint(0, len(bins[b2]) - 1)
                item1 = bins[b1][i1]
                item2 = bins[b2][i2]
                w1 = weights[item1]
                w2 = weights[item2]
                # Check if swap is feasible
                new_wt_b1 = wts[b1] - w1 + w2
                new_wt_b2 = wts[b2] - w2 + w1
                if new_wt_b1 <= bin_capacity and new_wt_b2 <= bin_capacity:
                    # Only do swap if it reduces wasted space in some useful way
                    # i.e., it makes one bin more full
                    if new_wt_b1 > wts[b1] or new_wt_b2 > wts[b2]:
                        bins[b1][i1] = item2
                        bins[b2][i2] = item1
                        wts[b1] = new_wt_b1
                        wts[b2] = new_wt_b2
                        swap_improved = True
            
            if swap_improved:
                improved = True
            
            iteration += 1
            if max_iter and iteration >= max_iter:
                break
        
        bins, wts = clean_empty(bins, wts)
        return bins, wts
    
    # Shaking procedures for different neighbourhoods
    def shake(bins, wts, k):
        """Perturb solution in neighbourhood k."""
        bins, wts = copy_solution(bins, wts)
        nb = len(bins)
        if nb <= 1:
            return bins, wts
        
        if k == 1:
            # Move a random item to a random different bin
            for _ in range(1):
                src = random.randint(0, nb - 1)
                if len(bins[src]) == 0:
                    continue
                idx = random.randint(0, len(bins[src]) - 1)
                item = bins[src][idx]
                w = weights[item]
                # Pick a random target (could create new bin)
                targets = [t for t in range(nb) if t != src and wts[t] + w <= bin_capacity]
                if targets:
                    tgt = random.choice(targets)
                    bins[src].pop(idx)
                    wts[src] -= w
                    bins[tgt].append(item)
                    wts[tgt] += w
        
        elif k == 2:
            # Swap items between two random bins
            for _ in range(2):
                b1, b2 = random.sample(range(nb), 2)
                if len(bins[b1]) == 0 or len(bins[b2]) == 0:
                    continue
                i1 = random.randint(0, len(bins[b1]) - 1)
                i2 = random.randint(0, len(bins[b2]) - 1)
                item1 = bins[b1][i1]
                item2 = bins[b2][i2]
                w1, w2 = weights[item1], weights[item2]
                if wts[b1] - w1 + w2 <= bin_capacity and wts[b2] - w2 + w1 <= bin_capacity:
                    bins[b1][i1] = item2
                    bins[b2][i2] = item1
                    wts[b1] = wts[b1] - w1 + w2
                    wts[b2] = wts[b2] - w2 + w1
        
        elif k == 3:
            # Move multiple random items
            for _ in range(3):
                src = random.randint(0, nb - 1)
                if len(bins[src]) == 0:
                    continue
                idx = random.randint(0, len(bins[src]) - 1)
                item = bins[src][idx]
                w = weights[item]
                targets = [t for t in range(nb) if t != src and wts[t] + w <= bin_capacity]
                if targets:
                    tgt = random.choice(targets)
                    bins[src].pop(idx)
                    wts[src] -= w
                    bins[tgt].append(item)
                    wts[tgt] += w
        
        elif k == 4:
            # Empty a random light bin and redistribute
            light_bins = sorted(range(nb), key=lambda j: wts[j])
            for src in light_bins[:max(1, nb // 5)]:
                if len(bins[src]) == 0:
                    continue
                items = list(bins[src])
                random.shuffle(items)
                success = True
                moves = []
                for item in items:
                    w = weights[item]
                    targets = [t for t in range(nb) if t != src and wts[t] + w <= bin_capacity]
                    if targets:
                        # best fit
                        tgt = min(targets, key=lambda t: bin_capacity - wts[t] - w)
                        moves.append((item, w, tgt))
                        wts[tgt] += w  # tentative
                    else:
                        success = False
                        # Undo tentative
                        for (it, ww, tt) in moves:
                            wts[tt] -= ww
                        moves = []
                        break
                if success and moves:
                    for (item, w, tgt) in moves:
                        bins[src].remove(item)
                        wts[src] -= w
                        bins[tgt].append(item)
                    break
        
        elif k == 5:
            # Larger perturbation: randomly redistribute items from multiple light bins
            light_bins = sorted(range(nb), key=lambda j: wts[j])
            num_to_perturb = min(3, max(1, nb // 4))
            all_items = []
            for src in light_bins[:num_to_perturb]:
                all_items.extend(bins[src])
                wts[src] = 0
                bins[src] = []
            
            random.shuffle(all_items)
            # Re-insert using best fit
            for item in all_items:
                w = weights[item]
                best_j = -1
                best_rem = bin_capacity + 1
                for j in range(nb):
                    rem = bin_capacity - wts[j]
                    if w <= rem and rem < best_rem:
                        best_j = j
                        best_rem = rem
                if best_j >= 0:
                    bins[best_j].append(item)
                    wts[best_j] += w
                else:
                    bins.append([item])
                    wts.append(w)
        
        elif k == 6:
            # Large scale: take items from several bins and repack with FFD-like
            num_to_destroy = min(max(2, nb // 3), nb)
            destroy_indices = random.sample(range(nb), num_to_destroy)
            all_items = []
            for src in destroy_indices:
                all_items.extend(bins[src])
                wts[src] = 0
                bins[src] = []
            
            # Sort items by weight descending
            all_items.sort(key=lambda i: -weights[i])
            # Re-insert using first fit decreasing
            for item in all_items:
                w = weights[item]
                best_j = -1
                best_rem = bin_capacity + 1
                for j in range(len(bins)):
                    rem = bin_capacity - wts[j]
                    if w <= rem and rem < best_rem:
                        best_j = j
                        best_rem = rem
                if best_j >= 0:
                    bins[best_j].append(item)
                    wts[best_j] += w
                else:
                    bins.append([item])
                    wts.append(w)
        
        bins, wts = clean_empty(bins, wts)
        return bins, wts
    
    # Apply local search to initial solution
    current_bins, current_wts = local_search(current_bins, current_wts)
    if len(current_bins) < best_cost:
        best_bins, best_wts = copy_solution(current_bins, current_wts)
        best_cost = len(best_bins)
    
    # VNS main loop
    k_max = 6
    k = 1
    max_no_improve = 50
    no_improve_count = 0
    
    while elapsed() < time_limit * 0.95:
        # Shaking
        shaken_bins, shaken_wts = shake(current_bins, current_wts, k)
        
        if elapsed() > time_limit * 0.95:
            break
        
        # Local search
        ls_bins, ls_wts = local_search(shaken_bins, shaken_wts, max_iter=5)
        
        ls_cost = len(ls_bins)
        current_cost = len(current_bins)
        
        # Move or not
        if ls_cost < current_cost:
            current_bins, current_wts = ls_bins, ls_wts
            k = 1
            no_improve_count = 0
            if ls_cost < best_cost:
                best_bins, best_wts = copy_solution(current_bins, current_wts)
                best_cost = ls_cost
        elif ls_cost == current_cost:
            # Accept lateral moves sometimes to diversify
            current_bins, current_wts = ls_bins, ls_wts
            k += 1
            if k > k_max:
                k = 1
                no_improve_count += 1
        else:
            k += 1
            if k > k_max:
                k = 1
                no_improve_count += 1
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Generate a new random-ish solution
            order = list(range(n))
            random.shuffle(order)
            # Sort with some noise
            order.sort(key=lambda i: -weights[i] + random.randint(0, max(1, bin_capacity // 10)))
            new_bins = []
            new_wts = []
            for idx in order:
                w = weights[idx]
                best_j = -1
                best_rem = bin_capacity + 1
                for j in range(len(new_bins)):
                    rem = bin_capacity - new_wts[j]
                    if w <= rem and rem < best_rem:
                        best_j = j
                        best_rem = rem
                if best_j >= 0:
                    new_bins[best_j].append(idx)
                    new_wts[best_j] += w
                else:
                    new_bins.append([idx])
                    new_wts.append(w)
            
            current_bins, current_wts = local_search(new_bins, new_wts, max_iter=5)
            if len(current_bins) < best_cost:
                best_bins, best_wts = copy_solution(current_bins, current_wts)
                best_cost = len(best_bins)
            k = 1
            no_improve_count = 0
    
    # Build final result
    return {
        "packing": best_bins,
        "bin_weights": best_wts
    }