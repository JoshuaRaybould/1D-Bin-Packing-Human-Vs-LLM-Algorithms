import random
import time
import bisect

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return time_limit - elapsed()
    
    total_weight = sum(weights)
    
    # L2 Lower Bound (Martello-Toth)
    def compute_lower_bound():
        simple_lb = (total_weight + C - 1) // C
        
        if n == 0:
            return simple_lb
        
        # Sort weights for efficient threshold computation
        sorted_w = sorted(weights, reverse=True)
        
        # Get distinct values that could serve as thresholds
        # We only need to check alpha values where alpha <= C//2
        half_C = C // 2
        
        # Collect distinct relevant weight values as potential thresholds
        distinct_vals = set()
        for w in sorted_w:
            if w > C:
                continue
            # alpha such that C - alpha = w, i.e., alpha = C - w
            a = C - w
            if 1 <= a <= half_C:
                distinct_vals.add(a)
            if 1 <= w <= half_C:
                distinct_vals.add(w)
        distinct_vals.add(1)
        if half_C >= 1:
            distinct_vals.add(half_C)
        
        best_lb = simple_lb
        
        for alpha in distinct_vals:
            if alpha < 1 or alpha > half_C:
                continue
            
            # N1: items with weight > C - alpha (these each need their own bin at minimum, paired with small items)
            # N2: items with weight in (C/2, C - alpha] -- note C/2 < w <= C - alpha
            # N3: items with weight in [alpha, C/2] -- note alpha <= w <= C/2
            # But careful: standard L2 uses:
            # N1 = items with w > C - alpha
            # N2 = items with C - alpha >= w > C/2  (i.e., C/2 < w <= C - alpha)
            # N3 items: alpha <= w <= C/2
            
            n1_count = 0
            n1_remaining = 0  # remaining capacity in N1 bins after placing N1 items
            n2_count = 0
            sum_n3 = 0
            n3_count = 0
            
            for w in sorted_w:
                if w > C - alpha:
                    n1_count += 1
                    n1_remaining += C - w
                elif w > half_C:  # C/2 < w <= C - alpha
                    n2_count += 1
                elif w >= alpha:  # alpha <= w <= C/2
                    sum_n3 += w
                    n3_count += 1
            
            # Each N2 item needs its own bin (can't pair with N1 or other N2 since both > C/2)
            # N3 items can potentially fill remaining space in N1 bins and N2 bins
            # Remaining capacity in N2 bins for N3 items: each N2 bin has C - w_n2 >= alpha
            # But we simplify: remaining cap in N1 bins = n1_remaining
            # remaining cap in N2 bins for N3 items: sum over N2 items of (C - w) 
            # but we don't track individual N2 weights, so approximate:
            # Actually let's compute it properly
            n2_remaining = 0
            for w in sorted_w:
                if C - alpha >= w > half_C:
                    n2_remaining += C - w
            
            # N3 items that can't fit into remaining space of N1+N2 bins
            leftover_n3 = max(0, sum_n3 - n1_remaining - n2_remaining)
            extra_bins = (leftover_n3 + C - 1) // C if leftover_n3 > 0 else 0
            
            lb = n1_count + n2_count + extra_bins
            if lb > best_lb:
                best_lb = lb
        
        return best_lb
    
    lower_bound = compute_lower_bound()
    
    def remove_empty_bins(bins, bin_wts):
        new_bins = []
        new_wts = []
        for i in range(len(bins)):
            if bins[i]:
                new_bins.append(bins[i])
                new_wts.append(bin_wts[i])
        return new_bins, new_wts
    
    def solution_score(bins, bin_wts):
        num_bins = len(bins)
        if num_bins == 0:
            return (0, 0)
        fill_score = sum(w * w for w in bin_wts)
        return (-num_bins, fill_score)
    
    def copy_solution(bins, bin_wts):
        return [list(b) for b in bins], list(bin_wts)
    
    # Best Fit Decreasing
    def bfd(indices_sorted=None):
        if indices_sorted is None:
            indices_sorted = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        for idx in indices_sorted:
            w = weights[idx]
            best_bin = -1
            best_remaining = C + 1
            for b in range(len(bins)):
                rem = C - bin_wts[b]
                if w <= rem and rem < best_remaining:
                    best_remaining = rem
                    best_bin = b
            if best_bin >= 0:
                bins[best_bin].append(idx)
                bin_wts[best_bin] += w
            else:
                bins.append([idx])
                bin_wts.append(w)
        return bins, bin_wts
    
    # First Fit Decreasing
    def ffd(indices_sorted=None):
        if indices_sorted is None:
            indices_sorted = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        for idx in indices_sorted:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if bin_wts[b] + w <= C:
                    bins[b].append(idx)
                    bin_wts[b] += w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_wts.append(w)
        return bins, bin_wts
    
    # Randomized Best Fit Decreasing (GRASP-like)
    def randomized_bfd():
        indices_sorted = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        for idx in indices_sorted:
            w = weights[idx]
            # Find top 3 best-fit bins
            candidates = []
            for b in range(len(bins)):
                rem = C - bin_wts[b]
                if w <= rem:
                    candidates.append((rem, b))
            if candidates:
                candidates.sort()
                # Pick from top 3
                top = candidates[:3]
                _, chosen = random.choice(top)
                bins[chosen].append(idx)
                bin_wts[chosen] += w
            else:
                bins.append([idx])
                bin_wts.append(w)
        return bins, bin_wts
    
    # Best fit insertion of items into existing bins
    def best_fit_insert(items, bins, bin_wts):
        for item in items:
            w = weights[item]
            best_b = -1
            best_rem = C + 1
            for b in range(len(bins)):
                rem = C - bin_wts[b]
                if w <= rem and rem < best_rem:
                    best_rem = rem
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(item)
                bin_wts[best_b] += w
            else:
                bins.append([item])
                bin_wts.append(w)
    
    # Local Search
    def local_search(bins, bin_wts, time_budget=None):
        if time_budget is None:
            time_budget = time_remaining() * 0.5
        
        ls_deadline = time.time() + time_budget
        
        # Phase A: Bin Elimination
        improved = True
        op_count = 0
        while improved:
            if time.time() > ls_deadline:
                break
            improved = False
            
            order = sorted(range(len(bins)), key=lambda i: bin_wts[i])
            
            for src_idx in order:
                if time.time() > ls_deadline:
                    bins, bin_wts = remove_empty_bins(bins, bin_wts)
                    return bins, bin_wts
                if not bins[src_idx]:
                    continue
                
                items = sorted(bins[src_idx], key=lambda i: -weights[i])
                
                # Try to redistribute all items using best-fit
                placements = {}
                temp_wts = list(bin_wts)
                can_empty = True
                
                for item in items:
                    w = weights[item]
                    best_b = -1
                    best_rem = C + 1
                    for b in range(len(bins)):
                        if b == src_idx:
                            continue
                        rem = C - temp_wts[b]
                        if w <= rem and rem < best_rem:
                            best_rem = rem
                            best_b = b
                    if best_b >= 0:
                        placements[item] = best_b
                        temp_wts[best_b] += w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    for item, target in placements.items():
                        bins[target].append(item)
                        bin_wts[target] += weights[item]
                    bins[src_idx] = []
                    bin_wts[src_idx] = 0
                    improved = True
                    continue
                
                # Try with (1,1)-swaps for items that don't fit
                # Reset and try again with swap capability
                placements = {}
                temp_wts = list(bin_wts)
                temp_bins = [list(b) for b in bins]
                can_empty = True
                
                for item in items:
                    w = weights[item]
                    # First try direct placement
                    best_b = -1
                    best_rem = C + 1
                    for b in range(len(temp_bins)):
                        if b == src_idx:
                            continue
                        rem = C - temp_wts[b]
                        if w <= rem and rem < best_rem:
                            best_rem = rem
                            best_b = b
                    
                    if best_b >= 0:
                        placements[item] = ('move', best_b)
                        temp_wts[best_b] += w
                        temp_bins[best_b].append(item)
                    else:
                        # Try (1,1)-swap: find item y in bin b such that
                        # w - weights[y] fits in b (i.e., temp_wts[b] - weights[y] + w <= C)
                        # and y fits somewhere else
                        found_swap = False
                        swap_candidates = []
                        for b in range(len(temp_bins)):
                            if b == src_idx:
                                continue
                            for ypos, y in enumerate(temp_bins[b]):
                                wy = weights[y]
                                if temp_wts[b] - wy + w <= C and wy < w:
                                    # y is displaced, check if y fits somewhere
                                    for b2 in range(len(temp_bins)):
                                        if b2 == src_idx or b2 == b:
                                            continue
                                        if temp_wts[b2] + wy <= C:
                                            swap_candidates.append((b, ypos, y, b2))
                                            found_swap = True
                                            break
                                if found_swap:
                                    break
                            if found_swap:
                                break
                        
                        if found_swap and swap_candidates:
                            b, ypos, y, b2 = swap_candidates[0]
                            # Move y from b to b2
                            temp_bins[b].remove(y)
                            temp_wts[b] -= weights[y]
                            temp_bins[b2].append(y)
                            temp_wts[b2] += weights[y]
                            # Place item in b
                            temp_bins[b].append(item)
                            temp_wts[b] += w
                            placements[item] = ('swap_done', b, y, b2)
                        else:
                            can_empty = False
                            break
                
                if can_empty:
                    # Apply the solution from temp
                    bins = temp_bins
                    bin_wts = temp_wts
                    bins[src_idx] = []
                    bin_wts[src_idx] = 0
                    improved = True
            
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
        
        # Phase B: Consolidation (improve fill scores)
        phase_b_deadline = min(ls_deadline, time.time() + time_budget * 0.3)
        
        # Single item relocations
        move_improved = True
        move_rounds = 0
        while move_improved and move_rounds < 3:
            if time.time() > phase_b_deadline:
                break
            move_improved = False
            move_rounds += 1
            
            for src in range(len(bins)):
                if time.time() > phase_b_deadline:
                    break
                for item_pos in range(len(bins[src]) - 1, -1, -1):
                    item = bins[src][item_pos]
                    w = weights[item]
                    old_src_wt = bin_wts[src]
                    new_src_wt = old_src_wt - w
                    
                    best_target = -1
                    best_score_gain = 0
                    
                    for tgt in range(len(bins)):
                        if tgt == src:
                            continue
                        if bin_wts[tgt] + w > C:
                            continue
                        new_tgt_wt = bin_wts[tgt] + w
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
        
        # (1,1)-swaps for consolidation
        if time.time() < phase_b_deadline:
            swap_improved = True
            swap_rounds = 0
            while swap_improved and swap_rounds < 2:
                if time.time() > phase_b_deadline:
                    break
                swap_improved = False
                swap_rounds += 1
                
                for b1 in range(len(bins)):
                    if time.time() > phase_b_deadline:
                        break
                    for b2 in range(b1 + 1, len(bins)):
                        if time.time() > phase_b_deadline:
                            break
                        for i1 in range(len(bins[b1])):
                            found = False
                            for i2 in range(len(bins[b2])):
                                item1 = bins[b1][i1]
                                item2 = bins[b2][i2]
                                w1 = weights[item1]
                                w2 = weights[item2]
                                
                                new_wt1 = bin_wts[b1] - w1 + w2
                                new_wt2 = bin_wts[b2] - w2 + w1
                                
                                if new_wt1 > C or new_wt2 > C:
                                    continue
                                
                                old_score = bin_wts[b1]**2 + bin_wts[b2]**2
                                new_score = new_wt1**2 + new_wt2**2
                                
                                if new_score > old_score:
                                    bins[b1][i1] = item2
                                    bins[b2][i2] = item1
                                    bin_wts[b1] = new_wt1
                                    bin_wts[b2] = new_wt2
                                    swap_improved = True
                                    found = True
                                    break
                            if found:
                                break
        
        return bins, bin_wts
    
    # Shaking procedures
    def shake(bins, bin_wts, k):
        num_bins = len(bins)
        if num_bins <= 1:
            return bins, bin_wts
        
        bins = [list(b) for b in bins]
        bin_wts = list(bin_wts)
        
        if k == 1:
            # Move 1 random item to a random feasible bin
            src = random.randint(0, num_bins - 1)
            if not bins[src]:
                return remove_empty_bins(bins, bin_wts)
            item_pos = random.randint(0, len(bins[src]) - 1)
            item = bins[src][item_pos]
            w = weights[item]
            feasible = [b for b in range(num_bins) if b != src and bin_wts[b] + w <= C]
            if feasible:
                tgt = random.choice(feasible)
                bins[src].pop(item_pos)
                bin_wts[src] -= w
                bins[tgt].append(item)
                bin_wts[tgt] += w
        
        elif k == 2:
            # Swap 2 items between 2 bins
            b1, b2 = random.sample(range(num_bins), 2)
            if not bins[b1] or not bins[b2]:
                return remove_empty_bins(bins, bin_wts)
            i1 = random.randint(0, len(bins[b1]) - 1)
            i2 = random.randint(0, len(bins[b2]) - 1)
            item1, item2 = bins[b1][i1], bins[b2][i2]
            w1, w2 = weights[item1], weights[item2]
            new_wt1 = bin_wts[b1] - w1 + w2
            new_wt2 = bin_wts[b2] - w2 + w1
            if new_wt1 <= C and new_wt2 <= C:
                bins[b1][i1] = item2
                bins[b2][i2] = item1
                bin_wts[b1] = new_wt1
                bin_wts[b2] = new_wt2
        
        elif k == 3:
            # Chain move
            src = random.randint(0, num_bins - 1)
            if not bins[src]:
                return remove_empty_bins(bins, bin_wts)
            item_pos = random.randint(0, len(bins[src]) - 1)
            item = bins[src][item_pos]
            w = weights[item]
            feasible = [b for b in range(num_bins) if b != src and bin_wts[b] + w <= C]
            if feasible:
                mid = random.choice(feasible)
                bins[src].pop(item_pos)
                bin_wts[src] -= w
                bins[mid].append(item)
                bin_wts[mid] += w
                # Now move a random item from mid to another bin
                if bins[mid]:
                    item_pos2 = random.randint(0, len(bins[mid]) - 1)
                    item2 = bins[mid][item_pos2]
                    w2 = weights[item2]
                    feasible2 = [b for b in range(num_bins) if b != mid and bin_wts[b] + w2 <= C]
                    if feasible2:
                        tgt = random.choice(feasible2)
                        bins[mid].pop(item_pos2)
                        bin_wts[mid] -= w2
                        bins[tgt].append(item2)
                        bin_wts[tgt] += w2
        
        elif k == 4:
            # Empty 1 lightest bin, redistribute with randomized best-fit
            min_idx = min(range(num_bins), key=lambda i: bin_wts[i] if bins[i] else float('inf'))
            items_to_place = list(bins[min_idx])
            bins[min_idx] = []
            bin_wts[min_idx] = 0
            items_to_place.sort(key=lambda i: -weights[i])
            
            for item in items_to_place:
                w = weights[item]
                candidates = []
                for b in range(len(bins)):
                    rem = C - bin_wts[b]
                    if w <= rem:
                        candidates.append((rem, b))
                if candidates:
                    candidates.sort()
                    # Pick from top 2 (randomized best-fit)
                    top = candidates[:2]
                    _, chosen = random.choice(top)
                    bins[chosen].append(item)
                    bin_wts[chosen] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
        
        elif k == 5:
            # Empty 2 lightest bins, redistribute
            if num_bins < 3:
                return bins, bin_wts
            sorted_indices = sorted(range(num_bins), key=lambda i: bin_wts[i] if bins[i] else float('inf'))
            items_to_place = []
            for idx in sorted_indices[:2]:
                if bins[idx]:
                    items_to_place.extend(bins[idx])
                    bins[idx] = []
                    bin_wts[idx] = 0
            items_to_place.sort(key=lambda i: -weights[i])
            for item in items_to_place:
                w = weights[item]
                best_b = -1
                best_rem = C + 1
                for b in range(len(bins)):
                    rem = C - bin_wts[b]
                    if w <= rem and rem < best_rem:
                        best_rem = rem
                        best_b = b
                if best_b >= 0:
                    bins[best_b].append(item)
                    bin_wts[best_b] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
        
        elif k == 6:
            # Empty 3 lightest bins
            if num_bins < 4:
                return shake(bins, bin_wts, 5)
            sorted_indices = sorted(range(num_bins), key=lambda i: bin_wts[i] if bins[i] else float('inf'))
            items_to_place = []
            for idx in sorted_indices[:3]:
                if bins[idx]:
                    items_to_place.extend(bins[idx])
                    bins[idx] = []
                    bin_wts[idx] = 0
            items_to_place.sort(key=lambda i: -weights[i])
            for item in items_to_place:
                w = weights[item]
                best_b = -1
                best_rem = C + 1
                for b in range(len(bins)):
                    rem = C - bin_wts[b]
                    if w <= rem and rem < best_rem:
                        best_rem = rem
                        best_b = b
                if best_b >= 0:
                    bins[best_b].append(item)
                    bin_wts[best_b] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
        
        elif k == 7:
            # Pick 2 random bins, merge their items, try to repack into fewer bins
            if num_bins < 2:
                return bins, bin_wts
            b1, b2 = random.sample(range(num_bins), 2)
            merged_items = bins[b1] + bins[b2]
            bins[b1] = []
            bin_wts[b1] = 0
            bins[b2] = []
            bin_wts[b2] = 0
            merged_items.sort(key=lambda i: -weights[i])
            # Try to fit into existing bins first, then create new ones
            for item in merged_items:
                w = weights[item]
                best_b = -1
                best_rem = C + 1
                for b in range(len(bins)):
                    rem = C - bin_wts[b]
                    if w <= rem and rem < best_rem:
                        best_rem = rem
                        best_b = b
                if best_b >= 0:
                    bins[best_b].append(item)
                    bin_wts[best_b] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
        
        elif k == 8:
            # Destroy random 20% of bins, rebuild with FFD
            num_destroy = max(1, num_bins // 5)
            destroy_indices = random.sample(range(num_bins), min(num_destroy, num_bins))
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(bins[idx])
                bins[idx] = []
                bin_wts[idx] = 0
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            items_to_place.sort(key=lambda i: -weights[i])
            # FFD insert
            for item in items_to_place:
                w = weights[item]
                placed = False
                for b in range(len(bins)):
                    if bin_wts[b] + w <= C:
                        bins[b].append(item)
                        bin_wts[b] += w
                        placed = True
                        break
                if not placed:
                    bins.append([item])
                    bin_wts.append(w)
        
        elif k == 9:
            # Destroy random 30% of bins, rebuild with BFD
            num_destroy = max(1, int(num_bins * 0.3))
            destroy_indices = random.sample(range(num_bins), min(num_destroy, num_bins))
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(bins[idx])
                bins[idx] = []
                bin_wts[idx] = 0
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            items_to_place.sort(key=lambda i: -weights[i])
            best_fit_insert(items_to_place, bins, bin_wts)
        
        elif k >= 10:
            # Destroy 40% of bins, rebuild with randomized heuristic
            num_destroy = max(1, int(num_bins * 0.4))
            destroy_indices = random.sample(range(num_bins), min(num_destroy, num_bins))
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(bins[idx])
                bins[idx] = []
                bin_wts[idx] = 0
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            items_to_place.sort(key=lambda i: -weights[i])
            for item in items_to_place:
                w = weights[item]
                if random.random() < 0.5:
                    # Best fit
                    best_b = -1
                    best_rem = C + 1
                    for b in range(len(bins)):
                        rem = C - bin_wts[b]
                        if w <= rem and rem < best_rem:
                            best_rem = rem
                            best_b = b
                    if best_b >= 0:
                        bins[best_b].append(item)
                        bin_wts[best_b] += w
                    else:
                        bins.append([item])
                        bin_wts.append(w)
                else:
                    placed = False
                    for b in range(len(bins)):
                        if bin_wts[b] + w <= C:
                            bins[b].append(item)
                            bin_wts[b] += w
                            placed = True
                            break
                    if not placed:
                        bins.append([item])
                        bin_wts.append(w)
        
        bins, bin_wts = remove_empty_bins(bins, bin_wts)
        return bins, bin_wts
    
    # Targeted bin reduction with backtracking
    def targeted_reduction(bins, bin_wts, max_time=0.5):
        deadline = time.time() + max_time
        num_bins = len(bins)
        if num_bins <= lower_bound:
            return bins, bin_wts
        
        # Find lightest bin
        min_idx = min(range(num_bins), key=lambda i: bin_wts[i])
        items = sorted(bins[min_idx], key=lambda i: -weights[i])
        total_to_place = bin_wts[min_idx]
        
        # Check if total remaining capacity (excluding this bin) is enough
        total_remaining = sum(C - bin_wts[b] for b in range(num_bins) if b != min_idx)
        if total_remaining < total_to_place:
            return bins, bin_wts
        
        # Try redistribution with backtracking (limited)
        temp_wts = list(bin_wts)
        temp_bins_items = [list(b) for b in bins]  # track for potential swaps
        
        def try_place(item_idx, remaining_items, temp_w):
            if time.time() > deadline:
                return None
            if item_idx >= len(remaining_items):
                return {}  # success
            
            item = remaining_items[item_idx]
            w = weights[item]
            
            # Sort bins by remaining capacity (ascending, best-fit first)
            candidates = []
            for b in range(num_bins):
                if b == min_idx:
                    continue
                rem = C - temp_w[b]
                if w <= rem:
                    candidates.append((rem, b))
            candidates.sort()
            
            for rem, b in candidates[:5]:  # limit branching
                temp_w[b] += w
                result = try_place(item_idx + 1, remaining_items, temp_w)
                if result is not None:
                    result[item] = b
                    return result
                temp_w[b] -= w
            
            return None
        
        result = try_place(0, items, list(bin_wts))
        if result is not None:
            for item, target in result.items():
                bins[target].append(item)
                bin_wts[target] += weights[item]
            bins[min_idx] = []
            bin_wts[min_idx] = 0
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
        
        return bins, bin_wts
    
    # === Multiple Initial Solutions ===
    init_deadline = time.time() + time_limit * 0.10
    
    best_bins, best_wts = ffd()
    best_score = solution_score(best_bins, best_wts)
    
    candidates = []
    
    # BFD
    b2, w2 = bfd()
    candidates.append((b2, w2))
    
    # Randomized variants
    for _ in range(5):
        if time.time() > init_deadline:
            break
        br, wr = randomized_bfd()
        candidates.append((br, wr))
    
    # Quick local search on each candidate
    for cb, cw in candidates:
        if time.time() > init_deadline:
            break
        ls_budget = max(0.01, (init_deadline - time.time()) * 0.3)
        cb, cw = local_search(cb, cw, time_budget=ls_budget)
        sc = solution_score(cb, cw)
        if sc > best_score:
            best_bins, best_wts = cb, cw
            best_score = sc
    
    # Apply more thorough local search to the best initial solution
    ls_budget = max(0.01, min(time_remaining() * 0.15, init_deadline - time.time() + time_limit * 0.05))
    best_bins, best_wts = local_search(best_bins, best_wts, time_budget=ls_budget)
    best_score = solution_score(best_bins, best_wts)
    
    if len(best_bins) <= lower_bound:
        return {"packing": best_bins, "bin_weights": best_wts}
    
    # === VNS Main Loop ===
    k_max = 10
    if n > 500:
        k_max = 7
    
    current_bins, current_wts = copy_solution(best_bins, best_wts)
    current_score = best_score
    
    stagnation_count = 0
    vns_deadline = start_time + time_limit * 0.97
    iteration = 0
    targeted_counter = 0
    
    while time.time() < vns_deadline:
        k = 1
        while k <= k_max and time.time() < vns_deadline:
            iteration += 1
            targeted_counter += 1
            
            # Shaking
            shaken_bins, shaken_wts = shake(
                current_bins, current_wts, k
            )
            
            # Local search with adaptive time budget
            remaining = vns_deadline - time.time()
            if remaining <= 0:
                break
            ls_time = min(remaining * 0.3, max(0.01, remaining / max(1, k_max - k + 1)))
            
            ls_bins, ls_wts = local_search(shaken_bins, shaken_wts, time_budget=ls_time)
            ls_score = solution_score(ls_bins, ls_wts)
            
            if ls_score > current_score:
                current_bins, current_wts = ls_bins, ls_wts
                current_score = ls_score
                k = 1
                
                if current_score > best_score:
                    best_bins, best_wts = copy_solution(current_bins, current_wts)
                    best_score = current_score
                    stagnation_count = 0
                    
                    if len(best_bins) <= lower_bound:
                        return {"packing": best_bins, "bin_weights": best_wts}
            else:
                k += 1
            
            # Periodically try targeted reduction
            if targeted_counter >= 5 and time.time() < vns_deadline:
                targeted_counter = 0
                tr_time = min(0.5, (vns_deadline - time.time()) * 0.1)
                tr_bins, tr_wts = targeted_reduction(
                    [list(b) for b in current_bins], list(current_wts), max_time=tr_time
                )
                tr_score = solution_score(tr_bins, tr_wts)
                if tr_score > current_score:
                    current_bins, current_wts = tr_bins, tr_wts
                    current_score = tr_score
                    k = 1
                    if current_score > best_score:
                        best_bins, best_wts = copy_solution(current_bins, current_wts)
                        best_score = current_score
                        stagnation_count = 0
                        if len(best_bins) <= lower_bound:
                            return {"packing": best_bins, "bin_weights": best_wts}
        
        # Restart from best with large perturbation
        stagnation_count += 1
        current_bins, current_wts = copy_solution(best_bins, best_wts)
        current_score = best_score
        
        num_bins_curr = len(current_bins)
        if num_bins_curr > 2:
            # Adaptive destruction percentage based on stagnation
            destroy_pct = min(0.5, 0.25 + stagnation_count * 0.05)
            num_destroy = max(2, int(num_bins_curr * destroy_pct))
            num_destroy = min(num_destroy, num_bins_curr - 1)
            destroy_indices = random.sample(range(num_bins_curr), num_destroy)
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(current_bins[idx])
                current_bins[idx] = []
                current_wts[idx] = 0
            
            current_bins, current_wts = remove_empty_bins(current_bins, current_wts)
            items_to_place.sort(key=lambda i: -weights[i])
            
            # Rebuild with randomized best-fit
            for item in items_to_place:
                w = weights[item]
                candidates = []
                for b in range(len(current_bins)):
                    rem = C - current_wts[b]
                    if w <= rem:
                        candidates.append((rem, b))
                if candidates:
                    candidates.sort()
                    top = candidates[:max(1, min(3, len(candidates)))]
                    _, chosen = random.choice(top)
                    current_bins[chosen].append(item)
                    current_wts[chosen] += w
                else:
                    current_bins.append([item])
                    current_wts.append(w)
            
            current_score = solution_score(current_bins, current_wts)
    
    return {
        "packing": best_bins,
        "bin_weights": best_wts
    }