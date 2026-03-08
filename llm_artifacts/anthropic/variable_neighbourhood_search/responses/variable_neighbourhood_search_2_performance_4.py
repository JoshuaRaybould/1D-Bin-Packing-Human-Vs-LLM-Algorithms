import random
import time
from bisect import bisect_left, insort

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
    
    # L2 Lower Bound
    def compute_lower_bound():
        simple_lb = (total_weight + C - 1) // C
        if n == 0:
            return simple_lb
        
        sorted_w = sorted(weights, reverse=True)
        half_C = C // 2
        
        best_lb = simple_lb
        
        distinct_vals = set()
        for w in sorted_w:
            if w > C:
                continue
            a = C - w
            if 1 <= a <= half_C:
                distinct_vals.add(a)
            if 1 <= w <= half_C:
                distinct_vals.add(w)
        distinct_vals.add(1)
        if half_C >= 1:
            distinct_vals.add(half_C)
        
        for alpha in distinct_vals:
            if alpha < 1 or alpha > half_C:
                continue
            
            n1_count = 0
            n1_remaining = 0
            n2_count = 0
            n2_remaining = 0
            sum_n3 = 0
            
            for w in sorted_w:
                if w > C - alpha:
                    n1_count += 1
                    n1_remaining += C - w
                elif w > half_C:
                    n2_count += 1
                    n2_remaining += C - w
                elif w >= alpha:
                    sum_n3 += w
            
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
    
    # Best Fit Decreasing with bisect for speed
    def bfd(indices_sorted=None):
        if indices_sorted is None:
            indices_sorted = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        # Track remaining capacities with sorted list for bisect
        # Each entry: (remaining_cap, bin_index)
        remaining_sorted = []  # sorted by remaining capacity
        
        for idx in indices_sorted:
            w = weights[idx]
            # Find bin with smallest remaining capacity >= w
            pos = bisect_left(remaining_sorted, (w,))
            if pos < len(remaining_sorted):
                rem, b = remaining_sorted[pos]
                # Remove old entry
                remaining_sorted.pop(pos)
                bins[b].append(idx)
                bin_wts[b] += w
                new_rem = rem - w
                if new_rem > 0:
                    insort(remaining_sorted, (new_rem, b))
            else:
                b = len(bins)
                bins.append([idx])
                bin_wts.append(w)
                new_rem = C - w
                if new_rem > 0:
                    insort(remaining_sorted, (new_rem, b))
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
    
    # Best fit insert items into existing bins using bisect
    def best_fit_insert_fast(items, bins, bin_wts):
        # Build sorted remaining list
        remaining_sorted = []
        for b in range(len(bins)):
            rem = C - bin_wts[b]
            if rem > 0:
                remaining_sorted.append((rem, b))
        remaining_sorted.sort()
        
        for item in items:
            w = weights[item]
            pos = bisect_left(remaining_sorted, (w,))
            if pos < len(remaining_sorted):
                rem, b = remaining_sorted[pos]
                remaining_sorted.pop(pos)
                bins[b].append(item)
                bin_wts[b] += w
                new_rem = rem - w
                if new_rem > 0:
                    insort(remaining_sorted, (new_rem, b))
            else:
                b = len(bins)
                bins.append([item])
                bin_wts.append(w)
                new_rem = C - w
                if new_rem > 0:
                    insort(remaining_sorted, (new_rem, b))
    
    # Randomized BFD
    def randomized_bfd():
        indices_sorted = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        remaining_sorted = []
        
        for idx in indices_sorted:
            w = weights[idx]
            pos = bisect_left(remaining_sorted, (w,))
            if pos < len(remaining_sorted):
                # Pick from top 3 best-fit candidates
                end = min(pos + 3, len(remaining_sorted))
                choice_idx = random.randint(pos, end - 1)
                rem, b = remaining_sorted[choice_idx]
                remaining_sorted.pop(choice_idx)
                bins[b].append(idx)
                bin_wts[b] += w
                new_rem = rem - w
                if new_rem > 0:
                    insort(remaining_sorted, (new_rem, b))
            else:
                b = len(bins)
                bins.append([idx])
                bin_wts.append(w)
                new_rem = C - w
                if new_rem > 0:
                    insort(remaining_sorted, (new_rem, b))
        return bins, bin_wts
    
    # Local Search
    def local_search(bins, bin_wts, time_budget=None):
        if time_budget is None:
            time_budget = time_remaining() * 0.5
        
        ls_deadline = time.time() + time_budget
        
        # Phase A: Bin Elimination - try to empty lightest bins
        improved = True
        while improved:
            if time.time() > ls_deadline:
                break
            improved = False
            
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            num_bins = len(bins)
            if num_bins <= lower_bound:
                return bins, bin_wts
            
            order = sorted(range(num_bins), key=lambda i: bin_wts[i])
            
            for src_idx in order:
                if time.time() > ls_deadline:
                    return remove_empty_bins(bins, bin_wts)
                if not bins[src_idx]:
                    continue
                
                items = sorted(bins[src_idx], key=lambda i: -weights[i])
                
                # Check feasibility: total remaining capacity in other bins
                total_rem = sum(C - bin_wts[b] for b in range(num_bins) if b != src_idx and bins[b])
                if total_rem < bin_wts[src_idx]:
                    continue
                
                # Try to redistribute all items using best-fit
                temp_wts = list(bin_wts)
                placements = {}
                can_empty = True
                
                for item in items:
                    w = weights[item]
                    best_b = -1
                    best_rem = C + 1
                    for b in range(num_bins):
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
                
                # Try with swaps: for each item that doesn't fit directly,
                # try swapping with a smaller item in another bin
                temp_wts = list(bin_wts)
                temp_bins = [list(b) for b in bins]
                can_empty = True
                
                for item in items:
                    w = weights[item]
                    best_b = -1
                    best_rem = C + 1
                    for b in range(num_bins):
                        if b == src_idx:
                            continue
                        rem = C - temp_wts[b]
                        if w <= rem and rem < best_rem:
                            best_rem = rem
                            best_b = b
                    
                    if best_b >= 0:
                        temp_wts[best_b] += w
                        temp_bins[best_b].append(item)
                    else:
                        found_swap = False
                        # Try finding a swap: item y in bin b where swapping allows placement
                        best_swap = None
                        best_swap_waste = C + 1
                        
                        for b in range(num_bins):
                            if b == src_idx:
                                continue
                            for ypos, y in enumerate(temp_bins[b]):
                                wy = weights[y]
                                if wy >= w:
                                    continue
                                if temp_wts[b] - wy + w > C:
                                    continue
                                # y needs a new home
                                for b2 in range(num_bins):
                                    if b2 == src_idx or b2 == b:
                                        continue
                                    if temp_wts[b2] + wy <= C:
                                        waste = (C - temp_wts[b] + wy - w) + (C - temp_wts[b2] - wy)
                                        if waste < best_swap_waste:
                                            best_swap_waste = waste
                                            best_swap = (b, ypos, y, b2)
                                            found_swap = True
                                        break
                                if found_swap:
                                    break
                            if found_swap:
                                break
                        
                        if found_swap and best_swap:
                            b, ypos, y, b2 = best_swap
                            temp_bins[b].remove(y)
                            temp_wts[b] -= weights[y]
                            temp_bins[b2].append(y)
                            temp_wts[b2] += weights[y]
                            temp_bins[b].append(item)
                            temp_wts[b] += w
                        else:
                            can_empty = False
                            break
                
                if can_empty:
                    bins = temp_bins
                    bin_wts = temp_wts
                    bins[src_idx] = []
                    bin_wts[src_idx] = 0
                    improved = True
            
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
        
        # Phase B: Consolidation moves
        phase_b_deadline = min(ls_deadline, time.time() + time_budget * 0.3)
        
        # Single item relocations to improve fill score
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
                                
                                old_score = bin_wts[b1] * bin_wts[b1] + bin_wts[b2] * bin_wts[b2]
                                new_score = new_wt1 * new_wt1 + new_wt2 * new_wt2
                                
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
    
    # Shaking procedures for VNS
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
            # Empty lightest bin and redistribute with BFD
            min_idx = min(range(num_bins), key=lambda i: bin_wts[i] if bins[i] else float('inf'))
            items_to_place = list(bins[min_idx])
            bins[min_idx] = []
            bin_wts[min_idx] = 0
            items_to_place.sort(key=lambda i: -weights[i])
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            best_fit_insert_fast(items_to_place, bins, bin_wts)
        
        elif k == 4:
            # Empty 2 lightest bins
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
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            best_fit_insert_fast(items_to_place, bins, bin_wts)
        
        elif k == 5:
            # Empty 3 lightest bins
            if num_bins < 4:
                return shake(bins, bin_wts, 4)
            sorted_indices = sorted(range(num_bins), key=lambda i: bin_wts[i] if bins[i] else float('inf'))
            items_to_place = []
            for idx in sorted_indices[:3]:
                if bins[idx]:
                    items_to_place.extend(bins[idx])
                    bins[idx] = []
                    bin_wts[idx] = 0
            items_to_place.sort(key=lambda i: -weights[i])
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            best_fit_insert_fast(items_to_place, bins, bin_wts)
        
        elif k == 6:
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
        
        elif k == 7:
            # Pick 2 random bins, merge and repack
            if num_bins < 2:
                return bins, bin_wts
            b1, b2 = random.sample(range(num_bins), 2)
            merged_items = bins[b1] + bins[b2]
            bins[b1] = []
            bin_wts[b1] = 0
            bins[b2] = []
            bin_wts[b2] = 0
            merged_items.sort(key=lambda i: -weights[i])
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            best_fit_insert_fast(merged_items, bins, bin_wts)
        
        elif k == 8:
            # Destroy random 20% of bins, rebuild with BFD
            num_destroy = max(1, num_bins // 5)
            destroy_indices = random.sample(range(num_bins), min(num_destroy, num_bins))
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(bins[idx])
                bins[idx] = []
                bin_wts[idx] = 0
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            items_to_place.sort(key=lambda i: -weights[i])
            best_fit_insert_fast(items_to_place, bins, bin_wts)
        
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
            best_fit_insert_fast(items_to_place, bins, bin_wts)
        
        elif k >= 10:
            # Destroy 40% of bins, rebuild
            num_destroy = max(1, int(num_bins * 0.4))
            destroy_indices = random.sample(range(num_bins), min(num_destroy, num_bins))
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(bins[idx])
                bins[idx] = []
                bin_wts[idx] = 0
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            items_to_place.sort(key=lambda i: -weights[i])
            # Randomized best fit
            remaining_sorted = []
            for b in range(len(bins)):
                rem = C - bin_wts[b]
                if rem > 0:
                    remaining_sorted.append((rem, b))
            remaining_sorted.sort()
            
            for item in items_to_place:
                w = weights[item]
                pos = bisect_left(remaining_sorted, (w,))
                if pos < len(remaining_sorted):
                    end = min(pos + 3, len(remaining_sorted))
                    choice_idx = random.randint(pos, end - 1)
                    rem, b = remaining_sorted[choice_idx]
                    remaining_sorted.pop(choice_idx)
                    bins[b].append(item)
                    bin_wts[b] += w
                    new_rem = rem - w
                    if new_rem > 0:
                        insort(remaining_sorted, (new_rem, b))
                else:
                    b = len(bins)
                    bins.append([item])
                    bin_wts.append(w)
                    new_rem = C - w
                    if new_rem > 0:
                        insort(remaining_sorted, (new_rem, b))
        
        bins, bin_wts = remove_empty_bins(bins, bin_wts)
        return bins, bin_wts
    
    # Targeted bin reduction with deeper backtracking
    def targeted_reduction(bins, bin_wts, max_time=1.0):
        deadline = time.time() + max_time
        num_bins = len(bins)
        if num_bins <= lower_bound:
            return bins, bin_wts
        
        # Try multiple lightest bins
        sorted_by_weight = sorted(range(num_bins), key=lambda i: bin_wts[i])
        
        for attempt_idx in sorted_by_weight[:3]:
            if time.time() > deadline:
                break
            
            min_idx = attempt_idx
            items = sorted(bins[min_idx], key=lambda i: -weights[i])
            total_to_place = bin_wts[min_idx]
            
            total_remaining = sum(C - bin_wts[b] for b in range(num_bins) if b != min_idx and bins[b])
            if total_remaining < total_to_place:
                continue
            
            nodes_visited = [0]
            max_nodes = 5000
            
            def try_place(item_idx, temp_w):
                if time.time() > deadline:
                    return None
                nodes_visited[0] += 1
                if nodes_visited[0] > max_nodes:
                    return None
                if item_idx >= len(items):
                    return {}
                
                item = items[item_idx]
                w = weights[item]
                
                candidates = []
                for b in range(num_bins):
                    if b == min_idx:
                        continue
                    rem = C - temp_w[b]
                    if w <= rem:
                        candidates.append((rem, b))
                candidates.sort()
                
                for rem, b in candidates[:8]:
                    temp_w[b] += w
                    result = try_place(item_idx + 1, temp_w)
                    if result is not None:
                        result[item] = b
                        return result
                    temp_w[b] -= w
                
                return None
            
            nodes_visited[0] = 0
            result = try_place(0, list(bin_wts))
            if result is not None:
                for item, target in result.items():
                    bins[target].append(item)
                    bin_wts[target] += weights[item]
                bins[min_idx] = []
                bin_wts[min_idx] = 0
                bins, bin_wts = remove_empty_bins(bins, bin_wts)
                num_bins = len(bins)
                if num_bins <= lower_bound:
                    return bins, bin_wts
                # Update sorted indices for next attempt
                sorted_by_weight = sorted(range(num_bins), key=lambda i: bin_wts[i])
        
        return bins, bin_wts
    
    # === Multiple Initial Solutions ===
    init_deadline = time.time() + time_limit * 0.08
    
    best_bins, best_wts = bfd()
    best_score = solution_score(best_bins, best_wts)
    
    # FFD
    b2, w2 = ffd()
    sc2 = solution_score(b2, w2)
    if sc2 > best_score:
        best_bins, best_wts = b2, w2
        best_score = sc2
    
    # Randomized variants
    for _ in range(8):
        if time.time() > init_deadline:
            break
        br, wr = randomized_bfd()
        scr = solution_score(br, wr)
        if scr > best_score:
            best_bins, best_wts = br, wr
            best_score = scr
    
    # Apply local search to best initial solution
    ls_budget = max(0.05, min(time_remaining() * 0.15, 3.0))
    best_bins, best_wts = local_search(best_bins, best_wts, time_budget=ls_budget)
    best_score = solution_score(best_bins, best_wts)
    
    if len(best_bins) <= lower_bound:
        return {"packing": best_bins, "bin_weights": best_wts}
    
    # Try targeted reduction on initial solution
    tr_time = min(2.0, time_remaining() * 0.1)
    best_bins, best_wts = targeted_reduction(best_bins, best_wts, max_time=tr_time)
    best_score = solution_score(best_bins, best_wts)
    
    if len(best_bins) <= lower_bound:
        return {"packing": best_bins, "bin_weights": best_wts}
    
    # === VNS Main Loop ===
    k_max = 10
    if n > 500:
        k_max = 8
    
    current_bins, current_wts = copy_solution(best_bins, best_wts)
    current_score = best_score
    
    stagnation_count = 0
    vns_deadline = start_time + time_limit * 0.98
    iteration = 0
    targeted_counter = 0
    
    while time.time() < vns_deadline:
        k = 1
        while k <= k_max and time.time() < vns_deadline:
            iteration += 1
            targeted_counter += 1
            
            # Shaking
            shaken_bins, shaken_wts = shake(current_bins, current_wts, k)
            
            # Local search with adaptive time budget
            remaining = vns_deadline - time.time()
            if remaining <= 0:
                break
            ls_time = min(remaining * 0.25, max(0.02, remaining / max(1, (k_max - k + 1) * 2)))
            
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
                tr_time = min(1.0, (vns_deadline - time.time()) * 0.08)
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
            # Adaptive destruction
            destroy_pct = min(0.5, 0.2 + stagnation_count * 0.05)
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
            
            # Rebuild with randomized best-fit using bisect
            remaining_sorted = []
            for b in range(len(current_bins)):
                rem = C - current_wts[b]
                if rem > 0:
                    remaining_sorted.append((rem, b))
            remaining_sorted.sort()
            
            for item in items_to_place:
                w = weights[item]
                pos = bisect_left(remaining_sorted, (w,))
                if pos < len(remaining_sorted):
                    end = min(pos + 3, len(remaining_sorted))
                    choice_idx = random.randint(pos, end - 1)
                    rem, b = remaining_sorted[choice_idx]
                    remaining_sorted.pop(choice_idx)
                    current_bins[b].append(item)
                    current_wts[b] += w
                    new_rem = rem - w
                    if new_rem > 0:
                        insort(remaining_sorted, (new_rem, b))
                else:
                    b = len(current_bins)
                    current_bins.append([item])
                    current_wts.append(w)
                    new_rem = C - w
                    if new_rem > 0:
                        insort(remaining_sorted, (new_rem, b))
            
            current_score = solution_score(current_bins, current_wts)
    
    return {
        "packing": best_bins,
        "bin_weights": best_wts
    }