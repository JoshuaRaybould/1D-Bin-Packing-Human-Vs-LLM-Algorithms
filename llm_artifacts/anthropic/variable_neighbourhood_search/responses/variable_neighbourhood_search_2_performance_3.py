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
    
    # L2 Lower Bound
    def compute_lower_bound():
        simple_lb = (total_weight + C - 1) // C
        if n == 0:
            return simple_lb
        
        sorted_w = sorted(weights, reverse=True)
        half_C = C // 2
        
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
        
        best_lb = simple_lb
        
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
    
    # Best Fit Decreasing using bisect for speed
    def bfd_fast(item_indices=None):
        if item_indices is None:
            item_indices = sorted(range(n), key=lambda i: -weights[i])
        # Maintain sorted list of (remaining_capacity, bin_index)
        bins = []
        bin_wts = []
        # We'll use a simple list sorted by remaining capacity
        # For each item, find the bin with smallest remaining capacity >= w
        remainders = []  # sorted list of (remaining, bin_idx)
        
        for idx in item_indices:
            w = weights[idx]
            # Binary search for smallest remaining >= w
            pos = bisect.bisect_left(remainders, (w,))
            if pos < len(remainders):
                rem, b = remainders[pos]
                # Remove this entry
                remainders.pop(pos)
                bins[b].append(idx)
                bin_wts[b] += w
                new_rem = rem - w
                if new_rem > 0:
                    bisect.insort(remainders, (new_rem, b))
            else:
                b = len(bins)
                bins.append([idx])
                bin_wts.append(w)
                new_rem = C - w
                if new_rem > 0:
                    bisect.insort(remainders, (new_rem, b))
        return bins, bin_wts
    
    def ffd_fast(item_indices=None):
        if item_indices is None:
            item_indices = sorted(range(n), key=lambda i: -weights[i])
        bins = []
        bin_wts = []
        # For FFD we need first fit, so we track bins by index order
        # Use a sorted structure of (remaining, bin_index) but pick first (smallest index) among feasible
        # Actually FFD: place in first bin that fits
        # For speed, maintain list of remaining capacities
        remainders = []
        for idx in item_indices:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if remainders[b] >= w:
                    bins[b].append(idx)
                    bin_wts[b] += w
                    remainders[b] -= w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_wts.append(w)
                remainders.append(C - w)
        return bins, bin_wts
    
    def best_fit_insert_fast(items, bins, bin_wts):
        """Insert items into bins using best-fit with bisect."""
        # Build sorted remainders
        remainders = []  # (remaining, bin_idx)
        for b in range(len(bins)):
            rem = C - bin_wts[b]
            if rem > 0:
                remainders.append((rem, b))
        remainders.sort()
        
        for item in items:
            w = weights[item]
            pos = bisect.bisect_left(remainders, (w,))
            if pos < len(remainders):
                rem, b = remainders[pos]
                remainders.pop(pos)
                bins[b].append(item)
                bin_wts[b] += w
                new_rem = rem - w
                if new_rem > 0:
                    bisect.insort(remainders, (new_rem, b))
            else:
                b = len(bins)
                bins.append([item])
                bin_wts.append(w)
                new_rem = C - w
                if new_rem > 0:
                    bisect.insort(remainders, (new_rem, b))
    
    # Local Search
    def local_search(bins, bin_wts, time_budget=None):
        if time_budget is None:
            time_budget = time_remaining() * 0.5
        
        ls_deadline = time.time() + time_budget
        
        # Build item-to-bin mapping
        item_bin = [0] * n
        for b in range(len(bins)):
            for item in bins[b]:
                item_bin[item] = b
        
        # Phase A: Bin Elimination
        improved = True
        while improved:
            if time.time() > ls_deadline:
                break
            improved = False
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            # Rebuild item_bin
            for b in range(len(bins)):
                for item in bins[b]:
                    item_bin[item] = b
            
            num_bins = len(bins)
            order = sorted(range(num_bins), key=lambda i: bin_wts[i])
            
            for src_idx in order:
                if time.time() > ls_deadline:
                    return remove_empty_bins(bins, bin_wts)
                if not bins[src_idx]:
                    continue
                
                items = sorted(bins[src_idx], key=lambda i: -weights[i])
                
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
                        item_bin[item] = target
                    bins[src_idx] = []
                    bin_wts[src_idx] = 0
                    improved = True
                    continue
                
                # Try with swaps for items that don't fit
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
                        for b in range(num_bins):
                            if b == src_idx:
                                continue
                            if found_swap:
                                break
                            for ypos, y in enumerate(temp_bins[b]):
                                wy = weights[y]
                                if temp_wts[b] - wy + w <= C and wy < w:
                                    for b2 in range(num_bins):
                                        if b2 == src_idx or b2 == b:
                                            continue
                                        if temp_wts[b2] + wy <= C:
                                            temp_bins[b].remove(y)
                                            temp_wts[b] -= wy
                                            temp_bins[b2].append(y)
                                            temp_wts[b2] += wy
                                            temp_bins[b].append(item)
                                            temp_wts[b] += w
                                            found_swap = True
                                            break
                                if found_swap:
                                    break
                        
                        if not found_swap:
                            can_empty = False
                            break
                
                if can_empty:
                    bins = temp_bins
                    bin_wts = temp_wts
                    bins[src_idx] = []
                    bin_wts[src_idx] = 0
                    for b in range(len(bins)):
                        for item in bins[b]:
                            item_bin[item] = b
                    improved = True
            
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
        
        # Phase B: Consolidation moves
        phase_b_deadline = min(ls_deadline, time.time() + time_budget * 0.4)
        
        # Rebuild item_bin
        for b in range(len(bins)):
            for item in bins[b]:
                item_bin[item] = b
        
        # Single item relocations
        move_improved = True
        move_rounds = 0
        while move_improved and move_rounds < 5:
            if time.time() > phase_b_deadline:
                break
            move_improved = False
            move_rounds += 1
            
            for src in range(len(bins)):
                if time.time() > phase_b_deadline:
                    break
                for item_pos in range(len(bins[src]) - 1, -1, -1):
                    if item_pos >= len(bins[src]):
                        continue
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
                        item_bin[item] = best_target
                        move_improved = True
            
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            for b in range(len(bins)):
                for item in bins[b]:
                    item_bin[item] = b
        
        # (1,1)-swaps
        if time.time() < phase_b_deadline:
            swap_improved = True
            swap_rounds = 0
            while swap_improved and swap_rounds < 3:
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
                                    item_bin[item1] = b2
                                    item_bin[item2] = b1
                                    swap_improved = True
                                    found = True
                                    break
                            if found:
                                break
        
        # (2,1) moves: move 2 items from src, 1 item back
        if time.time() < ls_deadline:
            phase_c_deadline = min(ls_deadline, time.time() + time_budget * 0.2)
            for b1 in range(len(bins)):
                if time.time() > phase_c_deadline:
                    break
                if len(bins[b1]) < 2:
                    continue
                for i1a in range(len(bins[b1])):
                    if time.time() > phase_c_deadline:
                        break
                    for i1b in range(i1a + 1, len(bins[b1])):
                        item_a = bins[b1][i1a]
                        item_b_val = bins[b1][i1b]
                        wa = weights[item_a]
                        wb = weights[item_b_val]
                        
                        for b2 in range(len(bins)):
                            if b2 == b1:
                                continue
                            # Move items a,b from b1 to b2, move item c from b2 to b1
                            for i2 in range(len(bins[b2])):
                                item_c = bins[b2][i2]
                                wc = weights[item_c]
                                
                                new_wt1 = bin_wts[b1] - wa - wb + wc
                                new_wt2 = bin_wts[b2] + wa + wb - wc
                                
                                if new_wt1 > C or new_wt2 > C or new_wt1 < 0 or new_wt2 < 0:
                                    continue
                                
                                old_score = bin_wts[b1] * bin_wts[b1] + bin_wts[b2] * bin_wts[b2]
                                new_score = new_wt1 * new_wt1 + new_wt2 * new_wt2
                                
                                if new_score > old_score:
                                    # Execute the move
                                    # Remove items from b1 (careful with indices)
                                    bins[b2].append(item_a)
                                    bins[b2].append(item_b_val)
                                    bins[b2].remove(item_c)
                                    bins[b1].append(item_c)
                                    # Remove in reverse order to keep indices valid
                                    if i1b > i1a:
                                        bins[b1].pop(i1b)
                                        bins[b1].pop(i1a)
                                    else:
                                        bins[b1].pop(i1a)
                                        bins[b1].pop(i1b)
                                    
                                    bin_wts[b1] = new_wt1
                                    bin_wts[b2] = new_wt2
                                    item_bin[item_a] = b2
                                    item_bin[item_b_val] = b2
                                    item_bin[item_c] = b1
                                    break
                            break
                    break
        
        bins, bin_wts = remove_empty_bins(bins, bin_wts)
        return bins, bin_wts
    
    # Shaking procedures
    def shake(bins, bin_wts, k):
        num_bins = len(bins)
        if num_bins <= 1:
            return bins, bin_wts
        
        bins = [list(b) for b in bins]
        bin_wts = list(bin_wts)
        
        if k == 1:
            # Move 1 random item
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
            # Swap 2 items
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
            # Move items from 2 random bins to other bins
            count = min(3, num_bins)
            selected = random.sample(range(num_bins), count)
            items_to_move = []
            for s in selected[:2]:
                # Move 1-2 random items from each
                if bins[s]:
                    nm = min(len(bins[s]), random.randint(1, 2))
                    for _ in range(nm):
                        if bins[s]:
                            pos = random.randint(0, len(bins[s]) - 1)
                            item = bins[s][pos]
                            items_to_move.append(item)
                            bins[s].pop(pos)
                            bin_wts[s] -= weights[item]
            
            items_to_move.sort(key=lambda i: -weights[i])
            best_fit_insert_fast(items_to_move, bins, bin_wts)
        
        elif k == 4:
            # Empty 1 lightest bin
            min_idx = min(range(num_bins), key=lambda i: bin_wts[i] if bins[i] else float('inf'))
            items_to_place = list(bins[min_idx])
            bins[min_idx] = []
            bin_wts[min_idx] = 0
            items_to_place.sort(key=lambda i: -weights[i])
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            best_fit_insert_fast(items_to_place, bins, bin_wts)
        
        elif k == 5:
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
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            best_fit_insert_fast(items_to_place, bins, bin_wts)
        
        elif k == 7:
            # Merge 2 random bins
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
            # Destroy 20% of bins
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
            # Destroy 30% of bins
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
        
        elif k == 10:
            # Destroy 40% of bins
            num_destroy = max(1, int(num_bins * 0.4))
            destroy_indices = random.sample(range(num_bins), min(num_destroy, num_bins))
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(bins[idx])
                bins[idx] = []
                bin_wts[idx] = 0
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            items_to_place.sort(key=lambda i: -weights[i])
            # Use randomized best fit
            for item in items_to_place:
                w = weights[item]
                candidates = []
                for b in range(len(bins)):
                    rem = C - bin_wts[b]
                    if w <= rem:
                        candidates.append((rem, b))
                if candidates:
                    candidates.sort()
                    top = candidates[:3]
                    _, chosen = random.choice(top)
                    bins[chosen].append(item)
                    bin_wts[chosen] += w
                else:
                    bins.append([item])
                    bin_wts.append(w)
        
        elif k >= 11:
            # Destroy 50% with randomized rebuild
            num_destroy = max(1, int(num_bins * 0.5))
            destroy_indices = random.sample(range(num_bins), min(num_destroy, num_bins))
            items_to_place = []
            for idx in destroy_indices:
                items_to_place.extend(bins[idx])
                bins[idx] = []
                bin_wts[idx] = 0
            bins, bin_wts = remove_empty_bins(bins, bin_wts)
            items_to_place.sort(key=lambda i: -weights[i])
            best_fit_insert_fast(items_to_place, bins, bin_wts)
        
        bins, bin_wts = remove_empty_bins(bins, bin_wts)
        return bins, bin_wts
    
    # Targeted bin reduction with deeper backtracking
    def targeted_reduction(bins, bin_wts, max_time=0.5):
        deadline = time.time() + max_time
        num_bins = len(bins)
        if num_bins <= lower_bound:
            return bins, bin_wts
        
        # Try multiple lightest bins
        sorted_by_weight = sorted(range(num_bins), key=lambda i: bin_wts[i])
        
        for try_idx in sorted_by_weight[:3]:
            if time.time() > deadline:
                break
            min_idx = try_idx
            items = sorted(bins[min_idx], key=lambda i: -weights[i])
            total_to_place = bin_wts[min_idx]
            
            total_remaining = sum(C - bin_wts[b] for b in range(num_bins) if b != min_idx)
            if total_remaining < total_to_place:
                continue
            
            def try_place(item_idx, remaining_items, temp_w):
                if time.time() > deadline:
                    return None
                if item_idx >= len(remaining_items):
                    return {}
                
                item = remaining_items[item_idx]
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
                num_bins = len(bins)
                if num_bins <= lower_bound:
                    return bins, bin_wts
                # Update sorted indices for next iteration
                sorted_by_weight = sorted(range(num_bins), key=lambda i: bin_wts[i])
        
        return bins, bin_wts
    
    # === Multiple Initial Solutions ===
    init_deadline = time.time() + time_limit * 0.08
    
    best_bins, best_wts = bfd_fast()
    best_score = solution_score(best_bins, best_wts)
    
    # FFD
    b2, w2 = ffd_fast()
    sc2 = solution_score(b2, w2)
    if sc2 > best_score:
        best_bins, best_wts = b2, w2
        best_score = sc2
    
    # Randomized BFD variants
    for _ in range(8):
        if time.time() > init_deadline:
            break
        indices = sorted(range(n), key=lambda i: -weights[i])
        # Add small perturbation to ordering
        for j in range(len(indices) - 1):
            if random.random() < 0.1 and weights[indices[j]] == weights[indices[j+1]]:
                indices[j], indices[j+1] = indices[j+1], indices[j]
        br, wr = bfd_fast(indices)
        sc = solution_score(br, wr)
        if sc > best_score:
            best_bins, best_wts = br, wr
            best_score = sc
    
    # Quick local search on best initial
    ls_budget = max(0.05, min(time_remaining() * 0.12, 2.0))
    best_bins, best_wts = local_search(best_bins, best_wts, time_budget=ls_budget)
    best_score = solution_score(best_bins, best_wts)
    
    if len(best_bins) <= lower_bound:
        return {"packing": best_bins, "bin_weights": best_wts}
    
    # Targeted reduction attempt
    tr_time = min(1.0, time_remaining() * 0.05)
    best_bins, best_wts = targeted_reduction(
        [list(b) for b in best_bins], list(best_wts), max_time=tr_time
    )
    best_score = solution_score(best_bins, best_wts)
    
    if len(best_bins) <= lower_bound:
        return {"packing": best_bins, "bin_weights": best_wts}
    
    # === VNS Main Loop ===
    k_max = 11
    if n > 500:
        k_max = 9
    if n > 1000:
        k_max = 7
    
    current_bins, current_wts = copy_solution(best_bins, best_wts)
    current_score = best_score
    
    stagnation_count = 0
    vns_deadline = start_time + time_limit * 0.98
    iteration = 0
    targeted_counter = 0
    last_improvement_time = time.time()
    
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
            
            # Adaptive LS time: more time for smaller k (more promising)
            if k <= 3:
                ls_time = min(remaining * 0.25, max(0.02, remaining / 8))
            else:
                ls_time = min(remaining * 0.15, max(0.01, remaining / 12))
            
            ls_bins, ls_wts = local_search(shaken_bins, shaken_wts, time_budget=ls_time)
            ls_score = solution_score(ls_bins, ls_wts)
            
            if ls_score > current_score:
                current_bins, current_wts = ls_bins, ls_wts
                current_score = ls_score
                k = 1
                last_improvement_time = time.time()
                
                if current_score > best_score:
                    best_bins, best_wts = copy_solution(current_bins, current_wts)
                    best_score = current_score
                    stagnation_count = 0
                    
                    if len(best_bins) <= lower_bound:
                        return {"packing": best_bins, "bin_weights": best_wts}
            else:
                k += 1
            
            # Periodically try targeted reduction
            if targeted_counter >= 4 and time.time() < vns_deadline:
                targeted_counter = 0
                tr_time = min(0.8, (vns_deadline - time.time()) * 0.08)
                tr_bins, tr_wts = targeted_reduction(
                    [list(b) for b in current_bins], list(current_wts), max_time=tr_time
                )
                tr_score = solution_score(tr_bins, tr_wts)
                if tr_score > current_score:
                    current_bins, current_wts = tr_bins, tr_wts
                    current_score = tr_score
                    k = 1
                    last_improvement_time = time.time()
                    if current_score > best_score:
                        best_bins, best_wts = copy_solution(current_bins, current_wts)
                        best_score = current_score
                        stagnation_count = 0
                        if len(best_bins) <= lower_bound:
                            return {"packing": best_bins, "bin_weights": best_wts}
        
        # Restart
        stagnation_count += 1
        current_bins, current_wts = copy_solution(best_bins, best_wts)
        current_score = best_score
        
        num_bins_curr = len(current_bins)
        if num_bins_curr > 2:
            destroy_pct = min(0.6, 0.2 + stagnation_count * 0.05)
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
            
            # Randomized best-fit rebuild
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