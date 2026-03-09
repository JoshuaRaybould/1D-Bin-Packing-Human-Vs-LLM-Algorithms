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
    w_arr = weights
    
    def elapsed():
        return time.time() - start_time
    
    # Sort indices by weight descending
    sorted_indices = sorted(range(n), key=lambda i: w_arr[i], reverse=True)
    
    total_weight = sum(weights)
    L1 = (total_weight + C - 1) // C
    
    # Better lower bound
    def compute_lower_bound():
        best_lb = L1
        half = C // 2
        # Precompute sorted weights
        sw = sorted(weights, reverse=True)
        
        for k in range(1, min(half + 1, 300)):
            n1 = 0
            n1_weight = 0
            n2 = 0
            n2_weight = 0
            s3 = 0
            threshold_high = C - k
            for wi in sw:
                if wi > threshold_high:
                    n1 += 1
                    n1_weight += wi
                elif wi > half:
                    n2 += 1
                    n2_weight += wi
                elif wi >= k:
                    s3 += wi
                elif wi < k:
                    break  # sorted desc, rest are smaller
            space_n1 = n1 * C - n1_weight
            remaining_s3 = max(0, s3 - space_n1)
            lb = n1 + n2 + ((remaining_s3 + C - 1) // C if remaining_s3 > 0 else 0)
            lb = max(lb, L1)
            if lb > best_lb:
                best_lb = lb
        return best_lb
    
    lower_bound = L1
    if n <= 200000:
        try:
            lower_bound = compute_lower_bound()
        except:
            lower_bound = L1
    
    best_packing_items = None
    best_packing_rem = None
    best_num_bins = n + 1
    
    def construct_bfd_fast(alpha=0.0):
        """BFD construction using bisect for fast best-fit."""
        # Each bin: (remaining_capacity)
        # We maintain a list of (remaining, bin_index) sorted by remaining
        bin_items_list = []
        bin_rem_list = []
        # sorted_bins: list of (remaining, bin_index)
        sorted_bins = []  # sorted by remaining capacity
        
        for idx in sorted_indices:
            w = w_arr[idx]
            if w > C:
                b = len(bin_items_list)
                bin_items_list.append([idx])
                bin_rem_list.append(C - w)
                continue
            
            if not sorted_bins:
                b = len(bin_items_list)
                bin_items_list.append([idx])
                bin_rem_list.append(C - w)
                insort(sorted_bins, (C - w, b))
                continue
            
            # Find best fit: smallest remaining >= w
            pos = bisect_left(sorted_bins, (w, -1))
            
            if pos >= len(sorted_bins):
                # No bin can fit this item
                b = len(bin_items_list)
                bin_items_list.append([idx])
                bin_rem_list.append(C - w)
                insort(sorted_bins, (C - w, b))
            else:
                if alpha == 0.0:
                    # Best fit: pick the one at pos (smallest remaining >= w)
                    rem, b = sorted_bins[pos]
                    sorted_bins.pop(pos)
                    new_rem = rem - w
                    bin_items_list[b].append(idx)
                    bin_rem_list[b] = new_rem
                    insort(sorted_bins, (new_rem, b))
                else:
                    # RCL: candidates from pos to end
                    num_candidates = len(sorted_bins) - pos
                    best_val = sorted_bins[pos][0] - w  # tightest fit
                    worst_val = sorted_bins[-1][0] - w  # loosest fit
                    threshold = best_val + alpha * (worst_val - best_val)
                    # Find upper bound
                    upper = bisect_left(sorted_bins, (w + threshold + 1, -1))
                    if upper <= pos:
                        upper = pos + 1
                    rcl_size = upper - pos
                    if rcl_size < 1:
                        rcl_size = 1
                    chosen_pos = pos + random.randint(0, min(rcl_size - 1, num_candidates - 1))
                    
                    rem, b = sorted_bins[chosen_pos]
                    sorted_bins.pop(chosen_pos)
                    new_rem = rem - w
                    bin_items_list[b].append(idx)
                    bin_rem_list[b] = new_rem
                    insort(sorted_bins, (new_rem, b))
        
        return bin_items_list, bin_rem_list
    
    def construct_ffd_fast(alpha=0.0):
        """FFD construction: first fit decreasing."""
        bin_items_list = []
        bin_rem_list = []
        
        for idx in sorted_indices:
            w = w_arr[idx]
            if w > C:
                bin_items_list.append([idx])
                bin_rem_list.append(C - w)
                continue
            
            placed = False
            if alpha == 0.0:
                for b in range(len(bin_items_list)):
                    if bin_rem_list[b] >= w:
                        bin_items_list[b].append(idx)
                        bin_rem_list[b] -= w
                        placed = True
                        break
            else:
                candidates = []
                for b in range(len(bin_items_list)):
                    if bin_rem_list[b] >= w:
                        candidates.append(b)
                        if len(candidates) > 20:
                            break
                if candidates:
                    if alpha > 0 and len(candidates) > 1:
                        k = max(1, int(len(candidates) * alpha))
                        chosen = candidates[random.randint(0, min(k, len(candidates)-1))]
                    else:
                        chosen = candidates[0]
                    bin_items_list[chosen].append(idx)
                    bin_rem_list[chosen] -= w
                    placed = True
            
            if not placed:
                bin_items_list.append([idx])
                bin_rem_list.append(C - w)
        
        return bin_items_list, bin_rem_list
    
    def local_search_empty_bins(bin_items, bin_remaining):
        """Try to empty bins by redistributing items using bisect-based best fit."""
        improved = True
        while improved:
            improved = False
            if elapsed() > time_limit * 0.95:
                break
            
            num_bins = len(bin_items)
            if num_bins <= lower_bound:
                break
            
            # Sort bins by total weight ascending (lightest first)
            bin_weight_idx = [(C - bin_remaining[i], i) for i in range(num_bins)]
            bin_weight_idx.sort()
            
            emptied = set()
            
            for bw, src_idx in bin_weight_idx:
                if src_idx in emptied:
                    continue
                if not bin_items[src_idx]:
                    continue
                if bw == 0:
                    emptied.add(src_idx)
                    continue
                if elapsed() > time_limit * 0.95:
                    break
                
                # Try to place all items from src_idx into other bins
                src_items_sorted = sorted(bin_items[src_idx], key=lambda x: w_arr[x], reverse=True)
                
                temp_rem = list(bin_remaining)
                moves = {}
                success = True
                
                for item_idx in src_items_sorted:
                    w = w_arr[item_idx]
                    best_target = -1
                    best_rem_after = C + 1
                    for t_idx in range(num_bins):
                        if t_idx == src_idx or t_idx in emptied:
                            continue
                        r = temp_rem[t_idx]
                        if r >= w and r - w < best_rem_after:
                            best_rem_after = r - w
                            best_target = t_idx
                    
                    if best_target == -1:
                        success = False
                        break
                    
                    moves[item_idx] = best_target
                    temp_rem[best_target] -= w
                
                if success:
                    for item_idx, target in moves.items():
                        bin_items[target].append(item_idx)
                        bin_remaining[target] -= w_arr[item_idx]
                    bin_items[src_idx] = []
                    bin_remaining[src_idx] = C
                    emptied.add(src_idx)
                    improved = True
            
            # Remove empty bins
            new_items = []
            new_rem = []
            for i in range(len(bin_items)):
                if bin_items[i]:
                    new_items.append(bin_items[i])
                    new_rem.append(bin_remaining[i])
            bin_items = new_items
            bin_remaining = new_rem
            
            if not improved:
                break
        
        return bin_items, bin_remaining
    
    def local_search_swap(bin_items, bin_remaining):
        """Try to empty bins using item swaps/displacement chains."""
        max_rounds = 3
        
        for round_count in range(max_rounds):
            if elapsed() > time_limit * 0.93:
                break
            
            num_bins = len(bin_items)
            if num_bins <= lower_bound:
                break
            
            bin_weight_idx = [(C - bin_remaining[i], i) for i in range(num_bins)]
            bin_weight_idx.sort()
            
            found_improvement = False
            candidates_to_try = min(max(1, num_bins // 4), 8)
            
            for bw, src_idx in bin_weight_idx[:candidates_to_try]:
                if elapsed() > time_limit * 0.93:
                    break
                if not bin_items[src_idx] or bw == 0:
                    continue
                
                # Build item-to-bin mapping for other bins
                temp_rem = {}
                temp_items = {}
                for i in range(num_bins):
                    if i != src_idx:
                        temp_rem[i] = bin_remaining[i]
                        temp_items[i] = list(bin_items[i])
                
                remaining_to_place = sorted(list(bin_items[src_idx]), key=lambda x: w_arr[x], reverse=True)
                
                max_iter = len(remaining_to_place) * 8 + 20
                iteration_count = 0
                
                while remaining_to_place and iteration_count < max_iter:
                    iteration_count += 1
                    if elapsed() > time_limit * 0.93:
                        break
                    
                    placed_one = False
                    for ri in range(len(remaining_to_place)):
                        item_idx = remaining_to_place[ri]
                        w = w_arr[item_idx]
                        
                        # Direct best-fit placement
                        best_target = -1
                        best_rem_after = C + 1
                        for t_idx, t_rem in temp_rem.items():
                            if t_rem >= w and t_rem - w < best_rem_after:
                                best_rem_after = t_rem - w
                                best_target = t_idx
                        
                        if best_target != -1:
                            temp_rem[best_target] -= w
                            temp_items[best_target].append(item_idx)
                            remaining_to_place.pop(ri)
                            placed_one = True
                            break
                        
                        # Displacement: swap out one item to make room
                        best_swap = None
                        best_net_gain = -1  # we want to displace lightest possible
                        
                        for t_idx, t_items in temp_items.items():
                            t_rem = temp_rem[t_idx]
                            for swap_item in t_items:
                                sw = w_arr[swap_item]
                                if sw < w and t_rem + sw >= w:
                                    gain = w - sw  # net reduction in what needs placing
                                    if gain > best_net_gain:
                                        best_net_gain = gain
                                        best_swap = (t_idx, swap_item)
                        
                        if best_swap is not None:
                            t_idx, swap_item = best_swap
                            sw = w_arr[swap_item]
                            temp_rem[t_idx] = temp_rem[t_idx] + sw - w
                            temp_items[t_idx].remove(swap_item)
                            temp_items[t_idx].append(item_idx)
                            remaining_to_place.pop(ri)
                            remaining_to_place.append(swap_item)
                            remaining_to_place.sort(key=lambda x: w_arr[x], reverse=True)
                            placed_one = True
                            break
                    
                    if not placed_one:
                        break
                
                if not remaining_to_place:
                    new_items = []
                    new_rem = []
                    for i in range(num_bins):
                        if i == src_idx:
                            continue
                        new_items.append(temp_items[i])
                        new_rem.append(temp_rem[i])
                    bin_items = new_items
                    bin_remaining = new_rem
                    found_improvement = True
                    break
            
            if not found_improvement:
                break
        
        return bin_items, bin_remaining
    
    def local_search_swap_pairs(bin_items, bin_remaining):
        """Try 1-1 and 2-1 swaps between bins to enable emptying."""
        if elapsed() > time_limit * 0.90:
            return bin_items, bin_remaining
        
        num_bins = len(bin_items)
        if num_bins <= lower_bound:
            return bin_items, bin_remaining
        
        # Build item-to-bin index
        item_bin = [0] * n
        for b in range(num_bins):
            for it in bin_items[b]:
                item_bin[it] = b
        
        # Try 1-1 swaps that increase packing tightness
        improved = True
        rounds = 0
        while improved and rounds < 2:
            improved = False
            rounds += 1
            if elapsed() > time_limit * 0.88:
                break
            
            # Find lightest bins
            bin_weight_idx = [(C - bin_remaining[i], i) for i in range(num_bins) if bin_items[i]]
            bin_weight_idx.sort()
            
            for bw, src in bin_weight_idx[:5]:
                if elapsed() > time_limit * 0.88:
                    break
                if not bin_items[src]:
                    continue
                
                # For each item in src, try swapping with an item in another bin
                # such that the src bin gets lighter
                src_copy = list(bin_items[src])
                for item_a in src_copy:
                    wa = w_arr[item_a]
                    # Find item_b in another bin where wb > wa and bin_b has room for the difference
                    for b2 in range(num_bins):
                        if b2 == src:
                            continue
                        for item_b in bin_items[b2]:
                            wb = w_arr[item_b]
                            if wb > wa:
                                diff = wb - wa
                                # src gains diff space, b2 loses diff space
                                if bin_remaining[b2] >= diff:
                                    # Do the swap - src gets lighter
                                    bin_items[src].remove(item_a)
                                    bin_items[b2].remove(item_b)
                                    bin_items[src].append(item_b)
                                    bin_items[b2].append(item_a)
                                    bin_remaining[src] -= diff
                                    bin_remaining[b2] += diff
                                    # Nope, this makes src heavier. We want src lighter.
                                    # Undo
                                    bin_items[src].remove(item_b)
                                    bin_items[b2].remove(item_a)
                                    bin_items[src].append(item_a)
                                    bin_items[b2].append(item_b)
                                    bin_remaining[src] += diff
                                    bin_remaining[b2] -= diff
                            elif wb < wa:
                                diff = wa - wb
                                # src gains diff space (gets lighter), b2 loses diff
                                if bin_remaining[b2] >= diff:
                                    bin_items[src].remove(item_a)
                                    bin_items[b2].remove(item_b)
                                    bin_items[src].append(item_b)
                                    bin_items[b2].append(item_a)
                                    bin_remaining[src] += diff
                                    bin_remaining[b2] -= diff
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                # Try emptying again
                bin_items, bin_remaining = local_search_empty_bins(bin_items, bin_remaining)
                num_bins = len(bin_items)
        
        return bin_items, bin_remaining
    
    def full_local_search(bi, br):
        """Apply all local search phases."""
        bi, br = local_search_empty_bins(bi, br)
        if len(bi) > lower_bound and elapsed() < time_limit * 0.90:
            bi, br = local_search_swap(bi, br)
        if len(bi) > lower_bound and elapsed() < time_limit * 0.85:
            bi, br = local_search_empty_bins(bi, br)
        return bi, br
    
    # Initial deterministic BFD solution
    bi, br = construct_bfd_fast(0.0)
    bi, br = full_local_search(bi, br)
    num_b = len(bi)
    if num_b < best_num_bins:
        best_num_bins = num_b
        best_packing_items = [list(b) for b in bi]
        best_packing_rem = list(br)
    
    if best_num_bins <= lower_bound:
        packing = best_packing_items
        bin_weights_out = [C - best_packing_rem[i] for i in range(len(best_packing_rem))]
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Also try FFD
    if elapsed() < time_limit * 0.85:
        bi2, br2 = construct_ffd_fast(0.0)
        bi2, br2 = full_local_search(bi2, br2)
        num_b2 = len(bi2)
        if num_b2 < best_num_bins:
            best_num_bins = num_b2
            best_packing_items = [list(b) for b in bi2]
            best_packing_rem = list(br2)
        
        if best_num_bins <= lower_bound:
            packing = best_packing_items
            bin_weights_out = [C - best_packing_rem[i] for i in range(len(best_packing_rem))]
            return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Reactive GRASP
    alphas = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    alpha_count = {a: 0 for a in alphas}
    alpha_quality_sum = {a: 0.0 for a in alphas}
    alpha_probs = {a: 1.0 / len(alphas) for a in alphas}
    
    def select_alpha():
        r = random.random()
        cumulative = 0.0
        for a in alphas:
            cumulative += alpha_probs[a]
            if r <= cumulative:
                return a
        return alphas[-1]
    
    def update_alpha_probs():
        avg_qualities = {}
        for a in alphas:
            if alpha_count[a] > 0:
                avg_qualities[a] = alpha_quality_sum[a] / alpha_count[a]
            else:
                avg_qualities[a] = 0.0
        
        total_q = sum(avg_qualities.values())
        if total_q > 0:
            for a in alphas:
                alpha_probs[a] = max(0.01 / len(alphas), avg_qualities[a] / total_q)
            total_p = sum(alpha_probs.values())
            for a in alphas:
                alpha_probs[a] /= total_p
    
    iteration = 0
    update_interval = 10
    
    while elapsed() < time_limit * 0.93:
        if iteration < len(alphas):
            alpha = alphas[iteration % len(alphas)]
        else:
            alpha = select_alpha()
        
        # Alternate construction methods
        if random.random() < 0.7:
            bi, br = construct_bfd_fast(alpha)
        else:
            bi, br = construct_ffd_fast(alpha)
        
        # Local search
        bi, br = local_search_empty_bins(bi, br)
        num_b = len(bi)
        
        # Deeper search if promising
        if num_b <= best_num_bins + 1 and elapsed() < time_limit * 0.90:
            bi, br = local_search_swap(bi, br)
            num_b = len(bi)
            if num_b <= best_num_bins + 1 and elapsed() < time_limit * 0.88:
                bi, br = local_search_empty_bins(bi, br)
                num_b = len(bi)
        
        alpha_count[alpha] += 1
        alpha_quality_sum[alpha] += 1.0 / num_b
        
        if num_b < best_num_bins:
            best_num_bins = num_b
            best_packing_items = [list(b) for b in bi]
            best_packing_rem = list(br)
        
        if best_num_bins <= lower_bound:
            break
        
        if iteration > 0 and iteration % update_interval == 0:
            update_alpha_probs()
        
        iteration += 1
    
    # Final intensification
    if elapsed() < time_limit * 0.97 and best_packing_items is not None:
        bi, br = local_search_swap(best_packing_items, best_packing_rem)
        if len(bi) < best_num_bins:
            best_num_bins = len(bi)
            best_packing_items = bi
            best_packing_rem = br
        if elapsed() < time_limit * 0.96:
            bi, br = local_search_empty_bins(bi, br)
            if len(bi) < best_num_bins:
                best_num_bins = len(bi)
                best_packing_items = bi
                best_packing_rem = br
    
    if best_packing_items is None:
        bi, br = construct_bfd_fast(0.0)
        best_packing_items = bi
        best_packing_rem = br
    
    packing = best_packing_items
    bin_weights_out = [C - best_packing_rem[i] for i in range(len(best_packing_rem))]
    
    return {"packing": packing, "bin_weights": bin_weights_out}