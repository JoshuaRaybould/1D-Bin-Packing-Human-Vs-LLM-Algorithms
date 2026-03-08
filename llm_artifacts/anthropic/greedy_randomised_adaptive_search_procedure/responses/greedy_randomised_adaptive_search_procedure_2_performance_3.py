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
    w = weights  # alias
    
    def elapsed():
        return time.time() - start_time
    
    total_weight = sum(weights)
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    # Lower bound
    L1 = math.ceil(total_weight / C) if C > 0 else n
    if C == 0:
        packing = [[i] for i in range(n)]
        bw = [weights[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bw}
    
    L2 = L1
    half_C = C / 2.0
    try:
        for t in range(1, C // 2 + 1):
            n_large = 0
            free_space = 0
            sum_medium = 0
            sum_small = 0
            for wi in weights:
                if wi > C - t:
                    n_large += 1
                    free_space += C - wi
                elif wi > t:
                    sum_medium += wi
                else:
                    sum_small += wi
            remaining_small = max(0, sum_small - free_space)
            total_need = sum_medium + remaining_small
            needed = n_large + (math.ceil(total_need / C) if total_need > 0 else 0)
            if needed > L2:
                L2 = needed
            if elapsed() > time_limit * 0.01:
                break
    except:
        pass
    
    lower_bound = L2
    
    # ---- Construction ----
    def construct_bfd(alpha=0.0, order=None):
        """Best-fit decreasing with optional RCL randomization."""
        if order is None:
            order = sorted_indices
        
        bin_items_list = []
        bin_remaining_list = []
        sorted_bins = []  # (remaining, bin_index)
        
        for idx in order:
            wi = weights[idx]
            if wi == 0:
                if bin_items_list:
                    bin_items_list[0].append(idx)
                else:
                    bin_items_list.append([idx])
                    bin_remaining_list.append(C)
                    insort(sorted_bins, (C, 0))
                continue
            if wi > C:
                bin_items_list.append([idx])
                bin_remaining_list.append(C - wi)
                continue
            
            pos = bisect_left(sorted_bins, (wi, -1))
            
            if pos >= len(sorted_bins):
                b_idx = len(bin_items_list)
                bin_items_list.append([idx])
                bin_remaining_list.append(C - wi)
                if C - wi > 0:
                    insort(sorted_bins, (C - wi, b_idx))
                continue
            
            if alpha <= 0.001:
                rem, b_idx = sorted_bins[pos]
                sorted_bins.pop(pos)
                bin_items_list[b_idx].append(idx)
                new_rem = rem - wi
                bin_remaining_list[b_idx] = new_rem
                if new_rem > 0:
                    insort(sorted_bins, (new_rem, b_idx))
            else:
                num_feasible = len(sorted_bins) - pos
                if num_feasible == 1:
                    rem, b_idx = sorted_bins[pos]
                    sorted_bins.pop(pos)
                    bin_items_list[b_idx].append(idx)
                    new_rem = rem - wi
                    bin_remaining_list[b_idx] = new_rem
                    if new_rem > 0:
                        insort(sorted_bins, (new_rem, b_idx))
                else:
                    min_res = sorted_bins[pos][0] - wi
                    max_res = sorted_bins[-1][0] - wi
                    threshold = min_res + alpha * (max_res - min_res)
                    rcl_end = bisect_left(sorted_bins, (threshold + wi + 1, -1))
                    if rcl_end <= pos:
                        rcl_end = pos + 1
                    chosen_pos = random.randint(pos, min(rcl_end - 1, len(sorted_bins) - 1))
                    rem, b_idx = sorted_bins[chosen_pos]
                    sorted_bins.pop(chosen_pos)
                    bin_items_list[b_idx].append(idx)
                    new_rem = rem - wi
                    bin_remaining_list[b_idx] = new_rem
                    if new_rem > 0:
                        insort(sorted_bins, (new_rem, b_idx))
        
        return bin_items_list, bin_remaining_list
    
    def construct_ffd(alpha=0.0, order=None):
        """First-fit decreasing with optional RCL."""
        if order is None:
            order = sorted_indices
        
        bin_items_list = []
        bin_remaining_list = []
        
        for idx in order:
            wi = weights[idx]
            if wi == 0:
                if bin_items_list:
                    bin_items_list[0].append(idx)
                else:
                    bin_items_list.append([idx])
                    bin_remaining_list.append(C)
                continue
            if wi > C:
                bin_items_list.append([idx])
                bin_remaining_list.append(C - wi)
                continue
            
            placed = False
            candidates = []
            for bi in range(len(bin_items_list)):
                if bin_remaining_list[bi] >= wi:
                    candidates.append(bi)
            
            if not candidates:
                bin_items_list.append([idx])
                bin_remaining_list.append(C - wi)
                continue
            
            if alpha <= 0.001 or len(candidates) == 1:
                chosen = candidates[0]
            else:
                k = max(1, int(len(candidates) * alpha))
                chosen = random.choice(candidates[:k])
            
            bin_items_list[chosen].append(idx)
            bin_remaining_list[chosen] -= wi
        
        return bin_items_list, bin_remaining_list
    
    # ---- Local Search: Bin Emptying ----
    def local_search(bin_items, bin_rem, time_frac=0.95):
        """Try to empty bins by redistributing their items."""
        max_rounds = 30
        
        for rnd in range(max_rounds):
            if elapsed() > time_limit * time_frac:
                break
            
            # Compact: remove empty bins
            new_items = []
            new_rem = []
            for i in range(len(bin_items)):
                if bin_items[i]:
                    new_items.append(bin_items[i])
                    new_rem.append(bin_rem[i])
            bin_items = new_items
            bin_rem = new_rem
            
            num_bins = len(bin_items)
            if num_bins <= lower_bound:
                break
            
            # Sort bins by weight (ascending) to try emptying lightest first
            bin_weights = [C - bin_rem[i] for i in range(num_bins)]
            order = sorted(range(num_bins), key=lambda i: bin_weights[i])
            
            emptied_any = False
            
            for src in order:
                if not bin_items[src]:
                    continue
                if elapsed() > time_limit * time_frac:
                    break
                
                items_to_place = sorted(bin_items[src], key=lambda i: weights[i], reverse=True)
                
                # Build sorted remaining for other bins
                sr = []
                for i in range(num_bins):
                    if i != src and bin_items[i] and bin_rem[i] > 0:
                        sr.append((bin_rem[i], i))
                sr.sort()
                
                placements = []
                failed = False
                
                for item in items_to_place:
                    wi = weights[item]
                    if wi == 0:
                        if sr:
                            placements.append((item, sr[0][1]))
                        elif placements:
                            placements.append((item, placements[0][1]))
                        continue
                    
                    pos = bisect_left(sr, (wi, -1))
                    if pos < len(sr):
                        rem_val, b_idx = sr[pos]
                        sr.pop(pos)
                        new_r = rem_val - wi
                        placements.append((item, b_idx))
                        if new_r > 0:
                            insort(sr, (new_r, b_idx))
                    else:
                        failed = True
                        break
                
                if not failed:
                    # Apply
                    for item, target in placements:
                        bin_items[target].append(item)
                        bin_rem[target] -= weights[item]
                    bin_items[src] = []
                    bin_rem[src] = 0
                    emptied_any = True
                    continue
                
                # Try swap-assisted emptying
                # Undo the partial simulation, redo with swap attempts
                if len(items_to_place) <= 20:
                    success = try_swap_empty_v2(bin_items, bin_rem, src, num_bins)
                    if success:
                        emptied_any = True
                        continue
            
            if not emptied_any:
                # Try consolidation moves to enable future emptying
                did_consol = consolidation_v2(bin_items, bin_rem, time_frac)
                if not did_consol:
                    break
        
        # Compact
        new_items = []
        new_rem = []
        for i in range(len(bin_items)):
            if bin_items[i]:
                new_items.append(bin_items[i])
                new_rem.append(bin_rem[i])
        return new_items, new_rem
    
    def try_swap_empty_v2(bin_items, bin_rem, src, num_bins):
        """Try emptying bin src using 1-swap assistance."""
        items_to_place = sorted(bin_items[src], key=lambda i: weights[i], reverse=True)
        
        # Build index of items per bin for fast lookup
        other_bins = [i for i in range(num_bins) if i != src and bin_items[i]]
        
        # For each possible single swap:
        # Remove one item from some other bin, creating more space there,
        # then try to BFD all src items + swapped item into remaining bins
        
        src_weight = sum(weights[item] for item in bin_items[src])
        
        # Find bins with small remaining capacity that could benefit from a swap
        for ob in other_bins:
            for swap_out in bin_items[ob]:
                ws = weights[swap_out]
                # After removing swap_out from ob, ob has bin_rem[ob] + ws space
                # We need to fit all src items + swap_out item into all other bins
                # This is complex - let's just try the simple BFD approach
                
                # Quick feasibility check
                total_free = sum(bin_rem[i] for i in other_bins)
                # With swap_out removed and re-added: total_free stays the same
                # But rearrangement might help
                if total_free < src_weight:
                    return False  # No way to fit
                
                # Simulate
                sr = []
                for i in other_bins:
                    rem_val = bin_rem[i] + (ws if i == ob else 0)
                    if rem_val > 0:
                        sr.append((rem_val, i))
                sr.sort()
                
                all_items = items_to_place + [swap_out]
                all_items_sorted = sorted(all_items, key=lambda i: weights[i], reverse=True)
                
                placements = []
                ok = True
                sr_copy = list(sr)
                
                for item in all_items_sorted:
                    wi = weights[item]
                    if wi == 0:
                        if sr_copy:
                            placements.append((item, sr_copy[0][1]))
                        continue
                    pos = bisect_left(sr_copy, (wi, -1))
                    if pos < len(sr_copy):
                        rem_val, b_idx = sr_copy[pos]
                        sr_copy.pop(pos)
                        new_r = rem_val - wi
                        placements.append((item, b_idx))
                        if new_r > 0:
                            insort(sr_copy, (new_r, b_idx))
                    else:
                        ok = False
                        break
                
                if ok:
                    # Apply: remove swap_out from ob
                    bin_items[ob].remove(swap_out)
                    bin_rem[ob] += ws
                    # Apply placements
                    for item, target in placements:
                        bin_items[target].append(item)
                        bin_rem[target] -= weights[item]
                    bin_items[src] = []
                    bin_rem[src] = 0
                    return True
            
            # Limit search to avoid timeout
            if elapsed() > time_limit * 0.9:
                return False
        
        return False
    
    def consolidation_v2(bin_items, bin_rem, time_frac):
        """Move items between bins to make some bins lighter (easier to empty)."""
        num_bins = len(bin_items)
        active = [i for i in range(num_bins) if bin_items[i]]
        if len(active) < 2:
            return False
        
        did_something = False
        
        # Sort by weight ascending
        active.sort(key=lambda i: C - bin_rem[i])
        
        # For lightest bins, try to move items to heavier bins
        for ai_idx in range(min(5, len(active))):
            i = active[ai_idx]
            if not bin_items[i]:
                continue
            if elapsed() > time_limit * time_frac:
                break
            
            for item in list(bin_items[i]):
                wi = weights[item]
                if wi == 0:
                    continue
                # Find best-fit bin for this item (not i)
                best_b = -1
                best_rem = C + 1
                for j in active:
                    if j == i:
                        continue
                    if bin_rem[j] >= wi and bin_rem[j] < best_rem:
                        best_rem = bin_rem[j]
                        best_b = j
                
                if best_b >= 0:
                    bin_items[i].remove(item)
                    bin_rem[i] += wi
                    bin_items[best_b].append(item)
                    bin_rem[best_b] -= wi
                    did_something = True
                    if not bin_items[i]:
                        break
            
            # Try 1-1 swaps to reduce weight of bin i
            if bin_items[i]:
                bw_i = C - bin_rem[i]
                for item_a in list(bin_items[i]):
                    wa = weights[item_a]
                    if wa == 0:
                        continue
                    best_swap_j = -1
                    best_swap_item = -1
                    best_reduction = 0
                    for j in active:
                        if j == i:
                            continue
                        for item_b in bin_items[j]:
                            wb = weights[item_b]
                            if wb >= wa:
                                continue
                            diff = wa - wb
                            # Check feasibility: bin j gets diff more weight
                            if bin_rem[j] >= diff:
                                if diff > best_reduction:
                                    best_reduction = diff
                                    best_swap_j = j
                                    best_swap_item = item_b
                    
                    if best_swap_j >= 0:
                        wb = weights[best_swap_item]
                        bin_items[i].remove(item_a)
                        bin_items[best_swap_j].remove(best_swap_item)
                        bin_items[i].append(best_swap_item)
                        bin_items[best_swap_j].append(item_a)
                        bin_rem[i] += wa - wb
                        bin_rem[best_swap_j] -= wa - wb
                        did_something = True
                    
                    if elapsed() > time_limit * time_frac:
                        break
        
        return did_something
    
    def perturb_solution(bin_items, bin_rem, num_remove=3):
        """Perturb by removing lightest bins and redistributing."""
        active = [i for i in range(len(bin_items)) if bin_items[i]]
        if len(active) <= 2:
            return bin_items, bin_rem
        
        num_remove = min(num_remove, len(active) - 1)
        
        # Remove lightest bins
        active.sort(key=lambda i: C - bin_rem[i])
        to_remove = active[:num_remove]
        
        freed_items = []
        for b in to_remove:
            freed_items.extend(bin_items[b])
            bin_items[b] = []
            bin_rem[b] = 0
        
        freed_items.sort(key=lambda i: weights[i], reverse=True)
        
        sr = []
        for i in range(len(bin_items)):
            if bin_items[i] and bin_rem[i] > 0:
                sr.append((bin_rem[i], i))
        sr.sort()
        
        for item in freed_items:
            wi = weights[item]
            if wi == 0:
                if sr:
                    bin_items[sr[0][1]].append(item)
                else:
                    b_idx = len(bin_items)
                    bin_items.append([item])
                    bin_rem.append(C)
                    insort(sr, (C, b_idx))
                continue
            
            pos = bisect_left(sr, (wi, -1))
            if pos < len(sr):
                rem_val, b_idx = sr[pos]
                sr.pop(pos)
                new_r = rem_val - wi
                bin_items[b_idx].append(item)
                bin_rem[b_idx] = new_r
                if new_r > 0:
                    insort(sr, (new_r, b_idx))
            else:
                b_idx = len(bin_items)
                bin_items.append([item])
                bin_rem.append(C - wi)
                if C - wi > 0:
                    insort(sr, (C - wi, b_idx))
        
        return bin_items, bin_rem
    
    def perturb_random(bin_items, bin_rem, num_remove=3):
        """Perturb by removing random bins and redistributing."""
        active = [i for i in range(len(bin_items)) if bin_items[i]]
        if len(active) <= 2:
            return bin_items, bin_rem
        
        num_remove = min(num_remove, len(active) - 1)
        to_remove = random.sample(active, num_remove)
        
        freed_items = []
        for b in to_remove:
            freed_items.extend(bin_items[b])
            bin_items[b] = []
            bin_rem[b] = 0
        
        freed_items.sort(key=lambda i: weights[i], reverse=True)
        
        sr = []
        for i in range(len(bin_items)):
            if bin_items[i] and bin_rem[i] > 0:
                sr.append((bin_rem[i], i))
        sr.sort()
        
        for item in freed_items:
            wi = weights[item]
            if wi == 0:
                if sr:
                    bin_items[sr[0][1]].append(item)
                else:
                    b_idx = len(bin_items)
                    bin_items.append([item])
                    bin_rem.append(C)
                    insort(sr, (C, b_idx))
                continue
            
            pos = bisect_left(sr, (wi, -1))
            if pos < len(sr):
                rem_val, b_idx = sr[pos]
                sr.pop(pos)
                new_r = rem_val - wi
                bin_items[b_idx].append(item)
                bin_rem[b_idx] = new_r
                if new_r > 0:
                    insort(sr, (new_r, b_idx))
            else:
                b_idx = len(bin_items)
                bin_items.append([item])
                bin_rem.append(C - wi)
                if C - wi > 0:
                    insort(sr, (C - wi, b_idx))
        
        return bin_items, bin_rem
    
    def count_bins(bi):
        return sum(1 for b in bi if b)
    
    def compact(bin_items, bin_rem):
        ni = []
        nr = []
        for i in range(len(bin_items)):
            if bin_items[i]:
                ni.append(bin_items[i])
                nr.append(bin_rem[i])
        return ni, nr
    
    def copy_solution(bi, br):
        return [list(b) for b in bi], list(br)
    
    # ---- Initial solution: BFD ----
    bi, br = construct_bfd(0.0)
    bi, br = local_search(bi, br, time_frac=0.3)
    bi, br = compact(bi, br)
    best_num_bins = len(bi)
    best_bi, best_br = copy_solution(bi, br)
    
    if best_num_bins <= lower_bound:
        packing = best_bi
        bw = [C - best_br[i] for i in range(len(best_bi))]
        return {"packing": packing, "bin_weights": bw}
    
    # ---- Reactive GRASP ----
    alphas = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    alpha_counts = [0] * len(alphas)
    alpha_sum_bins = [0.0] * len(alphas)
    alpha_best = [float('inf')] * len(alphas)
    
    def select_alpha_reactive():
        # Use average performance to select
        scores = []
        for i in range(len(alphas)):
            if alpha_counts[i] > 0:
                avg = alpha_sum_bins[i] / alpha_counts[i]
                # Score inversely proportional to average bins
                scores.append(1.0 / (avg - lower_bound + 1.0))
            else:
                scores.append(10.0)  # Encourage exploration
        total = sum(scores)
        r = random.random() * total
        cumsum = 0.0
        for i, s in enumerate(scores):
            cumsum += s
            if r <= cumsum:
                return i, alphas[i]
        return len(alphas) - 1, alphas[-1]
    
    iteration = 0
    no_improve = 0
    
    # Determine time allocation
    time_for_grasp = time_limit * 0.95
    
    while elapsed() < time_for_grasp:
        iteration += 1
        
        if iteration <= len(alphas):
            alpha_idx = (iteration - 1) % len(alphas)
            alpha = alphas[alpha_idx]
        else:
            alpha_idx, alpha = select_alpha_reactive()
        
        # Vary construction strategy
        r_strat = random.random()
        if r_strat < 0.6:
            bi, br = construct_bfd(alpha)
        elif r_strat < 0.8:
            # Slightly shuffled order
            order = list(sorted_indices)
            # Shuffle items of similar weight
            if alpha > 0:
                for i in range(len(order) - 1):
                    if random.random() < alpha * 0.3:
                        order[i], order[i+1] = order[i+1], order[i]
            bi, br = construct_bfd(0.0, order=order)
        else:
            bi, br = construct_bfd(alpha)
        
        # Time fraction for local search
        ls_time = min(time_for_grasp, elapsed() + (time_for_grasp - elapsed()) * 0.5)
        ls_frac = ls_time / time_limit if time_limit > 0 else 0.95
        
        bi, br = local_search(bi, br, time_frac=ls_frac)
        bi, br = compact(bi, br)
        num_bins = len(bi)
        
        # Update reactive scores
        alpha_counts[alpha_idx] += 1
        alpha_sum_bins[alpha_idx] += num_bins
        if num_bins < alpha_best[alpha_idx]:
            alpha_best[alpha_idx] = num_bins
        
        if num_bins < best_num_bins:
            best_num_bins = num_bins
            best_bi, best_br = copy_solution(bi, br)
            no_improve = 0
            if best_num_bins <= lower_bound:
                break
        else:
            no_improve += 1
        
        # Perturbation phase
        if no_improve >= 10 and elapsed() < time_for_grasp:
            for _ in range(3):
                if elapsed() > time_for_grasp:
                    break
                p_bi, p_br = copy_solution(best_bi, best_br)
                
                if random.random() < 0.5:
                    num_rem = random.randint(2, min(4, best_num_bins - 1))
                    p_bi, p_br = perturb_solution(p_bi, p_br, num_rem)
                else:
                    num_rem = random.randint(2, min(5, best_num_bins - 1))
                    p_bi, p_br = perturb_random(p_bi, p_br, num_rem)
                
                ls_time2 = min(time_for_grasp, elapsed() + (time_for_grasp - elapsed()) * 0.3)
                ls_frac2 = ls_time2 / time_limit if time_limit > 0 else 0.95
                p_bi, p_br = local_search(p_bi, p_br, time_frac=ls_frac2)
                p_bi, p_br = compact(p_bi, p_br)
                p_num = len(p_bi)
                
                if p_num < best_num_bins:
                    best_num_bins = p_num
                    best_bi, best_br = copy_solution(p_bi, p_br)
                    no_improve = 0
                    if best_num_bins <= lower_bound:
                        break
            
            if best_num_bins <= lower_bound:
                break
            no_improve = 0
    
    # Final intensive local search with remaining time
    if elapsed() < time_limit * 0.98 and best_num_bins > lower_bound:
        bi, br = copy_solution(best_bi, best_br)
        bi, br = local_search(bi, br, time_frac=0.98)
        bi, br = compact(bi, br)
        if len(bi) < best_num_bins:
            best_num_bins = len(bi)
            best_bi, best_br = bi, br
    
    packing = best_bi
    bw = [C - best_br[i] for i in range(len(best_bi))]
    return {"packing": packing, "bin_weights": bw}