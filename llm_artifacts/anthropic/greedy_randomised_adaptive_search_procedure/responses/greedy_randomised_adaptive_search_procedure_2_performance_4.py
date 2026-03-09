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
    w = weights
    
    def elapsed():
        return time.time() - start_time
    
    total_weight = sum(weights)
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    if C == 0:
        packing = [[i] for i in range(n)]
        bw = [weights[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bw}
    
    # Lower bound
    L1 = math.ceil(total_weight / C)
    L2 = L1
    try:
        half = C // 2
        for t in range(1, half + 1):
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
            if elapsed() > time_limit * 0.02:
                break
    except:
        pass
    
    lower_bound = max(L1, L2)
    
    # ---- Construction: BFD with RCL ----
    def construct_bfd(alpha=0.0, order=None):
        if order is None:
            order = sorted_indices
        
        bin_items_list = []
        bin_remaining_list = []
        sorted_bins = []  # (remaining, bin_index)
        
        for idx in order:
            wi = w[idx]
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
    
    # ---- Local Search ----
    def local_search(bin_items, bin_rem, deadline):
        """Try to empty bins by redistributing their items, with swap-assisted moves."""
        max_rounds = 50
        
        for rnd in range(max_rounds):
            if time.time() - start_time > deadline:
                break
            
            # Compact
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
            bin_weights_arr = [C - bin_rem[i] for i in range(num_bins)]
            order = sorted(range(num_bins), key=lambda i: bin_weights_arr[i])
            
            emptied_any = False
            
            for src in order:
                if not bin_items[src]:
                    continue
                if time.time() - start_time > deadline:
                    break
                
                src_items = sorted(bin_items[src], key=lambda i: w[i], reverse=True)
                
                # Build sorted remaining for other bins
                sr = []
                for i in range(num_bins):
                    if i != src and bin_items[i] and bin_rem[i] > 0:
                        sr.append((bin_rem[i], i))
                sr.sort()
                
                # Quick check: total free space
                total_free = sum(x[0] for x in sr)
                src_weight = bin_weights_arr[src]
                if total_free < src_weight:
                    continue
                
                placements = []
                failed = False
                sr_copy = list(sr)
                
                for item in src_items:
                    wi = w[item]
                    if wi == 0:
                        if sr_copy:
                            placements.append((item, sr_copy[0][1]))
                        elif placements:
                            placements.append((item, placements[0][1]))
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
                        failed = True
                        break
                
                if not failed:
                    for item, target in placements:
                        bin_items[target].append(item)
                        bin_rem[target] -= w[item]
                    bin_items[src] = []
                    bin_rem[src] = 0
                    bin_weights_arr[src] = 0
                    emptied_any = True
                    continue
                
                # Swap-assisted emptying
                if len(src_items) <= 30 and time.time() - start_time < deadline:
                    success = try_swap_empty(bin_items, bin_rem, src, num_bins, deadline)
                    if success:
                        bin_weights_arr[src] = 0
                        emptied_any = True
                        continue
            
            if not emptied_any:
                # Try consolidation moves
                did_consol = consolidation(bin_items, bin_rem, deadline)
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
    
    def try_swap_empty(bin_items, bin_rem, src, num_bins, deadline):
        """Try emptying bin src using 1-swap assistance."""
        src_items = sorted(bin_items[src], key=lambda i: w[i], reverse=True)
        other_bins = [i for i in range(num_bins) if i != src and bin_items[i]]
        
        src_weight = sum(w[item] for item in bin_items[src])
        total_free = sum(bin_rem[i] for i in other_bins)
        if total_free < src_weight:
            return False
        
        # For efficiency, build a map of items per bin with their weights
        # Try removing one item from another bin, creating space, then BFD all src items + removed item
        
        # Sort other bins by remaining space ascending
        other_bins_sorted = sorted(other_bins, key=lambda i: bin_rem[i])
        
        attempts = 0
        max_attempts = 200
        
        for ob in other_bins_sorted:
            if time.time() - start_time > deadline:
                return False
            # Try swapping out items from this bin that are large (create most space)
            items_in_ob = sorted(bin_items[ob], key=lambda i: w[i], reverse=True)
            
            for swap_out in items_in_ob[:5]:  # Only try largest items
                attempts += 1
                if attempts > max_attempts:
                    return False
                
                ws = w[swap_out]
                if ws == 0:
                    continue
                
                # Simulate: remove swap_out from ob, try to fit all src items + swap_out
                sr = []
                for i in other_bins:
                    rem_val = bin_rem[i] + (ws if i == ob else 0)
                    if rem_val > 0:
                        sr.append((rem_val, i))
                sr.sort()
                
                all_items = src_items + [swap_out]
                all_items_sorted = sorted(all_items, key=lambda i: w[i], reverse=True)
                
                ok = True
                placements = []
                sr_copy = list(sr)
                
                for item in all_items_sorted:
                    wi = w[item]
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
                    bin_items[ob].remove(swap_out)
                    bin_rem[ob] += ws
                    for item, target in placements:
                        bin_items[target].append(item)
                        bin_rem[target] -= w[item]
                    bin_items[src] = []
                    bin_rem[src] = 0
                    return True
        
        # Try 2-swap: remove one item each from two different bins
        if len(other_bins) >= 2 and time.time() - start_time < deadline:
            attempts2 = 0
            max_attempts2 = 100
            # Pick pairs of bins with small remaining
            for i_idx in range(min(len(other_bins_sorted), 8)):
                ob1 = other_bins_sorted[i_idx]
                items1 = sorted(bin_items[ob1], key=lambda x: w[x], reverse=True)
                for swap1 in items1[:3]:
                    for j_idx in range(i_idx + 1, min(len(other_bins_sorted), 8)):
                        ob2 = other_bins_sorted[j_idx]
                        items2 = sorted(bin_items[ob2], key=lambda x: w[x], reverse=True)
                        for swap2 in items2[:3]:
                            attempts2 += 1
                            if attempts2 > max_attempts2:
                                return False
                            if time.time() - start_time > deadline:
                                return False
                            
                            ws1 = w[swap1]
                            ws2 = w[swap2]
                            
                            sr = []
                            for i in other_bins:
                                rem_val = bin_rem[i]
                                if i == ob1:
                                    rem_val += ws1
                                if i == ob2:
                                    rem_val += ws2
                                if rem_val > 0:
                                    sr.append((rem_val, i))
                            sr.sort()
                            
                            all_items = src_items + [swap1, swap2]
                            all_items_sorted = sorted(all_items, key=lambda x: w[x], reverse=True)
                            
                            ok = True
                            placements = []
                            sr_copy = list(sr)
                            
                            for item in all_items_sorted:
                                wi = w[item]
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
                                bin_items[ob1].remove(swap1)
                                bin_rem[ob1] += ws1
                                bin_items[ob2].remove(swap2)
                                bin_rem[ob2] += ws2
                                for item, target in placements:
                                    bin_items[target].append(item)
                                    bin_rem[target] -= w[item]
                                bin_items[src] = []
                                bin_rem[src] = 0
                                return True
        
        return False
    
    def consolidation(bin_items, bin_rem, deadline):
        """Move items between bins to make some bins lighter."""
        num_bins = len(bin_items)
        active = [i for i in range(num_bins) if bin_items[i]]
        if len(active) < 2:
            return False
        
        did_something = False
        
        active.sort(key=lambda i: C - bin_rem[i])
        
        for ai_idx in range(min(8, len(active))):
            i = active[ai_idx]
            if not bin_items[i]:
                continue
            if time.time() - start_time > deadline:
                break
            
            for item in list(bin_items[i]):
                wi = w[item]
                if wi == 0:
                    continue
                best_b = -1
                best_rem = C + 1
                for j in active:
                    if j == i:
                        continue
                    if bin_rem[j] >= wi and bin_rem[j] < best_rem:
                        best_rem = bin_rem[j]
                        best_b = j
                
                if best_b >= 0 and best_rem - wi < bin_rem[i] + wi:  # Actually helps
                    bin_items[i].remove(item)
                    bin_rem[i] += wi
                    bin_items[best_b].append(item)
                    bin_rem[best_b] -= wi
                    did_something = True
                    if not bin_items[i]:
                        break
            
            # 1-1 swaps to reduce weight of bin i
            if bin_items[i] and time.time() - start_time < deadline:
                for item_a in list(bin_items[i]):
                    wa = w[item_a]
                    if wa == 0:
                        continue
                    best_swap_j = -1
                    best_swap_item = -1
                    best_reduction = 0
                    for j in active:
                        if j == i:
                            continue
                        for item_b in bin_items[j]:
                            wb = w[item_b]
                            if wb >= wa:
                                continue
                            diff = wa - wb
                            if bin_rem[j] >= diff and diff > best_reduction:
                                best_reduction = diff
                                best_swap_j = j
                                best_swap_item = item_b
                    
                    if best_swap_j >= 0:
                        wb = w[best_swap_item]
                        bin_items[i].remove(item_a)
                        bin_items[best_swap_j].remove(best_swap_item)
                        bin_items[i].append(best_swap_item)
                        bin_items[best_swap_j].append(item_a)
                        bin_rem[i] += wa - wb
                        bin_rem[best_swap_j] -= wa - wb
                        did_something = True
                    
                    if time.time() - start_time > deadline:
                        break
        
        return did_something
    
    def perturb_solution(bin_items, bin_rem, num_remove=3):
        active = [i for i in range(len(bin_items)) if bin_items[i]]
        if len(active) <= 2:
            return bin_items, bin_rem
        
        num_remove = min(num_remove, len(active) - 1)
        active.sort(key=lambda i: C - bin_rem[i])
        to_remove = active[:num_remove]
        
        freed_items = []
        for b in to_remove:
            freed_items.extend(bin_items[b])
            bin_items[b] = []
            bin_rem[b] = 0
        
        freed_items.sort(key=lambda i: w[i], reverse=True)
        
        sr = []
        for i in range(len(bin_items)):
            if bin_items[i] and bin_rem[i] > 0:
                sr.append((bin_rem[i], i))
        sr.sort()
        
        for item in freed_items:
            wi = w[item]
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
        
        freed_items.sort(key=lambda i: w[i], reverse=True)
        
        sr = []
        for i in range(len(bin_items)):
            if bin_items[i] and bin_rem[i] > 0:
                sr.append((bin_rem[i], i))
        sr.sort()
        
        for item in freed_items:
            wi = w[item]
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
    
    def perturb_large_random(bin_items, bin_rem, fraction=0.3):
        """Remove a fraction of all bins randomly and repack with BFD."""
        active = [i for i in range(len(bin_items)) if bin_items[i]]
        if len(active) <= 2:
            return bin_items, bin_rem
        
        num_remove = max(2, int(len(active) * fraction))
        num_remove = min(num_remove, len(active) - 1)
        to_remove = set(random.sample(active, num_remove))
        
        freed_items = []
        for b in to_remove:
            freed_items.extend(bin_items[b])
            bin_items[b] = []
            bin_rem[b] = 0
        
        freed_items.sort(key=lambda i: w[i], reverse=True)
        
        sr = []
        for i in range(len(bin_items)):
            if bin_items[i] and bin_rem[i] > 0:
                sr.append((bin_rem[i], i))
        sr.sort()
        
        for item in freed_items:
            wi = w[item]
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
    deadline_init = start_time + time_limit * 0.15
    bi, br = local_search(bi, br, deadline_init)
    bi, br = compact(bi, br)
    best_num_bins = len(bi)
    best_bi, best_br = copy_solution(bi, br)
    
    if best_num_bins <= lower_bound:
        packing = best_bi
        bw = [C - best_br[i] for i in range(len(best_bi))]
        return {"packing": packing, "bin_weights": bw}
    
    # ---- Reactive GRASP ----
    alphas = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    alpha_counts = [0] * len(alphas)
    alpha_sum_bins = [0.0] * len(alphas)
    
    def select_alpha_reactive():
        scores = []
        for i in range(len(alphas)):
            if alpha_counts[i] > 0:
                avg = alpha_sum_bins[i] / alpha_counts[i]
                scores.append(1.0 / (avg - lower_bound + 1.0))
            else:
                scores.append(10.0)
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
    time_for_grasp = time_limit * 0.97
    
    while elapsed() < time_for_grasp:
        iteration += 1
        
        if iteration <= len(alphas):
            alpha_idx = (iteration - 1) % len(alphas)
            alpha = alphas[alpha_idx]
        else:
            alpha_idx, alpha = select_alpha_reactive()
        
        # Vary construction strategy
        r_strat = random.random()
        if r_strat < 0.5:
            bi, br = construct_bfd(alpha)
        elif r_strat < 0.75:
            # Slightly shuffled order
            order = list(sorted_indices)
            if alpha > 0:
                for i in range(len(order) - 1):
                    if random.random() < alpha * 0.3:
                        order[i], order[i+1] = order[i+1], order[i]
            bi, br = construct_bfd(0.0, order=order)
        else:
            bi, br = construct_bfd(alpha)
        
        # Adaptive time for local search
        remaining_time = time_for_grasp - elapsed()
        if remaining_time <= 0:
            break
        ls_budget = min(remaining_time * 0.4, remaining_time)
        ls_deadline = time.time() + ls_budget
        ls_deadline = min(ls_deadline, start_time + time_for_grasp)
        
        bi, br = local_search(bi, br, ls_deadline)
        bi, br = compact(bi, br)
        num_bins = len(bi)
        
        alpha_counts[alpha_idx] += 1
        alpha_sum_bins[alpha_idx] += num_bins
        
        if num_bins < best_num_bins:
            best_num_bins = num_bins
            best_bi, best_br = copy_solution(bi, br)
            no_improve = 0
            if best_num_bins <= lower_bound:
                break
        else:
            no_improve += 1
        
        # Perturbation phase when stuck
        if no_improve >= 5 and elapsed() < time_for_grasp:
            for _ in range(5):
                if elapsed() > time_for_grasp:
                    break
                p_bi, p_br = copy_solution(best_bi, best_br)
                
                r_pert = random.random()
                if r_pert < 0.3:
                    num_rem = random.randint(2, min(5, best_num_bins - 1))
                    p_bi, p_br = perturb_solution(p_bi, p_br, num_rem)
                elif r_pert < 0.7:
                    num_rem = random.randint(2, min(6, best_num_bins - 1))
                    p_bi, p_br = perturb_random(p_bi, p_br, num_rem)
                else:
                    frac = random.uniform(0.15, 0.4)
                    p_bi, p_br = perturb_large_random(p_bi, p_br, frac)
                
                remaining_time2 = time_for_grasp - elapsed()
                if remaining_time2 <= 0:
                    break
                ls_budget2 = min(remaining_time2 * 0.3, remaining_time2)
                ls_deadline2 = time.time() + ls_budget2
                ls_deadline2 = min(ls_deadline2, start_time + time_for_grasp)
                
                p_bi, p_br = local_search(p_bi, p_br, ls_deadline2)
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
    
    # Final intensive local search
    if elapsed() < time_limit * 0.99 and best_num_bins > lower_bound:
        bi, br = copy_solution(best_bi, best_br)
        final_deadline = start_time + time_limit * 0.99
        bi, br = local_search(bi, br, final_deadline)
        bi, br = compact(bi, br)
        if len(bi) < best_num_bins:
            best_num_bins = len(bi)
            best_bi, best_br = bi, br
    
    packing = best_bi
    bw = [C - best_br[i] for i in range(len(best_bi))]
    return {"packing": packing, "bin_weights": bw}