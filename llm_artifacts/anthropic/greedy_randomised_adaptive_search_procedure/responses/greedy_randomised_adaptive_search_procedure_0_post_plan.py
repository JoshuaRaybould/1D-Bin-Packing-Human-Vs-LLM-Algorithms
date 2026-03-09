import random
import time
import math

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    
    def elapsed():
        return time.time() - start_time
    
    # Sort indices by weight descending
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    # Precompute weights array for fast access
    w_arr = weights  # alias
    
    # Lower bound computation (L2 - Martello & Toth style)
    total_weight = sum(weights)
    L1 = (total_weight + C - 1) // C  # ceil division
    
    # L2 lower bound
    def compute_L2():
        best_lb = L1
        # Try different thresholds
        for k in range(1, C // 2 + 1):
            # Items > C - k (large), items in (C/2, C-k] (medium-ish), items in [k, C/2]
            n1 = 0  # items with weight > C - k
            n2 = 0  # items with weight in (C//2, C - k]
            s3 = 0  # total weight of items in [k, C//2]
            n3 = 0
            for i in range(n):
                wi = w_arr[i]
                if wi > C - k:
                    n1 += 1
                elif wi > C // 2:
                    n2 += 1
                elif wi >= k:
                    s3 += wi
                    n3 += 1
            # Each n1 item needs its own bin, n2 items might pair up
            # Remaining space in n1 bins can absorb some small items
            waste_n1 = n1 * C - sum(wi for wi in weights if wi > C - k)
            # n2 items: each needs a bin, pairs don't fit (both > C/2)
            # Actually the L2 bound: LB = n1 + n2 + max(0, ceil((s3 - waste from n1 bins unused by n2) / C))
            lb = n1 + n2 + max(0, math.ceil((s3 - (n1 * C - sum(wi for wi in weights if wi > C - k) - 0)) / C))
            # Simplified: just use L1 and a quick estimate
            if lb > best_lb:
                best_lb = lb
        return best_lb
    
    # Simpler but effective L2
    def compute_lower_bound():
        best_lb = L1
        half = C / 2.0
        for k in range(1, min(C // 2 + 1, 100)):
            n1 = 0
            n1_weight = 0
            n2 = 0
            n2_weight = 0  
            s3 = 0
            for i in range(n):
                wi = w_arr[i]
                if wi > C - k:
                    n1 += 1
                    n1_weight += wi
                elif wi > C // 2:
                    n2 += 1
                    n2_weight += wi
                elif wi >= k:
                    s3 += wi
            # Space left in n1 bins after placing n1 items
            space_n1 = n1 * C - n1_weight
            # s3 items that can't fit in n1 leftover
            remaining_s3 = max(0, s3 - space_n1)
            lb = n1 + n2 + (remaining_s3 + C - 1) // C if remaining_s3 > 0 else n1 + n2
            # Also need at least ceil((n1_weight+n2_weight+s3)/C)
            lb = max(lb, L1)
            if lb > best_lb:
                best_lb = lb
        return best_lb
    
    lower_bound = L1
    if n <= 50000:
        try:
            lower_bound = compute_lower_bound()
        except:
            lower_bound = L1
    
    best_packing_items = None
    best_packing_rem = None
    best_num_bins = n + 1
    
    def construct_solution(alpha, use_ffd=False):
        """Greedy randomized construction."""
        bin_items = []
        bin_remaining = []
        
        for idx in sorted_indices:
            w = w_arr[idx]
            if w > C:
                bin_items.append([idx])
                bin_remaining.append(C - w)
                continue
            
            if not bin_items:
                bin_items.append([idx])
                bin_remaining.append(C - w)
                continue
            
            # Find candidates
            candidates = []
            for b_idx in range(len(bin_remaining)):
                rem = bin_remaining[b_idx]
                if rem >= w:
                    if use_ffd:
                        # For FFD: score is bin index (prefer earlier bins)
                        candidates.append((b_idx, b_idx))
                    else:
                        # For BFD: score is remaining after placing (lower=tighter)
                        candidates.append((b_idx, rem - w))
            
            if not candidates:
                bin_items.append([idx])
                bin_remaining.append(C - w)
            else:
                if alpha == 0.0:
                    # Deterministic: pick best
                    best_b = min(candidates, key=lambda c: c[1])
                    chosen = best_b[0]
                else:
                    best_val = min(c[1] for c in candidates)
                    worst_val = max(c[1] for c in candidates)
                    threshold = best_val + alpha * (worst_val - best_val)
                    rcl = [c for c in candidates if c[1] <= threshold]
                    if not rcl:
                        rcl = candidates[:1]
                    chosen = random.choice(rcl)[0]
                
                bin_items[chosen].append(idx)
                bin_remaining[chosen] -= w
        
        return bin_items, bin_remaining
    
    def local_search(bin_items, bin_remaining):
        """Try to empty bins by redistributing items."""
        improved = True
        while improved:
            improved = False
            if elapsed() > time_limit * 0.95:
                break
            
            num_bins = len(bin_items)
            # Sort bins by weight ascending (lightest first to try to empty)
            bin_weights_list = [(C - bin_remaining[i], i) for i in range(num_bins)]
            bin_weights_list.sort()
            
            emptied = set()
            
            for bw, src_idx in bin_weights_list:
                if src_idx in emptied:
                    continue
                if not bin_items[src_idx]:
                    continue
                if bw == 0:
                    continue
                if elapsed() > time_limit * 0.95:
                    break
                
                src_items_sorted = sorted(bin_items[src_idx], key=lambda x: w_arr[x], reverse=True)
                
                # Try to redistribute all items using best-fit
                temp_rem = list(bin_remaining)
                moves = {}  # item -> target
                success = True
                
                for item_idx in src_items_sorted:
                    w = w_arr[item_idx]
                    best_target = -1
                    best_rem_after = C + 1
                    for t_idx in range(num_bins):
                        if t_idx == src_idx or t_idx in emptied:
                            continue
                        if temp_rem[t_idx] >= w and temp_rem[t_idx] - w < best_rem_after:
                            best_rem_after = temp_rem[t_idx] - w
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
        """Enhanced swap-based local search with displacement chains."""
        improved = True
        max_rounds = 5
        round_count = 0
        
        while improved and round_count < max_rounds:
            improved = False
            round_count += 1
            
            if elapsed() > time_limit * 0.93:
                break
            
            num_bins = len(bin_items)
            bin_weights_list = [(C - bin_remaining[i], i) for i in range(num_bins)]
            bin_weights_list.sort()
            
            candidates_to_try = min(max(1, num_bins // 3), 10)
            
            for bw, src_idx in bin_weights_list[:candidates_to_try]:
                if elapsed() > time_limit * 0.93:
                    break
                if not bin_items[src_idx]:
                    continue
                
                # Try to empty src_idx using displacement chains
                temp_rem = {i: bin_remaining[i] for i in range(num_bins) if i != src_idx}
                temp_items = {i: list(bin_items[i]) for i in range(num_bins) if i != src_idx}
                remaining_to_place = sorted(list(bin_items[src_idx]), key=lambda x: w_arr[x], reverse=True)
                
                max_chain_depth = 4
                
                moved = True
                iteration_count = 0
                while moved and remaining_to_place:
                    moved = False
                    iteration_count += 1
                    if iteration_count > len(remaining_to_place) * max_chain_depth + 10:
                        break
                    if elapsed() > time_limit * 0.93:
                        break
                    
                    for ri, item_idx in enumerate(remaining_to_place):
                        w = w_arr[item_idx]
                        
                        # Direct placement (best fit)
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
                            moved = True
                            break
                        
                        # Displacement: find target bin where swapping out an item makes room
                        best_swap = None
                        best_displaced_w = float('inf')
                        
                        for t_idx, t_items in temp_items.items():
                            t_rem = temp_rem[t_idx]
                            for swap_item in t_items:
                                sw = w_arr[swap_item]
                                if sw < w and t_rem + sw >= w:
                                    # Prefer displacing lightest item
                                    if sw < best_displaced_w:
                                        best_displaced_w = sw
                                        best_swap = (t_idx, swap_item)
                        
                        if best_swap is not None:
                            t_idx, swap_item = best_swap
                            sw = w_arr[swap_item]
                            temp_rem[t_idx] = temp_rem[t_idx] + sw - w
                            temp_items[t_idx].remove(swap_item)
                            temp_items[t_idx].append(item_idx)
                            remaining_to_place.pop(ri)
                            remaining_to_place.append(swap_item)
                            moved = True
                            break
                
                if not remaining_to_place:
                    # Successfully emptied
                    new_items = []
                    new_rem = []
                    for i in range(num_bins):
                        if i == src_idx:
                            continue
                        if i in temp_items:
                            new_items.append(temp_items[i])
                            new_rem.append(temp_rem[i])
                    bin_items = new_items
                    bin_remaining = new_rem
                    improved = True
                    break
        
        return bin_items, bin_remaining
    
    # Reactive GRASP alpha setup
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
            # Normalize
            total_p = sum(alpha_probs.values())
            for a in alphas:
                alpha_probs[a] /= total_p
    
    # Elite pool
    elite_pool = []  # list of (num_bins, bin_items, bin_remaining)
    max_elite = 5
    
    def add_to_elite(num_bins, b_items, b_rem):
        # Deep copy
        items_copy = [list(bi) for bi in b_items]
        rem_copy = list(b_rem)
        
        if len(elite_pool) < max_elite:
            elite_pool.append((num_bins, items_copy, rem_copy))
            elite_pool.sort(key=lambda x: x[0])
        elif num_bins < elite_pool[-1][0]:
            elite_pool[-1] = (num_bins, items_copy, rem_copy)
            elite_pool.sort(key=lambda x: x[0])
    
    def path_relink(sol1_items, sol1_rem, sol2_items, sol2_rem):
        """Path relinking between two solutions."""
        if elapsed() > time_limit * 0.90:
            return None
        
        # Represent as item-to-bin assignment
        assign1 = [0] * n
        for b_idx, items in enumerate(sol1_items):
            for it in items:
                assign1[it] = b_idx
        
        assign2 = [0] * n
        for b_idx, items in enumerate(sol2_items):
            for it in items:
                assign2[it] = b_idx
        
        # Work from sol1 toward sol2
        # Current assignment starts as sol1
        cur_items = [list(bi) for bi in sol1_items]
        cur_rem = list(sol1_rem)
        cur_assign = list(assign1)
        
        # Find items that differ
        diff_items = [i for i in range(n) if assign1[i] != assign2[i]]
        random.shuffle(diff_items)
        
        best_local_bins = len(cur_items)
        best_local_items = None
        best_local_rem = None
        
        steps = min(len(diff_items), 50)
        
        for step_i in range(steps):
            if elapsed() > time_limit * 0.90:
                break
            
            item = diff_items[step_i]
            target_bin_in_sol2 = assign2[item]
            
            # Move item from current bin to its target in sol2
            # But bin indices may not correspond, so just try moving to a bin
            # that would reduce total bins
            # Simple approach: remove item from its current bin, place in best-fit among others
            cur_bin = cur_assign[item]
            w = w_arr[item]
            
            # Remove from current bin
            if item in cur_items[cur_bin]:
                cur_items[cur_bin].remove(item)
                cur_rem[cur_bin] += w
            
            # Place in best fit among other bins
            best_target = -1
            best_rem_after = C + 1
            for t_idx in range(len(cur_items)):
                if t_idx == cur_bin or not cur_items[t_idx]:
                    continue
                if cur_rem[t_idx] >= w and cur_rem[t_idx] - w < best_rem_after:
                    best_rem_after = cur_rem[t_idx] - w
                    best_target = t_idx
            
            if best_target != -1:
                cur_items[best_target].append(item)
                cur_rem[best_target] -= w
                cur_assign[item] = best_target
            else:
                # Put back
                cur_items[cur_bin].append(item)
                cur_rem[cur_bin] -= w
                continue
            
            # Check if current bin is now empty
            if not cur_items[cur_bin]:
                # Remove empty bin, update indices
                new_items = []
                new_rem = []
                idx_map = {}
                new_idx = 0
                for i in range(len(cur_items)):
                    if cur_items[i]:
                        idx_map[i] = new_idx
                        new_items.append(cur_items[i])
                        new_rem.append(cur_rem[i])
                        new_idx += 1
                cur_items = new_items
                cur_rem = new_rem
                # Update assignments
                for it_idx in range(n):
                    if cur_assign[it_idx] in idx_map:
                        cur_assign[it_idx] = idx_map[cur_assign[it_idx]]
                
                cur_bins_count = len(cur_items)
                if cur_bins_count < best_local_bins:
                    best_local_bins = cur_bins_count
                    best_local_items = [list(bi) for bi in cur_items]
                    best_local_rem = list(cur_rem)
        
        if best_local_items is not None:
            # Apply local search
            best_local_items, best_local_rem = local_search(best_local_items, best_local_rem)
            return (len(best_local_items), best_local_items, best_local_rem)
        return None
    
    # Initial deterministic solution (BFD, alpha=0)
    bi, br = construct_solution(0.0, use_ffd=False)
    bi, br = local_search(bi, br)
    num_b = len(bi)
    if num_b < best_num_bins:
        best_num_bins = num_b
        best_packing_items = [list(b) for b in bi]
        best_packing_rem = list(br)
        add_to_elite(num_b, bi, br)
    
    if best_num_bins <= lower_bound:
        packing = best_packing_items
        bin_weights_out = [C - best_packing_rem[i] for i in range(len(best_packing_rem))]
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Also try FFD
    bi2, br2 = construct_solution(0.0, use_ffd=True)
    bi2, br2 = local_search(bi2, br2)
    num_b2 = len(bi2)
    if num_b2 < best_num_bins:
        best_num_bins = num_b2
        best_packing_items = [list(b) for b in bi2]
        best_packing_rem = list(br2)
    add_to_elite(num_b2, bi2, br2)
    
    if best_num_bins <= lower_bound:
        packing = best_packing_items
        bin_weights_out = [C - best_packing_rem[i] for i in range(len(best_packing_rem))]
        return {"packing": packing, "bin_weights": bin_weights_out}
    
    # Main GRASP loop
    iteration = 0
    update_interval = 10
    relink_interval = 20
    
    while elapsed() < time_limit * 0.92:
        # Select alpha (reactive)
        if iteration < len(alphas):
            alpha = alphas[iteration % len(alphas)]
        else:
            alpha = select_alpha()
        
        # Alternate between BFD and FFD
        use_ffd = random.random() < 0.3
        
        # Construction
        bi, br = construct_solution(alpha, use_ffd=use_ffd)
        
        # Local search (essential for GRASP)
        bi, br = local_search(bi, br)
        num_b = len(bi)
        
        # Deeper local search if promising
        if num_b <= best_num_bins + 2 and elapsed() < time_limit * 0.88:
            bi, br = local_search_swap(bi, br)
            num_b = len(bi)
        
        # Update alpha tracking
        alpha_count[alpha] += 1
        alpha_quality_sum[alpha] += 1.0 / num_b
        
        if num_b < best_num_bins:
            best_num_bins = num_b
            best_packing_items = [list(b) for b in bi]
            best_packing_rem = list(br)
        
        add_to_elite(num_b, bi, br)
        
        # Check optimality
        if best_num_bins <= lower_bound:
            break
        
        # Update alpha probabilities
        if iteration > 0 and iteration % update_interval == 0:
            update_alpha_probs()
        
        # Path relinking
        if iteration > 0 and iteration % relink_interval == 0 and len(elite_pool) >= 2 and elapsed() < time_limit * 0.85:
            # Pick best and a random other
            best_elite = elite_pool[0]
            other_idx = random.randint(1, len(elite_pool) - 1)
            other_elite = elite_pool[other_idx]
            
            result = path_relink(best_elite[1], best_elite[2], other_elite[1], other_elite[2])
            if result is not None:
                pr_bins, pr_items, pr_rem = result
                if pr_bins < best_num_bins:
                    best_num_bins = pr_bins
                    best_packing_items = [list(b) for b in pr_items]
                    best_packing_rem = list(pr_rem)
                add_to_elite(pr_bins, pr_items, pr_rem)
                
                if best_num_bins <= lower_bound:
                    break
        
        iteration += 1
    
    # Final intensification on best solution
    if elapsed() < time_limit * 0.96 and best_packing_items is not None:
        bi, br = local_search_swap(best_packing_items, best_packing_rem)
        if len(bi) < best_num_bins:
            best_num_bins = len(bi)
            best_packing_items = bi
            best_packing_rem = br
    
    if best_packing_items is None:
        bi, br = construct_solution(0.0)
        best_packing_items = bi
        best_packing_rem = br
    
    packing = best_packing_items
    bin_weights_out = [C - best_packing_rem[i] for i in range(len(best_packing_rem))]
    
    return {"packing": packing, "bin_weights": bin_weights_out}
