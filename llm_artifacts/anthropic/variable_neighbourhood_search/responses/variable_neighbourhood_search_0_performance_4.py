import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    W = weights
    total_weight = sum(W)
    
    def elapsed():
        return time.time() - start_time
    
    # Lower bound
    continuous_lb = (total_weight + C - 1) // C
    
    def compute_L2():
        best_lb = continuous_lb
        half_C = C / 2.0
        sorted_w = sorted(W, reverse=True)
        
        alphas = set()
        for a in range(1, min(C // 2 + 1, 301)):
            alphas.add(a)
        for w in W:
            a = C - w
            if 1 <= a <= C // 2:
                alphas.add(a)
                if a + 1 <= C // 2:
                    alphas.add(a + 1)
                if a - 1 >= 1:
                    alphas.add(a - 1)
        
        for alpha in sorted(alphas):
            if elapsed() > time_limit * 0.02:
                break
            threshold_large = C - alpha
            n_large = 0
            sum_large = 0
            n_medium = 0
            sum_medium = 0
            sum_small = 0
            
            for w in sorted_w:
                if w > threshold_large:
                    n_large += 1
                    sum_large += w
                elif w > half_C:
                    n_medium += 1
                    sum_medium += w
                elif w >= alpha:
                    sum_small += w
            
            residual_large = n_large * C - sum_large
            residual_medium = n_medium * C - sum_medium
            remaining_small = max(0, sum_small - residual_large - residual_medium)
            
            lb = n_large + n_medium
            if remaining_small > 0:
                lb += (remaining_small + C - 1) // C
            
            if lb > best_lb:
                best_lb = lb
        
        return best_lb
    
    lower_bound = compute_L2()
    
    def best_fit_decreasing(order):
        assignment = [0] * n
        bin_items_list = []
        bin_wts_list = []
        residual_list = []
        num_bins = 0
        
        for idx in order:
            w = W[idx]
            best_b = -1
            best_rem = C + 1
            for b in range(num_bins):
                r = residual_list[b]
                if r >= w and r < best_rem:
                    best_b = b
                    best_rem = r
                    if r == w:
                        break
            if best_b >= 0:
                assignment[idx] = best_b
                bin_items_list[best_b].append(idx)
                bin_wts_list[best_b] += w
                residual_list[best_b] -= w
            else:
                assignment[idx] = num_bins
                bin_items_list.append([idx])
                bin_wts_list.append(w)
                residual_list.append(C - w)
                num_bins += 1
        
        return assignment, bin_items_list, bin_wts_list, residual_list, num_bins
    
    def first_fit_decreasing(order):
        assignment = [0] * n
        bin_items_list = []
        bin_wts_list = []
        residual_list = []
        num_bins = 0
        
        for idx in order:
            w = W[idx]
            placed = False
            for b in range(num_bins):
                if residual_list[b] >= w:
                    assignment[idx] = b
                    bin_items_list[b].append(idx)
                    bin_wts_list[b] += w
                    residual_list[b] -= w
                    placed = True
                    break
            if not placed:
                assignment[idx] = num_bins
                bin_items_list.append([idx])
                bin_wts_list.append(w)
                residual_list.append(C - w)
                num_bins += 1
        
        return assignment, bin_items_list, bin_wts_list, residual_list, num_bins
    
    def worst_fit_decreasing(order):
        assignment = [0] * n
        bin_items_list = []
        bin_wts_list = []
        residual_list = []
        num_bins = 0
        
        for idx in order:
            w = W[idx]
            best_b = -1
            best_rem = -1
            for b in range(num_bins):
                r = residual_list[b]
                if r >= w and r > best_rem:
                    best_b = b
                    best_rem = r
            if best_b >= 0:
                assignment[idx] = best_b
                bin_items_list[best_b].append(idx)
                bin_wts_list[best_b] += w
                residual_list[best_b] -= w
            else:
                assignment[idx] = num_bins
                bin_items_list.append([idx])
                bin_wts_list.append(w)
                residual_list.append(C - w)
                num_bins += 1
        
        return assignment, bin_items_list, bin_wts_list, residual_list, num_bins
    
    def sol_to_sets(assignment, bin_items_list, bin_wts_list, residual_list, num_bins):
        bi = [set(items) for items in bin_items_list]
        return list(assignment), bi, list(bin_wts_list), list(residual_list), num_bins
    
    def copy_sol(assignment, bi, bw, res, nb):
        return list(assignment), [set(s) for s in bi], list(bw), list(res), nb
    
    def compact_sol(assignment, bi, bw, res, nb):
        active = [b for b in range(len(bi)) if len(bi[b]) > 0]
        mapping = {}
        new_bi = []
        new_bw = []
        new_res = []
        for new_idx, old_idx in enumerate(active):
            mapping[old_idx] = new_idx
            new_bi.append(bi[old_idx])
            new_bw.append(bw[old_idx])
            new_res.append(res[old_idx])
        new_assignment = [0] * n
        for i in range(n):
            new_assignment[i] = mapping[assignment[i]]
        return new_assignment, new_bi, new_bw, new_res, len(active)
    
    def to_output(assignment, bi, bw, res, nb):
        assignment, bi, bw, res, nb = compact_sol(assignment, bi, bw, res, nb)
        packing = [sorted(bi[b]) for b in range(nb)]
        bin_weights = [bw[b] for b in range(nb)]
        return {"packing": packing, "bin_weights": bin_weights}
    
    def move_item(assignment, bi, bw, res, item, tgt):
        src = assignment[item]
        w = W[item]
        bi[src].discard(item)
        bw[src] -= w
        res[src] += w
        assignment[item] = tgt
        bi[tgt].add(item)
        bw[tgt] += w
        res[tgt] -= w
    
    def try_empty_bin_thorough(assignment, bi, bw, res, nb, src, time_lim):
        """Try to empty bin src using moves, (1,1) swaps, and (1,2) moves."""
        if len(bi[src]) == 0:
            return True
        
        src_items = sorted(bi[src], key=lambda i: -W[i])
        src_weight = bw[src]
        
        # Build list of other active bins
        others = [b for b in range(len(bi)) if len(bi[b]) > 0 and b != src]
        
        # Try direct moves first (greedy, largest items first)
        # We'll do a recursive/backtracking approach for small bins
        # For efficiency, limit depth
        
        if len(src_items) <= 20:
            # Try to place all items using combination of direct moves and swaps
            # Use iterative deepening with swap budget
            result = _try_empty_recursive(assignment, bi, bw, res, src, list(src_items), 0, others, time_lim, 0, 3)
            if result:
                return True
        
        return False
    
    def _try_empty_recursive(assignment, bi, bw, res, src, items, idx, others, time_lim, swaps_used, max_swaps):
        if idx >= len(items):
            return True
        if elapsed() > time_lim:
            return False
        
        item = items[idx]
        w = W[item]
        
        # Try direct move (best fit)
        candidates = []
        for b in others:
            if len(bi[b]) > 0 and res[b] >= w:
                candidates.append((res[b], b))
        candidates.sort()
        
        for _, b in candidates:
            if res[b] >= w:
                move_item(assignment, bi, bw, res, item, b)
                if _try_empty_recursive(assignment, bi, bw, res, src, items, idx + 1, others, time_lim, swaps_used, max_swaps):
                    return True
                move_item(assignment, bi, bw, res, item, src)
                if elapsed() > time_lim:
                    return False
                break  # Only try best-fit direct move
        
        # Try (1,1) swap: move item to bin b, move some item from b to another bin
        if swaps_used < max_swaps:
            for b in others:
                if len(bi[b]) == 0:
                    continue
                if elapsed() > time_lim:
                    return False
                needed = w - res[b]
                if needed <= 0:
                    continue  # already tried direct
                
                # Find item in b with weight >= needed and weight < w
                # that can fit somewhere else after being displaced
                b_items = sorted(bi[b], key=lambda i: W[i])
                for t_item in b_items:
                    wt = W[t_item]
                    if wt < needed:
                        continue
                    if wt >= w:
                        break  # no point, sorted ascending
                    # Check if t_item can go somewhere
                    # After removing t_item from b: res[b] + wt >= w, so item fits
                    # t_item needs a place
                    best_dest = -1
                    best_dest_r = C + 1
                    for b2 in others:
                        if b2 == b:
                            continue
                        if len(bi[b2]) == 0:
                            continue
                        if res[b2] >= wt and res[b2] < best_dest_r:
                            best_dest = b2
                            best_dest_r = res[b2]
                            if best_dest_r == wt:
                                break
                    # Also check if t_item fits in src (remaining items will also leave)
                    # Actually src will be emptied so don't put items there
                    
                    if best_dest >= 0:
                        # Do swap: move t_item from b to best_dest, move item from src to b
                        move_item(assignment, bi, bw, res, t_item, best_dest)
                        move_item(assignment, bi, bw, res, item, b)
                        if _try_empty_recursive(assignment, bi, bw, res, src, items, idx + 1, others, time_lim, swaps_used + 1, max_swaps):
                            return True
                        # Undo
                        move_item(assignment, bi, bw, res, item, src)
                        move_item(assignment, bi, bw, res, t_item, b)
                        if elapsed() > time_lim:
                            return False
                        break  # Only try first valid swap per bin
        
        return False
    
    def local_search(assignment, bi, bw, res, nb, time_frac=0.95):
        time_lim = start_time + time_limit * time_frac
        
        # Phase 1: Try to empty bins
        improved = True
        while improved and elapsed() < time_limit * time_frac:
            improved = False
            if nb <= lower_bound:
                break
            
            active = [b for b in range(len(bi)) if len(bi[b]) > 0]
            active.sort(key=lambda b: bw[b])
            
            for src in active:
                if elapsed() > time_limit * time_frac:
                    break
                if len(bi[src]) == 0:
                    continue
                
                result = try_empty_bin_thorough(assignment, bi, bw, res, nb, src, time_lim)
                if result and len(bi[src]) == 0:
                    nb = sum(1 for b in range(len(bi)) if len(bi[b]) > 0)
                    improved = True
                    break
        
        # Phase 2: (1,1) swaps to concentrate waste
        if nb > lower_bound:
            active = [b for b in range(len(bi)) if len(bi[b]) > 0]
            max_iters = min(len(active) * len(active) * 2, 2000)
            for _ in range(max_iters):
                if elapsed() > time_limit * time_frac:
                    break
                if len(active) < 2:
                    break
                b1, b2 = random.sample(active, 2)
                if len(bi[b1]) == 0 or len(bi[b2]) == 0:
                    continue
                item1 = random.choice(list(bi[b1]))
                item2 = random.choice(list(bi[b2]))
                w1, w2 = W[item1], W[item2]
                new_r1 = res[b1] + w1 - w2
                new_r2 = res[b2] + w2 - w1
                if new_r1 >= 0 and new_r2 >= 0:
                    old_waste = res[b1] * res[b1] + res[b2] * res[b2]
                    new_waste = new_r1 * new_r1 + new_r2 * new_r2
                    if new_waste > old_waste:
                        bi[b1].discard(item1)
                        bi[b1].add(item2)
                        bw[b1] += w2 - w1
                        res[b1] = new_r1
                        bi[b2].discard(item2)
                        bi[b2].add(item1)
                        bw[b2] += w1 - w2
                        res[b2] = new_r2
                        assignment[item1] = b2
                        assignment[item2] = b1
            
            # Phase 2b: (1,0) moves to concentrate waste
            active = [b for b in range(len(bi)) if len(bi[b]) > 0]
            for _ in range(max_iters):
                if elapsed() > time_limit * time_frac:
                    break
                if len(active) < 2:
                    break
                b1 = random.choice(active)
                if len(bi[b1]) == 0:
                    continue
                item1 = random.choice(list(bi[b1]))
                w1 = W[item1]
                # Find best bin to move to
                best_b = -1
                best_rem = C + 1
                for b2 in active:
                    if b2 == b1 or len(bi[b2]) == 0:
                        continue
                    r = res[b2]
                    if r >= w1 and r < best_rem:
                        best_b = b2
                        best_rem = r
                        if r == w1:
                            break
                if best_b >= 0:
                    old_waste = res[b1] * res[b1] + res[best_b] * res[best_b]
                    new_r1 = res[b1] + w1
                    new_r2 = res[best_b] - w1
                    new_waste = new_r1 * new_r1 + new_r2 * new_r2
                    if new_waste > old_waste:
                        move_item(assignment, bi, bw, res, item1, best_b)
        
        # Phase 3: Try emptying again after swaps
        improved2 = True
        while improved2 and elapsed() < time_limit * time_frac:
            improved2 = False
            if nb <= lower_bound:
                break
            active = [b for b in range(len(bi)) if len(bi[b]) > 0]
            active.sort(key=lambda b: bw[b])
            for src in active:
                if elapsed() > time_limit * time_frac:
                    break
                if len(bi[src]) == 0:
                    continue
                result = try_empty_bin_thorough(assignment, bi, bw, res, nb, src, time_lim)
                if result and len(bi[src]) == 0:
                    nb = sum(1 for b in range(len(bi)) if len(bi[b]) > 0)
                    improved2 = True
                    break
        
        return assignment, bi, bw, res, nb
    
    # --- Initial Solutions ---
    desc_order = sorted(range(n), key=lambda i: -W[i])
    
    a, bil, bwl, rl, nbl = best_fit_decreasing(desc_order)
    best_a, best_bi, best_bw, best_res, best_nb = sol_to_sets(a, bil, bwl, rl, nbl)
    
    if best_nb > lower_bound:
        a2, bil2, bwl2, rl2, nbl2 = first_fit_decreasing(desc_order)
        if nbl2 < best_nb:
            best_a, best_bi, best_bw, best_res, best_nb = sol_to_sets(a2, bil2, bwl2, rl2, nbl2)
    
    # Try multiple random orderings
    if best_nb > lower_bound:
        for _ in range(20):
            if elapsed() > time_limit * 0.05:
                break
            noise = max(1, C // 5)
            noisy_order = sorted(range(n), key=lambda i: -W[i] + random.randint(-noise, noise))
            a3, bil3, bwl3, rl3, nbl3 = best_fit_decreasing(noisy_order)
            if nbl3 < best_nb:
                best_a, best_bi, best_bw, best_res, best_nb = sol_to_sets(a3, bil3, bwl3, rl3, nbl3)
                if best_nb <= lower_bound:
                    break
    
    if best_nb <= lower_bound:
        return to_output(best_a, best_bi, best_bw, best_res, best_nb)
    
    # Apply initial local search
    cur_a, cur_bi, cur_bw, cur_res, cur_nb = copy_sol(best_a, best_bi, best_bw, best_res, best_nb)
    cur_a, cur_bi, cur_bw, cur_res, cur_nb = local_search(cur_a, cur_bi, cur_bw, cur_res, cur_nb, time_frac=0.25)
    
    if cur_nb < best_nb:
        best_a, best_bi, best_bw, best_res, best_nb = copy_sol(cur_a, cur_bi, cur_bw, cur_res, cur_nb)
    
    if best_nb <= lower_bound:
        return to_output(best_a, best_bi, best_bw, best_res, best_nb)
    
    # --- Shaking / Perturbation ---
    def destroy_and_rebuild(assignment, bi, bw, res, nb, frac, mode='bf', destroy_mode='light'):
        assignment, bi, bw, res, nb = compact_sol(assignment, bi, bw, res, nb)
        active = list(range(nb))
        
        num_destroy = max(1, int(nb * frac))
        num_destroy = min(num_destroy, nb)
        
        if destroy_mode == 'light':
            active_sorted = sorted(active, key=lambda b: bw[b])
            to_destroy = set(active_sorted[:num_destroy])
        elif destroy_mode == 'random':
            to_destroy = set(random.sample(active, min(num_destroy, len(active))))
        elif destroy_mode == 'heavy':
            active_sorted = sorted(active, key=lambda b: bw[b], reverse=True)
            to_destroy = set(active_sorted[:num_destroy])
        else:
            to_destroy = set(random.sample(active, min(num_destroy, len(active))))
        
        all_items = []
        for b in to_destroy:
            all_items.extend(bi[b])
        
        for item in all_items:
            b = assignment[item]
            bi[b].discard(item)
            w = W[item]
            bw[b] -= w
            res[b] += w
        
        all_items.sort(key=lambda i: -W[i])
        
        to_destroy_avail = set(to_destroy)
        
        for item in all_items:
            w = W[item]
            best_b = -1
            best_rem = C + 1
            for b in range(len(bi)):
                if len(bi[b]) == 0:
                    continue
                r = res[b]
                if r >= w and r < best_rem:
                    best_b = b
                    best_rem = r
                    if r == w:
                        break
            if best_b >= 0:
                assignment[item] = best_b
                bi[best_b].add(item)
                bw[best_b] += w
                res[best_b] -= w
            else:
                found = False
                for b in list(to_destroy_avail):
                    if len(bi[b]) == 0:
                        assignment[item] = b
                        bi[b].add(item)
                        bw[b] = w
                        res[b] = C - w
                        to_destroy_avail.discard(b)
                        found = True
                        break
                if not found:
                    new_b = len(bi)
                    bi.append(set([item]))
                    bw.append(w)
                    res.append(C - w)
                    assignment[item] = new_b
        
        nb = sum(1 for b in range(len(bi)) if len(bi[b]) > 0)
        return assignment, bi, bw, res, nb
    
    def shake(assignment, bi, bw, res, nb, k):
        assignment, bi, bw, res, nb = copy_sol(assignment, bi, bw, res, nb)
        
        if k == 1:
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 1.0 / max(nb, 1), destroy_mode='light')
        elif k == 2:
            frac = min(2.0 / max(nb, 1), 0.5)
            return destroy_and_rebuild(assignment, bi, bw, res, nb, frac, destroy_mode='light')
        elif k == 3:
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.15, destroy_mode='light')
        elif k == 4:
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.25, destroy_mode='light')
        elif k == 5:
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.40, destroy_mode='light')
        elif k == 6:
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.20, destroy_mode='random')
        elif k == 7:
            frac = random.uniform(0.25, 0.50)
            return destroy_and_rebuild(assignment, bi, bw, res, nb, frac, destroy_mode='random')
        elif k == 8:
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.60, destroy_mode='random')
        elif k == 9:
            # Full rebuild with noise
            noise = max(1, C // 4)
            noisy_order = sorted(range(n), key=lambda i: -W[i] + random.randint(-noise, noise))
            a, bil, bwl, rl, nbl = best_fit_decreasing(noisy_order)
            return sol_to_sets(a, bil, bwl, rl, nbl)
        elif k == 10:
            # Destroy light bins, rebuild with first fit
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.30, destroy_mode='light')
        else:
            noise = max(1, C // 3)
            noisy_order = sorted(range(n), key=lambda i: -W[i] + random.randint(-noise, noise))
            a, bil, bwl, rl, nbl = best_fit_decreasing(noisy_order)
            return sol_to_sets(a, bil, bwl, rl, nbl)
    
    # --- VNS Main Loop ---
    k_max = 11
    k = 1
    no_improve_count = 0
    max_no_improve = 40
    
    while elapsed() < time_limit * 0.97:
        if best_nb <= lower_bound:
            break
        
        # Shaking
        s_a, s_bi, s_bw, s_res, s_nb = shake(cur_a, cur_bi, cur_bw, cur_res, cur_nb, k)
        
        if elapsed() > time_limit * 0.97:
            if s_nb < best_nb:
                best_a, best_bi, best_bw, best_res, best_nb = s_a, s_bi, s_bw, s_res, s_nb
            break
        
        # Local search with adaptive time
        remaining = time_limit * 0.97 - elapsed()
        ls_time = min(remaining * 0.3, 5.0)
        ls_frac = min(0.97, (elapsed() + ls_time) / time_limit)
        ls_a, ls_bi, ls_bw, ls_res, ls_nb = local_search(s_a, s_bi, s_bw, s_res, s_nb, time_frac=ls_frac)
        
        if ls_nb < cur_nb:
            cur_a, cur_bi, cur_bw, cur_res, cur_nb = copy_sol(ls_a, ls_bi, ls_bw, ls_res, ls_nb)
            k = 1
            no_improve_count = 0
            if ls_nb < best_nb:
                best_a, best_bi, best_bw, best_res, best_nb = copy_sol(ls_a, ls_bi, ls_bw, ls_res, ls_nb)
                if best_nb <= lower_bound:
                    break
        elif ls_nb == cur_nb:
            if random.random() < 0.5:
                cur_a, cur_bi, cur_bw, cur_res, cur_nb = copy_sol(ls_a, ls_bi, ls_bw, ls_res, ls_nb)
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
            noise = max(1, C // 4)
            noisy_order = sorted(range(n), key=lambda i: -W[i] + random.randint(-noise, noise))
            a, bil, bwl, rl, nbl = best_fit_decreasing(noisy_order)
            cur_a, cur_bi, cur_bw, cur_res, cur_nb = sol_to_sets(a, bil, bwl, rl, nbl)
            remaining2 = time_limit * 0.97 - elapsed()
            ls_time2 = min(remaining2 * 0.3, 3.0)
            ls_frac2 = min(0.97, (elapsed() + ls_time2) / time_limit)
            cur_a, cur_bi, cur_bw, cur_res, cur_nb = local_search(cur_a, cur_bi, cur_bw, cur_res, cur_nb, time_frac=ls_frac2)
            if cur_nb < best_nb:
                best_a, best_bi, best_bw, best_res, best_nb = copy_sol(cur_a, cur_bi, cur_bw, cur_res, cur_nb)
            k = 1
            no_improve_count = 0
    
    return to_output(best_a, best_bi, best_bw, best_res, best_nb)
