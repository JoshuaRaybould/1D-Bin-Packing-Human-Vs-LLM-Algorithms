import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    W = weights  # list for fast indexing
    total_weight = sum(W)
    
    def elapsed():
        return time.time() - start_time
    
    # Lower bound
    continuous_lb = (total_weight + C - 1) // C
    
    # L2 lower bound
    def compute_L2():
        best_lb = continuous_lb
        sorted_w = sorted(W, reverse=True)
        half_C = C / 2.0
        
        # Only check select alpha values
        alphas = set()
        for a in range(1, min(C // 2 + 1, 201)):
            alphas.add(a)
        # Also add values related to item weights
        for w in W:
            a = C - w
            if 1 <= a <= C // 2:
                alphas.add(a)
                if a + 1 <= C // 2:
                    alphas.add(a + 1)
        
        for alpha in sorted(alphas):
            if elapsed() > time_limit * 0.03:
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
    
    # Compact representation using lists
    # assignment: item -> bin_id
    # bin_items: bin_id -> list of items (we'll use sets for O(1) removal)
    # bin_wts: bin_id -> weight
    # residual: bin_id -> remaining capacity
    
    def best_fit_decreasing(order):
        """Build solution using best-fit decreasing."""
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
        """Build solution using first-fit decreasing."""
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
    
    def sol_to_sets(assignment, bin_items_list, bin_wts_list, residual_list, num_bins):
        """Convert to set-based representation for local search."""
        bi = [set(items) for items in bin_items_list]
        return list(assignment), bi, list(bin_wts_list), list(residual_list), num_bins
    
    def copy_sol(assignment, bi, bw, res, nb):
        return list(assignment), [set(s) for s in bi], list(bw), list(res), nb
    
    def compact_sol(assignment, bi, bw, res, nb):
        """Remove empty bins and reindex."""
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
        for _ in range(10):
            if elapsed() > time_limit * 0.03:
                break
            noise = max(1, C // 6)
            noisy_order = sorted(range(n), key=lambda i: -W[i] + random.randint(-noise, noise))
            a3, bil3, bwl3, rl3, nbl3 = best_fit_decreasing(noisy_order)
            if nbl3 < best_nb:
                best_a, best_bi, best_bw, best_res, best_nb = sol_to_sets(a3, bil3, bwl3, rl3, nbl3)
                if best_nb <= lower_bound:
                    break
    
    if best_nb <= lower_bound:
        return to_output(best_a, best_bi, best_bw, best_res, best_nb)
    
    # --- Local Search: try to empty bins ---
    def try_empty_bin(assignment, bi, bw, res, nb, src, time_lim):
        """Try to empty bin src by moving/swapping items out. Returns True if emptied."""
        if len(bi[src]) == 0:
            return False
        
        items = sorted(bi[src], key=lambda i: -W[i])
        
        # First try direct moves (best-fit)
        temp_res = {}  # bin -> adjusted residual
        moves = []
        can_empty = True
        
        for item in items:
            w = W[item]
            best_b = -1
            best_rem = C + 1
            for b in range(len(bi)):
                if b == src or len(bi[b]) == 0:
                    if b == src:
                        continue
                    if len(bi[b]) == 0 and b not in temp_res:
                        continue
                r = temp_res.get(b, res[b])
                if r >= w and r < best_rem:
                    best_b = b
                    best_rem = r
                    if r == w:
                        break
            if best_b >= 0:
                moves.append((item, best_b))
                temp_res[best_b] = temp_res.get(best_b, res[best_b]) - w
            else:
                can_empty = False
                break
        
        if can_empty:
            for item, tgt in moves:
                w = W[item]
                old_b = assignment[item]
                bi[old_b].discard(item)
                bw[old_b] -= w
                res[old_b] += w
                assignment[item] = tgt
                bi[tgt].add(item)
                bw[tgt] += w
                res[tgt] -= w
            nb -= 1
            return True
        
        # Try swap-assisted emptying
        # For each item in src that couldn't fit, try (1,1) swap to create space
        if len(items) <= 12:
            # Backtracking approach for small bins
            result = try_empty_backtrack(assignment, bi, bw, res, nb, src, items, time_lim)
            if result:
                return True
        
        return False
    
    def try_empty_backtrack(assignment, bi, bw, res, nb, src, items, time_lim):
        """Try to empty bin using swap+move combinations."""
        # For each pair of items: one from src, one from another bin
        # Try swapping them if it helps place the src item
        
        src_items = list(bi[src])
        if not src_items:
            return False
        
        # Identify which items can be directly moved
        movable = []
        stuck = []
        for item in src_items:
            w = W[item]
            can_move = False
            for b in range(len(bi)):
                if b == src or len(bi[b]) == 0:
                    continue
                if res[b] >= w:
                    can_move = True
                    break
            if can_move:
                movable.append(item)
            else:
                stuck.append(item)
        
        if not stuck:
            # All movable, shouldn't happen since direct failed
            return False
        
        # For each stuck item, try (1,1) swap with items in other bins
        for s_item in stuck:
            if elapsed() > time_lim:
                return False
            ws = W[s_item]
            
            for b in range(len(bi)):
                if b == src or len(bi[b]) == 0:
                    continue
                if elapsed() > time_lim:
                    return False
                
                min_wt = ws - res[b]
                
                for t_item in list(bi[b]):
                    wt = W[t_item]
                    if wt < min_wt:
                        continue
                    if wt >= ws:
                        continue
                    # After swap: s_item goes to b (residual becomes res[b] - ws + wt >= 0)
                    # t_item needs to go somewhere other than src
                    # Check if t_item fits in some other bin
                    for b2 in range(len(bi)):
                        if b2 == src or b2 == b or len(bi[b2]) == 0:
                            continue
                        if res[b2] >= wt:
                            # Do the swap: move s_item from src to b, t_item from b to b2
                            # Move s_item
                            bi[src].discard(s_item)
                            bw[src] -= ws
                            res[src] += ws
                            assignment[s_item] = b
                            bi[b].add(s_item)
                            bw[b] += ws
                            res[b] -= ws
                            
                            # Move t_item
                            bi[b].discard(t_item)
                            bw[b] -= wt
                            res[b] += wt
                            assignment[t_item] = b2
                            bi[b2].add(t_item)
                            bw[b2] += wt
                            res[b2] -= wt
                            
                            return True  # partially emptied, caller should retry
                    
                    # Also try: t_item goes to src (temporarily), freeing space
                    # This only helps if other items in src can then be moved out
                    # Skip for now - too complex
        
        return False
    
    def local_search(assignment, bi, bw, res, nb, time_frac=0.95):
        """Try to reduce number of bins by emptying lightest bins."""
        time_lim = start_time + time_limit * time_frac
        
        improved = True
        while improved and elapsed() < time_limit * time_frac:
            improved = False
            if nb <= lower_bound:
                break
            
            # Sort bins by weight (lightest first)
            active = [b for b in range(len(bi)) if len(bi[b]) > 0]
            active.sort(key=lambda b: bw[b])
            
            for src in active:
                if elapsed() > time_limit * time_frac:
                    break
                if len(bi[src]) == 0:
                    continue
                
                result = try_empty_bin(assignment, bi, bw, res, nb, src, time_lim)
                if result:
                    # Recount
                    nb = sum(1 for b in range(len(bi)) if len(bi[b]) > 0)
                    improved = True
                    break
        
        # Phase 2: swap to improve packing tightness
        active = [b for b in range(len(bi)) if len(bi[b]) > 0]
        if len(active) >= 2 and nb > lower_bound:
            swap_rounds = 0
            max_swap_rounds = min(len(active) * len(active), 500)
            for _ in range(max_swap_rounds):
                if elapsed() > time_limit * time_frac:
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
                    # Accept if it reduces max residual or improves balance
                    old_waste = res[b1] * res[b1] + res[b2] * res[b2]
                    new_waste = new_r1 * new_r1 + new_r2 * new_r2
                    if new_waste > old_waste:  # More concentrated waste = easier to empty
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
        
        # Try emptying again after swaps
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
                result = try_empty_bin(assignment, bi, bw, res, nb, src, time_lim)
                if result:
                    nb = sum(1 for b in range(len(bi)) if len(bi[b]) > 0)
                    improved2 = True
                    break
        
        return assignment, bi, bw, res, nb
    
    # Apply initial local search
    cur_a, cur_bi, cur_bw, cur_res, cur_nb = copy_sol(best_a, best_bi, best_bw, best_res, best_nb)
    cur_a, cur_bi, cur_bw, cur_res, cur_nb = local_search(cur_a, cur_bi, cur_bw, cur_res, cur_nb)
    
    if cur_nb < best_nb:
        best_a, best_bi, best_bw, best_res, best_nb = copy_sol(cur_a, cur_bi, cur_bw, cur_res, cur_nb)
    
    if best_nb <= lower_bound:
        return to_output(best_a, best_bi, best_bw, best_res, best_nb)
    
    # --- Shaking / Perturbation ---
    def destroy_and_rebuild(assignment, bi, bw, res, nb, frac, mode='bf'):
        """Destroy fraction of bins and rebuild with best-fit."""
        assignment, bi, bw, res, nb = compact_sol(assignment, bi, bw, res, nb)
        active = list(range(nb))
        
        num_destroy = max(1, int(nb * frac))
        num_destroy = min(num_destroy, nb)
        
        # Choose bins to destroy (prefer lighter)
        active_sorted = sorted(active, key=lambda b: bw[b])
        to_destroy = set(active_sorted[:num_destroy])
        
        all_items = []
        for b in to_destroy:
            all_items.extend(bi[b])
        
        for item in all_items:
            b = assignment[item]
            bi[b].discard(item)
            w = W[item]
            bw[b] -= w
            res[b] += w
        
        # Sort items by weight descending
        all_items.sort(key=lambda i: -W[i])
        
        for item in all_items:
            w = W[item]
            best_b = -1
            best_rem = C + 1
            for b in range(len(bi)):
                if len(bi[b]) == 0 and b in to_destroy:
                    continue  # skip destroyed empty bins
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
                # Find an empty bin to use
                found = False
                for b in to_destroy:
                    if len(bi[b]) == 0:
                        assignment[item] = b
                        bi[b].add(item)
                        bw[b] = w
                        res[b] = C - w
                        to_destroy.discard(b)
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
        """Shaking: different neighborhood structures."""
        assignment, bi, bw, res, nb = copy_sol(assignment, bi, bw, res, nb)
        
        if k == 1:
            # Move 1-2 items from lightest bin
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 1.0 / max(nb, 1))
        elif k == 2:
            # Destroy 2 lightest bins
            frac = min(2.0 / max(nb, 1), 0.5)
            return destroy_and_rebuild(assignment, bi, bw, res, nb, frac)
        elif k == 3:
            # Destroy ~15% of bins (lightest)
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.15)
        elif k == 4:
            # Destroy ~25%
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.25)
        elif k == 5:
            # Destroy ~40%
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.40)
        elif k == 6:
            # Destroy ~60%
            return destroy_and_rebuild(assignment, bi, bw, res, nb, 0.60)
        elif k == 7:
            # Random destroy 20-50%
            # Instead of lightest, destroy random bins
            assignment, bi, bw, res, nb = compact_sol(assignment, bi, bw, res, nb)
            active = list(range(nb))
            frac = random.uniform(0.2, 0.5)
            num_destroy = max(1, int(nb * frac))
            to_destroy_list = random.sample(active, min(num_destroy, len(active)))
            to_destroy = set(to_destroy_list)
            
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
                    for b in to_destroy:
                        if len(bi[b]) == 0:
                            assignment[item] = b
                            bi[b].add(item)
                            bw[b] = w
                            res[b] = C - w
                            to_destroy.discard(b)
                            break
                    else:
                        new_b = len(bi)
                        bi.append(set([item]))
                        bw.append(w)
                        res.append(C - w)
                        assignment[item] = new_b
            nb = sum(1 for b in range(len(bi)) if len(bi[b]) > 0)
            return assignment, bi, bw, res, nb
        elif k == 8:
            # Full rebuild with noise
            noise = max(1, C // 4)
            noisy_order = sorted(range(n), key=lambda i: -W[i] + random.randint(-noise, noise))
            a, bil, bwl, rl, nbl = best_fit_decreasing(noisy_order)
            return sol_to_sets(a, bil, bwl, rl, nbl)
        else:
            return assignment, bi, bw, res, nb
    
    # --- VNS Main Loop ---
    k_max = 8
    k = 1
    no_improve_count = 0
    max_no_improve = 50
    
    # Dynamic time allocation for local search
    while elapsed() < time_limit * 0.97:
        if best_nb <= lower_bound:
            break
        
        # Shaking
        s_a, s_bi, s_bw, s_res, s_nb = shake(cur_a, cur_bi, cur_bw, cur_res, cur_nb, k)
        
        if elapsed() > time_limit * 0.97:
            if s_nb < best_nb:
                best_a, best_bi, best_bw, best_res, best_nb = s_a, s_bi, s_bw, s_res, s_nb
            break
        
        # Local search
        remaining = time_limit - elapsed()
        ls_frac = min(0.97, (elapsed() + min(remaining * 0.5, 2.0)) / time_limit)
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
            # Accept lateral moves sometimes
            if random.random() < 0.4:
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
            remaining2 = time_limit - elapsed()
            ls_frac2 = min(0.97, (elapsed() + min(remaining2 * 0.3, 1.5)) / time_limit)
            cur_a, cur_bi, cur_bw, cur_res, cur_nb = local_search(cur_a, cur_bi, cur_bw, cur_res, cur_nb, time_frac=ls_frac2)
            if cur_nb < best_nb:
                best_a, best_bi, best_bw, best_res, best_nb = copy_sol(cur_a, cur_bi, cur_bw, cur_res, cur_nb)
            k = 1
            no_improve_count = 0
    
    return to_output(best_a, best_bi, best_bw, best_res, best_nb)
