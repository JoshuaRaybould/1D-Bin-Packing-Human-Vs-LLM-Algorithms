import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    C = bin_capacity
    W = tuple(weights)  # faster indexing
    total_weight = sum(W)
    
    def elapsed():
        return time.time() - start_time
    
    # --- L2 Lower Bound (Martello-Toth) ---
    def compute_L2():
        continuous_lb = (total_weight + C - 1) // C  # ceil division
        best_lb = continuous_lb
        
        # Sort weights for efficient counting
        sorted_w = sorted(W, reverse=True)
        
        for alpha in range(1, C // 2 + 1):
            threshold_large = C - alpha
            # Large: w > C - alpha
            # Medium: C/2 < w <= C - alpha
            # Small: alpha <= w <= C/2
            
            n_large = 0
            n_medium = 0
            sum_small = 0
            n_small = 0
            
            half_C = C / 2.0
            
            for w in sorted_w:
                if w > threshold_large:
                    n_large += 1
                elif w > half_C:
                    n_medium += 1
                elif w >= alpha:
                    sum_small += w
                    n_small += 1
            
            # Each large item needs its own bin
            # Each medium item needs its own bin (can't pair two mediums since both > C/2)
            # But large items might have room for small items
            # Remaining capacity from large bins: sum(C - w_large) but that's complex
            # Simplified L2: 
            # Bins needed >= n_large + n_medium + max(0, ceil((sum_small - (n_medium * (C - ...) ... )))
            # Standard formula:
            # L2(alpha) = n_large + n_medium + max(0, ceil((sum_small - (n_large * C + n_medium * C - sum_large - sum_medium ... )))
            # Actually, the standard Martello-Toth L2 is:
            # For each alpha, count:
            #   J1 = items with w > C - alpha (need own bin)
            #   J2 = items with C - alpha >= w > C/2 (need own bin, can share with small)
            #   J3 = items with C/2 >= w >= alpha
            # residual capacity in J2 bins = sum over j2 bins of (C - w_j2)
            # But small items can fill residual in J1 and J2 bins
            # L2(alpha) = |J1| + |J2| + max(0, ceil((sum(J3) - sum_residual_J2) / C))
            # But computing sum_residual for J1 bins is harder.
            # Simplified: residual in J2 bins = n_medium * C - sum_medium_weights
            # We need sum of medium weights
            
            # Let me recompute with sums
            pass
        
        # Simpler approach: just compute for a few alpha values
        # Actually let me do it properly
        sorted_w_asc = sorted(W)
        
        for alpha in range(1, C // 2 + 1):
            threshold_large = C - alpha
            half_C_floor = C // 2
            
            n_large = 0
            sum_large = 0
            n_medium = 0
            sum_medium = 0
            sum_small = 0
            
            for w in sorted_w:
                if w > threshold_large:
                    n_large += 1
                    sum_large += w
                elif 2 * w > C:  # w > C/2
                    n_medium += 1
                    sum_medium += w
                elif w >= alpha:
                    sum_small += w
            
            # Residual capacity in large bins
            residual_large = n_large * C - sum_large
            # Residual capacity in medium bins
            residual_medium = n_medium * C - sum_medium
            
            # Small items fill residual in large+medium bins first
            remaining_small = max(0, sum_small - residual_large - residual_medium)
            
            lb = n_large + n_medium + (remaining_small + C - 1) // C if remaining_small > 0 else n_large + n_medium
            
            if lb > best_lb:
                best_lb = lb
            
            # Early termination: if alpha gets too large, diminishing returns
            if alpha > 20 and n == alpha:  # heuristic cutoff
                break
            if elapsed() > time_limit * 0.02:
                break
        
        return best_lb
    
    lower_bound = compute_L2()
    
    # --- Data Structure ---
    # We use: assignment[item] = bin_id, bin_items[bin_id] = set of items, 
    #         bin_wts[bin_id] = weight, residual[bin_id] = remaining capacity
    # We also track active_bins as a set of non-empty bin indices
    
    class Solution:
        __slots__ = ['assignment', 'bin_items', 'bin_wts', 'residual', 'num_bins', 'active_bins']
        
        def __init__(self):
            self.assignment = [0] * n  # item -> bin
            self.bin_items = []  # list of sets
            self.bin_wts = []  # list of ints
            self.residual = []  # list of ints
            self.num_bins = 0
            self.active_bins = set()
        
        def add_bin(self):
            idx = len(self.bin_items)
            self.bin_items.append(set())
            self.bin_wts.append(0)
            self.residual.append(C)
            self.num_bins += 1
            self.active_bins.add(idx)
            return idx
        
        def place_item(self, item, bin_id):
            self.assignment[item] = bin_id
            self.bin_items[bin_id].add(item)
            w = W[item]
            self.bin_wts[bin_id] += w
            self.residual[bin_id] -= w
        
        def move_item(self, item, to_bin):
            from_bin = self.assignment[item]
            w = W[item]
            self.bin_items[from_bin].discard(item)
            self.bin_wts[from_bin] -= w
            self.residual[from_bin] += w
            if len(self.bin_items[from_bin]) == 0:
                self.active_bins.discard(from_bin)
                self.num_bins -= 1
            self.assignment[item] = to_bin
            self.bin_items[to_bin].add(item)
            self.bin_wts[to_bin] += w
            self.residual[to_bin] -= w
        
        def swap_items(self, item1, item2):
            b1 = self.assignment[item1]
            b2 = self.assignment[item2]
            w1 = W[item1]
            w2 = W[item2]
            
            self.bin_items[b1].discard(item1)
            self.bin_items[b1].add(item2)
            self.bin_wts[b1] += w2 - w1
            self.residual[b1] += w1 - w2
            
            self.bin_items[b2].discard(item2)
            self.bin_items[b2].add(item1)
            self.bin_wts[b2] += w1 - w2
            self.residual[b2] += w2 - w1
            
            self.assignment[item1] = b2
            self.assignment[item2] = b1
        
        def cost(self):
            return self.num_bins
        
        def copy(self):
            s = Solution()
            s.assignment = list(self.assignment)
            s.bin_items = [set(b) for b in self.bin_items]
            s.bin_wts = list(self.bin_wts)
            s.residual = list(self.residual)
            s.num_bins = self.num_bins
            s.active_bins = set(self.active_bins)
            return s
        
        def compact(self):
            """Reindex bins to be compact (0..num_bins-1)"""
            active = sorted(self.active_bins)
            mapping = {old: new for new, old in enumerate(active)}
            new_bin_items = []
            new_bin_wts = []
            new_residual = []
            for old in active:
                new_bin_items.append(self.bin_items[old])
                new_bin_wts.append(self.bin_wts[old])
                new_residual.append(self.residual[old])
            for item in range(n):
                self.assignment[item] = mapping[self.assignment[item]]
            self.bin_items = new_bin_items
            self.bin_wts = new_bin_wts
            self.residual = new_residual
            self.active_bins = set(range(len(active)))
        
        def to_output(self):
            self.compact()
            packing = [sorted(self.bin_items[b]) for b in range(len(self.bin_items))]
            bin_weights = [self.bin_wts[b] for b in range(len(self.bin_wts))]
            return {"packing": packing, "bin_weights": bin_weights}
    
    def build_solution_from_order(order, mode='bf'):
        """Build solution from item order using first-fit or best-fit."""
        sol = Solution()
        for idx in order:
            w = W[idx]
            best_j = -1
            if mode == 'ff':
                for b in sorted(sol.active_bins):
                    if sol.residual[b] >= w:
                        best_j = b
                        break
            elif mode == 'bf':
                best_rem = C + 1
                for b in sol.active_bins:
                    r = sol.residual[b]
                    if r >= w and r < best_rem:
                        best_j = b
                        best_rem = r
                        if r == w:
                            break
            elif mode == 'wf':
                best_rem = -1
                for b in sol.active_bins:
                    r = sol.residual[b]
                    if r >= w and r > best_rem:
                        best_j = b
                        best_rem = r
            
            if best_j >= 0:
                sol.place_item(idx, best_j)
            else:
                new_b = sol.add_bin()
                sol.place_item(idx, new_b)
        return sol
    
    # --- Initial Solutions ---
    desc_order = sorted(range(n), key=lambda i: -W[i])
    
    sol_ffd = build_solution_from_order(desc_order, 'ff')
    best_sol = sol_ffd.copy()
    
    if best_sol.cost() > lower_bound:
        sol_bfd = build_solution_from_order(desc_order, 'bf')
        if sol_bfd.cost() < best_sol.cost():
            best_sol = sol_bfd.copy()
    
    if best_sol.cost() > lower_bound:
        sol_wfd = build_solution_from_order(desc_order, 'wf')
        if sol_wfd.cost() < best_sol.cost():
            best_sol = sol_wfd.copy()
    
    # Randomized BFD
    if best_sol.cost() > lower_bound:
        for _ in range(5):
            if elapsed() > time_limit * 0.05:
                break
            noisy_order = sorted(range(n), key=lambda i: -W[i] + random.randint(-max(1, C // 8), max(1, C // 8)))
            sol_r = build_solution_from_order(noisy_order, 'bf')
            if sol_r.cost() < best_sol.cost():
                best_sol = sol_r.copy()
    
    if best_sol.cost() <= lower_bound:
        return best_sol.to_output()
    
    # --- Local Search ---
    def local_search(sol, time_frac=0.98):
        """Strong local search: try to empty lightest bins."""
        time_lim = time_limit * time_frac
        
        improved = True
        while improved:
            if elapsed() > time_lim:
                break
            improved = False
            
            if sol.num_bins <= 1:
                break
            
            if sol.cost() <= lower_bound:
                break
            
            # Phase 1: Try to empty lightest bin
            active = sorted(sol.active_bins, key=lambda b: sol.bin_wts[b])
            
            for src in active:
                if elapsed() > time_lim:
                    return sol
                if src not in sol.active_bins:
                    continue
                if len(sol.bin_items[src]) == 0:
                    continue
                
                items_in_src = list(sol.bin_items[src])
                # Sort by weight descending (harder to place first)
                items_in_src.sort(key=lambda i: -W[i])
                
                # Try to move all items out
                can_empty = True
                moves = []  # (item, target_bin)
                # Simulate
                temp_residuals = {}  # bin -> adjusted residual
                
                for item in items_in_src:
                    w = W[item]
                    best_target = -1
                    best_rem = C + 1
                    for b in sol.active_bins:
                        if b == src:
                            continue
                        r = temp_residuals.get(b, sol.residual[b])
                        if r >= w and r < best_rem:
                            best_target = b
                            best_rem = r
                            if r == w:
                                break
                    if best_target >= 0:
                        moves.append((item, best_target))
                        temp_residuals[best_target] = temp_residuals.get(best_target, sol.residual[best_target]) - w
                    else:
                        can_empty = False
                        break
                
                if can_empty:
                    for item, target in moves:
                        sol.move_item(item, target)
                    improved = True
                    break  # restart from lightest
                
                # Phase 2: Swap-assisted emptying
                # For each item that couldn't be moved, try swapping
                if not can_empty and len(items_in_src) <= 8:
                    # Try (1,1) swap-assisted: for stuck item s, find item t in target bin
                    # such that swapping frees space
                    found_swap = False
                    
                    for stuck_item in items_in_src:
                        if elapsed() > time_lim:
                            return sol
                        ws = W[stuck_item]
                        
                        for b in sol.active_bins:
                            if b == src:
                                continue
                            # Can we swap stuck_item with some item t in b?
                            # Need: ws - wt <= residual[b] (i.e., wt >= ws - residual[b])
                            # And: wt fits somewhere else (including src which we're trying to empty - skip)
                            # Actually for swap: new residual[b] = residual[b] - ws + wt >= 0
                            # So wt >= ws - residual[b]
                            min_wt = ws - sol.residual[b]
                            if min_wt < 0:
                                min_wt = 0
                            
                            for t in sol.bin_items[b]:
                                wt = W[t]
                                if wt < min_wt:
                                    continue
                                if wt >= ws:
                                    continue  # no point swapping heavier item in
                                # Check if t fits somewhere other than src and b
                                can_place_t = False
                                for b2 in sol.active_bins:
                                    if b2 == src or b2 == b:
                                        continue
                                    if sol.residual[b2] >= wt:
                                        can_place_t = True
                                        # Do the swap + move
                                        sol.swap_items(stuck_item, t)
                                        sol.move_item(t, b2)
                                        found_swap = True
                                        break
                                if found_swap:
                                    break
                            if found_swap:
                                break
                    
                    if found_swap:
                        # Try again to empty this bin
                        improved = True
                        break
            
            if not improved:
                # Phase 4: Random swaps to improve packing
                nb = sol.num_bins
                active_list = list(sol.active_bins)
                swap_improved = False
                attempts = min(nb * nb, 300)
                for _ in range(attempts):
                    if elapsed() > time_lim:
                        return sol
                    if len(active_list) < 2:
                        break
                    b1, b2 = random.sample(active_list, 2)
                    if len(sol.bin_items[b1]) == 0 or len(sol.bin_items[b2]) == 0:
                        continue
                    item1 = random.choice(list(sol.bin_items[b1]))
                    item2 = random.choice(list(sol.bin_items[b2]))
                    w1, w2 = W[item1], W[item2]
                    new_r1 = sol.residual[b1] + w1 - w2
                    new_r2 = sol.residual[b2] + w2 - w1
                    if new_r1 >= 0 and new_r2 >= 0:
                        if (new_r1 < sol.residual[b1] or new_r2 < sol.residual[b2]):
                            sol.swap_items(item1, item2)
                            swap_improved = True
                if swap_improved:
                    improved = True
        
        return sol
    
    # Apply local search to best initial solution
    current_sol = best_sol.copy()
    current_sol = local_search(current_sol)
    if current_sol.cost() < best_sol.cost():
        best_sol = current_sol.copy()
    
    if best_sol.cost() <= lower_bound:
        return best_sol.to_output()
    
    # --- Shaking ---
    def shake(sol, k):
        sol = sol.copy()
        sol.compact()
        nb = sol.num_bins
        if nb <= 1:
            return sol
        
        active_list = sorted(sol.active_bins)
        
        if k == 1:
            # Move 1 random item to a random feasible bin
            src = random.choice(active_list)
            if len(sol.bin_items[src]) > 0:
                item = random.choice(list(sol.bin_items[src]))
                w = W[item]
                targets = [b for b in active_list if b != src and sol.residual[b] >= w]
                if targets:
                    tgt = random.choice(targets)
                    sol.move_item(item, tgt)
        
        elif k == 2:
            # Swap items between two random bins
            for _ in range(2):
                if len(active_list) < 2:
                    break
                b1, b2 = random.sample(active_list, 2)
                if len(sol.bin_items[b1]) == 0 or len(sol.bin_items[b2]) == 0:
                    continue
                item1 = random.choice(list(sol.bin_items[b1]))
                item2 = random.choice(list(sol.bin_items[b2]))
                w1, w2 = W[item1], W[item2]
                if sol.residual[b1] + w1 - w2 >= 0 and sol.residual[b2] + w2 - w1 >= 0:
                    sol.swap_items(item1, item2)
        
        elif k == 3:
            # Take 2 lightest bins, remove items, reinsert with best-fit
            sorted_bins = sorted(active_list, key=lambda b: sol.bin_wts[b])
            num = min(2, len(sorted_bins))
            to_destroy = sorted_bins[:num]
            all_items = []
            for b in to_destroy:
                all_items.extend(sol.bin_items[b])
            # Remove items
            for item in all_items:
                b = sol.assignment[item]
                sol.bin_items[b].discard(item)
                w = W[item]
                sol.bin_wts[b] -= w
                sol.residual[b] += w
                if len(sol.bin_items[b]) == 0:
                    sol.active_bins.discard(b)
                    sol.num_bins -= 1
            # Reinsert best-fit
            all_items.sort(key=lambda i: -W[i])
            for item in all_items:
                w = W[item]
                best_b = -1
                best_rem = C + 1
                for b in sol.active_bins:
                    r = sol.residual[b]
                    if r >= w and r < best_rem:
                        best_b = b
                        best_rem = r
                        if r == w:
                            break
                if best_b >= 0:
                    sol.assignment[item] = best_b
                    sol.bin_items[best_b].add(item)
                    sol.bin_wts[best_b] += w
                    sol.residual[best_b] -= w
                else:
                    new_b = sol.add_bin()
                    sol.assignment[item] = new_b
                    sol.bin_items[new_b].add(item)
                    sol.bin_wts[new_b] += w
                    sol.residual[new_b] -= w
        
        elif k == 4:
            # Destroy 15-25% of bins (prefer lighter)
            frac = random.uniform(0.15, 0.25)
            num_destroy = max(1, int(nb * frac))
            # Weighted selection: prefer lighter bins
            wt_list = [(b, 1.0 / (sol.bin_wts[b] + 1)) for b in active_list]
            total_w = sum(x[1] for x in wt_list)
            probs = [x[1] / total_w for x in wt_list]
            # Weighted sample without replacement
            chosen = set()
            pool = list(range(len(active_list)))
            for _ in range(min(num_destroy, len(pool))):
                r = random.random()
                cum = 0
                for idx in pool:
                    if idx in chosen:
                        continue
                    cum += probs[idx]
                    if r <= cum:
                        chosen.add(idx)
                        break
                else:
                    # fallback
                    remaining = [x for x in pool if x not in chosen]
                    if remaining:
                        chosen.add(random.choice(remaining))
            
            to_destroy = [active_list[i] for i in chosen]
            all_items = []
            for b in to_destroy:
                all_items.extend(sol.bin_items[b])
            for item in all_items:
                b = sol.assignment[item]
                sol.bin_items[b].discard(item)
                w = W[item]
                sol.bin_wts[b] -= w
                sol.residual[b] += w
                if len(sol.bin_items[b]) == 0:
                    sol.active_bins.discard(b)
                    sol.num_bins -= 1
            all_items.sort(key=lambda i: -W[i])
            for item in all_items:
                w = W[item]
                best_b = -1
                best_rem = C + 1
                for b in sol.active_bins:
                    r = sol.residual[b]
                    if r >= w and r < best_rem:
                        best_b = b
                        best_rem = r
                        if r == w:
                            break
                if best_b >= 0:
                    sol.assignment[item] = best_b
                    sol.bin_items[best_b].add(item)
                    sol.bin_wts[best_b] += w
                    sol.residual[best_b] -= w
                else:
                    new_b = sol.add_bin()
                    sol.assignment[item] = new_b
                    sol.bin_items[new_b].add(item)
                    sol.bin_wts[new_b] += w
                    sol.residual[new_b] -= w
        
        elif k == 5:
            # Destroy 30-50%
            frac = random.uniform(0.30, 0.50)
            num_destroy = max(1, int(nb * frac))
            sorted_bins = sorted(active_list, key=lambda b: sol.bin_wts[b])
            to_destroy = sorted_bins[:num_destroy]
            all_items = []
            for b in to_destroy:
                all_items.extend(sol.bin_items[b])
            for item in all_items:
                b = sol.assignment[item]
                sol.bin_items[b].discard(item)
                w = W[item]
                sol.bin_wts[b] -= w
                sol.residual[b] += w
                if len(sol.bin_items[b]) == 0:
                    sol.active_bins.discard(b)
                    sol.num_bins -= 1
            all_items.sort(key=lambda i: -W[i])
            for item in all_items:
                w = W[item]
                best_b = -1
                best_rem = C + 1
                for b in sol.active_bins:
                    r = sol.residual[b]
                    if r >= w and r < best_rem:
                        best_b = b
                        best_rem = r
                        if r == w:
                            break
                if best_b >= 0:
                    sol.assignment[item] = best_b
                    sol.bin_items[best_b].add(item)
                    sol.bin_wts[best_b] += w
                    sol.residual[best_b] -= w
                else:
                    new_b = sol.add_bin()
                    sol.assignment[item] = new_b
                    sol.bin_items[new_b].add(item)
                    sol.bin_wts[new_b] += w
                    sol.residual[new_b] -= w
        
        elif k == 6:
            # Destroy 60-80%
            frac = random.uniform(0.60, 0.80)
            num_destroy = max(2, int(nb * frac))
            num_destroy = min(num_destroy, nb)
            to_destroy = random.sample(active_list, num_destroy)
            all_items = []
            for b in to_destroy:
                all_items.extend(sol.bin_items[b])
            for item in all_items:
                b = sol.assignment[item]
                sol.bin_items[b].discard(item)
                w = W[item]
                sol.bin_wts[b] -= w
                sol.residual[b] += w
                if len(sol.bin_items[b]) == 0:
                    sol.active_bins.discard(b)
                    sol.num_bins -= 1
            all_items.sort(key=lambda i: -W[i])
            for item in all_items:
                w = W[item]
                best_b = -1
                best_rem = C + 1
                for b in sol.active_bins:
                    r = sol.residual[b]
                    if r >= w and r < best_rem:
                        best_b = b
                        best_rem = r
                        if r == w:
                            break
                if best_b >= 0:
                    sol.assignment[item] = best_b
                    sol.bin_items[best_b].add(item)
                    sol.bin_wts[best_b] += w
                    sol.residual[best_b] -= w
                else:
                    new_b = sol.add_bin()
                    sol.assignment[item] = new_b
                    sol.bin_items[new_b].add(item)
                    sol.bin_wts[new_b] += w
                    sol.residual[new_b] -= w
        
        return sol
    
    # --- VNS Main Loop ---
    k_max = 6
    k = 1
    max_no_improve = 30
    no_improve_count = 0
    
    while elapsed() < time_limit * 0.95:
        if best_sol.cost() <= lower_bound:
            break
        
        # Shaking
        shaken_sol = shake(current_sol, k)
        
        if elapsed() > time_limit * 0.95:
            break
        
        # Local search (full)
        ls_sol = local_search(shaken_sol)
        
        ls_cost = ls_sol.cost()
        current_cost = current_sol.cost()
        
        if ls_cost < current_cost:
            current_sol = ls_sol
            k = 1
            no_improve_count = 0
            if ls_cost < best_sol.cost():
                best_sol = current_sol.copy()
                if best_sol.cost() <= lower_bound:
                    break
        elif ls_cost == current_cost:
            # Accept lateral moves with probability 0.3
            if random.random() < 0.3:
                current_sol = ls_sol
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
            noisy_order = sorted(range(n), key=lambda i: -W[i] + random.randint(-max(1, C // 8), max(1, C // 8)))
            new_sol = build_solution_from_order(noisy_order, 'bf')
            new_sol = local_search(new_sol)
            if new_sol.cost() < best_sol.cost():
                best_sol = new_sol.copy()
            current_sol = new_sol
            k = 1
            no_improve_count = 0
    
    return best_sol.to_output()
