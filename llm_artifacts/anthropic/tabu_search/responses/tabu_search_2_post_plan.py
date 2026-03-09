def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    import random
    import math
    import bisect

    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = bin_capacity
    w_arr = tuple(weights)  # faster indexing

    # --- Compute L2 lower bound (Martello & Toth) ---
    total_weight = sum(w_arr)
    L1 = math.ceil(total_weight / C)

    # L2 computation
    def compute_L2():
        best_L2 = L1
        # Try different alpha values
        for alpha_num in range(1, C // 2 + 1):
            alpha = alpha_num
            if alpha > C // 2:
                break
            # Only try a few strategic alpha values
            # Items with weight > C - alpha (large)
            # Items with weight in (C/2, C - alpha] -- wait, standard L2:
            # Actually use the standard Martello-Toth L2:
            # For a given alpha (1 <= alpha <= C/2):
            #   J1 = items with weight > C - alpha
            #   J2 = items with C - alpha >= weight > C/2
            #   J3 = items with C/2 >= weight >= alpha
            # L2(alpha) = |J1| + |J2| + max(0, ceil((sum of J3 weights - (|J2|*C - sum of J2 weights - sum of J1 weights ... ))) )
            # This is complex; let's use a simpler version
            pass
        return best_L2

    # Simpler L2: use the standard formula
    # L2 = max over k in {1..C/2} of (n1(k) + n2(k) + max(0, ceil((S3(k) - slack) / C)))
    # where n1 = #{items > C-k}, n2 = #{items in (C/2, C-k]}, 
    # S3 = sum of items in [k, C/2], slack = n2*C - sum(J1) - sum(J2)
    # This is expensive for large C; sample a few values
    
    def compute_lower_bound():
        lb = L1
        # Try a few alpha values
        half_C = C / 2.0
        alphas = set()
        # Add strategic alpha values based on item weights
        for i in range(n):
            wi = w_arr[i]
            if wi <= C // 2:
                alphas.add(wi)
            diff = C - wi
            if 1 <= diff <= C // 2:
                alphas.add(diff)
        # Also add 1 and C//2
        alphas.add(1)
        if C // 2 >= 1:
            alphas.add(C // 2)
        # Limit number of alphas to avoid timeout
        if len(alphas) > 200:
            alphas = set(random.sample(sorted(alphas), 200))
        
        for alpha in alphas:
            if alpha < 1 or alpha > C // 2:
                continue
            n1 = 0
            s1 = 0
            n2 = 0
            s2 = 0
            s3 = 0
            threshold_high = C - alpha
            for i in range(n):
                wi = w_arr[i]
                if wi > threshold_high:
                    n1 += 1
                    s1 += wi
                elif wi > half_C:
                    n2 += 1
                    s2 += wi
                elif wi >= alpha:
                    s3 += wi
            # Free space in J2 bins after pairing with J1
            # Each J1 item needs its own bin, J2 items also need own bins
            # but J1 bins might have space for small items
            free_in_j1 = n1 * C - s1  # free space in bins dedicated to J1 items
            free_in_j2 = n2 * C - s2  # free space in bins dedicated to J2 items
            # J3 items need to fill remaining space
            remaining_s3 = max(0, s3 - free_in_j1 - free_in_j2)
            # But actually the standard formula doesn't subtract free_in_j1
            # Standard: L2(alpha) = n1 + n2 + max(0, ceil((s3 - (n2*C - s1 - s2)) / C))
            slack = n2 * C - s1 - s2
            extra = 0
            if s3 > slack:
                extra = math.ceil((s3 - slack) / C)
            val = n1 + n2 + max(0, extra)
            if val > lb:
                lb = val
        return lb

    lower_bound = compute_lower_bound()

    # --- Initial solution: FFD ---
    def first_fit_decreasing():
        sorted_indices = sorted(range(n), key=lambda i: w_arr[i], reverse=True)
        bins_ffd = []
        loads_ffd = []
        for idx in sorted_indices:
            w = w_arr[idx]
            placed = False
            for b in range(len(bins_ffd)):
                if loads_ffd[b] + w <= C:
                    bins_ffd[b].append(idx)
                    loads_ffd[b] += w
                    placed = True
                    break
            if not placed:
                bins_ffd.append([idx])
                loads_ffd.append(w)
        return bins_ffd, loads_ffd

    def best_fit_decreasing():
        sorted_indices = sorted(range(n), key=lambda i: w_arr[i], reverse=True)
        bins_bfd = []
        loads_bfd = []
        for idx in sorted_indices:
            w = w_arr[idx]
            best_b = -1
            best_remaining = float('inf')
            for b in range(len(bins_bfd)):
                remaining = C - loads_bfd[b] - w
                if 0 <= remaining < best_remaining:
                    best_remaining = remaining
                    best_b = b
            if best_b >= 0:
                bins_bfd[best_b].append(idx)
                loads_bfd[best_b] += w
            else:
                bins_bfd.append([idx])
                loads_bfd.append(w)
        return bins_bfd, loads_bfd

    ffd_bins, ffd_loads = first_fit_decreasing()
    bfd_bins, bfd_loads = best_fit_decreasing()

    if len(bfd_bins) < len(ffd_bins):
        init_bins = bfd_bins
        init_loads = bfd_loads
    else:
        init_bins = ffd_bins
        init_loads = ffd_loads

    best_packing = [sorted(b) for b in init_bins]
    best_bin_weights = list(init_loads)
    best_num_bins = len(init_bins)

    if best_num_bins <= lower_bound:
        return {"packing": best_packing, "bin_weights": best_bin_weights}

    # Try to reduce number of bins using tabu search
    target_bins = best_num_bins - 1

    while target_bins >= lower_bound and time.time() - start_time < time_limit * 0.95:
        time_at_target_start = time.time()
        remaining_time = time_limit * 0.95 - (time_at_target_start - start_time)
        if remaining_time <= 0:
            break
        time_limit_for_target = time_at_target_start + remaining_time * 0.85  # don't spend all time on one target

        # --- Improved merge strategy: try multiple random merges ---
        best_init_overflow = float('inf')
        best_init_bins_config = None
        best_init_loads_config = None
        best_init_item_bin = None

        num_merge_attempts = 5

        for attempt in range(num_merge_attempts):
            if time.time() - start_time >= time_limit * 0.95:
                break
            cur_bins = [list(b) for b in best_packing]
            cur_loads = list(best_bin_weights)

            if attempt == 0:
                # Deterministic: merge lightest bins
                while len(cur_bins) > target_bins:
                    min_idx = min(range(len(cur_bins)), key=lambda i: cur_loads[i])
                    items_to_redistribute = list(cur_bins[min_idx])
                    del cur_bins[min_idx]
                    del cur_loads[min_idx]
                    items_to_redistribute.sort(key=lambda i: w_arr[i], reverse=True)
                    for item in items_to_redistribute:
                        w = w_arr[item]
                        best_b = -1
                        best_rem = float('inf')
                        for b in range(len(cur_bins)):
                            rem = C - cur_loads[b] - w
                            if 0 <= rem < best_rem:
                                best_rem = rem
                                best_b = b
                        if best_b == -1:
                            best_b = min(range(len(cur_bins)), key=lambda b: cur_loads[b])
                        cur_bins[best_b].append(item)
                        cur_loads[best_b] += w
            else:
                # Random merge
                while len(cur_bins) > target_bins:
                    # Pick random bin to dissolve (biased towards lighter bins)
                    idxs = list(range(len(cur_bins)))
                    # Weight by inverse load
                    inv_loads = [1.0 / (cur_loads[i] + 1) for i in idxs]
                    total_inv = sum(inv_loads)
                    r = random.random() * total_inv
                    cum = 0
                    chosen_idx = idxs[-1]
                    for i in idxs:
                        cum += inv_loads[i]
                        if cum >= r:
                            chosen_idx = i
                            break
                    items_to_redistribute = list(cur_bins[chosen_idx])
                    del cur_bins[chosen_idx]
                    del cur_loads[chosen_idx]
                    random.shuffle(items_to_redistribute)
                    items_to_redistribute.sort(key=lambda i: w_arr[i], reverse=True)
                    for item in items_to_redistribute:
                        w = w_arr[item]
                        best_b = -1
                        best_rem = float('inf')
                        for b in range(len(cur_bins)):
                            rem = C - cur_loads[b] - w
                            if 0 <= rem < best_rem:
                                best_rem = rem
                                best_b = b
                        if best_b == -1:
                            best_b = min(range(len(cur_bins)), key=lambda b: cur_loads[b])
                        cur_bins[best_b].append(item)
                        cur_loads[best_b] += w

            ov = sum(max(0, cur_loads[b] - C) for b in range(len(cur_bins)))
            if ov < best_init_overflow:
                best_init_overflow = ov
                best_init_bins_config = cur_bins
                best_init_loads_config = cur_loads

        if best_init_bins_config is None:
            break

        current_bins_list = best_init_bins_config  # list of lists
        current_loads = best_init_loads_config
        num_bins = len(current_bins_list)

        # Build sets and item_bin mapping
        current_bins = [set(b) for b in current_bins_list]
        item_bin = [0] * n
        for b in range(num_bins):
            for item in current_bins[b]:
                item_bin[item] = b

        # Build sorted item lists per bin (sorted by weight)
        bin_sorted = [sorted(current_bins[b], key=lambda x: w_arr[x]) for b in range(num_bins)]
        bin_sorted_weights = [[w_arr[x] for x in bin_sorted[b]] for b in range(num_bins)]

        def remove_from_sorted(b, item):
            w = w_arr[item]
            idx = bisect.bisect_left(bin_sorted_weights[b], w)
            while idx < len(bin_sorted[b]) and bin_sorted[b][idx] != item:
                idx += 1
            if idx < len(bin_sorted[b]):
                del bin_sorted[b][idx]
                del bin_sorted_weights[b][idx]

        def add_to_sorted(b, item):
            w = w_arr[item]
            idx = bisect.bisect_left(bin_sorted_weights[b], w)
            bin_sorted[b].insert(idx, item)
            bin_sorted_weights[b].insert(idx, w)

        overflow = sum(max(0, current_loads[b] - C) for b in range(num_bins))

        if overflow == 0:
            best_packing = [sorted(b) for b in current_bins]
            best_bin_weights = list(current_loads)
            best_num_bins = target_bins
            target_bins -= 1
            continue

        # Tabu search
        tabu = {}
        tabu_tenure = max(7, int(n ** 0.5))

        best_overflow = overflow
        best_config_bins = [set(b) for b in current_bins]
        best_config_loads = list(current_loads)
        best_config_item_bin = list(item_bin)
        best_config_sorted = [list(bs) for bs in bin_sorted]
        best_config_sorted_weights = [list(bsw) for bsw in bin_sorted_weights]

        max_iter_no_improve = max(5000, n * 30)
        no_improve_count = 0
        iteration = 0
        restarts = 0

        # For adaptive tabu tenure - cycle detection
        recent_overflow_patterns = {}

        while time.time() - start_time < time_limit * 0.95:
            iteration += 1

            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                if elapsed >= time_limit * 0.95:
                    break
                if time.time() > time_limit_for_target:
                    break

            # Find overflowing bins
            overflow_bins = [b for b in range(num_bins) if current_loads[b] > C]

            if not overflow_bins:
                break

            # Generate candidate moves
            best_move = None
            best_move_delta = float('inf')
            tie_candidates = []

            # Sampling limits
            if num_bins < 10:
                max_ob_sample = len(overflow_bins)
                max_item_sample = 50
                max_target_swap = num_bins
                max_tb_items = 30
                max_target_12 = num_bins
                max_j_sample = 15
            else:
                max_ob_sample = min(5, len(overflow_bins))
                max_item_sample = 20
                max_target_swap = 8
                max_tb_items = 15
                max_target_12 = 5
                max_j_sample = 8

            sampled_overflow = overflow_bins if len(overflow_bins) <= max_ob_sample else random.sample(overflow_bins, max_ob_sample)

            for ob in sampled_overflow:
                ob_items = list(current_bins[ob])
                if len(ob_items) > max_item_sample:
                    sampled_items = random.sample(ob_items, max_item_sample)
                else:
                    sampled_items = ob_items

                old_ov_ob = max(0, current_loads[ob] - C)

                for item in sampled_items:
                    w = w_arr[item]
                    new_ov_ob = max(0, current_loads[ob] - w - C)

                    # Transfer moves
                    for tb in range(num_bins):
                        if tb == ob:
                            continue
                        old_ov_tb = max(0, current_loads[tb] - C)
                        new_ov_tb = max(0, current_loads[tb] + w - C)
                        delta = (new_ov_ob - old_ov_ob) + (new_ov_tb - old_ov_tb)

                        is_tabu = tabu.get((item, tb), 0) > iteration
                        aspiration = (overflow + delta < best_overflow) if is_tabu else False

                        if not is_tabu or aspiration:
                            if delta < best_move_delta:
                                best_move_delta = delta
                                tie_candidates = [('transfer', item, ob, tb)]
                            elif delta == best_move_delta:
                                tie_candidates.append(('transfer', item, ob, tb))

                    # (1,1)-Swap moves
                    non_overflow = [b for b in range(num_bins) if b != ob and current_loads[b] <= C]
                    if len(non_overflow) > max_target_swap:
                        sampled_targets = random.sample(non_overflow, max_target_swap)
                    else:
                        sampled_targets = non_overflow

                    for tb in sampled_targets:
                        tb_items_list = list(current_bins[tb])
                        if len(tb_items_list) > max_tb_items:
                            sampled_tb_items = random.sample(tb_items_list, max_tb_items)
                        else:
                            sampled_tb_items = tb_items_list

                        for item2 in sampled_tb_items:
                            w2 = w_arr[item2]
                            if w2 >= w:
                                continue

                            new_load_ob = current_loads[ob] - w + w2
                            new_load_tb = current_loads[tb] + w - w2

                            new_ov_ob2 = max(0, new_load_ob - C)
                            old_ov_tb2 = max(0, current_loads[tb] - C)
                            new_ov_tb2 = max(0, new_load_tb - C)

                            delta = (new_ov_ob2 - old_ov_ob) + (new_ov_tb2 - old_ov_tb2)

                            is_tabu1 = tabu.get((item, tb), 0) > iteration
                            is_tabu2 = tabu.get((item2, ob), 0) > iteration
                            is_tabu_move = is_tabu1 or is_tabu2
                            aspiration = (overflow + delta < best_overflow) if is_tabu_move else False

                            if not is_tabu_move or aspiration:
                                if delta < best_move_delta:
                                    best_move_delta = delta
                                    tie_candidates = [('swap', item, ob, item2, tb)]
                                elif delta == best_move_delta:
                                    tie_candidates.append(('swap', item, ob, item2, tb))

                    # (1,2)-Interchange moves
                    all_targets = [b for b in range(num_bins) if b != ob]
                    if len(all_targets) > max_target_12:
                        sampled_targets_12 = random.sample(all_targets, max_target_12)
                    else:
                        sampled_targets_12 = all_targets

                    for tb in sampled_targets_12:
                        if len(bin_sorted[tb]) < 2:
                            continue
                        # We move item (weight w) from ob to tb
                        # We move back items j, k from tb to ob
                        # Want w_j + w_k < w (so ob overflow decreases)
                        # Compute delta:
                        # ob: loses w, gains w_j + w_k
                        # tb: gains w, loses w_j + w_k
                        # For best result, maximize w_j + w_k (but < w)
                        
                        sorted_items_tb = bin_sorted[tb]
                        sorted_w_tb = bin_sorted_weights[tb]
                        len_tb = len(sorted_items_tb)
                        
                        # Sample j candidates
                        # j should be from the larger items in tb for maximum w_j+w_k
                        # But w_j < w (otherwise w_j + w_k >= w for any positive w_k)
                        # Find index where weight < w
                        max_j_idx = bisect.bisect_left(sorted_w_tb, w)
                        if max_j_idx < 1:  # need at least j and k
                            continue
                        
                        # Sample j from larger end
                        j_candidates_indices = list(range(max(0, max_j_idx - max_j_sample), max_j_idx))
                        if len(j_candidates_indices) > max_j_sample:
                            j_candidates_indices = random.sample(j_candidates_indices, max_j_sample)

                        for j_idx in j_candidates_indices:
                            j_item = sorted_items_tb[j_idx]
                            wj = sorted_w_tb[j_idx]
                            if j_item == item:
                                continue
                            # Need w_k <= w - wj - 1 and k != j
                            max_wk = w - wj - 1
                            if max_wk < 1:
                                continue
                            # Find largest k with weight <= max_wk, k != j
                            k_max_idx = bisect.bisect_right(sorted_w_tb, max_wk) - 1
                            if k_max_idx < 0:
                                continue
                            # Avoid j
                            k_idx = k_max_idx
                            if k_idx == j_idx:
                                k_idx -= 1
                            if k_idx < 0:
                                continue
                            k_item = sorted_items_tb[k_idx]
                            wk = sorted_w_tb[k_idx]
                            if k_item == item or k_item == j_item:
                                # try next
                                k_idx -= 1
                                if k_idx < 0:
                                    continue
                                if k_idx == j_idx:
                                    k_idx -= 1
                                if k_idx < 0:
                                    continue
                                k_item = sorted_items_tb[k_idx]
                                wk = sorted_w_tb[k_idx]

                            if wj + wk >= w:
                                continue

                            net_change = w - wj - wk  # positive: tb gains this net
                            new_load_ob_12 = current_loads[ob] - w + wj + wk
                            new_load_tb_12 = current_loads[tb] + net_change

                            new_ov_ob_12 = max(0, new_load_ob_12 - C)
                            old_ov_tb_12 = max(0, current_loads[tb] - C)
                            new_ov_tb_12 = max(0, new_load_tb_12 - C)

                            delta = (new_ov_ob_12 - old_ov_ob) + (new_ov_tb_12 - old_ov_tb_12)

                            is_tabu1 = tabu.get((item, tb), 0) > iteration
                            is_tabu2 = tabu.get((j_item, ob), 0) > iteration
                            is_tabu3 = tabu.get((k_item, ob), 0) > iteration
                            is_tabu_move = is_tabu1 or is_tabu2 or is_tabu3
                            aspiration = (overflow + delta < best_overflow) if is_tabu_move else False

                            if not is_tabu_move or aspiration:
                                if delta < best_move_delta:
                                    best_move_delta = delta
                                    tie_candidates = [('swap12', item, ob, j_item, k_item, tb)]
                                elif delta == best_move_delta:
                                    tie_candidates.append(('swap12', item, ob, j_item, k_item, tb))

            if not tie_candidates:
                # All moves tabu, force a random move
                ob = random.choice(overflow_bins)
                if current_bins[ob]:
                    item = random.choice(list(current_bins[ob]))
                    tb = random.choice([b for b in range(num_bins) if b != ob])
                    w = w_arr[item]
                    old_ov_ob = max(0, current_loads[ob] - C)
                    new_ov_ob = max(0, current_loads[ob] - w - C)
                    old_ov_tb = max(0, current_loads[tb] - C)
                    new_ov_tb = max(0, current_loads[tb] + w - C)
                    best_move_delta = (new_ov_ob - old_ov_ob) + (new_ov_tb - old_ov_tb)
                    tie_candidates = [('transfer', item, ob, tb)]
                else:
                    no_improve_count += 1
                    if no_improve_count >= max_iter_no_improve:
                        break
                    continue

            if not tie_candidates:
                break

            # Pick from ties
            if len(tie_candidates) > 1:
                chosen = random.choice(tie_candidates)
            else:
                chosen = tie_candidates[0]

            # Apply move
            if chosen[0] == 'transfer':
                _, item, ob, tb = chosen
                w = w_arr[item]
                current_bins[ob].discard(item)
                current_bins[tb].add(item)
                current_loads[ob] -= w
                current_loads[tb] += w
                remove_from_sorted(ob, item)
                add_to_sorted(tb, item)
                item_bin[item] = tb
                tabu[(item, ob)] = iteration + tabu_tenure + random.randint(0, 4)
                overflow += best_move_delta
            elif chosen[0] == 'swap':
                _, item1, ob, item2, tb = chosen
                w1 = w_arr[item1]
                w2 = w_arr[item2]
                current_bins[ob].discard(item1)
                current_bins[ob].add(item2)
                current_bins[tb].discard(item2)
                current_bins[tb].add(item1)
                current_loads[ob] = current_loads[ob] - w1 + w2
                current_loads[tb] = current_loads[tb] - w2 + w1
                remove_from_sorted(ob, item1)
                add_to_sorted(ob, item2)
                remove_from_sorted(tb, item2)
                add_to_sorted(tb, item1)
                item_bin[item1] = tb
                item_bin[item2] = ob
                tabu[(item1, ob)] = iteration + tabu_tenure + random.randint(0, 4)
                tabu[(item2, tb)] = iteration + tabu_tenure + random.randint(0, 4)
                overflow += best_move_delta
            elif chosen[0] == 'swap12':
                _, item_i, ob, j_item, k_item, tb = chosen
                wi = w_arr[item_i]
                wj = w_arr[j_item]
                wk = w_arr[k_item]
                # Move item_i from ob to tb
                current_bins[ob].discard(item_i)
                current_bins[tb].add(item_i)
                # Move j_item and k_item from tb to ob
                current_bins[tb].discard(j_item)
                current_bins[tb].discard(k_item)
                current_bins[ob].add(j_item)
                current_bins[ob].add(k_item)
                current_loads[ob] = current_loads[ob] - wi + wj + wk
                current_loads[tb] = current_loads[tb] + wi - wj - wk
                remove_from_sorted(ob, item_i)
                add_to_sorted(tb, item_i)
                remove_from_sorted(tb, j_item)
                remove_from_sorted(tb, k_item)
                add_to_sorted(ob, j_item)
                add_to_sorted(ob, k_item)
                item_bin[item_i] = tb
                item_bin[j_item] = ob
                item_bin[k_item] = ob
                tabu[(item_i, ob)] = iteration + tabu_tenure + random.randint(0, 4)
                tabu[(j_item, tb)] = iteration + tabu_tenure + random.randint(0, 4)
                tabu[(k_item, tb)] = iteration + tabu_tenure + random.randint(0, 4)
                overflow += best_move_delta

            # Adaptive tabu tenure
            if iteration % 50 == 0:
                ov_pattern = tuple(sorted([max(0, current_loads[b] - C) for b in range(num_bins) if current_loads[b] > C]))
                if ov_pattern in recent_overflow_patterns and iteration - recent_overflow_patterns[ov_pattern] < 50:
                    tabu_tenure = min(tabu_tenure + 3, n)
                recent_overflow_patterns[ov_pattern] = iteration
                # Clean old patterns
                if len(recent_overflow_patterns) > 1000:
                    recent_overflow_patterns.clear()

            if overflow < best_overflow:
                best_overflow = overflow
                best_config_bins = [set(b) for b in current_bins]
                best_config_loads = list(current_loads)
                best_config_item_bin = list(item_bin)
                best_config_sorted = [list(bs) for bs in bin_sorted]
                best_config_sorted_weights = [list(bsw) for bsw in bin_sorted_weights]
                no_improve_count = 0

                if overflow == 0:
                    break
            else:
                no_improve_count += 1
                # Adaptive: decrease tenure if stuck
                if no_improve_count % 500 == 0 and no_improve_count > 0:
                    tabu_tenure = max(5, tabu_tenure - 2)

            if no_improve_count >= max_iter_no_improve:
                restarts += 1
                # Stronger perturbation based on restart count
                current_bins = [set(b) for b in best_config_bins]
                current_loads = list(best_config_loads)
                item_bin = list(best_config_item_bin)
                bin_sorted = [list(bs) for bs in best_config_sorted]
                bin_sorted_weights = [list(bsw) for bsw in best_config_sorted_weights]

                if restarts <= 3:
                    perturb_count = max(1, n // 10)
                elif restarts <= 6:
                    perturb_count = max(1, n // 5)
                else:
                    # Reconstruct: empty 2-3 bins and redistribute
                    bins_to_empty = random.sample(range(num_bins), min(3, num_bins))
                    items_to_redistribute = []
                    for be in bins_to_empty:
                        items_to_redistribute.extend(list(current_bins[be]))
                        for it in list(current_bins[be]):
                            remove_from_sorted(be, it)
                        current_bins[be] = set()
                        current_loads[be] = 0
                    # BFD redistribute
                    items_to_redistribute.sort(key=lambda i: w_arr[i], reverse=True)
                    for it in items_to_redistribute:
                        w = w_arr[it]
                        best_b = -1
                        best_rem = float('inf')
                        for b in range(num_bins):
                            rem = C - current_loads[b] - w
                            if 0 <= rem < best_rem:
                                best_rem = rem
                                best_b = b
                        if best_b == -1:
                            best_b = min(range(num_bins), key=lambda b: current_loads[b])
                        current_bins[best_b].add(it)
                        current_loads[best_b] += w
                        add_to_sorted(best_b, it)
                        item_bin[it] = best_b
                    perturb_count = 0  # already perturbed

                if restarts <= 6 or True:
                    for _ in range(perturb_count if 'perturb_count' in dir() else 0):
                        b1 = random.randint(0, num_bins - 1)
                        if current_bins[b1]:
                            it = random.choice(list(current_bins[b1]))
                            b2 = random.randint(0, num_bins - 1)
                            if b2 != b1:
                                current_bins[b1].discard(it)
                                current_bins[b2].add(it)
                                current_loads[b1] -= w_arr[it]
                                current_loads[b2] += w_arr[it]
                                remove_from_sorted(b1, it)
                                add_to_sorted(b2, it)
                                item_bin[it] = b2

                overflow = sum(max(0, current_loads[b] - C) for b in range(num_bins))
                tabu.clear()
                recent_overflow_patterns.clear()
                no_improve_count = 0
                tabu_tenure = max(7, int(n ** 0.5)) + random.randint(0, 5)
                max_iter_no_improve = max(2000, n * 15)

        if best_overflow == 0 or overflow == 0:
            if overflow == 0:
                use_bins = current_bins
                use_loads = current_loads
            else:
                use_bins = best_config_bins
                use_loads = best_config_loads
            best_packing = [sorted(b) for b in use_bins]
            best_bin_weights = list(use_loads)
            best_num_bins = target_bins
            target_bins -= 1
        else:
            break

    # Remove empty bins
    final_packing = [b for b in best_packing if len(b) > 0]
    final_weights = [sum(w_arr[i] for i in b) for b in final_packing]

    return {"packing": final_packing, "bin_weights": final_weights}