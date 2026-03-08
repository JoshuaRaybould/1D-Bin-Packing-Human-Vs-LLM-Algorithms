def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    import random
    import math

    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = bin_capacity
    w = list(weights)

    total_weight = sum(w)
    L1 = math.ceil(total_weight / C)

    # Compute L2 lower bound
    def compute_lower_bound():
        lb = L1
        half_C = C / 2.0
        alphas = set()
        for i in range(n):
            wi = w[i]
            if wi <= C // 2:
                alphas.add(wi)
            diff = C - wi
            if 1 <= diff <= C // 2:
                alphas.add(diff)
        alphas.add(1)
        if C // 2 >= 1:
            alphas.add(C // 2)
        if len(alphas) > 300:
            alphas = set(random.sample(sorted(alphas), 300))
        for alpha in alphas:
            if alpha < 1 or alpha > C // 2:
                continue
            n1 = 0; s1 = 0; n2 = 0; s2 = 0; s3 = 0
            threshold_high = C - alpha
            for i in range(n):
                wi = w[i]
                if wi > threshold_high:
                    n1 += 1; s1 += wi
                elif wi > half_C:
                    n2 += 1; s2 += wi
                elif wi >= alpha:
                    s3 += wi
            slack = n2 * C - s1 - s2
            extra = 0
            if s3 > slack:
                extra = math.ceil((s3 - slack) / C)
            val = n1 + n2 + max(0, extra)
            if val > lb:
                lb = val
        return lb

    lower_bound = compute_lower_bound()

    # Multiple initial heuristics
    def first_fit_decreasing():
        order = sorted(range(n), key=lambda i: w[i], reverse=True)
        bins = []; loads = []
        for idx in order:
            wi = w[idx]
            placed = False
            for b in range(len(bins)):
                if loads[b] + wi <= C:
                    bins[b].append(idx); loads[b] += wi; placed = True; break
            if not placed:
                bins.append([idx]); loads.append(wi)
        return bins, loads

    def best_fit_decreasing():
        order = sorted(range(n), key=lambda i: w[i], reverse=True)
        bins = []; loads = []
        for idx in order:
            wi = w[idx]
            best_b = -1; best_rem = C + 1
            for b in range(len(bins)):
                rem = C - loads[b] - wi
                if 0 <= rem < best_rem:
                    best_rem = rem; best_b = b
            if best_b >= 0:
                bins[best_b].append(idx); loads[best_b] += wi
            else:
                bins.append([idx]); loads.append(wi)
        return bins, loads

    def worst_fit_decreasing():
        order = sorted(range(n), key=lambda i: w[i], reverse=True)
        bins = []; loads = []
        for idx in order:
            wi = w[idx]
            best_b = -1; best_rem = -1
            for b in range(len(bins)):
                rem = C - loads[b] - wi
                if rem >= 0 and rem > best_rem:
                    best_rem = rem; best_b = b
            if best_b >= 0:
                bins[best_b].append(idx); loads[best_b] += wi
            else:
                bins.append([idx]); loads.append(wi)
        return bins, loads

    ffd_b, ffd_l = first_fit_decreasing()
    bfd_b, bfd_l = best_fit_decreasing()
    wfd_b, wfd_l = worst_fit_decreasing()

    candidates = [(ffd_b, ffd_l), (bfd_b, bfd_l), (wfd_b, wfd_l)]
    best_init = min(candidates, key=lambda x: len(x[0]))
    
    best_packing = [sorted(b) for b in best_init[0]]
    best_bin_weights = list(best_init[1])
    best_num_bins = len(best_init[0])

    if best_num_bins <= lower_bound:
        return {"packing": best_packing, "bin_weights": best_bin_weights}

    target_bins = best_num_bins - 1
    
    time_fraction = 0.98

    while target_bins >= lower_bound and time.time() - start_time < time_limit * time_fraction:
        time_at_target = time.time()
        remaining = time_limit * time_fraction - (time_at_target - start_time)
        if remaining <= 0.05:
            break
        # Allocate time proportionally - more time for harder targets
        time_for_target = time_at_target + remaining * 0.7

        # Try multiple initializations
        best_init_overflow = float('inf')
        best_init_config = None

        num_attempts = min(10, max(3, int(remaining * 2)))

        for attempt in range(num_attempts):
            if time.time() - start_time >= time_limit * time_fraction:
                break
            cur_bins = [list(b) for b in best_packing]
            cur_loads = list(best_bin_weights)

            if attempt == 0:
                # Merge lightest bin
                while len(cur_bins) > target_bins:
                    min_idx = min(range(len(cur_bins)), key=lambda i: cur_loads[i])
                    items_r = list(cur_bins[min_idx])
                    del cur_bins[min_idx]; del cur_loads[min_idx]
                    items_r.sort(key=lambda i: w[i], reverse=True)
                    for it in items_r:
                        wi = w[it]
                        bb = -1; br = C + 1
                        for b in range(len(cur_bins)):
                            r = C - cur_loads[b] - wi
                            if 0 <= r < br:
                                br = r; bb = b
                        if bb == -1:
                            bb = min(range(len(cur_bins)), key=lambda b: cur_loads[b])
                        cur_bins[bb].append(it); cur_loads[bb] += wi
            elif attempt == 1:
                # Merge most overflowing potential pair
                while len(cur_bins) > target_bins:
                    min_idx = max(range(len(cur_bins)), key=lambda i: cur_loads[i])
                    # Actually merge lightest
                    min_idx = min(range(len(cur_bins)), key=lambda i: cur_loads[i])
                    items_r = list(cur_bins[min_idx])
                    del cur_bins[min_idx]; del cur_loads[min_idx]
                    # First fit
                    for it in sorted(items_r, key=lambda i: w[i], reverse=True):
                        wi = w[it]
                        placed = False
                        for b in range(len(cur_bins)):
                            if cur_loads[b] + wi <= C:
                                cur_bins[b].append(it); cur_loads[b] += wi; placed = True; break
                        if not placed:
                            bb = min(range(len(cur_bins)), key=lambda b: cur_loads[b])
                            cur_bins[bb].append(it); cur_loads[bb] += wi
            else:
                # Random merge
                while len(cur_bins) > target_bins:
                    idxs = list(range(len(cur_bins)))
                    inv_loads = [1.0 / (cur_loads[i] + 1) for i in idxs]
                    total_inv = sum(inv_loads)
                    r = random.random() * total_inv
                    cum = 0; chosen_idx = idxs[-1]
                    for i in idxs:
                        cum += inv_loads[i]
                        if cum >= r:
                            chosen_idx = i; break
                    items_r = list(cur_bins[chosen_idx])
                    del cur_bins[chosen_idx]; del cur_loads[chosen_idx]
                    random.shuffle(items_r)
                    items_r.sort(key=lambda i: w[i], reverse=True)
                    for it in items_r:
                        wi = w[it]
                        bb = -1; br = C + 1
                        for b in range(len(cur_bins)):
                            rem = C - cur_loads[b] - wi
                            if 0 <= rem < br:
                                br = rem; bb = b
                        if bb == -1:
                            bb = min(range(len(cur_bins)), key=lambda b: cur_loads[b])
                        cur_bins[bb].append(it); cur_loads[bb] += wi

            ov = sum(max(0, cur_loads[b] - C) for b in range(len(cur_bins)))
            if ov < best_init_overflow:
                best_init_overflow = ov
                best_init_config = (cur_bins, cur_loads)
                if ov == 0:
                    break

        if best_init_config is None:
            break

        if best_init_overflow == 0:
            cur_bins, cur_loads = best_init_config
            best_packing = [sorted(b) for b in cur_bins]
            best_bin_weights = list(cur_loads)
            best_num_bins = target_bins
            target_bins -= 1
            continue

        # Tabu search to eliminate overflow
        cur_bins_list, cur_loads = best_init_config
        num_bins = len(cur_bins_list)
        
        # item_bin[i] = which bin item i is in
        item_bin = [0] * n
        bin_items = [[] for _ in range(num_bins)]  # list of items per bin
        for b in range(num_bins):
            bin_items[b] = list(cur_bins_list[b])
            for it in bin_items[b]:
                item_bin[it] = b
        loads = list(cur_loads)

        overflow = sum(max(0, loads[b] - C) for b in range(num_bins))

        tabu = {}  # (item, bin) -> iteration when tabu expires
        base_tenure = max(7, int(math.sqrt(n)))
        tabu_tenure = base_tenure

        best_overflow = overflow
        best_state_item_bin = list(item_bin)
        best_state_loads = list(loads)
        best_state_bin_items = [list(b) for b in bin_items]

        no_improve = 0
        max_no_improve = max(3000, n * 20)
        iteration = 0
        
        found_feasible = False

        while True:
            iteration += 1
            
            if iteration % 50 == 0:
                if time.time() - start_time >= time_limit * time_fraction:
                    break
                if time.time() > time_for_target:
                    break

            # Find overflowing bins
            ov_bins = []
            for b in range(num_bins):
                if loads[b] > C:
                    ov_bins.append(b)
            
            if not ov_bins:
                found_feasible = True
                break

            best_delta = float('inf')
            best_moves = []

            # For each overflowing bin, try transfers and swaps
            if len(ov_bins) > 5:
                sampled_ov = random.sample(ov_bins, 5)
            else:
                sampled_ov = ov_bins

            for ob in sampled_ov:
                ob_ov = loads[ob] - C  # positive
                ob_items = bin_items[ob]
                
                if len(ob_items) > 30:
                    sampled_items = random.sample(ob_items, 30)
                else:
                    sampled_items = ob_items

                for item_i in sampled_items:
                    wi = w[item_i]
                    new_ob_ov = max(0, loads[ob] - wi - C)
                    delta_ob = new_ob_ov - ob_ov

                    # Transfer: move item_i from ob to tb
                    for tb in range(num_bins):
                        if tb == ob:
                            continue
                        tb_ov_old = max(0, loads[tb] - C)
                        tb_ov_new = max(0, loads[tb] + wi - C)
                        delta = delta_ob + (tb_ov_new - tb_ov_old)

                        is_tabu = tabu.get((item_i, tb), 0) > iteration
                        if is_tabu and not (overflow + delta < best_overflow):
                            continue

                        if delta < best_delta:
                            best_delta = delta
                            best_moves = [('t', item_i, ob, tb)]
                        elif delta == best_delta:
                            best_moves.append(('t', item_i, ob, tb))

                    # Swap: move item_i from ob to tb, move item_j from tb to ob
                    # Only consider non-overflow or less-overflow target bins
                    for tb in range(num_bins):
                        if tb == ob:
                            continue
                        tb_items = bin_items[tb]
                        if len(tb_items) > 20:
                            sampled_tb = random.sample(tb_items, 20)
                        else:
                            sampled_tb = tb_items
                        
                        for item_j in sampled_tb:
                            wj = w[item_j]
                            if wj >= wi:
                                continue
                            
                            new_load_ob = loads[ob] - wi + wj
                            new_load_tb = loads[tb] + wi - wj
                            
                            new_ob_ov2 = max(0, new_load_ob - C)
                            old_tb_ov = max(0, loads[tb] - C)
                            new_tb_ov = max(0, new_load_tb - C)
                            
                            delta = (new_ob_ov2 - ob_ov) + (new_tb_ov - old_tb_ov)

                            is_tabu1 = tabu.get((item_i, tb), 0) > iteration
                            is_tabu2 = tabu.get((item_j, ob), 0) > iteration
                            if (is_tabu1 or is_tabu2) and not (overflow + delta < best_overflow):
                                continue

                            if delta < best_delta:
                                best_delta = delta
                                best_moves = [('s', item_i, ob, item_j, tb)]
                            elif delta == best_delta:
                                best_moves.append(('s', item_i, ob, item_j, tb))

            if not best_moves:
                # Force random move
                ob = random.choice(ov_bins)
                if bin_items[ob]:
                    item_i = random.choice(bin_items[ob])
                    tb = random.choice([b for b in range(num_bins) if b != ob])
                    best_moves = [('t', item_i, ob, tb)]
                    wi = w[item_i]
                    old_ob_ov = max(0, loads[ob] - C)
                    new_ob_ov = max(0, loads[ob] - wi - C)
                    old_tb_ov = max(0, loads[tb] - C)
                    new_tb_ov = max(0, loads[tb] + wi - C)
                    best_delta = (new_ob_ov - old_ob_ov) + (new_tb_ov - old_tb_ov)
                else:
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break
                    continue

            chosen = random.choice(best_moves) if len(best_moves) > 1 else best_moves[0]

            # Apply move
            if chosen[0] == 't':
                _, item_i, ob, tb = chosen
                wi = w[item_i]
                bin_items[ob].remove(item_i)
                bin_items[tb].append(item_i)
                loads[ob] -= wi
                loads[tb] += wi
                item_bin[item_i] = tb
                tabu[(item_i, ob)] = iteration + tabu_tenure + random.randint(0, 4)
                overflow += best_delta
            elif chosen[0] == 's':
                _, item_i, ob, item_j, tb = chosen
                wi = w[item_i]; wj = w[item_j]
                bin_items[ob].remove(item_i)
                bin_items[ob].append(item_j)
                bin_items[tb].remove(item_j)
                bin_items[tb].append(item_i)
                loads[ob] = loads[ob] - wi + wj
                loads[tb] = loads[tb] - wj + wi
                item_bin[item_i] = tb
                item_bin[item_j] = ob
                tabu[(item_i, ob)] = iteration + tabu_tenure + random.randint(0, 4)
                tabu[(item_j, tb)] = iteration + tabu_tenure + random.randint(0, 4)
                overflow += best_delta

            if overflow < best_overflow:
                best_overflow = overflow
                best_state_item_bin = list(item_bin)
                best_state_loads = list(loads)
                best_state_bin_items = [list(b) for b in bin_items]
                no_improve = 0
                if overflow == 0:
                    found_feasible = True
                    break
            else:
                no_improve += 1

            # Adaptive tenure
            if no_improve > 0 and no_improve % 200 == 0:
                tabu_tenure = base_tenure + random.randint(-3, 5)
                if tabu_tenure < 4:
                    tabu_tenure = 4

            if no_improve >= max_no_improve:
                # Restart from best with perturbation
                item_bin = list(best_state_item_bin)
                loads = list(best_state_loads)
                bin_items = [list(b) for b in best_state_bin_items]
                overflow = best_overflow
                
                # Perturbation: randomly move some items
                perturb_count = max(2, n // 8)
                all_items = list(range(n))
                random.shuffle(all_items)
                for it in all_items[:perturb_count]:
                    old_b = item_bin[it]
                    new_b = random.randint(0, num_bins - 1)
                    if new_b != old_b:
                        bin_items[old_b].remove(it)
                        bin_items[new_b].append(it)
                        loads[old_b] -= w[it]
                        loads[new_b] += w[it]
                        item_bin[it] = new_b
                
                overflow = sum(max(0, loads[b] - C) for b in range(num_bins))
                tabu.clear()
                no_improve = 0
                tabu_tenure = base_tenure + random.randint(0, 5)
                max_no_improve = max(2000, n * 15)

        if found_feasible or best_overflow == 0:
            if found_feasible:
                use_bin_items = bin_items
                use_loads = loads
            else:
                use_bin_items = best_state_bin_items
                use_loads = best_state_loads
            best_packing = [sorted(b) for b in use_bin_items]
            best_bin_weights = list(use_loads)
            best_num_bins = target_bins
            target_bins -= 1
        else:
            break

    final_packing = [b for b in best_packing if len(b) > 0]
    final_weights = [sum(w[i] for i in b) for b in final_packing]
    return {"packing": final_packing, "bin_weights": final_weights}