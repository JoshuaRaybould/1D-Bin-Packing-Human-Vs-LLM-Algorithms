# anthropic
# tabu_search_2_performance_4.py

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    import random

    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Compute lower bound (L2)
    def compute_lower_bound():
        total_weight = sum(weights)
        lb1 = (total_weight + bin_capacity - 1) // bin_capacity
        # L2 bound
        half = bin_capacity / 2.0
        best_lb = lb1
        # Try different thresholds for L2
        for k in range(1, bin_capacity // 2 + 1):
            if time.time() - start_time > 0.5:
                break
            n1 = 0  # items > C - k
            n2 = 0  # items in (k, C-k]
            s2 = 0  # sum of items in (k, C-k]
            n3 = 0  # items <= k (not used directly in L2)
            for w in weights:
                if w > bin_capacity - k:
                    n1 += 1
                elif w > k:
                    n2 += 1
                    s2 += w
            lb = n1 + max(n2, (s2 + n1 * max(0, bin_capacity - (bin_capacity - k)) - n1 * bin_capacity + n1 * (bin_capacity - k) + bin_capacity - 1) // bin_capacity if False else 0)
            # Simpler L2: n1 + ceil(max(0, s2 - (n1*bin_capacity - sum of large items that fit with medium)) / C)
            # Actually let me use a cleaner L2
            pass
        # Simple but effective: continuous lower bound
        return lb1

    lower_bound = compute_lower_bound()

    # Initial solution: First Fit Decreasing
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    def first_fit_decreasing():
        bins = []
        bin_loads = []
        item_to_bin = [0] * n
        for idx in sorted_indices:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if bin_loads[b] + w <= bin_capacity:
                    bins[b].append(idx)
                    bin_loads[b] += w
                    item_to_bin[idx] = b
                    placed = True
                    break
            if not placed:
                item_to_bin[idx] = len(bins)
                bins.append([idx])
                bin_loads.append(w)
        return bins, bin_loads, item_to_bin

    def best_fit_decreasing():
        bins = []
        bin_loads = []
        item_to_bin = [0] * n
        for idx in sorted_indices:
            w = weights[idx]
            best_b = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bins)):
                remaining = bin_capacity - bin_loads[b]
                if remaining >= w and remaining - w < best_remaining:
                    best_remaining = remaining - w
                    best_b = b
            if best_b >= 0:
                bins[best_b].append(idx)
                bin_loads[best_b] += w
                item_to_bin[idx] = best_b
            else:
                item_to_bin[idx] = len(bins)
                bins.append([idx])
                bin_loads.append(w)
        return bins, bin_loads, item_to_bin

    bins_ffd, loads_ffd, _ = first_fit_decreasing()
    bins_bfd, loads_bfd, _ = best_fit_decreasing()

    if len(bins_bfd) <= len(bins_ffd):
        init_bins = [set(b) for b in bins_bfd]
        init_loads = list(loads_bfd)
    else:
        init_bins = [set(b) for b in bins_ffd]
        init_loads = list(loads_ffd)

    best_packing = [sorted(b) for b in init_bins]
    best_bin_weights = list(init_loads)
    best_num_bins = len(init_bins)

    # Try to reduce number of bins using tabu search
    target_bins = best_num_bins - 1

    while target_bins >= lower_bound and time.time() - start_time < time_limit * 0.95:
        # Initialize packing with target_bins bins by merging lightest bins
        current_bins = [set(b) for b in best_packing]
        current_loads = list(best_bin_weights)

        while len(current_bins) > target_bins:
            min_idx = min(range(len(current_bins)), key=lambda i: current_loads[i])
            items_to_redistribute = list(current_bins[min_idx])
            del current_bins[min_idx]
            del current_loads[min_idx]

            items_to_redistribute.sort(key=lambda i: weights[i], reverse=True)
            for item in items_to_redistribute:
                w = weights[item]
                best_b = -1
                best_remaining = float('inf')
                for b in range(len(current_bins)):
                    remaining = bin_capacity - current_loads[b] - w
                    if 0 <= remaining < best_remaining:
                        best_remaining = remaining
                        best_b = b
                if best_b == -1:
                    best_b = min(range(len(current_bins)), key=lambda b: current_loads[b])
                current_bins[best_b].add(item)
                current_loads[best_b] += w

        item_bin = [0] * n
        for b in range(len(current_bins)):
            for item in current_bins[b]:
                item_bin[item] = b

        num_bins = len(current_bins)

        overflow = sum(max(0, current_loads[b] - bin_capacity) for b in range(num_bins))

        if overflow == 0:
            best_packing = [sorted(b) for b in current_bins]
            best_bin_weights = list(current_loads)
            best_num_bins = target_bins
            target_bins -= 1
            continue

        # Tabu search to minimize total overflow
        tabu = {}
        base_tenure = max(7, int(n ** 0.5))
        tabu_tenure = base_tenure

        best_overflow = overflow
        best_config_bins = [set(b) for b in current_bins]
        best_config_loads = list(current_loads)
        best_config_item_bin = list(item_bin)

        max_iter_no_improve = max(5000, n * 30)
        no_improve_count = 0
        iteration = 0
        restarts = 0
        max_restarts = 10
        
        # Precompute weights array for quick access
        w_arr = weights

        while time.time() - start_time < time_limit * 0.95:
            iteration += 1

            if iteration % 500 == 0:
                if time.time() - start_time >= time_limit * 0.95:
                    break

            # Find overflowing bins
            overflow_bins = [b for b in range(num_bins) if current_loads[b] > bin_capacity]

            if not overflow_bins:
                break

            # Generate candidate moves
            best_move = None
            best_move_delta = float('inf')
            tie_count = 0

            # For transfer moves: iterate over items in overflow bins
            for ob in overflow_bins:
                ob_items = list(current_bins[ob])
                ob_load = current_loads[ob]
                old_ov_ob = ob_load - bin_capacity  # guaranteed > 0

                for item in ob_items:
                    w = w_arr[item]
                    new_ov_ob = max(0, ob_load - w - bin_capacity)

                    for tb in range(num_bins):
                        if tb == ob:
                            continue

                        tb_load = current_loads[tb]
                        old_ov_tb = max(0, tb_load - bin_capacity)
                        new_ov_tb = max(0, tb_load + w - bin_capacity)
                        delta = (new_ov_ob - old_ov_ob) + (new_ov_tb - old_ov_tb)

                        is_tabu = tabu.get((item, tb), 0) > iteration
                        if is_tabu and not (overflow + delta < best_overflow):
                            continue

                        if delta < best_move_delta:
                            best_move_delta = delta
                            best_move = ('t', item, ob, tb)
                            tie_count = 1
                        elif delta == best_move_delta:
                            tie_count += 1
                            if random.randint(1, tie_count) == 1:
                                best_move = ('t', item, ob, tb)

            # Swap moves: try swapping items between overflow and non-overflow bins
            non_overflow_bins = [b for b in range(num_bins) if current_loads[b] <= bin_capacity]
            
            # Limit swap exploration adaptively
            max_swap_obs = min(len(overflow_bins), 4)
            swap_obs = overflow_bins[:max_swap_obs] if len(overflow_bins) <= max_swap_obs else random.sample(overflow_bins, max_swap_obs)
            
            for ob in swap_obs:
                ob_items = list(current_bins[ob])
                ob_load = current_loads[ob]
                old_ov_ob = ob_load - bin_capacity

                # Sample items from ob if too many
                if len(ob_items) > 20:
                    s_ob_items = random.sample(ob_items, 20)
                else:
                    s_ob_items = ob_items

                max_swap_tbs = min(len(non_overflow_bins), 8)
                swap_tbs = non_overflow_bins[:max_swap_tbs] if len(non_overflow_bins) <= max_swap_tbs else random.sample(non_overflow_bins, max_swap_tbs)

                for tb in swap_tbs:
                    tb_items = list(current_bins[tb])
                    tb_load = current_loads[tb]
                    old_ov_tb = 0  # tb is non-overflow

                    if len(tb_items) > 15:
                        s_tb_items = random.sample(tb_items, 15)
                    else:
                        s_tb_items = tb_items

                    for item1 in s_ob_items:
                        w1 = w_arr[item1]
                        for item2 in s_tb_items:
                            w2 = w_arr[item2]
                            if w2 >= w1:
                                continue

                            new_load_ob = ob_load - w1 + w2
                            new_load_tb = tb_load + w1 - w2

                            new_ov_ob = max(0, new_load_ob - bin_capacity)
                            new_ov_tb = max(0, new_load_tb - bin_capacity)

                            delta = (new_ov_ob - old_ov_ob) + (new_ov_tb - old_ov_tb)

                            is_tabu1 = tabu.get((item1, tb), 0) > iteration
                            is_tabu2 = tabu.get((item2, ob), 0) > iteration
                            if (is_tabu1 or is_tabu2) and not (overflow + delta < best_overflow):
                                continue

                            if delta < best_move_delta:
                                best_move_delta = delta
                                best_move = ('s', item1, ob, item2, tb)
                                tie_count = 1
                            elif delta == best_move_delta:
                                tie_count += 1
                                if random.randint(1, tie_count) == 1:
                                    best_move = ('s', item1, ob, item2, tb)

            if best_move is None:
                # All moves tabu, force a random move
                ob = random.choice(overflow_bins)
                if current_bins[ob]:
                    item = random.choice(list(current_bins[ob]))
                    tb = random.choice([b for b in range(num_bins) if b != ob])
                    w = w_arr[item]
                    old_ov_ob = max(0, current_loads[ob] - bin_capacity)
                    new_ov_ob = max(0, current_loads[ob] - w - bin_capacity)
                    old_ov_tb = max(0, current_loads[tb] - bin_capacity)
                    new_ov_tb = max(0, current_loads[tb] + w - bin_capacity)
                    best_move_delta = (new_ov_ob - old_ov_ob) + (new_ov_tb - old_ov_tb)
                    best_move = ('t', item, ob, tb)
                else:
                    no_improve_count += 1
                    if no_improve_count >= max_iter_no_improve:
                        break
                    continue

            if best_move is None:
                break

            # Apply move
            if best_move[0] == 't':
                _, item, ob, tb = best_move
                w = w_arr[item]
                current_bins[ob].discard(item)
                current_bins[tb].add(item)
                current_loads[ob] -= w
                current_loads[tb] += w
                item_bin[item] = tb
                tabu[(item, ob)] = iteration + tabu_tenure + random.randint(0, 4)
                overflow += best_move_delta
            else:  # swap
                _, item1, ob, item2, tb = best_move
                w1 = w_arr[item1]
                w2 = w_arr[item2]
                current_bins[ob].discard(item1)
                current_bins[ob].add(item2)
                current_bins[tb].discard(item2)
                current_bins[tb].add(item1)
                current_loads[ob] = current_loads[ob] - w1 + w2
                current_loads[tb] = current_loads[tb] - w2 + w1
                item_bin[item1] = tb
                item_bin[item2] = ob
                tabu[(item1, ob)] = iteration + tabu_tenure + random.randint(0, 4)
                tabu[(item2, tb)] = iteration + tabu_tenure + random.randint(0, 4)
                overflow += best_move_delta

            if overflow < best_overflow:
                best_overflow = overflow
                best_config_bins = [set(b) for b in current_bins]
                best_config_loads = list(current_loads)
                best_config_item_bin = list(item_bin)
                no_improve_count = 0

                if overflow == 0:
                    break
            else:
                no_improve_count += 1

            # Dynamic tabu tenure
            if no_improve_count > 0 and no_improve_count % 500 == 0:
                tabu_tenure = base_tenure + random.randint(0, base_tenure)

            if no_improve_count >= max_iter_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Restart from best config with perturbation
                current_bins = [set(b) for b in best_config_bins]
                current_loads = list(best_config_loads)
                item_bin = list(best_config_item_bin)

                # Stronger perturbation with each restart
                perturb_count = max(2, n // 8) * min(restarts, 3)
                for _ in range(perturb_count):
                    b1 = random.randint(0, num_bins - 1)
                    if current_bins[b1]:
                        item = random.choice(list(current_bins[b1]))
                        b2 = random.randint(0, num_bins - 1)
                        if b2 != b1:
                            current_bins[b1].discard(item)
                            current_bins[b2].add(item)
                            current_loads[b1] -= w_arr[item]
                            current_loads[b2] += w_arr[item]
                            item_bin[item] = b2

                overflow = sum(max(0, current_loads[b] - bin_capacity) for b in range(num_bins))
                tabu.clear()
                no_improve_count = 0
                tabu_tenure = base_tenure + random.randint(0, base_tenure // 2)

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
            # Try a different initial configuration before giving up
            # Use a different merging strategy: merge two lightest bins
            break

    # Remove empty bins
    final_packing = [b for b in best_packing if len(b) > 0]
    final_weights = [sum(weights[i] for i in b) for b in final_packing]

    return {"packing": final_packing, "bin_weights": final_weights}
