def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    import random

    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Initial solution: First Fit Decreasing
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    bins = []  # list of sets of item indices
    bin_loads = []  # current load of each bin

    for idx in sorted_indices:
        w = weights[idx]
        placed = False
        for b in range(len(bins)):
            if bin_loads[b] + w <= bin_capacity:
                bins[b].add(idx)
                bin_loads[b] += w
                placed = True
                break
        if not placed:
            bins.append({idx})
            bin_loads.append(w)

    best_packing = [sorted(b) for b in bins]
    best_bin_weights = list(bin_loads)
    best_num_bins = len(bins)

    # Try to reduce number of bins using tabu search
    target_bins = best_num_bins - 1

    while target_bins >= 1 and time.time() - start_time < time_limit * 0.90:
        # Initialize packing with target_bins bins
        # Start from best packing, merge lightest bins
        current_bins = [set(b) for b in best_packing]
        current_loads = list(best_bin_weights)

        while len(current_bins) > target_bins:
            min_idx = min(range(len(current_bins)), key=lambda i: current_loads[i])
            items_to_redistribute = list(current_bins[min_idx])
            del current_bins[min_idx]
            del current_loads[min_idx]

            # Best fit decreasing for redistributed items
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
                    # No feasible bin, put in bin with most space (least load)
                    best_b = min(range(len(current_bins)), key=lambda b: current_loads[b])
                current_bins[best_b].add(item)
                current_loads[best_b] += w

        item_bin = [0] * n
        for b in range(len(current_bins)):
            for item in current_bins[b]:
                item_bin[item] = b

        num_bins = len(current_bins)

        def compute_overflow():
            return sum(max(0, current_loads[b] - bin_capacity) for b in range(num_bins))

        overflow = compute_overflow()

        if overflow == 0:
            best_packing = [sorted(b) for b in current_bins]
            best_bin_weights = list(current_loads)
            best_num_bins = target_bins
            target_bins -= 1
            continue

        # Tabu search to minimize total overflow
        tabu = {}  # (item, bin) -> iteration when tabu expires
        tabu_tenure = max(7, int(n ** 0.5))

        best_overflow = overflow
        best_config_bins = [set(b) for b in current_bins]
        best_config_loads = list(current_loads)
        best_config_item_bin = list(item_bin)

        max_iter_no_improve = max(3000, n * 20)
        no_improve_count = 0
        iteration = 0
        restarts = 0
        max_restarts = 5

        while time.time() - start_time < time_limit * 0.90:
            iteration += 1

            if iteration % 200 == 0:
                if time.time() - start_time >= time_limit * 0.90:
                    break

            # Find overflowing bins
            overflow_bins = [b for b in range(num_bins) if current_loads[b] > bin_capacity]

            if not overflow_bins:
                break

            # Generate candidate moves
            best_move = None
            best_move_delta = float('inf')
            best_move_type = None  # 'transfer' or 'swap'
            tie_candidates = []

            # Limit neighborhood exploration for performance
            # Sample overflow bins if too many
            sampled_overflow = overflow_bins if len(overflow_bins) <= 3 else random.sample(overflow_bins, 3)

            for ob in sampled_overflow:
                ob_items = list(current_bins[ob])
                # Sample items if too many
                if len(ob_items) > 15:
                    sampled_items = random.sample(ob_items, 15)
                else:
                    sampled_items = ob_items

                for item in sampled_items:
                    w = weights[item]
                    old_ov_ob = max(0, current_loads[ob] - bin_capacity)
                    new_ov_ob = max(0, current_loads[ob] - w - bin_capacity)

                    for tb in range(num_bins):
                        if tb == ob:
                            continue

                        # Transfer move
                        old_ov_tb = max(0, current_loads[tb] - bin_capacity)
                        new_ov_tb = max(0, current_loads[tb] + w - bin_capacity)
                        delta = (new_ov_ob - old_ov_ob) + (new_ov_tb - old_ov_tb)

                        is_tabu = tabu.get((item, tb), 0) > iteration
                        aspiration = (overflow + delta < best_overflow) if is_tabu else False

                        if not is_tabu or aspiration:
                            if delta < best_move_delta:
                                best_move_delta = delta
                                best_move = ('transfer', item, ob, tb)
                                tie_candidates = [best_move]
                            elif delta == best_move_delta:
                                tie_candidates.append(('transfer', item, ob, tb))

                    # Swap moves: swap item from ob with an item from a non-overflowing bin
                    # Sample target bins
                    non_overflow = [b for b in range(num_bins) if b != ob and current_loads[b] <= bin_capacity]
                    if len(non_overflow) > 5:
                        sampled_targets = random.sample(non_overflow, 5)
                    else:
                        sampled_targets = non_overflow

                    for tb in sampled_targets:
                        tb_items = list(current_bins[tb])
                        if len(tb_items) > 10:
                            sampled_tb_items = random.sample(tb_items, 10)
                        else:
                            sampled_tb_items = tb_items

                        for item2 in sampled_tb_items:
                            if weights[item2] >= w:
                                continue  # swapping won't help reduce overflow in ob

                            w2 = weights[item2]
                            # After swap: ob loses w, gains w2; tb gains w, loses w2
                            new_load_ob = current_loads[ob] - w + w2
                            new_load_tb = current_loads[tb] + w - w2

                            old_ov_ob2 = max(0, current_loads[ob] - bin_capacity)
                            new_ov_ob2 = max(0, new_load_ob - bin_capacity)
                            old_ov_tb2 = max(0, current_loads[tb] - bin_capacity)
                            new_ov_tb2 = max(0, new_load_tb - bin_capacity)

                            delta = (new_ov_ob2 - old_ov_ob2) + (new_ov_tb2 - old_ov_tb2)

                            is_tabu1 = tabu.get((item, tb), 0) > iteration
                            is_tabu2 = tabu.get((item2, ob), 0) > iteration
                            is_tabu_move = is_tabu1 or is_tabu2
                            aspiration = (overflow + delta < best_overflow) if is_tabu_move else False

                            if not is_tabu_move or aspiration:
                                if delta < best_move_delta:
                                    best_move_delta = delta
                                    best_move = ('swap', item, ob, item2, tb)
                                    tie_candidates = [best_move]
                                elif delta == best_move_delta:
                                    tie_candidates.append(('swap', item, ob, item2, tb))

            if not tie_candidates:
                # All moves tabu, force a random move
                ob = random.choice(overflow_bins)
                if current_bins[ob]:
                    item = random.choice(list(current_bins[ob]))
                    tb = random.choice([b for b in range(num_bins) if b != ob])
                    w = weights[item]
                    old_ov_ob = max(0, current_loads[ob] - bin_capacity)
                    new_ov_ob = max(0, current_loads[ob] - w - bin_capacity)
                    old_ov_tb = max(0, current_loads[tb] - bin_capacity)
                    new_ov_tb = max(0, current_loads[tb] + w - bin_capacity)
                    best_move_delta = (new_ov_ob - old_ov_ob) + (new_ov_tb - old_ov_tb)
                    best_move = ('transfer', item, ob, tb)
                else:
                    no_improve_count += 1
                    if no_improve_count >= max_iter_no_improve:
                        break
                    continue

            if best_move is None:
                break

            # Pick from ties
            if len(tie_candidates) > 1:
                chosen = random.choice(tie_candidates)
            elif tie_candidates:
                chosen = tie_candidates[0]
            else:
                chosen = best_move

            # Apply move
            if chosen[0] == 'transfer':
                _, item, ob, tb = chosen
                w = weights[item]
                current_bins[ob].discard(item)
                current_bins[tb].add(item)
                current_loads[ob] -= w
                current_loads[tb] += w
                item_bin[item] = tb
                tabu[(item, ob)] = iteration + tabu_tenure + random.randint(0, 4)
                overflow += best_move_delta
            else:  # swap
                _, item1, ob, item2, tb = chosen
                w1 = weights[item1]
                w2 = weights[item2]
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

            if no_improve_count >= max_iter_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Restart from best config with perturbation
                current_bins = [set(b) for b in best_config_bins]
                current_loads = list(best_config_loads)
                item_bin = list(best_config_item_bin)

                # Perturb
                perturb_count = max(1, n // 10)
                for _ in range(perturb_count):
                    b1 = random.randint(0, num_bins - 1)
                    if current_bins[b1]:
                        item = random.choice(list(current_bins[b1]))
                        b2 = random.randint(0, num_bins - 1)
                        if b2 != b1:
                            current_bins[b1].discard(item)
                            current_bins[b2].add(item)
                            current_loads[b1] -= weights[item]
                            current_loads[b2] += weights[item]
                            item_bin[item] = b2

                overflow = compute_overflow()
                tabu.clear()
                no_improve_count = 0
                # Adjust tabu tenure
                tabu_tenure = max(7, int(n ** 0.5)) + random.randint(0, 5)

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
    final_weights = [sum(weights[i] for i in b) for b in final_packing]

    return {"packing": final_packing, "bin_weights": final_weights}