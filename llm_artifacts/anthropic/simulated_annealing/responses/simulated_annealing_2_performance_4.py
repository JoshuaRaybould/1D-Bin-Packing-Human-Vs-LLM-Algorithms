def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import random
    import time as time_module
    import math

    start_time = time_module.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}
    if n == 1:
        return {"packing": [[0]], "bin_weights": [weights[0]]}

    C = bin_capacity
    W = weights
    exp = math.exp
    randint = random.randint
    rand = random.random
    time_time = time_module.time

    # Lower bound
    total_weight = sum(W)
    lb = (total_weight + C - 1) // C  # ceiling division

    # Sorted indices by weight descending
    indices_sorted = sorted(range(n), key=lambda i: W[i], reverse=True)

    def build_ffd():
        bins_list = []
        loads = []
        ibin = [0] * n
        for idx in indices_sorted:
            w = W[idx]
            placed = False
            for b in range(len(bins_list)):
                if loads[b] + w <= C:
                    bins_list[b].append(idx)
                    loads[b] += w
                    ibin[idx] = b
                    placed = True
                    break
            if not placed:
                ibin[idx] = len(bins_list)
                bins_list.append([idx])
                loads.append(w)
        return bins_list, loads, ibin

    def build_bfd():
        bins_list = []
        loads = []
        ibin = [0] * n
        for idx in indices_sorted:
            w = W[idx]
            best_b = -1
            best_remaining = C + 1
            for b in range(len(bins_list)):
                remaining = C - loads[b] - w
                if 0 <= remaining < best_remaining:
                    best_remaining = remaining
                    best_b = b
            if best_b >= 0:
                bins_list[best_b].append(idx)
                loads[best_b] += w
                ibin[idx] = best_b
            else:
                ibin[idx] = len(bins_list)
                bins_list.append([idx])
                loads.append(w)
        return bins_list, loads, ibin

    ffd_bins, ffd_loads, ffd_ibin = build_ffd()
    bfd_bins, bfd_loads, bfd_ibin = build_bfd()

    C2 = C * C
    BIG_M = (n + 1) * C2  # large enough to always prefer fewer bins

    def solution_cost(loads_list, nb):
        ss = 0
        for l in loads_list:
            ss += l * l
        return nb * BIG_M - ss

    # Pick best initial
    ffd_cost = solution_cost(ffd_loads, len(ffd_bins))
    bfd_cost = solution_cost(bfd_loads, len(bfd_bins))

    if len(bfd_bins) < len(ffd_bins) or (len(bfd_bins) == len(ffd_bins) and bfd_cost < ffd_cost):
        init_bins, init_loads, init_ibin = bfd_bins, bfd_loads, bfd_ibin
    else:
        init_bins, init_loads, init_ibin = ffd_bins, ffd_loads, ffd_ibin

    # --- Setup data structures ---
    num_bins = len(init_bins)
    max_bin_id = num_bins
    item_bin = init_ibin[:]
    bin_items = [[] for _ in range(max_bin_id)]
    bin_load = [0] * max_bin_id
    item_pos = [0] * n

    for b in range(num_bins):
        bin_items[b] = init_bins[b][:]
        bin_load[b] = init_loads[b]
        for pos, item in enumerate(bin_items[b]):
            item_pos[item] = pos

    active_list = list(range(num_bins))
    active_pos = {}
    for i, b in enumerate(active_list):
        active_pos[b] = i
    num_active = num_bins
    free_bins = []

    def activate_bin(b):
        nonlocal num_active
        if b not in active_pos:
            active_pos[b] = num_active
            if num_active < len(active_list):
                active_list[num_active] = b
            else:
                active_list.append(b)
            num_active += 1

    def deactivate_bin(b):
        nonlocal num_active
        if b in active_pos:
            pos = active_pos[b]
            num_active -= 1
            last_b = active_list[num_active]
            active_list[pos] = last_b
            active_pos[last_b] = pos
            del active_pos[b]

    def compute_cost():
        ss = 0
        for i in range(num_active):
            b = active_list[i]
            ss += bin_load[b] * bin_load[b]
        return num_active * BIG_M - ss

    current_cost = compute_cost()

    best_item_bin = item_bin[:]
    best_num_bins = num_bins
    best_cost = current_cost

    def remove_item(item):
        nonlocal num_bins
        b = item_bin[item]
        pos = item_pos[item]
        last_item = bin_items[b][-1]
        bin_items[b][pos] = last_item
        item_pos[last_item] = pos
        bin_items[b].pop()
        bin_load[b] -= W[item]
        if len(bin_items[b]) == 0:
            num_bins -= 1
            free_bins.append(b)
            deactivate_bin(b)
        return b

    def add_item(item, b):
        nonlocal num_bins, max_bin_id
        was_empty = len(bin_items[b]) == 0
        if was_empty:
            num_bins += 1
            activate_bin(b)
        item_pos[item] = len(bin_items[b])
        bin_items[b].append(item)
        bin_load[b] += W[item]
        item_bin[item] = b

    def get_new_bin():
        nonlocal max_bin_id
        if free_bins:
            return free_bins.pop()
        b = max_bin_id
        max_bin_id += 1
        bin_items.append([])
        bin_load.append(0)
        return b

    def save_best():
        nonlocal best_item_bin, best_num_bins, best_cost
        best_item_bin = item_bin[:]
        best_num_bins = num_bins
        best_cost = current_cost

    def restore_best():
        nonlocal num_bins, current_cost, max_bin_id, num_active
        nonlocal item_bin, bin_items, bin_load, item_pos, free_bins, active_list, active_pos
        item_bin = best_item_bin[:]
        max_b = max(item_bin) if n > 0 else -1
        max_bin_id = max_b + 1
        bin_items_new = [[] for _ in range(max_bin_id)]
        bin_load_new = [0] * max_bin_id
        for i in range(n):
            b = item_bin[i]
            item_pos[i] = len(bin_items_new[b])
            bin_items_new[b].append(i)
            bin_load_new[b] += W[i]
        bin_items = bin_items_new
        bin_load = bin_load_new
        free_bins = []
        active_list = []
        active_pos = {}
        num_active = 0
        for b in range(max_bin_id):
            if len(bin_items[b]) > 0:
                active_pos[b] = num_active
                active_list.append(b)
                num_active += 1
            else:
                free_bins.append(b)
        num_bins = best_num_bins
        current_cost = best_cost

    def try_compact():
        nonlocal num_bins, current_cost
        if num_active <= 1:
            return
        sorted_active = sorted(active_list[:num_active], key=lambda b: bin_load[b])
        for src in sorted_active[:min(8, len(sorted_active))]:
            if len(bin_items[src]) == 0:
                continue
            items_to_move = bin_items[src][:]
            items_to_move.sort(key=lambda i: W[i], reverse=True)
            assignments = []
            temp_extra = {}
            can_empty = True
            for item in items_to_move:
                w = W[item]
                best_dst = -1
                best_remaining = C + 1
                for j in range(num_active):
                    b2 = active_list[j]
                    if b2 == src:
                        continue
                    load = bin_load[b2] + temp_extra.get(b2, 0)
                    rem = C - load - w
                    if 0 <= rem < best_remaining:
                        best_remaining = rem
                        best_dst = b2
                        if rem == 0:
                            break
                if best_dst == -1:
                    can_empty = False
                    break
                assignments.append((item, best_dst))
                temp_extra[best_dst] = temp_extra.get(best_dst, 0) + w
            if can_empty:
                for item, dst in assignments:
                    remove_item(item)
                    add_item(item, dst)
                current_cost = compute_cost()
                if num_bins < best_num_bins or (num_bins == best_num_bins and current_cost < best_cost):
                    save_best()

    try_compact()

    if best_num_bins <= lb:
        # Already optimal
        bin_map = {}
        packing = []
        bin_weights_out = []
        for i in range(n):
            b = best_item_bin[i]
            if b not in bin_map:
                bin_map[b] = len(packing)
                packing.append([])
                bin_weights_out.append(0)
            idx = bin_map[b]
            packing[idx].append(i)
            bin_weights_out[idx] += W[i]
        return {"packing": packing, "bin_weights": bin_weights_out}

    # SA
    elapsed = time_time() - start_time
    remaining = time_limit - elapsed
    if remaining > 0.1:
        T_start = max(0.3 * C2, 1.0)
        T_end = 0.0001
        deadline = start_time + time_limit * 0.98
        total_time = deadline - start_time

        iteration = 0
        check_interval = 400
        no_improve_count = 0
        reheat_count = 0
        max_reheats = 15
        reheat_threshold = 100000

        log_ratio = math.log(T_end / T_start)

        now = time_time()
        progress = (now - start_time) / total_time
        if progress > 1.0:
            progress = 1.0
        T = T_start * exp(log_ratio * progress)

        while True:
            iteration += 1

            if iteration % check_interval == 0:
                now = time_time()
                if now >= deadline:
                    break
                progress = (now - start_time) / total_time
                if progress > 1.0:
                    progress = 1.0
                T = T_start * exp(log_ratio * progress)

                if no_improve_count > reheat_threshold and reheat_count < max_reheats:
                    restore_best()
                    T = T_start * (0.6 ** (reheat_count + 1))
                    reheat_count += 1
                    no_improve_count = 0

                if iteration % 15000 == 0:
                    try_compact()
                    if best_num_bins <= lb:
                        break

            if num_bins <= lb:
                current_cost = compute_cost()
                if num_bins < best_num_bins or (num_bins == best_num_bins and current_cost < best_cost):
                    save_best()
                break

            if num_active < 2:
                break

            r = rand()

            if r < 0.55:
                # Targeted relocate from least loaded bin
                # Sample 3-4 bins, pick least loaded
                s1 = active_list[randint(0, num_active - 1)]
                s2 = active_list[randint(0, num_active - 1)]
                s3 = active_list[randint(0, num_active - 1)]
                s4 = active_list[randint(0, num_active - 1)]

                src = s1
                if bin_load[s2] < bin_load[src]:
                    src = s2
                if bin_load[s3] < bin_load[src]:
                    src = s3
                if bin_load[s4] < bin_load[src]:
                    src = s4

                if len(bin_items[src]) == 0:
                    no_improve_count += 1
                    continue

                item = bin_items[src][randint(0, len(bin_items[src]) - 1)]
                w = W[item]

                best_dst = -1
                best_dst_load = -1
                for _ in range(7):
                    db = active_list[randint(0, num_active - 1)]
                    if db != src and bin_load[db] + w <= C and bin_load[db] > best_dst_load:
                        best_dst_load = bin_load[db]
                        best_dst = db

                if best_dst == -1:
                    no_improve_count += 1
                    continue

                dst = best_dst
                old_src_load = bin_load[src]
                old_dst_load = bin_load[dst]
                new_src_load = old_src_load - w
                new_dst_load = old_dst_load + w

                delta = -(new_src_load * new_src_load - old_src_load * old_src_load) - (new_dst_load * new_dst_load - old_dst_load * old_dst_load)

                if new_src_load == 0 and len(bin_items[src]) == 1:
                    delta -= BIG_M

                if delta <= 0 or (T > 0 and rand() < exp(-delta / T)):
                    remove_item(item)
                    add_item(item, dst)
                    current_cost += delta

                    if num_bins < best_num_bins or (num_bins == best_num_bins and current_cost < best_cost):
                        save_best()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

            elif r < 0.75:
                # Swap
                item1 = randint(0, n - 1)
                item2 = randint(0, n - 1)
                if item1 == item2:
                    continue

                b1 = item_bin[item1]
                b2 = item_bin[item2]
                if b1 == b2:
                    continue

                w1 = W[item1]
                w2 = W[item2]
                if w1 == w2:
                    continue

                new_load_b1 = bin_load[b1] - w1 + w2
                new_load_b2 = bin_load[b2] - w2 + w1

                if new_load_b1 > C or new_load_b2 > C:
                    continue

                old_load_b1 = bin_load[b1]
                old_load_b2 = bin_load[b2]

                delta = -(new_load_b1 * new_load_b1 - old_load_b1 * old_load_b1) - (new_load_b2 * new_load_b2 - old_load_b2 * old_load_b2)

                if delta <= 0 or (T > 0 and rand() < exp(-delta / T)):
                    pos1 = item_pos[item1]
                    pos2 = item_pos[item2]
                    bin_items[b1][pos1] = item2
                    bin_items[b2][pos2] = item1
                    item_pos[item2] = pos1
                    item_pos[item1] = pos2
                    item_bin[item1] = b2
                    item_bin[item2] = b1
                    bin_load[b1] = new_load_b1
                    bin_load[b2] = new_load_b2
                    current_cost += delta

                    if num_bins < best_num_bins or (num_bins == best_num_bins and current_cost < best_cost):
                        save_best()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

            else:
                # Bin emptying attempt
                min_load = C + 1
                min_bin = -1
                for _ in range(min(10, num_active)):
                    b = active_list[randint(0, num_active - 1)]
                    if bin_load[b] < min_load:
                        min_load = bin_load[b]
                        min_bin = b

                if min_bin == -1 or len(bin_items[min_bin]) == 0:
                    no_improve_count += 1
                    continue

                src = min_bin
                items_in_src = bin_items[src][:]
                items_in_src.sort(key=lambda i: W[i], reverse=True)

                assignments = []
                temp_extra = {}
                can_empty = True
                for item in items_in_src:
                    w = W[item]
                    best_dst = -1
                    best_remaining = C + 1
                    for j in range(num_active):
                        b2 = active_list[j]
                        if b2 == src:
                            continue
                        load = bin_load[b2] + temp_extra.get(b2, 0)
                        rem = C - load - w
                        if 0 <= rem < best_remaining:
                            best_remaining = rem
                            best_dst = b2
                            if rem == 0:
                                break
                    if best_dst == -1:
                        can_empty = False
                        break
                    assignments.append((item, best_dst))
                    temp_extra[best_dst] = temp_extra.get(best_dst, 0) + w

                if can_empty and len(assignments) > 0:
                    for item, dst in assignments:
                        remove_item(item)
                        add_item(item, dst)
                    current_cost = compute_cost()
                    if num_bins < best_num_bins or (num_bins == best_num_bins and current_cost < best_cost):
                        save_best()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Single item relocate from lightest bin
                    if len(bin_items[src]) > 0:
                        item = bin_items[src][randint(0, len(bin_items[src]) - 1)]
                        w = W[item]
                        best_dst = -1
                        best_dst_load = -1
                        for _ in range(min(12, num_active)):
                            b2 = active_list[randint(0, num_active - 1)]
                            if b2 != src and bin_load[b2] + w <= C and bin_load[b2] > best_dst_load:
                                best_dst_load = bin_load[b2]
                                best_dst = b2

                        if best_dst != -1:
                            dst = best_dst
                            old_src_load = bin_load[src]
                            old_dst_load = bin_load[dst]
                            new_src_load = old_src_load - w
                            new_dst_load = old_dst_load + w

                            delta = -(new_src_load * new_src_load - old_src_load * old_src_load) - (new_dst_load * new_dst_load - old_dst_load * old_dst_load)

                            if new_src_load == 0 and len(bin_items[src]) == 1:
                                delta -= BIG_M

                            if delta <= 0 or (T > 0 and rand() < exp(-delta / T)):
                                remove_item(item)
                                add_item(item, dst)
                                current_cost += delta

                                if num_bins < best_num_bins or (num_bins == best_num_bins and current_cost < best_cost):
                                    save_best()
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                            else:
                                no_improve_count += 1
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1

    # Final compaction
    restore_best()
    try_compact()

    # Reconstruct
    bin_map = {}
    packing = []
    bin_weights_out = []
    for i in range(n):
        b = best_item_bin[i]
        if b not in bin_map:
            bin_map[b] = len(packing)
            packing.append([])
            bin_weights_out.append(0)
        idx = bin_map[b]
        packing[idx].append(i)
        bin_weights_out[idx] += W[i]

    return {"packing": packing, "bin_weights": bin_weights_out}