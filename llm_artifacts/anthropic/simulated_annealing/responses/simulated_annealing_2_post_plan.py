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

    # --- FFD initial solution ---
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

    # --- BFD initial solution ---
    def build_bfd():
        bins_list = []
        loads = []
        ibin = [0] * n
        for idx in indices_sorted:
            w = W[idx]
            best_b = -1
            best_remaining = C + 1  # want minimum remaining = tightest fit
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

    def solution_cost(loads, num_bins):
        ss = sum(l * l for l in loads)
        return num_bins * C2 - ss

    ffd_cost = solution_cost(ffd_loads, len(ffd_bins))
    bfd_cost = solution_cost(bfd_loads, len(bfd_bins))

    if len(bfd_bins) < len(ffd_bins) or (len(bfd_bins) == len(ffd_bins) and bfd_cost < ffd_cost):
        init_bins, init_loads, init_ibin = bfd_bins, bfd_loads, bfd_ibin
    else:
        init_bins, init_loads, init_ibin = ffd_bins, ffd_loads, ffd_ibin

    # --- Setup O(1) data structures ---
    num_bins = len(init_bins)
    item_bin = init_ibin[:]
    # bin_items: list of lists, bin_loads: list of ints
    # item_pos[i]: position of item i in bin_items[item_bin[i]]
    bin_items = [[] for _ in range(num_bins)]
    bin_loads = [0] * num_bins
    item_pos = [0] * n

    for b in range(num_bins):
        bin_items[b] = init_bins[b][:]
        bin_loads[b] = init_loads[b]
        for pos, item in enumerate(bin_items[b]):
            item_pos[item] = pos

    # Free bin slots (reuse removed bin indices)
    free_bins = []

    def compute_cost():
        nb = num_bins
        ss = 0
        for b in range(len(bin_items)):
            if len(bin_items[b]) > 0:
                ss += bin_loads[b] * bin_loads[b]
        return nb * C2 - ss

    current_cost = compute_cost()

    # Best solution tracking: just copy item_bin
    best_item_bin = item_bin[:]
    best_num_bins = num_bins
    best_cost = current_cost

    # Helper: remove item from its bin (O(1))
    def remove_item(item):
        nonlocal num_bins
        b = item_bin[item]
        pos = item_pos[item]
        last_item = bin_items[b][-1]
        # swap with last
        bin_items[b][pos] = last_item
        item_pos[last_item] = pos
        bin_items[b].pop()
        bin_loads[b] -= W[item]
        # Check if bin is now empty
        if len(bin_items[b]) == 0:
            num_bins -= 1
            free_bins.append(b)
        return b

    def add_item(item, b):
        nonlocal num_bins
        was_empty = len(bin_items[b]) == 0
        if was_empty:
            num_bins += 1
            # remove from free_bins if present (shouldn't normally add to empty unless intended)
        item_pos[item] = len(bin_items[b])
        bin_items[b].append(item)
        bin_loads[b] += W[item]
        item_bin[item] = b

    # SA parameters
    elapsed = time_module.time() - start_time
    remaining = time_limit - elapsed
    if remaining <= 0.01:
        # reconstruct
        return _reconstruct(best_item_bin, n, W)

    T_start = max(0.3 * C2, 1.0)
    T_end = 0.01
    deadline = start_time + time_limit * 0.98

    total_time = deadline - start_time
    T = T_start

    iteration = 0
    check_interval = 200
    last_check_time = time_module.time()
    last_check_iter = 0
    last_T = T
    next_T = T  # T at next check

    no_improve_count = 0
    reheat_count = 0
    max_reheats = 5
    reheat_threshold = 50000

    # Active bins list (for sampling)
    # We'll just work with indices into bin_items, some may be empty
    total_bin_slots = len(bin_items)

    def get_active_bins():
        return [b for b in range(total_bin_slots) if len(bin_items[b]) > 0]

    def save_best():
        nonlocal best_item_bin, best_num_bins, best_cost
        best_item_bin = item_bin[:]
        best_num_bins = num_bins
        best_cost = current_cost

    def restore_best():
        nonlocal num_bins, current_cost, total_bin_slots
        # Rebuild from best_item_bin
        nonlocal item_bin, bin_items, bin_loads, item_pos, free_bins
        item_bin = best_item_bin[:]
        # Find max bin index
        max_b = max(item_bin) if n > 0 else -1
        total_bin_slots = max_b + 1
        bin_items_new = [[] for _ in range(total_bin_slots)]
        bin_loads_new = [0] * total_bin_slots
        for i in range(n):
            b = item_bin[i]
            item_pos[i] = len(bin_items_new[b])
            bin_items_new[b].append(i)
            bin_loads_new[b] += W[i]
        bin_items = bin_items_new
        bin_loads = bin_loads_new
        free_bins = [b for b in range(total_bin_slots) if len(bin_items[b]) == 0]
        num_bins = best_num_bins
        current_cost = best_cost

    # Compaction: try to empty least loaded bins
    def try_compact():
        nonlocal num_bins, current_cost
        active = get_active_bins()
        if len(active) <= 1:
            return
        # Sort by load ascending
        active.sort(key=lambda b: bin_loads[b])
        for src in active:
            if len(bin_items[src]) == 0:
                continue
            if num_bins <= best_num_bins - 1:
                break  # already improved enough
            # Try to redistribute all items from src to other bins
            items_to_move = bin_items[src][:]
            # Sort items by weight descending for better packing
            items_to_move.sort(key=lambda i: W[i], reverse=True)
            assignments = []
            temp_loads = {}
            can_empty = True
            for item in items_to_move:
                w = W[item]
                best_dst = -1
                best_remaining = C + 1
                for b2 in range(total_bin_slots):
                    if b2 == src:
                        continue
                    if len(bin_items[b2]) == 0 and b2 not in temp_loads:
                        continue
                    load = temp_loads.get(b2, bin_loads[b2])
                    rem = C - load - w
                    if 0 <= rem < best_remaining:
                        best_remaining = rem
                        best_dst = b2
                if best_dst == -1:
                    can_empty = False
                    break
                assignments.append((item, best_dst))
                temp_loads[best_dst] = temp_loads.get(best_dst, bin_loads[best_dst]) + w
            if can_empty:
                # Execute all moves
                for item, dst in assignments:
                    old_src_load = bin_loads[item_bin[item]]
                    remove_item(item)
                    add_item(item, dst)
                # Recompute cost
                current_cost = compute_cost()
                if num_bins < best_num_bins or (num_bins == best_num_bins and current_cost < best_cost):
                    save_best()

    while True:
        iteration += 1

        if iteration % check_interval == 0:
            now = time_module.time()
            if now >= deadline:
                break
            progress = (now - start_time) / total_time
            if progress > 1.0:
                progress = 1.0
            T = T_start * ((T_end / T_start) ** progress)
            last_check_time = now
            last_check_iter = iteration
            last_T = T

            # Reheat check
            if no_improve_count > reheat_threshold and reheat_count < max_reheats:
                restore_best()
                total_bin_slots = len(bin_items)
                T = T_start * 0.3
                reheat_count += 1
                no_improve_count = 0

            # Periodic compaction
            if iteration % 10000 == 0:
                try_compact()
                total_bin_slots = len(bin_items)
        else:
            # Interpolate T between checks
            frac = (iteration - last_check_iter) / check_interval
            now_progress = ((last_check_time - start_time) + frac * (check_interval * 0.00001)) / total_time
            if now_progress > 1.0:
                now_progress = 1.0
            T = T_start * ((T_end / T_start) ** now_progress)

        if num_bins <= 1:
            # Can't improve further
            break

        r = rand()

        if r < 0.50:
            # Move 1: Targeted Relocate - pick source biased toward least loaded
            # Sample 3 random active bins, pick least loaded as source
            active = None
            # Quick way: pick random items to find bins
            item1 = randint(0, n - 1)
            b1 = item_bin[item1]
            item2 = randint(0, n - 1)
            b2 = item_bin[item2]
            item3 = randint(0, n - 1)
            b3 = item_bin[item3]

            # Pick source = least loaded among b1, b2, b3
            candidates = [b1, b2, b3]
            src = min(candidates, key=lambda b: bin_loads[b])

            if len(bin_items[src]) == 0:
                no_improve_count += 1
                continue

            # Pick random item from src
            item = bin_items[src][randint(0, len(bin_items[src]) - 1)]
            w = W[item]

            # Pick destination: sample 3 random bins, pick tightest fit
            d1 = item_bin[randint(0, n - 1)]
            d2 = item_bin[randint(0, n - 1)]
            d3 = item_bin[randint(0, n - 1)]
            dst_candidates = [d1, d2, d3]

            best_dst = -1
            best_dst_load = -1
            for db in dst_candidates:
                if db == src:
                    continue
                if len(bin_items[db]) == 0:
                    continue
                if bin_loads[db] + w <= C and bin_loads[db] > best_dst_load:
                    best_dst_load = bin_loads[db]
                    best_dst = db

            if best_dst == -1:
                no_improve_count += 1
                continue

            dst = best_dst
            old_src_load = bin_loads[src]
            old_dst_load = bin_loads[dst]
            new_src_load = old_src_load - w
            new_dst_load = old_dst_load + w

            delta = -(new_src_load * new_src_load - old_src_load * old_src_load) - (new_dst_load * new_dst_load - old_dst_load * old_dst_load)

            bin_removed = False
            if new_src_load == 0 and len(bin_items[src]) == 1:
                delta -= C2
                bin_removed = True

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

        elif r < 0.65:
            # Move 2: Random Swap
            item1 = randint(0, n - 1)
            item2 = randint(0, n - 1)
            if item1 == item2:
                no_improve_count += 1
                continue

            b1 = item_bin[item1]
            b2 = item_bin[item2]
            if b1 == b2:
                no_improve_count += 1
                continue

            w1 = W[item1]
            w2 = W[item2]
            if w1 == w2:
                no_improve_count += 1
                continue

            new_load_b1 = bin_loads[b1] - w1 + w2
            new_load_b2 = bin_loads[b2] - w2 + w1

            if new_load_b1 > C or new_load_b2 > C:
                no_improve_count += 1
                continue

            old_load_b1 = bin_loads[b1]
            old_load_b2 = bin_loads[b2]

            delta = -(new_load_b1 * new_load_b1 - old_load_b1 * old_load_b1) - (new_load_b2 * new_load_b2 - old_load_b2 * old_load_b2)

            if delta <= 0 or (T > 0 and rand() < exp(-delta / T)):
                # Perform swap using O(1) operations
                pos1 = item_pos[item1]
                pos2 = item_pos[item2]
                bin_items[b1][pos1] = item2
                bin_items[b2][pos2] = item1
                item_pos[item2] = pos1
                item_pos[item1] = pos2
                item_bin[item1] = b2
                item_bin[item2] = b1
                bin_loads[b1] = new_load_b1
                bin_loads[b2] = new_load_b2
                current_cost += delta

                if num_bins < best_num_bins or (num_bins == best_num_bins and current_cost < best_cost):
                    save_best()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

        else:
            # Move 3: Bin Emptying Attempt
            # Find the least loaded bin
            min_load = C + 1
            min_bin = -1
            # Sample a few bins and pick the least loaded
            for _ in range(min(5, num_bins)):
                ri = randint(0, n - 1)
                b = item_bin[ri]
                if len(bin_items[b]) > 0 and bin_loads[b] < min_load:
                    min_load = bin_loads[b]
                    min_bin = b

            if min_bin == -1 or len(bin_items[min_bin]) == 0:
                no_improve_count += 1
                continue

            src = min_bin
            items_in_src = bin_items[src][:]
            # Sort by weight descending
            items_in_src.sort(key=lambda i: W[i], reverse=True)

            # Try to place all items from src into other bins
            assignments = []
            temp_extra = {}  # bin -> extra load added
            can_empty = True
            for item in items_in_src:
                w = W[item]
                best_dst = -1
                best_remaining = C + 1
                # Scan all active bins (or sample)
                # For efficiency, sample some bins
                checked = set()
                checked.add(src)
                found = False
                # Try up to 20 random bins
                for _ in range(min(30, num_bins)):
                    ri2 = randint(0, n - 1)
                    b2 = item_bin[ri2]
                    if b2 in checked:
                        continue
                    checked.add(b2)
                    if len(bin_items[b2]) == 0:
                        continue
                    load = bin_loads[b2] + temp_extra.get(b2, 0)
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
                # Execute all moves (always accept - reduces bin count)
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
                # Try moving a single item from the least loaded bin
                if len(bin_items[src]) > 0:
                    item = bin_items[src][randint(0, len(bin_items[src]) - 1)]
                    w = W[item]
                    # Find best destination
                    best_dst = -1
                    best_dst_load = -1
                    for _ in range(min(20, num_bins)):
                        ri2 = randint(0, n - 1)
                        b2 = item_bin[ri2]
                        if b2 == src or len(bin_items[b2]) == 0:
                            continue
                        if bin_loads[b2] + w <= C and bin_loads[b2] > best_dst_load:
                            best_dst_load = bin_loads[b2]
                            best_dst = b2

                    if best_dst != -1:
                        dst = best_dst
                        old_src_load = bin_loads[src]
                        old_dst_load = bin_loads[dst]
                        new_src_load = old_src_load - w
                        new_dst_load = old_dst_load + w

                        delta = -(new_src_load * new_src_load - old_src_load * old_src_load) - (new_dst_load * new_dst_load - old_dst_load * old_dst_load)

                        bin_removed_flag = False
                        if new_src_load == 0 and len(bin_items[src]) == 1:
                            delta -= C2
                            bin_removed_flag = True

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

    # Reconstruct from best_item_bin
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
