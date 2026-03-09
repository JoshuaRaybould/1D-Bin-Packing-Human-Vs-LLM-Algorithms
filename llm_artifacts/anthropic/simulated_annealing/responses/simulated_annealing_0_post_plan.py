def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import random
    import time
    import math

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start_time = time.time()
    deadline = start_time + time_limit * 0.95

    w = weights
    C = bin_capacity
    C2 = C * C

    # Sort indices by weight descending for heuristics
    indices_sorted = sorted(range(n), key=lambda i: w[i], reverse=True)

    # --- FFD ---
    def run_ffd():
        bins_items = []
        bins_loads = []
        ib = [0] * n
        for idx in indices_sorted:
            wi = w[idx]
            placed = False
            for b in range(len(bins_items)):
                if bins_loads[b] + wi <= C:
                    bins_items[b].append(idx)
                    bins_loads[b] += wi
                    ib[idx] = b
                    placed = True
                    break
            if not placed:
                ib[idx] = len(bins_items)
                bins_items.append([idx])
                bins_loads.append(wi)
        return bins_items, bins_loads, ib

    # --- BFD ---
    def run_bfd():
        bins_items = []
        bins_loads = []
        ib = [0] * n
        for idx in indices_sorted:
            wi = w[idx]
            best_b = -1
            best_remaining = C + 1
            for b in range(len(bins_items)):
                remaining = C - bins_loads[b]
                if wi <= remaining and remaining < best_remaining:
                    best_remaining = remaining
                    best_b = b
            if best_b >= 0:
                bins_items[best_b].append(idx)
                bins_loads[best_b] += wi
                ib[idx] = best_b
            else:
                ib[idx] = len(bins_items)
                bins_items.append([idx])
                bins_loads.append(wi)
        return bins_items, bins_loads, ib

    ffd_items, ffd_loads, ffd_ib = run_ffd()
    bfd_items, bfd_loads, bfd_ib = run_bfd()

    def score_of(loads):
        s = 0
        for ld in loads:
            s += ld * ld
        return -len(loads) * C2 * (n + 1) + s

    ffd_score = score_of(ffd_loads)
    bfd_score = score_of(bfd_loads)

    if len(bfd_items) < len(ffd_items) or (len(bfd_items) == len(ffd_items) and bfd_score > ffd_score):
        bin_items = bfd_items
        bin_loads = bfd_loads
        item_bin = bfd_ib
    else:
        bin_items = ffd_items
        bin_loads = ffd_loads
        item_bin = ffd_ib

    # Build item_pos_in_bin
    item_pos = [0] * n
    for b in range(len(bin_items)):
        for pos, itm in enumerate(bin_items[b]):
            item_pos[itm] = pos

    num_bins = len(bin_items)

    def compute_score():
        s = 0
        for ld in bin_loads:
            s += ld * ld
        return -len(bin_loads) * C2 * (n + 1) + s

    current_score = compute_score()

    # Best solution: store compact representation
    best_num_bins = num_bins
    best_score = current_score
    best_item_bin = item_bin[:]
    best_bin_loads = bin_loads[:]

    # Helper: remove item from its bin using swap-with-last
    def remove_item_from_bin(item, b):
        pos = item_pos[item]
        last_idx = len(bin_items[b]) - 1
        if pos != last_idx:
            other = bin_items[b][last_idx]
            bin_items[b][pos] = other
            item_pos[other] = pos
        bin_items[b].pop()
        bin_loads[b] -= w[item]

    def add_item_to_bin(item, b):
        item_pos[item] = len(bin_items[b])
        bin_items[b].append(item)
        bin_loads[b] += w[item]
        item_bin[item] = b

    def remove_empty_bin(b):
        nonlocal num_bins
        last = len(bin_items) - 1
        if b != last:
            bin_items[b] = bin_items[last]
            bin_loads[b] = bin_loads[last]
            for itm in bin_items[b]:
                item_bin[itm] = b
            # positions within the list are unchanged
        bin_items.pop()
        bin_loads.pop()
        num_bins = len(bin_items)

    # Find least-loaded bin
    def find_min_bin():
        min_b = 0
        min_ld = bin_loads[0]
        for b in range(1, len(bin_items)):
            if bin_loads[b] < min_ld:
                min_ld = bin_loads[b]
                min_b = b
        return min_b

    min_bin = find_min_bin()
    min_bin_load = bin_loads[min_bin]

    def update_min_bin_after_change(changed_bins):
        nonlocal min_bin, min_bin_load
        # If min_bin was removed (num_bins decreased), rescan
        if min_bin >= len(bin_items):
            min_bin = find_min_bin()
            min_bin_load = bin_loads[min_bin]
            return
        # Check if any changed bin is now smaller
        needs_rescan = False
        for b in changed_bins:
            if b >= len(bin_items):
                continue
            if bin_loads[b] < min_bin_load:
                min_bin = b
                min_bin_load = bin_loads[b]
        # If min_bin's load increased, rescan
        if bin_loads[min_bin] > min_bin_load:
            needs_rescan = True
        min_bin_load = bin_loads[min_bin]
        if needs_rescan:
            min_bin = find_min_bin()
            min_bin_load = bin_loads[min_bin]

    if time.time() >= deadline:
        packing = [[] for _ in range(best_num_bins)]
        for i, b in enumerate(best_item_bin):
            packing[b].append(i)
        bw = [sum(w[i] for i in p) for p in packing]
        return {"packing": packing, "bin_weights": bw}

    # Local aliases for speed
    _random = random.random
    _randrange = random.randrange
    _randint = random.randint
    _shuffle = random.shuffle
    _exp = math.exp
    _time = time.time

    # SA parameters - time-based adaptive temperature
    T_max = C2 * 0.5
    T_mid = C2 * 0.05
    T_low = C2 * 0.005
    T_min_val = 0.01
    total_time = deadline - start_time
    if total_time <= 0:
        total_time = 0.001

    T = T_max

    iteration = 0
    check_interval = 2000
    stagnation_count = 0
    last_best_bins = best_num_bins

    while True:
        iteration += 1
        if iteration % check_interval == 0:
            now = _time()
            if now >= deadline:
                break
            # Update temperature based on time fraction
            frac = (now - start_time) / total_time
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

            if frac < 0.3:
                f = frac / 0.3
                T = T_max * (1.0 - f) + T_mid * f
            elif frac < 0.8:
                f = (frac - 0.3) / 0.5
                T = T_mid * (1.0 - f) + T_low * f
            else:
                f = (frac - 0.8) / 0.2
                T = T_low * (1.0 - f) + T_min_val * f

            if T < T_min_val:
                T = T_min_val

            # Check stagnation
            if num_bins == last_best_bins:
                stagnation_count += check_interval
            else:
                stagnation_count = 0
                last_best_bins = num_bins

            # Diversification: perturbation after 100k iterations of stagnation
            if stagnation_count >= 100000 and len(bin_items) > 2:
                stagnation_count = 0
                # Pick 2 least loaded bins, redistribute their items via BFD into others
                if len(bin_items) > 2:
                    # Find 2 smallest bins
                    sorted_bins = sorted(range(len(bin_items)), key=lambda b: bin_loads[b])
                    to_empty = sorted_bins[:2]
                    displaced_items = []
                    # Collect items from bins to empty (process in reverse order to handle removal)
                    for b in sorted(to_empty, reverse=True):
                        for itm in list(bin_items[b]):
                            displaced_items.append(itm)
                        # Clear the bin
                        bin_items[b] = []
                        bin_loads[b] = 0

                    # Remove empty bins
                    # Process from high to low index
                    empties = sorted([b for b in range(len(bin_items)) if len(bin_items[b]) == 0], reverse=True)
                    for b in empties:
                        remove_empty_bin(b)

                    # Redistribute displaced items using BFD
                    displaced_items.sort(key=lambda i: w[i], reverse=True)
                    for itm in displaced_items:
                        wi = w[itm]
                        best_b = -1
                        best_rem = C + 1
                        for b in range(len(bin_items)):
                            rem = C - bin_loads[b]
                            if wi <= rem and rem < best_rem:
                                best_rem = rem
                                best_b = b
                        if best_b >= 0:
                            add_item_to_bin(itm, best_b)
                        else:
                            # Create new bin
                            new_b = len(bin_items)
                            bin_items.append([])
                            bin_loads.append(0)
                            num_bins = len(bin_items)
                            add_item_to_bin(itm, new_b)

                    current_score = compute_score()
                    min_bin = find_min_bin()
                    min_bin_load = bin_loads[min_bin]

                    # Temporarily boost temperature
                    T = T_mid

                    cur_nb = len(bin_items)
                    if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
                        best_num_bins = cur_nb
                        best_score = current_score
                        best_item_bin = item_bin[:]
                        best_bin_loads = bin_loads[:]

        nb = len(bin_items)
        if nb <= 1:
            break

        r = _random()

        if r < 0.50:
            # Move Type 1: Single Item Transfer
            if _random() < 0.7:
                src = min_bin
            else:
                src = _randrange(nb)

            src_len = len(bin_items[src])
            if src_len == 0:
                continue

            item = bin_items[src][_randrange(src_len)]
            wi = w[item]

            # Best-fit destination from random sample
            best_dst = -1
            best_dst_load = -1
            # Sample up to 8 random bins
            for _ in range(8):
                b = _randrange(nb)
                if b == src:
                    continue
                if bin_loads[b] + wi <= C and bin_loads[b] > best_dst_load:
                    best_dst_load = bin_loads[b]
                    best_dst = b

            if best_dst == -1:
                continue

            dst = best_dst
            old_src_load = bin_loads[src]
            old_dst_load = bin_loads[dst]
            new_src_load = old_src_load - wi
            new_dst_load = old_dst_load + wi

            src_becomes_empty = (src_len == 1)

            if src_becomes_empty:
                delta = -C2 * (n + 1) + new_dst_load * new_dst_load - old_dst_load * old_dst_load - old_src_load * old_src_load
            else:
                delta = (new_src_load * new_src_load - old_src_load * old_src_load +
                         new_dst_load * new_dst_load - old_dst_load * old_dst_load)

            if delta >= 0:
                accept = True
            else:
                dt = delta / T
                if dt < -500:
                    accept = False
                else:
                    accept = _random() < _exp(dt)

            if accept:
                remove_item_from_bin(item, src)
                add_item_to_bin(item, dst)
                current_score += delta

                changed = [src, dst]
                if src_becomes_empty:
                    remove_empty_bin(src)
                    changed = []

                if changed:
                    update_min_bin_after_change(changed)
                else:
                    min_bin = find_min_bin()
                    min_bin_load = bin_loads[min_bin] if bin_items else 0

                cur_nb = len(bin_items)
                if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
                    best_num_bins = cur_nb
                    best_score = current_score
                    best_item_bin = item_bin[:]
                    best_bin_loads = bin_loads[:]

        elif r < 0.80:
            # Move Type 2: Pairwise Swap
            accepted = False
            for _ in range(3):
                b1 = _randrange(nb)
                b2 = _randrange(nb)
                if b1 == b2:
                    continue
                if len(bin_items[b1]) == 0 or len(bin_items[b2]) == 0:
                    continue

                item1 = bin_items[b1][_randrange(len(bin_items[b1]))]
                item2 = bin_items[b2][_randrange(len(bin_items[b2]))]
                w1 = w[item1]
                w2 = w[item2]

                if w1 == w2:
                    continue

                new_load_b1 = bin_loads[b1] - w1 + w2
                new_load_b2 = bin_loads[b2] - w2 + w1

                if new_load_b1 > C or new_load_b2 > C:
                    continue
                if new_load_b1 < 0 or new_load_b2 < 0:
                    continue

                delta = (new_load_b1 * new_load_b1 - bin_loads[b1] * bin_loads[b1] +
                         new_load_b2 * new_load_b2 - bin_loads[b2] * bin_loads[b2])

                if delta >= 0:
                    accept = True
                else:
                    dt = delta / T
                    if dt < -500:
                        accept = False
                    else:
                        accept = _random() < _exp(dt)

                if accept:
                    # Remove both, add both
                    # Remove item1 from b1
                    remove_item_from_bin(item1, b1)
                    # Remove item2 from b2
                    remove_item_from_bin(item2, b2)
                    # Add item2 to b1
                    add_item_to_bin(item2, b1)
                    # Add item1 to b2
                    add_item_to_bin(item1, b2)

                    current_score += delta

                    update_min_bin_after_change([b1, b2])

                    cur_nb = len(bin_items)
                    if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
                        best_num_bins = cur_nb
                        best_score = current_score
                        best_item_bin = item_bin[:]
                        best_bin_loads = bin_loads[:]
                    accepted = True
                    break
                else:
                    break  # SA rejected, don't retry

        else:
            # Move Type 3: Targeted Bin Elimination
            src = min_bin
            items_in_src = list(bin_items[src])
            if len(items_in_src) == 0:
                continue

            _shuffle(items_in_src)

            for item in items_in_src:
                if item_bin[item] != src:
                    # Item was already moved (bin indices may have shifted)
                    continue
                wi = w[item]

                # Find best-fit destination
                best_dst = -1
                best_dst_load = -1
                for b in range(len(bin_items)):
                    if b == src:
                        continue
                    if bin_loads[b] + wi <= C and bin_loads[b] > best_dst_load:
                        best_dst_load = bin_loads[b]
                        best_dst = b

                if best_dst == -1:
                    continue

                dst = best_dst
                old_src_load = bin_loads[src]
                old_dst_load = bin_loads[dst]
                new_src_load = old_src_load - wi
                new_dst_load = old_dst_load + wi
                src_becomes_empty = (len(bin_items[src]) == 1)

                if src_becomes_empty:
                    delta = -C2 * (n + 1) + new_dst_load * new_dst_load - old_dst_load * old_dst_load - old_src_load * old_src_load
                else:
                    delta = (new_src_load * new_src_load - old_src_load * old_src_load +
                             new_dst_load * new_dst_load - old_dst_load * old_dst_load)

                if delta >= 0:
                    accept = True
                else:
                    dt = delta / T
                    if dt < -500:
                        accept = False
                    else:
                        accept = _random() < _exp(dt)

                if accept:
                    remove_item_from_bin(item, src)
                    add_item_to_bin(item, dst)
                    current_score += delta

                    if src_becomes_empty:
                        remove_empty_bin(src)
                        min_bin = find_min_bin()
                        min_bin_load = bin_loads[min_bin] if bin_items else 0

                        cur_nb = len(bin_items)
                        if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
                            best_num_bins = cur_nb
                            best_score = current_score
                            best_item_bin = item_bin[:]
                            best_bin_loads = bin_loads[:]
                        break
                    else:
                        update_min_bin_after_change([src, dst])

            cur_nb = len(bin_items)
            if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
                best_num_bins = cur_nb
                best_score = current_score
                best_item_bin = item_bin[:]
                best_bin_loads = bin_loads[:]

    # Reconstruct packing from best_item_bin
    packing = [[] for _ in range(best_num_bins)]
    for i, b in enumerate(best_item_bin):
        packing[b].append(i)

    # Filter out any empty bins (shouldn't happen but safety)
    packing = [p for p in packing if p]

    result_weights = [sum(w[i] for i in p) for p in packing]

    return {"packing": packing, "bin_weights": result_weights}
