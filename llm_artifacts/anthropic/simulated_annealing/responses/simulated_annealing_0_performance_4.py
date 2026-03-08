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

    indices_sorted = sorted(range(n), key=lambda i: w[i], reverse=True)

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

    def run_wfd():
        bins_items = []
        bins_loads = []
        ib = [0] * n
        for idx in indices_sorted:
            wi = w[idx]
            best_b = -1
            best_load = -1
            for b in range(len(bins_items)):
                if bins_loads[b] + wi <= C and bins_loads[b] > best_load:
                    best_load = bins_loads[b]
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

    candidates = []
    for runner in [run_ffd, run_bfd, run_wfd]:
        bi, bl, ib = runner()
        s = 0
        for ld in bl:
            s += ld * ld
        sc = -len(bl) * C2 * (n + 1) + s
        candidates.append((len(bl), -sc, bi, bl, ib))

    candidates.sort(key=lambda x: (x[0], x[1]))
    bin_items, bin_loads, item_bin = candidates[0][2], candidates[0][3], candidates[0][4]

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

    best_num_bins = num_bins
    best_score = current_score
    best_item_bin = item_bin[:]
    best_bin_loads = bin_loads[:]

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
        bin_items.pop()
        bin_loads.pop()
        num_bins = len(bin_items)

    def find_min_bin():
        min_b = 0
        min_ld = bin_loads[0]
        for b in range(1, len(bin_items)):
            if bin_loads[b] < min_ld:
                min_ld = bin_loads[b]
                min_b = b
        return min_b

    def find_two_min_bins():
        if len(bin_items) < 2:
            return find_min_bin(), -1
        if bin_loads[0] <= bin_loads[1]:
            m1, m2 = 0, 1
        else:
            m1, m2 = 1, 0
        for b in range(2, len(bin_items)):
            ld = bin_loads[b]
            if ld < bin_loads[m1]:
                m2 = m1
                m1 = b
            elif ld < bin_loads[m2]:
                m2 = b
        return m1, m2

    min_bin = find_min_bin()
    min_bin_load = bin_loads[min_bin]

    def update_min_bin_after_change(changed_bins):
        nonlocal min_bin, min_bin_load
        if min_bin >= len(bin_items):
            min_bin = find_min_bin()
            min_bin_load = bin_loads[min_bin]
            return
        for b in changed_bins:
            if b >= len(bin_items):
                continue
            if bin_loads[b] < min_bin_load:
                min_bin = b
                min_bin_load = bin_loads[b]
        if bin_loads[min_bin] > min_bin_load:
            min_bin = find_min_bin()
        min_bin_load = bin_loads[min_bin]

    def save_best():
        nonlocal best_num_bins, best_score, best_item_bin, best_bin_loads
        cur_nb = len(bin_items)
        if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
            best_num_bins = cur_nb
            best_score = current_score
            best_item_bin = item_bin[:]
            best_bin_loads = bin_loads[:]

    if time.time() >= deadline:
        packing = [[] for _ in range(best_num_bins)]
        for i, b in enumerate(best_item_bin):
            packing[b].append(i)
        bw = [sum(w[i] for i in p) for p in packing]
        return {"packing": packing, "bin_weights": bw}

    _random = random.random
    _randrange = random.randrange
    _randint = random.randint
    _shuffle = random.shuffle
    _exp = math.exp
    _time = time.time
    _w = w

    T_max = C2 * 0.5
    T_mid = C2 * 0.05
    T_low = C2 * 0.003
    T_min_val = 0.01
    total_time = deadline - start_time
    if total_time <= 0:
        total_time = 0.001

    T = T_max

    iteration = 0
    check_interval = 3000
    stagnation_count = 0
    last_best_bins = best_num_bins

    while True:
        iteration += 1
        if iteration % check_interval == 0:
            now = _time()
            if now >= deadline:
                break
            frac = (now - start_time) / total_time
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

            if frac < 0.25:
                f = frac / 0.25
                T = T_max * (1.0 - f) + T_mid * f
            elif frac < 0.75:
                f = (frac - 0.25) / 0.5
                T = T_mid * (1.0 - f) + T_low * f
            else:
                f = (frac - 0.75) / 0.25
                T = T_low * (1.0 - f) + T_min_val * f

            if T < T_min_val:
                T = T_min_val

            if num_bins == last_best_bins:
                stagnation_count += check_interval
            else:
                stagnation_count = 0
                last_best_bins = num_bins

            if stagnation_count >= 80000 and len(bin_items) > 2:
                stagnation_count = 0
                sorted_bins = sorted(range(len(bin_items)), key=lambda b: bin_loads[b])
                k = min(3, len(sorted_bins) - 1)
                to_empty = sorted_bins[:k]
                displaced_items = []
                for b in sorted(to_empty, reverse=True):
                    for itm in list(bin_items[b]):
                        displaced_items.append(itm)
                    bin_items[b] = []
                    bin_loads[b] = 0

                empties = sorted([b for b in range(len(bin_items)) if len(bin_items[b]) == 0], reverse=True)
                for b in empties:
                    remove_empty_bin(b)

                displaced_items.sort(key=lambda i: _w[i], reverse=True)
                for itm in displaced_items:
                    wi = _w[itm]
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
                        new_b = len(bin_items)
                        bin_items.append([])
                        bin_loads.append(0)
                        num_bins = len(bin_items)
                        add_item_to_bin(itm, new_b)

                current_score = compute_score()
                min_bin = find_min_bin()
                min_bin_load = bin_loads[min_bin]
                T = T_mid
                save_best()

        nb = len(bin_items)
        if nb <= 1:
            break

        r = _random()

        if r < 0.40:
            # Move Type 1: Single Item Transfer
            rr = _random()
            if rr < 0.6:
                src = min_bin
            elif rr < 0.8:
                # pick one of 2 least loaded
                m1, m2 = find_two_min_bins()
                src = m2 if m2 >= 0 and _random() < 0.5 else m1
            else:
                src = _randrange(nb)

            src_len = len(bin_items[src])
            if src_len == 0:
                continue

            item = bin_items[src][_randrange(src_len)]
            wi = _w[item]

            best_dst = -1
            best_dst_load = -1
            trials = min(nb - 1, 10)
            for _ in range(trials):
                b = _randrange(nb)
                if b == src:
                    continue
                if bin_loads[b] + wi <= C and bin_loads[b] > best_dst_load:
                    best_dst_load = bin_loads[b]
                    best_dst = b

            if best_dst == -1:
                # Try linear scan for feasible bin
                for b in range(nb):
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
                accept = dt > -500 and _random() < _exp(dt)

            if accept:
                remove_item_from_bin(item, src)
                add_item_to_bin(item, dst)
                current_score += delta

                if src_becomes_empty:
                    remove_empty_bin(src)
                    min_bin = find_min_bin()
                    min_bin_load = bin_loads[min_bin] if bin_items else 0
                else:
                    update_min_bin_after_change([src, dst])

                save_best()

        elif r < 0.60:
            # Move Type 2: Pairwise Swap
            b1 = _randrange(nb)
            b2 = _randrange(nb)
            if b1 == b2 or len(bin_items[b1]) == 0 or len(bin_items[b2]) == 0:
                continue

            item1 = bin_items[b1][_randrange(len(bin_items[b1]))]
            item2 = bin_items[b2][_randrange(len(bin_items[b2]))]
            w1 = _w[item1]
            w2 = _w[item2]

            if w1 == w2:
                continue

            new_load_b1 = bin_loads[b1] - w1 + w2
            new_load_b2 = bin_loads[b2] - w2 + w1

            if new_load_b1 > C or new_load_b2 > C:
                continue

            delta = (new_load_b1 * new_load_b1 - bin_loads[b1] * bin_loads[b1] +
                     new_load_b2 * new_load_b2 - bin_loads[b2] * bin_loads[b2])

            if delta >= 0:
                accept = True
            else:
                dt = delta / T
                accept = dt > -500 and _random() < _exp(dt)

            if accept:
                remove_item_from_bin(item1, b1)
                remove_item_from_bin(item2, b2)
                add_item_to_bin(item2, b1)
                add_item_to_bin(item1, b2)
                current_score += delta
                update_min_bin_after_change([b1, b2])
                save_best()

        elif r < 0.75:
            # Move Type 3: (2,1) exchange - move 2 items from src, 1 item from dst to src
            # This helps empty bins by reducing src load
            rr = _random()
            if rr < 0.7:
                src = min_bin
            else:
                src = _randrange(nb)
            
            src_len = len(bin_items[src])
            if src_len < 2:
                continue
            
            # Pick 2 items from src
            idx1 = _randrange(src_len)
            idx2 = _randrange(src_len - 1)
            if idx2 >= idx1:
                idx2 += 1
            item1 = bin_items[src][idx1]
            item2 = bin_items[src][idx2]
            w1 = _w[item1]
            w2 = _w[item2]
            
            # Find a dst bin with an item that fits in src after removing item1,item2
            dst = _randrange(nb)
            if dst == src or len(bin_items[dst]) == 0:
                continue
            
            # Pick item from dst
            item3 = bin_items[dst][_randrange(len(bin_items[dst]))]
            w3 = _w[item3]
            
            new_src_load = bin_loads[src] - w1 - w2 + w3
            new_dst_load = bin_loads[dst] - w3 + w1 + w2
            
            if new_src_load < 0 or new_src_load > C or new_dst_load > C:
                continue
            
            delta = (new_src_load * new_src_load - bin_loads[src] * bin_loads[src] +
                     new_dst_load * new_dst_load - bin_loads[dst] * bin_loads[dst])
            
            if delta >= 0:
                accept = True
            else:
                dt = delta / T
                accept = dt > -500 and _random() < _exp(dt)
            
            if accept:
                remove_item_from_bin(item1, src)
                remove_item_from_bin(item2, src)
                remove_item_from_bin(item3, dst)
                add_item_to_bin(item3, src)
                add_item_to_bin(item1, dst)
                add_item_to_bin(item2, dst)
                current_score += delta
                update_min_bin_after_change([src, dst])
                save_best()

        else:
            # Move Type 4: Targeted Bin Elimination
            src = min_bin
            items_in_src = list(bin_items[src])
            if len(items_in_src) == 0:
                continue

            _shuffle(items_in_src)

            for item in items_in_src:
                if item_bin[item] != src:
                    continue
                wi = _w[item]

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

                # For targeted elimination, be more lenient
                if delta >= 0:
                    accept = True
                else:
                    dt = delta / (T * 2.0)
                    accept = dt > -500 and _random() < _exp(dt)

                if accept:
                    remove_item_from_bin(item, src)
                    add_item_to_bin(item, dst)
                    current_score += delta

                    if src_becomes_empty:
                        remove_empty_bin(src)
                        min_bin = find_min_bin()
                        min_bin_load = bin_loads[min_bin] if bin_items else 0
                        save_best()
                        break
                    else:
                        update_min_bin_after_change([src, dst])

            save_best()

    packing = [[] for _ in range(best_num_bins)]
    for i, b in enumerate(best_item_bin):
        packing[b].append(i)

    packing = [p for p in packing if p]

    result_weights = [sum(w[i] for i in p) for p in packing]

    return {"packing": packing, "bin_weights": result_weights}