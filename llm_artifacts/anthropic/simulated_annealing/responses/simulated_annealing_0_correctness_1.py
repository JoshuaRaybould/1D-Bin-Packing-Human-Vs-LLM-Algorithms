def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import random
    import time
    import math

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start_time = time.time()
    deadline = start_time + time_limit * 0.95  # leave some margin

    # FFD initial solution
    indices_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    bins = []       # list of sets of item indices
    bin_loads = []   # current load of each bin
    item_bin = [0] * n  # which bin each item is in

    for idx in indices_sorted:
        w_item = weights[idx]
        placed = False
        for b in range(len(bins)):
            if bin_loads[b] + w_item <= bin_capacity:
                bins[b].add(idx)
                bin_loads[b] += w_item
                item_bin[idx] = b
                placed = True
                break
        if not placed:
            item_bin[idx] = len(bins)
            bins.append({idx})
            bin_loads.append(w_item)

    # Remove any empty bins
    def compact_bins():
        nonlocal bins, bin_loads, item_bin
        new_bins = []
        new_loads = []
        for b in range(len(bins)):
            if bins[b]:  # non-empty
                new_idx = len(new_bins)
                for item in bins[b]:
                    item_bin[item] = new_idx
                new_bins.append(bins[b])
                new_loads.append(bin_loads[b])
        bins = new_bins
        bin_loads = new_loads

    compact_bins()

    # Compute objective: we want to minimize bins, then maximize sum of load^2
    def compute_score():
        s = 0
        for load in bin_loads:
            s += load * load
        return -len(bins) * bin_capacity * bin_capacity * (n + 1) + s

    current_score = compute_score()
    num_bins = len(bins)

    # Save best solution
    best_num_bins = num_bins
    best_score = current_score
    best_packing = [set(b) for b in bins]
    best_bin_loads = list(bin_loads)

    # SA parameters
    C2 = bin_capacity * bin_capacity
    T = C2 * 0.5  # initial temperature
    T_min = 0.01

    elapsed = time.time() - start_time
    remaining = deadline - time.time()

    if remaining <= 0:
        result_packing = [list(b) for b in best_packing]
        result_weights = [sum(weights[i] for i in b) for b in result_packing]
        return {"packing": result_packing, "bin_weights": result_weights}

    # Alias
    w = weights
    exp = math.exp

    # Maintain a list version of each bin for random item selection
    bin_lists = [list(b) for b in bins]

    def rebuild_bin_list(b):
        bin_lists[b] = list(bins[b])

    iteration = 0
    check_interval = 500

    def find_smallest_bin():
        if not bins:
            return -1
        min_load = bin_loads[0]
        min_b = 0
        for b in range(1, len(bins)):
            if bin_loads[b] < min_load:
                min_load = bin_loads[b]
                min_b = b
        return min_b

    while True:
        iteration += 1
        if iteration % check_interval == 0:
            if time.time() >= deadline:
                break

        nb = len(bins)
        if nb <= 1:
            break

        # Choose move type
        r = random.random()

        if r < 0.5:
            # Move: pick item from a bin, try to move to another bin
            if random.random() < 0.6:
                src = find_smallest_bin()
            else:
                src = random.randrange(nb)

            if not bin_lists[src]:
                continue

            item = random.choice(bin_lists[src])
            wi = w[item]

            # Find a target bin
            dst = random.randrange(nb)
            if dst == src:
                dst = (dst + 1) % nb

            if bin_loads[dst] + wi > bin_capacity:
                found = False
                for _ in range(5):
                    dst = random.randrange(nb)
                    if dst != src and bin_loads[dst] + wi <= bin_capacity:
                        found = True
                        break
                if not found:
                    continue

            if dst == src:
                continue
            if bin_loads[dst] + wi > bin_capacity:
                continue

            old_src_load = bin_loads[src]
            old_dst_load = bin_loads[dst]
            new_src_load = old_src_load - wi
            new_dst_load = old_dst_load + wi

            src_becomes_empty = (len(bins[src]) == 1)

            if src_becomes_empty:
                delta = -C2 * (n + 1) + new_dst_load * new_dst_load - old_dst_load * old_dst_load - old_src_load * old_src_load
            else:
                delta = (new_src_load * new_src_load - old_src_load * old_src_load +
                         new_dst_load * new_dst_load - old_dst_load * old_dst_load)

            if delta >= 0:
                accept = True
            else:
                if T > T_min:
                    try:
                        accept = random.random() < exp(delta / T)
                    except:
                        accept = False
                else:
                    accept = False

            if accept:
                bins[src].remove(item)
                bins[dst].add(item)
                bin_loads[src] = new_src_load
                bin_loads[dst] = new_dst_load
                item_bin[item] = dst
                rebuild_bin_list(src)
                rebuild_bin_list(dst)
                current_score += delta

                if src_becomes_empty:
                    last = len(bins) - 1
                    if src != last:
                        bins[src] = bins[last]
                        bin_loads[src] = bin_loads[last]
                        bin_lists[src] = bin_lists[last]
                        for it in bins[src]:
                            item_bin[it] = src
                    bins.pop()
                    bin_loads.pop()
                    bin_lists.pop()
                    num_bins = len(bins)

                cur_nb = len(bins)
                if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
                    best_num_bins = cur_nb
                    best_score = current_score
                    best_packing = [set(b) for b in bins]
                    best_bin_loads = list(bin_loads)

        elif r < 0.85:
            # Swap: pick two items from different bins and swap them
            b1 = random.randrange(nb)
            b2 = random.randrange(nb)
            if b1 == b2:
                continue
            if not bin_lists[b1] or not bin_lists[b2]:
                continue

            item1 = random.choice(bin_lists[b1])
            item2 = random.choice(bin_lists[b2])
            w1 = w[item1]
            w2 = w[item2]

            if w1 == w2:
                continue

            new_load_b1 = bin_loads[b1] - w1 + w2
            new_load_b2 = bin_loads[b2] - w2 + w1

            if new_load_b1 > bin_capacity or new_load_b2 > bin_capacity:
                continue
            if new_load_b1 < 0 or new_load_b2 < 0:
                continue

            delta = (new_load_b1 * new_load_b1 - bin_loads[b1] * bin_loads[b1] +
                     new_load_b2 * new_load_b2 - bin_loads[b2] * bin_loads[b2])

            if delta >= 0:
                accept = True
            else:
                if T > T_min:
                    try:
                        accept = random.random() < exp(delta / T)
                    except:
                        accept = False
                else:
                    accept = False

            if accept:
                bins[b1].remove(item1)
                bins[b1].add(item2)
                bins[b2].remove(item2)
                bins[b2].add(item1)
                bin_loads[b1] = new_load_b1
                bin_loads[b2] = new_load_b2
                item_bin[item1] = b2
                item_bin[item2] = b1
                rebuild_bin_list(b1)
                rebuild_bin_list(b2)
                current_score += delta

                cur_nb = len(bins)
                if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
                    best_num_bins = cur_nb
                    best_score = current_score
                    best_packing = [set(b) for b in bins]
                    best_bin_loads = list(bin_loads)

        else:
            # Try to empty smallest bin by redistributing its items
            src = find_smallest_bin()
            items_in_src = list(bins[src])
            if len(items_in_src) == 0:
                continue

            random.shuffle(items_in_src)
            item = items_in_src[0]
            wi = w[item]

            # Find best fitting bin (most loaded that still fits)
            best_dst = -1
            best_dst_load = -1
            for b in range(len(bins)):
                if b == src:
                    continue
                if bin_loads[b] + wi <= bin_capacity and bin_loads[b] > best_dst_load:
                    best_dst_load = bin_loads[b]
                    best_dst = b

            if best_dst == -1:
                continue

            dst = best_dst
            old_src_load = bin_loads[src]
            old_dst_load = bin_loads[dst]
            new_src_load = old_src_load - wi
            new_dst_load = old_dst_load + wi
            src_becomes_empty = (len(bins[src]) == 1)

            if src_becomes_empty:
                delta = -C2 * (n + 1) + new_dst_load * new_dst_load - old_dst_load * old_dst_load - old_src_load * old_src_load
            else:
                delta = (new_src_load * new_src_load - old_src_load * old_src_load +
                         new_dst_load * new_dst_load - old_dst_load * old_dst_load)

            if delta >= 0:
                accept = True
            else:
                if T > T_min:
                    try:
                        accept = random.random() < exp(delta / T)
                    except:
                        accept = False
                else:
                    accept = False

            if accept:
                bins[src].remove(item)
                bins[dst].add(item)
                bin_loads[src] = new_src_load
                bin_loads[dst] = new_dst_load
                item_bin[item] = dst
                rebuild_bin_list(src)
                rebuild_bin_list(dst)
                current_score += delta

                if src_becomes_empty:
                    last = len(bins) - 1
                    if src != last:
                        bins[src] = bins[last]
                        bin_loads[src] = bin_loads[last]
                        bin_lists[src] = bin_lists[last]
                        for it in bins[src]:
                            item_bin[it] = src
                    bins.pop()
                    bin_loads.pop()
                    bin_lists.pop()
                    num_bins = len(bins)

                cur_nb = len(bins)
                if cur_nb < best_num_bins or (cur_nb == best_num_bins and current_score > best_score):
                    best_num_bins = cur_nb
                    best_score = current_score
                    best_packing = [set(b) for b in bins]
                    best_bin_loads = list(bin_loads)

        # Cool down
        T *= 0.99999
        if T < T_min:
            # Reheat
            T = C2 * 0.3

    # Build result from best solution
    result_packing = [sorted(list(b)) for b in best_packing]
    result_weights = [sum(weights[i] for i in b) for b in result_packing]

    return {"packing": result_packing, "bin_weights": result_weights}
