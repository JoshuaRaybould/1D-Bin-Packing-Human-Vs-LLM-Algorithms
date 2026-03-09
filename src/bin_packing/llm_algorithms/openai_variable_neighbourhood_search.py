# openai
# variable_neighbourhood_search_0_initial.py

import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = bin_capacity

    # ---------- Helpers: objective and solution container ----------
    def obj(num_bins: int, loads: List[int]) -> Tuple[int, int, int]:
        # Primary: minimize number of bins
        # Secondary: minimize total waste
        # Tertiary: prefer tighter packing (maximize sum of squares of loads)
        waste = num_bins * C - sum(loads)
        ss = sum(x * x for x in loads)
        return (num_bins, waste, -ss)

    def build_packing(assign: List[int], bins: List[List[int]], loads: List[int]) -> Tuple[List[List[int]], List[int]]:
        # Reindex bins to remove empties and make bin ids dense
        mapping = {}
        new_bins: List[List[int]] = []
        new_loads: List[int] = []
        for b, items in enumerate(bins):
            if items:
                mapping[b] = len(new_bins)
                new_bins.append(items[:])
                new_loads.append(loads[b])
        # Update assignment (not strictly needed for output, but used internally)
        for i in range(n):
            assign[i] = mapping[assign[i]]
        return new_bins, new_loads

    # ---------- Initial solution: randomized FFD ----------
    items = list(range(n))
    # Sort by weight descending, but allow slight random shuffling among equal weights
    items.sort(key=lambda i: weights[i], reverse=True)
    # Random perturbation within same-weight runs
    i = 0
    while i < n:
        j = i + 1
        while j < n and weights[items[j]] == weights[items[i]]:
            j += 1
        if j - i > 1:
            block = items[i:j]
            random.shuffle(block)
            items[i:j] = block
        i = j

    assign = [-1] * n
    bins: List[List[int]] = []
    loads: List[int] = []

    for it in items:
        w = weights[it]
        placed = False
        # First-fit with a small randomization: sample a few candidate bins first
        m = len(bins)
        if m > 0:
            # Try up to 6 random bins before scanning all
            tries = min(6, m)
            cand = random.sample(range(m), tries)
            for b in cand:
                if loads[b] + w <= C:
                    bins[b].append(it)
                    loads[b] += w
                    assign[it] = b
                    placed = True
                    break
        if not placed:
            for b in range(len(bins)):
                if loads[b] + w <= C:
                    bins[b].append(it)
                    loads[b] += w
                    assign[it] = b
                    placed = True
                    break
        if not placed:
            assign[it] = len(bins)
            bins.append([it])
            loads.append(w)

    best_assign = assign[:]
    best_bins = [b[:] for b in bins]
    best_loads = loads[:]
    best_obj = obj(len(best_bins), best_loads)

    # ---------- Neighborhood operations ----------
    def remove_item_from_bin(b: int, it: int):
        # swap-remove for speed
        lst = bins[b]
        idx = lst.index(it)
        last = lst[-1]
        lst[idx] = last
        lst.pop()

    def try_close_empty_bins():
        nonlocal bins, loads
        # Occasionally rebuild to remove empties
        if random.random() < 0.2:
            new_bins, new_loads = build_packing(assign, bins, loads)
            bins = new_bins
            loads = new_loads

    def feasible_relocate(it: int, to_b: int) -> bool:
        b_from = assign[it]
        if b_from == to_b:
            return False
        w = weights[it]
        return loads[to_b] + w <= C

    def do_relocate(it: int, to_b: int):
        b_from = assign[it]
        w = weights[it]
        remove_item_from_bin(b_from, it)
        loads[b_from] -= w
        bins[to_b].append(it)
        loads[to_b] += w
        assign[it] = to_b

    def feasible_swap(i1: int, i2: int) -> bool:
        b1 = assign[i1]
        b2 = assign[i2]
        if b1 == b2:
            return False
        w1 = weights[i1]
        w2 = weights[i2]
        return (loads[b1] - w1 + w2 <= C) and (loads[b2] - w2 + w1 <= C)

    def do_swap(i1: int, i2: int):
        b1 = assign[i1]
        b2 = assign[i2]
        # remove both
        remove_item_from_bin(b1, i1)
        loads[b1] -= weights[i1]
        remove_item_from_bin(b2, i2)
        loads[b2] -= weights[i2]
        # insert swapped
        bins[b1].append(i2)
        loads[b1] += weights[i2]
        assign[i2] = b1
        bins[b2].append(i1)
        loads[b2] += weights[i1]
        assign[i1] = b2

    def current_obj() -> Tuple[int, int, int]:
        # compute using only non-empty bins
        nb = sum(1 for b in bins if b)
        # loads may include empties; only count non-empty
        lds = [loads[i] for i in range(len(bins)) if bins[i]]
        return obj(nb, lds)

    # ---------- Local search: VND (improving moves) ----------
    def vnd(max_passes: int = 50):
        nonlocal best_obj, best_assign, best_bins, best_loads
        passes = 0
        while passes < max_passes:
            if time.time() - start > time_limit:
                return

            improved = False

            # Rebuild bin list occasionally to remove empties and tighten ids
            try_close_empty_bins()

            # Neighborhood 1: Relocate (prefer moves that empty a bin)
            # Iterate bins in increasing load (easier to empty small bins)
            bin_ids = [b for b in range(len(bins)) if bins[b]]
            bin_ids.sort(key=lambda b: loads[b])

            for b_from in bin_ids:
                if time.time() - start > time_limit:
                    return
                if not bins[b_from]:
                    continue
                # Try to move items out of b_from
                # Consider heavier items first to maximize chance of emptying
                candidates = bins[b_from][:]
                candidates.sort(key=lambda it: weights[it], reverse=True)

                for it in candidates:
                    # Try best-fit destination: minimal residual
                    w = weights[it]
                    dests = []
                    for b_to in bin_ids:
                        if b_to == b_from or not bins[b_to]:
                            continue
                        rem = C - loads[b_to]
                        if rem >= w:
                            dests.append((rem - w, b_to))
                    if dests:
                        dests.sort()
                        # try a few best destinations
                        for _, b_to in dests[:6]:
                            do_relocate(it, b_to)
                            new_o = current_obj()
                            if new_o < best_obj:
                                # accept
                                best_obj = new_o
                                best_assign = assign[:]
                                best_bins = [bb[:] for bb in bins]
                                best_loads = loads[:]
                                improved = True
                                break
                            else:
                                # revert
                                do_relocate(it, b_from)
                            if time.time() - start > time_limit:
                                return
                        if improved:
                            break
                if improved:
                    break

            if improved:
                passes += 1
                continue

            # Neighborhood 2: Swap
            # Sample limited number of pairs for speed
            all_items = list(range(n))
            # Focus on items in light bins and heavy bins
            light_bins = sorted(bin_ids, key=lambda b: loads[b])[: max(1, len(bin_ids) // 3)]
            heavy_bins = sorted(bin_ids, key=lambda b: loads[b], reverse=True)[: max(1, len(bin_ids) // 3)]
            pool = []
            for b in light_bins + heavy_bins:
                pool.extend(bins[b])
            if not pool:
                pool = all_items

            sample_items = random.sample(pool, min(len(pool), 40))
            sample_items2 = random.sample(all_items, min(n, 60))

            for i1 in sample_items:
                if time.time() - start > time_limit:
                    return
                for i2 in sample_items2:
                    if i1 == i2:
                        continue
                    if feasible_swap(i1, i2):
                        b1 = assign[i1]
                        b2 = assign[i2]
                        do_swap(i1, i2)
                        new_o = current_obj()
                        if new_o < best_obj:
                            best_obj = new_o
                            best_assign = assign[:]
                            best_bins = [bb[:] for bb in bins]
                            best_loads = loads[:]
                            improved = True
                            break
                        else:
                            # revert swap
                            do_swap(i1, i2)
                            # restore b1/b2 not needed because swap is involution
                if improved:
                    break

            if improved:
                passes += 1
                continue

            # Neighborhood 3: Try to empty a bin by moving two items out (2-0)
            # Choose a small bin and attempt to move its contents elsewhere
            bin_ids = [b for b in range(len(bins)) if bins[b]]
            bin_ids.sort(key=lambda b: loads[b])
            for b_from in bin_ids[: min(len(bin_ids), 8)]:
                if time.time() - start > time_limit:
                    return
                items_b = bins[b_from][:]
                if len(items_b) < 2:
                    continue
                # pick up to 10 pairs
                random.shuffle(items_b)
                pairs = []
                for a in range(min(len(items_b), 8)):
                    for c in range(a + 1, min(len(items_b), 8)):
                        pairs.append((items_b[a], items_b[c]))
                random.shuffle(pairs)
                pairs = pairs[:10]

                for it1, it2 in pairs:
                    # attempt to relocate both to other bins (greedy best-fit)
                    if assign[it1] != b_from or assign[it2] != b_from:
                        continue
                    # find destinations
                    def best_dest(it, forbid):
                        w = weights[it]
                        best = None
                        for b_to in bin_ids:
                            if b_to == forbid or not bins[b_to]:
                                continue
                            rem = C - loads[b_to]
                            if rem >= w:
                                score = rem - w
                                if best is None or score < best[0]:
                                    best = (score, b_to)
                        return best[1] if best else None

                    d1 = best_dest(it1, b_from)
                    if d1 is None:
                        continue
                    do_relocate(it1, d1)
                    d2 = best_dest(it2, b_from)
                    if d2 is None:
                        # revert
                        do_relocate(it1, b_from)
                        continue
                    do_relocate(it2, d2)

                    new_o = current_obj()
                    if new_o < best_obj:
                        best_obj = new_o
                        best_assign = assign[:]
                        best_bins = [bb[:] for bb in bins]
                        best_loads = loads[:]
                        improved = True
                        break
                    else:
                        # revert
                        do_relocate(it2, b_from)
                        do_relocate(it1, b_from)

                if improved:
                    break

            if improved:
                passes += 1
                continue

            # No improving move found
            break

    # ---------- Shaking for VNS ----------
    def shaking(k: int):
        # Apply k random moves (relocate/swap) to diversify
        if n <= 1:
            return
        bin_ids = [b for b in range(len(bins)) if bins[b]]
        if not bin_ids:
            return

        for _ in range(k):
            if time.time() - start > time_limit:
                return
            move_type = random.random()

            # Occasionally rebuild to clear empties
            if random.random() < 0.1:
                nonlocal_bins, nonlocal_loads = build_packing(assign, bins, loads)
                bins[:] = nonlocal_bins
                loads[:] = nonlocal_loads
                bin_ids = [b for b in range(len(bins)) if bins[b]]
                if not bin_ids:
                    return

            if move_type < 0.65:
                # random relocate (may create a new bin rarely)
                it = random.randrange(n)
                b_from = assign[it]
                # Choose destination among existing bins; sometimes force a new bin by picking an empty slot
                candidates = [b for b in bin_ids if b != b_from]
                if candidates:
                    b_to = random.choice(candidates)
                    if feasible_relocate(it, b_to):
                        do_relocate(it, b_to)
                    else:
                        # try a few random bins
                        for __ in range(5):
                            b_to = random.choice(candidates)
                            if feasible_relocate(it, b_to):
                                do_relocate(it, b_to)
                                break
                # If bin got empty, keep it (will be cleaned later)
            else:
                # random swap
                i1 = random.randrange(n)
                i2 = random.randrange(n)
                if i1 != i2 and feasible_swap(i1, i2):
                    do_swap(i1, i2)

    # ---------- Main VNS loop ----------
    # Fixed iteration budget; also stops on time.
    # k_max controls neighborhood size.
    k_max = 12
    max_iters = 250

    # Initial intensification
    vnd(max_passes=60)

    iters = 0
    while iters < max_iters:
        if time.time() - start > time_limit:
            break
        k = 1
        while k <= k_max:
            if time.time() - start > time_limit:
                break

            # Start from current best to make shaking meaningful
            assign = best_assign[:]
            bins = [bb[:] for bb in best_bins]
            loads = best_loads[:]

            shaking(k)
            vnd(max_passes=35)

            # Acceptance: if improved, restart neighborhood
            cur_o = current_obj()
            if cur_o < best_obj:
                best_obj = cur_o
                best_assign = assign[:]
                best_bins = [bb[:] for bb in bins]
                best_loads = loads[:]
                k = 1
            else:
                k += 1

        iters += 1

    # Final cleanup and output
    assign = best_assign[:]
    bins = [bb[:] for bb in best_bins]
    loads = best_loads[:]
    bins, loads = build_packing(assign, bins, loads)

    # Ensure bin_weights aligned
    return {"packing": bins, "bin_weights": loads}
