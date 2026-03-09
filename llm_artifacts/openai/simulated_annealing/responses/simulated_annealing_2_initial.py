import time
import math
import random
from typing import List, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    n = len(weights)
    C = bin_capacity
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    # ---- Helper: initial solution (robust greedy: Best-Fit Decreasing) ----
    items = list(range(n))
    items.sort(key=lambda i: weights[i], reverse=True)

    bin_loads: List[int] = []
    bins: List[List[int]] = []
    item_bin = [-1] * n

    for i in items:
        w = weights[i]
        best_b = -1
        best_rem = None
        # choose bin with minimal remaining capacity after placement (allow only feasible in init)
        for b, load in enumerate(bin_loads):
            if load + w <= C:
                rem = C - (load + w)
                if best_rem is None or rem < best_rem:
                    best_rem = rem
                    best_b = b
        if best_b == -1:
            best_b = len(bin_loads)
            bin_loads.append(0)
            bins.append([])
        bins[best_b].append(i)
        bin_loads[best_b] += w
        item_bin[i] = best_b

    # ---- Objective with penalty for infeasibility (SA can traverse infeasible states) ----
    # Large constant to prioritize fewer bins among feasible solutions.
    M = C * (n + 5)

    def score_state(loads: List[int]) -> float:
        # bins used = count of non-empty bins
        bins_used = 0
        over_sq = 0
        slack = 0
        for ld in loads:
            if ld > 0:
                bins_used += 1
                if ld > C:
                    d = ld - C
                    over_sq += d * d
                else:
                    slack += (C - ld)
        # Penalty multiplier; scale by M to dominate.
        return bins_used * M + over_sq * (M * 50) + slack * 0.01

    def is_feasible(loads: List[int]) -> bool:
        for ld in loads:
            if ld > C:
                return False
        return True

    def export_solution() -> Dict:
        packing = [lst[:] for lst in bins if lst]
        bw = []
        for lst in packing:
            s = 0
            for it in lst:
                s += weights[it]
            bw.append(s)
        return {"packing": packing, "bin_weights": bw}

    # Track best feasible
    cur_score = score_state(bin_loads)
    best_bins = None
    best_slack = None
    best_snapshot = None

    def update_best():
        nonlocal best_bins, best_slack, best_snapshot
        if not is_feasible(bin_loads):
            return
        used = sum(1 for b in bins if b)
        slack = 0
        for ld in bin_loads:
            if ld > 0:
                slack += (C - ld)
        if (best_bins is None) or (used < best_bins) or (used == best_bins and slack < best_slack):
            best_bins = used
            best_slack = slack
            best_snapshot = ( [lst[:] for lst in bins], bin_loads[:], item_bin[:] )

    update_best()

    # ---- SA parameters ----
    # Fixed iteration budget; also stop on time limit.
    # Budget scales with n but capped to be safe.
    max_iter = int(min(500000, max(20000, 4000 * math.sqrt(n) + 8000)))

    # Temperature schedule
    T0 = max(1.0, C * 0.5)
    Tf = 1e-3
    alpha = (Tf / T0) ** (1.0 / max(1, max_iter - 1))
    T = T0

    # Stagnation/reheating
    last_improve_iter = 0

    # ---- Fast ops for empty-bin cleanup ----
    def remove_empty_bin(b: int):
        # Keep data structures dense by swapping with last bin
        nonlocal bins, bin_loads, item_bin
        last = len(bins) - 1
        if b != last:
            bins[b], bins[last] = bins[last], bins[b]
            bin_loads[b], bin_loads[last] = bin_loads[last], bin_loads[b]
            for it in bins[b]:
                item_bin[it] = b
        bins.pop()
        bin_loads.pop()

    # ---- Neighborhood moves ----
    def move_item(i: int, b_from: int, b_to: int):
        # assumes b_to exists
        w = weights[i]
        # remove i from b_from list (swap-remove)
        lst = bins[b_from]
        pos = None
        # small bins; linear search ok
        for k, it in enumerate(lst):
            if it == i:
                pos = k
                break
        last_it = lst[-1]
        lst[pos] = last_it
        lst.pop()
        bin_loads[b_from] -= w

        # add to b_to
        bins[b_to].append(i)
        bin_loads[b_to] += w
        item_bin[i] = b_to

        # remove empty bin if needed
        if not bins[b_from]:
            remove_empty_bin(b_from)

    def swap_items(i: int, bi: int, j: int, bj: int):
        if bi == bj:
            return
        wi, wj = weights[i], weights[j]
        # swap in lists
        li, lj = bins[bi], bins[bj]
        pi = pj = None
        for k, it in enumerate(li):
            if it == i:
                pi = k
                break
        for k, it in enumerate(lj):
            if it == j:
                pj = k
                break
        li[pi], lj[pj] = j, i
        bin_loads[bi] += (wj - wi)
        bin_loads[bj] += (wi - wj)
        item_bin[i] = bj
        item_bin[j] = bi

    # Large-neighborhood move inside SA: try to eliminate a bin by reinserting its items greedily
    def try_eliminate_bin(b: int):
        nonlocal bins, bin_loads, item_bin
        if b < 0 or b >= len(bins) or not bins[b]:
            return False
        items_to_place = bins[b][:]
        # remove the bin entirely
        for it in items_to_place:
            item_bin[it] = -1
        # delete bin b
        remove_empty_bin(b)

        # reinsert items in decreasing weight; allow creating new bin if needed
        items_to_place.sort(key=lambda i: weights[i], reverse=True)
        for it in items_to_place:
            w = weights[it]
            best_b = -1
            best_rem = None
            for bb, ld in enumerate(bin_loads):
                if ld + w <= C:
                    rem = C - (ld + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_b = bb
            if best_b == -1:
                best_b = len(bins)
                bins.append([])
                bin_loads.append(0)
            bins[best_b].append(it)
            bin_loads[best_b] += w
            item_bin[it] = best_b
        return True

    # ---- SA loop ----
    # To avoid too-frequent time checks, check every k iterations
    time_check_period = 500

    for it in range(max_iter):
        if (it % time_check_period) == 0 and time.time() >= deadline:
            break

        # Occasionally attempt elimination as a proposal (part of SA neighborhood)
        do_elim = (random.random() < 0.03)

        # Snapshot for potential rollback (store just what we need)
        # For speed, we rollback by recording the exact move rather than cloning.
        accepted = False

        old_score = cur_score

        if do_elim and len(bins) >= 2:
            # choose a relatively light bin (more likely eliminable)
            candidates = list(range(len(bins)))
            b = min(candidates, key=lambda x: bin_loads[x] if bins[x] else 10**18)

            # full snapshot for this heavier move
            snapshot = ([lst[:] for lst in bins], bin_loads[:], item_bin[:])
            try_eliminate_bin(b)
            new_score = score_state(bin_loads)
            d = new_score - old_score
            if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                accepted = True
                cur_score = new_score
            else:
                bins, bin_loads, item_bin = snapshot
                accepted = False
        else:
            # Regular neighborhood: move or swap
            if len(bins) == 1:
                op = 0
            else:
                op = 0 if random.random() < 0.65 else 1

            if op == 0:
                # Move an item to another bin (or new bin)
                i = random.randrange(n)
                b_from = item_bin[i]
                if b_from == -1:
                    continue

                # choose destination
                # with small prob, open a new bin (useful for escaping)
                if random.random() < 0.05:
                    b_to = len(bins)
                    bins.append([])
                    bin_loads.append(0)
                    # apply move
                    # record for rollback
                    prev_len = len(bins[b_from])
                    move_item(i, b_from, b_to)
                    new_score = score_state(bin_loads)
                    d = new_score - old_score
                    if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                        accepted = True
                        cur_score = new_score
                    else:
                        # rollback by reversing move and removing new bin
                        # i currently in last bin
                        b_new = len(bins) - 1
                        # move back
                        move_item(i, item_bin[i], b_from)
                        # remove empty new bin (should be empty now)
                        if b_new < len(bins) and not bins[b_new]:
                            remove_empty_bin(b_new)
                        accepted = False
                else:
                    # pick an existing bin different from source
                    b_to = random.randrange(len(bins) - 1)
                    if b_to >= b_from:
                        b_to += 1
                    # Apply move with minimal rollback info
                    # Record whether source bin becomes empty and possible swap-removal side effects handled by full inverse move
                    src_before_empty = (len(bins[b_from]) == 1)
                    # Also record if destination bin index could change due to remove_empty_bin when src empties.
                    # We'll rollback using a full snapshot if src would become empty.
                    if src_before_empty:
                        snapshot = ([lst[:] for lst in bins], bin_loads[:], item_bin[:])
                        move_item(i, b_from, b_to)
                        new_score = score_state(bin_loads)
                        d = new_score - old_score
                        if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                            accepted = True
                            cur_score = new_score
                        else:
                            bins, bin_loads, item_bin = snapshot
                            accepted = False
                    else:
                        # cheap rollback: just move back
                        move_item(i, b_from, b_to)
                        new_score = score_state(bin_loads)
                        d = new_score - old_score
                        if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                            accepted = True
                            cur_score = new_score
                        else:
                            # move back
                            move_item(i, item_bin[i], b_from)
                            accepted = False
            else:
                # Swap two items from different bins
                i = random.randrange(n)
                j = random.randrange(n)
                if i == j:
                    continue
                bi = item_bin[i]
                bj = item_bin[j]
                if bi == -1 or bj == -1 or bi == bj:
                    continue
                # swap
                swap_items(i, bi, j, bj)
                new_score = score_state(bin_loads)
                d = new_score - old_score
                if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                    accepted = True
                    cur_score = new_score
                else:
                    # swap back
                    swap_items(i, item_bin[i], j, item_bin[j])
                    accepted = False

        if accepted:
            update_best()
            # track improvement in best objective proxy
            if best_bins is not None:
                last_improve_iter = it

        # Cooling
        T *= alpha

        # Reheat if long stagnation (keeps SA as the primary driver)
        if it - last_improve_iter > 15000:
            T = max(T, T0 * 0.25)
            last_improve_iter = it

    # Restore best feasible snapshot if exists; else return current (may be infeasible but should be rare)
    if best_snapshot is not None:
        b_bins, b_loads, b_item_bin = best_snapshot
        bins = b_bins
        bin_loads = b_loads
        item_bin = b_item_bin

    # Ensure alignment and feasibility cleanup (should already be feasible if best_snapshot used)
    # Remove any empty bins
    k = 0
    while k < len(bins):
        if not bins[k]:
            remove_empty_bin(k)
        else:
            k += 1

    return export_solution()
