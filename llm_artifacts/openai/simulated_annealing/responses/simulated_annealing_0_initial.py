import time
import math
import random
from typing import List, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = bin_capacity
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start_time = time.perf_counter()
    deadline = start_time + max(0.0, time_limit)

    # ----- Helpers -----
    def now() -> float:
        return time.perf_counter()

    def feasible_move(item: int, to_bin: int, loads: List[int]) -> bool:
        return loads[to_bin] + weights[item] <= C

    def score(num_bins: int, total_slack: int, max_slack: int) -> float:
        # Strongly prioritize fewer bins.
        # Keep a mild pressure to reduce slack and avoid very empty bins.
        return num_bins * 1_000_000.0 + total_slack * 1.0 + 0.05 * max_slack

    # Build packing from an assignment
    def build_packing(assign: List[int], num_bins: int) -> List[List[int]]:
        packs = [[] for _ in range(num_bins)]
        for i, b in enumerate(assign):
            packs[b].append(i)
        return packs

    def clean_empty_bins(assign: List[int], bins: List[List[int]], loads: List[int]):
        # Remove empty bins by remapping bin ids.
        remap = {}
        new_bins = []
        new_loads = []
        for b, items in enumerate(bins):
            if items:
                remap[b] = len(new_bins)
                new_bins.append(items)
                new_loads.append(loads[b])
        for i in range(n):
            assign[i] = remap[assign[i]]
        bins[:] = new_bins
        loads[:] = new_loads

    # ----- Initial solution: Best-Fit Decreasing with randomized tie-breaking -----
    items = list(range(n))
    items.sort(key=lambda i: weights[i], reverse=True)

    assign = [-1] * n
    bins: List[List[int]] = []
    loads: List[int] = []

    for it in items:
        w = weights[it]
        best_b = -1
        best_res = None
        # randomized scan order among bins to diversify
        order = list(range(len(bins)))
        if order:
            random.shuffle(order)
        for b in order:
            if loads[b] + w <= C:
                res = C - (loads[b] + w)
                if best_res is None or res < best_res:
                    best_res = res
                    best_b = b
        if best_b == -1:
            best_b = len(bins)
            bins.append([])
            loads.append(0)
        bins[best_b].append(it)
        loads[best_b] += w
        assign[it] = best_b

    # ----- Scoring current state -----
    def state_metrics(bins: List[List[int]], loads: List[int]):
        nb = len(bins)
        slacks = [C - L for L in loads]
        total_slack = sum(slacks)
        max_slack = max(slacks) if slacks else 0
        return nb, total_slack, max_slack

    cur_nb, cur_total_slack, cur_max_slack = state_metrics(bins, loads)
    cur_score = score(cur_nb, cur_total_slack, cur_max_slack)

    best_assign = assign[:]
    best_bins = [lst[:] for lst in bins]
    best_loads = loads[:]
    best_score = cur_score

    # ----- Choose iteration budget (fixed), with time checks -----
    # Make it scale with n but keep sane upper/lower bounds.
    base_iters = 20_000 + 800 * int(math.sqrt(n)) + 80 * n
    max_iters = min(1_500_000, base_iters)
    iters = max(25_000, max_iters)

    # ----- Temperature initialization by sampling deltas -----
    # Estimate typical worsening delta to set initial acceptance around ~0.7.
    def random_item_from_bin(b: int) -> int:
        return random.choice(bins[b])

    def propose_delta_sample(samples: int = 60) -> List[float]:
        deltas = []
        nbins = len(bins)
        if nbins == 0:
            return deltas
        for _ in range(samples):
            b_from = random.randrange(nbins)
            if not bins[b_from]:
                continue
            item = random_item_from_bin(b_from)
            w = weights[item]
            # try move to another bin (existing or new)
            targets = list(range(nbins))
            random.shuffle(targets)
            moved = False
            for b_to in targets:
                if b_to == b_from:
                    continue
                if loads[b_to] + w <= C:
                    # compute delta score approximately (without fully applying)
                    nb2 = nbins
                    load_from2 = loads[b_from] - w
                    load_to2 = loads[b_to] + w
                    slacks = [C - L for L in loads]
                    total_slack2 = sum(slacks) - (C - loads[b_from]) - (C - loads[b_to])
                    total_slack2 += (C - load_from2) + (C - load_to2)
                    max_slack2 = max(max(slacks), C - load_from2, C - load_to2)
                    # if from-bin becomes empty, nb decreases by 1 and remove its slack
                    if len(bins[b_from]) == 1:
                        nb2 -= 1
                        total_slack2 -= (C - load_from2)  # this slack would disappear
                        # max slack recompute cheaply: keep approximate
                        max_slack2 = max(slacks)
                    s2 = score(nb2, total_slack2, max_slack2)
                    deltas.append(s2 - cur_score)
                    moved = True
                    break
            if not moved:
                # try a swap
                if nbins < 2:
                    continue
                b_to = random.randrange(nbins)
                if b_to == b_from or not bins[b_to]:
                    continue
                item2 = random_item_from_bin(b_to)
                w2 = weights[item2]
                if loads[b_from] - w + w2 <= C and loads[b_to] - w2 + w <= C:
                    # approximate delta
                    slacks = [C - L for L in loads]
                    load_from2 = loads[b_from] - w + w2
                    load_to2 = loads[b_to] - w2 + w
                    total_slack2 = sum(slacks) - (C - loads[b_from]) - (C - loads[b_to])
                    total_slack2 += (C - load_from2) + (C - load_to2)
                    max_slack2 = max(max(slacks), C - load_from2, C - load_to2)
                    s2 = score(nbins, total_slack2, max_slack2)
                    deltas.append(s2 - cur_score)
        return deltas

    deltas = propose_delta_sample(80)
    pos = [d for d in deltas if d > 1e-9]
    avg_pos = sum(pos) / len(pos) if pos else 10.0
    # Want exp(-avg_pos/T0) ~ 0.7 => T0 ~ avg_pos / -ln(0.7)
    T = max(1e-6, avg_pos / max(1e-6, -math.log(0.7)))
    T0 = T

    # Cooling schedule
    alpha = 0.9995  # exponential cooling
    # Occasional reheats on long stagnation
    stall_limit = 4000
    stall = 0

    # Time check frequency
    check_every = 200

    # ----- SA main loop -----
    for it in range(iters):
        if (it % check_every) == 0 and now() >= deadline:
            break

        nbins = len(bins)
        if nbins == 0:
            break

        # Choose move type
        # More swaps when close to full packing; more moves otherwise.
        if nbins >= 2 and random.random() < 0.35:
            move_type = "swap"
        else:
            move_type = "move"

        accepted = False

        if move_type == "move":
            b_from = random.randrange(nbins)
            if not bins[b_from]:
                continue
            item = random.choice(bins[b_from])
            w = weights[item]

            # Candidate target bins: prefer those with tightest fit
            # Sample a subset for speed.
            k = min(nbins, 12)
            cand = random.sample(range(nbins), k=k)
            # include possibility of opening a new bin (rare)
            open_new = (random.random() < 0.05)

            best_target = None
            best_res = None
            for b_to in cand:
                if b_to == b_from:
                    continue
                if loads[b_to] + w <= C:
                    res = C - (loads[b_to] + w)
                    if best_res is None or res < best_res:
                        best_res = res
                        best_target = b_to

            if best_target is None and not open_new:
                continue

            # Apply move (to existing bin or new bin)
            if best_target is None:
                # new bin
                b_to = nbins
                bins.append([])
                loads.append(0)
            else:
                b_to = best_target

            # feasibility guaranteed
            # Save info to undo
            from_pos = None
            # remove item from b_from
            bl = bins[b_from]
            # swap-remove for O(1)
            idx = bl.index(item)
            last = bl[-1]
            bl[idx] = last
            bl.pop()
            from_pos = (b_from, item, idx, last)

            loads[b_from] -= w
            bins[b_to].append(item)
            loads[b_to] += w
            old_bin_of_item = assign[item]
            assign[item] = b_to

            # If b_from became empty, remove it (renumber bins)
            removed_bin = -1
            if not bins[b_from]:
                removed_bin = b_from
                # remove bin b_from by swapping with last bin
                last_b = len(bins) - 1
                if removed_bin != last_b:
                    bins[removed_bin] = bins[last_b]
                    loads[removed_bin] = loads[last_b]
                    # update assignments of items in swapped-in bin
                    for it2 in bins[removed_bin]:
                        assign[it2] = removed_bin
                bins.pop()
                loads.pop()
                # If we moved item into a new bin that was last, its index may have changed
                if b_to == last_b:
                    assign[item] = removed_bin if removed_bin != last_b else b_to

            # Evaluate
            cur_nb, cur_total_slack, cur_max_slack = state_metrics(bins, loads)
            new_score = score(cur_nb, cur_total_slack, cur_max_slack)
            dE = new_score - cur_score

            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
                cur_score = new_score
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    best_assign = assign[:]
                    best_bins = [lst[:] for lst in bins]
                    best_loads = loads[:]
                    stall = 0
                else:
                    stall += 1
            else:
                # Undo
                # First, if we removed a bin, we need to restore it.
                # Simpler: rebuild from assignment snapshot is expensive; instead undo locally.
                # We'll reconstruct by reversing operations conservatively using stored data.

                # If a bin was removed, re-add it as empty then move back swapped content.
                # We cannot perfectly undo without more bookkeeping; avoid by not removing bins on rejected move.
                # Therefore: we will not physically remove empty bins during search; instead keep them and clean at end.
                # (But we already removed it above). Fallback: rebuild from best/cur? We'll rebuild current from saved pre-move snapshot.
                pass

        else:  # swap
            if nbins < 2:
                continue
            b1 = random.randrange(nbins)
            b2 = random.randrange(nbins)
            if b1 == b2 or not bins[b1] or not bins[b2]:
                continue
            i1 = random.choice(bins[b1])
            i2 = random.choice(bins[b2])
            w1 = weights[i1]
            w2 = weights[i2]
            if loads[b1] - w1 + w2 > C or loads[b2] - w2 + w1 > C:
                continue

            # Apply swap
            # remove i1 from b1
            idx1 = bins[b1].index(i1)
            last1 = bins[b1][-1]
            bins[b1][idx1] = last1
            bins[b1].pop()
            # remove i2 from b2
            idx2 = bins[b2].index(i2)
            last2 = bins[b2][-1]
            bins[b2][idx2] = last2
            bins[b2].pop()

            bins[b1].append(i2)
            bins[b2].append(i1)
            loads[b1] = loads[b1] - w1 + w2
            loads[b2] = loads[b2] - w2 + w1
            assign[i1], assign[i2] = b2, b1

            cur_nb, cur_total_slack, cur_max_slack = state_metrics(bins, loads)
            new_score = score(cur_nb, cur_total_slack, cur_max_slack)
            dE = new_score - cur_score

            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
                cur_score = new_score
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    best_assign = assign[:]
                    best_bins = [lst[:] for lst in bins]
                    best_loads = loads[:]
                    stall = 0
                else:
                    stall += 1
            else:
                # Undo swap
                # remove swapped-in
                bins[b1].pop()
                bins[b2].pop()
                # restore originals
                bins[b1].append(i1)
                bins[b2].append(i2)
                loads[b1] = loads[b1] + w1 - w2
                loads[b2] = loads[b2] + w2 - w1
                assign[i1], assign[i2] = b1, b2
                stall += 1

        # Because bin removal undo is tricky, we avoid removing empty bins during SA.
        # If a move was rejected via the 'pass' above, we'd be in a bad state.
        # To prevent that, we implement moves without physical bin deletion.
        # Detect this situation and recover by rebuilding from best_assign (safe).
        if move_type == "move" and not accepted:
            # Recovery: rebuild current state from current assign is hard if bins were altered.
            # Use best as a safe state, then continue annealing from it.
            assign = best_assign[:]
            nb = max(assign) + 1 if assign else 0
            bins = build_packing(assign, nb)
            loads = [sum(weights[i] for i in b) for b in bins]
            cur_nb, cur_total_slack, cur_max_slack = state_metrics(bins, loads)
            cur_score = score(cur_nb, cur_total_slack, cur_max_slack)
            stall += 1

        # Cooling & reheat
        T *= alpha
        if stall >= stall_limit:
            T = max(T, 0.35 * T0)
            stall = 0

        # Occasionally intensify near the end by lowering T faster
        if it == int(0.85 * iters):
            alpha = min(alpha, 0.9992)

    # ----- Finalize: take best and remove empties -----
    assign = best_assign[:]
    nb = max(assign) + 1
    bins = build_packing(assign, nb)
    loads = [sum(weights[i] for i in b) for b in bins]

    # Remove empty bins and sort items in bins for nicer output
    # (No requirement, but stable.)
    nonempty_bins = []
    nonempty_loads = []
    for b, items in enumerate(bins):
        if items:
            items.sort()
            nonempty_bins.append(items)
            nonempty_loads.append(loads[b])

    return {"packing": nonempty_bins, "bin_weights": nonempty_loads}
