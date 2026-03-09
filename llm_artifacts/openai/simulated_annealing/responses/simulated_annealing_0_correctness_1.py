import time
import math
import random
from typing import List


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.perf_counter()
    deadline = start + max(0.0, float(time_limit))

    def now() -> float:
        return time.perf_counter()

    # ---------------- Scoring ----------------
    # Primary objective: minimize number of non-empty bins.
    # Secondary: minimize total slack, and mildly penalize very empty bins.
    def score(nonempty_bins: int, total_slack: int, max_slack: int) -> float:
        return nonempty_bins * 1_000_000.0 + total_slack * 1.0 + 0.05 * max_slack

    # ---------------- Build / metrics ----------------
    def build_bins_from_assign(assign: List[int], nbins: int) -> List[List[int]]:
        bins = [[] for _ in range(nbins)]
        for i, b in enumerate(assign):
            bins[b].append(i)
        return bins

    def compute_loads(bins: List[List[int]]) -> List[int]:
        return [sum(weights[i] for i in b) for b in bins]

    def metrics(bins: List[List[int]], loads: List[int]):
        slacks = [C - L for L in loads]
        total_slack = sum(slacks)
        max_slack = max(slacks) if slacks else 0
        nonempty = sum(1 for b in bins if b)
        return nonempty, total_slack, max_slack

    # ---------------- Initial solution: BFD with randomized tie breaking ----------------
    items = list(range(n))
    items.sort(key=lambda i: weights[i], reverse=True)

    assign = [-1] * n
    bins: List[List[int]] = []
    loads: List[int] = []

    for it in items:
        w = weights[it]
        best_b = -1
        best_res = None
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

    cur_nonempty, cur_total_slack, cur_max_slack = metrics(bins, loads)
    cur_score = score(cur_nonempty, cur_total_slack, cur_max_slack)

    best_assign = assign[:]
    best_bins = [b[:] for b in bins]
    best_loads = loads[:]
    best_score = cur_score

    # ---------------- SA temperature initialization ----------------
    def sample_positive_deltas(samples: int = 80) -> float:
        nbins = len(bins)
        if nbins <= 1:
            return 10.0
        deltas = []
        for _ in range(samples):
            b_from = random.randrange(nbins)
            if not bins[b_from]:
                continue
            item = random.choice(bins[b_from])
            w = weights[item]

            # attempt a feasible move to another bin
            targets = list(range(nbins))
            random.shuffle(targets)
            for b_to in targets:
                if b_to == b_from:
                    continue
                if loads[b_to] + w <= C:
                    # compute exact delta using local recomputation of metrics components
                    # (nbins fixed, we do NOT delete bins during SA)
                    old_slack_from = C - loads[b_from]
                    old_slack_to = C - loads[b_to]
                    new_slack_from = C - (loads[b_from] - w)
                    new_slack_to = C - (loads[b_to] + w)

                    total_slack2 = cur_total_slack - old_slack_from - old_slack_to + new_slack_from + new_slack_to

                    # max_slack recompute conservatively by looking at changed bins and old max
                    # (approx is fine for temperature)
                    max_slack2 = max(cur_max_slack, new_slack_from, new_slack_to)

                    nonempty2 = cur_nonempty
                    if len(bins[b_from]) == 1:
                        # would make b_from empty
                        nonempty2 -= 1
                    if len(bins[b_to]) == 0:
                        nonempty2 += 1

                    s2 = score(nonempty2, total_slack2, max_slack2)
                    deltas.append(s2 - cur_score)
                    break
        pos = [d for d in deltas if d > 1e-9]
        if not pos:
            return 10.0
        return sum(pos) / len(pos)

    avg_pos = sample_positive_deltas(100)
    T0 = max(1e-6, avg_pos / max(1e-9, -math.log(0.7)))
    T = T0

    alpha = 0.9995
    stall = 0
    stall_limit = 6000

    # Use time budget: choose a *fixed* very large iteration cap, but stop at deadline.
    # This satisfies “fixed number of iterations of your choosing” while ensuring
    # we don't terminate early when time remains.
    max_iters = 50_000_000
    check_every = 400

    # ---------------- SA main loop ----------------
    for it in range(max_iters):
        if (it % check_every) == 0 and now() >= deadline:
            break

        nbins = len(bins)
        if nbins == 0:
            break

        # choose move type
        move_type = "swap" if (nbins >= 2 and random.random() < 0.35) else "move"

        if move_type == "move":
            b_from = random.randrange(nbins)
            if not bins[b_from]:
                stall += 1
                T *= alpha
                continue

            item = random.choice(bins[b_from])
            w = weights[item]

            # choose candidate target bins (including possibly empty bins)
            # prefer tight fit among sampled bins
            k = min(nbins, 14)
            cand = random.sample(range(nbins), k=k)
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

            if best_target is None:
                stall += 1
                T *= alpha
                continue

            b_to = best_target

            # Apply move (NO bin deletion; undo is easy)
            # remove item from b_from by swap-remove
            bl_from = bins[b_from]
            idx = bl_from.index(item)
            last_item = bl_from[-1]
            bl_from[idx] = last_item
            bl_from.pop()

            old_load_from = loads[b_from]
            old_load_to = loads[b_to]
            old_nonempty = cur_nonempty

            loads[b_from] -= w
            bins[b_to].append(item)
            loads[b_to] += w
            old_bin = assign[item]
            assign[item] = b_to

            # update metrics incrementally
            old_slack_from = C - old_load_from
            old_slack_to = C - old_load_to
            new_slack_from = C - loads[b_from]
            new_slack_to = C - loads[b_to]

            new_total_slack = cur_total_slack - old_slack_from - old_slack_to + new_slack_from + new_slack_to

            new_nonempty = old_nonempty
            if old_load_to == 0:
                new_nonempty += 1
            if loads[b_from] == 0:
                new_nonempty -= 1

            # max slack recompute cheaply with occasional full recompute
            # (full recompute is still O(#bins), but done rarely)
            if random.random() < 0.02:
                new_nonempty2, new_total_slack2, new_max_slack2 = metrics(bins, loads)
                new_nonempty, new_total_slack, new_max_slack = new_nonempty2, new_total_slack2, new_max_slack2
            else:
                new_max_slack = max(cur_max_slack, new_slack_from, new_slack_to)

            new_score = score(new_nonempty, new_total_slack, new_max_slack)
            dE = new_score - cur_score

            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                # accept
                cur_score = new_score
                cur_nonempty, cur_total_slack, cur_max_slack = new_nonempty, new_total_slack, new_max_slack

                if new_score < best_score - 1e-9:
                    best_score = new_score
                    best_assign = assign[:]
                    best_bins = [b[:] for b in bins]
                    best_loads = loads[:]
                    stall = 0
                else:
                    stall += 1
            else:
                # undo
                bins[b_to].pop()
                loads[b_to] = old_load_to

                bins[b_from].append(item)
                loads[b_from] = old_load_from

                assign[item] = old_bin

                # restore metrics
                stall += 1

        else:  # swap
            if nbins < 2:
                stall += 1
                T *= alpha
                continue

            b1 = random.randrange(nbins)
            b2 = random.randrange(nbins)
            if b1 == b2 or not bins[b1] or not bins[b2]:
                stall += 1
                T *= alpha
                continue

            i1 = random.choice(bins[b1])
            i2 = random.choice(bins[b2])
            w1 = weights[i1]
            w2 = weights[i2]

            if loads[b1] - w1 + w2 > C or loads[b2] - w2 + w1 > C:
                stall += 1
                T *= alpha
                continue

            # Apply swap with undo info
            idx1 = bins[b1].index(i1)
            last1 = bins[b1][-1]
            bins[b1][idx1] = last1
            bins[b1].pop()

            idx2 = bins[b2].index(i2)
            last2 = bins[b2][-1]
            bins[b2][idx2] = last2
            bins[b2].pop()

            old_load1 = loads[b1]
            old_load2 = loads[b2]

            bins[b1].append(i2)
            bins[b2].append(i1)
            loads[b1] = old_load1 - w1 + w2
            loads[b2] = old_load2 - w2 + w1
            assign[i1], assign[i2] = b2, b1

            # Incremental total slack (nonempty unchanged because both bins were nonempty)
            old_slack1 = C - old_load1
            old_slack2 = C - old_load2
            new_slack1 = C - loads[b1]
            new_slack2 = C - loads[b2]
            new_total_slack = cur_total_slack - old_slack1 - old_slack2 + new_slack1 + new_slack2

            if random.random() < 0.02:
                ne, ts, ms = metrics(bins, loads)
                new_nonempty, new_total_slack, new_max_slack = ne, ts, ms
            else:
                new_nonempty = cur_nonempty
                new_max_slack = max(cur_max_slack, new_slack1, new_slack2)

            new_score = score(new_nonempty, new_total_slack, new_max_slack)
            dE = new_score - cur_score

            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                cur_score = new_score
                cur_nonempty, cur_total_slack, cur_max_slack = new_nonempty, new_total_slack, new_max_slack

                if new_score < best_score - 1e-9:
                    best_score = new_score
                    best_assign = assign[:]
                    best_bins = [b[:] for b in bins]
                    best_loads = loads[:]
                    stall = 0
                else:
                    stall += 1
            else:
                # undo swap
                bins[b1].pop()
                bins[b2].pop()

                bins[b1].append(i1)
                bins[b2].append(i2)

                loads[b1] = old_load1
                loads[b2] = old_load2
                assign[i1], assign[i2] = b1, b2
                stall += 1

        # Cooling / reheat
        T *= alpha
        if stall >= stall_limit:
            T = max(T, 0.35 * T0)
            stall = 0

        if it == 2_000_000:
            alpha = min(alpha, 0.9993)

    # ---------------- Finalize: best solution, remove empty bins ----------------
    # Rebuild from best_bins/best_loads (already consistent), then compact.
    compact_bins: List[List[int]] = []
    compact_loads: List[int] = []
    for b, L in zip(best_bins, best_loads):
        if b:
            b_sorted = sorted(b)
            compact_bins.append(b_sorted)
            compact_loads.append(L)

    return {"packing": compact_bins, "bin_weights": compact_loads}
