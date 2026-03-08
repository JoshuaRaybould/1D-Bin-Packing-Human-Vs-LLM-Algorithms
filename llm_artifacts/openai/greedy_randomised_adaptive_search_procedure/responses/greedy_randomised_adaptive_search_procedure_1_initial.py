import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Basic feasibility check (if any item exceeds capacity, we still pack it alone as best-effort)
    # but classic bin packing assumes all items <= capacity.

    items = list(range(n))

    # ---- Helpers ----
    def time_exceeded() -> bool:
        return (time.time() - start) >= time_limit

    def compute_bins_weight(packing: List[List[int]]) -> List[int]:
        return [sum(weights[i] for i in b) for b in packing]

    def deep_copy_solution(packing: List[List[int]], bin_w: List[int]) -> Tuple[List[List[int]], List[int]]:
        return [b[:] for b in packing], bin_w[:]

    # Score = primary minimize #bins, secondary maximize total fill (or minimize waste)
    def solution_key(packing: List[List[int]], bin_w: List[int]) -> Tuple[int, int]:
        # waste is sum(C - load) over bins; minimize bins first, then waste
        waste = sum(bin_capacity - w for w in bin_w)
        return (len(packing), waste)

    # ---- GRASP Construction ----
    # Parameter defaults chosen for robustness across sizes.
    # alpha controls RCL size by quality threshold.
    def construct(alpha: float) -> Tuple[List[List[int]], List[int]]:
        # Order items with bias toward large first but randomized
        # Use a noisy key: weight + noise*weight
        order = items[:]
        # Larger items first; randomness to diversify
        order.sort(key=lambda i: weights[i] * (1.0 + 0.15 * (random.random() - 0.5)), reverse=True)

        packing: List[List[int]] = []
        bin_w: List[int] = []

        for it in order:
            w = weights[it]
            if w > bin_capacity:
                # Put alone; infeasible instance fallback
                packing.append([it])
                bin_w.append(w)
                continue

            # Evaluate candidate bins by resulting slack (best-fit) and also by current fill
            candidates = []  # (slack_after, -fill_before, bin_index)
            for b_idx, bw in enumerate(bin_w):
                if bw + w <= bin_capacity:
                    slack = bin_capacity - (bw + w)
                    candidates.append((slack, -bw, b_idx))

            if not candidates:
                packing.append([it])
                bin_w.append(w)
                continue

            # Build RCL based on slack quality
            # Lower slack is better. Let min_slack, max_slack over feasible.
            candidates.sort()
            min_slack = candidates[0][0]
            max_slack = candidates[-1][0]
            threshold = min_slack + alpha * (max_slack - min_slack)
            rcl = [c for c in candidates if c[0] <= threshold]

            chosen = random.choice(rcl)
            b = chosen[2]
            packing[b].append(it)
            bin_w[b] += w

        return packing, bin_w

    # ---- Local search (essential GRASP component) ----
    # Aim: reduce number of bins; secondarily reduce waste.
    # Neighborhood: move one item from a bin into another; swap items between bins.
    def local_search(packing: List[List[int]], bin_w: List[int], time_check_period: int = 2000) -> Tuple[List[List[int]], List[int]]:
        # Build item->bin mapping for fast updates
        item_bin = [-1] * n
        for bi, b in enumerate(packing):
            for it in b:
                if it < n:
                    item_bin[it] = bi

        # For speed, we work on indices; bins may become empty; we will compact occasionally.
        def compact():
            nonlocal packing, bin_w
            new_p, new_w = [], []
            remap = {}
            for old_i, b in enumerate(packing):
                if b:
                    remap[old_i] = len(new_p)
                    new_p.append(b)
                    new_w.append(bin_w[old_i])
            packing, bin_w = new_p, new_w
            for it in range(n):
                bi = item_bin[it]
                if bi in remap:
                    item_bin[it] = remap[bi]
                elif bi != -1:
                    item_bin[it] = -1

        # Try to eliminate bins starting from the lightest (often easiest) but also consider heavy bins.
        move_attempts = 0
        improved = True
        while improved:
            improved = False
            if time_exceeded():
                break

            # Recompute bin order each outer loop
            bin_indices = list(range(len(packing)))
            # Try both lightest-first and heaviest-first intermittently
            if random.random() < 0.5:
                bin_indices.sort(key=lambda b: bin_w[b])
            else:
                bin_indices.sort(key=lambda b: bin_w[b], reverse=True)

            for src in bin_indices:
                if time_exceeded():
                    break
                if not packing[src]:
                    continue

                # Attempt to move all items out of src to eliminate it
                # Greedy: move larger items first.
                src_items = packing[src][:]
                src_items.sort(key=lambda i: weights[i], reverse=True)

                # Snapshot changes to allow rollback if fail
                moved = []  # (item, old_bin, new_bin)
                ok = True

                for it in src_items:
                    w = weights[it]

                    # Find best destination bin (best-fit) among other bins
                    best_dest = -1
                    best_slack = None
                    for dst in range(len(packing)):
                        if dst == src or not packing[dst]:
                            continue
                        if bin_w[dst] + w <= bin_capacity:
                            slack = bin_capacity - (bin_w[dst] + w)
                            if best_slack is None or slack < best_slack:
                                best_slack = slack
                                best_dest = dst

                    if best_dest == -1:
                        ok = False
                        break

                    # perform move
                    packing[src].remove(it)
                    bin_w[src] -= w
                    packing[best_dest].append(it)
                    bin_w[best_dest] += w
                    item_bin[it] = best_dest
                    moved.append((it, src, best_dest))

                    move_attempts += 1
                    if move_attempts % time_check_period == 0 and time_exceeded():
                        ok = False
                        break

                if ok:
                    # src eliminated or at least reduced; if empty -> improvement in #bins
                    if not packing[src]:
                        improved = True
                        compact()
                        break
                else:
                    # rollback
                    for it, oldb, newb in reversed(moved):
                        w = weights[it]
                        # undo move
                        packing[newb].remove(it)
                        bin_w[newb] -= w
                        packing[oldb].append(it)
                        bin_w[oldb] += w
                        item_bin[it] = oldb

            if improved:
                continue

            # If no bin elimination, try improving waste via relocations/swaps (bounded)
            # This can enable future eliminations.
            # Limit neighborhood sampling to keep time predictable.
            if time_exceeded():
                break

            B = len(packing)
            if B <= 1:
                break

            # Sample bins/items
            sample_bins = list(range(B))
            random.shuffle(sample_bins)
            sample_bins = sample_bins[: min(B, 12)]

            best_delta = 0
            best_move = None  # ('reloc', it, src, dst) or ('swap', it1, b1, it2, b2)

            # Relocations
            for src in sample_bins:
                if not packing[src]:
                    continue
                for it in packing[src][: min(len(packing[src]), 10)]:
                    w = weights[it]
                    for dst in sample_bins:
                        if dst == src or not packing[dst]:
                            continue
                        if bin_w[dst] + w <= bin_capacity:
                            # Prefer moves that increase fill of dst and decrease waste overall
                            # Waste change: (C-(dst+w))+(C-(src-w)) - ((C-dst)+(C-src)) = -w + w = 0
                            # So use secondary criterion: make one bin tighter (smaller slack max)
                            # We use improvement as reduction of max slack among (src,dst).
                            old = max(bin_capacity - bin_w[src], bin_capacity - bin_w[dst])
                            new = max(bin_capacity - (bin_w[src] - w), bin_capacity - (bin_w[dst] + w))
                            delta = old - new
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('reloc', it, src, dst)

                    move_attempts += 1
                    if move_attempts % time_check_period == 0 and time_exceeded():
                        break
                if time_exceeded():
                    break

            # Swaps
            if not time_exceeded():
                for b1 in sample_bins:
                    if not packing[b1]:
                        continue
                    for b2 in sample_bins:
                        if b2 <= b1 or not packing[b2]:
                            continue
                        # sample items
                        items1 = packing[b1][: min(len(packing[b1]), 8)]
                        items2 = packing[b2][: min(len(packing[b2]), 8)]
                        for it1 in items1:
                            w1 = weights[it1]
                            for it2 in items2:
                                w2 = weights[it2]
                                nb1 = bin_w[b1] - w1 + w2
                                nb2 = bin_w[b2] - w2 + w1
                                if nb1 <= bin_capacity and nb2 <= bin_capacity:
                                    old = max(bin_capacity - bin_w[b1], bin_capacity - bin_w[b2])
                                    new = max(bin_capacity - nb1, bin_capacity - nb2)
                                    delta = old - new
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = ('swap', it1, b1, it2, b2)

                            move_attempts += 1
                            if move_attempts % time_check_period == 0 and time_exceeded():
                                break
                        if time_exceeded():
                            break
                    if time_exceeded():
                        break

            if best_move is not None and best_delta > 0:
                if best_move[0] == 'reloc':
                    _, it, src, dst = best_move
                    w = weights[it]
                    packing[src].remove(it)
                    bin_w[src] -= w
                    packing[dst].append(it)
                    bin_w[dst] += w
                    item_bin[it] = dst
                    if not packing[src]:
                        compact()
                    improved = True
                else:
                    _, it1, b1, it2, b2 = best_move
                    w1, w2 = weights[it1], weights[it2]
                    packing[b1].remove(it1)
                    packing[b2].remove(it2)
                    packing[b1].append(it2)
                    packing[b2].append(it1)
                    bin_w[b1] = bin_w[b1] - w1 + w2
                    bin_w[b2] = bin_w[b2] - w2 + w1
                    item_bin[it1] = b2
                    item_bin[it2] = b1
                    improved = True

        return packing, bin_w

    # ---- Initial best: deterministic-ish First Fit Decreasing ----
    def ffd() -> Tuple[List[List[int]], List[int]]:
        order = items[:]
        order.sort(key=lambda i: weights[i], reverse=True)
        packing: List[List[int]] = []
        bin_w: List[int] = []
        for it in order:
            w = weights[it]
            placed = False
            for b in range(len(packing)):
                if bin_w[b] + w <= bin_capacity:
                    packing[b].append(it)
                    bin_w[b] += w
                    placed = True
                    break
            if not placed:
                packing.append([it])
                bin_w.append(w)
        return packing, bin_w

    best_p, best_w = ffd()
    best_key = solution_key(best_p, best_w)

    # ---- Main GRASP loop ----
    # Fixed iteration budget, but we also obey time_limit.
    # Scale iterations with n somewhat; still fixed chosen here.
    max_iter = 200 + int(30 * (n ** 0.5))

    # Use varying alpha schedule to balance greediness and randomness.
    alphas = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]

    for it in range(max_iter):
        if time_exceeded():
            break

        alpha = alphas[it % len(alphas)]
        p, w = construct(alpha)
        if time_exceeded():
            break
        p, w = local_search(p, w)

        k = solution_key(p, w)
        if k < best_key:
            best_key = k
            best_p, best_w = deep_copy_solution(p, w)

    # Ensure alignment and no empty bins
    final_p = [b for b in best_p if b]
    final_w = [sum(weights[i] for i in b) for b in final_p]

    return {"packing": final_p, "bin_weights": final_w}
