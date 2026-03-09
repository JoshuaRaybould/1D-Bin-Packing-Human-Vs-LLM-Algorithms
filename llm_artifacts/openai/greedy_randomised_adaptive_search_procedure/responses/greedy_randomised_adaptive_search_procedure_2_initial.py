import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Basic feasibility check: if any item exceeds capacity, cannot pack.
    # Here we still return each such item in its own "bin" (violating capacity) is not acceptable.
    # Better: raise? The prompt doesn't specify; assume inputs are feasible.
    # We'll handle gracefully by putting in separate bins (still overweight) to avoid crash.

    # Precompute items sorted by weight descending (ties broken randomly for diversification).
    items = list(range(n))
    items.sort(key=lambda i: (weights[i], random.random()), reverse=True)

    total_weight = sum(weights)

    def solution_score(bin_weights: List[int]) -> Tuple[int, int]:
        # Primary: minimize bins.
        # Secondary: minimize total waste (equivalently maximize utilization).
        # Return (bins, waste)
        b = len(bin_weights)
        waste = b * bin_capacity - sum(bin_weights)
        return (b, waste)

    def construct(alpha: float) -> Tuple[List[List[int]], List[int]]:
        # GRASP construction using RCL based on best-fit residual.
        packing: List[List[int]] = []
        bw: List[int] = []
        rem: List[int] = []

        for it in items:
            w = weights[it]
            # Find feasible bins and compute residual after placement.
            feasible = []  # (residual_after, bin_index)
            for b in range(len(rem)):
                if rem[b] >= w:
                    feasible.append((rem[b] - w, b))

            if not feasible:
                packing.append([it])
                bw.append(w)
                rem.append(bin_capacity - w)
                continue

            # Best-fit: smaller residual_after is better.
            feasible.sort(key=lambda x: x[0])
            best = feasible[0][0]
            worst = feasible[-1][0]
            # Threshold per GRASP: allow residual <= best + alpha*(worst-best)
            thr = best + alpha * (worst - best)
            rcl = [b for (resid, b) in feasible if resid <= thr]
            chosen = random.choice(rcl)

            packing[chosen].append(it)
            bw[chosen] += w
            rem[chosen] -= w

        return packing, bw

    def local_search(packing: List[List[int]], bw: List[int]) -> Tuple[List[List[int]], List[int]]:
        # Standard GRASP local search: try to reduce number of bins by relocating items
        # and simple swaps. First-improvement with randomization.
        #
        # Representation invariant: bw[b] == sum(weights[i] for i in packing[b])

        def try_relocate() -> bool:
            # Attempt to move a single item from one bin to another.
            # Bias: try to empty light bins first.
            order_bins = list(range(len(packing)))
            order_bins.sort(key=lambda b: bw[b])

            for b_from in order_bins:
                if not packing[b_from]:
                    continue
                # Randomize items within bin (small bins first helps emptying)
                items_from = packing[b_from][:]
                random.shuffle(items_from)
                for it in items_from:
                    w = weights[it]
                    # Try to place into other bins (best-fit target)
                    targets = []
                    for b_to in range(len(packing)):
                        if b_to == b_from:
                            continue
                        if bw[b_to] + w <= bin_capacity:
                            targets.append((bin_capacity - (bw[b_to] + w), b_to))
                    if not targets:
                        continue
                    targets.sort(key=lambda x: x[0])
                    # try a few best targets
                    for _, b_to in targets[:5]:
                        # perform move
                        packing[b_from].remove(it)
                        packing[b_to].append(it)
                        bw[b_from] -= w
                        bw[b_to] += w
                        # remove empty bin
                        if bw[b_from] == 0:
                            packing.pop(b_from)
                            bw.pop(b_from)
                        return True
            return False

        def try_swap() -> bool:
            # Swap items between bins to enable later relocations / emptying.
            # Keep bin count constant; accept if it improves total waste distribution
            # and/or makes a bin empty immediately (rare).
            m = len(packing)
            if m <= 1:
                return False

            # Consider a limited number of random bin pairs
            pairs = []
            for _ in range(min(40, m * 2)):
                a = random.randrange(m)
                b = random.randrange(m)
                if a != b:
                    pairs.append((a, b))

            for a, b in pairs:
                if not packing[a] or not packing[b]:
                    continue
                ia = random.choice(packing[a])
                ib = random.choice(packing[b])
                wa, wb = weights[ia], weights[ib]

                new_wa = bw[a] - wa + wb
                new_wb = bw[b] - wb + wa
                if new_wa <= bin_capacity and new_wb <= bin_capacity:
                    # Accept swap if it reduces sum of residuals squares (tightens packing)
                    ra0 = bin_capacity - bw[a]
                    rb0 = bin_capacity - bw[b]
                    ra1 = bin_capacity - new_wa
                    rb1 = bin_capacity - new_wb
                    if (ra1 * ra1 + rb1 * rb1) < (ra0 * ra0 + rb0 * rb0):
                        # Do swap
                        packing[a].remove(ia)
                        packing[b].remove(ib)
                        packing[a].append(ib)
                        packing[b].append(ia)
                        bw[a] = new_wa
                        bw[b] = new_wb
                        return True
            return False

        # Iterate local improvements with a move budget
        moves = 0
        move_budget = 500 + 10 * len(packing)
        while moves < move_budget:
            moves += 1
            # time check
            if time.time() - start >= time_limit:
                break
            if try_relocate():
                continue
            if try_swap():
                continue
            break
        return packing, bw

    # Initial best: deterministic-ish First Fit Decreasing as baseline.
    def ffd() -> Tuple[List[List[int]], List[int]]:
        pack: List[List[int]] = []
        bw0: List[int] = []
        rem0: List[int] = []
        for it in items:
            w = weights[it]
            placed = False
            best_b = -1
            best_res = None
            for b in range(len(rem0)):
                if rem0[b] >= w:
                    res = rem0[b] - w
                    if best_res is None or res < best_res:
                        best_res = res
                        best_b = b
            if best_b >= 0:
                pack[best_b].append(it)
                bw0[best_b] += w
                rem0[best_b] -= w
                placed = True
            if not placed:
                pack.append([it])
                bw0.append(w)
                rem0.append(bin_capacity - w)
        return pack, bw0

    best_packing, best_bw = ffd()
    best_packing, best_bw = local_search(best_packing, best_bw)
    best_score = solution_score(best_bw)

    # Fixed iteration budget; also stop by time limit.
    # Scale with n but cap to keep predictable.
    max_iter = min(2000, 200 + 10 * int(n ** 0.5))

    # Adaptive alpha schedule: mix greedy and random.
    # GRASP often samples alpha in [0,1].
    alphas = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]

    it = 0
    while it < max_iter:
        it += 1
        if it % 5 == 0 and (time.time() - start) >= time_limit:
            break

        alpha = random.choice(alphas)
        p, bw = construct(alpha)
        p, bw = local_search(p, bw)
        sc = solution_score(bw)

        if sc < best_score:
            best_score = sc
            best_packing, best_bw = p, bw

        # Mild intensification: if close to current best, retry greedier alpha
        if it % 25 == 0 and (time.time() - start) < time_limit:
            p2, bw2 = construct(0.0)
            p2, bw2 = local_search(p2, bw2)
            sc2 = solution_score(bw2)
            if sc2 < best_score:
                best_score = sc2
                best_packing, best_bw = p2, bw2

        # Extra time checks
        if it % 50 == 0 and (time.time() - start) >= time_limit:
            break

    # Ensure bin_weights aligned and correct (recompute to be safe)
    final_bw = [sum(weights[i] for i in b) for b in best_packing]
    return {"packing": best_packing, "bin_weights": final_bw}
