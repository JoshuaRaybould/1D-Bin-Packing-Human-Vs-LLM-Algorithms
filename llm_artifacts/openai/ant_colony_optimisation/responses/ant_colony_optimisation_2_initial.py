import time
import random
import math
from typing import List, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = bin_capacity
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.perf_counter()
    deadline = start + max(0.0, float(time_limit))

    # --- Preprocess: sort items by weight (descending), keep original indices ---
    items = list(range(n))
    items.sort(key=lambda i: weights[i], reverse=True)
    w_sorted = [weights[i] for i in items]

    # Quick bounds for scaling
    total_w = sum(weights)
    lb = (total_w + C - 1) // C

    # --- Parameters (robust defaults) ---
    # Number of ants per iteration and iterations chosen to respect time.
    # We'll loop until time ends or until max_iter reached.
    ants_per_iter = 12 if n <= 200 else 8
    max_iter = 10_000  # hard cap; time limit will stop earlier

    # ACO exponents
    alpha = 1.2  # pheromone
    beta = 2.5   # heuristic

    # Evaporation
    rho = 0.12
    tau_min = 1e-4
    tau_max = 5.0

    # Candidate list sizes
    # For choosing next item: focus on heavier items; add a few random.
    cand_top = min(25, n)
    cand_rand = 10

    # For choosing bin: consider all feasible bins + new bin (bins are usually not huge)

    # --- Pheromone on transitions between items in sorted order space ---
    # We'll index pheromones by positions 0..n-1 in the sorted list.
    # Include a special START node index = n.
    START = n
    tau = [[1.0 for _ in range(n)] for _ in range(n + 1)]  # (prev in 0..n or START) -> next in 0..n-1

    # Heuristic for selecting next item: prefer large items.
    # Use (w/C) plus a small epsilon.
    eta_item = [(w_sorted[i] / C) + 1e-6 for i in range(n)]

    # Helper: build packing from an order of item positions.
    def pack_from_order(order_pos: List[int]) -> tuple[list[list[int]], list[int]]:
        # Best-Fit Decreasing style placement following the provided order
        bin_weights: List[int] = []
        packing_pos: List[List[int]] = []
        for p in order_pos:
            w = w_sorted[p]
            best_bin = -1
            best_residual = None
            for b, bw in enumerate(bin_weights):
                if bw + w <= C:
                    residual = C - (bw + w)
                    if best_residual is None or residual < best_residual:
                        best_residual = residual
                        best_bin = b
                        if residual == 0:
                            break
            if best_bin == -1:
                bin_weights.append(w)
                packing_pos.append([p])
            else:
                bin_weights[best_bin] += w
                packing_pos[best_bin].append(p)
        # Convert positions to original indices
        packing = [[items[p] for p in bin_list] for bin_list in packing_pos]
        return packing, bin_weights

    # Greedy baseline to initialize best
    greedy_order = list(range(n))
    best_packing, best_bin_weights = pack_from_order(greedy_order)
    best_bins = len(best_packing)

    # Objective comparison: primary minimize #bins, secondary maximize fullness / minimize slack.
    def solution_key(bin_weights: List[int]) -> tuple[int, int]:
        # Slack sum as tie-breaker
        slack = sum(C - bw for bw in bin_weights)
        return (len(bin_weights), slack)

    best_key = solution_key(best_bin_weights)

    # --- Ant construction ---
    def construct_solution() -> tuple[list[list[int]], list[int], List[int]]:
        # We will choose next item positions probabilistically, then place each item using a bin-choice rule.
        remaining = set(range(n))
        order: List[int] = []
        prev = START

        # Bins maintained during construction to allow bin-choice heuristic depending on current state.
        bin_weights: List[int] = []
        packing_pos: List[List[int]] = []

        while remaining:
            if time.perf_counter() >= deadline:
                break

            # Candidate list for next item
            cand = []
            # top heavy remaining
            for p in range(cand_top):
                if p in remaining:
                    cand.append(p)
            # add some random remaining positions
            if len(remaining) > len(cand):
                rem_list = None
                # sample without materializing huge list repeatedly
                # but for simplicity, materialize occasionally
                rem_list = list(remaining)
                for _ in range(cand_rand):
                    cand.append(rem_list[random.randrange(len(rem_list))])
            # unique
            if cand:
                cand = list(dict.fromkeys(cand))
            else:
                cand = list(remaining)

            # Compute probabilities for choosing next item
            probs = []
            s = 0.0
            for j in cand:
                tj = tau[prev][j]
                # Heuristic: prefer larger items; also prefer items that fit tightly into some existing bin.
                # Estimate tight-fit potential: best residual if placed into current bins.
                wj = w_sorted[j]
                best_res = None
                for bw in bin_weights:
                    if bw + wj <= C:
                        res = C - (bw + wj)
                        if best_res is None or res < best_res:
                            best_res = res
                            if res == 0:
                                break
                if best_res is None:
                    fit_bonus = 1.0  # may open new bin
                else:
                    # smaller residual => bigger bonus
                    fit_bonus = 1.0 + (1.0 - (best_res / C))

                eta = eta_item[j] * fit_bonus
                val = (tj ** alpha) * (eta ** beta)
                probs.append(val)
                s += val

            # Roulette selection
            if s <= 0.0:
                nxt = random.choice(cand)
            else:
                r = random.random() * s
                acc = 0.0
                nxt = cand[-1]
                for j, v in zip(cand, probs):
                    acc += v
                    if acc >= r:
                        nxt = j
                        break

            # Place the selected item into a bin (probabilistic best-fit)
            w = w_sorted[nxt]
            feasible_bins = []
            scores = []
            total_score = 0.0

            for b, bw in enumerate(bin_weights):
                if bw + w <= C:
                    res = C - (bw + w)
                    # Heuristic strongly favors tighter fits
                    h = 1e-6 + (1.0 - (res / C))
                    # Add mild preference for fuller bins (to reduce bin count)
                    h *= 1.0 + (bw / C)
                    # Use pheromone from prev->nxt to bias placement indirectly, so keep h dominant
                    sc = h
                    feasible_bins.append(b)
                    scores.append(sc)
                    total_score += sc

            # Option to open a new bin
            # Encourage opening new bin only when existing placements are weak.
            open_score = 0.15 + 0.85 * (w / C)

            choose_new = False
            if feasible_bins:
                total = total_score + open_score
                r = random.random() * total
                if r >= total_score:
                    choose_new = True
                else:
                    # roulette among bins
                    rr = r
                    acc = 0.0
                    chosen_b = feasible_bins[-1]
                    for b, sc in zip(feasible_bins, scores):
                        acc += sc
                        if acc >= rr:
                            chosen_b = b
                            break
                    bin_weights[chosen_b] += w
                    packing_pos[chosen_b].append(nxt)
            else:
                choose_new = True

            if choose_new:
                bin_weights.append(w)
                packing_pos.append([nxt])

            # Commit selection
            remaining.remove(nxt)
            order.append(nxt)
            prev = nxt

        # If time cut construction early, pack remaining greedily into current state
        if remaining:
            rem = sorted(remaining, key=lambda p: w_sorted[p], reverse=True)
            for p in rem:
                w = w_sorted[p]
                best_bin = -1
                best_res = None
                for b, bw in enumerate(bin_weights):
                    if bw + w <= C:
                        res = C - (bw + w)
                        if best_res is None or res < best_res:
                            best_res = res
                            best_bin = b
                            if res == 0:
                                break
                if best_bin == -1:
                    bin_weights.append(w)
                    packing_pos.append([p])
                else:
                    bin_weights[best_bin] += w
                    packing_pos[best_bin].append(p)
                order.append(p)

        packing = [[items[p] for p in bin_list] for bin_list in packing_pos]
        return packing, bin_weights, order

    # --- Pheromone update ---
    def evaporate():
        for i in range(n + 1):
            row = tau[i]
            for j in range(n):
                row[j] *= (1.0 - rho)
                if row[j] < tau_min:
                    row[j] = tau_min

    def deposit(order: List[int], quality: float, strength: float = 1.0):
        # Deposit along START->order[0], order[k-1]->order[k]
        if not order:
            return
        add = strength * quality
        prev = START
        for nxt in order:
            v = tau[prev][nxt] + add
            tau[prev][nxt] = tau_max if v > tau_max else v
            prev = nxt

    # Quality function: larger is better. Strongly reward fewer bins.
    def quality_from(bin_weights: List[int]) -> float:
        bins = len(bin_weights)
        slack = sum(C - bw for bw in bin_weights)
        # Normalize slack to [0, n*C]
        # Primary term: inverse bins, secondary: inverse slack.
        return 1.0 / (bins - lb + 1.0) + 0.05 / (1.0 + (slack / C))

    # --- Main loop ---
    it = 0
    while it < max_iter and time.perf_counter() < deadline:
        it += 1

        iter_best_packing = None
        iter_best_bw = None
        iter_best_order = None
        iter_best_key = None

        for _ in range(ants_per_iter):
            if time.perf_counter() >= deadline:
                break
            packing, bw, order = construct_solution()
            k = solution_key(bw)
            if iter_best_key is None or k < iter_best_key:
                iter_best_key = k
                iter_best_packing = packing
                iter_best_bw = bw
                iter_best_order = order

        if iter_best_key is None:
            break

        # Update global best
        if iter_best_key < best_key:
            best_key = iter_best_key
            best_packing = iter_best_packing
            best_bin_weights = iter_best_bw
            best_bins = len(best_packing)
            # Early stop if optimal (hits lower bound)
            if best_bins == lb:
                break

        # Pheromone update
        evaporate()
        q_iter = quality_from(iter_best_bw)
        deposit(iter_best_order, q_iter, strength=1.0)

        # Extra reinforcement for global-best occasionally
        if it % 5 == 0:
            q_best = quality_from(best_bin_weights)
            # stronger deposit
            # Need order of best: rebuild from best packing to a sequence (approximate) using bin concatenation.
            # We'll derive an order by listing items in each bin by descending weight.
            best_positions = []
            for b in best_packing:
                pos = [items.index(i) for i in b]  # O(n^2) if used often; avoid by mapping
                best_positions.extend(sorted(pos, key=lambda p: w_sorted[p], reverse=True))
            # Optimize mapping to avoid O(n^2)
            # (Only executed rarely; still, fix with a precomputed map)
            # We'll rebuild properly below:
            pos_map = {items[p]: p for p in range(n)}
            best_positions = []
            for b in best_packing:
                pos = [pos_map[i] for i in b]
                best_positions.extend(sorted(pos, key=lambda p: w_sorted[p], reverse=True))
            deposit(best_positions, q_best, strength=1.5)

    # Ensure bin_weights aligned with packing and valid totals
    # Recompute bin weights from packing to be safe.
    final_bw = []
    for bin_items in best_packing:
        final_bw.append(sum(weights[i] for i in bin_items))

    return {"packing": best_packing, "bin_weights": final_bw}
