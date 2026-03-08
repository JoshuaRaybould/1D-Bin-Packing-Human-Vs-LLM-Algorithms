import time
import random
import math
from typing import List, Dict


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    # ------------------------
    # Basic checks / setup
    # ------------------------
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    w = weights

    # If any item exceeds capacity, still return something deterministic (each in its own bin).
    # (Classic bin packing assumes all items <= C; handle gracefully.)
    for wi in w:
        if wi > C:
            packing = [[i] for i in range(n)]
            bin_weights = [w[i] for i in range(n)]
            return {"packing": packing, "bin_weights": bin_weights}

    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    # Order items (offline): decreasing weight tends to help construction.
    order = sorted(range(n), key=lambda i: w[i], reverse=True)

    # Lower bound for bins.
    total_w = sum(w)
    lb = (total_w + C - 1) // C

    # Upper bound: First-Fit Decreasing for initial solution.
    def first_fit_decreasing():
        bins = []  # list of (remaining, items)
        for i in order:
            placed = False
            wi = w[i]
            for b in range(len(bins)):
                rem, items = bins[b]
                if rem >= wi:
                    bins[b] = (rem - wi, items + [i])
                    placed = True
                    break
            if not placed:
                bins.append((C - wi, [i]))
        packing = [items for _, items in bins]
        bin_weights = [sum(w[i] for i in items) for items in packing]
        return packing, bin_weights

    best_packing, best_bin_weights = first_fit_decreasing()
    best_k = len(best_packing)

    # Max bins we model in pheromone slots.
    # Keep it close to best and lower bound; include some slack.
    max_bins = min(n, max(best_k, lb) + 6)

    # ------------------------
    # ACO parameters
    # ------------------------
    # Number of ants per iteration (small but effective).
    ants = 18 if n <= 250 else 14 if n <= 600 else 10

    # Iterations: fixed cap, but will stop on time.
    # Use a cap that scales with n; still checks time.
    iter_cap = 400 if n <= 120 else 260 if n <= 300 else 180 if n <= 700 else 120

    alpha = 1.0  # pheromone importance
    beta = 3.0   # heuristic importance
    rho = 0.18   # evaporation rate

    # Encourage early bins (often yields fewer bins).
    # Used as a mild prior in heuristic.
    pos_bias = [1.0 / (1.0 + 0.06 * j) for j in range(max_bins)]

    # Candidate list size for feasible existing bins.
    cand_k = 7 if n <= 400 else 6

    # Pheromone matrix: tau[item][bin_slot]
    # Initialize moderately.
    tau0 = 1.0
    tau = [[tau0 for _ in range(max_bins)] for _ in range(n)]

    # Helper: compute solution score (lower is better)
    def solution_key(packing, bin_weights):
        k = len(packing)
        leftover = k * C - sum(bin_weights)
        # Primary: bins, secondary: leftover
        return (k, leftover)

    best_key = solution_key(best_packing, best_bin_weights)

    # ------------------------
    # Construction (one ant)
    # ------------------------
    def construct_solution():
        # bins_rem[j] remaining capacity for bin j
        bins_rem: List[int] = []
        bins_items: List[List[int]] = []

        for idx_in_seq, item in enumerate(order):
            wi = w[item]

            # Determine feasible existing bins
            feas = []
            for j, rem in enumerate(bins_rem):
                if rem >= wi:
                    # tight fit heuristic: prefer small remainder after placement
                    after = rem - wi
                    # heuristic component: tighter is better -> 1/(1+after)
                    eta_fit = 1.0 / (1.0 + after)
                    # mild preference for earlier bins
                    eta_pos = pos_bias[j] if j < max_bins else 0.5
                    # combine
                    eta = eta_fit * (0.85 + 0.15 * eta_pos)
                    feas.append((j, after, eta))

            # Sort feasible bins by best fit (smallest after)
            feas.sort(key=lambda t: t[1])

            # Candidate restriction
            cand = feas[:cand_k]

            # Include an option to open a new bin if we can.
            # This is treated as bin_slot = len(bins_rem) if within max_bins.
            open_new_allowed = len(bins_rem) < max_bins

            # Build roulette wheel over candidates and possibly open-new.
            choices = []  # (action_type, j, weight)
            total = 0.0

            for (j, after, eta) in cand:
                t = tau[item][j] if j < max_bins else tau0
                val = (t ** alpha) * (eta ** beta)
                if val > 0.0:
                    choices.append((0, j, val))
                    total += val

            if open_new_allowed:
                jnew = len(bins_rem)
                # Heuristic for opening new bin: generally worse than placing in existing.
                # However, if no feasible existing bins, it must be chosen.
                # Use a bin-reuse penalty via small eta.
                after = C - wi
                eta_fit = 1.0 / (1.0 + after)
                # Penalize opening new bin.
                reuse_penalty = 0.30
                eta = eta_fit * reuse_penalty * (0.9 + 0.1 * pos_bias[jnew])
                t = tau[item][jnew]
                val = (t ** alpha) * (eta ** beta)
                # If no existing feasible bins, force open-new by giving it positive mass.
                if not choices:
                    val = max(val, 1e-6)
                if val > 0.0:
                    choices.append((1, jnew, val))
                    total += val

            # If still no choice (shouldn't happen), open a new bin as fallback.
            if not choices:
                bins_rem.append(C - wi)
                bins_items.append([item])
                continue

            # Roulette selection
            r = random.random() * total
            acc = 0.0
            chosen_type = None
            chosen_j = None
            for typ, j, val in choices:
                acc += val
                if acc >= r:
                    chosen_type = typ
                    chosen_j = j
                    break
            if chosen_type is None:
                chosen_type, chosen_j = choices[-1][0], choices[-1][1]

            if chosen_type == 1:
                # open new
                bins_rem.append(C - wi)
                bins_items.append([item])
            else:
                # place into existing bin
                bins_rem[chosen_j] -= wi
                bins_items[chosen_j].append(item)

        # Prepare output
        packing = bins_items
        bin_weights = [C - rem for rem in bins_rem]
        return packing, bin_weights

    # ------------------------
    # Pheromone update
    # ------------------------
    def evaporate():
        keep = 1.0 - rho
        for i in range(n):
            row = tau[i]
            for j in range(max_bins):
                row[j] *= keep
                # Avoid pheromone vanishing
                if row[j] < 1e-6:
                    row[j] = 1e-6

    def deposit(packing, bin_weights, strength):
        # Deposit based on item->bin_slot assignment
        for j, items in enumerate(packing):
            if j >= max_bins:
                break
            for item in items:
                tau[item][j] += strength

    # Strength schedule
    # Use inverse of (bins - lb + 1) and also reward better fill.
    def compute_strength(packing, bin_weights):
        k = len(packing)
        # quality term: closer to lower bound -> larger
        gap = max(0, k - lb)
        q1 = 1.0 / (1.0 + gap)
        # fill term
        leftover = k * C - sum(bin_weights)
        q2 = 1.0 / (1.0 + leftover / max(1, C))
        return 2.5 * q1 * (0.6 + 0.4 * q2)

    # ------------------------
    # Main ACO loop
    # ------------------------
    it = 0
    while it < iter_cap:
        # Time check each iteration
        if time.time() >= deadline:
            break

        solutions = []
        iter_best = None
        iter_best_key = None

        for _ in range(ants):
            if time.time() >= deadline:
                break
            p, bw = construct_solution()
            key = solution_key(p, bw)
            solutions.append((p, bw, key))
            if iter_best is None or key < iter_best_key:
                iter_best = (p, bw)
                iter_best_key = key

        # Update global best
        if iter_best is not None and iter_best_key < best_key:
            best_packing, best_bin_weights = iter_best
            best_key = iter_best_key
            best_k = best_key[0]
            # Optionally shrink max_bins if we found much better (keep stable here).

        # Pheromone update: evaporate then deposit from iteration-best and global-best
        evaporate()

        if iter_best is not None:
            p, bw = iter_best
            deposit(p, bw, compute_strength(p, bw))

        # Extra elitist reinforcement of global best
        deposit(best_packing, best_bin_weights, 0.7 * compute_strength(best_packing, best_bin_weights))

        it += 1

    # Normalize output: bins with indices lists and aligned weights
    # Ensure indices are original item indices (they are).
    # Also ensure no empty bins.
    packing = [b for b in best_packing if b]
    bin_weights = [sum(w[i] for i in b) for b in packing]
    return {"packing": packing, "bin_weights": bin_weights}
