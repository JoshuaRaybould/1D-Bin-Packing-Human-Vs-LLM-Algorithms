# openai
# ant_colony_optimisation_2_post_plan.py

import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.perf_counter()
    deadline = start + max(0.0, float(time_limit))

    # --- 1) Preprocess: sort items by weight desc, keep mapping ---
    pos_to_item = list(range(n))
    pos_to_item.sort(key=lambda i: weights[i], reverse=True)
    w = [weights[i] for i in pos_to_item]
    item_to_pos = [0] * n
    for p, it in enumerate(pos_to_item):
        item_to_pos[it] = p

    total_w = sum(weights)

    # --- 2) Lower bounds: LB1 and stronger LB2 (items > C/2 each need a bin) ---
    lb1 = (total_w + C - 1) // C
    big = 0
    sum_others = 0
    half = C / 2
    for wi in w:
        if wi > half:
            big += 1
        else:
            sum_others += wi
    lb2 = big + ((sum_others + C - 1) // C)
    lb = max(lb1, lb2)

    # --- 3) Strong baseline: efficient Best-Fit Decreasing (BFD) in sorted pos space ---
    def bfd_packing() -> Tuple[List[List[int]], List[int]]:
        # Maintain list of remaining capacities per bin.
        rem_caps: List[int] = []
        bins: List[List[int]] = []
        for p in range(n):
            wp = w[p]
            best_b = -1
            best_rem = None
            for b, rcap in enumerate(rem_caps):
                if wp <= rcap:
                    new_rem = rcap - wp
                    if best_rem is None or new_rem < best_rem:
                        best_rem = new_rem
                        best_b = b
                        if new_rem == 0:
                            break
            if best_b < 0:
                rem_caps.append(C - wp)
                bins.append([p])
            else:
                rem_caps[best_b] -= wp
                bins[best_b].append(p)
        bin_weights = [C - r for r in rem_caps]
        return bins, bin_weights

    best_bins_pos, best_bin_weights = bfd_packing()

    def key_from_binweights(bw: List[int]) -> Tuple[int, int]:
        slack = sum(C - x for x in bw)
        return (len(bw), slack)

    best_key = key_from_binweights(best_bin_weights)

    # --- 4) ACO parameters / budgeting (fixed max_iter) ---
    if n <= 200:
        ants_per_iter = 24
    elif n <= 800:
        ants_per_iter = 16
    else:
        ants_per_iter = 10

    max_iter = 50_000

    # Exponents
    alpha = 1.4
    beta = 3.0
    alpha_open = 1.0
    beta_seed = 2.0
    gamma_seed = 1.0

    # ACS exploitation parameter (can increase mildly with iterations)
    q0_start = 0.60
    q0_end = 0.90

    # Evaporation
    rho = 0.10

    # Candidate list sizes
    K_fit = 30
    M_rand = 20
    K_seed = 30
    M_seed_rand = 10

    # Deposit control
    elite_every = 5
    stagnation_S = 200

    # --- 5) Pheromone structures: tau_pair (compatibility) + tau_open (seed desirability) ---
    # Full n x n matrix; tau_pair[i][j] used for i<j (keep symmetric).
    tau_init = 1.0
    tau_pair = [[tau_init] * n for _ in range(n)]
    tau_open = [tau_init] * n

    # MMAS-ish bounds (robust variant)
    # tau_max depends on current best gap; tau_min = tau_max / A
    A_bound = 100.0

    def compute_tau_bounds(best_bins: int) -> Tuple[float, float]:
        gap = max(0, best_bins - lb)
        # Larger when close to LB; smaller when far.
        tau_max = 1.0 / (rho * (1.0 + gap))
        tau_min = tau_max / A_bound
        return tau_min, tau_max

    tau_min, tau_max = compute_tau_bounds(best_key[0])

    # Clamp helper
    def clamp(v: float) -> float:
        if v < tau_min:
            return tau_min
        if v > tau_max:
            return tau_max
        return v

    # --- 6) Fast remainder look-ahead support: counts by weight ---
    # Note: weights are integers.
    def build_count_by_weight(unpacked: List[bool]) -> Dict[int, int]:
        d: Dict[int, int] = {}
        for p in range(n):
            if unpacked[p]:
                wp = w[p]
                d[wp] = d.get(wp, 0) + 1
        return d

    # Candidate builder: top-K heaviest that fit + M random that fit
    def candidates_that_fit(unpacked: List[bool], remaining_positions: List[int], R: int) -> List[int]:
        cand: List[int] = []
        # top-K heaviest that fit (scan sorted positions)
        for p in range(n):
            if len(cand) >= K_fit:
                break
            if unpacked[p] and w[p] <= R:
                cand.append(p)
        if not remaining_positions:
            return cand
        # random samples that fit
        tries = 0
        while tries < M_rand and len(cand) < (K_fit + M_rand):
            tries += 1
            p = remaining_positions[random.randrange(len(remaining_positions))]
            if unpacked[p] and w[p] <= R:
                cand.append(p)
        # unique preserve order
        if len(cand) > 1:
            cand = list(dict.fromkeys(cand))
        return cand

    # Seed candidates: top-K remaining + some random
    def seed_candidates(unpacked: List[bool], remaining_positions: List[int]) -> List[int]:
        cand: List[int] = []
        for p in range(n):
            if len(cand) >= K_seed:
                break
            if unpacked[p]:
                cand.append(p)
        if remaining_positions:
            tries = 0
            while tries < M_seed_rand and len(cand) < (K_seed + M_seed_rand):
                tries += 1
                p = remaining_positions[random.randrange(len(remaining_positions))]
                if unpacked[p]:
                    cand.append(p)
        if len(cand) > 1:
            cand = list(dict.fromkeys(cand))
        return cand

    # Average pheromone between candidate j and current bin items B
    def avg_pheromone_with_bin(j: int, B: List[int]) -> float:
        if not B:
            return 1.0
        s = 0.0
        for i in B:
            if i < j:
                s += tau_pair[i][j]
            else:
                s += tau_pair[j][i]
        return s / len(B)

    # Approximate compatibility score for seed: average of top few pheromones to other unpacked items
    def seed_pher_preview(i: int, unpacked: List[bool], preview: int = 12) -> float:
        # scan a small prefix among heavy unpacked items for speed
        vals: List[float] = []
        cnt = 0
        for j in range(n):
            if cnt >= preview:
                break
            if j == i or (not unpacked[j]):
                continue
            v = tau_pair[i][j] if i < j else tau_pair[j][i]
            vals.append(v)
            cnt += 1
        if not vals:
            return 1.0
        vals.sort(reverse=True)
        # average top min(4, len)
        t = min(4, len(vals))
        return sum(vals[:t]) / t

    # --- 7) Constructive ant: bin-by-bin filling ---
    def construct_solution(iter_frac: float) -> Tuple[List[List[int]], List[int]]:
        # Data structures for fast removals
        unpacked = [True] * n
        remaining_positions = list(range(n))
        pos_in_remaining = list(range(n))

        def remove_pos(p: int) -> None:
            # swap-pop from remaining_positions
            idx = pos_in_remaining[p]
            last = remaining_positions[-1]
            remaining_positions[idx] = last
            pos_in_remaining[last] = idx
            remaining_positions.pop()

        count_by_weight = build_count_by_weight(unpacked)

        bins: List[List[int]] = []
        bin_weights: List[int] = []

        # time checks: per bin and every 32 insertions
        inserts = 0

        q0 = q0_start + (q0_end - q0_start) * iter_frac

        left = n
        while left > 0:
            if time.perf_counter() >= deadline:
                break

            # --- choose seed item for new bin ---
            seeds = seed_candidates(unpacked, remaining_positions)
            # exploitation/exploration using roulette on seed score
            best_seed = seeds[0]
            best_score = -1.0
            seed_scores: List[float] = []
            ssum = 0.0
            for i in seeds:
                # prefer large items + tau_open + some compatibility preview
                preview = seed_pher_preview(i, unpacked)
                h = (w[i] / C) + 1e-9
                score = (tau_open[i] ** alpha_open) * (h ** beta_seed) * (preview ** gamma_seed)
                seed_scores.append(score)
                ssum += score
                if score > best_score:
                    best_score = score
                    best_seed = i

            if (random.random() < q0) or ssum <= 0.0:
                seed = best_seed
            else:
                r = random.random() * ssum
                acc = 0.0
                seed = seeds[-1]
                for i, sc in zip(seeds, seed_scores):
                    acc += sc
                    if acc >= r:
                        seed = i
                        break

            # start bin
            B: List[int] = [seed]
            bw = w[seed]
            R = C - bw

            unpacked[seed] = False
            remove_pos(seed)
            count_by_weight[w[seed]] -= 1
            if count_by_weight[w[seed]] == 0:
                del count_by_weight[w[seed]]
            left -= 1

            # --- fill bin iteratively ---
            while R > 0 and left > 0:
                inserts += 1
                if (inserts & 31) == 0 and time.perf_counter() >= deadline:
                    break

                # Deterministic closure: if exact fit exists, take it.
                if R in count_by_weight and count_by_weight[R] > 0:
                    # pick some unpacked position with weight R; scan prefix for speed
                    chosen = None
                    for p in range(n):
                        if unpacked[p] and w[p] == R:
                            chosen = p
                            break
                    if chosen is not None:
                        B.append(chosen)
                        bw += w[chosen]
                        R = 0
                        unpacked[chosen] = False
                        remove_pos(chosen)
                        count_by_weight[w[chosen]] -= 1
                        if count_by_weight[w[chosen]] == 0:
                            del count_by_weight[w[chosen]]
                        left -= 1
                        break

                cand = candidates_that_fit(unpacked, remaining_positions, R)
                if not cand:
                    break

                # compute scores
                best_j = cand[0]
                best_sc = -1.0
                scores: List[float] = []
                ssum = 0.0

                for j in cand:
                    # pheromone compatibility with current bin
                    pher = avg_pheromone_with_bin(j, B)

                    # heuristic: tight fill wrt remaining capacity
                    hj = (w[j] / R) + 1e-9

                    # look-ahead: reward leaving a remainder that is exactly fillable
                    rem = R - w[j]
                    look = 1.0
                    if rem > 0 and (rem in count_by_weight):
                        look = 1.35

                    sc = (pher ** alpha) * ((hj ** beta) * look)
                    scores.append(sc)
                    ssum += sc
                    if sc > best_sc:
                        best_sc = sc
                        best_j = j

                if (random.random() < q0) or ssum <= 0.0:
                    chosen = best_j
                else:
                    r = random.random() * ssum
                    acc = 0.0
                    chosen = cand[-1]
                    for j, sc in zip(cand, scores):
                        acc += sc
                        if acc >= r:
                            chosen = j
                            break

                # add chosen
                B.append(chosen)
                bw += w[chosen]
                R -= w[chosen]
                unpacked[chosen] = False
                remove_pos(chosen)
                count_by_weight[w[chosen]] -= 1
                if count_by_weight[w[chosen]] == 0:
                    del count_by_weight[w[chosen]]
                left -= 1

            bins.append(B)
            bin_weights.append(bw)

        # If time cut short, pack remaining with greedy BFD into new bins (safe fallback)
        if left > 0:
            # collect remaining positions in descending weight order (already sorted by pos)
            rem_pos = [p for p in range(n) if unpacked[p]]
            # place each into best-fit among existing bins; else open new
            rem_caps = [C - bw for bw in bin_weights]
            for p in rem_pos:
                wp = w[p]
                best_b = -1
                best_rem = None
                for b, rcap in enumerate(rem_caps):
                    if wp <= rcap:
                        new_rem = rcap - wp
                        if best_rem is None or new_rem < best_rem:
                            best_rem = new_rem
                            best_b = b
                            if new_rem == 0:
                                break
                if best_b < 0:
                    bins.append([p])
                    bin_weights.append(wp)
                    rem_caps.append(C - wp)
                else:
                    bins[best_b].append(p)
                    bin_weights[best_b] += wp
                    rem_caps[best_b] -= wp

        return bins, bin_weights

    # --- 8) MMAS evaporation / deposit with pairwise reinforcement ---
    def evaporate() -> None:
        mult = 1.0 - rho
        for i in range(n):
            row = tau_pair[i]
            # start j at 0; keep symmetric matrix maintained by updates
            for j in range(n):
                row[j] = clamp(row[j] * mult)
        for i in range(n):
            tau_open[i] = clamp(tau_open[i] * mult)

    def deposit_from_solution(bins_pos: List[List[int]], bw: List[int], strength: float) -> None:
        bins_cnt = len(bw)
        slack = sum(C - x for x in bw)
        # scalarized objective (smaller better)
        f = bins_cnt + (slack / (C * max(1, n)))

        gap = max(0, bins_cnt - lb)
        # Strong separation near LB
        delta = strength * (1.0 / ((1.0 + gap) ** 2.2)) * (1.0 / max(1e-9, f))

        # Deposit within bins on pairs; cap work for large bins
        m_cap = 25
        heavy_t = 12

        for B in bins_pos:
            if not B:
                continue
            # deposit on seed
            seed = B[0]
            tau_open[seed] = clamp(tau_open[seed] + delta)

            m = len(B)
            if m <= 1:
                continue

            if m <= m_cap:
                for a in range(m - 1):
                    i = B[a]
                    rowi = tau_pair[i]
                    for b in range(a + 1, m):
                        j = B[b]
                        # symmetric update
                        if i < j:
                            v = rowi[j] + delta
                            v = tau_max if v > tau_max else v
                            rowi[j] = v
                            tau_pair[j][i] = v
                        else:
                            v = tau_pair[j][i] + delta
                            v = tau_max if v > tau_max else v
                            tau_pair[j][i] = v
                            rowi[j] = v
            else:
                # only deposit among heaviest heavy_t items in the bin
                # (heaviest items have smaller position index)
                B_sorted = sorted(B)[:heavy_t]
                mm = len(B_sorted)
                for a in range(mm - 1):
                    i = B_sorted[a]
                    rowi = tau_pair[i]
                    for b in range(a + 1, mm):
                        j = B_sorted[b]
                        if i < j:
                            v = rowi[j] + delta
                            v = tau_max if v > tau_max else v
                            rowi[j] = v
                            tau_pair[j][i] = v
                        else:
                            v = tau_pair[j][i] + delta
                            v = tau_max if v > tau_max else v
                            tau_pair[j][i] = v
                            rowi[j] = v

    # Stagnation reset
    def partial_reset() -> None:
        for i in range(n):
            row = tau_pair[i]
            for j in range(n):
                row[j] = 0.5 * row[j] + 0.5 * tau_init
            tau_open[i] = 0.5 * tau_open[i] + 0.5 * tau_init

    # --- 9) Main loop ---
    best_no_improve = 0

    for it in range(1, max_iter + 1):
        if time.perf_counter() >= deadline:
            break

        iter_best_bins = None
        iter_best_bw = None
        iter_best_key = None

        iter_frac = (it - 1) / max(1, (max_iter - 1))

        for _ in range(ants_per_iter):
            if time.perf_counter() >= deadline:
                break
            bins_pos, bw = construct_solution(iter_frac)
            k = key_from_binweights(bw)
            if iter_best_key is None or k < iter_best_key:
                iter_best_key = k
                iter_best_bins = bins_pos
                iter_best_bw = bw

        if iter_best_key is None:
            break

        # Update global best
        if iter_best_key < best_key:
            best_key = iter_best_key
            best_bins_pos = iter_best_bins
            best_bin_weights = iter_best_bw
            best_no_improve = 0
            # Update bounds when best improves
            tau_min, tau_max = compute_tau_bounds(best_key[0])
            if best_key[0] == lb:
                # reached lower bound
                break
        else:
            best_no_improve += 1

        # MMAS evaporation then deposits (iter-best always, global-best periodically)
        evaporate()
        deposit_from_solution(iter_best_bins, iter_best_bw, strength=1.0)
        if (it % elite_every) == 0:
            deposit_from_solution(best_bins_pos, best_bin_weights, strength=1.8)

        if best_no_improve >= stagnation_S:
            partial_reset()
            best_no_improve = 0

    # --- 10) Output reconstruction: map positions -> original indices, recompute bin weights ---
    packing: List[List[int]] = []
    for B in best_bins_pos:
        packing.append([pos_to_item[p] for p in B])

    final_bw = [sum(weights[i] for i in bin_items) for bin_items in packing]

    return {"packing": packing, "bin_weights": final_bw}
