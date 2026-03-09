import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.perf_counter()
    # user says execution will stop at 100 seconds; we can safely run up to their provided time_limit
    deadline = start + max(0.0, float(time_limit))

    # ---- Sort items by decreasing weight, keep mapping ----
    pos_to_item = list(range(n))
    pos_to_item.sort(key=lambda i: weights[i], reverse=True)
    w = [weights[i] for i in pos_to_item]

    total_w = sum(w)

    # ---- Lower bounds ----
    lb1 = (total_w + C - 1) // C

    # LB2: items > C/2 each need a bin + ceil(remaining/C)
    half = C / 2
    big = 0
    sum_others = 0
    for wi in w:
        if wi > half:
            big += 1
        else:
            sum_others += wi
    lb2 = big + ((sum_others + C - 1) // C)

    # LB3: items > C/3 at most 2 per bin
    third = C / 3
    count_third = 0
    for wi in w:
        if wi > third:
            count_third += 1
    lb3 = (count_third + 1) // 2

    # LB4 (fast MT-like strengthening using large-item pairing logic)
    # Count items in (C/2, C], (C/3, C/2], (C/4, C/3], ... and force bins based on limited pairability.
    # A lightweight safe version: consider > C/2 and > C/3 separately with leftover space accounting.
    # For each item > C/2, it can only accept at most one partner of size <= C - wi.
    # We compute how many small items can be absorbed; the rest need bins.
    gt_half = []
    le_half = []
    for wi in w:
        if wi > half:
            gt_half.append(wi)
        else:
            le_half.append(wi)
    le_half.sort()  # ascending for greedy absorption

    # try to pair the largest >C/2 with the largest possible <= residual (classic greedy)
    j = len(le_half) - 1
    absorbed = 0
    for wi in gt_half:
        res = C - wi
        while j >= 0 and le_half[j] > res:
            j -= 1
        if j >= 0:
            absorbed += 1
            j -= 1
    # remaining items after absorbing into the >half bins
    remaining_weight = sum(le_half[: j + 1])
    # bins for gt_half are fixed, plus bins for remaining
    lb4 = len(gt_half) + ((remaining_weight + C - 1) // C)

    lb = max(lb1, lb2, lb3, lb4)

    # ---- Greedy baseline: Best-Fit Decreasing with randomized ties ----
    def bfd(order: List[int], rand_ties: bool) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        rem: List[int] = []
        for p in order:
            wp = w[p]
            best_b = -1
            best_after = None
            for b, r in enumerate(rem):
                if wp <= r:
                    after = r - wp
                    if best_after is None or after < best_after:
                        best_after = after
                        best_b = b
                    elif rand_ties and after == best_after and random.random() < 0.35:
                        best_b = b
            if best_b < 0:
                bins.append([p])
                rem.append(C - wp)
            else:
                bins[best_b].append(p)
                rem[best_b] -= wp
        bw = [C - r for r in rem]
        return bins, bw

    def key_from_bw(bw: List[int]) -> Tuple[int, int, int]:
        # primary: bins, secondary: slack, tertiary: number of very empty bins
        slack = sum(C - x for x in bw)
        very_empty = sum(1 for x in bw if x <= C // 3)
        return (len(bw), slack, very_empty)

    order0 = list(range(n))
    best_bins_pos, best_bw = bfd(order0, rand_ties=False)
    best_key = key_from_bw(best_bw)

    restarts = 18 if n <= 500 else 12 if n <= 1500 else 8
    for _ in range(restarts):
        if time.perf_counter() >= deadline:
            break
        bins0, bw0 = bfd(order0, rand_ties=True)
        k0 = key_from_bw(bw0)
        if k0 < best_key:
            best_bins_pos, best_bw, best_key = bins0, bw0, k0

    # ---- ACO / MMAS parameters ----
    # Fixed iteration budget; we also respect the time limit.
    max_iter = 220_000

    if n <= 200:
        ants_per_iter = 38
    elif n <= 600:
        ants_per_iter = 28
    elif n <= 1500:
        ants_per_iter = 18
    else:
        ants_per_iter = 12

    alpha = 1.3
    beta = 5.2
    rho = 0.055

    q0_start = 0.60
    q0_end = 0.95

    elite_every = 5
    stagnation_S = 320

    # candidate sizes
    K_item = 34
    M_item_rand = 18
    K_bin = 12

    # pheromone base
    tau0 = 1.0

    # pairwise pheromone among top heavy items
    H = min(n, 520)
    tau_pair_rows: List[Dict[int, float]] = [dict() for _ in range(H)]

    # opening pheromone
    tau_open = [tau0] * n

    # discretized remaining-capacity class pheromone (helps learn residual patterns)
    # classes are 0..B-1 for rem in [0..C]
    BCLS = 24
    tau_remcls = [tau0] * BCLS

    # lazy evaporation scales
    scale_pair = 1.0
    scale_open = 1.0
    scale_rem = 1.0

    A_bound = 140.0

    def compute_tau_bounds(best_bins: int) -> Tuple[float, float]:
        gap = max(0, best_bins - lb)
        tau_max = 1.0 / (rho * (1.0 + gap))
        tau_min = tau_max / A_bound
        return tau_min, tau_max

    tau_min, tau_max = compute_tau_bounds(best_key[0])

    def clamp_tau(v: float) -> float:
        if v < tau_min:
            return tau_min
        if v > tau_max:
            return tau_max
        return v

    def get_open(i: int) -> float:
        v = tau_open[i] * scale_open
        if v < tau_min:
            return tau_min
        if v > tau_max:
            return tau_max
        return v

    def add_open(i: int, delta: float) -> None:
        nonlocal scale_open
        if scale_open == 0.0:
            return
        cur = tau_open[i] * scale_open
        cur = clamp_tau(cur + delta)
        tau_open[i] = cur / scale_open

    def get_remcls(rem: int) -> float:
        # rem in [0..C]
        idx = (rem * (BCLS - 1)) // C
        v = tau_remcls[idx] * scale_rem
        if v < tau_min:
            return tau_min
        if v > tau_max:
            return tau_max
        return v

    def add_remcls(rem: int, delta: float) -> None:
        nonlocal scale_rem
        if scale_rem == 0.0:
            return
        idx = (rem * (BCLS - 1)) // C
        cur = tau_remcls[idx] * scale_rem
        cur = clamp_tau(cur + delta)
        tau_remcls[idx] = cur / scale_rem

    def get_pair(i: int, j: int) -> float:
        if i == j:
            return tau_max
        if i > j:
            i, j = j, i
        if i >= H or j >= H:
            v = tau0 * scale_pair
            if v < tau_min:
                return tau_min
            if v > tau_max:
                return tau_max
            return v
        row = tau_pair_rows[i]
        base = row.get(j, tau0)
        v = base * scale_pair
        if v < tau_min:
            return tau_min
        if v > tau_max:
            return tau_max
        return v

    def add_pair(i: int, j: int, delta: float) -> None:
        nonlocal scale_pair
        if i == j:
            return
        if i > j:
            i, j = j, i
        if i >= H or j >= H:
            return
        if scale_pair == 0.0:
            return
        row = tau_pair_rows[i]
        cur = row.get(j, tau0) * scale_pair
        cur = clamp_tau(cur + delta)
        row[j] = cur / scale_pair

    def evaporate() -> None:
        nonlocal scale_pair, scale_open, scale_rem
        mult = 1.0 - rho
        scale_pair *= mult
        scale_open *= mult
        scale_rem *= mult

        # renormalize
        if scale_pair < 1e-7:
            for i in range(H):
                row = tau_pair_rows[i]
                for j in list(row.keys()):
                    v = row[j] * scale_pair
                    if v < tau_min:
                        v = tau_min
                    elif v > tau_max:
                        v = tau_max
                    row[j] = v
            scale_pair = 1.0

        if scale_open < 1e-7:
            for i in range(n):
                v = tau_open[i] * scale_open
                if v < tau_min:
                    v = tau_min
                elif v > tau_max:
                    v = tau_max
                tau_open[i] = v
            scale_open = 1.0

        if scale_rem < 1e-7:
            for k in range(BCLS):
                v = tau_remcls[k] * scale_rem
                if v < tau_min:
                    v = tau_min
                elif v > tau_max:
                    v = tau_max
                tau_remcls[k] = v
            scale_rem = 1.0

    def partial_reset() -> None:
        nonlocal scale_pair, scale_open, scale_rem
        # materialize
        if scale_pair != 1.0:
            for i in range(H):
                row = tau_pair_rows[i]
                for j in list(row.keys()):
                    row[j] *= scale_pair
            scale_pair = 1.0
        if scale_open != 1.0:
            for i in range(n):
                tau_open[i] *= scale_open
            scale_open = 1.0
        if scale_rem != 1.0:
            for k in range(BCLS):
                tau_remcls[k] *= scale_rem
            scale_rem = 1.0

        # blend towards tau0
        for i in range(H):
            row = tau_pair_rows[i]
            for j in list(row.keys()):
                row[j] = clamp_tau(0.50 * row[j] + 0.50 * tau0)
        for i in range(n):
            tau_open[i] = clamp_tau(0.50 * tau_open[i] + 0.50 * tau0)
        for k in range(BCLS):
            tau_remcls[k] = clamp_tau(0.50 * tau_remcls[k] + 0.50 * tau0)

    # ---- Construction: ACS-like on (item, bin) moves ----
    def construct_solution(iter_frac: float) -> Tuple[List[List[int]], List[int]]:
        q0 = q0_start + (q0_end - q0_start) * iter_frac

        unpacked = [True] * n
        remaining_positions = list(range(n))
        pos_in_remaining = list(range(n))

        def remove_pos(p: int) -> None:
            idx = pos_in_remaining[p]
            last = remaining_positions[-1]
            remaining_positions[idx] = last
            pos_in_remaining[last] = idx
            remaining_positions.pop()

        bins: List[List[int]] = []
        rem_caps: List[int] = []

        # seed first bin from top items using tau_open
        top_seed = min(K_item, n)
        best_p = 0
        best_sc = -1.0
        for p in range(top_seed):
            if not unpacked[p]:
                continue
            # prefer heavier seeds but allow pheromone
            size_h = (w[p] / C) + 1e-12
            sc = (get_open(p) ** 1.0) * (size_h ** 3.0)
            if sc > best_sc:
                best_sc = sc
                best_p = p
        seed = best_p
        bins.append([seed])
        rem_caps.append(C - w[seed])
        unpacked[seed] = False
        remove_pos(seed)
        left = n - 1

        step = 0
        while left > 0:
            step += 1
            if (step & 255) == 0 and time.perf_counter() >= deadline:
                break

            # candidate items: top-heavy remaining + random samples
            cand_items: List[int] = []
            # scan from start for heavies
            for p in range(n):
                if len(cand_items) >= K_item:
                    break
                if unpacked[p]:
                    cand_items.append(p)
            # random add
            tries = 0
            while tries < M_item_rand and len(cand_items) < K_item + M_item_rand and remaining_positions:
                tries += 1
                p = remaining_positions[random.randrange(len(remaining_positions))]
                if unpacked[p]:
                    cand_items.append(p)
            if len(cand_items) > 1:
                cand_items = list(dict.fromkeys(cand_items))

            # Preselect some bins to consider for best-fit-like evaluation
            # bins with smallest remaining are most constrained
            if rem_caps:
                idxs = list(range(len(rem_caps)))
                idxs.sort(key=lambda b: rem_caps[b])
                bin_short = idxs[: max(6, K_bin)]
            else:
                bin_short = []

            # Evaluate moves (p -> existing bin b) and (p -> new bin)
            best_move = None  # (score, p, b)
            scores: List[Tuple[float, int, int]] = []
            ssum = 0.0

            for p in cand_items:
                wp = w[p]
                size_h = (wp / C) + 1e-12

                # option: open new bin
                # heuristic: opening is worse unless item is large; also influenced by open pheromone
                # estimate desirability of remaining class after placing alone
                newrem = C - wp
                pher_open = get_open(p)
                pher_rem = get_remcls(newrem)
                # penalize opening bins slightly to encourage packing
                h_open = (0.25 + 0.75 * size_h)
                sc_open = (pher_open ** 1.0) * (pher_rem ** 0.6) * (h_open ** beta) * 0.55
                scores.append((sc_open, p, -1))
                ssum += sc_open

                # existing bins
                # shortlist feasible bins by best-fit among bin_short
                feasible = []
                for b in bin_short:
                    r = rem_caps[b]
                    if wp <= r:
                        feasible.append(b)
                if feasible:
                    feasible.sort(key=lambda b: rem_caps[b] - wp)
                    feasible = feasible[:K_bin]

                for b in feasible:
                    r = rem_caps[b]
                    after = r - wp

                    # pheromone: compatibility with early items in that bin
                    B = bins[b]
                    take = 6 if len(B) > 6 else len(B)
                    if take == 0:
                        pher_pair = 1.0
                    else:
                        # max tends to work well for compatibility
                        m = 1.0
                        for t in range(take):
                            v = get_pair(p, B[t])
                            if v > m:
                                m = v
                        pher_pair = m

                    pher_rem = get_remcls(after)

                    # heuristic: tight fit, plus discourage tiny unusable residuals
                    tight = 1.0 / (1.0 + after)
                    # residual usability: prefer after close to 0 or reasonably large; penalize mid small waste
                    # use a smooth function peaked at 0 and also not too low at larger after
                    frac = after / C
                    usability = 1.0 - 0.75 * (frac * (1.0 - frac))  # minimum at 0.5

                    h = (0.72 * tight + 0.28 * usability) + 1e-12

                    sc = (pher_pair ** alpha) * (pher_rem ** 0.7) * ((size_h + 0.2) ** 1.2) * (h ** beta)
                    scores.append((sc, p, b))
                    ssum += sc

            # choose move ACS-style
            if not scores:
                break

            if ssum <= 0.0 or random.random() < q0:
                # greedy
                sc, p, b = max(scores, key=lambda x: x[0])
            else:
                r = random.random() * ssum
                acc = 0.0
                p = scores[-1][1]
                b = scores[-1][2]
                for sc, pp, bb in scores:
                    acc += sc
                    if acc >= r:
                        p, b = pp, bb
                        break

            # apply move
            wp = w[p]
            if b < 0:
                bins.append([p])
                rem_caps.append(C - wp)
            else:
                bins[b].append(p)
                rem_caps[b] -= wp
            unpacked[p] = False
            remove_pos(p)
            left -= 1

        # If cut short, finish with deterministic BFD into existing bins
        if left > 0:
            rem_pos = [p for p in range(n) if unpacked[p]]
            for p in rem_pos:
                wp = w[p]
                best_b = -1
                best_after = None
                for b, r in enumerate(rem_caps):
                    if wp <= r:
                        after = r - wp
                        if best_after is None or after < best_after:
                            best_after = after
                            best_b = b
                            if after == 0:
                                break
                if best_b < 0:
                    bins.append([p])
                    rem_caps.append(C - wp)
                else:
                    bins[best_b].append(p)
                    rem_caps[best_b] -= wp

        bw = [C - r for r in rem_caps]
        return bins, bw

    # ---- Deposit (MMAS) ----
    def deposit(bins_pos: List[List[int]], bw: List[int], strength: float) -> None:
        bins_cnt = len(bw)
        slack = sum(C - x for x in bw)
        gap = max(0, bins_cnt - lb)

        f = bins_cnt + 0.30 * (slack / (C * max(1, n)))

        # stronger reward close to LB
        delta = strength * (1.0 / ((1.0 + gap) ** 2.6)) * (1.0 / max(1e-12, f))

        for B in bins_pos:
            if not B:
                continue
            # open pheromone for first item
            add_open(B[0], delta)

            # rem-class pheromone for the bin's final remaining capacity
            # bw is filled weight => rem = C - bw
            bsum = 0
            for p in B:
                bsum += w[p]
            rem = C - bsum
            if rem < 0:
                rem = 0
            add_remcls(rem, delta)

            # pair pheromone within bin among first t items
            m = len(B)
            if m <= 1:
                continue
            t = 16 if m > 16 else m
            for a in range(t - 1):
                i = B[a]
                for b in range(a + 1, t):
                    j = B[b]
                    add_pair(i, j, delta)

    best_no_improve = 0

    # ---- Main loop ----
    for it in range(1, max_iter + 1):
        if time.perf_counter() >= deadline:
            break

        iter_best_bins = None
        iter_best_bw = None
        iter_best_key = None

        iter_frac = (it - 1) / max(1, (max_iter - 1))

        for a in range(ants_per_iter):
            if (a & 3) == 0 and time.perf_counter() >= deadline:
                break
            bins_pos, bw = construct_solution(iter_frac)
            k = key_from_bw(bw)
            if iter_best_key is None or k < iter_best_key:
                iter_best_key = k
                iter_best_bins = bins_pos
                iter_best_bw = bw

        if iter_best_key is None:
            break

        # update global best
        if iter_best_key < best_key:
            best_key = iter_best_key
            best_bins_pos = iter_best_bins
            best_bw = iter_best_bw
            best_no_improve = 0
            tau_min, tau_max = compute_tau_bounds(best_key[0])
            if best_key[0] == lb:
                break
        else:
            best_no_improve += 1

        evaporate()

        # MMAS-style: deposit iter-best, plus global-best periodically
        deposit(iter_best_bins, iter_best_bw, strength=1.0)
        if (it % elite_every) == 0:
            deposit(best_bins_pos, best_bw, strength=2.4)

        if best_no_improve >= stagnation_S:
            partial_reset()
            best_no_improve = 0

    # ---- Map back to original indices ----
    packing: List[List[int]] = [[pos_to_item[p] for p in B] for B in best_bins_pos]
    bin_weights = [sum(weights[i] for i in B) for B in packing]
    return {"packing": packing, "bin_weights": bin_weights}
