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

    # ---- Sort items by decreasing weight, keep mappings ----
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

    # LB3 (fast, safe strengthening): account for > C/3 as well (simple counting)
    # Any bin can contain at most 2 items > C/3.
    third = C / 3
    count_third = sum(1 for wi in w if wi > third)
    lb3 = (count_third + 1) // 2

    lb = max(lb1, lb2, lb3)

    # ---- Greedy baseline portfolio ----
    def bfd_order(order: List[int], rand_ties: bool = False) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        rem: List[int] = []
        for p in order:
            wp = w[p]
            best_b = -1
            best_rem = None
            # scan all bins (n is moderate typically; good baseline)
            for b, r in enumerate(rem):
                if wp <= r:
                    nr = r - wp
                    if best_rem is None or nr < best_rem:
                        best_rem = nr
                        best_b = b
                    elif rand_ties and nr == best_rem and random.random() < 0.25:
                        best_b = b
            if best_b < 0:
                bins.append([p])
                rem.append(C - wp)
            else:
                bins[best_b].append(p)
                rem[best_b] -= wp
        bw = [C - r for r in rem]
        return bins, bw

    def key_from_bw(bw: List[int]) -> Tuple[int, int]:
        # primary: #bins, secondary: total slack
        return (len(bw), sum(C - x for x in bw))

    base_best_bins = None
    base_best_bw = None
    base_best_key = None

    order0 = list(range(n))  # already sorted by weight
    # Deterministic BFD
    for _ in range(2):
        bins0, bw0 = bfd_order(order0, rand_ties=False)
        k0 = key_from_bw(bw0)
        if base_best_key is None or k0 < base_best_key:
            base_best_key, base_best_bins, base_best_bw = k0, bins0, bw0

    # Randomized tie-breaking restarts
    # Keep small so we don't waste budget, but enough to diversify seed
    restarts = 12 if n <= 400 else 8
    for _ in range(restarts):
        bins0, bw0 = bfd_order(order0, rand_ties=True)
        k0 = key_from_bw(bw0)
        if k0 < base_best_key:
            base_best_key, base_best_bins, base_best_bw = k0, bins0, bw0
        if time.perf_counter() >= deadline:
            break

    best_bins_pos = base_best_bins
    best_bw = base_best_bw
    best_key = base_best_key

    # ---- ACO/MMAS parameters ----
    # Spend time: more ants per iter when small, fewer when large.
    if n <= 200:
        ants_per_iter = 40
    elif n <= 600:
        ants_per_iter = 28
    elif n <= 1500:
        ants_per_iter = 18
    else:
        ants_per_iter = 12

    # Fixed iteration budget (hard requirement). Time checks can stop earlier.
    max_iter = 120_000

    # ACS / MMAS behavior
    alpha = 1.2
    beta = 4.0
    rho = 0.06

    q0_start = 0.55
    q0_end = 0.92

    elite_every = 6
    stagnation_S = 260

    # Candidate sizes
    K_item = 28
    M_item_rand = 14
    K_bin = 10

    # ---- Pheromones with lazy evaporation ----
    tau0 = 1.0

    # Keep pairwise pheromone only among top heavy items (most impactful).
    H = min(n, 380)

    # tau_pair_rows[i] is dict j->tau (for i<j<H). Missing implies tau0.
    tau_pair_rows: List[Dict[int, float]] = [dict() for _ in range(H)]

    # tau_open for seed/early placement desirability (lazy evaporation via scale)
    tau_open = [tau0] * n

    # Lazy evaporation scaling factors
    scale_pair = 1.0
    scale_open = 1.0

    A_bound = 120.0

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
        # real tau = stored * scale_open
        v = tau_open[i] * scale_open
        if v < tau_min:
            return tau_min
        if v > tau_max:
            return tau_max
        return v

    def add_open(i: int, delta: float) -> None:
        # convert delta in real domain to stored domain
        if scale_open == 0.0:
            return
        cur = tau_open[i] * scale_open
        cur = clamp_tau(cur + delta)
        tau_open[i] = cur / scale_open

    def get_pair(i: int, j: int) -> float:
        if i == j:
            return tau_max
        if i > j:
            i, j = j, i
        if i >= H or j >= H:
            # implicit
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
        nonlocal scale_pair, scale_open
        mult = 1.0 - rho
        scale_pair *= mult
        scale_open *= mult
        # renormalize scales if too small to maintain numeric stability
        if scale_pair < 1e-6:
            # push scaling into stored values
            for i in range(H):
                row = tau_pair_rows[i]
                for j in list(row.keys()):
                    row[j] *= scale_pair
                    # clamp in stored domain relative to scale=1
                    if row[j] < tau_min:
                        row[j] = tau_min
                    elif row[j] > tau_max:
                        row[j] = tau_max
            scale_pair = 1.0
        if scale_open < 1e-6:
            for i in range(n):
                tau_open[i] *= scale_open
                if tau_open[i] < tau_min:
                    tau_open[i] = tau_min
                elif tau_open[i] > tau_max:
                    tau_open[i] = tau_max
            scale_open = 1.0

    def partial_reset() -> None:
        # Blend towards tau0 (in real domain)
        nonlocal scale_pair, scale_open
        # materialize into stored
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
        # blend
        for i in range(H):
            row = tau_pair_rows[i]
            for j in list(row.keys()):
                row[j] = clamp_tau(0.6 * row[j] + 0.4 * tau0)
        for i in range(n):
            tau_open[i] = clamp_tau(0.6 * tau_open[i] + 0.4 * tau0)

    # ---- Construction (ACS-like): place items one by one into chosen bins ----
    # Heuristic for item/bin: prefer tight fit (less remaining) and high fill.
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

        # bins representation
        bins: List[List[int]] = []
        rem_caps: List[int] = []

        # start with a seed (heavy) to open first bin
        # choose among top candidates using tau_open and size
        def choose_next_item_seed() -> int:
            # top-K heavy unpacked + random
            cand: List[int] = []
            for p in range(n):
                if len(cand) >= K_item:
                    break
                if unpacked[p]:
                    cand.append(p)
            tries = 0
            while tries < M_item_rand and len(cand) < K_item + M_item_rand and remaining_positions:
                tries += 1
                p = remaining_positions[random.randrange(len(remaining_positions))]
                if unpacked[p]:
                    cand.append(p)
            if len(cand) > 1:
                cand = list(dict.fromkeys(cand))

            best = cand[0]
            best_sc = -1.0
            scores = []
            ssum = 0.0
            for p in cand:
                h = (w[p] / C) + 1e-12
                sc = (get_open(p) ** 1.0) * (h ** 2.4)
                scores.append(sc)
                ssum += sc
                if sc > best_sc:
                    best_sc = sc
                    best = p
            if ssum <= 0.0 or random.random() < q0:
                return best
            r = random.random() * ssum
            acc = 0.0
            for p, sc in zip(cand, scores):
                acc += sc
                if acc >= r:
                    return p
            return best

        # choose next item among those that can fit somewhere (or open new bin)
        def choose_next_item() -> int:
            # prefer heavy remaining items; randomize a bit
            cand: List[int] = []
            for p in range(n):
                if len(cand) >= K_item:
                    break
                if unpacked[p]:
                    cand.append(p)
            tries = 0
            while tries < M_item_rand and len(cand) < K_item + M_item_rand and remaining_positions:
                tries += 1
                p = remaining_positions[random.randrange(len(remaining_positions))]
                if unpacked[p]:
                    cand.append(p)
            if len(cand) > 1:
                cand = list(dict.fromkeys(cand))

            # score depends on how well it might pair with existing bins' contents
            # approximate: use best pair pheromone to any item already placed in a candidate bin
            best = cand[0]
            best_sc = -1.0
            scores = []
            ssum = 0.0

            # sample up to K_bin bins with smallest remaining (most constrained) to estimate compatibility
            if bins:
                bin_idx = list(range(len(bins)))
                bin_idx.sort(key=lambda b: rem_caps[b])
                bin_idx = bin_idx[:K_bin]
            else:
                bin_idx = []

            for p in cand:
                size_h = (w[p] / C) + 1e-12
                comp = 1.0
                if bin_idx:
                    best_comp = 1.0
                    # take max pair pheromone with any item in those bins (limited)
                    for b in bin_idx:
                        if w[p] > rem_caps[b]:
                            continue
                        # compare against first few items in bin (heavy items are earlier by construction tendency)
                        B = bins[b]
                        take = 4 if len(B) > 4 else len(B)
                        for t in range(take):
                            v = get_pair(p, B[t])
                            if v > best_comp:
                                best_comp = v
                    comp = best_comp

                sc = (comp ** alpha) * (size_h ** 2.0)
                scores.append(sc)
                ssum += sc
                if sc > best_sc:
                    best_sc = sc
                    best = p

            if ssum <= 0.0 or random.random() < q0:
                return best
            r = random.random() * ssum
            acc = 0.0
            for p, sc in zip(cand, scores):
                acc += sc
                if acc >= r:
                    return p
            return best

        # choose bin for item p: among feasible bins pick best (ACS)
        def choose_bin_for_item(p: int) -> int:
            wp = w[p]
            feasible = [b for b, r in enumerate(rem_caps) if wp <= r]
            if not feasible:
                return -1
            # candidate: bins with smallest remaining after placement (best fit)
            feasible.sort(key=lambda b: rem_caps[b] - wp)
            feasible = feasible[:K_bin]

            best_b = feasible[0]
            best_sc = -1.0
            scores = []
            ssum = 0.0
            for b in feasible:
                # pheromone: avg pair with up to first few items in bin
                B = bins[b]
                take = 6 if len(B) > 6 else len(B)
                if take == 0:
                    pher = 1.0
                else:
                    s = 0.0
                    for t in range(take):
                        s += get_pair(p, B[t])
                    pher = s / take

                # heuristic: tighter fit and higher utilization
                new_rem = rem_caps[b] - wp
                tight = 1.0 / (1.0 + new_rem)
                fill = wp / max(1, rem_caps[b])
                h = (0.70 * tight + 0.30 * fill) + 1e-12

                sc = (pher ** alpha) * (h ** beta)
                scores.append(sc)
                ssum += sc
                if sc > best_sc:
                    best_sc = sc
                    best_b = b

            # exploitation/exploration
            if ssum <= 0.0 or random.random() < (q0_start + (q0_end - q0_start) * iter_frac):
                return best_b
            r = random.random() * ssum
            acc = 0.0
            for b, sc in zip(feasible, scores):
                acc += sc
                if acc >= r:
                    return b
            return best_b

        left = n
        inserts = 0

        # open first bin
        seed = choose_next_item_seed()
        bins.append([seed])
        rem_caps.append(C - w[seed])
        unpacked[seed] = False
        remove_pos(seed)
        left -= 1

        while left > 0:
            inserts += 1
            if (inserts & 255) == 0 and time.perf_counter() >= deadline:
                break

            p = choose_next_item()
            wp = w[p]
            b = choose_bin_for_item(p)
            if b < 0:
                bins.append([p])
                rem_caps.append(C - wp)
            else:
                bins[b].append(p)
                rem_caps[b] -= wp
            unpacked[p] = False
            remove_pos(p)
            left -= 1

        # If cut short, finish with greedy BFD into existing bins
        if left > 0:
            rem_pos = [p for p in range(n) if unpacked[p]]
            # descending by weight already
            for p in rem_pos:
                wp = w[p]
                best_b = -1
                best_rem = None
                for b, r in enumerate(rem_caps):
                    if wp <= r:
                        nr = r - wp
                        if best_rem is None or nr < best_rem:
                            best_rem = nr
                            best_b = b
                            if nr == 0:
                                break
                if best_b < 0:
                    bins.append([p])
                    rem_caps.append(C - wp)
                else:
                    bins[best_b].append(p)
                    rem_caps[best_b] -= wp

        bw = [C - r for r in rem_caps]
        return bins, bw

    # ---- Deposit ----
    def deposit(bins_pos: List[List[int]], bw: List[int], strength: float) -> None:
        bins_cnt = len(bw)
        slack = sum(C - x for x in bw)
        gap = max(0, bins_cnt - lb)

        # objective scalar (smaller is better)
        f = bins_cnt + slack / (C * max(1, n))

        # delta stronger near LB
        delta = strength * (1.0 / ((1.0 + gap) ** 2.4)) * (1.0 / max(1e-12, f))

        # Update open pheromone on first items of bins (seed) and pairwise within bins (limited)
        for B in bins_pos:
            if not B:
                continue
            add_open(B[0], delta)
            m = len(B)
            if m <= 1:
                continue
            # deposit on pairs among first t items (heavier items)
            t = 14 if m > 14 else m
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

        for _ in range(ants_per_iter):
            if time.perf_counter() >= deadline:
                break
            bins_pos, bw = construct_solution(iter_frac)
            k = key_from_bw(bw)
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
            best_bw = iter_best_bw
            best_no_improve = 0
            tau_min, tau_max = compute_tau_bounds(best_key[0])
            if best_key[0] == lb:
                break
        else:
            best_no_improve += 1

        evaporate()
        deposit(iter_best_bins, iter_best_bw, strength=1.0)
        if (it % elite_every) == 0:
            deposit(best_bins_pos, best_bw, strength=2.0)

        if best_no_improve >= stagnation_S:
            partial_reset()
            best_no_improve = 0

    # ---- Map back to original indices ----
    packing: List[List[int]] = [[pos_to_item[p] for p in B] for B in best_bins_pos]
    bin_weights = [sum(weights[i] for i in B) for B in packing]
    return {"packing": packing, "bin_weights": bin_weights}
