import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    start = time.perf_counter()
    deadline = start + max(0.0, float(time_limit))

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    wts = weights

    maxw = max(wts)
    if maxw <= 0:
        return {"packing": [list(range(n))], "bin_weights": [0]}

    # normalized weights for priority shaping
    inv_maxw = 1.0 / maxw
    norm_w = [w * inv_maxw for w in wts]

    # ---------- helpers ----------
    def timed_out() -> bool:
        return time.perf_counter() >= deadline

    # almost-full threshold (adaptive): encourage closing bins tightly
    # larger instances: slightly tighter threshold
    t_thr = max(1, C // (65 if n <= 400 else 80))
    almost_full_thr = C - t_thr

    # fitness: lexicographic tuple; bins dominates
    def fitness(bin_w: List[int]) -> Tuple[int, int, int, int]:
        k = len(bin_w)
        af = 0
        sum_sq = 0
        total_slack = 0
        # add a mild penalty for "very open" bins too
        very_open = 0
        vo_thr = C // 3
        for bw in bin_w:
            s = C - bw
            total_slack += s
            sum_sq += s * s
            if bw >= almost_full_thr:
                af += 1
            if bw <= vo_thr:
                very_open += 1
        # maximize af -> minimize -af; minimize very_open
        return (k, -af, very_open, sum_sq + total_slack)

    def scalar_from_fit(fit: Tuple[int, int, int, int]) -> float:
        k, neg_af, very_open, mix = fit
        # bins dominates heavily; other terms shape WF
        return k + 0.02 * very_open + 0.0000005 * mix + 0.0005 * (-neg_af)

    lo, hi = -1.0, 1.0

    def clamp(x: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    # ---------- precomputed structure ----------
    idx_by_weight_desc = sorted(range(n), key=wts.__getitem__, reverse=True)
    heavy_cut = max(1, int(0.35 * n))
    heavy_indices = idx_by_weight_desc[:heavy_cut]

    # weight groups (for occasional block moves)
    groups = {}
    for i, w in enumerate(wts):
        groups.setdefault(w, []).append(i)
    weight_groups = list(groups.values())

    # ---------- decoding (best-fit with bitset buckets + pair-closure bias) ----------
    # buckets[r] holds bins with remaining capacity r
    # bitset has bit r if buckets[r] non-empty

    def _bitset_find_ge(bitset: int, w: int) -> int:
        masked = bitset >> w
        if masked == 0:
            return -1
        lsb = masked & -masked
        off = lsb.bit_length() - 1
        return w + off

    def _bitset_find_ge_limited(bitset: int, w: int, limit: int) -> List[int]:
        masked = bitset >> w
        if masked == 0:
            return []
        opts = []
        m = masked
        for _ in range(limit):
            if m == 0:
                break
            lsb = m & -m
            off = lsb.bit_length() - 1
            opts.append(w + off)
            m ^= lsb
        return opts

    def decode_order(order: List[int], diversify_p: float,
                     best_bins_so_far: List[List[int]], best_binw_so_far: List[int]) -> Tuple[List[List[int]], List[int], bool]:
        bins: List[List[int]] = []
        bin_w: List[int] = []
        buckets: List[List[int]] = [[] for _ in range(C + 1)]
        nonempty = 0

        # For large n, check time less frequently for speed
        check_mask = 127 if n >= 1500 else 63

        for t_idx, i in enumerate(order):
            if (t_idx & check_mask) == 0 and timed_out():
                return best_bins_so_far, best_binw_so_far, True

            w = wts[i]

            # primary: best-fit (smallest remaining >= w)
            # diversification: sample among a few feasible remaining capacities,
            # but bias toward near-closure (small remaining after placement).
            chosen_r = -1
            if diversify_p > 0.0 and random.random() < diversify_p:
                opts = _bitset_find_ge_limited(nonempty, w, 6)
                if opts:
                    # score r by resulting remaining after placement: (r-w) small is good
                    # use a softmax-ish roulette on inverse
                    scores = []
                    for r in opts:
                        after = r - w
                        # prefer closing: after==0 best
                        scores.append(1.0 / (1.0 + after))
                    chosen_r = random.choices(opts, weights=scores, k=1)[0]
                else:
                    chosen_r = -1
            else:
                chosen_r = _bitset_find_ge(nonempty, w)

            if chosen_r == -1:
                j = len(bins)
                bins.append([i])
                bw = w
                bin_w.append(bw)
                rem = C - bw
                buckets[rem].append(j)
                nonempty |= (1 << rem)
            else:
                bucket = buckets[chosen_r]
                j = bucket.pop()
                if not bucket:
                    nonempty &= ~(1 << chosen_r)

                bins[j].append(i)
                bw = bin_w[j] + w
                bin_w[j] = bw
                rem = C - bw
                buckets[rem].append(j)
                nonempty |= (1 << rem)

        return bins, bin_w, False

    # ---------- order portfolio ----------
    def build_orders(priorities: List[float], K: int) -> List[List[int]]:
        # Base: (priority, weight) desc
        base = list(range(n))
        base.sort(key=lambda i: (priorities[i], wts[i]), reverse=True)
        if K <= 1:
            return [base]

        orders = [base]

        # Variant 1: weight-first but still guided
        if K >= 2:
            a = list(range(n))
            a.sort(key=lambda i: (wts[i], priorities[i]), reverse=True)
            orders.append(a)

        # Variant 2: complement around C/2 blended with priority
        if K >= 3:
            half = C / 2.0
            b = list(range(n))
            b.sort(key=lambda i: (
                0.75 * priorities[i] + 0.25 * (1.0 - abs(half - wts[i]) / (half + 1.0)),
                wts[i]
            ), reverse=True)
            orders.append(b)

        # Variant 3: small structured shuffle in heavy prefix
        if K >= 4:
            c = base[:]
            m = min(len(c), max(24, n // 12))
            if m >= 8:
                # shuffle a slice of the heavy prefix; keeps items heavy early but changes packing
                s = 0
                seg = m
                tmp = c[s:s + seg]
                random.shuffle(tmp)
                c[s:s + seg] = tmp
            orders.append(c)

        # Variant 4: reverse a mid segment (rarely used)
        if K >= 5:
            d = base[:]
            seg = max(16, n // 14)
            if seg < n:
                s = random.randrange(0, n - seg + 1)
                d[s:s + seg] = reversed(d[s:s + seg])
            orders.append(d)

        return orders

    def decode_portfolio(priorities: List[float], it: int, max_iters: int,
                         best_bins_so_far: List[List[int]], best_binw_so_far: List[int]) -> Tuple[List[List[int]], List[int], Tuple[int, int, int, int], bool]:
        # K adapts by n (more variants helps quality for small/medium)
        if n <= 220:
            K = 5
        elif n <= 900:
            K = 4
        else:
            K = 3

        # diversification probability: higher early, lower late, but never zero
        frac = it / max_iters if max_iters > 0 else 1.0
        base_p = 0.22 * (1.0 - frac) + 0.06

        orders = build_orders(priorities, K)

        best_local_bins: Optional[List[List[int]]] = None
        best_local_binw: Optional[List[int]] = None
        best_local_fit: Optional[Tuple[int, int, int, int]] = None

        for ord_idx, order in enumerate(orders):
            # slightly different diversify per variant
            diversify_p = base_p * (1.0 if ord_idx == 0 else 0.85)
            bins, binw, to = decode_order(order, diversify_p, best_bins_so_far, best_binw_so_far)
            if to:
                return best_bins_so_far, best_binw_so_far, fitness(best_binw_so_far), True
            fitv = fitness(binw)
            if best_local_fit is None or fitv < best_local_fit:
                best_local_fit = fitv
                best_local_bins = bins
                best_local_binw = binw

        return best_local_bins, best_local_binw, best_local_fit, False

    # ---------- priority seeding ----------
    def seeded_priority_vector(kind: int) -> List[float]:
        if kind == 0:
            # FFD-like: mostly weight
            return [clamp(0.95 * norm_w[i] + 1e-6 * (n - i)) for i in range(n)]
        if kind == 1:
            # BFD-like tie flip
            return [clamp(0.95 * norm_w[i] + 1e-6 * i) for i in range(n)]
        if kind == 2:
            # near half first
            half = C / 2.0
            return [clamp(1.0 - abs(half - wts[i]) / (half + 1.0)) for i in range(n)]
        if kind == 3:
            # emphasize heavy strongly
            return [clamp(1.05 * norm_w[i] - 0.05) for i in range(n)]
        if kind == 4:
            # mixed noise
            return [clamp(0.85 * norm_w[i] + random.uniform(-0.25, 0.25)) for i in range(n)]
        # fallback
        return [random.uniform(lo, hi) for _ in range(n)]

    # ---------- quick baseline: plain FFD order on weight ----------
    # (still decoded by our best-fit engine)
    ffd_order = idx_by_weight_desc
    ffd_bins, ffd_binw, to = decode_order(ffd_order, 0.0, [], [])
    if to:
        return {"packing": [], "bin_weights": []}

    best_bins = [b[:] for b in ffd_bins]
    best_binw = ffd_binw[:]
    best_fit = fitness(best_binw)
    best_pos: Optional[List[float]] = None

    # ---------- budgeting: scale iterations to time_limit and n ----------
    # Use a fixed number of iterations (deterministic given time_limit and n),
    # but still return early on time.
    if n <= 250:
        base_iters = 4500
    elif n <= 900:
        base_iters = 3200
    else:
        base_iters = 2200

    # more time => more iterations, capped to keep overhead bounded
    max_iters = int(base_iters * (0.6 + 0.02 * min(100.0, time_limit)))
    if max_iters < 800:
        max_iters = 800
    if max_iters > 20000:
        max_iters = 20000

    # population size: increase moderately for better coverage
    pop_size = int(28 + 8 * (n ** 0.5))
    if pop_size < 32:
        pop_size = 32
    if pop_size > 110:
        pop_size = 110

    scouts: List[List[float]] = []
    scout_fit: List[Tuple[int, int, int, int]] = []
    scout_bins: List[List[List[int]]] = []
    scout_binw: List[List[int]] = []
    no_improve_i: List[int] = []

    seed_kinds = [0, 1, 2, 3, 4]

    # include a scout seeded from FFD baseline priorities
    ffd_pos = [clamp(0.95 * norm_w[i] + 1e-6 * (n - i)) for i in range(n)]

    init_positions: List[List[float]] = [ffd_pos]
    for k in seed_kinds:
        init_positions.append(seeded_priority_vector(k))

    # fill rest
    while len(init_positions) < pop_size:
        if len(init_positions) < pop_size // 3:
            init_positions.append([random.uniform(lo, hi) for _ in range(n)])
        else:
            ns = 0.30 if len(init_positions) < (2 * pop_size) // 3 else 0.18
            init_positions.append([clamp(norm_w[i] + random.uniform(-ns, ns)) for i in range(n)])

    for p in range(pop_size):
        if timed_out():
            return {"packing": best_bins, "bin_weights": best_binw}

        pos = init_positions[p]
        bins, binw, fitv, to = decode_portfolio(pos, 0, max_iters, best_bins, best_binw)
        if to:
            return {"packing": best_bins, "bin_weights": best_binw}

        scouts.append(pos)
        scout_fit.append(fitv)
        scout_bins.append(bins)
        scout_binw.append(binw)
        no_improve_i.append(0)

        if fitv < best_fit:
            best_fit = fitv
            best_bins = [b[:] for b in bins]
            best_binw = binw[:]
            best_pos = pos[:]

    # ---------- FDO main loop ----------
    no_improve_global = 0

    # stagnation thresholds
    Sg = 55 if n <= 900 else 70
    Si = 30 if n <= 900 else 40

    def pick_leader(elite_indices: List[int]) -> List[float]:
        # rank-based roulette, with slight randomness to avoid over-convergence
        m = len(elite_indices)
        if m == 1:
            return scouts[elite_indices[0]]
        weights_rank = [1.0 / (1 + r) for r in range(m)]
        if random.random() < 0.12 and m >= 3:
            # occasionally prefer not-the-best elite
            weights_rank[0] *= 0.35
        chosen = random.choices(elite_indices, weights=weights_rank, k=1)[0]
        return scouts[chosen]

    def reinit_scout() -> List[float]:
        r = random.random()
        if r < 0.55:
            return seeded_priority_vector(random.choice(seed_kinds))
        if r < 0.80:
            return [random.uniform(lo, hi) for _ in range(n)]
        ns = 0.26
        return [clamp(norm_w[i] + random.uniform(-ns, ns)) for i in range(n)]

    # number of dimensions updated for large n
    def choose_update_set() -> Optional[List[int]]:
        if n <= 240:
            return None
        upd = max(80, n // 4)
        heavy_k = min(len(heavy_indices), int(0.72 * upd))
        rest_k = upd - heavy_k
        idxs: List[int] = []
        if heavy_k > 0:
            idxs.extend(random.sample(heavy_indices, heavy_k) if heavy_k < len(heavy_indices) else heavy_indices)
        if rest_k > 0:
            pool = idx_by_weight_desc[heavy_cut:]
            if pool:
                if rest_k >= len(pool):
                    idxs.extend(pool)
                else:
                    idxs.extend(random.sample(pool, rest_k))
        return idxs

    for it in range(max_iters):
        if timed_out():
            break

        # dynamic elite size
        E = 6 if pop_size >= 50 else 5
        if it > max_iters * 0.6:
            E = max(4, E - 1)

        idx_sorted = sorted(range(pop_size), key=lambda i: scout_fit[i])
        elite_indices = idx_sorted[: min(E, pop_size)]

        # refresh global best
        bi = idx_sorted[0]
        if scout_fit[bi] < best_fit:
            best_fit = scout_fit[bi]
            best_bins = [b[:] for b in scout_bins[bi]]
            best_binw = scout_binw[bi][:]
            best_pos = scouts[bi][:]
            no_improve_global = 0
        else:
            no_improve_global += 1

        best_scalar = scalar_from_fit(best_fit)

        # adaptive exploration
        frac = it / max_iters
        base_noise = 0.16 * (1.0 - frac) + 0.03
        if no_improve_global >= Sg // 2:
            base_noise *= 1.5
        if base_noise > 0.28:
            base_noise = 0.28

        wf_max = 1.0 + (0.6 if no_improve_global >= Sg // 2 else 0.0)

        # partial restart on global stagnation
        if no_improve_global >= Sg:
            no_improve_global = 0
            worst_count = max(1, pop_size // 4)
            worst = idx_sorted[-worst_count:]
            for wi in worst:
                if timed_out():
                    break
                pos_new = reinit_scout()
                bins, binw, fitv, to = decode_portfolio(pos_new, it, max_iters, best_bins, best_binw)
                if to:
                    return {"packing": best_bins, "bin_weights": best_binw}
                scouts[wi] = pos_new
                scout_fit[wi] = fitv
                scout_bins[wi] = bins
                scout_binw[wi] = binw
                no_improve_i[wi] = 0

        # update scouts
        for i in range(pop_size):
            if timed_out():
                break

            if no_improve_i[i] >= Si:
                no_improve_i[i] = 0
                # opposition or restart
                if random.random() < 0.45:
                    old = scouts[i]
                    pos = [clamp(-x + random.uniform(-0.06, 0.06)) for x in old]
                else:
                    pos = reinit_scout()

                bins, binw, fitv, to = decode_portfolio(pos, it, max_iters, best_bins, best_binw)
                if to:
                    return {"packing": best_bins, "bin_weights": best_binw}

                scouts[i] = pos
                scout_fit[i] = fitv
                scout_bins[i] = bins
                scout_binw[i] = binw

                if fitv < best_fit:
                    best_fit = fitv
                    best_bins = [b[:] for b in bins]
                    best_binw = binw[:]
                    best_pos = pos[:]
                    no_improve_global = 0
                continue

            pos = scouts[i]
            fit_i = scout_fit[i]
            scalar_i = scalar_from_fit(fit_i)

            if scalar_i <= best_scalar:
                wf = 0.0
            else:
                gap = (scalar_i - best_scalar) / (best_scalar + 1e-9)
                wf = gap
                if wf > wf_max:
                    wf = wf_max

            leader_pos = pick_leader(elite_indices)

            # dimension subset update
            idxs = choose_update_set()

            r = random.random()
            noise_amp = base_noise

            # occasional group block perturbation
            do_block = (weight_groups and random.random() < (0.10 if n > 300 else 0.14))
            block = random.choice(weight_groups) if do_block else None

            new_pos = pos[:]

            if idxs is None:
                for d in range(n):
                    x = new_pos[d]
                    step = r * wf * (leader_pos[d] - x)
                    # light pull to weight signal to reduce drift
                    pull = 0.03 * (norm_w[d] - (x * 0.5 + 0.5))
                    y = x + step + random.uniform(-noise_amp, noise_amp) + pull
                    new_pos[d] = clamp(y)
            else:
                for d in idxs:
                    x = new_pos[d]
                    step = r * wf * (leader_pos[d] - x)
                    pull = 0.03 * (norm_w[d] - (x * 0.5 + 0.5))
                    y = x + step + random.uniform(-noise_amp, noise_amp) + pull
                    new_pos[d] = clamp(y)

                if block is not None:
                    delta = random.uniform(-noise_amp, noise_amp)
                    for d in block:
                        new_pos[d] = clamp(new_pos[d] + delta)

            bins, binw, fit_new, to = decode_portfolio(new_pos, it, max_iters, best_bins, best_binw)
            if to:
                return {"packing": best_bins, "bin_weights": best_binw}

            if fit_new <= fit_i:
                scouts[i] = new_pos
                scout_fit[i] = fit_new
                scout_bins[i] = bins
                scout_binw[i] = binw
                no_improve_i[i] = 0

                if fit_new < best_fit:
                    best_fit = fit_new
                    best_bins = [b[:] for b in bins]
                    best_binw = binw[:]
                    best_pos = new_pos[:]
                    no_improve_global = 0
            else:
                no_improve_i[i] += 1

    return {"packing": best_bins, "bin_weights": best_binw}
