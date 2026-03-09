import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    start = time.perf_counter()
    deadline = start + max(0.0, time_limit)

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    wts = weights

    maxw = max(wts)
    if maxw <= 0:
        # All zero weights: put everything in one bin.
        return {"packing": [list(range(n))], "bin_weights": [0]}

    norm_w = [w / maxw for w in wts]

    # -------------------- Fitness shaping (Plan §3) --------------------
    t = max(1, C // 50)  # almost-full threshold

    def fitness(bin_w: List[int]) -> Tuple[int, int, int, int]:
        # (bins, -almost_full_count, sum_sq_slack, total_slack)
        k = len(bin_w)
        total_slack = 0
        sum_sq = 0
        almost_full = 0
        thr = C - t
        for bw in bin_w:
            s = C - bw
            total_slack += s
            sum_sq += s * s
            if bw >= thr:
                almost_full += 1
        return (k, -almost_full, sum_sq, total_slack)

    # scalar used for wf gap (Plan §3.2)
    def scalar_from_fit(fit: Tuple[int, int, int, int]) -> float:
        k, neg_af, sum_sq, total_slack = fit
        # k dominates; include mild slack structure
        # Normalize sum_sq by (C^2 * k) and total_slack by (C*n)
        denom_sq = (C * C) * (k if k > 0 else 1)
        denom_ts = (C * n + 1)
        return (
            k
            + 0.05 * (sum_sq / (denom_sq + 1.0))
            + 0.01 * (total_slack / (denom_ts + 1.0))
            + 0.001 * (-neg_af)  # prefer more almost-full bins
        )

    # -------------------- Structured subset indices (Plan §1.4) --------------------
    # Heaviest items dominate feasibility; bias updates to them.
    idx_by_weight_desc = sorted(range(n), key=wts.__getitem__, reverse=True)
    heavy_cut = max(1, int(0.30 * n))
    heavy_indices = idx_by_weight_desc[:heavy_cut]

    # Group by weight to allow block perturbations.
    # For huge n, cap groups by sampling distinct weights only.
    groups = {}
    for i, w in enumerate(wts):
        groups.setdefault(w, []).append(i)
    weight_groups = list(groups.values())

    # -------------------- Decoding with bitset next-nonempty (Plan §2.2) --------------------
    # We implement Best-Fit via smallest remaining capacity r >= w.
    # Maintain buckets[r] of bin indices with remaining capacity r, and a bitset of non-empty r.

    def _bitset_find_ge(bitset: int, w: int) -> int:
        """Return smallest r>=w with bitset having bit r set; -1 if none."""
        # Mask off bits < w
        masked = bitset >> w
        if masked == 0:
            return -1
        # least significant set bit in masked
        lsb = masked & -masked
        # index in masked:
        off = lsb.bit_length() - 1
        return w + off

    def _bitset_find_ge_limited(bitset: int, w: int, limit: int) -> int:
        """Return up to 'limit' smallest feasible r>=w as list-like via repeated extraction.
        Here we return just one sampled among the first few feasible r's (Plan §2.3)."""
        masked = bitset >> w
        if masked == 0:
            return -1
        # Extract up to limit options from low bits.
        opts = []
        m = masked
        for _ in range(limit):
            if m == 0:
                break
            lsb = m & -m
            off = lsb.bit_length() - 1
            opts.append(w + off)
            m ^= lsb
        return random.choice(opts) if opts else -1

    # Deterministic choice within bucket: pop last (LIFO), but we push deterministically by bin index.
    # (Plan §2.4): reduce extra randomness.

    def decode_order(order: List[int], mix_rule: float, best_bins_so_far, best_binw_so_far) -> Tuple[List[List[int]], List[int], bool]:
        # mix_rule: probability of using "less myopic" selection (Plan §2.3)
        bins: List[List[int]] = []
        bin_w: List[int] = []
        # buckets[r] = stack of bin indices with remaining capacity r
        buckets: List[List[int]] = [[] for _ in range(C + 1)]
        nonempty = 0  # bitset

        check_mask = 63

        for t_idx, i in enumerate(order):
            if (t_idx & check_mask) == 0:
                if time.perf_counter() >= deadline:
                    return best_bins_so_far, best_binw_so_far, True

            w = wts[i]

            chosen_bin = -1
            chosen_r = -1

            # Best-Fit usually; sometimes pick among a few feasible r (adds diversity)
            if mix_rule > 0.0 and random.random() < mix_rule:
                chosen_r = _bitset_find_ge_limited(nonempty, w, 4)
            else:
                chosen_r = _bitset_find_ge(nonempty, w)

            if chosen_r == -1:
                # open new bin
                j = len(bins)
                bins.append([i])
                bw = w
                bin_w.append(bw)
                rem = C - bw
                buckets[rem].append(j)
                nonempty |= (1 << rem)
            else:
                # take a bin from that remaining capacity
                bucket = buckets[chosen_r]
                chosen_bin = bucket.pop()
                if not bucket:
                    nonempty &= ~(1 << chosen_r)

                bins[chosen_bin].append(i)
                bw = bin_w[chosen_bin] + w
                bin_w[chosen_bin] = bw
                rem = C - bw
                buckets[rem].append(chosen_bin)
                nonempty |= (1 << rem)

        return bins, bin_w, False

    # -------------------- Portfolio decoding per candidate (Plan §2.1) --------------------
    # Build a few alternative induced orders cheaply.

    def build_orders(priorities: List[float], K: int) -> List[List[int]]:
        # Base order: (priority, weight) desc
        base = list(range(n))
        base.sort(key=lambda i: (priorities[i], wts[i]), reverse=True)
        if K <= 1:
            return [base]

        orders = [base]

        # Variant A: flip tie-break
        if K >= 2:
            a = list(range(n))
            a.sort(key=lambda i: (priorities[i], -wts[i]), reverse=True)
            orders.append(a)

        # Variant B: blended score
        if K >= 3:
            alpha = 0.85
            b = list(range(n))
            b.sort(key=lambda i: (alpha * priorities[i] + (1.0 - alpha) * norm_w[i], wts[i]), reverse=True)
            orders.append(b)

        # Variant C (rare/only if K==4): segment reversal on base to diversify while keeping weight structure
        if K >= 4:
            c = base[:]  # start from base
            seg = max(10, n // 10)
            if seg < n:
                s = random.randrange(0, n - seg + 1)
                c[s:s + seg] = reversed(c[s:s + seg])
            orders.append(c)

        return orders

    def decode_portfolio(priorities: List[float], it: int, max_iters: int,
                         best_bins_so_far, best_binw_so_far) -> Tuple[List[List[int]], List[int], Tuple[int, int, int, int], bool]:
        # Adaptive K by n (Plan §2.1)
        if n <= 200:
            K = 4
        elif n <= 800:
            K = 3
        else:
            K = 2

        # Placement mix prob: a bit higher early, lower later (Plan §2.3)
        base_p = 0.18 if it < max_iters * 0.4 else 0.10

        orders = build_orders(priorities, K)

        best_local_bins = None
        best_local_binw = None
        best_local_fit = None

        for ord_idx, order in enumerate(orders):
            # slightly vary mix between variants
            mix_p = base_p * (1.0 if ord_idx == 0 else 0.8)
            bins, binw, timed_out = decode_order(order, mix_p, best_bins_so_far, best_binw_so_far)
            if timed_out:
                return best_bins_so_far, best_binw_so_far, fitness(best_binw_so_far), True
            fitv = fitness(binw)
            if best_local_fit is None or fitv < best_local_fit:
                best_local_fit = fitv
                best_local_bins = bins
                best_local_binw = binw

        return best_local_bins, best_local_binw, best_local_fit, False

    # -------------------- Initialization (Plan §5.1) --------------------
    lo, hi = -1.0, 1.0

    def clamp01(x: float) -> float:
        return lo if x < lo else (hi if x > hi else x)

    def seeded_priority_vector(kind: int) -> List[float]:
        # Deterministic-ish seeds (no random needed, but fine if used elsewhere)
        if kind == 0:
            # FFD-like: weight + tiny index jitter
            return [clamp01(0.9 * norm_w[i] + 1e-6 * (n - i)) for i in range(n)]
        if kind == 1:
            # BFD-like: weight + tiny increasing jitter (different tie)
            return [clamp01(0.9 * norm_w[i] + 1e-6 * i) for i in range(n)]
        if kind == 2:
            # Complement: items near C/2 first
            half = C / 2.0
            # closer to half -> higher priority
            return [clamp01(1.0 - abs(half - wts[i]) / (half + 1.0)) for i in range(n)]
        if kind == 3:
            # Mixed: mostly weight
            return [clamp01(0.7 * norm_w[i] + 0.3 * (2.0 * random.random() - 1.0) * 0.2) for i in range(n)]
        if kind == 4:
            # Another mixed: stronger weight
            return [clamp01(1.0 * norm_w[i] + (2.0 * random.random() - 1.0) * 0.15) for i in range(n)]
        # fallback noisy
        noise_scale = 0.25
        return [clamp01(norm_w[i] + random.uniform(-noise_scale, noise_scale)) for i in range(n)]

    # -------------------- Pop sizing and iteration budgeting (Plan §4) --------------------
    pop_size = int(20 + 6 * (n ** 0.5))
    if pop_size < 25:
        pop_size = 25
    elif pop_size > 80:
        pop_size = 80

    if n <= 200:
        max_iters = 2000
    elif n <= 800:
        max_iters = 1200
    else:
        max_iters = 800

    # -------------------- FDO population state --------------------
    scouts: List[List[float]] = []
    scout_fit: List[Tuple[int, int, int, int]] = []
    scout_bins: List[List[List[int]]] = []
    scout_binw: List[List[int]] = []
    no_improve_i: List[int] = []

    best_pos: Optional[List[float]] = None
    best_fit: Optional[Tuple[int, int, int, int]] = None
    best_bins: List[List[int]] = []
    best_binw: List[int] = []

    # Initial seeds first, then random/noisy
    seed_kinds = [0, 1, 2, 3, 4]
    init_count = pop_size
    for p in range(init_count):
        if time.perf_counter() >= deadline:
            break

        if p < len(seed_kinds):
            pos = seeded_priority_vector(seed_kinds[p])
        elif p < len(seed_kinds) + pop_size // 3:
            pos = [random.uniform(lo, hi) for _ in range(n)]
        else:
            # weight-based with noise
            ns = 0.35 if p < (2 * pop_size) // 3 else 0.18
            pos = [clamp01(norm_w[i] + random.uniform(-ns, ns)) for i in range(n)]

        bins, binw, fitv, timed_out = decode_portfolio(pos, 0, max_iters, best_bins, best_binw)
        if timed_out:
            return {"packing": best_bins, "bin_weights": best_binw}

        scouts.append(pos)
        scout_fit.append(fitv)
        scout_bins.append(bins)
        scout_binw.append(binw)
        no_improve_i.append(0)

        if best_fit is None or fitv < best_fit:
            best_fit = fitv
            best_pos = pos[:]
            best_bins = [b[:] for b in bins]
            best_binw = binw[:]

    if not scouts:
        return {"packing": [], "bin_weights": []}

    assert best_pos is not None and best_fit is not None

    # -------------------- FDO main loop with elites + restarts (Plan §1) --------------------
    E = 5 if pop_size >= 25 else 3
    Sg = 40
    Si = 25
    no_improve_global = 0

    def pick_leader(elite_indices: List[int]) -> List[float]:
        # Roulette biased by rank: weight ~ 1/(rank+1)
        weights_rank = [1.0 / (r + 1) for r in range(len(elite_indices))]
        chosen = random.choices(elite_indices, weights=weights_rank, k=1)[0]
        return scouts[chosen]

    def reinit_scout(strategy: int) -> List[float]:
        # 0: seeded, 1: random, 2: mixed noisy
        if strategy == 0:
            return seeded_priority_vector(random.choice(seed_kinds))
        if strategy == 1:
            return [random.uniform(lo, hi) for _ in range(n)]
        ns = 0.30
        return [clamp01(norm_w[i] + random.uniform(-ns, ns)) for i in range(n)]

    for it in range(max_iters):
        if time.perf_counter() >= deadline:
            break

        # Build elite set (Plan §1.1)
        idx_sorted = sorted(range(len(scouts)), key=lambda i: scout_fit[i])
        elite_indices = idx_sorted[: min(E, len(idx_sorted))]

        # Refresh global best from population
        best_idx = idx_sorted[0]
        if scout_fit[best_idx] < best_fit:
            best_fit = scout_fit[best_idx]
            best_pos = scouts[best_idx][:]
            best_bins = [b[:] for b in scout_bins[best_idx]]
            best_binw = scout_binw[best_idx][:]
            no_improve_global = 0
        else:
            no_improve_global += 1

        best_scalar = scalar_from_fit(best_fit)

        # Adaptive exploration when stuck (Plan §1.3)
        base_noise = 0.14 * (1.0 - it / max_iters)
        if no_improve_global >= Sg // 2:
            base_noise *= 1.7
        if base_noise > 0.25:
            base_noise = 0.25

        wf_max = 1.0
        if no_improve_global >= Sg // 2:
            wf_max = 1.5

        # Partial restart on global stagnation (Plan §1.2)
        if no_improve_global >= Sg:
            no_improve_global = 0
            worst_count = max(1, (len(scouts) + 3) // 4)  # ceil 25%
            worst_indices = idx_sorted[-worst_count:]
            for wi in worst_indices:
                if time.perf_counter() >= deadline:
                    break
                pos_new = reinit_scout(strategy=0 if random.random() < 0.6 else 1)
                bins, binw, fitv, timed_out = decode_portfolio(pos_new, it, max_iters, best_bins, best_binw)
                if timed_out:
                    return {"packing": best_bins, "bin_weights": best_binw}
                scouts[wi] = pos_new
                scout_fit[wi] = fitv
                scout_bins[wi] = bins
                scout_binw[wi] = binw
                no_improve_i[wi] = 0

        # Update scouts (steady-state)
        for i in range(len(scouts)):
            if time.perf_counter() >= deadline:
                break

            # Scout-level stagnation handling (Plan §1.2)
            if no_improve_i[i] >= Si:
                no_improve_i[i] = 0
                if random.random() < 0.5:
                    # opposition-based
                    old = scouts[i]
                    pos = [clamp01(lo + hi - x + random.uniform(-0.05, 0.05)) for x in old]
                else:
                    pos = reinit_scout(strategy=0 if random.random() < 0.7 else 2)

                bins, binw, fitv, timed_out = decode_portfolio(pos, it, max_iters, best_bins, best_binw)
                if timed_out:
                    return {"packing": best_bins, "bin_weights": best_binw}

                scouts[i] = pos
                scout_fit[i] = fitv
                scout_bins[i] = bins
                scout_binw[i] = binw

                if fitv < best_fit:
                    best_fit = fitv
                    best_pos = pos[:]
                    best_bins = [b[:] for b in bins]
                    best_binw = binw[:]
                    no_improve_global = 0
                continue

            pos = scouts[i]
            fit_i = scout_fit[i]
            scalar_i = scalar_from_fit(fit_i)

            # wf depends on gap
            if scalar_i <= best_scalar:
                wf = 0.0
            else:
                gap = (scalar_i - best_scalar) / (best_scalar + 1e-9)
                wf = gap
                if wf > wf_max:
                    wf = wf_max

            # Choose leader from elites (Plan §1.1)
            leader_pos = pick_leader(elite_indices)

            # Structured dimension updates (Plan §1.4)
            idxset = None
            block = None
            if n > 200:
                upd = max(60, n // 5)
                # bias: 70% from heavy, 30% from the rest
                heavy_k = min(len(heavy_indices), int(0.7 * upd))
                rest_k = upd - heavy_k
                idxs = []
                if heavy_k > 0:
                    idxs.extend(random.sample(heavy_indices, heavy_k))
                if rest_k > 0:
                    # sample from remaining indices
                    # create a cheap pool by slicing a shuffled permutation of idx_by_weight_desc
                    # but avoid heavy part
                    pool = idx_by_weight_desc[heavy_cut:]
                    if rest_k >= len(pool):
                        idxs.extend(pool)
                    else:
                        idxs.extend(random.sample(pool, rest_k))
                idxset = set(idxs)

                # occasional block perturbation over a weight group
                if weight_groups and random.random() < 0.12:
                    block = random.choice(weight_groups)

            r = random.random()
            noise_amp = base_noise

            new_pos = pos[:]  # copy then modify selected dims

            # Apply updates
            if idxset is None:
                # small n: update all
                for d in range(n):
                    x = new_pos[d]
                    step = r * wf * (leader_pos[d] - x)
                    y = x + step + random.uniform(-noise_amp, noise_amp)
                    new_pos[d] = clamp01(y)
            else:
                for d in idxset:
                    x = new_pos[d]
                    step = r * wf * (leader_pos[d] - x)
                    y = x + step + random.uniform(-noise_amp, noise_amp)
                    new_pos[d] = clamp01(y)

                if block is not None:
                    # perturb whole group together (same delta)
                    delta = random.uniform(-noise_amp, noise_amp)
                    for d in block:
                        new_pos[d] = clamp01(new_pos[d] + delta)

            bins, binw, fit_new, timed_out = decode_portfolio(new_pos, it, max_iters, best_bins, best_binw)
            if timed_out:
                return {"packing": best_bins, "bin_weights": best_binw}

            if fit_new <= fit_i:
                scouts[i] = new_pos
                scout_fit[i] = fit_new
                scout_bins[i] = bins
                scout_binw[i] = binw
                no_improve_i[i] = 0

                if fit_new < best_fit:
                    best_fit = fit_new
                    best_pos = new_pos[:]
                    best_bins = [b[:] for b in bins]
                    best_binw = binw[:]
                    no_improve_global = 0
            else:
                no_improve_i[i] += 1

    return {"packing": best_bins, "bin_weights": best_binw}
