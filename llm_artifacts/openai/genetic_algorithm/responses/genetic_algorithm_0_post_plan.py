import time
import random
from bisect import bisect_left

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    start = time.perf_counter()
    deadline = start + max(0.0, float(time_limit))

    # ---- Lower bound (cheap) ----
    total_w = sum(weights)
    lb1 = (total_w + C - 1) // C
    lb = lb1  # can be extended, keep cheap and safe

    # ---- Helpers: order from keys ----
    idxs = list(range(n))

    def order_from_keys(keys):
        # Tie by index for determinism
        return sorted(idxs, key=lambda i: (keys[i], i))

    # ---- Decoder implementations ----
    # We will evaluate an individual by trying a few deterministic decode variants
    # and taking the best fitness among them.

    BUCKET_THRESHOLD = 50000  # plan: buckets up to threshold else bisect list

    def _decode_bf_buckets(order, stop_if_bins_gt=None, window=None):
        # Best-Fit using remaining-capacity buckets.
        # window: if set, once best rem found, optionally allow scanning up to best+window
        #         and choose the tightest (still deterministic). Here, since scanning is
        #         from w..C, the first hit is tightest; window not needed for BF buckets.
        packing = []
        bin_weights = []
        # buckets[rem] -> list of bin indices with that remaining
        buckets = [[] for _ in range(C + 1)]

        for it in order:
            w = weights[it]
            # Find smallest rem >= w
            chosen_bin = -1
            chosen_rem = -1
            # scan upward
            for rem in range(w, C + 1):
                if buckets[rem]:
                    chosen_bin = buckets[rem].pop()
                    chosen_rem = rem
                    break

            if chosen_bin < 0:
                # open new bin
                b = len(bin_weights)
                bin_weights.append(w)
                packing.append([it])
                rem = C - w
                buckets[rem].append(b)
                if stop_if_bins_gt is not None and (b + 1) > stop_if_bins_gt:
                    return None, None, b + 1  # aborted
            else:
                packing[chosen_bin].append(it)
                bin_weights[chosen_bin] += w
                new_rem = chosen_rem - w
                buckets[new_rem].append(chosen_bin)

        return packing, bin_weights, len(bin_weights)

    def _decode_ff(order, stop_if_bins_gt=None):
        # First-Fit in opening order.
        packing = []
        bin_weights = []
        rem = []
        for it in order:
            w = weights[it]
            placed = False
            for b in range(len(rem)):
                if rem[b] >= w:
                    rem[b] -= w
                    bin_weights[b] += w
                    packing[b].append(it)
                    placed = True
                    break
            if not placed:
                b = len(rem)
                rem.append(C - w)
                bin_weights.append(w)
                packing.append([it])
                if stop_if_bins_gt is not None and (b + 1) > stop_if_bins_gt:
                    return None, None, b + 1
        return packing, bin_weights, len(bin_weights)

    def _decode_bf_bisect(order, stop_if_bins_gt=None, lookahead=0):
        # Best-Fit using sorted list of (remaining, bin_index)
        # lookahead: deterministic limited lookahead among near-tight bins.
        rem_sorted = []
        packing = []
        bin_weights = []

        for it in order:
            w = weights[it]
            pos = bisect_left(rem_sorted, (w, -1))
            if pos >= len(rem_sorted):
                b = len(bin_weights)
                bin_weights.append(w)
                packing.append([it])
                rem = C - w
                ins = bisect_left(rem_sorted, (rem, b))
                rem_sorted.insert(ins, (rem, b))
                if stop_if_bins_gt is not None and (b + 1) > stop_if_bins_gt:
                    return None, None, b + 1
            else:
                # choose the tightest; if lookahead>0 choose best among a small window deterministically
                chosen = pos
                if lookahead > 0:
                    end = min(len(rem_sorted), pos + 1 + lookahead)
                    # pick minimal remaining (which is pos) - so lookahead only matters if we choose among equals;
                    # we deterministically pick smallest bin index among those with same rem.
                    best_rem, best_b = rem_sorted[pos]
                    for j in range(pos + 1, end):
                        r, b = rem_sorted[j]
                        if r != best_rem:
                            break
                        if b < best_b:
                            best_b = b
                            chosen = j
                rem, b = rem_sorted.pop(chosen)
                packing[b].append(it)
                bin_weights[b] += w
                new_rem = rem - w
                ins = bisect_left(rem_sorted, (new_rem, b))
                rem_sorted.insert(ins, (new_rem, b))

        return packing, bin_weights, len(bin_weights)

    use_buckets = (C <= BUCKET_THRESHOLD)

    def decode_variants(order, stop_if_bins_gt=None):
        # Try a few deterministic policies; return best (packing, bin_weights)
        # Early abort if all variants exceed stop_if_bins_gt.
        best_pack = None
        best_bw = None
        best_k = 10**18
        best_fit = None

        # Variant A: Best-Fit (bucketed or bisect)
        if use_buckets:
            p, bw, k = _decode_bf_buckets(order, stop_if_bins_gt=stop_if_bins_gt)
        else:
            p, bw, k = _decode_bf_bisect(order, stop_if_bins_gt=stop_if_bins_gt, lookahead=0)
        if p is not None:
            fit = fitness_from_bin_weights(bw)
            best_pack, best_bw, best_k, best_fit = p, bw, k, fit
        else:
            best_k = min(best_k, k)

        # Variant B: First-Fit
        p, bw, k = _decode_ff(order, stop_if_bins_gt=stop_if_bins_gt)
        if p is not None:
            fit = fitness_from_bin_weights(bw)
            if best_fit is None or fit < best_fit:
                best_pack, best_bw, best_k, best_fit = p, bw, k, fit
        else:
            best_k = min(best_k, k)

        # Variant C: Best-Fit with deterministic lookahead (only meaningful on bisect)
        if not use_buckets:
            p, bw, k = _decode_bf_bisect(order, stop_if_bins_gt=stop_if_bins_gt, lookahead=3)
            if p is not None:
                fit = fitness_from_bin_weights(bw)
                if best_fit is None or fit < best_fit:
                    best_pack, best_bw, best_k, best_fit = p, bw, k, fit
            else:
                best_k = min(best_k, k)

        return best_pack, best_bw

    # ---- Fitness ----
    def fitness_from_bin_weights(bin_weights_list):
        k = len(bin_weights_list)
        total_waste = k * C - total_w
        thr = C // 2
        big_waste = 0
        sq = 0
        for bw in bin_weights_list:
            waste = C - bw
            if waste >= thr:
                big_waste += 1
            sq += waste * waste
        return (k, total_waste, big_waste, sq)

    # ---- Build an order from an existing packing (for memetic step conversion) ----
    def order_from_packing(packing, bin_weights_list):
        # bins sorted by fill descending; inside bin weights descending
        bins = list(range(len(packing)))
        bins.sort(key=lambda b: (-bin_weights_list[b], b))
        out = []
        for b in bins:
            items_in_bin = packing[b]
            items_in_bin.sort(key=lambda i: (-weights[i], i))
            out.extend(items_in_bin)
        return out

    def keys_from_order(order):
        # Assign ranks with tiny jitter to avoid ties
        keys = [0.0] * n
        inv = [0] * n
        for pos, it in enumerate(order):
            inv[it] = pos
        # normalize to [0,1)
        denom = max(1, n)
        for i in range(n):
            keys[i] = (inv[i] + (random.random() * 1e-6)) / denom
        return keys

    # ---- Signatures for diversity ----
    def signature_from_keys(keys, k_bins_hint=None):
        # Simple and cheap: hash of first m items in induced order + bins used hint
        m = 24 if n >= 24 else n
        order = sorted(idxs, key=lambda i: (keys[i], i))
        h = 1469598103934665603
        for j in range(m):
            v = order[j] + 1
            h ^= v
            h *= 1099511628211
            h &= (1 << 64) - 1
        return (k_bins_hint if k_bins_hint is not None else -1, h)

    # ---- Initialization (seeds) ----
    by_w_desc = sorted(idxs, key=lambda i: (-weights[i], i))
    by_w_asc = sorted(idxs, key=lambda i: (weights[i], i))

    # heavy-light interleave
    heavy_light = []
    l, r = 0, n - 1
    tmp = by_w_asc[:]  # ascending
    while l <= r:
        heavy_light.append(tmp[r]); r -= 1
        if l <= r:
            heavy_light.append(tmp[l]); l += 1

    # Population parameters
    if n <= 300:
        mu = 80
    elif n <= 1000:
        mu = 60
    else:
        mu = 40
    lam = 10
    tournament_k = 3

    # base mutation settings
    pm_base = 0.30
    noise_base = 0.15

    # Memetic rates
    rr_rate_base = 0.30

    # ---- Incumbent best ----
    best_fit = None
    best_packing = None
    best_bin_weights = None
    best_bins = 10**18

    # ---- Evaluate an individual keys -> fitness, and maybe update incumbent with full packing ----
    def evaluate_keys(keys, stop_if_bins_gt=None, want_packing_for_best=False):
        order = order_from_keys(keys)
        pack, bw = decode_variants(order, stop_if_bins_gt=stop_if_bins_gt)
        if pack is None:
            # dominated / aborted: create a weak fitness
            k = stop_if_bins_gt + 1 if stop_if_bins_gt is not None else (n + 1)
            return (k, k * C - total_w, k, k * C), None, None
        fit = fitness_from_bin_weights(bw)
        return fit, pack, bw

    # ---- Individual structure ----
    # store: (fitness_tuple, keys, bins_used, signature)

    def make_individual_from_order(order):
        keys = keys_from_order(order)
        fit, pack, bw = evaluate_keys(keys, stop_if_bins_gt=None)
        k = fit[0]
        sig = signature_from_keys(keys, k)
        return [fit, keys, k, sig], pack, bw

    def make_weight_biased_keys(alpha):
        # key ~ alpha * weight_rank + (1-alpha)*rand
        # lower key => earlier.
        # Build rank by descending weight
        rank = [0] * n
        for rpos, it in enumerate(by_w_desc):
            rank[it] = rpos
        denom = max(1, n - 1)
        keys = [0.0] * n
        for i in range(n):
            base = rank[i] / denom
            keys[i] = alpha * base + (1.0 - alpha) * random.random()
        return keys

    # ---- Build initial population with diverse seeds ----
    population = []
    sig_set = set()

    def try_add_individual(ind, pack=None, bw=None):
        nonlocal best_fit, best_packing, best_bin_weights, best_bins
        fit, keys, k, sig = ind
        # enforce some uniqueness
        if sig in sig_set and len(population) >= mu // 2:
            return False
        sig_set.add(sig)
        population.append(ind)
        # update incumbent if improved
        if best_fit is None or fit < best_fit:
            if pack is None or bw is None:
                _, pack2, bw2 = fit, None, None
                # reconstruct packing for the best
                fit2, pack2, bw2 = evaluate_keys(keys, stop_if_bins_gt=None)
                pack, bw = pack2, bw2
                fit = fit2
                ind[0] = fit
                ind[2] = fit[0]
                ind[3] = signature_from_keys(keys, fit[0])
            best_fit = fit
            best_bins = fit[0]
            best_packing = pack
            best_bin_weights = bw
        return True

    # Seed orders
    for order in (by_w_desc, by_w_asc, heavy_light):
        ind, pack, bw = make_individual_from_order(order[:])
        try_add_individual(ind, pack, bw)

    # Weight-biased random keys
    for a in (0.70, 0.85, 0.95):
        keys = make_weight_biased_keys(a)
        fit, pack, bw = evaluate_keys(keys)
        ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
        try_add_individual(ind, pack, bw)

    # Random permutations / random keys fill
    while len(population) < mu:
        keys = [random.random() for _ in range(n)]
        fit, pack, bw = evaluate_keys(keys)
        ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
        try_add_individual(ind, pack, bw)
        if time.perf_counter() >= deadline:
            break

    # ---- Selection ----
    def tournament_select():
        best = None
        for _ in range(tournament_k):
            cand = population[random.randrange(len(population))]
            if best is None or cand[0] < best[0]:
                best = cand
        return best

    # ---- Crossover operators on random keys ----
    def crossover_uniform(kA, kB):
        child = [0.0] * n
        for i in range(n):
            child[i] = kA[i] if (random.random() < 0.5) else kB[i]
        return child

    def crossover_blend(kA, kB):
        a = random.uniform(0.3, 0.7)
        child = [0.0] * n
        for i in range(n):
            child[i] = a * kA[i] + (1.0 - a) * kB[i]
        return child

    def bix_crossover(parentA_keys, parentB_keys):
        # Bin Inheritance Crossover:
        # decode parentA, take its best-filled r bins, keep those items grouped first,
        # then append remaining items ordered by parentB keys.
        orderA = order_from_keys(parentA_keys)
        packA, bwA = decode_variants(orderA, stop_if_bins_gt=None)
        if packA is None:
            return crossover_uniform(parentA_keys, parentB_keys)

        # pick r
        u = random.random()
        if u < 0.6:
            r = 1
        elif u < 0.9:
            r = 2
        else:
            r = 3
        r = min(r, len(packA))

        bins = list(range(len(packA)))
        bins.sort(key=lambda b: (-bwA[b], b))
        chosen_bins = bins[:r]

        fixed_items = []
        fixed_set = set()
        for b in chosen_bins:
            # keep bin items in descending weight to preserve structure
            items_in_bin = packA[b][:]
            items_in_bin.sort(key=lambda i: (-weights[i], i))
            for it in items_in_bin:
                fixed_items.append(it)
                fixed_set.add(it)

        rest = [i for i in idxs if i not in fixed_set]
        rest.sort(key=lambda i: (parentB_keys[i], i))

        child_order = fixed_items + rest
        return keys_from_order(child_order)

    # ---- Mutation: targeted key perturbation ----
    def mutate(keys, pm, noise_scale):
        if random.random() >= pm:
            return keys
        out = keys[:]  # copy
        # choose t with geometric-ish distribution
        u = random.random()
        if u < 0.55:
            t = 1
        elif u < 0.80:
            t = 2
        elif u < 0.93:
            t = 4
        else:
            t = 8
        t = min(t, n)
        for _ in range(t):
            i = random.randrange(n)
            if random.random() < 0.30:
                out[i] = random.random()
            else:
                out[i] = out[i] + (random.random() - 0.5) * noise_scale
        return out

    # ---- Memetic improvement: Ruin & Recreate ----
    def ruin_recreate_from_packing(packing, bin_weights_list, time_check_deadline):
        # choose r bins to ruin (emptiest bias)
        k = len(packing)
        if k <= 1:
            return packing, bin_weights_list

        u = random.random()
        if u < 0.6:
            r = 1
        elif u < 0.9:
            r = 2
        else:
            r = 3
        r = min(r, k)

        bins = list(range(k))
        # emptiest first: low fill
        bins.sort(key=lambda b: (bin_weights_list[b], b))
        ruin_bins = bins[:r]
        ruin_set = set(ruin_bins)

        removed = []
        new_packing = []
        new_bw = []
        for b in range(k):
            if b in ruin_set:
                removed.extend(packing[b])
            else:
                new_packing.append(packing[b][:])
                new_bw.append(bin_weights_list[b])

        # Reinsert removed items by descending weight using Best-Fit
        removed.sort(key=lambda i: (-weights[i], i))

        if use_buckets and C <= BUCKET_THRESHOLD:
            # Build buckets for current bins
            buckets = [[] for _ in range(C + 1)]
            for b in range(len(new_bw)):
                buckets[C - new_bw[b]].append(b)

            for it in removed:
                if time.perf_counter() >= time_check_deadline:
                    break
                w = weights[it]
                chosen_bin = -1
                chosen_rem = -1
                for rem in range(w, C + 1):
                    if buckets[rem]:
                        chosen_bin = buckets[rem].pop()
                        chosen_rem = rem
                        break
                if chosen_bin < 0:
                    b = len(new_bw)
                    new_bw.append(w)
                    new_packing.append([it])
                    buckets[C - w].append(b)
                else:
                    new_packing[chosen_bin].append(it)
                    new_bw[chosen_bin] += w
                    buckets[chosen_rem - w].append(chosen_bin)
        else:
            # Bisect-based best-fit structure
            rem_sorted = []
            for b in range(len(new_bw)):
                rem = C - new_bw[b]
                ins = bisect_left(rem_sorted, (rem, b))
                rem_sorted.insert(ins, (rem, b))

            for it in removed:
                if time.perf_counter() >= time_check_deadline:
                    break
                w = weights[it]
                pos = bisect_left(rem_sorted, (w, -1))
                if pos >= len(rem_sorted):
                    b = len(new_bw)
                    new_bw.append(w)
                    new_packing.append([it])
                    rem = C - w
                    ins = bisect_left(rem_sorted, (rem, b))
                    rem_sorted.insert(ins, (rem, b))
                else:
                    rem, b = rem_sorted.pop(pos)
                    new_packing[b].append(it)
                    new_bw[b] += w
                    new_rem = rem - w
                    ins = bisect_left(rem_sorted, (new_rem, b))
                    rem_sorted.insert(ins, (new_rem, b))

        return new_packing, new_bw

    # ---- Steady-state loop ----
    # Fixed iteration budget, but stop by time. Also stop early if hit LB.
    max_iterations = 200000  # fixed cap; time will typically cut earlier
    evals_since_check = 0
    stagnation = 0
    best_bins_last = best_bins

    def time_up():
        return time.perf_counter() >= deadline

    # Ensure we have at least one individual
    if not population:
        # Fallback
        pack, bw = decode_variants(by_w_desc)
        return {"packing": pack, "bin_weights": bw}

    # Main loop
    it = 0
    while it < max_iterations:
        it += 1

        # Early optimality vs LB
        if best_bins <= lb:
            break

        # periodic time check
        evals_since_check += 1
        if (evals_since_check >= 25) and time_up():
            break
        if evals_since_check >= 25:
            evals_since_check = 0

        # adaptive parameters on stagnation
        if best_bins < best_bins_last:
            best_bins_last = best_bins
            stagnation = 0
        else:
            stagnation += 1

        pm = pm_base
        noise = noise_base
        rr_rate = rr_rate_base
        if stagnation > 200:
            pm = min(0.70, pm_base + 0.25)
            noise = min(0.50, noise_base + 0.20)
            rr_rate = min(0.60, rr_rate_base + 0.30)

        # Create a small offspring batch (lambda)
        offspring = []
        for _ in range(lam):
            if time_up():
                break
            pA = tournament_select()
            pB = tournament_select()
            kA = pA[1]
            kB = pB[1]

            r = random.random()
            if r < 0.60:
                child_keys = crossover_uniform(kA, kB)
            elif r < 0.80:
                child_keys = crossover_blend(kA, kB)
            else:
                child_keys = bix_crossover(kA, kB)

            child_keys = mutate(child_keys, pm, noise)

            # Evaluate with early abort if already worse than incumbent (bin count)
            stop_bins = (best_bins - 1) if best_bins < 10**18 else None
            fit, pack, bw = evaluate_keys(child_keys, stop_if_bins_gt=stop_bins)
            if pack is None:
                # Aborted / weak; still may keep for diversity only rarely
                ind = [fit, child_keys, fit[0], signature_from_keys(child_keys, fit[0])]
                offspring.append((ind, None, None))
                continue

            # Memetic R&R on a fraction of offspring (canonical memetic GA for BPP)
            if random.random() < rr_rate:
                if time_up():
                    pass
                else:
                    # Apply R&R with a tight time check budget
                    rr_deadline = min(deadline, time.perf_counter() + 0.005)
                    new_pack, new_bw = ruin_recreate_from_packing(pack, bw, rr_deadline)
                    new_fit = fitness_from_bin_weights(new_bw)
                    if new_fit < fit:
                        # convert improved packing back to keys
                        new_order = order_from_packing(new_pack, new_bw)
                        child_keys = keys_from_order(new_order)
                        fit, pack, bw = new_fit, new_pack, new_bw

            ind = [fit, child_keys, fit[0], signature_from_keys(child_keys, fit[0])]
            offspring.append((ind, pack, bw))

        # Insert offspring, remove worst to keep mu; enforce some uniqueness
        for ind, pack, bw in offspring:
            if time_up():
                break

            # If too similar to many, optionally force extra mutation once
            if ind[3] in sig_set and random.random() < 0.50:
                ind[1] = mutate(ind[1], pm=1.0, noise_scale=min(0.6, noise + 0.25))
                fit2, pack2, bw2 = evaluate_keys(ind[1], stop_if_bins_gt=(best_bins - 1))
                if pack2 is not None:
                    ind[0] = fit2
                    ind[2] = fit2[0]
                    ind[3] = signature_from_keys(ind[1], fit2[0])
                    pack, bw = pack2, bw2

            # Insert if unique enough or if it's good
            if ind[3] not in sig_set or ind[0] < max(population, key=lambda x: x[0])[0]:
                try_add_individual(ind, pack, bw)

            # Maintain population size by removing worst
            if len(population) > mu:
                # remove worst (linear scan, mu small)
                worst_idx = 0
                worst_fit = population[0][0]
                for i in range(1, len(population)):
                    if population[i][0] > worst_fit:
                        worst_fit = population[i][0]
                        worst_idx = i
                worst = population.pop(worst_idx)
                # rebuild sig_set occasionally to avoid drift (cheap at mu<=80)
                if random.random() < 0.05:
                    sig_set = set(p[3] for p in population)
                else:
                    sig_set.discard(worst[3])

        # Stagnation restart/injection
        if stagnation > 400 and len(population) >= mu:
            # Inject 5-10% fresh individuals
            inject = max(2, mu // 10)
            for _ in range(inject):
                if time_up():
                    break
                if random.random() < 0.5:
                    keys = make_weight_biased_keys(random.choice([0.70, 0.85, 0.95]))
                else:
                    keys = [random.random() for _ in range(n)]
                fit, pack, bw = evaluate_keys(keys, stop_if_bins_gt=(best_bins - 1))
                if pack is None:
                    continue
                ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
                try_add_individual(ind, pack, bw)
            stagnation = 0

    # Safety reconstruct for incumbent (already stored), but ensure valid
    if best_packing is None or best_bin_weights is None:
        # fallback
        order = by_w_desc
        best_packing, best_bin_weights = decode_variants(order)

    return {"packing": best_packing, "bin_weights": best_bin_weights}
