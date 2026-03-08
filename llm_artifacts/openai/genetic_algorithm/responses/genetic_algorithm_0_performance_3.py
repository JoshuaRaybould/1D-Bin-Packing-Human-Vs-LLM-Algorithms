import time
import random
from bisect import bisect_left


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    wts = weights

    start = time.perf_counter()
    deadline = start + max(0.0, float(time_limit))

    # ---------------- Lower bounds ----------------
    total_w = sum(wts)
    lb_vol = (total_w + C - 1) // C

    # L2 bound: ceil( sum_{i<=k} w_i / C ) where k is largest with sum(w_i)> (k-1)C ???
    # Use a common cheap safe strengthening: L2 = max_{t in (C/2,C]} ceil( (W_gt + sum_small) / C )
    # We'll use: let S = {w > C/2}; these must occupy distinct bins (can pair with small).
    # LB = max(lb_vol, count_big)
    count_big = sum(1 for x in wts if x > C // 2)
    lb = max(lb_vol, count_big)

    idxs = list(range(n))
    by_w_desc = sorted(idxs, key=lambda i: (-wts[i], i))

    # ---------------- Decoders ----------------
    # We will use three classic order-based decoders:
    # - FFD (first-fit with bins ordered by opening order, but items already sorted by keys order)
    # - BFD (best-fit decreasing style along the given item order)
    # - BF (tightest remaining using sorted list)

    BUCKET_THRESHOLD = 60000
    use_buckets = (C <= BUCKET_THRESHOLD)

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

    def order_from_keys(keys):
        return sorted(idxs, key=lambda i: (keys[i], i))

    def _decode_ff(order, stop_if_bins_gt=None):
        packing = []
        bin_weights = []
        rem = []
        for it in order:
            w = wts[it]
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

    def _decode_bf_bisect(order, stop_if_bins_gt=None):
        # Sorted by remaining capacity
        rem_sorted = []  # (rem, bin_index)
        packing = []
        bin_weights = []
        for it in order:
            w = wts[it]
            pos = bisect_left(rem_sorted, (w, -1))
            if pos >= len(rem_sorted):
                b = len(bin_weights)
                bin_weights.append(w)
                packing.append([it])
                rem = C - w
                rem_sorted.insert(bisect_left(rem_sorted, (rem, b)), (rem, b))
                if stop_if_bins_gt is not None and (b + 1) > stop_if_bins_gt:
                    return None, None, b + 1
            else:
                rem, b = rem_sorted.pop(pos)
                packing[b].append(it)
                bin_weights[b] += w
                new_rem = rem - w
                rem_sorted.insert(bisect_left(rem_sorted, (new_rem, b)), (new_rem, b))
        return packing, bin_weights, len(bin_weights)

    def _decode_bf_buckets(order, stop_if_bins_gt=None):
        packing = []
        bin_weights = []
        buckets = [[] for _ in range(C + 1)]
        for it in order:
            w = wts[it]
            chosen_bin = -1
            chosen_rem = -1
            for rem in range(w, C + 1):
                if buckets[rem]:
                    chosen_bin = buckets[rem].pop()
                    chosen_rem = rem
                    break
            if chosen_bin < 0:
                b = len(bin_weights)
                bin_weights.append(w)
                packing.append([it])
                buckets[C - w].append(b)
                if stop_if_bins_gt is not None and (b + 1) > stop_if_bins_gt:
                    return None, None, b + 1
            else:
                packing[chosen_bin].append(it)
                bin_weights[chosen_bin] += w
                buckets[chosen_rem - w].append(chosen_bin)
        return packing, bin_weights, len(bin_weights)

    def decode_variants(order, stop_if_bins_gt=None):
        best = None
        best_fit = None

        # Variant 1: Best-Fit
        if use_buckets:
            p, bw, _ = _decode_bf_buckets(order, stop_if_bins_gt=stop_if_bins_gt)
        else:
            p, bw, _ = _decode_bf_bisect(order, stop_if_bins_gt=stop_if_bins_gt)
        if p is not None:
            f = fitness_from_bin_weights(bw)
            best, best_fit = (p, bw), f

        # Variant 2: First-Fit
        p, bw, _ = _decode_ff(order, stop_if_bins_gt=stop_if_bins_gt)
        if p is not None:
            f = fitness_from_bin_weights(bw)
            if best_fit is None or f < best_fit:
                best, best_fit = (p, bw), f

        # Variant 3: FFD-like: sort by weight desc but keep key order within weight bands
        # (helps on instances where keys are noisy)
        if n >= 40:
            # stable sort by (-weight, key)
            # create local keys map for order
            pos = {it: j for j, it in enumerate(order)}
            order_ffd = sorted(idxs, key=lambda i: (-wts[i], pos[i]))
            if use_buckets:
                p, bw, _ = _decode_bf_buckets(order_ffd, stop_if_bins_gt=stop_if_bins_gt)
            else:
                p, bw, _ = _decode_bf_bisect(order_ffd, stop_if_bins_gt=stop_if_bins_gt)
            if p is not None:
                f = fitness_from_bin_weights(bw)
                if best_fit is None or f < best_fit:
                    best, best_fit = (p, bw), f

        if best is None:
            return None, None
        return best

    # ---------------- Key helpers ----------------
    def keys_from_order(order):
        keys = [0.0] * n
        inv = [0] * n
        for pos, it in enumerate(order):
            inv[it] = pos
        denom = max(1, n)
        for i in range(n):
            keys[i] = (inv[i] + random.random() * 1e-6) / denom
        return keys

    def signature_from_keys(keys, k_bins_hint=None):
        m = 28 if n >= 28 else n
        order = sorted(idxs, key=lambda i: (keys[i], i))
        h = 1469598103934665603
        for j in range(m):
            v = order[j] + 1
            h ^= v
            h = (h * 1099511628211) & ((1 << 64) - 1)
        return (k_bins_hint if k_bins_hint is not None else -1, h)

    # ---------------- Local improvement: bin elimination ----------------
    # Classic memetic component for BPP: try to empty a light bin by relocating its items.
    # We keep it bounded and deterministic-ish (only small search) to be safe on time.

    def _try_place_item_best_fit(item, packing, bin_w, rem_sorted):
        # rem_sorted: sorted list of (rem, bin)
        w = wts[item]
        pos = bisect_left(rem_sorted, (w, -1))
        if pos >= len(rem_sorted):
            return False
        rem, b = rem_sorted.pop(pos)
        packing[b].append(item)
        bin_w[b] += w
        new_rem = rem - w
        rem_sorted.insert(bisect_left(rem_sorted, (new_rem, b)), (new_rem, b))
        return True

    def bin_elimination_local_search(packing, bin_w, time_check_deadline, max_passes=2):
        # Work on a copy
        packing = [lst[:] for lst in packing]
        bin_w = bin_w[:]

        for _pass in range(max_passes):
            if time.perf_counter() >= time_check_deadline:
                break
            k = len(packing)
            if k <= lb:
                break

            # Candidate bins: lightest few
            bins = list(range(k))
            bins.sort(key=lambda b: (bin_w[b], b))
            candidates = bins[: min(4, k)]
            improved = False

            for b0 in candidates:
                if time.perf_counter() >= time_check_deadline:
                    break

                items = packing[b0][:]
                if not items:
                    continue
                # Try to move heavier items first
                items.sort(key=lambda i: (-wts[i], i))

                # Build rem structure excluding b0
                rem_sorted = []
                for b in range(k):
                    if b == b0:
                        continue
                    rem = C - bin_w[b]
                    rem_sorted.insert(bisect_left(rem_sorted, (rem, b)), (rem, b))

                # Attempt pure relocations
                moved_all = True
                for it in items:
                    if time.perf_counter() >= time_check_deadline:
                        moved_all = False
                        break
                    if not _try_place_item_best_fit(it, packing, bin_w, rem_sorted):
                        moved_all = False
                        break

                if moved_all:
                    # Remove emptied bin b0
                    packing.pop(b0)
                    bin_w.pop(b0)
                    improved = True
                    break

                # Limited swap attempt: for first failed item, try swapping with one item from a tight bin
                # Revert and try one swap only to keep time bounded.
                # Reconstruct state from scratch for swap attempt.
                packing2 = [lst[:] for lst in packing]
                bin_w2 = bin_w[:]

                # remove items in b0 temporarily
                items0 = packing2[b0][:]
                packing2[b0].clear()
                bin_w2[b0] = 0

                # target fill b0 via taking one item from another bin to make room elsewhere
                items0.sort(key=lambda i: (-wts[i], i))

                # Build rem for other bins
                rem_sorted2 = []
                for b in range(k):
                    if b == b0:
                        continue
                    rem = C - bin_w2[b]
                    rem_sorted2.insert(bisect_left(rem_sorted2, (rem, b)), (rem, b))

                failed_item = None
                for it in items0:
                    if time.perf_counter() >= time_check_deadline:
                        break
                    if not _try_place_item_best_fit(it, packing2, bin_w2, rem_sorted2):
                        failed_item = it
                        break

                if failed_item is None:
                    # all moved (should have been caught)
                    packing2.pop(b0)
                    bin_w2.pop(b0)
                    packing, bin_w = packing2, bin_w2
                    improved = True
                    break

                # One swap attempt
                # Choose a bin with small remaining (hard bin) and try swapping one of its items with failed_item.
                # This is a standard small neighborhood in BPP memetic search.
                hard_bins = list(range(len(packing2)))
                hard_bins.sort(key=lambda b: (C - bin_w2[b], b))
                tried = 0
                for hb in hard_bins:
                    if time.perf_counter() >= time_check_deadline:
                        break
                    if hb == b0:
                        continue
                    # if hb has any item that could be swapped
                    for j, x in enumerate(packing2[hb]):
                        if tried >= 25:
                            break
                        tried += 1
                        wx = wts[x]
                        wf = wts[failed_item]
                        # Swap feasibility: put failed into hb and x elsewhere (best-fit)
                        if bin_w2[hb] - wx + wf > C:
                            continue

                        # Create temp state for this swap
                        pk = [lst[:] for lst in packing2]
                        bwk = bin_w2[:]

                        # perform swap inside hb
                        pk[hb].pop(j)
                        bwk[hb] -= wx
                        pk[hb].append(failed_item)
                        bwk[hb] += wf

                        # now try to place x using best-fit among all bins except b0 (which is being eliminated)
                        rem_sortedk = []
                        for b in range(len(pk)):
                            if b == b0:
                                continue
                            rem_sortedk.insert(bisect_left(rem_sortedk, (C - bwk[b], b)), (C - bwk[b], b))

                        # remove current remaining for hb (will be reinserted by placement call when it pops)
                        # simplest: just attempt placement with rem_sortedk directly
                        if _try_place_item_best_fit(x, pk, bwk, rem_sortedk):
                            # then try to place the rest of items0 after failed_item (not placed)
                            rest = [it for it in items0 if it != failed_item]
                            rest.sort(key=lambda i: (-wts[i], i))
                            ok = True
                            # b0 currently empty
                            for it2 in rest:
                                if time.perf_counter() >= time_check_deadline:
                                    ok = False
                                    break
                                if not _try_place_item_best_fit(it2, pk, bwk, rem_sortedk):
                                    ok = False
                                    break
                            if ok:
                                # eliminate b0
                                pk.pop(b0)
                                bwk.pop(b0)
                                packing, bin_w = pk, bwk
                                improved = True
                                break
                    if improved:
                        break

                if improved:
                    break

            if not improved:
                break

        return packing, bin_w

    # ---------------- GA structures ----------------
    def evaluate_keys(keys, stop_if_bins_gt=None):
        order = order_from_keys(keys)
        pack, bw = decode_variants(order, stop_if_bins_gt=stop_if_bins_gt)
        if pack is None:
            k = (stop_if_bins_gt + 1) if stop_if_bins_gt is not None else (n + 1)
            return (k, k * C - total_w, k, k * C), None, None
        fit = fitness_from_bin_weights(bw)
        return fit, pack, bw

    def make_weight_biased_keys(alpha):
        rank = [0] * n
        for rpos, it in enumerate(by_w_desc):
            rank[it] = rpos
        denom = max(1, n - 1)
        keys = [0.0] * n
        for i in range(n):
            base = rank[i] / denom
            keys[i] = alpha * base + (1.0 - alpha) * random.random()
        return keys

    def tournament_select(pop, k=3):
        best = None
        for _ in range(k):
            cand = pop[random.randrange(len(pop))]
            if best is None or cand[0] < best[0]:
                best = cand
        return best

    def crossover_uniform(kA, kB):
        child = [0.0] * n
        for i in range(n):
            child[i] = kA[i] if (random.random() < 0.5) else kB[i]
        return child

    def crossover_blend(kA, kB):
        a = random.uniform(0.2, 0.8)
        child = [0.0] * n
        for i in range(n):
            child[i] = a * kA[i] + (1.0 - a) * kB[i]
        return child

    def bix_crossover(kA, kB):
        # Bin inheritance: inherit a few fullest bins from A, complete using B order
        orderA = order_from_keys(kA)
        packA, bwA = decode_variants(orderA, stop_if_bins_gt=None)
        if packA is None:
            return crossover_uniform(kA, kB)

        u = random.random()
        if u < 0.55:
            r = 1
        elif u < 0.85:
            r = 2
        else:
            r = 3
        r = min(r, len(packA))

        bins = list(range(len(packA)))
        bins.sort(key=lambda b: (-bwA[b], b))
        chosen = bins[:r]

        fixed = []
        fixed_set = set()
        for b in chosen:
            items = packA[b][:]
            items.sort(key=lambda i: (-wts[i], i))
            for it in items:
                fixed.append(it)
                fixed_set.add(it)

        rest = [i for i in idxs if i not in fixed_set]
        rest.sort(key=lambda i: (kB[i], i))
        return keys_from_order(fixed + rest)

    def mutate(keys, pm, noise_scale):
        if random.random() >= pm:
            return keys
        out = keys[:]
        u = random.random()
        if u < 0.45:
            t = 1
        elif u < 0.75:
            t = 2
        elif u < 0.92:
            t = 4
        else:
            t = 8
        t = min(t, n)
        for _ in range(t):
            i = random.randrange(n)
            if random.random() < 0.25:
                out[i] = random.random()
            else:
                out[i] = out[i] + (random.random() - 0.5) * noise_scale
        return out

    # ---------------- Initialization ----------------
    by_w_asc = sorted(idxs, key=lambda i: (wts[i], i))

    heavy_light = []
    l, r = 0, n - 1
    tmp = by_w_asc[:]
    while l <= r:
        heavy_light.append(tmp[r])
        r -= 1
        if l <= r:
            heavy_light.append(tmp[l])
            l += 1

    if n <= 300:
        mu = 90
        lam = 30
    elif n <= 1200:
        mu = 70
        lam = 24
    else:
        mu = 55
        lam = 20

    tournament_k = 3
    elite_keep = max(4, mu // 10)

    pm_base = 0.28
    noise_base = 0.18

    # Local search rate (memetic GA)
    ls_rate_base = 0.35

    # Incumbent
    best_fit = None
    best_packing = None
    best_bin_w = None
    best_bins = 10**18

    population = []
    sig_set = set()

    def try_add(ind, pack=None, bw=None):
        nonlocal best_fit, best_packing, best_bin_w, best_bins, sig_set
        fit, keys, k, sig = ind
        if sig in sig_set and len(population) >= mu // 2:
            return False
        sig_set.add(sig)
        population.append(ind)
        if best_fit is None or fit < best_fit:
            if pack is None or bw is None:
                fit2, pack2, bw2 = evaluate_keys(keys)
                fit = fit2
                pack = pack2
                bw = bw2
                ind[0] = fit
                ind[2] = fit[0]
                ind[3] = signature_from_keys(keys, fit[0])
            best_fit = fit
            best_packing = pack
            best_bin_w = bw
            best_bins = fit[0]
        return True

    def add_seed_order(order):
        keys = keys_from_order(order)
        fit, pack, bw = evaluate_keys(keys)
        ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
        try_add(ind, pack, bw)

    add_seed_order(by_w_desc)
    add_seed_order(by_w_asc)
    add_seed_order(heavy_light)

    for a in (0.65, 0.75, 0.85, 0.93, 0.97):
        keys = make_weight_biased_keys(a)
        fit, pack, bw = evaluate_keys(keys)
        ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
        try_add(ind, pack, bw)

    # Fill remaining
    while len(population) < mu and time.perf_counter() < deadline:
        if random.random() < 0.55:
            keys = make_weight_biased_keys(random.choice([0.75, 0.85, 0.93, 0.97]))
        else:
            keys = [random.random() for _ in range(n)]
        fit, pack, bw = evaluate_keys(keys)
        ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
        try_add(ind, pack, bw)

    if not population:
        order = by_w_desc
        pack, bw = decode_variants(order)
        return {"packing": pack, "bin_weights": bw}

    # ---------------- Main loop ----------------
    def time_up():
        return time.perf_counter() >= deadline

    # Fixed iteration cap (requirement). Chosen to be large; time checks will stop earlier.
    # Increase for large n to keep searching; still fixed once computed.
    max_iterations = 250000 if n <= 600 else 350000

    evals_since_check = 0
    stagnation = 0
    best_bins_last = best_bins

    it = 0
    while it < max_iterations:
        it += 1

        if best_bins <= lb:
            break

        evals_since_check += 1
        if evals_since_check >= 30:
            evals_since_check = 0
            if time_up():
                break

        if best_bins < best_bins_last:
            best_bins_last = best_bins
            stagnation = 0
        else:
            stagnation += 1

        pm = pm_base
        noise = noise_base
        ls_rate = ls_rate_base

        if stagnation > 300:
            pm = min(0.70, pm_base + 0.25)
            noise = min(0.65, noise_base + 0.25)
            ls_rate = min(0.65, ls_rate_base + 0.25)

        # Elitism: keep best few
        population.sort(key=lambda ind: ind[0])
        elites = population[:elite_keep]

        # Generate offspring
        offspring = []
        for _ in range(lam):
            if time_up():
                break
            pA = tournament_select(population, tournament_k)
            pB = tournament_select(population, tournament_k)
            kA, kB = pA[1], pB[1]

            r = random.random()
            if r < 0.50:
                child_keys = crossover_uniform(kA, kB)
            elif r < 0.75:
                child_keys = crossover_blend(kA, kB)
            else:
                child_keys = bix_crossover(kA, kB)

            child_keys = mutate(child_keys, pm, noise)

            stop_bins = (best_bins - 1) if best_bins < 10**18 else None
            fit, pack, bw = evaluate_keys(child_keys, stop_if_bins_gt=stop_bins)

            if pack is not None and random.random() < ls_rate:
                # bounded local search budget per offspring
                ls_deadline = min(deadline, time.perf_counter() + 0.004)
                pack2, bw2 = bin_elimination_local_search(pack, bw, ls_deadline, max_passes=2)
                fit2 = fitness_from_bin_weights(bw2)
                if fit2 < fit:
                    pack, bw, fit = pack2, bw2, fit2
                    # derive keys from improved packing: bins by fill desc, items by wt desc
                    bins = list(range(len(pack)))
                    bins.sort(key=lambda b: (-bw[b], b))
                    order = []
                    for b in bins:
                        items = pack[b][:]
                        items.sort(key=lambda i: (-wts[i], i))
                        order.extend(items)
                    child_keys = keys_from_order(order)

            ind = [fit, child_keys, fit[0], signature_from_keys(child_keys, fit[0])]
            offspring.append((ind, pack, bw))

        # Replacement: (mu + offspring) -> keep elites + best diverse rest
        # Start with elites
        new_pop = elites[:]
        new_sig = set(ind[3] for ind in new_pop)

        # Combine candidates (current + offspring) but bias toward offspring
        candidates = []
        candidates.extend((ind, None, None) for ind in population[elite_keep:])
        candidates.extend(offspring)

        # Sort by fitness
        candidates.sort(key=lambda x: x[0][0])

        for ind, pack, bw in candidates:
            if len(new_pop) >= mu:
                break
            sig = ind[3]
            if sig in new_sig:
                # allow duplicates only if very good and population is still small
                if len(new_pop) < mu // 3:
                    pass
                else:
                    continue
            new_pop.append(ind)
            new_sig.add(sig)

            # Update incumbent
            if best_fit is None or ind[0] < best_fit:
                if pack is None or bw is None:
                    fit2, pack2, bw2 = evaluate_keys(ind[1])
                    ind[0] = fit2
                    ind[2] = fit2[0]
                    ind[3] = signature_from_keys(ind[1], fit2[0])
                    pack, bw = pack2, bw2
                best_fit = ind[0]
                best_packing = pack
                best_bin_w = bw
                best_bins = best_fit[0]

        population = new_pop
        sig_set = new_sig

        # Diversification on stagnation: inject random/biased individuals
        if stagnation > 700 and not time_up():
            inject = max(3, mu // 8)
            for _ in range(inject):
                if time_up():
                    break
                if random.random() < 0.65:
                    keys = make_weight_biased_keys(random.choice([0.75, 0.85, 0.93, 0.97]))
                else:
                    keys = [random.random() for _ in range(n)]
                stop_bins = (best_bins - 1) if best_bins < 10**18 else None
                fit, pack, bw = evaluate_keys(keys, stop_if_bins_gt=stop_bins)
                if pack is None:
                    continue
                # occasional local search on injected too
                if random.random() < 0.25:
                    ls_deadline = min(deadline, time.perf_counter() + 0.003)
                    pack, bw = bin_elimination_local_search(pack, bw, ls_deadline, max_passes=1)
                    fit = fitness_from_bin_weights(bw)
                ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
                try_add(ind, pack, bw)
            stagnation = 0

        # Occasionally intensify incumbent
        if (it % 250 == 0) and best_packing is not None and not time_up():
            ls_deadline = min(deadline, time.perf_counter() + 0.006)
            pack2, bw2 = bin_elimination_local_search(best_packing, best_bin_w, ls_deadline, max_passes=2)
            fit2 = fitness_from_bin_weights(bw2)
            if fit2 < best_fit:
                best_fit = fit2
                best_packing = pack2
                best_bin_w = bw2
                best_bins = fit2[0]

    if best_packing is None or best_bin_w is None:
        order = by_w_desc
        best_packing, best_bin_w = decode_variants(order)

    return {"packing": best_packing, "bin_weights": best_bin_w}
