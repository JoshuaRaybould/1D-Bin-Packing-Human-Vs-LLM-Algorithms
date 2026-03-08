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

    total_w = sum(wts)

    # ---------------- Lower bounds ----------------
    lb_vol = (total_w + C - 1) // C
    count_big = sum(1 for x in wts if x > C // 2)

    # A safe extra bound inspired by MT L3 ideas:
    # Let A = items in (C/2, C], each needs its own bin.
    # Let B = items in (C/3, C/2], at most one can accompany an A item.
    # Unpaired B items then require packing with <= C/3 items, which is harder.
    # A simple safe tightening:
    #   lb = max(lb_vol, |A| + ceil(max(0, |B|-|A|)/2))
    # because two (C/3, C/2] items cannot always fit together (they might),
    # but using /2 is still safe as a lower bound (optimistic). We can further
    # strengthen by using volume of remaining after reserving |A| bins.
    A = [x for x in wts if x > C / 2]
    B = [x for x in wts if (C / 3) < x <= (C / 2)]
    lb_mt = len(A)
    if len(B) > len(A):
        lb_mt += (len(B) - len(A) + 1) // 2

    lb = max(lb_vol, count_big, lb_mt)

    idxs = list(range(n))
    by_w_desc = sorted(idxs, key=lambda i: (-wts[i], i))
    by_w_asc = sorted(idxs, key=lambda i: (wts[i], i))

    # ---------------- Decoder helpers ----------------
    BUCKET_THRESHOLD = 70000
    use_buckets = (C <= BUCKET_THRESHOLD)
    min_w = min(wts)

    def fitness_from_bin_weights(bin_weights_list):
        # Lexicographic fitness
        k = len(bin_weights_list)
        total_waste = k * C - total_w
        # hard gaps: big holes or tiny holes (tiny holes are unusable for most items)
        hard = 0
        sq = 0
        for bw in bin_weights_list:
            rem = C - bw
            if rem >= C / 2:
                hard += 1
            elif 0 < rem < min_w:
                hard += 1
            sq += rem * rem
        return (k, total_waste, hard, sq)

    def order_from_keys(keys):
        return sorted(idxs, key=lambda i: (keys[i], i))

    # --- core best-fit decoders ---
    def _decode_bf_bisect(order, stop_if_bins_gt=None):
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
                    return None, None
            else:
                rem, b = rem_sorted.pop(pos)
                packing[b].append(it)
                bin_weights[b] += w
                new_rem = rem - w
                rem_sorted.insert(bisect_left(rem_sorted, (new_rem, b)), (new_rem, b))
        return packing, bin_weights

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
                    return None, None
            else:
                packing[chosen_bin].append(it)
                bin_weights[chosen_bin] += w
                buckets[chosen_rem - w].append(chosen_bin)
        return packing, bin_weights

    # A slightly smarter decoder: best-fit with a small lookahead score.
    # When multiple bins can take the item, prefer the placement that creates
    # a remainder close to some item weight (encourages future exact fills).
    # Implemented cheaply by sampling a few candidate bins around the best-fit position.
    def _decode_bf_lookahead(order, stop_if_bins_gt=None):
        # Use bisect structure; sample neighborhood around best-fit
        rem_sorted = []  # (rem, bin)
        packing = []
        bin_weights = []

        # Precompute a small set of "target" remainders: frequent item sizes (or complements)
        # Cheap proxy: take top distinct weights.
        distinct = sorted(set(wts), reverse=True)
        targets = distinct[: min(24, len(distinct))]

        def score_rem(r):
            # smaller is better: distance to nearest target
            # targets are weights; good remainder equals some weight
            best = r
            for t in targets:
                d = abs(r - t)
                if d < best:
                    best = d
                    if best == 0:
                        break
            # also penalize very large holes
            if r >= C / 2:
                best += C * 0.15
            return best

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
                    return None, None
            else:
                # consider a few candidates near pos
                cand_positions = [pos]
                if pos + 1 < len(rem_sorted):
                    cand_positions.append(pos + 1)
                if pos + 2 < len(rem_sorted):
                    cand_positions.append(pos + 2)
                if pos - 1 >= 0:
                    cand_positions.append(pos - 1)

                best_j = None
                best_sc = None
                for j in cand_positions:
                    rem, b = rem_sorted[j]
                    new_rem = rem - w
                    sc = score_rem(new_rem)
                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_j = j

                rem, b = rem_sorted.pop(best_j)
                packing[b].append(it)
                bin_weights[b] += w
                new_rem = rem - w
                rem_sorted.insert(bisect_left(rem_sorted, (new_rem, b)), (new_rem, b))

        return packing, bin_weights

    def decode_variants(order, stop_if_bins_gt=None):
        best = None
        best_fit = None

        # Variant A: BF (fast)
        if use_buckets:
            p, bw = _decode_bf_buckets(order, stop_if_bins_gt)
        else:
            p, bw = _decode_bf_bisect(order, stop_if_bins_gt)
        if p is not None:
            f = fitness_from_bin_weights(bw)
            best, best_fit = (p, bw), f

        # Variant B: Lookahead BF (usually stronger, a bit slower)
        if n >= 60:
            p2, bw2 = _decode_bf_lookahead(order, stop_if_bins_gt)
            if p2 is not None:
                f2 = fitness_from_bin_weights(bw2)
                if best_fit is None or f2 < best_fit:
                    best, best_fit = (p2, bw2), f2

        # Variant C: FFD-like stabilization: sort by (-w, key_pos)
        if n >= 40:
            posmap = {it: j for j, it in enumerate(order)}
            order_ffd = sorted(idxs, key=lambda i: (-wts[i], posmap[i]))
            if use_buckets:
                p3, bw3 = _decode_bf_buckets(order_ffd, stop_if_bins_gt)
            else:
                p3, bw3 = _decode_bf_bisect(order_ffd, stop_if_bins_gt)
            if p3 is not None:
                f3 = fitness_from_bin_weights(bw3)
                if best_fit is None or f3 < best_fit:
                    best, best_fit = (p3, bw3), f3

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
        m = 32 if n >= 32 else n
        order = sorted(idxs, key=lambda i: (keys[i], i))
        h = 1469598103934665603
        for j in range(m):
            v = order[j] + 1
            h ^= v
            h = (h * 1099511628211) & ((1 << 64) - 1)
        return (k_bins_hint if k_bins_hint is not None else -1, h)

    # ---------------- Local improvement (memetic GA; standard for BPP) ----------------
    def _try_place_item_best_fit(item, packing, bin_w, rem_sorted):
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

    def bin_elimination_local_search(packing, bin_w, time_check_deadline, max_passes=3):
        packing = [lst[:] for lst in packing]
        bin_w = bin_w[:]

        for _ in range(max_passes):
            if time.perf_counter() >= time_check_deadline:
                break
            k = len(packing)
            if k <= lb:
                break

            # try multiple of the lightest bins
            bins = list(range(k))
            bins.sort(key=lambda b: (bin_w[b], b))
            candidates = bins[: min(7, k)]

            improved = False
            for b0 in candidates:
                if time.perf_counter() >= time_check_deadline:
                    break
                if not packing[b0]:
                    continue

                items0 = packing[b0][:]
                items0.sort(key=lambda i: (-wts[i], i))

                # Build rem structure excluding b0
                rem_sorted = []
                for b in range(k):
                    if b == b0:
                        continue
                    rem = C - bin_w[b]
                    rem_sorted.insert(bisect_left(rem_sorted, (rem, b)), (rem, b))

                moved = []
                ok = True
                for it in items0:
                    if time.perf_counter() >= time_check_deadline:
                        ok = False
                        break
                    if _try_place_item_best_fit(it, packing, bin_w, rem_sorted):
                        moved.append(it)
                    else:
                        ok = False
                        break

                if ok:
                    packing.pop(b0)
                    bin_w.pop(b0)
                    improved = True
                    break

                # rollback moved items (fast rollback by reconstructing b0 and undoing placements)
                # To keep it simple and safe, reconstruct from scratch for a short ejection chain attempt.
                packing2 = [lst[:] for lst in packing]
                bin_w2 = bin_w[:]

                # remove b0 contents
                items0 = packing2[b0][:]
                packing2[b0].clear()
                bin_w2[b0] = 0

                # helper to attempt depth-2 ejection: place it into some bin by ejecting one item
                def try_place_with_eject(it):
                    w = wts[it]
                    # scan a few tight bins
                    # Build list of bins that can take it directly first
                    best_direct = None
                    best_rem = None
                    for b in range(len(packing2)):
                        if b == b0:
                            continue
                        rem = C - bin_w2[b]
                        if rem >= w:
                            if best_rem is None or rem - w < best_rem:
                                best_rem = rem - w
                                best_direct = b
                    if best_direct is not None:
                        packing2[best_direct].append(it)
                        bin_w2[best_direct] += w
                        return True

                    # otherwise attempt eject from a bin: pick a bin where removing x makes room
                    # and then place x elsewhere directly.
                    bins_by_rem = sorted(
                        [b for b in range(len(packing2)) if b != b0],
                        key=lambda b: (C - bin_w2[b])
                    )
                    trials = 0
                    for b in bins_by_rem[: min(10, len(bins_by_rem))]:
                        rem = C - bin_w2[b]
                        need = w - rem
                        # choose an item x with weight >= need
                        for j, x in enumerate(packing2[b]):
                            if trials >= 35:
                                return False
                            trials += 1
                            wx = wts[x]
                            if wx < need:
                                continue
                            # eject x
                            packing2[b].pop(j)
                            bin_w2[b] -= wx
                            # place it
                            packing2[b].append(it)
                            bin_w2[b] += w

                            # now try to place x somewhere else directly
                            placed_x = False
                            best_b = None
                            best_r = None
                            for bb in range(len(packing2)):
                                if bb == b0:
                                    continue
                                if bb == b:
                                    continue
                                rrem = C - bin_w2[bb]
                                if rrem >= wx:
                                    rr = rrem - wx
                                    if best_r is None or rr < best_r:
                                        best_r = rr
                                        best_b = bb
                            if best_b is not None:
                                packing2[best_b].append(x)
                                bin_w2[best_b] += wx
                                placed_x = True

                            if placed_x:
                                return True

                            # revert and continue
                            packing2[b].pop()  # remove it
                            bin_w2[b] -= w
                            packing2[b].insert(j, x)
                            bin_w2[b] += wx

                    return False

                items0_sorted = sorted(items0, key=lambda i: (-wts[i], i))
                ok2 = True
                for it in items0_sorted:
                    if time.perf_counter() >= time_check_deadline:
                        ok2 = False
                        break
                    if not try_place_with_eject(it):
                        ok2 = False
                        break

                if ok2:
                    # b0 eliminated
                    packing2.pop(b0)
                    bin_w2.pop(b0)
                    packing, bin_w = packing2, bin_w2
                    improved = True
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
        a = random.uniform(0.15, 0.85)
        child = [0.0] * n
        for i in range(n):
            child[i] = a * kA[i] + (1.0 - a) * kB[i]
        return child

    def crossover_two_point_order(kA, kB):
        # Order-based crossover on permutations induced by keys.
        # Standard in order-based GA for BPP.
        oA = order_from_keys(kA)
        oB = order_from_keys(kB)
        a = random.randrange(n)
        b = random.randrange(n)
        if a > b:
            a, b = b, a
        segment = oA[a:b]
        segset = set(segment)
        child_order = segment + [x for x in oB if x not in segset]
        return keys_from_order(child_order)

    def bix_crossover(kA, kB):
        orderA = order_from_keys(kA)
        packA, bwA = decode_variants(orderA, stop_if_bins_gt=None)
        if packA is None:
            return crossover_two_point_order(kA, kB)

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
        # mixed mutation: a few gaussian-like jitters + occasional resets
        u = random.random()
        if u < 0.40:
            t = 1
        elif u < 0.70:
            t = 2
        elif u < 0.90:
            t = 5
        else:
            t = 10
        t = min(t, n)
        for _ in range(t):
            i = random.randrange(n)
            if random.random() < 0.18:
                out[i] = random.random()
            else:
                out[i] = out[i] + (random.random() - 0.5) * noise_scale
        return out

    # ---------------- Initialization ----------------
    heavy_light = []
    l, r = 0, n - 1
    tmp = by_w_asc[:]
    while l <= r:
        heavy_light.append(tmp[r])
        r -= 1
        if l <= r:
            heavy_light.append(tmp[l])
            l += 1

    # Larger populations help with long budgets; keep manageable for huge n.
    if n <= 300:
        mu = 140
        lam = 70
    elif n <= 1200:
        mu = 110
        lam = 55
    else:
        mu = 90
        lam = 45

    tournament_k = 3
    elite_keep = max(6, mu // 9)

    pm_base = 0.24
    noise_base = 0.22
    ls_rate_base = 0.45

    best_fit = None
    best_packing = None
    best_bin_w = None
    best_bins = 10**18

    population = []
    sig_set = set()

    def try_add(ind, pack=None, bw=None):
        nonlocal best_fit, best_packing, best_bin_w, best_bins
        fit, keys, _, sig = ind
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

    for a in (0.60, 0.70, 0.80, 0.88, 0.93, 0.97):
        keys = make_weight_biased_keys(a)
        fit, pack, bw = evaluate_keys(keys)
        ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
        try_add(ind, pack, bw)

    # Fill remaining with biased keys
    while len(population) < mu and time.perf_counter() < deadline:
        if random.random() < 0.72:
            keys = make_weight_biased_keys(random.choice([0.75, 0.85, 0.93, 0.97]))
        else:
            keys = [random.random() for _ in range(n)]
        fit, pack, bw = evaluate_keys(keys)
        ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
        try_add(ind, pack, bw)

    if not population:
        pack, bw = decode_variants(by_w_desc)
        return {"packing": pack, "bin_weights": bw}

    def time_up():
        return time.perf_counter() >= deadline

    # Fixed iteration cap + periodic time checks.
    max_iterations = 420000 if n <= 600 else 520000

    evals_since_check = 0
    stagnation = 0
    best_bins_last = best_bins

    it = 0
    while it < max_iterations:
        it += 1

        if best_bins <= lb:
            break

        evals_since_check += 1
        if evals_since_check >= 45:
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

        if stagnation > 350:
            pm = min(0.75, pm_base + 0.28)
            noise = min(0.75, noise_base + 0.30)
            ls_rate = min(0.70, ls_rate_base + 0.20)

        population.sort(key=lambda ind: ind[0])
        elites = population[:elite_keep]

        offspring = []
        for _ in range(lam):
            if time_up():
                break

            pA = tournament_select(population, tournament_k)
            pB = tournament_select(population, tournament_k)
            kA, kB = pA[1], pB[1]

            r = random.random()
            if r < 0.38:
                child_keys = crossover_two_point_order(kA, kB)
            elif r < 0.62:
                child_keys = crossover_uniform(kA, kB)
            elif r < 0.80:
                child_keys = crossover_blend(kA, kB)
            else:
                child_keys = bix_crossover(kA, kB)

            # Adaptive mutation: stronger when close to best bins
            if best_bins < 10**18 and random.random() < 0.35:
                pm_eff = min(0.85, pm + 0.10)
                noise_eff = min(0.85, noise + 0.10)
            else:
                pm_eff = pm
                noise_eff = noise

            child_keys = mutate(child_keys, pm_eff, noise_eff)

            stop_bins = (best_bins - 1) if best_bins < 10**18 else None
            fit, pack, bw = evaluate_keys(child_keys, stop_if_bins_gt=stop_bins)

            if pack is not None and random.random() < ls_rate:
                ls_deadline = min(deadline, time.perf_counter() + 0.006)
                pack2, bw2 = bin_elimination_local_search(pack, bw, ls_deadline, max_passes=3)
                fit2 = fitness_from_bin_weights(bw2)
                if fit2 < fit:
                    pack, bw, fit = pack2, bw2, fit2
                    # rebuild keys from packing
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

        # Replacement with diversity
        new_pop = elites[:]
        new_sig = set(ind[3] for ind in new_pop)

        candidates = []
        candidates.extend((ind, None, None) for ind in population[elite_keep:])
        candidates.extend(offspring)
        candidates.sort(key=lambda x: x[0][0])

        for ind, pack, bw in candidates:
            if len(new_pop) >= mu:
                break
            sig = ind[3]
            if sig in new_sig and len(new_pop) >= mu // 3:
                continue
            new_pop.append(ind)
            new_sig.add(sig)

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

        # Diversification restart on long stagnation
        if stagnation > 850 and not time_up():
            inject = max(6, mu // 6)
            for _ in range(inject):
                if time_up():
                    break
                if random.random() < 0.80:
                    keys = make_weight_biased_keys(random.choice([0.80, 0.88, 0.93, 0.97]))
                else:
                    keys = [random.random() for _ in range(n)]
                stop_bins = (best_bins - 1) if best_bins < 10**18 else None
                fit, pack, bw = evaluate_keys(keys, stop_if_bins_gt=stop_bins)
                if pack is None:
                    continue
                if random.random() < 0.35:
                    ls_deadline = min(deadline, time.perf_counter() + 0.004)
                    pack, bw = bin_elimination_local_search(pack, bw, ls_deadline, max_passes=2)
                    fit = fitness_from_bin_weights(bw)
                ind = [fit, keys, fit[0], signature_from_keys(keys, fit[0])]
                try_add(ind, pack, bw)
            stagnation = 0

        # Intensify incumbent periodically
        if (it % 220 == 0) and best_packing is not None and not time_up():
            ls_deadline = min(deadline, time.perf_counter() + 0.010)
            pack2, bw2 = bin_elimination_local_search(best_packing, best_bin_w, ls_deadline, max_passes=4)
            fit2 = fitness_from_bin_weights(bw2)
            if fit2 < best_fit:
                best_fit = fit2
                best_packing = pack2
                best_bin_w = bw2
                best_bins = fit2[0]

    if best_packing is None or best_bin_w is None:
        best_packing, best_bin_w = decode_variants(by_w_desc)

    return {"packing": best_packing, "bin_weights": best_bin_w}
