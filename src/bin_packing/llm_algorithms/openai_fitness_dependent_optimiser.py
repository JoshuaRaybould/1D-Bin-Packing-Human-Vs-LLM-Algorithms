# openai
# fitness_dependent_optimiser_0_post_plan.py

import time
import random
from typing import List, Tuple, Dict, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = bin_capacity
    # --- (1) Instrumentation ---
    total_w = sum(weights)
    LB = (total_w + C - 1) // C
    max_w = max(weights)
    # If invalid instance (item larger than capacity) we still return something sensible:
    # put each item in its own bin.
    if max_w > C:
        packing = [[i] for i in range(n)]
        bw = [weights[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bw}

    # Counters (internal)
    decodes_full = 0
    decodes_partial = 0
    accepted_moves = 0
    last_improvement_time = start

    items = list(range(n))

    # Global weight rank for heavy-prefix operations
    weight_rank = sorted(items, key=lambda i: weights[i], reverse=True)
    rank_pos = [0] * n
    for r, it in enumerate(weight_rank):
        rank_pos[it] = r

    # ---------- (2) Strong decoder: bucketed Best-Fit with hardness tie-break + lookahead ----------

    def decode_best_fit_lookahead(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        """Order-based deterministic decoder with enhanced tie-breaks.
        Returns (packing, bin_weights)."""
        nonlocal decodes_full
        decodes_full += 1

        bins: List[List[int]] = []
        bw: List[int] = []

        # frequency of remaining items by weight
        freq = [0] * (C + 1)
        for idx in order:
            freq[weights[idx]] += 1

        # buckets[rem] -> list of bin indices whose remaining capacity equals rem
        buckets: List[List[int]] = [[] for _ in range(C + 1)]

        # For each bin b, track current remaining capacity
        rem_of: List[int] = []

        def hardness(r: int) -> float:
            # Cheap proxy: prefer remainders that are easy to fill (low hardness)
            # so we penalize remainders lacking matching weights.
            if r <= 0:
                return 0.0
            h = 0.0
            # 1/(1+freq) => larger when freq small => harder
            h += 1.0 / (1.0 + freq[r])
            if r - 1 >= 1:
                h += 1.0 / (1.0 + freq[r - 1])
            if r + 1 <= C:
                h += 1.0 / (1.0 + freq[r + 1])
            return h

        # candidate search: find best rem >= w by scanning buckets from w upward
        def find_best_bin_for(w: int, use_lookahead: bool) -> int:
            # return bin index or -1
            best_b = -1
            best_rem2 = C + 1
            best_h = 1e18

            # Optionally collect top2 candidates for lookahead
            cand1 = -1
            cand2 = -1
            cand1_rem2 = C + 1
            cand2_rem2 = C + 1
            cand1_h = 1e18
            cand2_h = 1e18

            for rem in range(w, C + 1):
                if not buckets[rem]:
                    continue
                # For this remainder rem, all those bins give rem2 = rem-w.
                rem2 = rem - w
                h = hardness(rem2)

                # We need best by (rem2, hardness, older bin)
                # Pick the oldest bin (smallest index) among this remainder bucket.
                b_oldest = buckets[rem][0]

                better_local = (rem2 < best_rem2) or (rem2 == best_rem2 and (h < best_h or (h == best_h and b_oldest < best_b)))
                if better_local:
                    best_b = b_oldest
                    best_rem2 = rem2
                    best_h = h

                # Track top2 distinct candidates if lookahead enabled
                if use_lookahead:
                    # Compare against cand1/cand2 by same criteria
                    def better_tuple(rem2a, ha, ba, rem2b, hb, bb):
                        return (rem2a < rem2b) or (rem2a == rem2b and (ha < hb or (ha == hb and ba < bb)))

                    if cand1 == -1 or better_tuple(rem2, h, b_oldest, cand1_rem2, cand1_h, cand1):
                        cand2, cand2_rem2, cand2_h = cand1, cand1_rem2, cand1_h
                        cand1, cand1_rem2, cand1_h = b_oldest, rem2, h
                    elif b_oldest != cand1 and (cand2 == -1 or better_tuple(rem2, h, b_oldest, cand2_rem2, cand2_h, cand2)):
                        cand2, cand2_rem2, cand2_h = b_oldest, rem2, h

                # Early stop: perfect fit
                if best_rem2 == 0:
                    break

            if not use_lookahead or cand2 == -1:
                return best_b

            # (2.3) 1-step lookahead for large items: choose between cand1 and cand2
            # using complement availability proxy: prefer rem2 that has an exact complement in remaining.
            # Evaluate a small score; lower is better.
            def look_score(rem2: int) -> float:
                # encourage exact fill: if there is an item of weight rem2 remaining, good.
                # Otherwise use hardness.
                if rem2 <= 0:
                    return -10.0
                exact_bonus = -2.0 * (1.0 if freq[rem2] > 0 else 0.0)
                near_bonus = -0.5 * (1.0 if (rem2 - 1 >= 1 and freq[rem2 - 1] > 0) else 0.0)
                near_bonus += -0.5 * (1.0 if (rem2 + 1 <= C and freq[rem2 + 1] > 0) else 0.0)
                return hardness(rem2) + exact_bonus + near_bonus

            s1 = look_score(cand1_rem2)
            s2 = look_score(cand2_rem2)
            if s2 < s1 - 1e-12:
                return cand2
            return cand1

        # How many early items to allow lookahead for
        # Keep it modest and scale slightly with n.
        K = 30 if n < 150 else 50 if n < 500 else 80

        for t, idx in enumerate(order):
            w = weights[idx]
            freq[w] -= 1

            use_look = (t < K and w * 10 >= 6 * C)  # w >= 0.6C
            b = find_best_bin_for(w, use_lookahead=use_look)
            if b == -1:
                # open new bin
                b = len(bins)
                bins.append([idx])
                bw.append(w)
                rem = C - w
                rem_of.append(rem)
                buckets[rem].append(b)
            else:
                # remove b from old bucket (swap-remove in list)
                old_rem = rem_of[b]
                lst = buckets[old_rem]
                # find and remove b (bucket sizes tend to be small)
                # keep order in bucket: older first, so removal by linear scan.
                for k in range(len(lst)):
                    if lst[k] == b:
                        lst.pop(k)
                        break

                bins[b].append(idx)
                bw[b] += w
                new_rem = old_rem - w
                rem_of[b] = new_rem
                buckets[new_rem].append(b)

        return bins, bw

    # (7.1) Partial screening: decode only heavy prefix and count bins opened
    def partial_bins_used(order: List[int], m: int) -> int:
        nonlocal decodes_partial
        decodes_partial += 1
        if m <= 0:
            return 0
        # Use simple best-fit scan for partial; keep it cheap.
        bw_local: List[int] = []
        for idx in order[:m]:
            w = weights[idx]
            best = -1
            best_rem2 = C + 1
            for b in range(len(bw_local)):
                rem = C - bw_local[b]
                if w <= rem:
                    rem2 = rem - w
                    if rem2 < best_rem2:
                        best_rem2 = rem2
                        best = b
                        if rem2 == 0:
                            break
            if best == -1:
                bw_local.append(w)
            else:
                bw_local[best] += w
        return len(bw_local)

    # ---------- Fitness / comparison ----------
    def fitness_from(bins: List[List[int]], bw: List[int]) -> Tuple[int, int]:
        waste = 0
        for s in bw:
            waste += (C - s)
        return (len(bins), waste)

    def better(f1: Tuple[int, int], f2: Tuple[int, int]) -> bool:
        return f1[0] < f2[0] or (f1[0] == f2[0] and f1[1] < f2[1])

    # ---------- (6) Better initialization: multiple deterministic seed orders ----------
    items_desc = sorted(items, key=lambda i: weights[i], reverse=True)
    items_asc = sorted(items, key=lambda i: weights[i])

    def order_by_mod(k: int) -> List[int]:
        # group by weight mod k (descending mod group, then descending weight)
        return sorted(items, key=lambda i: ((weights[i] % k), weights[i]), reverse=True)

    def order_by_half_closeness() -> List[int]:
        # prioritize medium items close to C/2 (descending distance closeness => small |w-C/2| first)
        half = C / 2.0
        return sorted(items, key=lambda i: (abs(weights[i] - half), -weights[i]))

    def heavy_front_random_tail() -> List[int]:
        # keep heavy prefix fixed by weight rank, shuffle the rest
        cut = min(n, 25)
        head = items_desc[:cut]
        tail = items_desc[cut:]
        random.shuffle(tail)
        return head + tail

    seed_orders: List[List[int]] = []
    seed_orders.append(items_desc)
    seed_orders.append(items_asc)
    seed_orders.append(order_by_half_closeness())
    for k in (3, 4, 5):
        if C >= k:
            seed_orders.append(order_by_mod(k))
    seed_orders.append(heavy_front_random_tail())

    # de-dup seeds while preserving order
    seen_seed = set()
    uniq_seeds = []
    for o in seed_orders:
        t = tuple(o)
        if t not in seen_seed:
            seen_seed.add(t)
            uniq_seeds.append(o)
    seed_orders = uniq_seeds

    # ---------- Population representation with cached decode structure (4.1) ----------
    # Each scout stores: order, fit, bins, bw, signature
    class Scout:
        __slots__ = ("order", "fit", "bins", "bw", "sig")

        def __init__(self, order: List[int], bins: List[List[int]], bw: List[int], fit: Tuple[int, int], sig: Tuple[int, ...]):
            self.order = order
            self.bins = bins
            self.bw = bw
            self.fit = fit
            self.sig = sig

    def signature_from_bw(bw: List[int]) -> Tuple[int, ...]:
        # packing-based signature: sorted bin weights, plus a short prefix of largest bins
        # (tuple kept reasonably small)
        s = sorted(bw)
        if len(s) <= 24:
            return tuple(s)
        top = tuple(sorted(s[-6:], reverse=True))
        # include length to avoid ambiguity
        return (len(s),) + top + tuple(s[:6])

    # ---------- Permutation perturbations / mutations ----------
    def random_perturb(base: List[int]) -> List[int]:
        a = base[:]
        if n <= 1:
            return a
        k = 3 if n < 120 else 4
        for _ in range(k):
            r = random.random()
            if r < 0.6:
                i = random.randrange(n)
                j = random.randrange(n)
                a[i], a[j] = a[j], a[i]
            else:
                i = random.randrange(n)
                j = random.randrange(n)
                if i > j:
                    i, j = j, i
                if j - i >= 2:
                    a[i:j] = reversed(a[i:j])
        return a

    def mutate(order: List[int], rate: float) -> List[int]:
        a = order[:]
        if n <= 1:
            return a
        swaps = 1 if rate <= 0.22 else 2 if rate <= 0.55 else 3
        for _ in range(swaps):
            r = random.random()
            if r < 0.72:
                i = random.randrange(n)
                j = random.randrange(n)
                a[i], a[j] = a[j], a[i]
            else:
                i = random.randrange(n)
                j = random.randrange(n)
                if i > j:
                    i, j = j, i
                if j - i >= 2:
                    a[i:j] = reversed(a[i:j])
        return a

    # Precompute complement groups from a packing (4.3)
    def tight_groups_from_packing(bins: List[List[int]], bw: List[int], max_groups: int = 20) -> List[List[int]]:
        # Prefer bins with small waste and small group sizes (2-3 items)
        groups = []
        candidates = list(range(len(bins)))
        candidates.sort(key=lambda b: (C - bw[b], len(bins[b])))
        for b in candidates:
            if len(groups) >= max_groups:
                break
            if len(bins[b]) < 2 or len(bins[b]) > 4:
                continue
            waste = C - bw[b]
            if waste > max(3, C // 30):
                continue
            groups.append(bins[b][:])
        return groups

    def move_groups_consecutive(order: List[int], groups: List[List[int]], how_many: int) -> List[int]:
        if not groups or how_many <= 0:
            return order[:]
        a = order[:]
        pos = {it: i for i, it in enumerate(a)}

        chosen = []
        for _ in range(min(how_many, len(groups))):
            chosen.append(groups[random.randrange(len(groups))])

        # apply each group: remove its items then insert them consecutively
        for g in chosen:
            # ensure unique items
            g2 = list(dict.fromkeys(g))
            # sort by descending weight
            g2.sort(key=lambda it: weights[it], reverse=True)

            # anchor: earliest current position among group items
            anchor = min(pos[it] for it in g2 if it in pos)

            # remove items (in descending index order)
            rem_idx = sorted((pos[it] for it in g2 if it in pos), reverse=True)
            for idx in rem_idx:
                a.pop(idx)
            # clamp anchor
            if anchor > len(a):
                anchor = len(a)
            for j, it in enumerate(g2):
                a.insert(anchor + j, it)

            # rebuild pos locally (groups are few)
            pos = {it: i for i, it in enumerate(a)}
        return a

    # (4.4) Heavy-prefix preservation alignment toward best
    def heavy_prefix_align(order: List[int], best_order: List[int], prefix_len: int) -> List[int]:
        if prefix_len <= 0:
            return order[:]
        prefix_len = min(prefix_len, n)
        # determine the set of heavy items (by global weight rank)
        heavy_set = set(weight_rank[:prefix_len])
        # Extract heavy items in the order they appear in best
        heavy_in_best = [it for it in best_order if it in heavy_set]
        # Build new order: place heavy_in_best first, then the remaining items as they appear in current order
        tail = [it for it in order if it not in heavy_set]
        return heavy_in_best + tail

    # (4.2) Bin-emptying guided relocation: permutation edits based on decoded bins
    def bin_emptying_move(order: List[int], bins: List[List[int]], bw: List[int], pace: float) -> List[int]:
        if len(bins) <= 1:
            return order[:]
        # pick candidate bin: small weight or high waste
        cand_bins = list(range(len(bins)))
        cand_bins.sort(key=lambda b: (bw[b], C - bw[b]), reverse=False)
        # choose among a few, biased by pace (higher pace -> more aggressive on worst bins)
        pick_pool = min(6, len(cand_bins))
        bsel = cand_bins[random.randrange(pick_pool)] if random.random() < 0.75 else cand_bins[-1]

        items_to_move = bins[bsel][:]
        if not items_to_move:
            return order[:]
        # move heavier first
        items_to_move.sort(key=lambda it: weights[it], reverse=True)

        # compute remaining capacities of target bins (excluding selected bin)
        rem = [C - bw[b] for b in range(len(bins))]

        # build a mapping item -> target_bin (greedy best-fit by remaining)
        targets: Dict[int, int] = {}
        for it in items_to_move:
            w = weights[it]
            best_b = -1
            best_rem2 = C + 1
            for b in range(len(bins)):
                if b == bsel:
                    continue
                if w <= rem[b]:
                    rem2 = rem[b] - w
                    if rem2 < best_rem2:
                        best_rem2 = rem2
                        best_b = b
                        if rem2 == 0:
                            break
            if best_b != -1:
                targets[it] = best_b
                rem[best_b] -= w
            else:
                # cannot place; keep it in its own area (no-op for this item)
                targets[it] = -1

        # Convert reassignment to order edits: move each item right after a representative of target bin.
        a = order[:]
        pos = {it: i for i, it in enumerate(a)}

        # representative per target bin: first item in that bin
        rep = {}
        for b in range(len(bins)):
            if b == bsel:
                continue
            if bins[b]:
                rep[b] = bins[b][0]

        # Remove items to move (in descending index order)
        rem_positions = sorted((pos[it] for it in items_to_move if it in pos), reverse=True)
        for p in rem_positions:
            a.pop(p)

        # Recompute positions after removals
        pos = {it: i for i, it in enumerate(a)}

        # Insert items near their reps; add noise depending on (1-pace)
        noise = int((1.0 - pace) * 10)
        for it in items_to_move:
            tb = targets.get(it, -1)
            if tb == -1 or tb not in rep or rep[tb] not in pos:
                # fallback: insert near front for heavy items, else random
                base = min(len(a), 5)
                ins = base if weights[it] * 10 >= 7 * C else random.randrange(len(a) + 1)
            else:
                ins = pos[rep[tb]] + 1
            if noise > 0:
                ins += random.randint(-noise, noise)
            if ins < 0:
                ins = 0
            if ins > len(a):
                ins = len(a)
            a.insert(ins, it)
            # update pos locally (cheap incremental update avoided; rebuild occasionally)
            if len(items_to_move) <= 10:
                pos = {x: i for i, x in enumerate(a)}
        return a

    # ---------- (3) Pace function uses gap-to-LB and normalized waste ----------
    def pace_from(f: Tuple[int, int], bestf: Tuple[int, int]) -> float:
        fb, fw = f
        bb, bwaste = bestf

        gap = fb - LB
        best_gap = bb - LB

        # bins difference dominates
        if fb > bb:
            # worse bin count: strong pull
            base = 0.75 + 0.10 * min(2, fb - bb)
        else:
            # equal bins: pace depends on whether we're above LB and waste
            if best_gap > 0:
                base = 0.35
            else:
                # already at LB: stabilize
                base = 0.18

        # incorporate gap-to-LB
        if gap > best_gap:
            base = min(1.0, base + 0.15)
        elif gap == best_gap and fb == bb and fw > bwaste:
            base = min(1.0, base + 0.08)

        # normalized waste per bin to smooth
        if fb > 0:
            wpn = (fw / fb) / C  # ~ [0,1)
        else:
            wpn = 0.0
        base = min(1.0, base + 0.10 * wpn)

        return max(0.06, min(1.0, base))

    # ---------- Build initial population (6 + 9) ----------
    # Slightly higher pop; keep bounded.
    pop_size = max(30, min(80, 28 + n // 20))

    # Start with best among seeds
    best_order: List[int] = items_desc[:]
    best_packing, best_bw = decode_best_fit_lookahead(best_order)
    best_fit = fitness_from(best_packing, best_bw)
    best_sig = signature_from_bw(best_bw)

    # initialize scouts with seeds + perturbed/random
    scouts: List[Scout] = []

    # Add decoded seeds
    for o in seed_orders:
        bins, bw = decode_best_fit_lookahead(o)
        f = fitness_from(bins, bw)
        sig = signature_from_bw(bw)
        scouts.append(Scout(o[:], bins, bw, f, sig))
        if better(f, best_fit):
            best_fit, best_order, best_packing, best_bw, best_sig = f, o[:], bins, bw, sig

    # Fill the rest
    while len(scouts) < pop_size:
        if len(scouts) % 3 == 0:
            o = random_perturb(best_order)
        elif len(scouts) % 3 == 1:
            o = heavy_front_random_tail()
        else:
            o = items[:]
            random.shuffle(o)
        bins, bw = decode_best_fit_lookahead(o)
        f = fitness_from(bins, bw)
        sig = signature_from_bw(bw)
        scouts.append(Scout(o, bins, bw, f, sig))
        if better(f, best_fit):
            best_fit, best_order, best_packing, best_bw, best_sig = f, o[:], bins, bw, sig

    # complement groups from best for operator 4.3
    best_groups = tight_groups_from_packing(best_packing, best_bw)

    # precomputed best positions (7.2)
    best_pos = [0] * n
    for i, it in enumerate(best_order):
        best_pos[it] = i

    def update_best(order: List[int], bins: List[List[int]], bw: List[int], fit: Tuple[int, int], sig: Tuple[int, ...]):
        nonlocal best_order, best_packing, best_bw, best_fit, best_sig, best_groups, best_pos, last_improvement_time
        best_order = order[:]
        best_packing = bins
        best_bw = bw
        best_fit = fit
        best_sig = sig
        best_groups = tight_groups_from_packing(best_packing, best_bw)
        for i, it in enumerate(best_order):
            best_pos[it] = i
        last_improvement_time = time.time()

    # Guided insertion using cached best_pos
    def guided_insertion(order: List[int], strength: float) -> List[int]:
        if n <= 1:
            return order[:]
        a = order[:]
        m = 1 + int(strength * min(36, n))
        for _ in range(m):
            if random.random() < 0.7:
                i = random.randrange(min(n, 30))
            else:
                i = random.randrange(n)
            item = a[i]
            a.pop(i)
            target = best_pos[item]
            ins = int((target / max(1, n - 1)) * max(1, len(a) - 1))
            noise = int((1.0 - strength) * 9)
            if noise > 0:
                ins += random.randint(-noise, noise)
            if ins < 0:
                ins = 0
            if ins > len(a):
                ins = len(a)
            a.insert(ins, item)
        return a

    # ---------- (5.1) Annealed acceptance on secondary objective ----------
    def accept_move(cur_fit: Tuple[int, int], new_fit: Tuple[int, int], elapsed: float, limit: float) -> bool:
        if better(new_fit, cur_fit):
            return True
        # same bin-count but worse waste: sometimes accept early with threshold
        if new_fit[0] == cur_fit[0] and new_fit[1] > cur_fit[1]:
            progress = 1.0 if limit <= 1e-9 else min(1.0, elapsed / limit)
            p0 = 0.15
            p = p0 * (1.0 - progress) * (1.0 - progress)
            # threshold: allow limited regression
            bins_cnt = max(1, new_fit[0])
            delta = new_fit[1] - cur_fit[1]
            # allow up to ~ (0.03*C per bin) early
            thr = int((0.03 + 0.04 * (1.0 - progress)) * C * bins_cnt)
            if delta <= thr and random.random() < p:
                return True
        # exact tie: tiny acceptance for churn
        if new_fit == cur_fit and random.random() < 0.03:
            return True
        return False

    # ---------- (5.2) Diversity management by signatures ----------
    def signature_counts() -> Dict[Tuple[int, ...], int]:
        d: Dict[Tuple[int, ...], int] = {}
        for sc in scouts:
            d[sc.sig] = d.get(sc.sig, 0) + 1
        return d

    # ---------- Main loop: time-driven with fixed iteration cap (8) ----------
    # Respect provided time_limit strictly.
    effective_limit = max(0.0, time_limit)

    # large fixed iteration budget; time checks will stop earlier
    iter_budget = 20000 if n < 400 else 12000 if n < 1200 else 8000

    check_every = 25

    # adaptive restart trigger
    no_improve_trigger = max(0.15 * effective_limit, 0.8)

    # partial screening parameter
    partial_M = min(n, 120)

    for it in range(iter_budget):
        if it % check_every == 0:
            if (time.time() - start) >= effective_limit:
                break

        elapsed = time.time() - start
        if elapsed >= effective_limit:
            break

        best_bins = best_fit[0]
        best_gap = best_bins - LB

        # occasional diversity repair if many identical packings
        if it % 120 == 119:
            counts = signature_counts()
            # if too concentrated, reinit a couple
            most = max(counts.values()) if counts else 1
            if most >= max(4, pop_size // 5):
                # reinitialize 2-4 scouts (not best)
                k = 2 if pop_size < 50 else 4
                for _ in range(k):
                    if (time.time() - start) >= effective_limit:
                        break
                    idx = random.randrange(pop_size)
                    if scouts[idx].fit == best_fit:
                        continue
                    # new constructive
                    o = heavy_front_random_tail() if random.random() < 0.5 else random_perturb(items_desc)
                    bins, bw = decode_best_fit_lookahead(o)
                    f = fitness_from(bins, bw)
                    sig = signature_from_bw(bw)
                    scouts[idx] = Scout(o, bins, bw, f, sig)
                    if better(f, best_fit):
                        update_best(o, bins, bw, f, sig)

        # (6.2) Adaptive restarts when no improvement for some time
        if effective_limit > 0.0 and (time.time() - last_improvement_time) >= no_improve_trigger and it % 20 == 0:
            # restart 20-35% of population (excluding best)
            frac = 0.22 if best_gap <= 1 else 0.32
            k = max(1, int(frac * pop_size))
            # pick worst by fitness
            idxs = list(range(pop_size))
            idxs.sort(key=lambda i: (scouts[i].fit[0], scouts[i].fit[1]), reverse=True)
            restarted = 0
            for i in idxs:
                if restarted >= k:
                    break
                if (time.time() - start) >= effective_limit:
                    break
                if scouts[i].fit == best_fit:
                    continue
                if random.random() < 0.5:
                    # apply bin-emptying move to best, then perturb
                    o = bin_emptying_move(best_order, best_packing, best_bw, pace=0.85)
                    o = random_perturb(o)
                else:
                    # new randomized constructive
                    base = seed_orders[random.randrange(len(seed_orders))]
                    o = random_perturb(base)
                    if random.random() < 0.25:
                        o = heavy_front_random_tail()
                bins, bw = decode_best_fit_lookahead(o)
                f = fitness_from(bins, bw)
                sig = signature_from_bw(bw)
                scouts[i] = Scout(o, bins, bw, f, sig)
                restarted += 1
                if better(f, best_fit):
                    update_best(o, bins, bw, f, sig)
            last_improvement_time = time.time()  # avoid repeated triggers

        # Iterate scouts
        # small random permutation of scout visitation to reduce bias
        order_scouts = list(range(pop_size))
        random.shuffle(order_scouts)

        for si in order_scouts:
            if (time.time() - start) >= effective_limit:
                break

            sc = scouts[si]
            cur_order = sc.order
            cur_fit = sc.fit

            pace = pace_from(cur_fit, best_fit)

            # (9) Operator schedule driven by bin relation
            if cur_fit == best_fit:
                # best scout: mostly micro-mutate, occasional bin-empty attempt
                if random.random() < 0.70:
                    new_order = mutate(cur_order, rate=0.18)
                else:
                    new_order = bin_emptying_move(cur_order, sc.bins, sc.bw, pace=0.35)
                    if random.random() < 0.30:
                        new_order = mutate(new_order, rate=0.20)
            else:
                if cur_fit[0] > best_fit[0]:
                    # worse in bin count: strong exploitation
                    r = random.random()
                    if r < 0.60:
                        pfx = 20 if n < 200 else 35
                        pfx = min(pfx, n)
                        new_order = heavy_prefix_align(cur_order, best_order, prefix_len=pfx)
                        if random.random() < 0.70:
                            new_order = guided_insertion(new_order, strength=min(1.0, pace))
                        if random.random() < 0.50:
                            new_order = mutate(new_order, rate=0.25 + 0.35 * (1.0 - pace))
                    elif r < 0.90:
                        new_order = guided_insertion(cur_order, strength=min(1.0, pace))
                        if random.random() < 0.55:
                            new_order = mutate(new_order, rate=0.22 + 0.45 * (1.0 - pace))
                    else:
                        new_order = mutate(cur_order, rate=0.45)
                else:
                    # equal bin count: try structure moves to break through
                    r = random.random()
                    if r < 0.40:
                        new_order = bin_emptying_move(cur_order, sc.bins, sc.bw, pace=pace)
                        if random.random() < 0.35:
                            new_order = mutate(new_order, rate=0.22)
                    elif r < 0.80:
                        # complement pairing injection from best
                        how_many = 1 if n < 200 else 2
                        new_order = move_groups_consecutive(cur_order, best_groups, how_many=how_many)
                        if random.random() < 0.40:
                            new_order = mutate(new_order, rate=0.18)
                    else:
                        new_order = mutate(cur_order, rate=0.30)

            # (7.1) Two-stage screening
            # If partial heavy-prefix already opens too many bins vs current, skip full decode.
            # Use a lenient margin.
            if n > 60 and random.random() < 0.85:
                pb_new = partial_bins_used(new_order, partial_M)
                pb_cur = partial_bins_used(cur_order, partial_M)
                # if clearly worse, reject
                if pb_new > pb_cur + 1 and cur_fit[0] <= best_fit[0] + 2:
                    continue

            # Full decode
            new_bins, new_bw = decode_best_fit_lookahead(new_order)
            new_fit = fitness_from(new_bins, new_bw)
            new_sig = signature_from_bw(new_bw)

            # Acceptance
            if accept_move(cur_fit, new_fit, elapsed=(time.time() - start), limit=effective_limit):
                scouts[si] = Scout(new_order, new_bins, new_bw, new_fit, new_sig)
                accepted_moves += 1

                # Update global best
                if better(new_fit, best_fit):
                    update_best(new_order, new_bins, new_bw, new_fit, new_sig)

    return {"packing": best_packing, "bin_weights": best_bw}
