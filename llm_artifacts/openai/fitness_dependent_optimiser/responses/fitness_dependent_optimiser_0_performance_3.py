import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = bin_capacity
    total_w = sum(weights)
    LB = (total_w + C - 1) // C

    if max(weights) > C:
        packing = [[i] for i in range(n)]
        bw = [weights[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bw}

    items = list(range(n))

    # Heavies / ranks
    items_desc = sorted(items, key=lambda i: weights[i], reverse=True)
    rank_pos = [0] * n
    for r, it in enumerate(items_desc):
        rank_pos[it] = r

    # ---------------- Decoder (strong, fast) ----------------
    # Bucketed best-fit with remainder-hardness and slight randomization among best candidates.
    def decode(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bw: List[int] = []

        # Remaining-weight frequency for hardness
        freq = [0] * (C + 1)
        for idx in order:
            freq[weights[idx]] += 1

        buckets: List[List[int]] = [[] for _ in range(C + 1)]  # by remaining
        rem_of: List[int] = []

        def hardness(r: int) -> float:
            if r <= 0:
                return 0.0
            # penalize remainders that seem hard to fill
            h = 1.0 / (1.0 + freq[r])
            if r - 1 >= 1:
                h += 0.35 / (1.0 + freq[r - 1])
            if r + 1 <= C:
                h += 0.35 / (1.0 + freq[r + 1])
            return h

        for t, idx in enumerate(order):
            w = weights[idx]
            freq[w] -= 1

            best_b = -1
            best_rem2 = C + 1
            best_h = 1e18

            # Scan possible remainders from w..C (bucketed)
            # Slightly randomized tie-break: choose oldest or 2nd-oldest sometimes.
            for rem in range(w, C + 1):
                lst = buckets[rem]
                if not lst:
                    continue
                rem2 = rem - w
                h = hardness(rem2)

                # choose candidate bin index from this bucket
                if len(lst) >= 2 and random.random() < 0.08:
                    b = lst[1]
                else:
                    b = lst[0]

                if (rem2 < best_rem2) or (rem2 == best_rem2 and (h < best_h)):
                    best_b = b
                    best_rem2 = rem2
                    best_h = h
                    if best_rem2 == 0:
                        break

            if best_b == -1:
                b = len(bins)
                bins.append([idx])
                bw.append(w)
                rem = C - w
                rem_of.append(rem)
                buckets[rem].append(b)
            else:
                # remove best_b from its old bucket
                old_rem = rem_of[best_b]
                lst = buckets[old_rem]
                # linear scan; buckets tend to be small
                for k in range(len(lst)):
                    if lst[k] == best_b:
                        lst.pop(k)
                        break
                # place item
                bins[best_b].append(idx)
                bw[best_b] += w
                new_rem = old_rem - w
                rem_of[best_b] = new_rem
                buckets[new_rem].append(best_b)

        return bins, bw

    def fitness(bw: List[int]) -> Tuple[int, int]:
        # primary: bins; secondary: total waste
        waste = 0
        for s in bw:
            waste += (C - s)
        return (len(bw), waste)

    def better(f1: Tuple[int, int], f2: Tuple[int, int]) -> bool:
        return f1[0] < f2[0] or (f1[0] == f2[0] and f1[1] < f2[1])

    # ---------------- Seed orders (diverse constructives) ----------------
    items_asc = sorted(items, key=lambda i: weights[i])

    def order_half_closeness() -> List[int]:
        half = C / 2.0
        return sorted(items, key=lambda i: (abs(weights[i] - half), -weights[i]))

    def order_mod(k: int) -> List[int]:
        return sorted(items, key=lambda i: ((weights[i] % k), weights[i]), reverse=True)

    def order_chunked(shuffle_within: bool, chunk: int) -> List[int]:
        # sort by weight desc, but shuffle inside equal-weight blocks or inside chunks
        a = items_desc[:]
        if shuffle_within:
            i = 0
            while i < n:
                j = i + 1
                wi = weights[a[i]]
                while j < n and weights[a[j]] == wi:
                    j += 1
                if j - i >= 2:
                    block = a[i:j]
                    random.shuffle(block)
                    a[i:j] = block
                i = j
        if chunk > 1:
            out = []
            for i in range(0, n, chunk):
                block = a[i:i + chunk]
                if len(block) >= 2:
                    random.shuffle(block)
                out.extend(block)
            a = out
        return a

    def heavy_front_random_tail(cut: int) -> List[int]:
        cut = min(n, cut)
        head = items_desc[:cut]
        tail = items_desc[cut:]
        random.shuffle(tail)
        return head + tail

    seeds: List[List[int]] = [
        items_desc,
        items_asc,
        order_half_closeness(),
        order_chunked(True, 0),
        order_chunked(False, 11 if n > 50 else 7),
        heavy_front_random_tail(30),
        heavy_front_random_tail(60),
    ]
    for k in (3, 4, 5, 7):
        if C >= k:
            seeds.append(order_mod(k))

    # de-dup
    seen = set()
    uniq = []
    for o in seeds:
        t = tuple(o)
        if t not in seen:
            seen.add(t)
            uniq.append(o)
    seeds = uniq

    # ---------------- FDO population ----------------
    # FDO is typically swarm-like; keep moderately large population.
    pop_size = max(50, min(140, 60 + n // 18))

    class Scout:
        __slots__ = ("order", "bins", "bw", "fit")

        def __init__(self, order: List[int], bins: List[List[int]], bw: List[int], fit: Tuple[int, int]):
            self.order = order
            self.bins = bins
            self.bw = bw
            self.fit = fit

    def eval_order(o: List[int]) -> Scout:
        b, bw = decode(o)
        f = fitness(bw)
        return Scout(o, b, bw, f)

    # initial best and 2nd best (elite-2 helps avoid stagnation)
    best = eval_order(items_desc[:])
    best2 = eval_order(order_chunked(True, 13 if n > 120 else 9))
    if better(best2.fit, best.fit):
        best, best2 = best2, best

    scouts: List[Scout] = []
    # add seeds
    for o in seeds:
        sc = eval_order(o[:])
        scouts.append(sc)
        if better(sc.fit, best.fit):
            best2 = best
            best = sc
        elif sc.fit != best.fit and better(sc.fit, best2.fit):
            best2 = sc

    # fill rest with perturbed variants
    def mutate_simple(order: List[int], swaps: int) -> List[int]:
        a = order[:]
        for _ in range(swaps):
            i = random.randrange(n)
            j = random.randrange(n)
            a[i], a[j] = a[j], a[i]
        if n >= 6 and random.random() < 0.35:
            i = random.randrange(n)
            j = random.randrange(n)
            if i > j:
                i, j = j, i
            if j - i >= 3:
                a[i:j] = reversed(a[i:j])
        return a

    while len(scouts) < pop_size:
        r = len(scouts) % 5
        if r == 0:
            o = mutate_simple(best.order, 3)
        elif r == 1:
            o = mutate_simple(best2.order, 3)
        elif r == 2:
            o = heavy_front_random_tail(45)
        elif r == 3:
            o = items[:]
            random.shuffle(o)
        else:
            base = seeds[random.randrange(len(seeds))]
            o = mutate_simple(base, 4)
        sc = eval_order(o)
        scouts.append(sc)
        if better(sc.fit, best.fit):
            best2 = best
            best = sc
        elif sc.fit != best.fit and better(sc.fit, best2.fit):
            best2 = sc

    # positions of elites for guided movement
    best_pos = [0] * n
    best2_pos = [0] * n

    def refresh_elite_positions():
        for i, it in enumerate(best.order):
            best_pos[it] = i
        for i, it in enumerate(best2.order):
            best2_pos[it] = i

    refresh_elite_positions()

    # ---------------- FDO movement operators ----------------
    # Implement FDO-style "pace" / step depending on relative fitness.
    # We translate step into number of guided relocations toward elites.
    def pace(cur: Tuple[int, int], elite: Tuple[int, int]) -> float:
        # Larger => more aggressive
        cb, cw = cur
        eb, ew = elite
        if cb > eb:
            d = cb - eb
            return min(1.0, 0.65 + 0.12 * d)
        if cb == eb and cw > ew:
            # only worse on waste
            # normalize by capacity
            return min(0.75, 0.18 + 0.45 * min(1.0, (cw - ew) / max(1, cb * C)))
        # already elite-quality-ish
        return 0.10

    def fdo_step(order: List[int], cur_fit: Tuple[int, int], target_pos: List[int], step_strength: float) -> List[int]:
        # Move a few sampled items toward their target positions.
        a = order[:]
        m = 2 + int(step_strength * min(60, n))

        # bias sample toward heavy items early in permutation
        for _ in range(m):
            if random.random() < 0.72:
                i = random.randrange(min(n, 80))
            else:
                i = random.randrange(n)
            it = a[i]
            a.pop(i)

            tp = target_pos[it]
            # scale target position to current length
            ins = int((tp / max(1, n - 1)) * max(1, len(a)))

            # noise inversely proportional to strength
            noise = int((1.0 - step_strength) * 12)
            if noise:
                ins += random.randint(-noise, noise)
            if ins < 0:
                ins = 0
            if ins > len(a):
                ins = len(a)
            a.insert(ins, it)

        # small random mutation to avoid lock-in
        if random.random() < 0.55:
            a = mutate_simple(a, 1 if step_strength < 0.4 else 2)
        return a

    def crossover_like(cur: List[int], a_pos: List[int], b_pos: List[int], alpha: float) -> List[int]:
        # FDO variant: mix pull from best and best2 by picking items closest to front in a convex combination.
        # Score item by alpha*posA + (1-alpha)*posB, then stably sort by score with small noise.
        scores = []
        for it in cur:
            s = alpha * a_pos[it] + (1.0 - alpha) * b_pos[it]
            # tiny noise
            s += (random.random() - 0.5) * 0.05
            scores.append((s, -weights[it], it))
        scores.sort()
        return [it for _, _, it in scores]

    # ---------------- Main loop ----------------
    effective_limit = max(0.0, float(time_limit))
    # required periodic time check; also fixed iteration count
    iter_budget = 60000 if n < 400 else 45000 if n < 1200 else 32000
    check_every = 40

    last_improve = start

    for itn in range(iter_budget):
        if itn % check_every == 0:
            if time.time() - start >= effective_limit:
                break

        # refresh elite positions occasionally
        if itn % 120 == 0:
            refresh_elite_positions()

        # stagnation handling inside FDO: re-scout a fraction
        if (itn % 500 == 0) and (time.time() - last_improve > max(1.2, 0.20 * effective_limit)):
            k = max(3, pop_size // 8)
            # replace worst k
            idxs = list(range(pop_size))
            idxs.sort(key=lambda i: (scouts[i].fit[0], scouts[i].fit[1]), reverse=True)
            for z in range(k):
                if time.time() - start >= effective_limit:
                    break
                i = idxs[z]
                if scouts[i] is best:
                    continue
                base = best.order if random.random() < 0.6 else seeds[random.randrange(len(seeds))]
                o = mutate_simple(base, 5)
                if random.random() < 0.35:
                    o = heavy_front_random_tail(60)
                sc = eval_order(o)
                scouts[i] = sc
                if better(sc.fit, best.fit):
                    best2 = best
                    best = sc
                    last_improve = time.time()
                elif sc.fit != best.fit and better(sc.fit, best2.fit):
                    best2 = sc
            last_improve = time.time()  # prevent repeated immediate triggers

        # visit scouts in random order
        idxs = list(range(pop_size))
        random.shuffle(idxs)

        for si in idxs:
            if time.time() - start >= effective_limit:
                break

            sc = scouts[si]
            cur_fit = sc.fit

            # choose leader: best most of the time, best2 sometimes
            if random.random() < 0.78:
                leader = best
                leader_pos = best_pos
            else:
                leader = best2
                leader_pos = best2_pos

            st = pace(cur_fit, leader.fit)

            # Fitness-dependent movement choices
            r = random.random()
            if r < 0.55:
                new_order = fdo_step(sc.order, cur_fit, leader_pos, st)
            elif r < 0.80:
                # mix both elites (helps escaping local structures)
                alpha = 0.65 if leader is best else 0.35
                new_order = crossover_like(sc.order, best_pos, best2_pos, alpha=alpha)
                if random.random() < 0.60:
                    new_order = mutate_simple(new_order, 2)
            else:
                # exploration
                if random.random() < 0.5:
                    new_order = heavy_front_random_tail(45)
                else:
                    new_order = mutate_simple(sc.order, 4 if st < 0.4 else 3)

            new_bins, new_bw = decode(new_order)
            new_fit = fitness(new_bw)

            # FDO acceptance: greedy on primary; fitness-dependent tolerance on waste
            accept = False
            if better(new_fit, cur_fit):
                accept = True
            elif new_fit[0] == cur_fit[0]:
                # sometimes accept slightly worse waste early
                elapsed = time.time() - start
                prog = 1.0 if effective_limit <= 1e-9 else min(1.0, elapsed / effective_limit)
                # tolerance shrinks over time
                tol = int((0.06 * (1.0 - prog) + 0.01) * C * max(1, new_fit[0]))
                if new_fit[1] <= cur_fit[1] + tol and random.random() < (0.22 * (1.0 - prog)):
                    accept = True

            if accept:
                scouts[si] = Scout(new_order, new_bins, new_bw, new_fit)
                sc = scouts[si]

                if better(sc.fit, best.fit):
                    best2 = best
                    best = sc
                    last_improve = time.time()
                elif sc.fit != best.fit and better(sc.fit, best2.fit):
                    best2 = sc

    return {"packing": best.bins, "bin_weights": best.bw}
