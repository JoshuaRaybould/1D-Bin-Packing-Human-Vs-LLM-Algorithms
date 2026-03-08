import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)

    if max(weights) > C:
        packing = [[i] for i in range(n)]
        bw = [weights[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bw}

    total_w = sum(weights)
    LB = (total_w + C - 1) // C

    items = list(range(n))
    items_desc = sorted(items, key=lambda i: weights[i], reverse=True)
    items_asc = sorted(items, key=lambda i: weights[i])

    # ---------------- Fenwick tree for fast "best remainder >= w" queries ----------------
    class Fenwick:
        __slots__ = ("n", "bit")

        def __init__(self, n: int):
            self.n = n
            self.bit = [0] * (n + 1)

        def add(self, i: int, delta: int) -> None:
            # i in [0..n-1]
            i += 1
            bit = self.bit
            nn = self.n
            while i <= nn:
                bit[i] += delta
                i += i & -i

        def sum(self, i: int) -> int:
            # prefix sum [0..i]
            i += 1
            s = 0
            bit = self.bit
            while i > 0:
                s += bit[i]
                i -= i & -i
            return s

        def range_sum(self, l: int, r: int) -> int:
            if r < l:
                return 0
            return self.sum(r) - (self.sum(l - 1) if l > 0 else 0)

        def find_by_prefix(self, target: int) -> int:
            # smallest idx such that prefix sum >= target, assuming target in [1..total]
            idx = 0
            bit = self.bit
            bitmask = 1 << (self.n.bit_length() - 1)
            while bitmask:
                t = idx + bitmask
                if t <= self.n and bit[t] < target:
                    idx = t
                    target -= bit[t]
                bitmask >>= 1
            return idx  # 0-based index

        def first_ge(self, l: int) -> int:
            # first index >= l with positive count, else -1
            if l >= self.n:
                return -1
            before = self.sum(l - 1) if l > 0 else 0
            total = self.sum(self.n - 1)
            if total == before:
                return -1
            # find first prefix >= before+1
            return self.find_by_prefix(before + 1)

    # ---------------- Decoder ----------------
    # Best-fit based on remainder, with small RCL randomization.
    def decode(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bw: List[int] = []

        # freq used for remainder-hardness (approx)
        freq = [0] * (C + 1)
        for idx in order:
            freq[weights[idx]] += 1

        # bins grouped by remainder (remaining capacity)
        # Use lists as stacks for speed; keep also a count Fenwick.
        by_rem: List[List[int]] = [[] for _ in range(C + 1)]
        rem_of: List[int] = []
        fw = Fenwick(C + 1)

        def hardness(r: int) -> float:
            if r <= 0:
                return 0.0
            # small heuristic: remainders with low frequency nearby are "hard"
            h = 1.0 / (1.0 + freq[r])
            if r - 1 >= 1:
                h += 0.30 / (1.0 + freq[r - 1])
            if r + 1 <= C:
                h += 0.30 / (1.0 + freq[r + 1])
            return h

        for idx in order:
            w = weights[idx]
            freq[w] -= 1

            # Find best remainder >= w.
            # Candidate selection: mostly best remainder, sometimes pick among next few.
            r0 = fw.first_ge(w)
            if r0 == -1:
                # open new bin
                b = len(bins)
                bins.append([idx])
                bw.append(w)
                rnew = C - w
                rem_of.append(rnew)
                by_rem[rnew].append(b)
                fw.add(rnew, 1)
                continue

            # RCL: consider up to 3 remainder levels (best-fit + a couple next levels)
            # but only if they exist. Prefer smaller remainder after placement.
            candidates = [r0]
            if random.random() < 0.20:
                # try to add next non-empty remainder levels
                # (scan forward a bit; typically tiny)
                r = r0 + 1
                while r <= C and len(candidates) < 3:
                    if by_rem[r]:
                        candidates.append(r)
                    r += 1

            # choose best by (rem_after, hardness) with small random tie
            best_r = candidates[0]
            best_ra = best_r - w
            best_h = hardness(best_ra)
            for r in candidates[1:]:
                ra = r - w
                h = hardness(ra)
                if ra < best_ra or (ra == best_ra and h < best_h - 1e-12):
                    best_r, best_ra, best_h = r, ra, h

            # Occasionally diversify even if worse by 1 remainder if hardness suggests.
            if len(candidates) >= 2 and random.random() < 0.10:
                r_alt = candidates[-1]
                ra_alt = r_alt - w
                if ra_alt <= best_ra + 1 and hardness(ra_alt) + 0.05 < best_h:
                    best_r, best_ra = r_alt, ra_alt

            # pick a bin from by_rem[best_r]
            lst = by_rem[best_r]
            b = lst.pop()  # LIFO
            if not lst:
                fw.add(best_r, -1)

            # place item
            bins[b].append(idx)
            bw[b] += w
            new_r = best_r - w
            rem_of[b] = new_r
            if new_r >= 0:
                if not by_rem[new_r]:
                    fw.add(new_r, 1)
                by_rem[new_r].append(b)

        return bins, bw

    # ---------------- Fitness ----------------
    # Primary: bins, secondary: total waste. Third: sum of squared remaining (smaller is better)
    # to encourage tight packing structure at equal bins.
    def fitness(bw: List[int]) -> Tuple[int, int, int]:
        waste = 0
        sq = 0
        for s in bw:
            r = C - s
            waste += r
            sq += r * r
        return (len(bw), waste, sq)

    def better(f1: Tuple[int, int, int], f2: Tuple[int, int, int]) -> bool:
        return f1 < f2

    # ---------------- Seed orders ----------------
    def order_half_closeness() -> List[int]:
        half = C / 2.0
        return sorted(items, key=lambda i: (abs(weights[i] - half), -weights[i]))

    def order_mod(k: int) -> List[int]:
        return sorted(items, key=lambda i: ((weights[i] % k), weights[i]), reverse=True)

    def order_chunked(chunk: int) -> List[int]:
        a = items_desc[:]
        if chunk <= 1:
            return a
        out = []
        for i in range(0, n, chunk):
            block = a[i:i + chunk]
            if len(block) >= 2:
                random.shuffle(block)
            out.extend(block)
        return out

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
        order_chunked(7 if n <= 70 else 11),
        order_chunked(17 if n > 200 else 13),
        heavy_front_random_tail(30),
        heavy_front_random_tail(60),
    ]
    for k in (3, 4, 5, 7, 8, 9):
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
    # Slightly larger population now that decode is faster.
    pop_size = max(70, min(220, 90 + n // 14))

    class Scout:
        __slots__ = ("order", "bins", "bw", "fit")

        def __init__(self, order: List[int], bins: List[List[int]], bw: List[int], fit: Tuple[int, int, int]):
            self.order = order
            self.bins = bins
            self.bw = bw
            self.fit = fit

    def eval_order(o: List[int]) -> Scout:
        b, bws = decode(o)
        f = fitness(bws)
        return Scout(o, b, bws, f)

    def mutate_simple(order: List[int], swaps: int) -> List[int]:
        a = order[:]
        for _ in range(swaps):
            i = random.randrange(n)
            j = random.randrange(n)
            a[i], a[j] = a[j], a[i]
        # occasional segment reverse
        if n >= 8 and random.random() < 0.30:
            i = random.randrange(n)
            j = random.randrange(n)
            if i > j:
                i, j = j, i
            if j - i >= 4:
                a[i:j] = reversed(a[i:j])
        return a

    # Evaluate initial elites
    best = eval_order(items_desc[:])
    best2 = eval_order(order_chunked(13 if n > 140 else 9))
    if better(best2.fit, best.fit):
        best, best2 = best2, best

    scouts: List[Scout] = []
    archive: List[Scout] = []  # small elite archive
    ARCH = 8

    def archive_add(sc: Scout) -> None:
        nonlocal archive
        archive.append(sc)
        archive.sort(key=lambda s: s.fit)
        # de-dup by fitness a bit
        out = []
        seenf = set()
        for s in archive:
            if s.fit in seenf:
                continue
            seenf.add(s.fit)
            out.append(s)
            if len(out) >= ARCH:
                break
        archive = out

    # add seeds
    for o in seeds:
        sc = eval_order(o[:])
        scouts.append(sc)
        archive_add(sc)
        if better(sc.fit, best.fit):
            best2 = best
            best = sc
        elif sc.fit != best.fit and better(sc.fit, best2.fit):
            best2 = sc

    # fill rest
    while len(scouts) < pop_size:
        r = len(scouts) % 6
        if r == 0:
            o = mutate_simple(best.order, 3)
        elif r == 1:
            o = mutate_simple(best2.order, 3)
        elif r == 2:
            o = heavy_front_random_tail(50)
        elif r == 3:
            o = items[:]
            random.shuffle(o)
        elif r == 4:
            base = seeds[random.randrange(len(seeds))]
            o = mutate_simple(base, 4)
        else:
            base = archive[random.randrange(len(archive))].order if archive else best.order
            o = mutate_simple(base, 5)
        sc = eval_order(o)
        scouts.append(sc)
        archive_add(sc)
        if better(sc.fit, best.fit):
            best2 = best
            best = sc
        elif sc.fit != best.fit and better(sc.fit, best2.fit):
            best2 = sc

    best_pos = [0] * n
    best2_pos = [0] * n

    def refresh_elite_positions() -> None:
        for i, it in enumerate(best.order):
            best_pos[it] = i
        for i, it in enumerate(best2.order):
            best2_pos[it] = i

    refresh_elite_positions()

    # ---------------- FDO movement ----------------
    def pace(cur: Tuple[int, int, int], elite: Tuple[int, int, int]) -> float:
        cb, cw, _ = cur
        eb, ew, _ = elite
        if cb > eb:
            d = cb - eb
            return min(1.0, 0.70 + 0.10 * d)
        if cb == eb and cw > ew:
            # normalize by total capacity used
            denom = max(1, cb * C)
            return min(0.85, 0.20 + 0.55 * min(1.0, (cw - ew) / denom))
        return 0.10

    def guided_relocate(order: List[int], target_pos: List[int], strength: float) -> List[int]:
        a = order[:]
        m = 3 + int(strength * min(80, n))
        # prefer heavy/early
        front = min(n, 120)
        for _ in range(m):
            if random.random() < 0.75:
                i = random.randrange(front)
            else:
                i = random.randrange(n)
            it = a.pop(i)
            tp = target_pos[it]
            ins = int((tp / max(1, n - 1)) * len(a))
            noise = int((1.0 - strength) * 16)
            if noise:
                ins += random.randint(-noise, noise)
            if ins < 0:
                ins = 0
            if ins > len(a):
                ins = len(a)
            a.insert(ins, it)
        # light mutation
        if random.random() < 0.45:
            a = mutate_simple(a, 1 if strength < 0.5 else 2)
        return a

    def block_move_toward(order: List[int], target_pos: List[int], strength: float) -> List[int]:
        # pick a block and reinsert roughly where target suggests (keeps relative order inside block)
        a = order[:]
        if n < 10:
            return mutate_simple(a, 2)
        L = 2 + int((0.10 + 0.25 * strength) * min(60, n))
        L = max(2, min(L, n // 2))
        i = random.randrange(0, n - L + 1)
        block = a[i:i + L]
        del a[i:i + L]
        # target insertion position = median of target positions in block
        tps = sorted(target_pos[it] for it in block)
        med = tps[len(tps) // 2]
        ins = int((med / max(1, n - 1)) * len(a))
        noise = int((1.0 - strength) * 20)
        if noise:
            ins += random.randint(-noise, noise)
        if ins < 0:
            ins = 0
        if ins > len(a):
            ins = len(a)
        a[ins:ins] = block
        if random.random() < 0.35:
            a = mutate_simple(a, 1)
        return a

    def crossover_like(cur: List[int], a_pos: List[int], b_pos: List[int], alpha: float) -> List[int]:
        scores = []
        for it in cur:
            s = alpha * a_pos[it] + (1.0 - alpha) * b_pos[it]
            s += (random.random() - 0.5) * 0.07
            scores.append((s, -weights[it], it))
        scores.sort()
        return [it for _, _, it in scores]

    def random_destroy_repair(cur: List[int], strength: float) -> List[int]:
        # still permutation-based (FDO exploration): remove some items and reinsert randomly biased to front
        a = cur[:]
        k = 2 + int((0.05 + 0.12 * strength) * n)
        k = min(k, max(6, n // 3))
        removed = []
        for _ in range(k):
            i = random.randrange(len(a))
            removed.append(a.pop(i))
        # reinsert with heavy bias early
        removed.sort(key=lambda it: weights[it], reverse=True)
        for it in removed:
            if random.random() < 0.65:
                pos = random.randrange(0, min(len(a) + 1, 40))
            else:
                pos = random.randrange(len(a) + 1)
            a.insert(pos, it)
        return a

    # ---------------- Main loop ----------------
    # Allowed to run longer; cap at 100 seconds.
    effective_limit = min(100.0, max(0.0, float(time_limit)))
    if effective_limit <= 0.0:
        effective_limit = 1.0

    # fixed iteration budget (time will cut it earlier)
    iter_budget = 180000 if n < 400 else 140000 if n < 1200 else 100000
    check_every = 60

    last_improve = start

    for itn in range(iter_budget):
        if itn % check_every == 0:
            if time.time() - start >= effective_limit:
                break

        if itn % 150 == 0:
            refresh_elite_positions()

        # stagnation: rebuild worst portion, but also inject archive-guided mixes
        if itn % 700 == 0:
            elapsed_since = time.time() - last_improve
            if elapsed_since > max(1.5, 0.22 * effective_limit):
                # replace worst
                k = max(5, pop_size // 6)
                idxs = list(range(pop_size))
                idxs.sort(key=lambda i: scouts[i].fit, reverse=True)
                for z in range(k):
                    if time.time() - start >= effective_limit:
                        break
                    i = idxs[z]
                    if scouts[i] is best:
                        continue
                    if archive and random.random() < 0.60:
                        baseA = archive[random.randrange(len(archive))].order
                        baseB = best.order if random.random() < 0.7 else best2.order
                        # build mix by position-scoring
                        # make temporary pos for baseA
                        posA = [0] * n
                        for p, it in enumerate(baseA):
                            posA[it] = p
                        alpha = 0.55 + 0.35 * random.random()
                        o = crossover_like(baseB, posA, best_pos, alpha)
                        o = mutate_simple(o, 3)
                    else:
                        base = best.order if random.random() < 0.65 else seeds[random.randrange(len(seeds))]
                        o = mutate_simple(base, 6)
                        if random.random() < 0.35:
                            o = heavy_front_random_tail(70)
                    sc = eval_order(o)
                    scouts[i] = sc
                    archive_add(sc)
                    if better(sc.fit, best.fit):
                        best2 = best
                        best = sc
                        last_improve = time.time()
                    elif sc.fit != best.fit and better(sc.fit, best2.fit):
                        best2 = sc
                last_improve = time.time()  # avoid rapid retrigger

        idxs = list(range(pop_size))
        random.shuffle(idxs)

        for si in idxs:
            if time.time() - start >= effective_limit:
                break

            sc = scouts[si]
            cur_fit = sc.fit

            # leader selection: mostly best, sometimes best2 or archive leader
            rlead = random.random()
            if rlead < 0.72:
                leader = best
                leader_pos = best_pos
            elif rlead < 0.90:
                leader = best2
                leader_pos = best2_pos
            else:
                leader = archive[random.randrange(len(archive))] if archive else best
                # build pos for this archive leader
                leader_pos = [0] * n
                for p, it in enumerate(leader.order):
                    leader_pos[it] = p

            st = pace(cur_fit, leader.fit)

            r = random.random()
            if r < 0.45:
                new_order = guided_relocate(sc.order, leader_pos, st)
            elif r < 0.65:
                new_order = block_move_toward(sc.order, leader_pos, st)
            elif r < 0.82:
                alpha = 0.70 if leader is best else 0.45
                new_order = crossover_like(sc.order, best_pos, best2_pos, alpha=alpha)
                if random.random() < 0.65:
                    new_order = mutate_simple(new_order, 2)
            else:
                if random.random() < 0.55:
                    new_order = random_destroy_repair(sc.order, st)
                else:
                    new_order = mutate_simple(sc.order, 5 if st < 0.4 else 4)

            new_bins, new_bw = decode(new_order)
            new_fit = fitness(new_bw)

            # Acceptance: greedy on bins; allow limited uphill on waste early.
            accept = False
            if better(new_fit, cur_fit):
                accept = True
            elif new_fit[0] == cur_fit[0]:
                elapsed = time.time() - start
                prog = min(1.0, elapsed / effective_limit) if effective_limit > 1e-9 else 1.0
                # waste tolerance shrinks over time
                # scale by bins and capacity
                tol = int((0.085 * (1.0 - prog) + 0.012) * C * max(1, new_fit[0]))
                if new_fit[1] <= cur_fit[1] + tol:
                    # probabilistic acceptance decreases with time
                    if random.random() < (0.28 * (1.0 - prog)):
                        accept = True

            if accept:
                sc2 = Scout(new_order, new_bins, new_bw, new_fit)
                scouts[si] = sc2
                archive_add(sc2)

                if better(sc2.fit, best.fit):
                    best2 = best
                    best = sc2
                    last_improve = time.time()
                    # early stop if hit LB
                    if best.fit[0] <= LB:
                        # can't beat LB on bins; keep going a bit for tie improvements? not needed.
                        pass
                elif sc2.fit != best.fit and better(sc2.fit, best2.fit):
                    best2 = sc2

    return {"packing": best.bins, "bin_weights": best.bw}
