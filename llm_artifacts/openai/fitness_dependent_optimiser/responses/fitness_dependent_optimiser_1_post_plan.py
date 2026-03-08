import time
import random
from bisect import bisect_left, insort
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    def time_up() -> bool:
        return time.time() >= deadline

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    w = list(map(int, weights))

    # Trivial feasibility handling
    if C <= 0:
        # No positive capacity: each item its own bin (or impossible). Return singleton bins.
        packing = [[i] for i in range(n)]
        bin_w = [w[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bin_w}

    # -------------------- 1) Stronger safe LB --------------------
    total_w = sum(w)
    L0 = (total_w + C - 1) // C
    big_count = sum(1 for wi in w if wi > C // 2)
    sum_small = total_w - sum(wi for wi in w if wi > C // 2)
    # Safe pairing bound: each > C/2 needs its own bin; the rest need at least ceil(sum_small/C)
    LB = max(L0, big_count + (sum_small + C - 1) // C)

    # -------------------- Helpers to build priorities from an order --------------------
    def priority_from_order(order: List[int]) -> List[float]:
        # Assign descending ranks mapped into (0,1).
        pr = [0.0] * n
        denom = max(1, n)
        for rank, idx in enumerate(order):
            pr[idx] = (denom - rank) / (denom + 1.0)
        return pr

    # -------------------- 2) Decoder: class-ordered + bisect-based best-fit --------------------
    # Cache decode results for identical order (n<=200)
    cache_enabled = n <= 200
    decode_cache: Dict[Tuple[int, ...], Tuple[List[List[int]], List[int], float]] = {}
    cache_queue: List[Tuple[int, ...]] = []
    cache_max = 400

    def _cache_get(key: Tuple[int, ...]):
        return decode_cache.get(key)

    def _cache_put(key: Tuple[int, ...], val):
        if not cache_enabled:
            return
        if key in decode_cache:
            return
        decode_cache[key] = val
        cache_queue.append(key)
        if len(cache_queue) > cache_max:
            old = cache_queue.pop(0)
            decode_cache.pop(old, None)

    def build_order(priority: List[float]) -> List[int]:
        # Weight class partition heavy->light, within class sort by (priority, weight)
        # Classes:
        # A: >2/3C
        # B: (1/2C,2/3C]
        # C: (1/3C,1/2C]
        # D: <=1/3C
        t23 = (2 * C) / 3.0
        t12 = C / 2.0
        t13 = C / 3.0

        A, B, Cc, D = [], [], [], []
        for i, wi in enumerate(w):
            if wi > t23:
                A.append(i)
            elif wi > t12:
                B.append(i)
            elif wi > t13:
                Cc.append(i)
            else:
                D.append(i)

        def sort_class(lst: List[int]) -> None:
            # Desc by priority, tie-break by heavier
            lst.sort(key=lambda i: (priority[i], w[i]), reverse=True)

        sort_class(A)
        sort_class(B)
        sort_class(Cc)
        sort_class(D)

        return A + B + Cc + D

    def best_fit_pack_from_order(order: List[int], check_time: bool = True) -> Tuple[List[List[int]], List[int]]:
        # Maintain sorted residual list and mapping residual->stack of bin indices.
        # Use bisect_left on residuals to find first residual >= wi (tightest fit).
        packing: List[List[int]] = []
        bin_w: List[int] = []

        residuals: List[int] = []  # sorted residual values
        bins_for_residual: Dict[int, List[int]] = {}

        # Localize
        C_loc = C
        w_loc = w
        residuals_append = residuals.append
        ins = insort
        bleft = bisect_left

        def add_bin(bi: int, residual: int) -> None:
            # insert residual into sorted list and map
            ins(residuals, residual)
            bins_for_residual.setdefault(residual, []).append(bi)

        def pop_bin_for_residual(residual: int) -> int:
            lst = bins_for_residual[residual]
            bi = lst.pop()
            if not lst:
                del bins_for_residual[residual]
            return bi

        def remove_one_residual_at_pos(pos: int, residual: int) -> None:
            # remove residuals[pos]==residual
            residuals.pop(pos)

        # Pack
        for k, i in enumerate(order):
            if check_time and (k & 127) == 0 and time_up():
                break
            wi = w_loc[i]
            pos = bleft(residuals, wi)
            if pos == len(residuals):
                # open new bin
                bi = len(bin_w)
                packing.append([i])
                bw = wi
                bin_w.append(bw)
                add_bin(bi, C_loc - bw)
            else:
                res = residuals[pos]
                remove_one_residual_at_pos(pos, res)
                bi = pop_bin_for_residual(res)

                packing[bi].append(i)
                nbw = bin_w[bi] + wi
                bin_w[bi] = nbw
                add_bin(bi, C_loc - nbw)

        return packing, bin_w

    def repair_empty_one_bin(
        packing: List[List[int]],
        bin_w: List[int],
        priority: List[float],
        remaining_time: float,
    ) -> Tuple[List[List[int]], List[int]]:
        # Try to eliminate 1-2 lightest bins by reinserting their items with best-fit.
        # Kept cheap and deterministic; conditioned on remaining_time.
        if len(bin_w) <= 1:
            return packing, bin_w

        # Budget: skip repair if near deadline
        if remaining_time < 0.01:
            return packing, bin_w

        # Pick up to 2 candidate bins: smallest total weight (lightest bins)
        candidates = sorted(range(len(bin_w)), key=lambda j: bin_w[j])[: (2 if remaining_time > 0.05 else 1)]

        # Precompute residual structures for current bins
        # We'll attempt to reinsert items from candidate bin into other bins.
        for bj in candidates:
            if time_up():
                break
            if bj >= len(bin_w) or not packing[bj]:
                continue

            items = packing[bj][:]
            # Reinsert heavier first to increase chance
            items.sort(key=lambda i: (w[i], priority[i]), reverse=True)

            # Build residual index excluding bj
            residuals: List[int] = []
            bins_for_residual: Dict[int, List[int]] = {}

            for j in range(len(bin_w)):
                if j == bj:
                    continue
                res = C - bin_w[j]
                if res < 0:
                    res = 0
                insort(residuals, res)
                bins_for_residual.setdefault(res, []).append(j)

            def take_bin(res: int) -> int:
                lst = bins_for_residual[res]
                x = lst.pop()
                if not lst:
                    del bins_for_residual[res]
                return x

            ok = True
            moves: List[Tuple[int, int]] = []  # (item, dest_bin)

            for t, i in enumerate(items):
                if (t & 31) == 0 and time_up():
                    ok = False
                    break
                wi = w[i]
                pos = bisect_left(residuals, wi)
                if pos == len(residuals):
                    ok = False
                    break
                res = residuals.pop(pos)
                dest = take_bin(res)
                moves.append((i, dest))
                # update dest residual
                bin_w[dest] += wi
                packing[dest].append(i)
                new_res = C - bin_w[dest]
                insort(residuals, new_res)
                bins_for_residual.setdefault(new_res, []).append(dest)

            if ok:
                # remove emptied bin bj
                packing.pop(bj)
                bin_w.pop(bj)
                # Note: indices of bins > bj shifted; but packing/bin_w alignment preserved.
                return packing, bin_w
            else:
                # rollback partial changes
                for i, dest in reversed(moves):
                    packing[dest].pop()
                    bin_w[dest] -= w[i]

        return packing, bin_w

    def repair_fill_moves(
        packing: List[List[int]],
        bin_w: List[int],
        priority: List[float],
        remaining_time: float,
    ) -> Tuple[List[List[int]], List[int]]:
        # One cheap pass: try moving one item from slack bins into tight bins to reduce worst residuals.
        if len(bin_w) <= 1:
            return packing, bin_w
        if remaining_time < 0.02:
            return packing, bin_w

        residual = [C - bw for bw in bin_w]
        tight_bins = sorted(range(len(bin_w)), key=lambda j: residual[j])  # smallest residual first
        slack_bins = sorted(range(len(bin_w)), key=lambda j: residual[j], reverse=True)

        # Budget of checks
        budget = min(2 * n, 600)
        checks = 0

        # For faster candidate selection, sample a few items from slack bins
        for sb in slack_bins[: min(8, len(slack_bins))]:
            if time_up():
                break
            if not packing[sb] or residual[sb] <= 0:
                continue

            # Prefer moving smaller items out of slack bins (keep them packable)
            items = packing[sb]
            # sample up to 6 items biased by low weight / low priority
            if len(items) <= 6:
                sample_items = items[:]
            else:
                # take a few smallest by weight, and a few random
                smallest = sorted(items, key=lambda i: w[i])[:3]
                sample_items = smallest + random.sample(items, 3)

            for i in sample_items:
                wi = w[i]
                # try to place into a tight bin where it fits best
                best_tb = -1
                best_new_res = None
                for tb in tight_bins[: min(10, len(tight_bins))]:
                    if tb == sb:
                        continue
                    if bin_w[tb] + wi <= C:
                        new_res = C - (bin_w[tb] + wi)
                        if best_new_res is None or new_res < best_new_res:
                            best_new_res = new_res
                            best_tb = tb
                            if new_res == 0:
                                break
                    checks += 1
                    if checks >= budget or time_up():
                        break
                if checks >= budget or time_up():
                    break

                if best_tb != -1:
                    # execute move if it doesn't make source bin worse in a way that obviously harms
                    # (keep deterministic and cheap): accept if it reduces max residual among tb/sb.
                    old_pair = max(C - bin_w[best_tb], C - bin_w[sb])
                    # move
                    packing[sb].remove(i)
                    bin_w[sb] -= wi
                    packing[best_tb].append(i)
                    bin_w[best_tb] += wi
                    new_pair = max(C - bin_w[best_tb], C - bin_w[sb])
                    if new_pair <= old_pair:
                        return packing, bin_w
                    # rollback
                    packing[best_tb].pop()
                    bin_w[best_tb] -= wi
                    packing[sb].append(i)
                    bin_w[sb] += wi

        return packing, bin_w

    def decode(priority: List[float], do_repair: bool) -> Tuple[List[List[int]], List[int]]:
        order = build_order(priority)
        key = tuple(order) if cache_enabled else None
        if cache_enabled:
            hit = _cache_get(key)  # type: ignore[arg-type]
            if hit is not None:
                pack, bw, _f = hit
                # Return copies to avoid accidental mutation by repair
                return [b[:] for b in pack], bw[:]

        pack, bw = best_fit_pack_from_order(order, check_time=True)

        if do_repair:
            rem = deadline - time.time()
            pack, bw = repair_empty_one_bin(pack, bw, priority, rem)
            rem = deadline - time.time()
            pack, bw = repair_fill_moves(pack, bw, priority, rem)

        if cache_enabled:
            _cache_put(key, (pack, bw, 0.0))  # fitness filled later if desired
        return pack, bw

    # -------------------- 3) Fitness shaping --------------------
    def fitness(bin_w: List[int]) -> float:
        bins = len(bin_w)
        if bins == 0:
            return 0.0
        res = [C - bw for bw in bin_w]
        waste = sum(res)
        k = 5 if bins >= 5 else bins
        if k > 0:
            # sum of largest residuals (worst bins)
            topk = sum(sorted(res, reverse=True)[:k])
            topk_term = (topk / (C * k)) * 1e-2
        else:
            topk_term = 0.0
        waste_term = (waste / (C * bins)) * 1e-3
        empties = sum(1 for r in res if r > C / 3.0)
        empty_term = empties * 1e-3
        return bins + topk_term + waste_term + empty_term

    # -------------------- 5) Better initialization --------------------
    idxs = list(range(n))

    def order_ffd() -> List[int]:
        return sorted(idxs, key=lambda i: w[i], reverse=True)

    def order_bfd_jitter() -> List[int]:
        # weight with tiny jitter
        return sorted(idxs, key=lambda i: (w[i], random.random()), reverse=True)

    def order_complement() -> List[int]:
        half = C / 2.0
        return sorted(idxs, key=lambda i: (abs(w[i] - half), w[i]), reverse=True)

    def order_random() -> List[int]:
        o = idxs[:]
        random.shuffle(o)
        return o

    # Population size and iteration budget (fixed iterations; time limit can stop early)
    if n <= 60:
        pop_size = 28
        max_iters = 9000
    elif n <= 200:
        pop_size = 32
        max_iters = 12000
    else:
        pop_size = 36
        max_iters = 15000

    # Build initial population with structured seeds
    pop: List[List[float]] = []
    seeds = [order_ffd(), order_bfd_jitter(), order_complement()]
    # add a couple randomized seeds
    seeds.append(order_random())
    seeds.append(order_random())

    for o in seeds:
        if len(pop) >= pop_size:
            break
        pop.append(priority_from_order(o))

    # Biased random priorities toward weight
    for _ in range(pop_size - len(pop)):
        beta = 0.5 + 0.3 * random.random()  # [0.5,0.8]
        pr = [0.0] * n
        invC = 1.0 / C
        for i in range(n):
            pr[i] = beta * random.random() + (1.0 - beta) * (w[i] * invC)
            # clip into [0,1)
            if pr[i] >= 1.0:
                pr[i] = 0.999999
        pop.append(pr)

    # -------------------- Evaluate initial population --------------------
    best_priority: Optional[List[float]] = None
    best_pack: Optional[List[List[int]]] = None
    best_bw: Optional[List[int]] = None
    best_fit = float("inf")

    fits = [float("inf")] * pop_size

    # Initial evaluation: allow repair (usually plenty of time at start)
    for p in range(pop_size):
        if time_up():
            break
        do_repair = (deadline - time.time()) > 0.05
        pack, bw = decode(pop[p], do_repair=do_repair)
        f = fitness(bw)
        fits[p] = f
        if f < best_fit:
            best_fit = f
            best_priority = pop[p][:]
            best_pack, best_bw = pack, bw
            if len(best_bw) == LB:
                return {"packing": best_pack, "bin_weights": best_bw}

    if best_pack is None or best_priority is None or best_bw is None:
        return {"packing": [], "bin_weights": []}

    # -------------------- 4) FDO upgrades --------------------
    def clip01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x >= 1.0:
            return 0.999999
        return x

    eval_counter = 0
    eval_check_period = 6

    stall = 0
    best_bins = len(best_bw)

    elite_E = 3 if pop_size >= 30 else 2

    for it in range(max_iters):
        if time_up():
            break

        # Rank population
        order_pop = sorted(range(pop_size), key=lambda i: fits[i])

        # Track stalling based on best bin count improvement
        cur_best_idx = order_pop[0]
        cur_best_bins = int(fits[cur_best_idx])  # floor gives bins since fitness=bins+small
        if cur_best_bins < best_bins:
            best_bins = cur_best_bins
            stall = 0
        else:
            stall += 1

        # Adaptive parameters
        t = it / max(1, max_iters - 1)
        base_alpha = 0.85 * (1.0 - 0.5 * t)
        min_alpha = 0.05
        alpha = max(min_alpha, base_alpha)

        # Stall-based exploration boost
        if stall > 150:
            noise_boost = min(2.5, 1.0 + (stall - 150) / 200.0)
            swap_p = min(0.25, 0.05 + 0.002 * (stall - 150))
            k_frac = min(0.75, 0.35 + 0.002 * (stall - 150))
        else:
            noise_boost = 1.0
            swap_p = 0.05
            k_frac = 0.35

        # Leaders (foods)
        best_idx = order_pop[0]
        food1 = pop[best_idx]
        topK = max(2, pop_size // 5)
        food2 = pop[order_pop[random.randrange(topK)]]

        fmin = fits[order_pop[0]]
        fmax = fits[order_pop[-1]]
        denom = (fmax - fmin) if (fmax > fmin) else 1.0

        # Periodic reinit of worst individuals (explorers)
        if (it + 1) % 250 == 0 and not time_up():
            R = max(1, pop_size // 6)  # ~15-20%
            for idx in order_pop[-R:]:
                if time_up():
                    break
                beta = 0.55 + 0.25 * random.random()
                invC = 1.0 / C
                pr = pop[idx]
                for d in range(n):
                    pr[d] = beta * random.random() + (1.0 - beta) * (w[d] * invC)
                    if pr[d] >= 1.0:
                        pr[d] = 0.999999
                pack, bw = decode(pr, do_repair=(deadline - time.time()) > 0.07)
                f = fitness(bw)
                fits[idx] = f
                if f < best_fit:
                    best_fit = f
                    best_priority = pr[:]
                    best_pack, best_bw = pack, bw
                    if len(best_bw) == LB:
                        return {"packing": best_pack, "bin_weights": best_bw}

        # Elitism: keep top E unchanged
        elites = set(order_pop[:elite_E])

        for i in range(pop_size):
            if i in elites:
                continue

            eval_counter += 1
            if (eval_counter % eval_check_period) == 0 and time_up():
                break

            xi = pop[i]
            fi = fits[i]

            fw = (fi - fmin) / denom  # 0..1
            # choose leader
            leader = food1 if (fw > 0.55 or random.random() < 0.6) else food2

            r = random.random() * 2.0 - 1.0
            noise_scale = (0.015 + 0.09 * fw) * noise_boost

            # number of perturbed dimensions
            if n <= 120:
                dims = range(n)
            else:
                k = max(18, int(k_frac * n * (0.6 + 0.8 * fw)))
                k = min(n, k)
                dims = random.sample(range(n), k)

            newx = xi[:]
            for d in dims:
                step = alpha * r * fw * (leader[d] - xi[d])
                step += (random.random() - random.random()) * noise_scale
                newx[d] = clip01(newx[d] + step)

            # 4.1) rank-jitter + swap mutation
            m = max(2, int((0.02 + 0.10 * fw) * n))
            m = min(n, m)
            if n <= 80:
                mut_dims = random.sample(range(n), m)
            else:
                mut_dims = random.sample(range(n), m)
            sigma = (0.05 + 0.20 * fw) * noise_boost
            for d in mut_dims:
                newx[d] = clip01(newx[d] + (random.random() - random.random()) * sigma)

            if random.random() < swap_p:
                a = random.randrange(n)
                b = random.randrange(n)
                newx[a], newx[b] = newx[b], newx[a]

            # Occasional heavy reset for very poor individuals
            if fw > 0.85 and random.random() < 0.07:
                reset_k = min(n, max(10, int(0.25 * n)))
                for d in random.sample(range(n), reset_k):
                    newx[d] = random.random()

            # Decode with repair depending on remaining time
            do_repair = (deadline - time.time()) > 0.04
            pack, bw = decode(newx, do_repair=do_repair)
            fnew = fitness(bw)

            # Accept if improved
            if fnew <= fi:
                pop[i] = newx
                fits[i] = fnew
                if fnew < best_fit:
                    best_fit = fnew
                    best_priority = newx[:]
                    best_pack, best_bw = pack, bw
                    if len(best_bw) == LB:
                        return {"packing": best_pack, "bin_weights": best_bw}

    return {
        "packing": best_pack if best_pack is not None else [],
        "bin_weights": best_bw if best_bw is not None else [],
    }
