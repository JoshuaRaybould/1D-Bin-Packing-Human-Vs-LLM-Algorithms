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

    if C <= 0:
        packing = [[i] for i in range(n)]
        bin_w = [w[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bin_w}

    # -------------------- Lower bounds (safe) --------------------
    total_w = sum(w)
    L0 = (total_w + C - 1) // C

    # Simple large-item bounds
    gt12 = [wi for wi in w if wi > C / 2]
    big_count = len(gt12)
    sum_small = total_w - sum(gt12)
    LB1 = big_count + (sum_small + C - 1) // C

    # Additional safe bound using >C/3 items: in any bin, at most 2 items > C/3.
    gt13_count = sum(1 for wi in w if wi > C / 3)
    LB2 = (gt13_count + 1) // 2

    LB = max(L0, LB1, LB2)

    # -------------------- Priority <-> order helpers --------------------
    idxs = list(range(n))

    def priority_from_order(order: List[int]) -> List[float]:
        pr = [0.0] * n
        denom = max(1, n)
        for rank, idx in enumerate(order):
            pr[idx] = (denom - rank) / (denom + 1.0)
        return pr

    # Build order with weight classes; within class sort by (priority, weight)
    def build_order(priority: List[float]) -> List[int]:
        t34 = (3 * C) / 4.0
        t23 = (2 * C) / 3.0
        t12 = C / 2.0
        t13 = C / 3.0
        t14 = C / 4.0

        A, B, Cc, D, E = [], [], [], [], []
        for i, wi in enumerate(w):
            if wi > t34:
                A.append(i)
            elif wi > t23:
                B.append(i)
            elif wi > t12:
                Cc.append(i)
            elif wi > t13:
                D.append(i)
            else:
                E.append(i)

        def sort_class(lst: List[int]) -> None:
            lst.sort(key=lambda i: (priority[i], w[i]), reverse=True)

        sort_class(A)
        sort_class(B)
        sort_class(Cc)
        sort_class(D)
        # for very small items, sometimes better to place heavier small first
        E.sort(key=lambda i: (priority[i], w[i]), reverse=True)

        return A + B + Cc + D + E

    # -------------------- Decoder --------------------
    # Cache decoded solutions for repeated orders (small/medium n)
    cache_enabled = n <= 260
    cache_max = 700
    # key -> (bin_weights, assign[item]=bin_index)
    decode_cache: Dict[Tuple[int, ...], Tuple[List[int], List[int]]] = {}
    cache_queue: List[Tuple[int, ...]] = []

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

    def pack_from_order(order: List[int], check_time: bool = True) -> Tuple[List[int], List[int]]:
        """Return (bin_weights, assign[item]=bin_index)."""
        bin_w: List[int] = []
        assign = [-1] * n

        residuals: List[int] = []  # sorted
        bins_for_residual: Dict[int, List[int]] = {}

        def add_bin(bi: int, res: int) -> None:
            insort(residuals, res)
            bins_for_residual.setdefault(res, []).append(bi)

        def pop_bin(res: int) -> int:
            lst = bins_for_residual[res]
            bi = lst.pop()
            if not lst:
                del bins_for_residual[res]
            return bi

        # Two-phase: heavy first with pure best-fit, then lighter with a tiny look-ahead scoring.
        split = int(0.55 * n)
        # Put all >C/3 in phase1 regardless
        phase1 = []
        phase2 = []
        for k, i in enumerate(order):
            if w[i] > C / 3 or k < split:
                phase1.append(i)
            else:
                phase2.append(i)

        # Phase 1: best-fit
        for k, i in enumerate(phase1):
            if check_time and (k & 127) == 0 and time_up():
                break
            wi = w[i]
            pos = bisect_left(residuals, wi)
            if pos == len(residuals):
                bi = len(bin_w)
                bin_w.append(wi)
                assign[i] = bi
                add_bin(bi, C - wi)
            else:
                res = residuals.pop(pos)
                bi = pop_bin(res)
                bin_w[bi] += wi
                assign[i] = bi
                add_bin(bi, C - bin_w[bi])

        # Phase 2: look-ahead among a few candidate bins to avoid creating nasty gaps
        # Candidates: best-fit bin, second-best-fit, and a "largest residual" bin if it fits.
        for t, i in enumerate(phase2):
            if check_time and (t & 127) == 0 and time_up():
                break
            wi = w[i]
            if not residuals:
                bi = len(bin_w)
                bin_w.append(wi)
                assign[i] = bi
                add_bin(bi, C - wi)
                continue

            pos = bisect_left(residuals, wi)
            if pos == len(residuals):
                bi = len(bin_w)
                bin_w.append(wi)
                assign[i] = bi
                add_bin(bi, C - wi)
                continue

            cand_res = []
            cand_res.append(residuals[pos])
            if pos + 1 < len(residuals):
                cand_res.append(residuals[pos + 1])
            # try a very slack bin too
            cand_res.append(residuals[-1])

            best_choice = None
            best_score = None
            # score: prefer tight fill, but penalize leaving residual in (0, C/6)
            for res in cand_res:
                if res < wi:
                    continue
                new_res = res - wi
                score = new_res
                if 0 < new_res < C / 6:
                    score += C / 5  # discourage tiny gaps
                if best_score is None or score < best_score:
                    best_score = score
                    best_choice = res

            if best_choice is None:
                bi = len(bin_w)
                bin_w.append(wi)
                assign[i] = bi
                add_bin(bi, C - wi)
            else:
                # remove one occurrence of best_choice at its leftmost position
                p2 = bisect_left(residuals, best_choice)
                residuals.pop(p2)
                bi = pop_bin(best_choice)
                bin_w[bi] += wi
                assign[i] = bi
                add_bin(bi, C - bin_w[bi])

        return bin_w, assign

    def reconstruct_packing(assign: List[int], bin_w: List[int]) -> List[List[int]]:
        packing = [[] for _ in range(len(bin_w))]
        for i, b in enumerate(assign):
            if b >= 0:
                packing[b].append(i)
        # Remove any empty bins (shouldn't happen, but safe)
        new_packing = []
        new_bw = []
        for b, items in enumerate(packing):
            if items:
                new_packing.append(items)
                new_bw.append(bin_w[b])
        bin_w[:] = new_bw
        return new_packing

    def decode(priority: List[float]) -> Tuple[List[int], List[int]]:
        order = build_order(priority)
        key = tuple(order) if cache_enabled else None
        if cache_enabled:
            hit = _cache_get(key)  # type: ignore[arg-type]
            if hit is not None:
                bw, assign = hit
                return bw[:], assign[:]

        bw, assign = pack_from_order(order, check_time=True)
        if cache_enabled:
            _cache_put(key, (bw[:], assign[:]))
        return bw, assign

    # -------------------- Fitness (bins primary) --------------------
    def fitness(bin_w: List[int]) -> float:
        bins = len(bin_w)
        if bins == 0:
            return 0.0
        res = [C - x for x in bin_w]
        waste = sum(res)
        # Stronger tie-breaking on bad gaps
        k = 7 if bins >= 7 else bins
        topk = sum(sorted(res, reverse=True)[:k]) if k > 0 else 0
        tiny = sum(1 for r in res if 0 < r < C / 10)
        big = sum(1 for r in res if r > C / 3)
        return (
            bins
            + (topk / (C * max(1, k))) * 2e-2
            + (waste / (C * bins)) * 2e-3
            + tiny * 5e-4
            + big * 1e-3
        )

    # -------------------- Initialization (strong seeds) --------------------
    def order_ffd() -> List[int]:
        return sorted(idxs, key=lambda i: w[i], reverse=True)

    def order_bfd_jitter() -> List[int]:
        return sorted(idxs, key=lambda i: (w[i], random.random()), reverse=True)

    def order_complement() -> List[int]:
        half = C / 2.0
        return sorted(idxs, key=lambda i: (abs(w[i] - half), w[i], random.random()), reverse=True)

    def order_mod_class() -> List[int]:
        # push awkward near 1/3 and 2/3 early
        t13 = C / 3.0
        t23 = 2 * C / 3.0
        return sorted(idxs, key=lambda i: (min(abs(w[i] - t13), abs(w[i] - t23)), w[i]), reverse=True)

    def order_random() -> List[int]:
        o = idxs[:]
        random.shuffle(o)
        return o

    # Larger budgets (quality); still fixed-iteration with time checks
    if n <= 60:
        pop_size = 42
        max_iters = 18000
    elif n <= 200:
        pop_size = 50
        max_iters = 24000
    else:
        pop_size = 58
        max_iters = 30000

    pop: List[List[float]] = []
    seed_orders = [
        order_ffd(),
        order_bfd_jitter(),
        order_complement(),
        order_mod_class(),
        order_random(),
        order_random(),
    ]

    for o in seed_orders:
        if len(pop) >= pop_size:
            break
        pop.append(priority_from_order(o))

    invC = 1.0 / C
    while len(pop) < pop_size:
        beta = 0.45 + 0.40 * random.random()  # [0.45,0.85]
        pr = [0.0] * n
        for i in range(n):
            # bias with weight and a little nonlinearity
            base = (w[i] * invC) ** (0.8 + 0.6 * random.random())
            pr[i] = beta * random.random() + (1.0 - beta) * base
            if pr[i] >= 1.0:
                pr[i] = 0.999999
        pop.append(pr)

    # -------------------- Evaluate initial population --------------------
    fits = [float("inf")] * pop_size
    best_priority: Optional[List[float]] = None
    best_bw: Optional[List[int]] = None
    best_assign: Optional[List[int]] = None
    best_fit = float("inf")

    for p in range(pop_size):
        if time_up():
            break
        bw, assign = decode(pop[p])
        f = fitness(bw)
        fits[p] = f
        if f < best_fit:
            best_fit = f
            best_priority = pop[p][:]
            best_bw = bw[:]
            best_assign = assign[:]
            if len(best_bw) == LB:
                pack = reconstruct_packing(best_assign, best_bw)
                return {"packing": pack, "bin_weights": best_bw}

    if best_priority is None or best_bw is None or best_assign is None:
        return {"packing": [], "bin_weights": []}

    # -------------------- FDO main loop (primary method) --------------------
    def clip01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x >= 1.0:
            return 0.999999
        return x

    best_bins = len(best_bw)
    stall = 0

    elite_E = max(4, pop_size // 12)
    eval_counter = 0
    eval_check_period = 8

    for it in range(max_iters):
        if time_up():
            break

        order_pop = sorted(range(pop_size), key=lambda i: fits[i])
        best_idx = order_pop[0]

        cur_best_bins = int(fits[best_idx])
        if cur_best_bins < best_bins:
            best_bins = cur_best_bins
            stall = 0
        else:
            stall += 1

        # Foods: best, random from top quantile, and a diversifying food (bottom quantile)
        food1 = pop[best_idx]
        topQ = max(3, pop_size // 6)
        food2 = pop[order_pop[random.randrange(topQ)]]
        botQ = max(3, pop_size // 8)
        food3 = pop[order_pop[-random.randrange(1, botQ + 1)]]

        fmin = fits[order_pop[0]]
        fmax = fits[order_pop[-1]]
        denom = (fmax - fmin) if (fmax > fmin) else 1.0

        t = it / max(1, max_iters - 1)
        alpha = max(0.03, 0.95 * (1.0 - 0.55 * t))

        # Stall-driven exploration
        if stall > 220:
            noise_boost = min(3.0, 1.0 + (stall - 220) / 180.0)
            reinit_rate = min(0.28, 0.10 + (stall - 220) / 900.0)
            k_frac = min(0.85, 0.45 + (stall - 220) / 700.0)
        else:
            noise_boost = 1.0
            reinit_rate = 0.08
            k_frac = 0.45

        # Periodic partial reinit of worst
        if (it + 1) % 180 == 0 and not time_up():
            R = max(1, int(reinit_rate * pop_size))
            for idx in order_pop[-R:]:
                if time_up():
                    break
                pr = pop[idx]
                beta = 0.50 + 0.35 * random.random()
                powp = 0.7 + 0.9 * random.random()
                for d in range(n):
                    base = (w[d] * invC) ** powp
                    pr[d] = beta * random.random() + (1.0 - beta) * base
                    if pr[d] >= 1.0:
                        pr[d] = 0.999999
                bw, assign = decode(pr)
                f = fitness(bw)
                fits[idx] = f
                if f < best_fit:
                    best_fit = f
                    best_priority = pr[:]
                    best_bw = bw[:]
                    best_assign = assign[:]
                    if len(best_bw) == LB:
                        pack = reconstruct_packing(best_assign, best_bw)
                        return {"packing": pack, "bin_weights": best_bw}

        elites = set(order_pop[:elite_E])

        for i in range(pop_size):
            if i in elites:
                continue

            eval_counter += 1
            if (eval_counter % eval_check_period) == 0 and time_up():
                break

            xi = pop[i]
            fi = fits[i]
            fw = (fi - fmin) / denom  # 0..1 (worse -> larger)

            # Choose a food (leader) depending on quality
            rsel = random.random()
            if fw < 0.35:
                leader = food1 if rsel < 0.7 else food2
            elif fw < 0.75:
                leader = food2 if rsel < 0.6 else food1
            else:
                leader = food3 if rsel < 0.55 else food1

            # Subset of dimensions for speed and better mixing
            if n <= 120:
                dims = range(n)
            else:
                k = int(max(24, min(n, k_frac * n * (0.55 + 0.9 * fw))))
                dims = random.sample(range(n), k)

            # FDO-inspired update: move proportional to (leader-x), scaled by fitness dependency
            # Better individuals (small fw) do smaller, exploitative moves.
            # Worse individuals explore more.
            step_scale = alpha * (0.10 + 1.25 * fw) * (0.9 + 0.2 * random.random())
            noise_scale = (0.010 + 0.10 * fw) * noise_boost

            newx = xi[:]
            for d in dims:
                diff = leader[d] - xi[d]
                # signed random factor, like FDO pace
                pace = (random.random() * 2.0 - 1.0) * step_scale
                newx[d] = clip01(newx[d] + pace * diff + (random.random() - random.random()) * noise_scale)

            # Small mutation on a few dims
            m = max(2, int((0.02 + 0.08 * fw) * n))
            m = min(n, m)
            for d in random.sample(range(n), m):
                newx[d] = clip01(newx[d] + (random.random() - random.random()) * (0.03 + 0.18 * fw) * noise_boost)

            # Rare swap-like perturbation
            if random.random() < (0.04 + 0.10 * fw):
                a = random.randrange(n)
                b = random.randrange(n)
                newx[a], newx[b] = newx[b], newx[a]

            bw, assign = decode(newx)
            fnew = fitness(bw)

            # Accept rule: greedy plus tiny probability of equal-worse when stalling (still FDO-style)
            if fnew <= fi or (stall > 300 and random.random() < 0.02 and fnew < fi + 0.08):
                pop[i] = newx
                fits[i] = fnew
                if fnew < best_fit:
                    best_fit = fnew
                    best_priority = newx[:]
                    best_bw = bw[:]
                    best_assign = assign[:]
                    if len(best_bw) == LB:
                        pack = reconstruct_packing(best_assign, best_bw)
                        return {"packing": pack, "bin_weights": best_bw}

        # Micro-jitter to non-best elites to avoid clone stagnation
        if (it + 1) % 90 == 0 and not time_up() and elite_E > 1:
            for eidx in order_pop[1:elite_E]:
                pr = pop[eidx]
                if random.random() < 0.6:
                    for d in random.sample(range(n), min(n, 6)):
                        pr[d] = clip01(pr[d] + (random.random() - random.random()) * 0.02)
                    bw, assign = decode(pr)
                    f = fitness(bw)
                    fits[eidx] = f
                    if f < best_fit:
                        best_fit = f
                        best_priority = pr[:]
                        best_bw = bw[:]
                        best_assign = assign[:]
                        if len(best_bw) == LB:
                            pack = reconstruct_packing(best_assign, best_bw)
                            return {"packing": pack, "bin_weights": best_bw}

    # Reconstruct packing from best assignment
    packing = reconstruct_packing(best_assign, best_bw)
    return {"packing": packing, "bin_weights": best_bw}
