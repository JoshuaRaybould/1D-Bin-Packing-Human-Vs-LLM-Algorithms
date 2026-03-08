import random
import time
from bisect import bisect_left, insort
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)

    # ---------------- Time helpers ----------------
    def elapsed() -> float:
        return time.perf_counter() - start

    def time_left() -> float:
        return time_limit - elapsed()

    def time_exceeded() -> bool:
        return elapsed() >= time_limit

    # Keep a small guard so we don’t start heavy work at the end.
    def safe_to_decode() -> bool:
        return time_left() > 0.003

    # ---------------- Lower bounds ----------------
    total_w = sum(weights)
    lb = (total_w + C - 1) // C
    lb2 = sum(1 for w in weights if 2 * w > C)
    lb = max(lb, lb2)

    # ---------------- Base ranks and item classes ----------------
    base_order = list(range(n))
    base_order.sort(key=lambda i: weights[i], reverse=True)
    rank_pos = [0] * n
    for r, i in enumerate(base_order):
        rank_pos[i] = r

    sorted_ws = sorted(weights)

    def percentile_value(p: float) -> int:
        if n == 1:
            return sorted_ws[0]
        idx = int(p * (n - 1))
        return sorted_ws[idx]

    p90 = percentile_value(0.9)
    p50 = percentile_value(0.5)

    heavy_items = [i for i, w in enumerate(weights) if w >= p90]
    mid_items = [i for i, w in enumerate(weights) if p50 <= w < p90]
    small_items = [i for i, w in enumerate(weights) if w < p50]

    # ---------------- Random-keys individuals ----------------
    # Each individual: (order_keys, fit_keys)

    def make_biased_keys(noise_sigma: float) -> Tuple[List[float], List[float]]:
        invn = 1.0 / n
        ok = [0.0] * n
        fk = [0.0] * n
        for i in range(n):
            base = rank_pos[i] * invn
            ok[i] = base + random.uniform(-noise_sigma, noise_sigma)
            fk[i] = random.random()
        return ok, fk

    def make_random_keys() -> Tuple[List[float], List[float]]:
        return [random.random() for _ in range(n)], [random.random() for _ in range(n)]

    def keys_to_order(order_keys: List[float]) -> List[int]:
        # Stable tie-break: heavier first
        return sorted(range(n), key=lambda i: (order_keys[i], -weights[i]))

    def keys_from_order(order: List[int], noise_sigma: float) -> Tuple[List[float], List[float]]:
        invn = 1.0 / n
        ok = [0.0] * n
        fk = [random.random() for _ in range(n)]
        for pos, item in enumerate(order):
            ok[item] = pos * invn + random.uniform(-noise_sigma, noise_sigma)
        return ok, fk

    # ---------------- Strong exact Best-Fit decoder ----------------
    # Data structure: sorted list of (remaining, bin_id).

    def decode(order: List[int], fit_keys: List[float], variant: int) -> Tuple[List[List[int]], List[int]]:
        # bins contents and weights
        bins: List[List[int]] = []
        bws: List[int] = []

        # current remaining per bin (array)
        rems: List[int] = []

        # sorted list of (remaining, bin_id) for bins; kept consistent
        # Note: to update a bin, we remove its old pair by linear search;
        # however, we keep it efficient by storing positions in an array
        # only when needed. For typical bin counts, this is fast enough
        # and produces stronger packings than approximate bucket scans.
        #
        # We implement remove by searching for exact tuple (old_rem, id)
        # via bisect_left then scanning neighbors.
        sorted_bins: List[Tuple[int, int]] = []

        def remove_pair(pair: Tuple[int, int]) -> None:
            # find pair in sorted_bins
            i = bisect_left(sorted_bins, pair)
            # scan forward for exact match (rarely more than a couple)
            while i < len(sorted_bins) and sorted_bins[i][0] == pair[0]:
                if sorted_bins[i] == pair:
                    sorted_bins.pop(i)
                    return
                i += 1
            # if not found (should be rare), scan backward
            i = bisect_left(sorted_bins, pair) - 1
            while i >= 0 and sorted_bins[i][0] == pair[0]:
                if sorted_bins[i] == pair:
                    sorted_bins.pop(i)
                    return
                i -= 1

        def choose_bin(w: int, item_idx: int) -> int:
            # Find first bin with remaining >= w (best-fit by default: take the smallest such remaining).
            pos = bisect_left(sorted_bins, (w, -1))
            if pos >= len(sorted_bins):
                return -1

            if variant == 2:
                # Slightly more first-fit-ish: among first few feasible, pick one influenced by fit_keys.
                # This diversifies without breaking feasibility.
                window = 4
                end = min(len(sorted_bins), pos + window)
                best = -1
                best_score = None
                for j in range(pos, end):
                    rem, bid = sorted_bins[j]
                    # prefer tight fit; add small fit-key perturbation
                    res_after = rem - w
                    score = (-(res_after), fit_keys[item_idx])
                    if best_score is None or score > best_score:
                        best_score = score
                        best = bid
                return best

            if variant == 1:
                # Best-fit but break ties toward more loaded bins (smaller remaining), and then by fit-key.
                # Since sorted by remaining, we are already at tightest feasible; but there may be ties.
                rem0 = sorted_bins[pos][0]
                best_bid = sorted_bins[pos][1]
                best_fk = fit_keys[item_idx]
                j = pos + 1
                while j < len(sorted_bins) and sorted_bins[j][0] == rem0:
                    # pick deterministically using fit-key and bin_id
                    bid = sorted_bins[j][1]
                    fk = (fit_keys[item_idx] * 1315423911.0) % 1.0
                    if (fk, -bid) > (best_fk, -best_bid):
                        best_fk = fk
                        best_bid = bid
                    j += 1
                return best_bid

            if variant == 3:
                # “Almost best-fit”: allow the next bucket if it significantly reduces future slack patterns.
                # We use a small lookahead on remaining values.
                # Choose between pos and pos+1 based on fit_key.
                bid0 = sorted_bins[pos][1]
                if pos + 1 < len(sorted_bins):
                    bid1 = sorted_bins[pos + 1][1]
                    if fit_keys[item_idx] > 0.70:
                        return bid1
                return bid0

            # variant 0 strict best-fit
            return sorted_bins[pos][1]

        for idx in order:
            w = weights[idx]
            b = choose_bin(w, idx)
            if b == -1:
                # open new bin
                bid = len(bins)
                bins.append([idx])
                bws.append(w)
                rem = C - w
                rems.append(rem)
                insort(sorted_bins, (rem, bid))
            else:
                old_rem = rems[b]
                remove_pair((old_rem, b))
                bins[b].append(idx)
                bws[b] += w
                new_rem = old_rem - w
                rems[b] = new_rem
                insort(sorted_bins, (new_rem, b))

        return bins, bws

    # ---------------- Fitness ----------------
    def fitness_from_bws(bws: List[int]) -> float:
        m = len(bws)
        # primary objective: minimize #bins
        # secondary: encourage tight packing
        slack_sum = 0
        slack_sq = 0
        slack_max = 0
        for bw in bws:
            s = C - bw
            slack_sum += s
            slack_sq += s * s
            if s > slack_max:
                slack_max = s

        invC = 1.0 / C
        f = float(m)
        # scales tuned to keep bin count dominant
        f += 2e-4 * (slack_sum * invC)
        f += 2e-7 * (slack_sq * (invC * invC))
        f += 5e-7 * (slack_max * invC)
        return f

    # ---------------- Evaluation + memoization ----------------
    TOPK = min(96, n)
    MEMO_CAP = 6000 if n <= 900 else 3500
    memo: Dict[Tuple[int, ...], Tuple[float, List[List[int]], List[int]]] = {}
    memo_age: Dict[Tuple[int, ...], int] = {}
    age_counter = 0

    def signature_from_order(order: List[int]) -> Tuple[int, ...]:
        return tuple(order[:TOPK])

    def eval_individual(order_keys: List[float], fit_keys: List[float], decodes: int) -> Tuple[float, List[List[int]], List[int], List[int]]:
        nonlocal age_counter
        order = keys_to_order(order_keys)
        sig = signature_from_order(order)
        age_counter += 1

        hit = memo.get(sig)
        if hit is not None:
            memo_age[sig] = age_counter
            f, pack, bws = hit
            return f, pack, bws, order

        # Deterministic base variant from a few fit keys (stable, reduces noise)
        s = 0.0
        step = max(1, n // 24)
        for i in range(0, n, step):
            s += fit_keys[i]
        base_variant = int(s * 1021) & 3

        best_f = float("inf")
        best_pack: List[List[int]] = []
        best_bws: List[int] = []

        for t in range(decodes):
            v = (base_variant + t) & 3
            pack, bws = decode(order, fit_keys, v)
            f = fitness_from_bws(bws)
            if f < best_f:
                best_f, best_pack, best_bws = f, pack, bws

        memo[sig] = (best_f, best_pack, best_bws)
        memo_age[sig] = age_counter

        if len(memo) > MEMO_CAP:
            # evict ~15% oldest
            k = max(1, MEMO_CAP // 7)
            for key, _ in sorted(memo_age.items(), key=lambda kv: kv[1])[:k]:
                memo.pop(key, None)
                memo_age.pop(key, None)

        return best_f, best_pack, best_bws, order

    # ---------------- Genetic operators ----------------
    def blend_crossover(p1_ok: List[float], p1_fk: List[float], p2_ok: List[float], p2_fk: List[float]) -> Tuple[List[float], List[float]]:
        lam = random.random()
        c_ok = [lam * a + (1.0 - lam) * b for a, b in zip(p1_ok, p2_ok)]
        c_fk = [lam * a + (1.0 - lam) * b for a, b in zip(p1_fk, p2_fk)]
        return c_ok, c_fk

    def order_ox_crossover(p1_order: List[int], p2_order: List[int]) -> List[int]:
        # Order crossover (OX) on permutations; returns a child permutation.
        if n <= 2:
            return p1_order[:]
        a = random.randrange(n)
        b = random.randrange(n)
        if a > b:
            a, b = b, a
        if a == b:
            b = min(n, a + 1)
        child = [-1] * n
        # copy slice from p1
        child[a:b] = p1_order[a:b]
        used = set(child[a:b])
        # fill from p2 in order
        pos = b
        for x in p2_order:
            if x in used:
                continue
            if pos >= n:
                pos = 0
            while child[pos] != -1:
                pos += 1
                if pos >= n:
                    pos = 0
            child[pos] = x
        return child

    def chunk_inheritance(child_ok: List[float], packing: List[List[int]]) -> None:
        if not packing:
            return
        k = 2 if len(packing) < 10 else 3
        k = min(k, len(packing))
        chosen = random.sample(range(len(packing)), k)
        mn = min(child_ok)
        base = mn - 0.12
        spread = 0.02
        t = 0
        for bi in chosen:
            for item in packing[bi]:
                child_ok[item] = base + spread * (t / max(1, n))
                t += 1

    def mutate(order_keys: List[float], fit_keys: List[float], noise: float, pm_band: float) -> None:
        # light noise
        for i in range(n):
            r = random.random()
            if r < 0.018:
                order_keys[i] += random.uniform(-noise, noise)
            if r > 0.982:
                fit_keys[i] += random.uniform(-0.30, 0.30)

        # targeted bursts
        if random.random() < 0.18:
            reps = 4 if n <= 600 else 3
            for _ in range(reps):
                i = random.randrange(n)
                order_keys[i] += random.uniform(-2.5 * noise, 2.5 * noise)

        # band swaps
        if random.random() < pm_band:
            band = random.random()
            if band < 0.34 and heavy_items:
                items = heavy_items
            elif band < 0.68 and mid_items:
                items = mid_items
            else:
                items = small_items if small_items else list(range(n))
            reps = 6 if len(items) >= 6 else len(items)
            for _ in range(reps):
                a = random.choice(items)
                b = random.choice(items)
                if a != b:
                    order_keys[a], order_keys[b] = order_keys[b], order_keys[a]

        # keep some fk values in range
        for _ in range(3):
            i = random.randrange(n)
            if fit_keys[i] < 0.0 or fit_keys[i] >= 1.0:
                fit_keys[i] %= 1.0

    # ---------------- GA parameters ----------------
    pop_size = max(90, min(240, 70 + n // 12))
    tournament_k = 4 if n <= 700 else 3

    # More decode variants per evaluation (quality > speed; time budget can be up to 100s)
    base_decodes = 5 if n <= 350 else (4 if n <= 900 else 3)

    # Fixed iteration budget, with time checks to stop early
    max_steps = 420000 if n <= 450 else (300000 if n <= 1000 else 220000)

    init_sigma = 0.10 if n <= 250 else (0.075 if n <= 900 else 0.055)
    mut_noise = 0.06 if n <= 250 else (0.045 if n <= 900 else 0.035)
    pm_band = 0.22 if n <= 700 else 0.16

    crossover_rate = 0.92
    ox_rate = 0.22 if n <= 1200 else 0.16

    stagnation_window = 2600 if n <= 600 else (1800 if n <= 1100 else 1300)
    steps_since_improve = 0

    immigrant_every = 55

    # ---------------- Seeding ----------------
    seeds: List[Tuple[List[float], List[float]]] = []

    # 1) biased decreasing
    seeds.append(make_biased_keys(init_sigma))

    # 2) pure decreasing (FFD/BFD use decreasing, so it’s a strong baseline order)
    dec_order = list(range(n))
    dec_order.sort(key=lambda i: weights[i], reverse=True)
    seeds.append(keys_from_order(dec_order, init_sigma * 0.35))

    # 3) increasing (occasionally helps with some structures)
    inc_order = list(range(n))
    inc_order.sort(key=lambda i: weights[i])
    seeds.append(keys_from_order(inc_order, init_sigma * 0.55))

    # 4) interleave heavy + small to reduce fragmentation
    heavy_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    a, b = 0, n - 1
    inter = []
    while a <= b:
        inter.append(heavy_sorted[a])
        a += 1
        if a <= b:
            inter.append(heavy_sorted[b])
            b -= 1
    seeds.append(keys_from_order(inter, init_sigma * 0.45))

    # 5) modulo grouping seed
    g = 9
    mod_order = list(range(n))
    mod_order.sort(key=lambda i: (weights[i] % g, -weights[i]))
    seeds.append(keys_from_order(mod_order, init_sigma * 0.60))

    # 6) blockwise shuffled decreasing (preserve good decreasing structure but diversify)
    block = 16 if n <= 600 else 24
    for _ in range(8):
        perm = dec_order[:]
        # shuffle within blocks
        for s in range(0, n, block):
            e = min(n, s + block)
            sub = perm[s:e]
            random.shuffle(sub)
            perm[s:e] = sub
        seeds.append(keys_from_order(perm, init_sigma * 0.85))

    # additional biased + random
    for _ in range(10):
        seeds.append(make_biased_keys(init_sigma * 1.6))
    for _ in range(10):
        seeds.append(make_random_keys())

    while len(seeds) < pop_size:
        if random.random() < 0.62:
            seeds.append(make_biased_keys(init_sigma * 2.0))
        else:
            seeds.append(make_random_keys())

    # ---------------- Initialize population ----------------
    population: List[Dict[str, object]] = []

    best_fit = float("inf")
    best_pack: List[List[int]] = []
    best_bws: List[int] = []

    for ok, fk in seeds[:pop_size]:
        if time_exceeded() or not safe_to_decode():
            break
        f, pack, bws, order = eval_individual(ok, fk, base_decodes)
        population.append({"ok": ok, "fk": fk, "fit": f, "pack": pack, "bws": bws, "order": order})
        if f < best_fit:
            best_fit, best_pack, best_bws = f, pack, bws
            steps_since_improve = 0

    if not population:
        return {"packing": best_pack, "bin_weights": best_bws}

    def tournament_pick() -> int:
        best_i = 0
        best_f = float("inf")
        for _ in range(tournament_k):
            i = random.randrange(len(population))
            f = population[i]["fit"]  # type: ignore[index]
            if f < best_f:
                best_f = f
                best_i = i
        return best_i

    def worst_of_random_subset(k: int) -> int:
        worst_i = 0
        worst_f = -1.0
        for _ in range(k):
            i = random.randrange(len(population))
            f = population[i]["fit"]  # type: ignore[index]
            if f > worst_f:
                worst_f = f
                worst_i = i
        return worst_i

    # ---------------- Main GA loop ----------------
    # periodic time check stride (reduce overhead)
    time_check_stride = 64

    for step in range(max_steps):
        if (step & (time_check_stride - 1)) == 0:
            if time_exceeded() or not safe_to_decode():
                break

        best_bins = len(best_bws) if best_bws else 10**9
        near_lb = best_bins <= lb + 1

        # adaptive evaluation intensity
        decodes = base_decodes + (1 if near_lb else 0)

        # stagnation response
        if steps_since_improve >= stagnation_window:
            mut_noise *= 1.35
            pm_band = min(0.50, pm_band * 1.20)
            immigrant_every = max(18, immigrant_every - 10)
            steps_since_improve = 0
        else:
            base_noise = 0.06 if n <= 250 else (0.045 if n <= 900 else 0.035)
            mut_noise = 0.985 * mut_noise + 0.015 * base_noise
            base_pm = 0.22 if n <= 700 else 0.16
            pm_band = 0.985 * pm_band + 0.015 * base_pm
            immigrant_every = int(0.99 * immigrant_every + 0.01 * 55)

        # immigrants
        if step > 0 and immigrant_every > 0 and (step % immigrant_every) == 0:
            ok, fk = make_biased_keys(init_sigma * 2.4)
            if time_exceeded() or not safe_to_decode():
                break
            f, pack, bws, order = eval_individual(ok, fk, decodes)
            rep = worst_of_random_subset(12)
            if f < population[rep]["fit"]:  # type: ignore[index]
                population[rep] = {"ok": ok, "fk": fk, "fit": f, "pack": pack, "bws": bws, "order": order}
                if f < best_fit:
                    best_fit, best_pack, best_bws = f, pack, bws
            continue

        # selection
        i1 = tournament_pick()
        i2 = tournament_pick()
        p1 = population[i1]
        p2 = population[i2]

        p1_ok = p1["ok"]  # type: ignore[assignment]
        p1_fk = p1["fk"]  # type: ignore[assignment]
        p2_ok = p2["ok"]  # type: ignore[assignment]
        p2_fk = p2["fk"]  # type: ignore[assignment]

        # crossover
        if random.random() < crossover_rate:
            # sometimes do permutation OX crossover for stronger subsequence inheritance
            if random.random() < ox_rate:
                o1 = p1["order"]  # type: ignore[assignment]
                o2 = p2["order"]  # type: ignore[assignment]
                child_order = order_ox_crossover(o1, o2)
                c_ok, c_fk = keys_from_order(child_order, init_sigma * 0.25)
                # mix fit-keys by blending
                lam = random.random()
                c_fk = [lam * a + (1.0 - lam) * b for a, b in zip(p1_fk, p2_fk)]
            else:
                c_ok, c_fk = blend_crossover(p1_ok, p1_fk, p2_ok, p2_fk)  # type: ignore[arg-type]
        else:
            # clone better
            if p1["fit"] <= p2["fit"]:  # type: ignore[index]
                c_ok = list(p1_ok)  # type: ignore[arg-type]
                c_fk = list(p1_fk)  # type: ignore[arg-type]
            else:
                c_ok = list(p2_ok)  # type: ignore[arg-type]
                c_fk = list(p2_fk)  # type: ignore[arg-type]

        # chunk inheritance (memetic within GA)
        if random.random() < (0.20 if not near_lb else 0.12):
            donor = p1 if p1["fit"] <= p2["fit"] else p2  # type: ignore[index]
            chunk_inheritance(c_ok, donor["pack"])  # type: ignore[arg-type]

        # mutation (exploit more near lb)
        eff_noise = mut_noise * (0.65 if near_lb else 1.0)
        mutate(c_ok, c_fk, eff_noise, pm_band * (0.75 if near_lb else 1.0))

        if time_exceeded() or not safe_to_decode():
            break

        f, pack, bws, order = eval_individual(c_ok, c_fk, decodes)

        # replacement
        rep = worst_of_random_subset(8)
        if f < population[rep]["fit"]:  # type: ignore[index]
            population[rep] = {"ok": c_ok, "fk": c_fk, "fit": f, "pack": pack, "bws": bws, "order": order}

        if f < best_fit:
            best_fit, best_pack, best_bws = f, pack, bws
            steps_since_improve = 0
        else:
            steps_since_improve += 1

    return {"packing": best_pack, "bin_weights": best_bws}
