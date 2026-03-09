import random
import time
from bisect import bisect_left
from typing import List, Dict, Tuple, Optional


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

    # Avoid starting a decode when we're extremely close to the limit.
    # (Decodes should complete to avoid noisy/partial evaluations.)
    def safe_to_decode() -> bool:
        # Heuristic guard; keep small to not prematurely stop on tiny limits.
        return time_left() > 0.002

    # ---------------- Precomputation ----------------
    total_w = sum(weights)
    lb = (total_w + C - 1) // C
    # Simple L2-ish component: items > C/2 require own bin (at least).
    lb2 = sum(1 for w in weights if w * 2 > C)
    lb = max(lb, lb2)

    # Base decreasing order by weight
    base_order = list(range(n))
    base_order.sort(key=lambda i: weights[i], reverse=True)
    rank_pos = [0] * n
    for r, i in enumerate(base_order):
        rank_pos[i] = r

    # Weight classes (for band mutations)
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

    # ---------------- Chromosome (random keys with bias) ----------------
    # Each individual: (order_keys, fit_keys)

    def make_biased_keys(noise_sigma: float, random_fit: bool = True) -> Tuple[List[float], List[float]]:
        invn = 1.0 / n
        ok = [0.0] * n
        fk = [0.0] * n
        for i in range(n):
            base = rank_pos[i] * invn
            ok[i] = base + random.uniform(-noise_sigma, noise_sigma)
            fk[i] = random.random() if random_fit else (0.5 + random.uniform(-0.25, 0.25))
        return ok, fk

    def make_random_keys() -> Tuple[List[float], List[float]]:
        ok = [random.random() for _ in range(n)]
        fk = [random.random() for _ in range(n)]
        return ok, fk

    def keys_to_order(order_keys: List[float]) -> List[int]:
        # Stable sort: tie by heavier first to avoid pathological equal keys.
        return sorted(range(n), key=lambda i: (order_keys[i], -weights[i]))

    # ---------------- Bucketed decoder (Best-Fit family) ----------------
    # Coarse residual buckets + within-bucket scan.

    # Bucket size: target ~2048 buckets, but keep >=1.
    B = max(1, C // 2048)

    def decode(order: List[int], fit_keys: List[float], variant: int) -> Tuple[List[List[int]], List[int]]:
        """Constructive decoder.

        variant controls tie-breaking among candidate bins and whether we are
        slightly more best-fit vs first-fit.
        """
        # bins and weights
        bins: List[List[int]] = []
        bws: List[int] = []

        # map bin_id -> current remaining
        rems: List[int] = []

        # buckets: bucket_id -> list of bin_ids
        buckets: Dict[int, List[int]] = {}

        def add_bin_to_bucket(bin_id: int, rem: int) -> None:
            bid = rem // B
            lst = buckets.get(bid)
            if lst is None:
                buckets[bid] = [bin_id]
            else:
                lst.append(bin_id)

        def rebuild_buckets() -> None:
            buckets.clear()
            for bid, rem in enumerate(rems):
                add_bin_to_bucket(bid, rem)

        # limited lookahead buffer for "open-bin choice" policy
        # We'll keep a small window of not-yet-placed items we can swap within.
        # Implemented by scanning next K when opening a new bin.
        K = 4 if n <= 400 else 3

        # Periodic rebuild to mitigate drift of stale bucket lists.
        rebuild_every = 256 if n <= 1000 else 512

        pos = 0
        while pos < n:
            idx = order[pos]
            w = weights[idx]

            # Find best candidate bin using buckets
            start_bucket = w // B
            best_bin = -1
            best_res_after = C + 1
            best_secondary = None

            # Determine search direction/behavior by variant
            # variant 0: strict best-fit (min residual after)
            # variant 1: best-fit but with stronger load preference
            # variant 2: slightly more first-fit-ish (stop early)
            # variant 3: like 0 but different fit_key influence
            stop_early = (variant == 2)

            # search buckets from start_bucket upward
            b = start_bucket
            # cap bucket loop to avoid rare worst-case if C huge: at most explore
            # up to current max bucket.
            max_bucket = (max(rems) // B) if rems else -1
            while b <= max_bucket:
                cand = buckets.get(b)
                if cand:
                    # scan candidates
                    for bin_id in cand:
                        rem = rems[bin_id]
                        if rem < w:
                            continue
                        res_after = rem - w

                        # Primary: minimize res_after
                        if res_after < best_res_after:
                            best_res_after = res_after
                            best_bin = bin_id
                        elif res_after == best_res_after and best_bin != -1:
                            # Tie-breaks
                            # Prefer higher current load (lower rem)
                            # plus chromosome fit_key influences deterministically.
                            # Use different mixing per variant.
                            rem_best = rems[best_bin]
                            load_pref = rem_best - rem  # >0 means candidate more loaded
                            if variant == 1:
                                # stronger load preference
                                score = (load_pref, fit_keys[idx] - 0.5)
                            elif variant == 3:
                                # more fit_key influence
                                score = (fit_keys[idx] - 0.5, load_pref)
                            else:
                                score = (load_pref, -best_bin)

                            if best_secondary is None:
                                best_secondary = score
                            else:
                                # Compare current candidate score to best
                                if score > best_secondary:
                                    best_bin = bin_id
                                    best_secondary = score

                        if stop_early and best_res_after == 0:
                            break
                    if stop_early and best_res_after == 0:
                        break
                # If we already found a perfect fit, no need to search more.
                if best_res_after == 0:
                    break
                b += 1

            if best_bin != -1:
                # Place into best_bin: remove from old bucket by lazy strategy
                # (bucket lists can contain stale bins; we rebuild periodically).
                bins[best_bin].append(idx)
                bws[best_bin] += w
                rems[best_bin] -= w
                add_bin_to_bucket(best_bin, rems[best_bin])
                pos += 1
            else:
                # Open-bin choice policy: before opening, try swapping with next K
                # items to see if any can fit into an existing bin.
                swapped = False
                if rems and K > 0 and (pos + 1) < n:
                    end = min(n, pos + 1 + K)
                    # Try a few upcoming items; pick the first that fits somewhere.
                    # Use same bucket search but only to find existence.
                    for j in range(pos + 1, end):
                        idx2 = order[j]
                        w2 = weights[idx2]
                        sb2 = w2 // B
                        max_bucket2 = (max(rems) // B)
                        found = False
                        b2 = sb2
                        while b2 <= max_bucket2:
                            cand2 = buckets.get(b2)
                            if cand2:
                                for bin_id in cand2:
                                    if rems[bin_id] >= w2:
                                        found = True
                                        break
                                if found:
                                    break
                            b2 += 1
                        if found:
                            # swap this item forward
                            order[pos], order[j] = order[j], order[pos]
                            swapped = True
                            break

                if swapped:
                    continue

                # Actually open new bin
                bin_id = len(bins)
                bins.append([idx])
                bws.append(w)
                rem = C - w
                rems.append(rem)
                add_bin_to_bucket(bin_id, rem)
                pos += 1

            if (pos & 255) == 0:
                if (pos % rebuild_every) == 0:
                    rebuild_buckets()

        return bins, bws

    # ---------------- Fitness ----------------
    def fitness_from_bws(bws: List[int]) -> float:
        m = len(bws)
        # Slack stats
        slack_sum = 0
        slack_max = 0
        slack_sq = 0
        slacks = []
        for bw in bws:
            s = C - bw
            slack_sum += s
            if s > slack_max:
                slack_max = s
            slack_sq += s * s
            slacks.append(s)

        # Pressure: many bins with large slack are undesirable
        slacks.sort()
        median_slack = slacks[m // 2]
        big_slack_cnt = 0
        thr = max(median_slack, C // 4)
        for s in slacks:
            if s >= thr:
                big_slack_cnt += 1

        # Lexicographic-ish float: 1 bin dominates
        # Keep terms scale-stable across instance sizes.
        invC = 1.0 / C
        f = float(m)
        f += 1e-4 * (slack_sum * invC)
        f += 1e-6 * (slack_max * invC)
        f += 1e-7 * (slack_sq * (invC * invC))
        f += 5e-7 * (big_slack_cnt / max(1, m))
        return f

    # ---------------- Multi-decoder evaluation + memoization ----------------
    # LRU-ish dict for near-duplicate top-K orders
    TOPK = 64 if n >= 64 else n
    MEMO_CAP = 3000 if n <= 800 else 1500
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

        if sig in memo:
            memo_age[sig] = age_counter
            f, pack, bws = memo[sig]
            return f, pack, bws, order

        best_f = float("inf")
        best_pack: List[List[int]] = []
        best_bws: List[int] = []

        # Deterministic variant selection based on fit_keys aggregate
        # (stable across generations; reduces noisy selection)
        s = 0.0
        step = max(1, n // 16)
        for i in range(0, n, step):
            s += fit_keys[i]
        base_variant = int(s * 997) % 4

        # Evaluate a few variants
        for t in range(decodes):
            v = (base_variant + t) & 3
            pack, bws = decode(order[:], fit_keys, v)
            f = fitness_from_bws(bws)
            if f < best_f:
                best_f, best_pack, best_bws = f, pack, bws

        # Memo insert + eviction
        memo[sig] = (best_f, best_pack, best_bws)
        memo_age[sig] = age_counter
        if len(memo) > MEMO_CAP:
            # evict ~10% oldest
            k = max(1, MEMO_CAP // 10)
            oldest = sorted(memo_age.items(), key=lambda kv: kv[1])[:k]
            for key, _a in oldest:
                memo.pop(key, None)
                memo_age.pop(key, None)

        return best_f, best_pack, best_bws, order

    # ---------------- Crossover operators ----------------
    def blend_crossover(p1_ok: List[float], p1_fk: List[float], p2_ok: List[float], p2_fk: List[float]) -> Tuple[List[float], List[float]]:
        lam = random.random()
        child_ok = [0.0] * n
        child_fk = [0.0] * n
        for i in range(n):
            child_ok[i] = lam * p1_ok[i] + (1.0 - lam) * p2_ok[i]
            child_fk[i] = lam * p1_fk[i] + (1.0 - lam) * p2_fk[i]
        return child_ok, child_fk

    def chunk_inheritance(child_ok: List[float], packing: List[List[int]]) -> None:
        # Select a few bins and push their items earlier by lowering their keys.
        if not packing:
            return
        # number of bins to inherit
        k = 2 if len(packing) < 8 else 3
        k = min(k, len(packing))
        chosen = random.sample(range(len(packing)), k)
        # find current minimum key to anchor earlier placement
        mn = min(child_ok)
        # push selected items to near-front with small spread
        base = mn - 0.1
        spread = 0.02
        t = 0
        for bi in chosen:
            for item in packing[bi]:
                child_ok[item] = base + spread * (t / max(1, n))
                t += 1

    # ---------------- Mutation ----------------
    def mutate(order_keys: List[float], fit_keys: List[float], noise: float, pm_band: float) -> None:
        # Always a touch of noise
        for i in range(n):
            if random.random() < 0.02:
                order_keys[i] += random.uniform(-noise, noise)
            if random.random() < 0.02:
                fit_keys[i] += random.uniform(-0.25, 0.25)

        # Occasional stronger per-gene noise burst
        if random.random() < 0.15:
            for _ in range(3):
                i = random.randrange(n)
                order_keys[i] += random.uniform(-2.0 * noise, 2.0 * noise)

        # Weight-class mutation (permute within band by nudging keys)
        if random.random() < pm_band:
            band = random.random()
            if band < 0.33 and heavy_items:
                items = heavy_items
            elif band < 0.66 and mid_items:
                items = mid_items
            else:
                items = small_items if small_items else list(range(n))

            reps = 4 if len(items) >= 4 else len(items)
            for _ in range(reps):
                a = random.choice(items)
                b = random.choice(items)
                if a != b:
                    order_keys[a], order_keys[b] = order_keys[b], order_keys[a]

        # Keep fit_keys in [0,1) loosely to remain well-behaved
        for _ in range(2):
            i = random.randrange(n)
            if fit_keys[i] < 0.0:
                fit_keys[i] = 0.0
            elif fit_keys[i] >= 1.0:
                fit_keys[i] = fit_keys[i] % 1.0

    def late_bin_disruption(order_keys: List[float], fit_keys: List[float], packing: List[List[int]], bws: List[int]) -> None:
        # Push items from the worst-filled bins earlier.
        if len(packing) <= 1:
            return
        # choose last 1-2 bins by fill (highest slack)
        slacks = [(C - bw, bi) for bi, bw in enumerate(bws)]
        slacks.sort(reverse=True)
        take_bins = [slacks[0][1]]
        if len(slacks) >= 2 and random.random() < 0.5:
            take_bins.append(slacks[1][1])

        mn = min(order_keys)
        base = mn - 0.05
        for bi in take_bins:
            items = packing[bi]
            # perturb a few items (not all)
            t = 0
            for item in items:
                if random.random() < 0.6:
                    order_keys[item] = base + 0.01 * (t / max(1, len(items)))
                    t += 1

    # ---------------- GA parameters (steady-state) ----------------
    pop_size = max(50, min(160, 40 + n // 20))
    tournament_k = 4 if n <= 600 else 3

    # Multi-decode count
    decodes = 4 if n <= 250 else (3 if n <= 700 else 2)

    # iteration budget (fixed steps)
    max_steps = 200000 if n <= 400 else (120000 if n <= 900 else 60000)

    # noise sigma for initialization
    init_sigma = 0.12 if n <= 200 else (0.08 if n <= 800 else 0.05)

    # Base mutation noise (adaptive)
    mut_noise = 0.08 if n <= 200 else (0.05 if n <= 800 else 0.035)
    pm_band = 0.25 if n <= 500 else 0.18

    crossover_rate = 0.9

    # stagnation controls
    stagnation_window = 2000 if n <= 400 else (1200 if n <= 900 else 600)
    steps_since_improve = 0

    # biased immigrants
    immigrant_every = 50  # may adapt

    # ---------------- Population storage ----------------
    # Each entry: dict with keys: ok, fk, fit, pack, bws
    population: List[Dict[str, object]] = []

    def tournament_pick() -> int:
        best = None
        best_f = None
        for _ in range(tournament_k):
            i = random.randrange(len(population))
            f = population[i]["fit"]  # type: ignore[index]
            if best is None or f < best_f:
                best, best_f = i, f
        return best  # type: ignore[return-value]

    def worst_of_random_subset(k: int = 6) -> int:
        worst = None
        worst_f = None
        for _ in range(k):
            i = random.randrange(len(population))
            f = population[i]["fit"]  # type: ignore[index]
            if worst is None or f > worst_f:
                worst, worst_f = i, f
        return worst  # type: ignore[return-value]

    # ---------------- Seeding (raise floor) ----------------
    def keys_from_order(order: List[int], noise_sigma: float) -> Tuple[List[float], List[float]]:
        ok = [0.0] * n
        fk = [random.random() for _ in range(n)]
        invn = 1.0 / n
        for pos, item in enumerate(order):
            ok[item] = pos * invn + random.uniform(-noise_sigma, noise_sigma)
        return ok, fk

    # Deterministic seeds
    seeds: List[Tuple[List[float], List[float]]] = []
    seeds.append(make_biased_keys(init_sigma, random_fit=True))

    inc_order = list(range(n))
    inc_order.sort(key=lambda i: weights[i])
    seeds.append(keys_from_order(inc_order, init_sigma))

    # Modulo classes seed (group by weight % g then decreasing)
    g = 7
    mod_order = list(range(n))
    mod_order.sort(key=lambda i: (weights[i] % g, -weights[i]))
    seeds.append(keys_from_order(mod_order, init_sigma))

    # A few more biased perturbations
    for _ in range(6):
        seeds.append(make_biased_keys(init_sigma * 1.5, random_fit=True))

    # Some fully random
    for _ in range(6):
        seeds.append(make_random_keys())

    # Fill seeds to pop_size
    while len(seeds) < pop_size:
        # Mix biased and random
        if random.random() < 0.65:
            seeds.append(make_biased_keys(init_sigma * 2.0, random_fit=True))
        else:
            seeds.append(make_random_keys())

    # Evaluate initial population
    best_fit = float("inf")
    best_pack: List[List[int]] = []
    best_bws: List[int] = []

    for ok, fk in seeds[:pop_size]:
        if time_exceeded() or not safe_to_decode():
            break
        f, pack, bws, _order = eval_individual(ok, fk, decodes)
        population.append({"ok": ok, "fk": fk, "fit": f, "pack": pack, "bws": bws})
        if f < best_fit:
            best_fit, best_pack, best_bws = f, pack, bws
            steps_since_improve = 0

    if not population:
        return {"packing": best_pack, "bin_weights": best_bws}

    # If time ran out during init
    if time_exceeded() or not safe_to_decode():
        return {"packing": best_pack, "bin_weights": best_bws}

    # ---------------- Steady-state GA loop ----------------
    for step in range(max_steps):
        if time_exceeded() or not safe_to_decode():
            break

        # Adaptive behavior near LB: if we are within +1 bin of LB, intensify
        best_bins = len(best_bws) if best_bws else 10**9
        near_lb = (best_bins <= lb + 1)

        # Adapt mutation if stagnant
        if steps_since_improve >= stagnation_window:
            mut_noise *= 1.5
            pm_band = min(0.5, pm_band * 1.25)
            immigrant_every = max(20, immigrant_every - 10)
            steps_since_improve = 0  # reset the counter for next phase
        else:
            # gentle decay back toward baseline
            base_noise = 0.08 if n <= 200 else (0.05 if n <= 800 else 0.035)
            mut_noise = 0.98 * mut_noise + 0.02 * base_noise
            base_pm_band = 0.25 if n <= 500 else 0.18
            pm_band = 0.98 * pm_band + 0.02 * base_pm_band
            immigrant_every = int(0.98 * immigrant_every + 0.02 * 50)

        # Occasionally inject a biased immigrant
        if immigrant_every > 0 and (step % immigrant_every) == 0 and step > 0:
            ok, fk = make_biased_keys(init_sigma * 2.5, random_fit=True)
            if time_exceeded() or not safe_to_decode():
                break
            f, pack, bws, _ = eval_individual(ok, fk, decodes if not near_lb else max(2, decodes - 1))
            replace = worst_of_random_subset(10)
            if f < population[replace]["fit"]:  # type: ignore[index]
                population[replace] = {"ok": ok, "fk": fk, "fit": f, "pack": pack, "bws": bws}
                if f < best_fit:
                    best_fit, best_pack, best_bws = f, pack, bws
                    steps_since_improve = 0
            continue

        # Parent selection
        i1 = tournament_pick()
        i2 = tournament_pick()
        p1 = population[i1]
        p2 = population[i2]
        p1_ok = p1["ok"]  # type: ignore[assignment]
        p1_fk = p1["fk"]  # type: ignore[assignment]
        p2_ok = p2["ok"]  # type: ignore[assignment]
        p2_fk = p2["fk"]  # type: ignore[assignment]

        # Crossover
        if random.random() < crossover_rate:
            c_ok, c_fk = blend_crossover(p1_ok, p1_fk, p2_ok, p2_fk)  # type: ignore[arg-type]
        else:
            # clone better parent
            if p1["fit"] <= p2["fit"]:  # type: ignore[index]
                c_ok = list(p1_ok)  # type: ignore[arg-type]
                c_fk = list(p1_fk)  # type: ignore[arg-type]
            else:
                c_ok = list(p2_ok)  # type: ignore[arg-type]
                c_fk = list(p2_fk)  # type: ignore[arg-type]

        # Educated chunk inheritance occasionally, favoring better parent
        if random.random() < (0.18 if not near_lb else 0.10):
            donor = p1 if p1["fit"] <= p2["fit"] else p2  # type: ignore[index]
            chunk_inheritance(c_ok, donor["pack"])  # type: ignore[arg-type]

        # Mutation
        # If near LB, reduce noise to exploit
        eff_noise = mut_noise * (0.6 if near_lb else 1.0)
        mutate(c_ok, c_fk, eff_noise, pm_band * (0.7 if near_lb else 1.0))

        # Late-bin disruption occasionally (memetic-style mutation, still GA)
        if (step % (200 if n <= 600 else 300) == 0) and random.random() < 0.7:
            # Use best individual as guidance
            # (avoid extra decode; reuse stored best packing)
            late_bin_disruption(c_ok, c_fk, best_pack, best_bws)

        # Evaluate child
        if time_exceeded() or not safe_to_decode():
            break
        f, pack, bws, _ = eval_individual(c_ok, c_fk, decodes if not near_lb else max(2, decodes - 1))

        # Replacement: replace worst among a random subset, but avoid inserting exact signature duplicates too often
        replace = worst_of_random_subset(7)
        if f < population[replace]["fit"]:  # type: ignore[index]
            population[replace] = {"ok": c_ok, "fk": c_fk, "fit": f, "pack": pack, "bws": bws}

        # Update global best
        if f < best_fit:
            best_fit, best_pack, best_bws = f, pack, bws
            steps_since_improve = 0
            # If we hit LB exactly, still continue but we are effectively optimal under LB.
        else:
            steps_since_improve += 1

    return {"packing": best_pack, "bin_weights": best_bws}
