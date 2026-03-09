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

    # guard to avoid starting a decode too close to the end
    def safe_to_decode() -> bool:
        # decoder can be a bit heavy for large n
        return time_left() > (0.004 if n <= 1200 else 0.007)

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

    p92 = percentile_value(0.92)
    p55 = percentile_value(0.55)

    heavy_items = [i for i, w in enumerate(weights) if w >= p92]
    mid_items = [i for i, w in enumerate(weights) if p55 <= w < p92]
    small_items = [i for i, w in enumerate(weights) if w < p55]

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
        # stable tie-break: heavier first
        return sorted(range(n), key=lambda i: (order_keys[i], -weights[i]))

    def keys_from_order(order: List[int], noise_sigma: float) -> Tuple[List[float], List[float]]:
        invn = 1.0 / n
        ok = [0.0] * n
        fk = [random.random() for _ in range(n)]
        for pos, item in enumerate(order):
            ok[item] = pos * invn + random.uniform(-noise_sigma, noise_sigma)
        return ok, fk

    # ---------------- Strong decoder: best-fit core + controlled diversification ----------------
    # Data structure: sorted list of (remaining, bin_id).

    def decode(order: List[int], fit_keys: List[float], variant: int) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bws: List[int] = []
        rems: List[int] = []
        sorted_bins: List[Tuple[int, int]] = []

        def remove_pair(pair: Tuple[int, int]) -> None:
            i = bisect_left(sorted_bins, pair)
            while i < len(sorted_bins) and sorted_bins[i][0] == pair[0]:
                if sorted_bins[i] == pair:
                    sorted_bins.pop(i)
                    return
                i += 1
            i = bisect_left(sorted_bins, pair) - 1
            while i >= 0 and sorted_bins[i][0] == pair[0]:
                if sorted_bins[i] == pair:
                    sorted_bins.pop(i)
                    return
                i -= 1

        def choose_bin(w: int, item_idx: int) -> int:
            pos = bisect_left(sorted_bins, (w, -1))
            if pos >= len(sorted_bins):
                return -1

            # variant 0: strict best-fit
            if variant == 0:
                return sorted_bins[pos][1]

            # variant 1: best-fit with deterministic tie-breaking influenced by fit_keys
            if variant == 1:
                rem0 = sorted_bins[pos][0]
                best_bid = sorted_bins[pos][1]
                # derived pseudo-random but deterministic per item
                base_fk = fit_keys[item_idx]
                j = pos + 1
                while j < len(sorted_bins) and sorted_bins[j][0] == rem0:
                    bid = sorted_bins[j][1]
                    # prefer larger bid sometimes to diversify
                    score = ((base_fk * 1315423911.0) % 1.0, bid)
                    best_score = ((base_fk * 2654435761.0) % 1.0, best_bid)
                    if score > best_score:
                        best_bid = bid
                    j += 1
                return best_bid

            # variant 2: short-window regret/beam among feasible bins
            if variant == 2:
                window = 6
                end = min(len(sorted_bins), pos + window)
                best_bid = -1
                best_val = None
                fk = fit_keys[item_idx]
                # prefer tight but also avoid creating "awkward" large remainders near C/2
                half = C // 2
                for j in range(pos, end):
                    rem, bid = sorted_bins[j]
                    after = rem - w
                    # penalty if after is near half (harder to fill) and not small
                    awkward = abs(after - half)
                    # combine: smaller after is better; awkwardness is worse; fk adds mild shuffle
                    val = (-(after), -awkward, fk)
                    if best_val is None or val > best_val:
                        best_val = val
                        best_bid = bid
                return best_bid

            # variant 3: allow next/next+2 candidate occasionally (adds exploration)
            if variant == 3:
                # mostly best-fit, but fk selects among first few feasible
                window = 3
                end = min(len(sorted_bins), pos + window)
                if end == pos + 1:
                    return sorted_bins[pos][1]
                t = fit_keys[item_idx]
                pick = pos + int(t * (end - pos))
                if pick >= end:
                    pick = end - 1
                return sorted_bins[pick][1]

            return sorted_bins[pos][1]

        for idx in order:
            w = weights[idx]
            b = choose_bin(w, idx)
            if b == -1:
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
        slack_sum = 0
        slack_sq = 0
        slack_max = 0
        near_full = 0
        # count bins with very low slack (good)
        thr = max(1, C // 30)  # ~3.3%

        for bw in bws:
            s = C - bw
            slack_sum += s
            slack_sq += s * s
            if s > slack_max:
                slack_max = s
            if s <= thr:
                near_full += 1

        invC = 1.0 / C
        # Dominant: bin count
        f = float(m)
        # Secondary: reduce total slack and variance; encourage many near-full bins
        f += 1.5e-4 * (slack_sum * invC)
        f += 2.5e-7 * (slack_sq * (invC * invC))
        f += 4.0e-7 * (slack_max * invC)
        f += 6.0e-5 * ((m - near_full) / max(1.0, m))
        return f

    # ---------------- Evaluation + memoization ----------------
    TOPK = min(120, n)  # more robust signature
    MEMO_CAP = 9000 if n <= 900 else 5500
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

        # deterministic base variant from a few keys
        s = 0.0
        step = max(1, n // 32)
        for i in range(0, n, step):
            s += fit_keys[i]
        base_variant = int(s * 4099) & 3

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
            k = max(1, MEMO_CAP // 8)
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
        if n <= 2:
            return p1_order[:]
        a = random.randrange(n)
        b = random.randrange(n)
        if a > b:
            a, b = b, a
        if a == b:
            b = min(n, a + 1)
        child = [-1] * n
        child[a:b] = p1_order[a:b]
        used = set(child[a:b])
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
        # inherit a few well-packed bins by forcing their items early
        k = 3 if len(packing) >= 14 else 2
        k = min(k, len(packing))
        chosen = random.sample(range(len(packing)), k)
        mn = min(child_ok)
        base = mn - 0.18
        spread = 0.015
        t = 0
        for bi in chosen:
            for item in packing[bi]:
                child_ok[item] = base + spread * (t / max(1, n))
                t += 1

    def mutate(order_keys: List[float], fit_keys: List[float], noise: float, pm_band: float) -> None:
        # light noise
        for i in range(n):
            r = random.random()
            if r < 0.020:
                order_keys[i] += random.uniform(-noise, noise)
            if r > 0.985:
                fit_keys[i] += random.uniform(-0.35, 0.35)

        # targeted bursts
        if random.random() < 0.20:
            reps = 5 if n <= 600 else 4
            for _ in range(reps):
                i = random.randrange(n)
                order_keys[i] += random.uniform(-2.8 * noise, 2.8 * noise)

        # band swaps
        if random.random() < pm_band:
            band = random.random()
            if band < 0.36 and heavy_items:
                items = heavy_items
            elif band < 0.70 and mid_items:
                items = mid_items
            else:
                items = small_items if small_items else list(range(n))
            reps = 8 if len(items) >= 8 else len(items)
            for _ in range(reps):
                a = random.choice(items)
                b = random.choice(items)
                if a != b:
                    order_keys[a], order_keys[b] = order_keys[b], order_keys[a]

        # keep fk in range
        for _ in range(4):
            i = random.randrange(n)
            if fit_keys[i] < 0.0 or fit_keys[i] >= 1.0:
                fit_keys[i] %= 1.0

    # ---------------- Constructive seed orders ----------------
    def greedy_order_ffd() -> List[int]:
        return sorted(range(n), key=lambda i: weights[i], reverse=True)

    def greedy_order_bfd_tiebreak() -> List[int]:
        # still returns an order; the decoder will do BFD-like placement.
        # We create an order that helps BFD: decreasing, but with randomized tie-breaking in equal weights.
        idxs = list(range(n))
        idxs.sort(key=lambda i: (-weights[i], (i * 2654435761) % 1024))
        return idxs

    def greedy_pair_interleave() -> List[int]:
        # heavy + small alternation
        dec = sorted(range(n), key=lambda i: weights[i], reverse=True)
        a, b = 0, n - 1
        out = []
        while a <= b:
            out.append(dec[a])
            a += 1
            if a <= b:
                out.append(dec[b])
                b -= 1
        return out

    # ---------------- GA parameters (quality-leaning) ----------------
    pop_size = max(120, min(320, 90 + n // 10))
    elite_keep = max(6, pop_size // 18)
    tournament_k = 5 if n <= 900 else 4

    # more decode variants (we have more time budget now)
    base_decodes = 7 if n <= 350 else (6 if n <= 900 else 5)

    # fixed iteration budget; time checks stop early
    max_steps = 900000 if n <= 450 else (650000 if n <= 1000 else 480000)

    init_sigma = 0.095 if n <= 250 else (0.070 if n <= 900 else 0.052)
    mut_noise = 0.055 if n <= 250 else (0.042 if n <= 900 else 0.033)
    pm_band = 0.24 if n <= 700 else 0.18

    crossover_rate = 0.93
    ox_rate = 0.28 if n <= 1200 else 0.22

    stagnation_window = 4200 if n <= 600 else (3000 if n <= 1100 else 2400)
    steps_since_improve = 0

    immigrant_every = 65

    # ---------------- Seeding ----------------
    seeds: List[Tuple[List[float], List[float]]] = []

    dec_order = greedy_order_ffd()
    seeds.append(keys_from_order(dec_order, init_sigma * 0.30))
    seeds.append(keys_from_order(greedy_order_bfd_tiebreak(), init_sigma * 0.40))

    inc_order = list(range(n))
    inc_order.sort(key=lambda i: weights[i])
    seeds.append(keys_from_order(inc_order, init_sigma * 0.65))

    seeds.append(keys_from_order(greedy_pair_interleave(), init_sigma * 0.45))

    # modulo grouping seeds with a few moduli
    for g in (7, 9, 11, 13):
        mod_order = list(range(n))
        mod_order.sort(key=lambda i: (weights[i] % g, -weights[i]))
        seeds.append(keys_from_order(mod_order, init_sigma * 0.65))

    # blockwise shuffled decreasing
    block = 14 if n <= 600 else 22
    for _ in range(14):
        perm = dec_order[:]
        for s in range(0, n, block):
            e = min(n, s + block)
            sub = perm[s:e]
            random.shuffle(sub)
            perm[s:e] = sub
        seeds.append(keys_from_order(perm, init_sigma * 0.90))

    # biased + random
    for _ in range(16):
        seeds.append(make_biased_keys(init_sigma * 1.6))
    for _ in range(12):
        seeds.append(make_random_keys())

    while len(seeds) < pop_size:
        if random.random() < 0.70:
            seeds.append(make_biased_keys(init_sigma * 2.1))
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

    def worst_index() -> int:
        worst_i = 0
        worst_f = -1.0
        for i, ind in enumerate(population):
            f = ind["fit"]  # type: ignore[index]
            if f > worst_f:
                worst_f = f
                worst_i = i
        return worst_i

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
    time_check_stride = 96

    for step in range(max_steps):
        if (step % time_check_stride) == 0:
            if time_exceeded() or not safe_to_decode():
                break

        best_bins = len(best_bws) if best_bws else 10**9
        near_lb = best_bins <= lb + 1
        very_near = best_bins <= lb

        # adaptive evaluation intensity: increase near LB and late in the run
        prog = elapsed() / max(1e-9, time_limit)
        decodes = base_decodes
        if near_lb:
            decodes += 1
        if prog > 0.70:
            decodes += 1
        if very_near and prog > 0.82:
            decodes += 1

        # stagnation response
        if steps_since_improve >= stagnation_window:
            mut_noise *= 1.40
            pm_band = min(0.55, pm_band * 1.22)
            immigrant_every = max(20, immigrant_every - 12)
            steps_since_improve = 0
        else:
            base_noise = 0.055 if n <= 250 else (0.042 if n <= 900 else 0.033)
            mut_noise = 0.987 * mut_noise + 0.013 * base_noise
            base_pm = 0.24 if n <= 700 else 0.18
            pm_band = 0.987 * pm_band + 0.013 * base_pm
            immigrant_every = int(0.992 * immigrant_every + 0.008 * 65)

        # elitism refresh: periodically re-evaluate elites with higher decode count
        if step > 0 and (step % 2500) == 0 and safe_to_decode():
            # pick current top elites
            elites = sorted(population, key=lambda ind: ind["fit"])[:elite_keep]  # type: ignore[index]
            for ind in elites:
                ok = ind["ok"]  # type: ignore[assignment]
                fk = ind["fk"]  # type: ignore[assignment]
                f, pack, bws, order = eval_individual(ok, fk, decodes + 2)
                ind["fit"] = f
                ind["pack"] = pack
                ind["bws"] = bws
                ind["order"] = order
                if f < best_fit:
                    best_fit, best_pack, best_bws = f, pack, bws

        # immigrants
        if step > 0 and immigrant_every > 0 and (step % immigrant_every) == 0:
            ok, fk = make_biased_keys(init_sigma * 2.5)
            if time_exceeded() or not safe_to_decode():
                break
            f, pack, bws, order = eval_individual(ok, fk, decodes)
            rep = worst_of_random_subset(14)
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
            if random.random() < ox_rate:
                o1 = p1["order"]  # type: ignore[assignment]
                o2 = p2["order"]  # type: ignore[assignment]
                child_order = order_ox_crossover(o1, o2)
                c_ok, _ = keys_from_order(child_order, init_sigma * 0.22)
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

        # chunk inheritance
        if random.random() < (0.22 if not near_lb else 0.14):
            donor = p1 if p1["fit"] <= p2["fit"] else p2  # type: ignore[index]
            chunk_inheritance(c_ok, donor["pack"])  # type: ignore[arg-type]

        # mutation (more exploitation near LB)
        eff_noise = mut_noise * (0.60 if near_lb else 1.0)
        mutate(c_ok, c_fk, eff_noise, pm_band * (0.72 if near_lb else 1.0))

        if time_exceeded() or not safe_to_decode():
            break

        f, pack, bws, order = eval_individual(c_ok, c_fk, decodes)

        # replacement with elitism protection
        # do not overwrite top elites unless strictly better
        rep = worst_of_random_subset(10)
        if rep < len(population):
            # crude elite protection: if rep is among best elite_keep, use global worst instead
            if rep in {i for i, _ in enumerate(sorted(population, key=lambda ind: ind["fit"])[:elite_keep])}:
                rep = worst_index()

        if f < population[rep]["fit"]:  # type: ignore[index]
            population[rep] = {"ok": c_ok, "fk": c_fk, "fit": f, "pack": pack, "bws": bws, "order": order}

        if f < best_fit:
            best_fit, best_pack, best_bws = f, pack, bws
            steps_since_improve = 0
        else:
            steps_since_improve += 1

    return {"packing": best_pack, "bin_weights": best_bws}
