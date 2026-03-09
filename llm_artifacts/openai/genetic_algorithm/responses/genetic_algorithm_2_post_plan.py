import time
import random
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()
    deadline = start + max(0.0, time_limit)

    C = int(bin_capacity)
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # If time is extremely tiny, return a fast FFD packing.
    if time_limit <= 1e-4:
        order = sorted(range(n), key=lambda i: (-weights[i], i))
        packing, loads = _ffd_decode(order, weights, C)
        return {"packing": packing, "bin_weights": loads}

    # ----------------- Preprocessing & lower bound -----------------
    total_weight = sum(weights)
    lb0 = (total_weight + C - 1) // C
    half = C / 2.0
    lb1 = max(lb0, sum(1 for w in weights if w > half))
    LB = lb1

    # Items and some helper orders
    idxs = list(range(n))
    order_desc = sorted(idxs, key=lambda i: (-weights[i], i))  # FFD order
    order_asc = sorted(idxs, key=lambda i: (weights[i], i))

    # ----------------- Decoder: permutation -> bins -----------------
    # Multi-rule decoder (FF/BF-ish) with time checks.

    # Remaining-capacity buckets for BF when C is moderate.
    # For large C, we fallback to capped candidate scanning.
    BUCKET_MAX_C = 5000

    def decode_perm(perm: List[int], rule: int) -> Optional[Tuple[List[List[int]], List[int]]]:
        # rule: 0=best-fit (linear/capped), 1=first-fit, 2=best-fit-buckets (if possible)
        # Time check periodically; caller may call often.
        if time.perf_counter() >= deadline:
            return None

        bins: List[List[int]] = []
        loads: List[int] = []

        if rule == 2 and C <= BUCKET_MAX_C:
            # buckets[r] = list of bin indices that currently have remaining capacity r
            buckets: List[List[int]] = [[] for _ in range(C + 1)]
            rems: List[int] = []

            for t, it in enumerate(perm):
                if (t & 255) == 0 and time.perf_counter() >= deadline:
                    return None
                w = weights[it]
                if w > C:
                    # Infeasible instance; still pack as singleton to avoid crash.
                    bins.append([it])
                    loads.append(w)
                    rems.append(max(0, C - w))
                    continue

                # Best-fit: find smallest remaining >= w -> rem_after minimal.
                best_bin = -1
                # Scan remaining from w..C, stop at first non-empty bucket.
                for r in range(w, C + 1):
                    if buckets[r]:
                        # Tie-break: choose bin with larger load (equivalently smaller remaining);
                        # within same remaining bucket, any is fine.
                        best_bin = buckets[r].pop()
                        break

                if best_bin == -1:
                    j = len(bins)
                    bins.append([it])
                    load = w
                    loads.append(load)
                    rnew = C - load
                    rems.append(rnew)
                    buckets[rnew].append(j)
                else:
                    # Update bucket membership
                    old_r = rems[best_bin]
                    # old_r bucket already popped
                    bins[best_bin].append(it)
                    loads[best_bin] += w
                    new_r = old_r - w
                    rems[best_bin] = new_r
                    buckets[new_r].append(best_bin)

            return bins, loads

        # Fallback: keep rems and perform either FF or BF with capped candidates.
        rems: List[int] = []

        # Candidate cap for BF when C large.
        CAP = 16

        for t, it in enumerate(perm):
            if (t & 255) == 0 and time.perf_counter() >= deadline:
                return None
            w = weights[it]
            if w > C:
                bins.append([it])
                loads.append(w)
                rems.append(max(0, C - w))
                continue

            if rule == 1:
                # First-fit with mild tie-breaking: earliest bin that fits.
                placed = False
                for j, r in enumerate(rems):
                    if w <= r:
                        bins[j].append(it)
                        loads[j] += w
                        rems[j] -= w
                        placed = True
                        break
                if not placed:
                    bins.append([it])
                    loads.append(w)
                    rems.append(C - w)
            else:
                # Best-fit: choose bin with minimal rem_after, tie-break by smaller rem.
                best_j = -1
                best_rem_after = C + 1

                m = len(rems)
                if m <= CAP:
                    cand_idx = range(m)
                else:
                    # sample some bins + always include a few most promising by remaining
                    cand = set()
                    for _ in range(CAP):
                        cand.add(random.randrange(m))
                    # Add a few bins with smallest remaining (tighter bins are usually better)
                    # Cheap selection of k tightest via partial scan
                    k = min(6, m)
                    tight = sorted(range(m), key=lambda j: rems[j])[:k]
                    cand.update(tight)
                    cand_idx = cand

                for j in cand_idx:
                    r = rems[j]
                    if w <= r:
                        ra = r - w
                        if ra < best_rem_after:
                            best_rem_after = ra
                            best_j = j
                            if ra == 0:
                                break
                        elif ra == best_rem_after and best_j != -1:
                            # tie-break: choose tighter (smaller remaining before placement)
                            if r < rems[best_j]:
                                best_j = j

                if best_j == -1:
                    bins.append([it])
                    loads.append(w)
                    rems.append(C - w)
                else:
                    bins[best_j].append(it)
                    loads[best_j] += w
                    rems[best_j] -= w

        return bins, loads

    # ----------------- Fitness: (bins, sum_sq_leftover, waste) -----------------
    def fitness_of_decoded(loads: List[int]) -> Tuple[int, int, int]:
        b = len(loads)
        # squared leftover measure
        ss = 0
        for load in loads:
            r = C - load
            ss += r * r
        waste = b * C - sum(loads)
        return (b, ss, waste)

    # ----------------- Cache (small LRU) -----------------
    # Cache by signature to reduce decoding of duplicates.
    # For large n, signature uses first/last K elements.
    CACHE_MAX = 4000
    SIG_K = 18

    cache: "OrderedDict[Tuple[int, ...], Tuple[Tuple[int, int, int], int]]" = OrderedDict()

    def signature(perm: List[int]) -> Tuple[int, ...]:
        if n <= 120:
            return tuple(perm)
        k = min(SIG_K, n)
        return tuple(perm[:k] + perm[-k:])

    def evaluate_perm(perm: List[int]) -> Optional[Tuple[Tuple[int, int, int], List[List[int]], List[int]]]:
        if time.perf_counter() >= deadline:
            return None
        sig = signature(perm)
        if sig in cache:
            fit, rule = cache[sig]
            cache.move_to_end(sig)
            dec = decode_perm(perm, rule)
            if dec is None:
                return None
            packing, loads = dec
            # If rule produced a different bin count due to time-dependent randomness in decoder
            # (shouldn't happen for deterministic rules), recompute.
            fit2 = fitness_of_decoded(loads)
            if fit2 != fit:
                fit = fit2
                cache[sig] = (fit, rule)
            return fit, packing, loads

        # Choose among rules; mild randomization helps diversification.
        r = random.random()
        if r < 0.55:
            rule = 2  # buckets BF when possible
        elif r < 0.80:
            rule = 0  # BF capped
        else:
            rule = 1  # FF

        dec = decode_perm(perm, rule)
        if dec is None:
            return None
        packing, loads = dec
        fit = fitness_of_decoded(loads)

        cache[sig] = (fit, rule)
        if len(cache) > CACHE_MAX:
            cache.popitem(last=False)
        return fit, packing, loads

    # ----------------- Initialization -----------------
    def noisy_desc(scale: float = 0.35) -> List[int]:
        return sorted(idxs, key=lambda i: -(weights[i] * (1.0 + scale * (random.random() - 0.5))))

    def grouped_shuffle() -> List[int]:
        # shuffle within equal-weight groups after sorting descending by weight
        groups: Dict[int, List[int]] = {}
        for i in idxs:
            groups.setdefault(weights[i], []).append(i)
        ws = sorted(groups.keys(), reverse=True)
        out: List[int] = []
        for w in ws:
            g = groups[w]
            random.shuffle(g)
            out.extend(g)
        return out

    # Population size (plan J)
    if n <= 200:
        pop_size = 100
    elif n <= 1000:
        pop_size = 80
    else:
        pop_size = 60

    elite_target = max(4, pop_size // 12)
    tour_k = 3 if pop_size < 70 else 4

    population: List[List[int]] = []
    population.append(order_desc[:])
    population.append(order_asc[:])
    population.append(grouped_shuffle())
    for _ in range(min(8, pop_size // 6)):
        population.append(noisy_desc())

    while len(population) < pop_size:
        p = idxs[:]
        random.shuffle(p)
        population.append(p)

    # Incumbent from FFD
    init_dec = decode_perm(order_desc, 2 if C <= BUCKET_MAX_C else 0)
    if init_dec is None:
        packing, loads = _ffd_decode(order_desc, weights, C)
        return {"packing": packing, "bin_weights": loads}

    best_pack, best_loads = init_dec
    best_fit = fitness_of_decoded(best_loads)
    best_bins = best_fit[0]

    # Evaluate initial population
    fits: List[Tuple[int, int, int]] = [(10**9, 10**18, 10**18)] * pop_size
    for i in range(pop_size):
        if time.perf_counter() >= deadline:
            return {"packing": best_pack, "bin_weights": best_loads}
        ev = evaluate_perm(population[i])
        if ev is None:
            return {"packing": best_pack, "bin_weights": best_loads}
        fit, pack, loads = ev
        fits[i] = fit
        if fit < best_fit:
            best_fit, best_pack, best_loads = fit, pack, loads
            best_bins = best_fit[0]

    # ----------------- Diversity filtering for elites -----------------
    sample_pos = [0, n // 4, n // 2, (3 * n) // 4, n - 1] if n >= 5 else list(range(n))

    def similarity(a: List[int], b: List[int]) -> int:
        # cheap: identical items in sampled positions
        s = 0
        for p in sample_pos:
            if a[p] == b[p]:
                s += 1
        return s

    # ----------------- Selection -----------------
    def tournament_select() -> int:
        cand = random.randrange(pop_size)
        for _ in range(tour_k - 1):
            j = random.randrange(pop_size)
            if fits[j] < fits[cand]:
                cand = j
        return cand

    # ----------------- Crossovers -----------------
    def ox(p1: List[int], p2: List[int]) -> List[int]:
        a = random.randrange(n)
        b = random.randrange(n)
        if a > b:
            a, b = b, a
        child = [-1] * n
        child[a:b+1] = p1[a:b+1]
        used = set(child[a:b+1])
        pos = (b + 1) % n
        for x in p2:
            if x not in used:
                child[pos] = x
                pos = (pos + 1) % n
        return child

    def pmx(p1: List[int], p2: List[int]) -> List[int]:
        a = random.randrange(n)
        b = random.randrange(n)
        if a > b:
            a, b = b, a
        child = [-1] * n
        child[a:b+1] = p1[a:b+1]
        pos_in_child = {}
        for i in range(a, b + 1):
            pos_in_child[child[i]] = i

        for i in range(a, b + 1):
            x = p2[i]
            if x in pos_in_child:
                continue
            y = p1[i]
            # find where to place x by following mapping
            j = i
            while True:
                y2 = p2[j]
                if y2 not in pos_in_child:
                    break
                j = pos_in_child[y2]
            child[j] = x
            pos_in_child[x] = j

        # fill remaining from p2
        for i in range(n):
            if child[i] == -1:
                child[i] = p2[i]
        return child

    def bin_preserving(p1: List[int], p2: List[int]) -> Optional[List[int]]:
        # decode both parents (time-bounded); copy some bins from p1 then fill from p2 order.
        if time.perf_counter() >= deadline:
            return None
        d1 = decode_perm(p1, 2 if C <= BUCKET_MAX_C else 0)
        if d1 is None:
            return None
        bins1, loads1 = d1
        if len(bins1) == 0:
            return p2[:]

        # prefer taking some of the fullest bins from p1
        bcount = len(bins1)
        idx_bins = list(range(bcount))
        idx_bins.sort(key=lambda j: loads1[j], reverse=True)
        take = 1 if bcount < 6 else random.choice([1, 2, 2, 3])
        take = min(take, bcount)
        chosen = set(idx_bins[:take])

        prefix: List[int] = []
        used = set()
        for j in idx_bins:
            if j in chosen:
                for it in bins1[j]:
                    if it not in used:
                        prefix.append(it)
                        used.add(it)

        child = prefix
        for it in p2:
            if it not in used:
                child.append(it)
                used.add(it)
        if len(child) != n:
            # fallback safety
            rem = [it for it in idxs if it not in used]
            child.extend(rem)
        return child

    # ----------------- Mutation (adaptive intensity) -----------------
    def mutate(perm: List[int], intensity: int) -> None:
        # intensity controls how many operations
        # ops: swap, insertion, inversion; occasional scramble
        ops = 1 + intensity
        for _ in range(ops):
            r = random.random()
            if r < 0.45:
                i = random.randrange(n)
                j = random.randrange(n)
                perm[i], perm[j] = perm[j], perm[i]
            elif r < 0.75:
                i = random.randrange(n)
                j = random.randrange(n)
                if i != j:
                    x = perm.pop(i)
                    perm.insert(j, x)
            else:
                a = random.randrange(n)
                b = random.randrange(n)
                if a > b:
                    a, b = b, a
                if b - a >= 2:
                    perm[a:b+1] = reversed(perm[a:b+1])

        if random.random() < (0.04 + 0.02 * intensity) and n >= 10:
            a = random.randrange(n)
            b = random.randrange(n)
            if a > b:
                a, b = b, a
            if b - a >= 3:
                seg = perm[a:b+1]
                random.shuffle(seg)
                perm[a:b+1] = seg

    # ----------------- Memetic local improvement -----------------
    # Use bin-emptying ruin-and-recreate + quick relocates.

    def local_improve(perm: List[int], base_fit: Tuple[int, int, int]) -> Optional[Tuple[List[int], Tuple[int, int, int], List[List[int]], List[int]]]:
        if time.perf_counter() >= deadline:
            return None

        # Only intensify near incumbent
        if base_fit[0] > best_bins + 1:
            return None

        dec = decode_perm(perm, 2 if C <= BUCKET_MAX_C else 0)
        if dec is None:
            return None
        packing, loads = dec
        fit = fitness_of_decoded(loads)

        # pick 1-3 least-filled bins to empty
        b = len(loads)
        if b <= 1:
            return perm, fit, packing, loads

        # Attempt budget
        max_steps = 220
        step = 0

        # Prepare rems
        rems = [C - ld for ld in loads]

        # Choose bins to ruin
        bin_order = sorted(range(b), key=lambda j: loads[j])
        k = 1 if b < 8 else random.choice([1, 2, 2, 3])
        k = min(k, b - 1)
        to_remove = set(bin_order[:k])

        pool: List[int] = []
        new_packing: List[List[int]] = []
        new_loads: List[int] = []
        new_rems: List[int] = []

        for j in range(b):
            if j in to_remove:
                pool.extend(packing[j])
            else:
                new_packing.append(packing[j][:])
                new_loads.append(loads[j])
                new_rems.append(rems[j])

        # Reinsert pool items heavy-to-light
        pool.sort(key=lambda i: weights[i], reverse=True)

        # reinsertion via best-fit on current bins
        for it in pool:
            if (step & 63) == 0 and time.perf_counter() >= deadline:
                return None
            step += 1
            w = weights[it]
            best_j = -1
            best_ra = C + 1
            # exact-fit preference
            for j, r in enumerate(new_rems):
                if w <= r:
                    ra = r - w
                    if ra < best_ra:
                        best_ra = ra
                        best_j = j
                        if ra == 0:
                            break
            if best_j == -1:
                new_packing.append([it])
                new_loads.append(w)
                new_rems.append(C - w if w <= C else 0)
            else:
                new_packing[best_j].append(it)
                new_loads[best_j] += w
                new_rems[best_j] -= w

        # Quick relocate attempts: try to empty a bin by moving its items elsewhere.
        # Do limited random attempts.
        for _ in range(max_steps):
            if (_ & 31) == 0 and time.perf_counter() >= deadline:
                return None
            if len(new_loads) <= 1:
                break
            # pick a light bin
            j = min(range(len(new_loads)), key=lambda x: new_loads[x])
            items_j = new_packing[j]
            if not items_j:
                # remove empty
                new_packing.pop(j)
                new_loads.pop(j)
                new_rems.pop(j)
                continue

            moved_all = True
            # try move items one by one (heaviest first)
            for it in sorted(items_j, key=lambda i: weights[i], reverse=True):
                w = weights[it]
                dest = -1
                best_ra = C + 1
                for d, r in enumerate(new_rems):
                    if d == j:
                        continue
                    if w <= r:
                        ra = r - w
                        if ra < best_ra:
                            best_ra = ra
                            dest = d
                            if ra == 0:
                                break
                if dest == -1:
                    moved_all = False
                    break
                # move
                new_packing[dest].append(it)
                new_loads[dest] += w
                new_rems[dest] -= w

            if moved_all:
                # eliminate bin j
                new_packing.pop(j)
                new_loads.pop(j)
                new_rems.pop(j)
            else:
                break

        new_fit = fitness_of_decoded(new_loads)
        if new_fit <= fit:
            # Build an improved permutation by concatenating bins order (heaviest bins first)
            # This is a standard way to turn packing back into an order.
            order_bins = sorted(range(len(new_loads)), key=lambda j: new_loads[j], reverse=True)
            improved_perm: List[int] = []
            used = set()
            for j in order_bins:
                for it in new_packing[j]:
                    if it not in used:
                        improved_perm.append(it)
                        used.add(it)
            # append any missing (shouldn't happen)
            if len(improved_perm) != n:
                for it in idxs:
                    if it not in used:
                        improved_perm.append(it)
                        used.add(it)
            return improved_perm, new_fit, new_packing, new_loads

        return None

    # ----------------- Steady-state memetic GA loop (fixed iterations) -----------------
    MAX_ITERS = 200000 if n <= 400 else 140000

    crossover_rate = 0.92
    p_ls_early = 0.15
    p_ls_late = 0.35

    stall = 0
    best_seen_at = 0

    def replace_worst(child: List[int], child_fit: Tuple[int, int, int], child_pack: List[List[int]], child_loads: List[int]) -> None:
        nonlocal best_fit, best_pack, best_loads, best_bins, stall, best_seen_at

        # find worst index
        worst = 0
        for i in range(1, pop_size):
            if fits[i] > fits[worst]:
                worst = i

        # Diversity: avoid inserting near-duplicate of best individual
        bi = min(range(pop_size), key=lambda i: fits[i])
        if similarity(child, population[bi]) >= len(sample_pos) - 1:
            # accept only if clearly better than worst
            if child_fit >= fits[worst]:
                return

        if child_fit < fits[worst]:
            population[worst] = child
            fits[worst] = child_fit

        if child_fit < best_fit:
            best_fit = child_fit
            best_pack = child_pack
            best_loads = child_loads
            best_bins = best_fit[0]
            stall = 0
            best_seen_at = it
        else:
            stall += 1

    for it in range(MAX_ITERS):
        if (it & 63) == 0:
            if time.perf_counter() >= deadline:
                break
            if best_bins <= LB:
                # LB is a valid bound; if met, we can stop early.
                break

        # Adaptive phase
        frac = it / MAX_ITERS
        p_ls = p_ls_early if frac < 0.20 else (p_ls_late if frac > 0.65 else 0.22)

        # Increase mutation intensity on stall
        if stall < 800:
            intensity = 0
        elif stall < 2500:
            intensity = 1
        else:
            intensity = 2

        # Diversification injection when very stalled
        if stall > 6000 and (it - best_seen_at) > 6000:
            # replace a few worst with noisy desc and mutated-best
            bi = min(range(pop_size), key=lambda i: fits[i])
            base = population[bi][:]
            for _ in range(max(2, pop_size // 12)):
                if time.perf_counter() >= deadline:
                    break
                widx = max(range(pop_size), key=lambda i: fits[i])
                if random.random() < 0.6:
                    population[widx] = noisy_desc()
                else:
                    nb = base[:]
                    mutate(nb, 3)
                    population[widx] = nb
                ev = evaluate_perm(population[widx])
                if ev is None:
                    break
                fits[widx] = ev[0]
            stall = 0

        # Parent selection
        p1 = population[tournament_select()]
        p2 = population[tournament_select()]
        if p1 is p2 and pop_size > 1:
            p2 = population[(population.index(p1) + random.randrange(1, pop_size)) % pop_size]

        # Crossover
        child: Optional[List[int]]
        if random.random() < crossover_rate:
            r = random.random()
            if r < 0.20:
                child = bin_preserving(p1, p2)
                if child is None:
                    child = ox(p1, p2)
            elif r < 0.70:
                child = ox(p1, p2)
            else:
                child = pmx(p1, p2)
        else:
            child = p1[:]

        # Mutation (child-level)
        if random.random() < (0.28 + 0.08 * intensity):
            mutate(child, intensity)

        # Evaluate
        ev = evaluate_perm(child)
        if ev is None:
            break
        child_fit, child_pack, child_loads = ev

        # Memetic improvement
        if random.random() < p_ls:
            imp = local_improve(child, child_fit)
            if imp is None:
                pass
            else:
                child, child_fit, child_pack, child_loads = imp

        # Replacement
        replace_worst(child, child_fit, child_pack, child_loads)

        # Occasionally improve elites (cheap because few)
        if (it % 2000) == 0 and it > 0:
            if time.perf_counter() >= deadline:
                break
            # pick a few best and try improvement
            elite_idx = sorted(range(pop_size), key=lambda i: fits[i])[:elite_target]
            # filter near-duplicates
            filtered = []
            for i in elite_idx:
                ok = True
                for j in filtered:
                    if similarity(population[i], population[j]) >= len(sample_pos) - 1:
                        ok = False
                        break
                if ok:
                    filtered.append(i)
            for i in filtered[:max(2, elite_target // 2)]:
                if time.perf_counter() >= deadline:
                    break
                imp = local_improve(population[i], fits[i])
                if imp is None:
                    continue
                new_perm, new_fit, new_pack, new_loads = imp
                if new_fit < fits[i]:
                    population[i] = new_perm
                    fits[i] = new_fit
                    if new_fit < best_fit:
                        best_fit, best_pack, best_loads = new_fit, new_pack, new_loads
                        best_bins = best_fit[0]
                        stall = 0
                        best_seen_at = it

    return {"packing": best_pack, "bin_weights": best_loads}


def _ffd_decode(order: List[int], weights: List[int], C: int) -> Tuple[List[List[int]], List[int]]:
    bins: List[List[int]] = []
    loads: List[int] = []
    rems: List[int] = []
    for it in order:
        w = weights[it]
        best_j = -1
        best_ra = C + 1
        for j, r in enumerate(rems):
            if w <= r:
                ra = r - w
                if ra < best_ra:
                    best_ra = ra
                    best_j = j
                    if ra == 0:
                        break
        if best_j == -1:
            bins.append([it])
            loads.append(w)
            rems.append(max(0, C - w))
        else:
            bins[best_j].append(it)
            loads[best_j] += w
            rems[best_j] -= w
    return bins, loads
