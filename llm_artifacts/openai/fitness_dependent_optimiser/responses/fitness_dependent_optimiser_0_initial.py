import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---------- Decoder: Best-Fit on a given order ----------
    # Returns (packing, bin_weights)
    def decode_best_fit(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bw: List[int] = []
        for idx in order:
            w = weights[idx]
            # best fit: smallest remaining space that can accommodate
            best_bin = -1
            best_rem = bin_capacity + 1
            for b in range(len(bins)):
                rem = bin_capacity - bw[b]
                if w <= rem:
                    rem2 = rem - w
                    if rem2 < best_rem:
                        best_rem = rem2
                        best_bin = b
                        if best_rem == 0:
                            break
            if best_bin == -1:
                bins.append([idx])
                bw.append(w)
            else:
                bins[best_bin].append(idx)
                bw[best_bin] += w
        return bins, bw

    # Fitness: primary minimize bins, secondary maximize total utilization (equiv minimize waste)
    # We use a scalar where lower is better.
    def fitness_from(bins: List[List[int]], bw: List[int]) -> Tuple[int, int]:
        # (bins_count, waste)
        waste = 0
        for s in bw:
            waste += (bin_capacity - s)
        return (len(bins), waste)

    # Compare (bins, waste)
    def better(f1: Tuple[int, int], f2: Tuple[int, int]) -> bool:
        return f1[0] < f2[0] or (f1[0] == f2[0] and f1[1] < f2[1])

    # ---------- Initial solutions ----------
    items = list(range(n))

    # A strong deterministic seed: sort by weight descending
    items_sorted = sorted(items, key=lambda i: weights[i], reverse=True)

    best_packing, best_bw = decode_best_fit(items_sorted)
    best_fit = fitness_from(best_packing, best_bw)
    best_order = items_sorted[:]

    # ---------- Population (scouts) ----------
    # Keep population modest for speed.
    pop_size = max(12, min(40, 10 + n // 25))

    def random_perturb(base: List[int]) -> List[int]:
        a = base[:]
        # apply a few random swaps/inversions
        k = 2 if n < 80 else 3
        for _ in range(k):
            r = random.random()
            if r < 0.55:
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

    pop_orders: List[List[int]] = [best_order[:]]
    # Mix of random permutations and perturbed sorted order
    for p in range(pop_size - 1):
        if p % 2 == 0:
            o = random_perturb(items_sorted)
        else:
            o = items[:]
            random.shuffle(o)
        pop_orders.append(o)

    pop_fit: List[Tuple[int, int]] = []
    for o in pop_orders:
        bins, bw = decode_best_fit(o)
        f = fitness_from(bins, bw)
        pop_fit.append(f)
        if better(f, best_fit):
            best_fit, best_packing, best_bw, best_order = f, bins, bw, o[:]

    # ---------- FDO Movement Operators (permutation moves) ----------
    # We generate a new order influenced by best_order.

    # Move 1: guided insertion: take some items and place them to match relative position in best.
    def guided_insertion(order: List[int], best: List[int], strength: float) -> List[int]:
        # strength in [0,1]; higher => more alignment with best.
        if n <= 1:
            return order[:]
        pos_in_best = {item: i for i, item in enumerate(best)}
        a = order[:]
        # number of guided moves
        m = 1 + int(strength * min(30, n))
        for _ in range(m):
            # pick an item; more likely from first part (heavier) to matter more
            if random.random() < 0.7:
                i = random.randrange(min(n, 30))
            else:
                i = random.randrange(n)
            item = a[i]
            # remove
            a.pop(i)
            # target insertion location based on best position, with noise
            target = pos_in_best[item]
            # map best position to current list length
            ins = int((target / max(1, n - 1)) * max(1, len(a) - 1))
            # add noise depending on (1-strength)
            noise = int((1.0 - strength) * 8)
            if noise > 0:
                ins += random.randint(-noise, noise)
            if ins < 0:
                ins = 0
            if ins > len(a):
                ins = len(a)
            a.insert(ins, item)
        return a

    # Move 2: small random mutation
    def mutate(order: List[int], rate: float) -> List[int]:
        a = order[:]
        # expected swaps proportional to rate
        swaps = 1 if rate <= 0.25 else 2 if rate <= 0.6 else 3
        for _ in range(swaps):
            r = random.random()
            if r < 0.7:
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

    # Convert fitness to a "pace" (strength toward best). Worse -> larger pace.
    def pace_from(f: Tuple[int, int], bestf: Tuple[int, int], worst_bins: int) -> float:
        # bins difference dominates
        db = f[0] - bestf[0]
        if db <= 0:
            base = 0.10
        else:
            base = min(1.0, 0.20 + 0.20 * db)
        # add slight dependence on waste among same bin-count
        if f[0] == bestf[0] and f[1] > bestf[1]:
            base = min(1.0, base + 0.10)
        # scale with how bad relative to worst_bins to stabilize for large n
        if worst_bins > 0:
            base = min(1.0, base * (1.0 + 0.5 * (f[0] / worst_bins)))
        return max(0.05, min(1.0, base))

    # ---------- Main loop ----------
    # Fixed iteration budget, but will stop early on time limit.
    # Keep it proportional to n but capped.
    iter_budget = 300 + 25 * min(60, n // 10)
    iter_budget = min(iter_budget, 3000)

    # periodic time checks
    check_every = 10

    for it in range(iter_budget):
        if it % check_every == 0 and (time.time() - start) >= time_limit:
            break

        # Determine worst bins currently to normalize pace.
        worst_bins = max(f[0] for f in pop_fit)

        # Elitism: keep best_order as one of scouts
        # Update each scout
        for s in range(pop_size):
            if (time.time() - start) >= time_limit:
                break

            cur = pop_orders[s]
            cur_fit = pop_fit[s]

            # If this scout is the best itself, do a small exploration mutation
            if cur_fit == best_fit and random.random() < 0.6:
                new_order = mutate(cur, rate=0.3)
            else:
                pace = pace_from(cur_fit, best_fit, worst_bins)
                # Fitness-dependent strength: move towards best with prob=pace
                if random.random() < pace:
                    new_order = guided_insertion(cur, best_order, strength=pace)
                    # add slight mutation inversely proportional to pace
                    if random.random() < (0.25 + 0.5 * (1.0 - pace)):
                        new_order = mutate(new_order, rate=0.2 + 0.5 * (1.0 - pace))
                else:
                    # random walk
                    new_order = mutate(cur, rate=0.4)

            bins, bw = decode_best_fit(new_order)
            f = fitness_from(bins, bw)

            # Accept if better or with small probability if equal bins (diversity)
            accept = False
            if better(f, cur_fit):
                accept = True
            elif f[0] == cur_fit[0] and f[1] == cur_fit[1] and random.random() < 0.05:
                accept = True
            elif f[0] == cur_fit[0] and f[1] > cur_fit[1] and random.random() < 0.02:
                accept = True

            if accept:
                pop_orders[s] = new_order
                pop_fit[s] = f
                cur_fit = f

            if better(cur_fit, best_fit):
                best_fit = cur_fit
                best_packing, best_bw = bins, bw
                best_order = pop_orders[s][:]

        # occasional re-diversification of worst scouts
        if it % 40 == 39 and pop_size >= 6:
            # identify a few worst by fitness
            idxs = list(range(pop_size))
            idxs.sort(key=lambda i: (pop_fit[i][0], pop_fit[i][1]), reverse=True)
            k = max(1, pop_size // 6)
            for j in range(k):
                if (time.time() - start) >= time_limit:
                    break
                i = idxs[j]
                # restart around best with heavy mutation
                o = random_perturb(best_order)
                if random.random() < 0.3:
                    # sometimes shuffle tail to create different structure
                    cut = min(n, 20)
                    head = o[:cut]
                    tail = o[cut:]
                    random.shuffle(tail)
                    o = head + tail
                bins, bw = decode_best_fit(o)
                f = fitness_from(bins, bw)
                pop_orders[i] = o
                pop_fit[i] = f
                if better(f, best_fit):
                    best_fit = f
                    best_packing, best_bw = bins, bw
                    best_order = o[:]

    return {"packing": best_packing, "bin_weights": best_bw}
