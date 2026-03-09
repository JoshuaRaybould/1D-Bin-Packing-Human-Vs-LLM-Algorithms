import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    start = time.perf_counter()
    deadline = start + max(0.0, time_limit)

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = bin_capacity

    # ---- Decoding: priority vector -> feasible packing (Best-Fit with induced order) ----
    # Uses buckets of bins keyed by remaining capacity to implement near O(n*C) best-fit.
    # Also checks time periodically to respect deadline.
    def decode(priorities: List[float], best_bins_so_far, best_binw_so_far) -> Tuple[List[List[int]], List[int], bool]:
        # Return (bins, bin_w, timed_out)
        order = list(range(n))
        order.sort(key=lambda i: (priorities[i], weights[i]), reverse=True)

        bins: List[List[int]] = []
        bin_w: List[int] = []

        # buckets[r] = list of bin indices whose remaining capacity == r
        # remaining is integer in [0..C]
        buckets: List[List[int]] = [[] for _ in range(C + 1)]

        # Check cadence inside decode (dominant cost)
        check_mask = 63  # check every 64 items

        for t, i in enumerate(order):
            if (t & check_mask) == 0:
                if time.perf_counter() >= deadline:
                    # Timed out: return best-so-far passed in
                    return best_bins_so_far, best_binw_so_far, True

            w = weights[i]
            # Find smallest remaining capacity r >= w
            # This yields minimal slack (best-fit).
            chosen = -1
            for r in range(w, C + 1):
                if buckets[r]:
                    chosen = buckets[r].pop()
                    break

            if chosen == -1:
                # open new bin
                j = len(bins)
                bins.append([i])
                bw = w
                bin_w.append(bw)
                rem = C - bw
                buckets[rem].append(j)
            else:
                bins[chosen].append(i)
                bw = bin_w[chosen] + w
                bin_w[chosen] = bw
                rem = C - bw
                buckets[rem].append(chosen)

        return bins, bin_w, False

    # ---- Fitness: minimize bins, then minimize total slack ----
    def fitness(bin_w: List[int]) -> Tuple[int, int]:
        k = len(bin_w)
        slack = sum(C - bw for bw in bin_w)
        return (k, slack)

    maxw = max(weights)

    def seeded_priorities(noise_scale: float) -> List[float]:
        return [
            (weights[i] / maxw) + random.uniform(-noise_scale, noise_scale)
            for i in range(n)
        ]

    # ---- FDO parameters ----
    pop_size = max(12, min(50, 10 + int(n ** 0.5) * 4))

    # Use a fixed iteration cap, but time limit is authoritative.
    max_iters = 600

    lo, hi = -1.0, 1.0
    wf_min, wf_max = 0.0, 1.0

    scouts: List[List[float]] = []
    scout_fit: List[Tuple[int, int]] = []
    scout_bins: List[List[List[int]]] = []
    scout_binw: List[List[int]] = []

    # Initialize with mix of random and seeded
    best_pos: Optional[List[float]] = None
    best_fit: Optional[Tuple[int, int]] = None
    best_bins: List[List[int]] = []
    best_binw: List[int] = []

    for p in range(pop_size):
        if time.perf_counter() >= deadline:
            break

        if p < pop_size // 3:
            pos = [random.uniform(lo, hi) for _ in range(n)]
        else:
            noise = 0.35 if p < (2 * pop_size) // 3 else 0.15
            pos = seeded_priorities(noise)
            pos = [min(hi, max(lo, x)) for x in pos]

        # During initialization, if timeout occurs, return best found so far.
        bins, binw, timed_out = decode(pos, best_bins, best_binw)
        if timed_out:
            return {"packing": best_bins, "bin_weights": best_binw}

        fit = fitness(binw)

        scouts.append(pos)
        scout_fit.append(fit)
        scout_bins.append(bins)
        scout_binw.append(binw)

        if best_fit is None or fit < best_fit:
            best_fit = fit
            best_pos = pos[:]
            best_bins = [b[:] for b in bins]
            best_binw = binw[:]

    if not scouts:
        # Nothing evaluated in time; must return empty/valid.
        return {"packing": [], "bin_weights": []}

    assert best_pos is not None and best_fit is not None

    # ---- Main FDO loop ----
    for it in range(max_iters):
        if time.perf_counter() >= deadline:
            break

        # Refresh global best from population
        best_idx = min(range(len(scouts)), key=lambda i: scout_fit[i])
        if scout_fit[best_idx] < best_fit:
            best_fit = scout_fit[best_idx]
            best_pos = scouts[best_idx][:]
            best_bins = [b[:] for b in scout_bins[best_idx]]
            best_binw = scout_binw[best_idx][:]

        best_scalar = best_fit[0] + (best_fit[1] / (C * n + 1.0))

        # Iterate through population; check time frequently
        for i in range(len(scouts)):
            if time.perf_counter() >= deadline:
                break

            pos = scouts[i]
            fit_i = scout_fit[i]
            scalar_i = fit_i[0] + (fit_i[1] / (C * n + 1.0))

            if scalar_i <= best_scalar:
                wf = wf_min
            else:
                gap = (scalar_i - best_scalar) / (best_scalar + 1e-9)
                wf = min(wf_max, max(wf_min, gap))

            r = random.random()
            noise_amp = 0.15 * (1.0 - it / max_iters)

            new_pos = [0.0] * n

            # Subsample dimensions for big n
            idxset = None
            if n > 200:
                upd = max(50, n // 5)
                idxset = set(random.sample(range(n), upd))

            for d in range(n):
                x = pos[d]
                if idxset is not None and d not in idxset:
                    new_pos[d] = x
                    continue
                step = r * wf * (best_pos[d] - x)
                noise = random.uniform(-noise_amp, noise_amp)
                y = x + step + noise
                if y < lo:
                    y = lo
                elif y > hi:
                    y = hi
                new_pos[d] = y

            bins, binw, timed_out = decode(new_pos, best_bins, best_binw)
            if timed_out:
                return {"packing": best_bins, "bin_weights": best_binw}

            fit_new = fitness(binw)

            if fit_new <= fit_i:
                scouts[i] = new_pos
                scout_fit[i] = fit_new
                scout_bins[i] = bins
                scout_binw[i] = binw

                if fit_new < best_fit:
                    best_fit = fit_new
                    best_pos = new_pos[:]
                    best_bins = [b[:] for b in bins]
                    best_binw = binw[:]

    return {"packing": best_bins, "bin_weights": best_binw}
