import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    start = time.perf_counter()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---- Decoding: priority vector -> feasible packing (Best-Fit with induced order) ----
    def decode(priorities: List[float]) -> Tuple[List[List[int]], List[int]]:
        # Order items by priority (desc), then by weight (desc) to stabilize ties
        order = list(range(n))
        order.sort(key=lambda i: (priorities[i], weights[i]), reverse=True)

        bins: List[List[int]] = []
        bin_w: List[int] = []

        for i in order:
            w = weights[i]
            # Best-fit: choose bin with minimum remaining capacity after placing (>=0)
            best_j = -1
            best_rem = bin_capacity + 1
            for j, bw in enumerate(bin_w):
                rem = bin_capacity - (bw + w)
                if rem >= 0 and rem < best_rem:
                    best_rem = rem
                    best_j = j
            if best_j == -1:
                bins.append([i])
                bin_w.append(w)
            else:
                bins[best_j].append(i)
                bin_w[best_j] += w

        return bins, bin_w

    # ---- Fitness: minimize bins, then minimize total slack ----
    def fitness(bin_w: List[int]) -> Tuple[int, int]:
        k = len(bin_w)
        slack = sum(bin_capacity - bw for bw in bin_w)
        return (k, slack)

    # ---- Quick initial heuristic seed: weight-based priorities (gives BFD-like) ----
    maxw = max(weights)
    def seeded_priorities(noise_scale: float) -> List[float]:
        # Higher weight => higher base priority; add noise for diversity
        return [
            (weights[i] / maxw) + random.uniform(-noise_scale, noise_scale)
            for i in range(n)
        ]

    # ---- FDO parameters ----
    # Population size: moderate, scale mildly with n
    pop_size = max(12, min(50, 10 + int(n ** 0.5) * 4))

    # Iterations: fixed; time limit will cut short if needed
    # Chosen to provide meaningful search while staying safe on large instances.
    max_iters = 600

    # Position bounds for priorities
    lo, hi = -1.0, 1.0

    # Fitness-dependent weight factor range
    wf_min, wf_max = 0.0, 1.0

    # ---- Initialize population ----
    scouts: List[List[float]] = []
    scout_fit: List[Tuple[int, int]] = []
    scout_bins: List[List[List[int]]] = []
    scout_binw: List[List[int]] = []

    # Mix of: pure random, seeded (BFD-like) with varying noise
    for p in range(pop_size):
        if p < pop_size // 3:
            pos = [random.uniform(lo, hi) for _ in range(n)]
        else:
            noise = 0.35 if p < (2 * pop_size) // 3 else 0.15
            pos = seeded_priorities(noise)
            # clamp
            pos = [min(hi, max(lo, x)) for x in pos]

        bins, binw = decode(pos)
        fit = fitness(binw)
        scouts.append(pos)
        scout_fit.append(fit)
        scout_bins.append(bins)
        scout_binw.append(binw)

    # Best-so-far
    best_idx = min(range(pop_size), key=lambda i: scout_fit[i])
    best_pos = scouts[best_idx][:]
    best_fit = scout_fit[best_idx]
    best_bins = [b[:] for b in scout_bins[best_idx]]
    best_binw = scout_binw[best_idx][:]

    # ---- Main FDO loop ----
    # Time check cadence
    check_every = 10

    for it in range(max_iters):
        if it % check_every == 0:
            if time.perf_counter() - start >= time_limit:
                break

        # Recompute global best index (in case improved)
        best_idx = min(range(pop_size), key=lambda i: scout_fit[i])
        if scout_fit[best_idx] < best_fit:
            best_fit = scout_fit[best_idx]
            best_pos = scouts[best_idx][:]
            best_bins = [b[:] for b in scout_bins[best_idx]]
            best_binw = scout_binw[best_idx][:]

        # Convert lexicographic fitness to a scalar for WF computation
        # Primary bins dominates; slack provides tie-breaking.
        # Scale slack by capacity*n to keep bins dominant.
        best_scalar = best_fit[0] + (best_fit[1] / (bin_capacity * n + 1.0))

        for i in range(pop_size):
            pos = scouts[i]
            fit_i = scout_fit[i]
            scalar_i = fit_i[0] + (fit_i[1] / (bin_capacity * n + 1.0))

            # Fitness-dependent weight factor (better individuals move less)
            # wf in [0,1], higher when worse than best.
            if scalar_i <= best_scalar:
                wf = wf_min
            else:
                # Normalize by relative gap; keep bounded
                gap = (scalar_i - best_scalar) / (best_scalar + 1e-9)
                wf = min(wf_max, max(wf_min, gap))

            # FDO update towards best with stochastic perturbation
            # new = x + r * wf * (best - x) + noise
            r = random.random()
            # Decrease noise over time
            noise_amp = 0.15 * (1.0 - it / max_iters)

            new_pos = [0.0] * n
            # To reduce overhead, reuse random calls sparsely
            # and allow occasional dimension sub-sampling.
            if n > 200:
                # Update a subset of dimensions; keep others
                # This retains feasibility while lowering per-iter cost.
                upd = max(50, n // 5)
                idxs = random.sample(range(n), upd)
                idxset = set(idxs)
            else:
                idxset = None

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

            # Evaluate candidate
            bins, binw = decode(new_pos)
            fit_new = fitness(binw)

            # Greedy selection
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

    # Ensure return bins aligned with weights
    return {"packing": best_bins, "bin_weights": best_binw}
