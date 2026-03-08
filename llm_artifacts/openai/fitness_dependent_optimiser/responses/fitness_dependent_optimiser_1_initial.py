import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Quick feasibility / normalization
    C = int(bin_capacity)
    w = list(map(int, weights))

    # --- Decoding: priority -> order -> best-fit packing ---
    def best_fit_from_order(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        packing: List[List[int]] = []
        bin_w: List[int] = []
        for i in order:
            wi = w[i]
            # place in bin with minimum residual after placement (best fit)
            best_j = -1
            best_residual = None
            for j, bw in enumerate(bin_w):
                if bw + wi <= C:
                    residual = C - (bw + wi)
                    if best_residual is None or residual < best_residual:
                        best_residual = residual
                        best_j = j
                        if residual == 0:
                            break
            if best_j == -1:
                packing.append([i])
                bin_w.append(wi)
            else:
                packing[best_j].append(i)
                bin_w[best_j] += wi
        return packing, bin_w

    def decode(priority: List[float]) -> Tuple[List[List[int]], List[int]]:
        # stable sort by priority descending; tie-break by heavier items first
        order = list(range(n))
        order.sort(key=lambda i: (priority[i], w[i]), reverse=True)
        return best_fit_from_order(order)

    # --- Fitness: primary bins, secondary unused space (lower is better) ---
    def fitness(bin_w: List[int]) -> float:
        bins = len(bin_w)
        unused = sum(C - bw for bw in bin_w)
        # Primary objective dominates; secondary just breaks ties.
        return bins + (unused / (C * max(1, bins))) * 1e-3

    # --- FDO parameters (tuned for bin packing decoding) ---
    # Population size: keep moderate for speed.
    if n <= 60:
        pop_size = 25
        max_iters = 1200
    elif n <= 200:
        pop_size = 30
        max_iters = 1800
    else:
        pop_size = 35
        max_iters = 2200

    # Ensure fixed iteration count; time limit can cut it short.
    # FDO step control
    base_alpha = 0.8  # exploration weight
    min_alpha = 0.05

    # Initialize population as random priorities in [0,1)
    pop = [[random.random() for _ in range(n)] for _ in range(pop_size)]

    # Evaluate initial population
    best_priority = None
    best_pack = None
    best_bw = None
    best_fit = float("inf")

    fits = [0.0] * pop_size
    for p in range(pop_size):
        pack, bw = decode(pop[p])
        f = fitness(bw)
        fits[p] = f
        if f < best_fit:
            best_fit = f
            best_priority = pop[p][:]
            best_pack, best_bw = pack, bw

    # Helper to clip priorities into [0,1)
    def clip01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x >= 1.0:
            # keep below 1 to avoid edge ties
            return 0.999999
        return x

    # --- Main FDO loop ---
    # Fitness-dependent weight factor: smaller step near best fitness
    # Use population min/max each iteration to scale.
    check_every = 10

    for it in range(max_iters):
        if it % check_every == 0 and (time.time() - start) >= time_limit:
            break

        fmin = min(fits)
        fmax = max(fits)
        denom = (fmax - fmin) if (fmax > fmin) else 1.0

        # Decrease alpha slightly over time (more exploitation later)
        t = it / max(1, max_iters - 1)
        alpha = max(min_alpha, base_alpha * (1.0 - 0.6 * t))

        # Update each agent
        for i in range(pop_size):
            # Another time check inside the inner loop for large instances
            if (i == 0) and (it % check_every == 0) and (time.time() - start) >= time_limit:
                break

            xi = pop[i]
            fi = fits[i]

            # Fitness weight: 0 (best) .. 1 (worst)
            fw = (fi - fmin) / denom

            # Random perturbation direction around best (food)
            # FDO-like update: x_new = x + alpha * r * fw * (food - x) + noise
            r = random.random() * 2.0 - 1.0  # [-1,1]

            # Small dimension-wise noise to avoid premature convergence
            # Noise magnitude depends on fw (worse solutions explore more)
            noise_scale = 0.02 + 0.10 * fw

            newx = xi[:]  # copy
            food = best_priority  # global best

            # Update a subset of dimensions for speed on large n
            if n <= 120:
                dims = range(n)
            else:
                # update ~40% dims
                k = max(20, int(0.4 * n))
                dims = random.sample(range(n), k)

            for d in dims:
                step = alpha * r * fw * (food[d] - xi[d])
                step += (random.random() * 2.0 - 1.0) * noise_scale
                newx[d] = clip01(newx[d] + step)

            # Occasionally do a "restart" for very poor agents
            if fw > 0.85 and random.random() < 0.05:
                for d in dims:
                    newx[d] = random.random()

            # Evaluate and accept if improved (elitist)
            pack, bw = decode(newx)
            fnew = fitness(bw)
            if fnew <= fi:
                pop[i] = newx
                fits[i] = fnew
                if fnew < best_fit:
                    best_fit = fnew
                    best_priority = newx[:]
                    best_pack, best_bw = pack, bw

        # Light diversification: shuffle a couple of agents around best
        # (Still within FDO population-based search; no separate local search.)
        if (it + 1) % 200 == 0 and pop_size >= 6:
            for _ in range(2):
                j = random.randrange(pop_size)
                if best_priority is None:
                    continue
                xj = pop[j][:]
                # mix with best
                mix = 0.5 + 0.4 * random.random()
                for d in (range(n) if n <= 80 else random.sample(range(n), max(10, n // 5))):
                    xj[d] = clip01(mix * best_priority[d] + (1.0 - mix) * random.random())
                pack, bw = decode(xj)
                f = fitness(bw)
                pop[j] = xj
                fits[j] = f
                if f < best_fit:
                    best_fit = f
                    best_priority = xj[:]
                    best_pack, best_bw = pack, bw

    # Return best found
    # Ensure outputs are aligned
    return {
        "packing": best_pack if best_pack is not None else [],
        "bin_weights": best_bw if best_bw is not None else [],
    }
