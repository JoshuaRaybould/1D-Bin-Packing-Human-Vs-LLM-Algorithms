import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    start = time.time()
    # Respect time_limit even if it is very small/zero.
    deadline = start + max(0.0, float(time_limit))

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    w = list(map(int, weights))

    # Lower bound (can early terminate if achieved)
    total_w = sum(w)
    lb = (total_w + C - 1) // C if C > 0 else n

    def time_up() -> bool:
        return time.time() >= deadline

    # --- Decoding: priority -> order -> best-fit packing ---
    def best_fit_from_order(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        packing: List[List[int]] = []
        bin_w: List[int] = []
        for i in order:
            wi = w[i]
            best_j = -1
            best_residual = None
            for j, bw in enumerate(bin_w):
                nbw = bw + wi
                if nbw <= C:
                    residual = C - nbw
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
        # Avoid building tuples repeatedly in the key: use (priority, weight)
        order.sort(key=lambda i: (priority[i], w[i]), reverse=True)
        return best_fit_from_order(order)

    # --- Fitness: primary bins, secondary unused space (lower is better) ---
    def fitness(bin_w: List[int]) -> float:
        bins = len(bin_w)
        if bins == 0:
            return 0.0
        unused = sum(C - bw for bw in bin_w)
        return bins + (unused / (C * bins)) * 1e-3

    # --- FDO parameters ---
    if n <= 60:
        pop_size = 25
        max_iters = 1200
    elif n <= 200:
        pop_size = 30
        max_iters = 1800
    else:
        pop_size = 35
        max_iters = 2200

    base_alpha = 0.8
    min_alpha = 0.05

    # Initialize population as random priorities in [0,1)
    pop = [[random.random() for _ in range(n)] for _ in range(pop_size)]

    best_priority = None
    best_pack = None
    best_bw = None
    best_fit = float("inf")

    fits = [0.0] * pop_size

    # Evaluate initial population with time checks
    for p in range(pop_size):
        if time_up():
            break
        pack, bw = decode(pop[p])
        f = fitness(bw)
        fits[p] = f
        if f < best_fit:
            best_fit = f
            best_priority = pop[p][:]
            best_pack, best_bw = pack, bw
            if len(best_bw) == lb:
                return {"packing": best_pack, "bin_weights": best_bw}

    # If we ran out of time during initialization
    if best_pack is None:
        return {"packing": [], "bin_weights": []}

    def clip01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x >= 1.0:
            return 0.999999
        return x

    # Check time frequently by counting evaluations (cheap)
    eval_counter = 0
    eval_check_period = 5  # check time every few candidate evaluations

    for it in range(max_iters):
        if time_up():
            break

        fmin = min(fits)
        fmax = max(fits)
        denom = (fmax - fmin) if (fmax > fmin) else 1.0

        t = it / max(1, max_iters - 1)
        alpha = max(min_alpha, base_alpha * (1.0 - 0.6 * t))

        for i in range(pop_size):
            eval_counter += 1
            if eval_counter % eval_check_period == 0 and time_up():
                break

            xi = pop[i]
            fi = fits[i]

            fw = (fi - fmin) / denom  # 0 (best) .. 1 (worst)
            r = random.random() * 2.0 - 1.0  # [-1,1]
            noise_scale = 0.02 + 0.10 * fw

            food = best_priority
            newx = xi[:]

            if n <= 120:
                dims = range(n)
            else:
                k = max(20, int(0.4 * n))
                dims = random.sample(range(n), k)

            for d in dims:
                step = alpha * r * fw * (food[d] - xi[d])
                step += (random.random() * 2.0 - 1.0) * noise_scale
                newx[d] = clip01(newx[d] + step)

            if fw > 0.85 and random.random() < 0.05:
                for d in dims:
                    newx[d] = random.random()

            pack, bw = decode(newx)
            fnew = fitness(bw)

            if fnew <= fi:
                pop[i] = newx
                fits[i] = fnew
                if fnew < best_fit:
                    best_fit = fnew
                    best_priority = newx[:]
                    best_pack, best_bw = pack, bw
                    if len(best_bw) == lb:
                        return {"packing": best_pack, "bin_weights": best_bw}

        # Diversification step, with time checks inside
        if (it + 1) % 200 == 0 and pop_size >= 6 and not time_up():
            for _ in range(2):
                if time_up():
                    break
                j = random.randrange(pop_size)
                xj = pop[j][:]
                mix = 0.5 + 0.4 * random.random()
                if n <= 80:
                    dims2 = range(n)
                else:
                    dims2 = random.sample(range(n), max(10, n // 5))
                for d in dims2:
                    xj[d] = clip01(mix * best_priority[d] + (1.0 - mix) * random.random())

                pack, bw = decode(xj)
                f = fitness(bw)
                pop[j] = xj
                fits[j] = f
                if f < best_fit:
                    best_fit = f
                    best_priority = xj[:]
                    best_pack, best_bw = pack, bw
                    if len(best_bw) == lb:
                        return {"packing": best_pack, "bin_weights": best_bw}

    return {"packing": best_pack if best_pack is not None else [],
            "bin_weights": best_bw if best_bw is not None else []}
