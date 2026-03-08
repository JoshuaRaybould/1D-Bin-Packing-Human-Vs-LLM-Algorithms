import time
import random
from typing import List, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()
    deadline = start + max(0.0, time_limit)

    n = len(weights)
    C = bin_capacity

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ----------------- Decoder: order -> bins via Best-Fit -----------------
    # Return None on timeout instead of raising.
    def decode_from_order(order: List[int]) -> Optional[Tuple[List[List[int]], List[int]]]:
        bins: List[List[int]] = []
        loads: List[int] = []

        for t, i in enumerate(order):
            if (t & 255) == 0 and time.perf_counter() >= deadline:
                return None

            w = weights[i]
            best_j = -1
            best_rem_after = C + 1
            for j, load in enumerate(loads):
                rem = C - load
                if w <= rem:
                    rem_after = rem - w
                    if rem_after < best_rem_after:
                        best_rem_after = rem_after
                        best_j = j
                        if best_rem_after == 0:
                            break
            if best_j == -1:
                bins.append([i])
                loads.append(w)
            else:
                bins[best_j].append(i)
                loads[best_j] += w
        return bins, loads

    def decode_keys(keys: List[float]) -> Optional[Tuple[List[List[int]], List[int]]]:
        if time.perf_counter() >= deadline:
            return None
        order = sorted(range(n), key=lambda i: keys[i])
        return decode_from_order(order)

    # ----------------- Fitness -----------------
    BIG = (n + 1) * C + 1

    def fitness_of(keys: List[float]) -> Optional[Tuple[int, int, int]]:
        dec = decode_keys(keys)
        if dec is None:
            return None
        _, loads = dec
        b = len(loads)
        waste = b * C - sum(loads)
        return b * BIG + waste, b, waste

    # ----------------- Heuristic seeds (orders -> keys) -----------------
    def keys_from_order(order: List[int]) -> List[float]:
        keys = [0.0] * n
        for pos, idx in enumerate(order):
            keys[idx] = pos + random.random() * 1e-6
        return keys

    idxs = list(range(n))
    order_desc = sorted(idxs, key=lambda i: (-weights[i], i))
    order_asc = sorted(idxs, key=lambda i: (weights[i], i))

    def noisy_weight_order(scale: float = 0.35) -> List[int]:
        return sorted(
            idxs,
            key=lambda i: -(weights[i] * (1.0 + scale * (random.random() - 0.5))),
        )

    # ----------------- GA parameters -----------------
    pop_size = max(30, min(120, 20 + int(2.5 * (n ** 0.5))))
    elite = max(2, pop_size // 12)
    tour_k = 3 if pop_size < 60 else 4

    mut_rate = 0.10 if n < 200 else 0.06
    mut_strength = 0.15
    crossover_rate = 0.90

    max_gens = 2000 if n <= 200 else 1200

    # ----------------- Population initialization -----------------
    population: List[List[float]] = []
    population.append(keys_from_order(order_desc))
    population.append(keys_from_order(order_asc))
    for _ in range(min(6, pop_size // 6)):
        population.append(keys_from_order(noisy_weight_order()))

    while len(population) < pop_size:
        population.append([random.random() for _ in range(n)])

    # Always keep a valid incumbent packing available from the start.
    # Use FFD seed as fallback incumbent (no time checks here besides decoder loop).
    init_dec = decode_from_order(order_desc)
    if init_dec is None:
        # Extremely tiny time_limit: return trivial singletons quickly.
        packing = [[i] for i in order_desc]
        return {"packing": packing, "bin_weights": [weights[i] for i in order_desc]}

    best_pack, best_loads = init_dec
    best_keys = population[0][:]
    best_score = (len(best_loads) * BIG + (len(best_loads) * C - sum(best_loads)), len(best_loads), len(best_loads) * C - sum(best_loads))

    scores: List[Tuple[int, int, int]] = [(10**18, 10**9, 10**9)] * pop_size

    def evaluate_all() -> bool:
        nonlocal best_keys, best_score, best_pack, best_loads
        for i in range(pop_size):
            if time.perf_counter() >= deadline:
                return False
            fit = fitness_of(population[i])
            if fit is None:
                return False
            scores[i] = fit
            if fit[0] < best_score[0]:
                best_score = fit
                best_keys = population[i][:]
                dec = decode_keys(best_keys)
                if dec is None:
                    return False
                best_pack, best_loads = dec
        return True

    if not evaluate_all():
        return {"packing": best_pack, "bin_weights": best_loads}

    def best_index() -> int:
        best = 0
        for i in range(1, pop_size):
            if scores[i][0] < scores[best][0]:
                best = i
        return best

    # ----------------- Selection -----------------
    def tournament_select() -> int:
        cand = random.randrange(pop_size)
        for _ in range(tour_k - 1):
            j = random.randrange(pop_size)
            if scores[j][0] < scores[cand][0]:
                cand = j
        return cand

    # ----------------- Variation operators -----------------
    def crossover(p1: List[float], p2: List[float]) -> List[float]:
        child = [0.0] * n
        if random.random() < 0.2:
            a = random.random()
            for i in range(n):
                child[i] = a * p1[i] + (1.0 - a) * p2[i]
        else:
            for i in range(n):
                child[i] = p1[i] if random.random() < 0.5 else p2[i]
        return child

    def mutate(ind: List[float]) -> None:
        # Segment perturbation (acts like partial re-ranking in random-keys space)
        if random.random() < 0.08 and n >= 6:
            a = random.randrange(n)
            b = random.randrange(n)
            if a > b:
                a, b = b, a
            if b - a >= 2:
                base = random.random() * n
                span = (b - a + 1)
                for i in range(a, b + 1):
                    ind[i] = base + random.random() * span

        for i in range(n):
            if random.random() < mut_rate:
                ind[i] += (random.random() * 2.0 - 1.0) * mut_strength

    # ----------------- Main GA loop -----------------
    for gen in range(max_gens):
        if time.perf_counter() >= deadline:
            break

        elite_idx = sorted(range(pop_size), key=lambda i: scores[i][0])[:elite]
        new_pop = [population[i][:] for i in elite_idx]

        while len(new_pop) < pop_size:
            if time.perf_counter() >= deadline:
                break

            if random.random() < crossover_rate:
                i1 = tournament_select()
                i2 = tournament_select()
                if i2 == i1 and pop_size > 1:
                    i2 = (i2 + random.randrange(1, pop_size)) % pop_size
                child = crossover(population[i1], population[i2])
            else:
                child = population[tournament_select()][:]

            mutate(child)
            new_pop.append(child)

        population = new_pop
        if time.perf_counter() >= deadline:
            break

        # Evaluate new population
        scores = [(10**18, 10**9, 10**9)] * pop_size
        ok = True
        for i in range(pop_size):
            if time.perf_counter() >= deadline:
                ok = False
                break
            fit = fitness_of(population[i])
            if fit is None:
                ok = False
                break
            scores[i] = fit
            if fit[0] < best_score[0]:
                best_score = fit
                best_keys = population[i][:]
                dec = decode_keys(best_keys)
                if dec is None:
                    ok = False
                    break
                best_pack, best_loads = dec

        if not ok:
            break

        # Occasional materialization of best in current generation (safe)
        bi = best_index()
        if scores[bi][0] < best_score[0]:
            best_score = scores[bi]
            best_keys = population[bi][:]
            dec = decode_keys(best_keys)
            if dec is None:
                break
            best_pack, best_loads = dec

        # Diversification
        if gen > 0 and (gen % 200) == 0:
            if time.perf_counter() >= deadline:
                break
            current_best = min(scores, key=lambda s: s[0])[0]
            if current_best >= best_score[0]:
                worst_count = max(2, pop_size // 10)
                worst_idx = sorted(range(pop_size), key=lambda i: scores[i][0], reverse=True)[:worst_count]
                for wi in worst_idx:
                    if wi in elite_idx:
                        continue
                    if random.random() < 0.5:
                        population[wi] = [random.random() for _ in range(n)]
                    else:
                        population[wi] = keys_from_order(noisy_weight_order())

                # Re-evaluate injected individuals
                for wi in worst_idx:
                    if time.perf_counter() >= deadline:
                        break
                    fit = fitness_of(population[wi])
                    if fit is None:
                        break
                    scores[wi] = fit
                    if fit[0] < best_score[0]:
                        best_score = fit
                        best_keys = population[wi][:]
                        dec = decode_keys(best_keys)
                        if dec is None:
                            break
                        best_pack, best_loads = dec

    # Return best-so-far (always valid)
    return {"packing": best_pack, "bin_weights": best_loads}
