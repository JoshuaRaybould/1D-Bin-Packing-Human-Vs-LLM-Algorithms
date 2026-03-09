import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()
    n = len(weights)
    C = bin_capacity

    # Trivial cases
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ----- Decoder: order -> bins via Best-Fit -----
    def decode_from_order(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []

        for i in order:
            w = weights[i]
            # best-fit: choose bin with minimum remaining capacity that can fit
            best_j = -1
            best_rem = C + 1
            for j, load in enumerate(loads):
                rem = C - load
                if w <= rem:
                    if rem - w < best_rem:
                        best_rem = rem - w
                        best_j = j
                        if best_rem == 0:
                            break
            if best_j == -1:
                bins.append([i])
                loads.append(w)
            else:
                bins[best_j].append(i)
                loads[best_j] += w
        return bins, loads

    def decode_keys(keys: List[float]) -> Tuple[List[List[int]], List[int]]:
        order = sorted(range(n), key=lambda i: keys[i])
        return decode_from_order(order)

    # ----- Fitness -----
    # Upper bound on waste: at most bins*C. bins <= n.
    BIG = (n + 1) * C + 1

    def fitness_of(keys: List[float]) -> Tuple[int, int, int]:
        # returns (scalar, bins, waste)
        packing, loads = decode_keys(keys)
        b = len(loads)
        waste = b * C - sum(loads)
        return b * BIG + waste, b, waste

    # ----- Heuristic seeds converted to key representation -----
    def keys_from_order(order: List[int]) -> List[float]:
        # Assign increasing keys following the order; add tiny jitter to break ties
        keys = [0.0] * n
        for pos, idx in enumerate(order):
            keys[idx] = pos + random.random() * 1e-6
        return keys

    idxs = list(range(n))
    # Common good orderings
    order_desc = sorted(idxs, key=lambda i: (-weights[i], i))
    order_asc = sorted(idxs, key=lambda i: (weights[i], i))

    # Some randomized greedy orders: sort by weight but with noise
    def noisy_weight_order(scale: float = 0.35) -> List[int]:
        return sorted(idxs, key=lambda i: -(weights[i] * (1.0 + scale * (random.random() - 0.5))))

    # ----- GA parameters (adaptive to size) -----
    # Keep these moderate; time_limit governs stopping.
    pop_size = max(30, min(120, 20 + int(2.5 * (n ** 0.5))))
    elite = max(2, pop_size // 12)
    tour_k = 3 if pop_size < 60 else 4

    mut_rate = 0.10 if n < 200 else 0.06
    mut_strength = 0.15
    crossover_rate = 0.90

    # Fixed iteration budget (hard requirement) + time checks
    max_gens = 2000 if n <= 200 else 1200

    # ----- Population initialization -----
    population: List[List[float]] = []

    # Seeds
    population.append(keys_from_order(order_desc))  # like FFD order
    population.append(keys_from_order(order_asc))
    for _ in range(min(6, pop_size // 6)):
        population.append(keys_from_order(noisy_weight_order()))

    # Random keys
    while len(population) < pop_size:
        population.append([random.random() for _ in range(n)])

    # Evaluate
    scores: List[Tuple[int, int, int]] = [fitness_of(ind) for ind in population]

    def best_index() -> int:
        best = 0
        for i in range(1, pop_size):
            if scores[i][0] < scores[best][0]:
                best = i
        return best

    best_i = best_index()
    best_keys = population[best_i][:]
    best_score = scores[best_i]
    best_pack, best_loads = decode_keys(best_keys)

    # ----- Selection -----
    def tournament_select() -> int:
        # Minimization
        cand = random.randrange(pop_size)
        for _ in range(tour_k - 1):
            j = random.randrange(pop_size)
            if scores[j][0] < scores[cand][0]:
                cand = j
        return cand

    # ----- Variation operators -----
    def crossover(p1: List[float], p2: List[float]) -> List[float]:
        # Uniform, with occasional blend
        child = [0.0] * n
        if random.random() < 0.2:
            # arithmetic blend
            a = random.random()
            for i in range(n):
                child[i] = a * p1[i] + (1.0 - a) * p2[i]
        else:
            for i in range(n):
                child[i] = p1[i] if random.random() < 0.5 else p2[i]
        return child

    def mutate(ind: List[float]) -> None:
        # Key perturbations; keeps representation valid.
        # Occasionally apply a stronger disruption on a segment.
        if random.random() < 0.08 and n >= 6:
            # segment shuffle by resetting keys in a random segment
            a = random.randrange(n)
            b = random.randrange(n)
            if a > b:
                a, b = b, a
            if b - a >= 2:
                base = random.random() * n
                for i in range(a, b + 1):
                    ind[i] = base + random.random() * (b - a + 1)

        for i in range(n):
            if random.random() < mut_rate:
                # bounded random walk
                ind[i] += (random.random() * 2.0 - 1.0) * mut_strength

    # ----- Main GA loop -----
    # Periodic time checks
    check_every = 5

    for gen in range(max_gens):
        if gen % check_every == 0:
            if time.perf_counter() - start >= time_limit:
                break

        # Elitism: take indices of best elites
        elite_idx = sorted(range(pop_size), key=lambda i: scores[i][0])[:elite]
        new_pop = [population[i][:] for i in elite_idx]

        # Generate offspring
        while len(new_pop) < pop_size:
            if random.random() < crossover_rate:
                i1 = tournament_select()
                i2 = tournament_select()
                # Avoid identical parents sometimes
                if i2 == i1 and pop_size > 1:
                    i2 = (i2 + random.randrange(1, pop_size)) % pop_size
                child = crossover(population[i1], population[i2])
            else:
                # Clone selected
                child = population[tournament_select()][:]

            mutate(child)
            new_pop.append(child)

        population = new_pop
        scores = [fitness_of(ind) for ind in population]

        # Track best
        bi = best_index()
        if scores[bi][0] < best_score[0]:
            best_score = scores[bi]
            best_keys = population[bi][:]
            best_pack, best_loads = decode_keys(best_keys)

        # Mild diversification if stagnating: re-randomize a few worst
        # (Still within GA family: it's a population management tactic.)
        if gen > 0 and gen % 200 == 0:
            # If no improvement recently, inject randomness into worst individuals
            # Heuristic proxy: compare current best to historical; if equal, inject.
            current_best = min(scores, key=lambda s: s[0])[0]
            if current_best >= best_score[0]:
                worst_idx = sorted(range(pop_size), key=lambda i: scores[i][0], reverse=True)[: max(2, pop_size // 10)]
                for wi in worst_idx:
                    if wi in elite_idx:
                        continue
                    if random.random() < 0.5:
                        population[wi] = [random.random() for _ in range(n)]
                    else:
                        population[wi] = keys_from_order(noisy_weight_order())
                scores = [fitness_of(ind) for ind in population]
                bi = best_index()
                if scores[bi][0] < best_score[0]:
                    best_score = scores[bi]
                    best_keys = population[bi][:]
                    best_pack, best_loads = decode_keys(best_keys)

    return {"packing": best_pack, "bin_weights": best_loads}
