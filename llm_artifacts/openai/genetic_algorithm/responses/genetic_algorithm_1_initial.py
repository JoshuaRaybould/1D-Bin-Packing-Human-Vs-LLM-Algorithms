import random
import time
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---- Helpers ----
    def time_exceeded() -> bool:
        return (time.perf_counter() - start) >= time_limit

    # Precompute a strong baseline order: decreasing by weight.
    # We'll mix it with random permutations as a seed population.
    base_order = list(range(n))
    base_order.sort(key=lambda i: weights[i], reverse=True)

    def decode_perm(perm: List[int]) -> Tuple[List[List[int]], List[int]]:
        """Decode permutation into bins using Best-Fit among current bins.

        For each item in the perm order:
          - place into bin with minimum residual capacity after placement (best fit)
          - if none fits, open a new bin.

        Returns packing and bin_weights.
        """
        bins: List[List[int]] = []
        bws: List[int] = []

        # residuals maintained implicitly by bws
        for idx in perm:
            w = weights[idx]
            best_bin = -1
            best_residual = bin_capacity + 1

            # Find best fitting existing bin
            # (scan is OK; GA is time-limited and decode dominates anyway)
            for b, bw in enumerate(bws):
                rem = bin_capacity - bw
                if w <= rem:
                    res_after = rem - w
                    if res_after < best_residual:
                        best_residual = res_after
                        best_bin = b
                        if best_residual == 0:
                            break

            if best_bin == -1:
                bins.append([idx])
                bws.append(w)
            else:
                bins[best_bin].append(idx)
                bws[best_bin] += w

            # occasional time check in long decodes
            # (cheap and keeps us safe on large instances)
            if (len(bws) & 1023) == 0 and time_exceeded():
                break

        return bins, bws

    def fitness_from_bws(bws: List[int]) -> float:
        """Lower is better.

        Primary: number of bins.
        Secondary: prefer higher average fill (maximize sum of squared fills).

        We return a scalar to minimize: bins - eps * fill_score.
        """
        m = len(bws)
        # Normalize fill score to [0, m] roughly.
        fill_score = 0.0
        invC = 1.0 / bin_capacity
        for bw in bws:
            x = bw * invC
            fill_score += x * x

        # eps chosen so that 1 bin dominates any secondary differences.
        return m - 1e-3 * (fill_score)

    def evaluate(perm: List[int]) -> Tuple[float, List[List[int]], List[int]]:
        pack, bws = decode_perm(perm)
        fit = fitness_from_bws(bws)
        return fit, pack, bws

    def tournament_select(pop: List[List[int]], fits: List[float], k: int = 3) -> List[int]:
        best_i = None
        best_f = None
        for _ in range(k):
            i = random.randrange(len(pop))
            f = fits[i]
            if best_i is None or f < best_f:
                best_i = i
                best_f = f
        return pop[best_i]

    def order_crossover(p1: List[int], p2: List[int]) -> List[int]:
        """OX crossover."""
        size = len(p1)
        a = random.randrange(size)
        b = random.randrange(size)
        if a > b:
            a, b = b, a
        child = [-1] * size
        # Copy segment
        child[a:b + 1] = p1[a:b + 1]
        used = set(child[a:b + 1])
        # Fill remaining positions from p2 in order
        pos = (b + 1) % size
        for gene in p2:
            if gene in used:
                continue
            child[pos] = gene
            pos = (pos + 1) % size
        return child

    def mutate(perm: List[int], pm_swap: float, pm_ins: float) -> None:
        """In-place mutation: swap and insertion."""
        size = len(perm)
        if random.random() < pm_swap:
            i = random.randrange(size)
            j = random.randrange(size)
            perm[i], perm[j] = perm[j], perm[i]
        if random.random() < pm_ins:
            i = random.randrange(size)
            j = random.randrange(size)
            if i != j:
                gene = perm.pop(i)
                perm.insert(j, gene)

    # ---- Parameters (fixed iteration budget; time limit also enforced) ----
    # Keep parameters conservative but effective across sizes.
    # Population size scaled mildly with n.
    pop_size = max(30, min(120, 30 + n // 10))
    elite = max(1, pop_size // 10)
    tournament_k = 3
    crossover_rate = 0.9

    # Mutation rates tuned to permutation problems.
    # Slightly higher for small instances.
    pm_swap = 0.2 if n < 200 else 0.12
    pm_ins = 0.15 if n < 200 else 0.08

    # Random immigrants fraction per generation (standard GA diversity mechanism)
    immigrant_frac = 0.05

    # Fixed iteration budget (generations).
    # Also stops early if time exceeded.
    max_gens = 400 if n <= 200 else 250

    # ---- Initialize population ----
    population: List[List[int]] = []

    # Seed with a few structured permutations derived from base_order.
    population.append(base_order[:])

    # A few perturbations of base_order
    for _ in range(min(6, pop_size - 1)):
        p = base_order[:]
        # apply a handful of random swaps
        swaps = 5 if n < 200 else 10
        for _s in range(swaps):
            i = random.randrange(n)
            j = random.randrange(n)
            p[i], p[j] = p[j], p[i]
        population.append(p)

    # Fill rest with random permutations
    while len(population) < pop_size:
        p = list(range(n))
        random.shuffle(p)
        population.append(p)

    # Evaluate initial population
    fits: List[float] = []
    decoded_cache: List[Tuple[List[List[int]], List[int]]] = []
    best_fit = float("inf")
    best_pack: List[List[int]] = []
    best_bws: List[int] = []

    for perm in population:
        if time_exceeded():
            break
        f, pack, bws = evaluate(perm)
        fits.append(f)
        decoded_cache.append((pack, bws))
        if f < best_fit:
            best_fit = f
            best_pack, best_bws = pack, bws

    # If time ran out during init
    if time_exceeded():
        return {"packing": best_pack, "bin_weights": best_bws}

    # ---- GA loop ----
    for _gen in range(max_gens):
        if time_exceeded():
            break

        # Rank population by fitness
        idxs = list(range(len(population)))
        idxs.sort(key=lambda i: fits[i])

        new_pop: List[List[int]] = []

        # Elitism
        for i in idxs[:elite]:
            new_pop.append(population[i][:])

        # Immigrants (random new individuals)
        immigrants = int(pop_size * immigrant_frac)

        # Generate offspring
        while len(new_pop) < pop_size - immigrants:
            if time_exceeded():
                break
            p1 = tournament_select(population, fits, tournament_k)
            p2 = tournament_select(population, fits, tournament_k)
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]  # clone
            mutate(child, pm_swap, pm_ins)
            new_pop.append(child)

        # Add immigrants
        while len(new_pop) < pop_size:
            if time_exceeded():
                break
            p = list(range(n))
            random.shuffle(p)
            new_pop.append(p)

        # Evaluate new population
        population = new_pop
        fits = []
        decoded_cache = []

        for perm in population:
            if time_exceeded():
                break
            f, pack, bws = evaluate(perm)
            fits.append(f)
            decoded_cache.append((pack, bws))
            if f < best_fit:
                best_fit = f
                best_pack, best_bws = pack, bws

        # If time exceeded mid-evaluation, we keep best-so-far
        if time_exceeded():
            break

    return {"packing": best_pack, "bin_weights": best_bws}
