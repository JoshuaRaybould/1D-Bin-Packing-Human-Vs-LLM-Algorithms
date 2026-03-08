import time
import random
from bisect import bisect_left


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.perf_counter()
    deadline = start + max(0.0, time_limit)

    # --- Helper: decode a permutation into bins using Best-Fit (deterministic with light random tie-break) ---
    def decode_best_fit(order, jitter=False):
        # bins: list of (remaining_capacity, bin_index)
        # keep bins sorted by remaining capacity ascending so we can find tightest fit.
        rem_sorted = []
        bin_weights = []
        packing = []

        for idx in order:
            w = weights[idx]
            # Find first bin with remaining >= w (since rem_sorted is ascending)
            pos = bisect_left(rem_sorted, (w, -1))
            chosen_pos = None

            if pos < len(rem_sorted):
                # Candidate bins at positions >= pos; choose the one that yields minimal remaining after placement.
                # The best is the smallest remaining >= w => exactly rem_sorted[pos].
                chosen_pos = pos

                if jitter:
                    # With small probability, consider a few near-best candidates to diversify.
                    # We look ahead a limited window.
                    window = 3
                    end = min(len(rem_sorted), pos + window)
                    if end - pos > 1 and random.random() < 0.25:
                        chosen_pos = random.randrange(pos, end)

            if chosen_pos is None:
                # open new bin
                b = len(bin_weights)
                bin_weights.append(w)
                packing.append([idx])
                rem = bin_capacity - w
                # insert into sorted list
                ins = bisect_left(rem_sorted, (rem, b))
                rem_sorted.insert(ins, (rem, b))
            else:
                rem, b = rem_sorted.pop(chosen_pos)
                # place in bin b
                packing[b].append(idx)
                bin_weights[b] += w
                new_rem = rem - w
                # reinsert
                ins = bisect_left(rem_sorted, (new_rem, b))
                rem_sorted.insert(ins, (new_rem, b))

        return packing, bin_weights

    def fitness_from_bins(bin_weights):
        # Primary: minimize number of bins
        k = len(bin_weights)
        # Secondary: minimize total slack, and slightly penalize many low-filled bins.
        total_slack = k * bin_capacity - sum(bin_weights)
        # Encourage fuller bins: sum of squared slack (bigger if many partially empty bins)
        sq_slack = 0
        for bw in bin_weights:
            s = bin_capacity - bw
            sq_slack += s * s
        # Return tuple for lexicographic minimization
        return (k, total_slack, sq_slack)

    # --- Seed solutions: start from classic heuristics + random permutations ---
    items = list(range(n))
    # Heuristic base permutations
    by_weight_desc = sorted(items, key=lambda i: (-weights[i], i))
    by_weight_asc = sorted(items, key=lambda i: (weights[i], i))

    # --- GA parameters (robust defaults) ---
    # Keep population moderate for time-limited use.
    pop_size = 60 if n <= 500 else 40
    elite_size = max(2, pop_size // 10)
    tournament_k = 3
    mutation_rate = 0.25

    # Fixed number of generations; still stop by time.
    max_generations = 10_000  # will be time-limited anyway

    # Build initial population
    population = []

    def add_individual(order, jitter=False):
        packing, bw = decode_best_fit(order, jitter=jitter)
        fit = fitness_from_bins(bw)
        population.append((fit, order, packing, bw))

    # Add strong heuristic seeds
    add_individual(by_weight_desc, jitter=False)
    add_individual(by_weight_desc, jitter=True)
    add_individual(by_weight_asc, jitter=True)

    # Add a few partially shuffled variants of the descending order
    for _ in range(min(10, pop_size // 2)):
        order = by_weight_desc[:]
        # Do some random swaps to diversify
        swaps = 5 if n < 200 else 10
        for __ in range(swaps):
            a = random.randrange(n)
            b = random.randrange(n)
            order[a], order[b] = order[b], order[a]
        add_individual(order, jitter=True)

    # Fill the rest with random permutations
    while len(population) < pop_size:
        order = items[:]
        random.shuffle(order)
        add_individual(order, jitter=True)

    # Track best solution
    population.sort(key=lambda t: t[0])
    best_fit, best_order, best_packing, best_bw = population[0]

    # --- GA operators ---
    def tournament_select(pop):
        best = None
        for _ in range(tournament_k):
            cand = pop[random.randrange(len(pop))]
            if best is None or cand[0] < best[0]:
                best = cand
        return best

    def order_crossover(parent1, parent2):
        # OX: keep a slice from p1, preserve order of remaining from p2.
        p1 = parent1
        p2 = parent2
        if n <= 1:
            return p1[:]
        a = random.randrange(n)
        b = random.randrange(n)
        if a > b:
            a, b = b, a
        child = [None] * n
        child[a:b+1] = p1[a:b+1]
        in_slice = set(child[a:b+1])
        fill_pos = (b + 1) % n
        for gene in p2:
            if gene in in_slice:
                continue
            child[fill_pos] = gene
            fill_pos = (fill_pos + 1) % n
        return child

    def mutate_swap(order):
        # Swap mutation; sometimes do multiple swaps.
        o = order[:]
        if n <= 1:
            return o
        swaps = 1
        if random.random() < 0.20:
            swaps = 2
        if random.random() < 0.10:
            swaps = 3
        for _ in range(swaps):
            i = random.randrange(n)
            j = random.randrange(n)
            o[i], o[j] = o[j], o[i]
        return o

    def maybe_time_up():
        return time.perf_counter() >= deadline

    # --- Main GA loop ---
    generation = 0
    while generation < max_generations:
        generation += 1
        if (generation & 31) == 0 and maybe_time_up():
            break

        # Sort and keep elites
        population.sort(key=lambda t: t[0])
        if population[0][0] < best_fit:
            best_fit, best_order, best_packing, best_bw = population[0]

        elites = population[:elite_size]

        # Create next generation
        next_pop = elites[:]  # copy tuples

        # Adaptive mutation: if stagnant on bins, increase mutation slightly
        # (still GA-consistent; no local search)
        cur_best_bins = elites[0][0][0]
        if cur_best_bins == best_fit[0] and random.random() < 0.10:
            mut = min(0.60, mutation_rate + 0.10)
        else:
            mut = mutation_rate

        while len(next_pop) < pop_size:
            if maybe_time_up():
                break

            pA = tournament_select(population)
            pB = tournament_select(population)

            # Crossover
            if random.random() < 0.90:
                child_order = order_crossover(pA[1], pB[1])
            else:
                child_order = pA[1][:]

            # Mutation
            if random.random() < mut:
                child_order = mutate_swap(child_order)

            # Decode and evaluate
            packing, bw = decode_best_fit(child_order, jitter=True)
            fit = fitness_from_bins(bw)
            next_pop.append((fit, child_order, packing, bw))

        population = next_pop

        # Update best
        population.sort(key=lambda t: t[0])
        if population[0][0] < best_fit:
            best_fit, best_order, best_packing, best_bw = population[0]

        if maybe_time_up():
            break

    # Return best found
    # Ensure bins are aligned and weights correct
    return {"packing": best_packing, "bin_weights": best_bw}
