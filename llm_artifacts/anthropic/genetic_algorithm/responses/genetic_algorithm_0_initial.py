import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Helper: decode a permutation into a packing using First Fit
    def decode_first_fit(perm):
        bin_remaining = []
        bin_items = []
        for idx in perm:
            w = weights[idx]
            placed = False
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w:
                    bin_remaining[b] -= w
                    bin_items[b].append(idx)
                    placed = True
                    break
            if not placed:
                bin_remaining.append(bin_capacity - w)
                bin_items.append([idx])
        return bin_items
    
    # Faster decode using Best Fit
    def decode_best_fit(perm):
        bin_remaining = []
        bin_items = []
        for idx in perm:
            w = weights[idx]
            best_b = -1
            best_rem = bin_capacity + 1
            for b in range(len(bin_remaining)):
                rem = bin_remaining[b]
                if rem >= w and rem < best_rem:
                    best_rem = rem
                    best_b = b
            if best_b >= 0:
                bin_remaining[best_b] -= w
                bin_items[best_b].append(idx)
            else:
                bin_remaining.append(bin_capacity - w)
                bin_items.append([idx])
        return bin_items
    
    def evaluate(perm):
        return len(decode_best_fit(perm))
    
    def format_solution(perm):
        packing = decode_best_fit(perm)
        bin_weights = [sum(weights[i] for i in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}
    
    # Generate initial permutations
    # FFD: sort by decreasing weight
    ffd_perm = sorted(range(n), key=lambda i: -weights[i])
    
    # Population size
    pop_size = min(100, max(20, n))
    
    population = []
    fitness = []
    
    # Add FFD permutation
    population.append(list(ffd_perm))
    fitness.append(evaluate(ffd_perm))
    
    best_perm = list(ffd_perm)
    best_fitness = fitness[0]
    
    # Generate rest of population
    for i in range(pop_size - 1):
        if time.time() - start_time > time_limit * 0.9:
            return format_solution(best_perm)
        if i < pop_size // 3:
            # Perturbed FFD
            perm = list(ffd_perm)
            num_swaps = random.randint(1, max(1, n // 5))
            for _ in range(num_swaps):
                a, b = random.randrange(n), random.randrange(n)
                perm[a], perm[b] = perm[b], perm[a]
        else:
            # Random permutation
            perm = list(range(n))
            random.shuffle(perm)
        population.append(perm)
        f = evaluate(perm)
        fitness.append(f)
        if f < best_fitness:
            best_fitness = f
            best_perm = list(perm)
    
    # Tournament selection
    def tournament_select(k=3):
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best_c = candidates[0]
        for c in candidates[1:]:
            if fitness[c] < fitness[best_c]:
                best_c = c
        return best_c
    
    # Order Crossover (OX)
    def order_crossover(p1, p2):
        size = len(p1)
        if size <= 1:
            return list(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        # Copy segment from p1
        child[a:b+1] = p1[a:b+1]
        segment_set = set(child[a:b+1])
        # Fill remaining from p2 in order
        pos = (b + 1) % size
        for item in p2:
            if item not in segment_set:
                child[pos] = item
                pos = (pos + 1) % size
        return child
    
    # Partially Mapped Crossover (PMX)
    def pmx_crossover(p1, p2):
        size = len(p1)
        if size <= 1:
            return list(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = list(p2)
        mapping = {}
        for i in range(a, b + 1):
            child[i] = p1[i]
            mapping[p1[i]] = p2[i]
        
        used = set(child[a:b+1])
        for i in list(range(0, a)) + list(range(b+1, size)):
            val = child[i]
            while val in used:
                val = mapping.get(val, val)
                if val == child[i]:
                    break
            child[i] = val
        
        # Fallback: if invalid, use OX
        if len(set(child)) != size:
            return order_crossover(p1, p2)
        return child
    
    # Mutation: swap mutation
    def mutate_swap(perm, num_swaps=1):
        perm = list(perm)
        for _ in range(num_swaps):
            a, b = random.randrange(n), random.randrange(n)
            perm[a], perm[b] = perm[b], perm[a]
        return perm
    
    # Mutation: inversion
    def mutate_inversion(perm):
        perm = list(perm)
        if n <= 1:
            return perm
        a, b = sorted(random.sample(range(n), 2))
        perm[a:b+1] = reversed(perm[a:b+1])
        return perm
    
    # Mutation: scramble
    def mutate_scramble(perm):
        perm = list(perm)
        if n <= 1:
            return perm
        a, b = sorted(random.sample(range(n), 2))
        sub = perm[a:b+1]
        random.shuffle(sub)
        perm[a:b+1] = sub
        return perm
    
    # GA parameters
    elite_count = max(2, pop_size // 10)
    mutation_rate = 0.3
    crossover_rate = 0.8
    
    generation = 0
    max_generations = 10000
    
    # Lower bound for early termination
    total_weight = sum(weights)
    lower_bound = (total_weight + bin_capacity - 1) // bin_capacity
    
    while generation < max_generations:
        if time.time() - start_time > time_limit * 0.85:
            break
        
        if best_fitness <= lower_bound:
            break
        
        generation += 1
        
        # Sort population by fitness
        paired = list(zip(fitness, population))
        paired.sort(key=lambda x: x[0])
        fitness = [p[0] for p in paired]
        population = [p[1] for p in paired]
        
        new_population = []
        new_fitness = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(list(population[i]))
            new_fitness.append(fitness[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit * 0.85:
                break
            
            # Selection
            p1_idx = tournament_select(3)
            p2_idx = tournament_select(3)
            
            # Crossover
            if random.random() < crossover_rate:
                if random.random() < 0.5:
                    child = order_crossover(population[p1_idx], population[p2_idx])
                else:
                    child = order_crossover(population[p2_idx], population[p1_idx])
            else:
                child = list(population[p1_idx])
            
            # Mutation
            if random.random() < mutation_rate:
                r = random.random()
                if r < 0.4:
                    child = mutate_swap(child, random.randint(1, max(1, n // 10)))
                elif r < 0.7:
                    child = mutate_inversion(child)
                else:
                    child = mutate_scramble(child)
            
            f = evaluate(child)
            new_population.append(child)
            new_fitness.append(f)
            
            if f < best_fitness:
                best_fitness = f
                best_perm = list(child)
                if best_fitness <= lower_bound:
                    break
        
        population = new_population
        fitness = new_fitness
    
    return format_solution(best_perm)
