import random

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Decode a permutation into a packing using First Fit Decreasing-style
    def decode_first_fit(perm):
        bin_remaining = []
        bin_contents = []
        for idx in perm:
            w = weights[idx]
            # First Fit: place in first bin that fits
            placed = False
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w:
                    bin_remaining[b] -= w
                    bin_contents[b].append(idx)
                    placed = True
                    break
            if not placed:
                bin_remaining.append(bin_capacity - w)
                bin_contents.append([idx])
        return bin_contents
    
    def decode_best_fit(perm):
        bin_remaining = []
        bin_contents = []
        for idx in perm:
            w = weights[idx]
            # Best Fit: place in bin with least remaining capacity that still fits
            best_b = -1
            best_rem = bin_capacity + 1
            for b in range(len(bin_remaining)):
                if bin_remaining[b] >= w and bin_remaining[b] < best_rem:
                    best_rem = bin_remaining[b]
                    best_b = b
            if best_b >= 0:
                bin_remaining[best_b] -= w
                bin_contents[best_b].append(idx)
            else:
                bin_remaining.append(bin_capacity - w)
                bin_contents.append([idx])
        return bin_contents
    
    def fitness(packing):
        return len(packing)
    
    def make_result(packing):
        bin_weights = [sum(weights[i] for i in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}
    
    # For small instances, just use FFD
    if n <= 1:
        packing = [[i] for i in range(n)]
        return make_result(packing)
    
    # Generate initial permutations
    def ffd_permutation():
        return sorted(range(n), key=lambda i: -weights[i])
    
    def random_permutation():
        perm = list(range(n))
        random.shuffle(perm)
        return perm
    
    # Choose decode method - use best fit for better quality
    decode = decode_best_fit
    
    # GA Parameters
    if n <= 50:
        pop_size = 80
    elif n <= 200:
        pop_size = 60
    elif n <= 500:
        pop_size = 40
    else:
        pop_size = 30
    
    tournament_size = 4
    crossover_rate = 0.85
    mutation_rate = 0.4
    elite_count = max(2, pop_size // 10)
    
    # Initialize population
    population = []
    fitnesses = []
    
    # Add FFD-based individual
    ffd_perm = ffd_permutation()
    packing = decode(ffd_perm)
    population.append(ffd_perm)
    fitnesses.append(fitness(packing))
    
    # Add LFD (largest first) sorted variants with slight perturbation
    for _ in range(min(5, pop_size // 4)):
        perm = ffd_perm[:]
        # Small perturbations
        for _ in range(max(1, n // 10)):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
        packing = decode(perm)
        population.append(perm)
        fitnesses.append(fitness(packing))
    
    # Fill rest with random permutations
    while len(population) < pop_size:
        perm = random_permutation()
        packing = decode(perm)
        population.append(perm)
        fitnesses.append(fitness(packing))
    
    # Track best
    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    best_perm = population[best_idx][:]
    best_packing = decode(best_perm)
    
    # Order Crossover (OX)
    def order_crossover(p1, p2):
        size = len(p1)
        c1 = [-1] * size
        c2 = [-1] * size
        
        # Select two crossover points
        pt1 = random.randint(0, size - 1)
        pt2 = random.randint(0, size - 1)
        if pt1 > pt2:
            pt1, pt2 = pt2, pt1
        
        # Copy segment from parents
        c1[pt1:pt2+1] = p1[pt1:pt2+1]
        c2[pt1:pt2+1] = p2[pt1:pt2+1]
        
        # Fill remaining from other parent
        s1 = set(c1[pt1:pt2+1])
        s2 = set(c2[pt1:pt2+1])
        
        # For c1, fill from p2
        pos = (pt2 + 1) % size
        for i in range(size):
            idx = (pt2 + 1 + i) % size
            val = p2[idx]
            if val not in s1:
                c1[pos] = val
                pos = (pos + 1) % size
        
        # For c2, fill from p1
        pos = (pt2 + 1) % size
        for i in range(size):
            idx = (pt2 + 1 + i) % size
            val = p1[idx]
            if val not in s2:
                c2[pos] = val
                pos = (pos + 1) % size
        
        return c1, c2
    
    # PMX Crossover (alternative)
    def pmx_crossover(p1, p2):
        size = len(p1)
        pt1 = random.randint(0, size - 2)
        pt2 = random.randint(pt1 + 1, size - 1)
        
        c1 = p1[:]
        c2 = p2[:]
        
        # Create mappings
        map1to2 = {}
        map2to1 = {}
        for i in range(pt1, pt2 + 1):
            c1[i] = p2[i]
            c2[i] = p1[i]
            map1to2[p1[i]] = p2[i]
            map2to1[p2[i]] = p1[i]
        
        # Fix duplicates in c1
        for i in range(size):
            if i < pt1 or i > pt2:
                while c1[i] in map1to2:
                    c1[i] = map1to2[c1[i]]
                    if c1[i] == p1[i]:
                        break
        
        for i in range(size):
            if i < pt1 or i > pt2:
                while c2[i] in map2to1:
                    c2[i] = map2to1[c2[i]]
                    if c2[i] == p2[i]:
                        break
        
        return c1, c2
    
    # Mutation operators
    def swap_mutation(perm):
        perm = perm[:]
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def inversion_mutation(perm):
        perm = perm[:]
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i > j:
            i, j = j, i
        perm[i:j+1] = perm[i:j+1][::-1]
        return perm
    
    def scramble_mutation(perm):
        perm = perm[:]
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i > j:
            i, j = j, i
        sub = perm[i:j+1]
        random.shuffle(sub)
        perm[i:j+1] = sub
        return perm
    
    def insertion_mutation(perm):
        perm = perm[:]
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        item = perm.pop(i)
        perm.insert(j, item)
        return perm
    
    def mutate(perm):
        r = random.random()
        if r < 0.3:
            return swap_mutation(perm)
        elif r < 0.55:
            return inversion_mutation(perm)
        elif r < 0.8:
            return insertion_mutation(perm)
        else:
            return scramble_mutation(perm)
    
    def tournament_select(pop, fits, k):
        best = None
        best_f = float('inf')
        for _ in range(k):
            i = random.randint(0, len(pop) - 1)
            if fits[i] < best_f:
                best_f = fits[i]
                best = i
        return best
    
    # Main GA loop
    generation = 0
    stagnation = 0
    max_stagnation = 50
    
    # Also try first-fit decoding for the best perm
    ff_packing = decode_first_fit(ffd_perm)
    if len(ff_packing) < best_fitness:
        best_fitness = len(ff_packing)
        best_packing = ff_packing
        best_perm = ffd_perm[:]
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break
        
        generation += 1
        prev_best = best_fitness
        
        # Sort population by fitness for elitism
        combined = list(zip(fitnesses, population))
        combined.sort(key=lambda x: x[0])
        fitnesses = [c[0] for c in combined]
        population = [c[1] for c in combined]
        
        new_population = []
        new_fitnesses = []
        
        # Elitism: keep best individuals
        for i in range(elite_count):
            new_population.append(population[i][:])
            new_fitnesses.append(fitnesses[i])
        
        # Generate rest of new population
        while len(new_population) < pop_size:
            # Selection
            p1_idx = tournament_select(population, fitnesses, tournament_size)
            p2_idx = tournament_select(population, fitnesses, tournament_size)
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover
            if random.random() < crossover_rate:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1 = p1[:]
                c2 = p2[:]
            
            # Mutation
            if random.random() < mutation_rate:
                c1 = mutate(c1)
            if random.random() < mutation_rate:
                c2 = mutate(c2)
            
            # Evaluate
            packing1 = decode(c1)
            f1 = fitness(packing1)
            new_population.append(c1)
            new_fitnesses.append(f1)
            
            if f1 < best_fitness:
                best_fitness = f1
                best_perm = c1[:]
                best_packing = packing1
            
            if len(new_population) < pop_size:
                packing2 = decode(c2)
                f2 = fitness(packing2)
                new_population.append(c2)
                new_fitnesses.append(f2)
                
                if f2 < best_fitness:
                    best_fitness = f2
                    best_perm = c2[:]
                    best_packing = packing2
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Track stagnation
        if best_fitness < prev_best:
            stagnation = 0
        else:
            stagnation += 1
        
        # If stagnating, inject diversity
        if stagnation >= max_stagnation:
            stagnation = 0
            # Replace worst half with new random individuals (keep elites)
            # Sort first
            combined = list(zip(fitnesses, population))
            combined.sort(key=lambda x: x[0])
            fitnesses = [c[0] for c in combined]
            population = [c[1] for c in combined]
            
            keep = pop_size // 3
            for i in range(keep, pop_size):
                if random.random() < 0.7:
                    # Random permutation
                    perm = random_permutation()
                else:
                    # Perturbation of best
                    perm = best_perm[:]
                    num_swaps = max(1, n // 5)
                    for _ in range(num_swaps):
                        a = random.randint(0, n - 1)
                        b = random.randint(0, n - 1)
                        perm[a], perm[b] = perm[b], perm[a]
                packing = decode(perm)
                f = fitness(packing)
                population[i] = perm
                fitnesses[i] = f
                if f < best_fitness:
                    best_fitness = f
                    best_perm = perm[:]
                    best_packing = packing
        
        # Check time periodically
        if generation % 10 == 0:
            if time.time() - start_time >= time_limit * 0.95:
                break
    
    # Final: try both decodings on best perm
    packing_bf = decode_best_fit(best_perm)
    packing_ff = decode_first_fit(best_perm)
    if len(packing_ff) < len(packing_bf):
        best_packing = packing_ff
    else:
        best_packing = packing_bf
    
    return make_result(best_packing)