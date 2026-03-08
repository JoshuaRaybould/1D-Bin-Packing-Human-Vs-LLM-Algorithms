import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # Decode a permutation into a packing using Best Fit
    def decode_best_fit(perm):
        bin_weights = []
        bin_items = []
        for idx in perm:
            w = weights[idx]
            # Find best fit bin (least remaining space after placing)
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bin_weights)):
                remaining = bin_capacity - bin_weights[b]
                if remaining >= w and remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b
                    if remaining == w:
                        break  # perfect fit
            if best_bin >= 0:
                bin_weights[best_bin] += w
                bin_items[best_bin].append(idx)
            else:
                bin_weights.append(w)
                bin_items.append([idx])
        return bin_items, bin_weights
    
    # Decode using First Fit
    def decode_first_fit(perm):
        bin_weights = []
        bin_items = []
        for idx in perm:
            w = weights[idx]
            placed = False
            for b in range(len(bin_weights)):
                if bin_capacity - bin_weights[b] >= w:
                    bin_weights[b] += w
                    bin_items[b].append(idx)
                    placed = True
                    break
            if not placed:
                bin_weights.append(w)
                bin_items.append([idx])
        return bin_items, bin_weights
    
    def decode(perm):
        return decode_best_fit(perm)
    
    def fitness(perm):
        _, bw = decode(perm)
        return len(bw)
    
    # Generate initial solutions
    def make_ffd_perm():
        # Sort by weight descending
        return sorted(range(n), key=lambda i: -weights[i])
    
    def make_random_perm():
        p = list(range(n))
        random.shuffle(p)
        return p
    
    def make_ffd_random_perm():
        # Sort descending with some randomness
        return sorted(range(n), key=lambda i: -weights[i] + random.uniform(-0.5, 0.5) * weights[i])
    
    # Order Crossover (OX)
    def order_crossover(p1, p2):
        size = len(p1)
        if size <= 2:
            return p1[:], p2[:]
        
        c1, c2 = random.sample(range(size), 2)
        if c1 > c2:
            c1, c2 = c2, c1
        
        # Child 1
        child1 = [-1] * size
        child1[c1:c2+1] = p1[c1:c2+1]
        segment_set = set(p1[c1:c2+1])
        
        pos = (c2 + 1) % size
        for i in range(size):
            idx = (c2 + 1 + i) % size
            gene = p2[idx]
            if gene not in segment_set:
                child1[pos] = gene
                pos = (pos + 1) % size
        
        # Child 2
        child2 = [-1] * size
        child2[c1:c2+1] = p2[c1:c2+1]
        segment_set2 = set(p2[c1:c2+1])
        
        pos = (c2 + 1) % size
        for i in range(size):
            idx = (c2 + 1 + i) % size
            gene = p1[idx]
            if gene not in segment_set2:
                child2[pos] = gene
                pos = (pos + 1) % size
        
        return child1, child2
    
    # Partially Mapped Crossover (PMX)
    def pmx_crossover(p1, p2):
        size = len(p1)
        if size <= 2:
            return p1[:], p2[:]
        
        c1, c2 = sorted(random.sample(range(size), 2))
        
        child1 = p1[:]
        child2 = p2[:]
        
        # Build position maps
        pos1 = [0] * size
        pos2 = [0] * size
        for i in range(size):
            pos1[child1[i]] = i
            pos2[child2[i]] = i
        
        for i in range(c1, c2 + 1):
            val1 = child1[i]
            val2 = child2[i]
            if val1 != val2:
                # Swap in child1
                p1_pos = pos1[val2]
                child1[i], child1[p1_pos] = child1[p1_pos], child1[i]
                pos1[val1] = p1_pos
                pos1[val2] = i
                
                # Swap in child2
                p2_pos = pos2[val1]
                child2[i], child2[p2_pos] = child2[p2_pos], child2[i]
                pos2[val2] = p2_pos
                pos2[val1] = i
        
        return child1, child2
    
    # Mutation: swap two random positions
    def swap_mutation(perm):
        p = perm[:]
        if len(p) <= 1:
            return p
        i, j = random.sample(range(len(p)), 2)
        p[i], p[j] = p[j], p[i]
        return p
    
    # Mutation: insert - remove an element and insert it elsewhere
    def insert_mutation(perm):
        p = perm[:]
        if len(p) <= 1:
            return p
        i = random.randrange(len(p))
        gene = p.pop(i)
        j = random.randrange(len(p) + 1)
        p.insert(j, gene)
        return p
    
    # Mutation: inversion - reverse a subsequence
    def inversion_mutation(perm):
        p = perm[:]
        if len(p) <= 2:
            return p
        i, j = sorted(random.sample(range(len(p)), 2))
        p[i:j+1] = reversed(p[i:j+1])
        return p
    
    # Scramble mutation: scramble a small segment
    def scramble_mutation(perm):
        p = perm[:]
        if len(p) <= 2:
            return p
        seg_len = random.randint(2, min(8, len(p)))
        start = random.randint(0, len(p) - seg_len)
        segment = p[start:start+seg_len]
        random.shuffle(segment)
        p[start:start+seg_len] = segment
        return p
    
    def mutate(perm):
        r = random.random()
        if r < 0.3:
            return swap_mutation(perm)
        elif r < 0.6:
            return insert_mutation(perm)
        elif r < 0.8:
            return inversion_mutation(perm)
        else:
            return scramble_mutation(perm)
    
    # Tournament selection
    def tournament_select(population, fitnesses, k=3):
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best = min(candidates, key=lambda i: fitnesses[i])
        return population[best]
    
    # Parameters
    if n <= 20:
        pop_size = 30
    elif n <= 100:
        pop_size = 50
    elif n <= 500:
        pop_size = 60
    else:
        pop_size = 40
    
    crossover_rate = 0.85
    mutation_rate = 0.4
    elite_count = max(2, pop_size // 10)
    tournament_size = 4
    
    # Initialize population
    population = []
    
    # FFD solution
    ffd_perm = make_ffd_perm()
    population.append(ffd_perm)
    
    # Several FFD-like with noise
    for _ in range(min(pop_size // 4, 10)):
        population.append(make_ffd_random_perm())
    
    # Fill rest with random
    while len(population) < pop_size:
        population.append(make_random_perm())
    
    # Evaluate initial population
    fitnesses = [fitness(p) for p in population]
    
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_perm = population[best_idx][:]
    best_fitness = fitnesses[best_idx]
    
    # Also try first fit decoding for the FFD perm to see if it's better
    ffd_items_bf, ffd_weights_bf = decode_best_fit(ffd_perm)
    ffd_items_ff, ffd_weights_ff = decode_first_fit(ffd_perm)
    
    if len(ffd_weights_ff) < best_fitness:
        best_fitness = len(ffd_weights_ff)
        best_perm = ffd_perm[:]
    
    generation = 0
    no_improve_count = 0
    
    elapsed = time.time() - start_time
    time_budget = time_limit * 0.95  # leave some margin
    
    while elapsed < time_budget:
        generation += 1
        
        # Sort population by fitness for elitism
        combined = list(zip(fitnesses, population))
        combined.sort(key=lambda x: x[0])
        fitnesses = [c[0] for c in combined]
        population = [c[1] for c in combined]
        
        new_population = []
        new_fitnesses = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(population[i][:])
            new_fitnesses.append(fitnesses[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time >= time_budget:
                break
            
            parent1 = tournament_select(population, fitnesses, tournament_size)
            parent2 = tournament_select(population, fitnesses, tournament_size)
            
            if random.random() < crossover_rate:
                if random.random() < 0.5:
                    child1, child2 = order_crossover(parent1, parent2)
                else:
                    child1, child2 = pmx_crossover(parent1, parent2)
            else:
                child1 = parent1[:]
                child2 = parent2[:]
            
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            
            f1 = fitness(child1)
            new_population.append(child1)
            new_fitnesses.append(f1)
            
            if len(new_population) < pop_size:
                f2 = fitness(child2)
                new_population.append(child2)
                new_fitnesses.append(f2)
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Update best
        gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_perm = population[gen_best_idx][:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If no improvement for a while, inject diversity
        if no_improve_count > 20:
            # Replace worst half with new random/FFD-noise permutations
            combined = list(zip(fitnesses, population))
            combined.sort(key=lambda x: x[0])
            keep = pop_size // 3
            population = [c[1] for c in combined[:keep]]
            fitnesses = [c[0] for c in combined[:keep]]
            
            # Ensure best is in there
            population.insert(0, best_perm[:])
            fitnesses.insert(0, best_fitness)
            
            while len(population) < pop_size:
                if random.random() < 0.3:
                    p = make_ffd_random_perm()
                else:
                    p = make_random_perm()
                population.append(p)
                fitnesses.append(fitness(p))
            
            no_improve_count = 0
        
        elapsed = time.time() - start_time
    
    # Build final solution from best permutation
    packing, bin_weights = decode(best_perm)
    
    # Also try first fit on best perm
    packing_ff, bin_weights_ff = decode_first_fit(best_perm)
    if len(bin_weights_ff) < len(bin_weights):
        packing = packing_ff
        bin_weights = bin_weights_ff
    
    return {"packing": packing, "bin_weights": bin_weights}