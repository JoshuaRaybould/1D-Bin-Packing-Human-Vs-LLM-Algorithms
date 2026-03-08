import time
import random

def solve(bin_capacity, weights, time_limit):
    start_time = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Parameters
    num_ants = min(10, n)
    rho = 0.1  # Evaporation rate
    alpha = 1.0 # Pheromone importance
    beta = 2.0  # Heuristic importance
    q0 = 0.5    # Exploitation vs Exploration

    # Initialize pheromones
    # pheromone[i][j] represents the preference of item j being in the same bin as item i
    pheromones = [[1.0 for _ in range(n)] for _ in range(n)]

    best_packing = []
    best_num_bins = n + 1

    def get_packing_weights(packing):
        return [sum(weights[i] for i in b) for b in packing]

    while time.time() - start_time < time_limit * 0.9:
        iteration_solutions = []

        for _ in range(num_ants):
            if time.time() - start_time > time_limit * 0.95: break
            
            unplaced = list(range(n))
            random.shuffle(unplaced)
            current_packing = []
            
            while unplaced:
                bin_contents = []
                remaining_cap = bin_capacity
                
                # Seed the bin with an item
                idx = unplaced.pop(0)
                bin_contents.append(idx)
                remaining_cap -= weights[idx]
                
                # Try to fill the rest of the bin
                while True:
                    candidates = [i for i in unplaced if weights[i] <= remaining_cap]
                    if not candidates: break
                    
                    # ACO Selection
                    probs = []
                    last_item = bin_contents[-1]
                    for c in candidates:
                        tau = pheromones[last_item][c] ** alpha
                        # Heuristic: favor items that leave less space
                        eta = (1.0 / (remaining_cap - weights[c] + 1)) ** beta
                        probs.append(tau * eta)
                    
                    if random.random() < q0:
                        # Exploitation
                        max_p = -1
                        best_c_idx = 0
                        for i, p in enumerate(probs):
                            if p > max_p:
                                max_p = p
                                best_c_idx = i
                        selected_item = candidates[best_c_idx]
                    else:
                        # Biased Exploration
                        total_p = sum(probs)
                        pick = random.random() * total_p
                        current = 0
                        selected_item = candidates[-1]
                        for i, p in enumerate(probs):
                            current += p
                            if current >= pick:
                                selected_item = candidates[i]
                                break
                    
                    bin_contents.append(selected_item)
                    remaining_cap -= weights[selected_item]
                    unplaced.remove(selected_item)
                
                current_packing.append(bin_contents)
            
            iteration_solutions.append(current_packing)
            
            if len(current_packing) < best_num_bins:
                best_num_bins = len(current_packing)
                best_packing = current_packing

        # Pheromone Update (Evaporation)
        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= (1.0 - rho)

        # Pheromone Update (Reinforcement from global best)
        if best_packing:
            reward = 1.0 / len(best_packing)
            for b in best_packing:
                for i in range(len(b)):
                    for j in range(i + 1, len(b)):
                        u, v = b[i], b[j]
                        pheromones[u][v] += reward
                        pheromones[v][u] += reward

    if not best_packing:
        # Fallback to simple First Fit if ACO didn't complete any
        items = sorted(range(n), key=lambda x: weights[x], reverse=True)
        best_packing = []
        for idx in items:
            placed = False
            for b in best_packing:
                if sum(weights[i] for i in b) + weights[idx] <= bin_capacity:
                    b.append(idx)
                    placed = True
                    break
            if not placed:
                best_packing.append([idx])

    return {
        "packing": best_packing,
        "bin_weights": get_packing_weights(best_packing)
    }