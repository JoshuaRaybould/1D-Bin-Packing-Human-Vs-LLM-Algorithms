import time
import random

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Sort items by weight descending to aid heuristic construction
    indexed_items = sorted([(w, i) for i, w in enumerate(weights)], key=lambda x: x[0], reverse=True)
    sorted_weights = [x[0] for x in indexed_items]
    orig_indices = [x[1] for x in indexed_items]

    def get_ffd_solution():
        bins = []
        for i in range(n):
            placed = False
            w = sorted_weights[i]
            for b in bins:
                current_weight = sum(sorted_weights[item] for item in b)
                if current_weight + w <= bin_capacity:
                    b.append(i)
                    placed = True
                    break
            if not placed:
                bins.append([i])
        return bins

    # Initial solution using First Fit Decreasing
    best_bins_indices = get_ffd_solution()
    best_count = len(best_bins_indices)

    # ACO Parameters
    alpha = 1.0  # Pheromone importance
    beta = 2.0   # Heuristic importance
    rho = 0.05   # Evaporation rate
    
    # Pheromone Matrix (closeness of two items in the same bin)
    tau_max = 1.0 / (rho * best_count)
    tau_min = tau_max / (2.0 * n)
    pheromones = [[tau_max for _ in range(n)] for _ in range(n)]

    while time.time() - start_time < time_limit * 0.95:
        # Construct solutions with ants
        num_ants = 4
        iteration_solutions = []
        
        for _ in range(num_ants):
            if time.time() - start_time > time_limit * 0.98: break
            
            unplaced = list(range(n))
            current_sol = []
            
            while unplaced:
                # Start a new bin with the largest available item
                bin_items = [unplaced.pop(0)]
                rem_cap = bin_capacity - sorted_weights[bin_items[0]]
                
                while True:
                    candidates = []
                    for idx, item_idx in enumerate(unplaced):
                        if sorted_weights[item_idx] <= rem_cap:
                            candidates.append((idx, item_idx))
                    
                    if not candidates: break
                    
                    # Calculate scores for candidates
                    scores = []
                    for _, c_idx in candidates:
                        # Average pheromone between candidate and items already in bin
                        avg_tau = sum(pheromones[bin_items[j]][c_idx] for j in range(len(bin_items))) / len(bin_items)
                        # Heuristic: prioritize items that fill the remaining capacity well
                        eta = 1.0 / (rem_cap - sorted_weights[c_idx] + 1)
                        scores.append((avg_tau ** alpha) * (eta ** beta))
                    
                    total_score = sum(scores)
                    pick = random.random() * total_score
                    cumulative = 0
                    chosen_idx = len(candidates) - 1
                    for i, s in enumerate(scores):
                        cumulative += s
                        if cumulative >= pick:
                            chosen_idx = i
                            break
                    
                    pop_idx, item_idx = candidates[chosen_idx]
                    bin_items.append(item_idx)
                    rem_cap -= sorted_weights[item_idx]
                    unplaced.pop(pop_idx)
                
                current_sol.append(bin_items)
            
            iteration_solutions.append(current_sol)
            if len(current_sol) < best_count:
                best_count = len(current_sol)
                best_bins_indices = current_sol
                tau_max = 1.0 / (rho * best_count)
                tau_min = tau_max / (2.0 * n)

        # Evaporation
        for i in range(n):
            for j in range(i + 1, n):
                pheromones[i][j] = max(tau_min, pheromones[i][j] * (1.0 - rho))
                pheromones[j][i] = pheromones[i][j]

        # Pheromone Update (using best-so-far solution)
        d_tau = 1.0 / best_count
        for b in best_bins_indices:
            for i in range(len(b)):
                for j in range(i + 1, len(b)):
                    u, v = b[i], b[j]
                    pheromones[u][v] = min(tau_max, pheromones[u][v] + d_tau)
                    pheromones[v][u] = pheromones[u][v]

    # Prepare result
    final_packing = []
    final_weights = []
    for b in best_bins_indices:
        indices = [orig_indices[idx] for idx in b]
        final_packing.append(indices)
        final_weights.append(sum(weights[idx] for idx in indices))

    return {"packing": final_packing, "bin_weights": final_weights}