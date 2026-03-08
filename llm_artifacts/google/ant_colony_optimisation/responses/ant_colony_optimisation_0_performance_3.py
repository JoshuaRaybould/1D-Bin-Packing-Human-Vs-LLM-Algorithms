import time
import random

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Sort items by weight descending to aid construction
    indexed_items = sorted([(w, i) for i, w in enumerate(weights)], key=lambda x: x[0], reverse=True)
    sorted_weights = [x[0] for x in indexed_items]
    orig_indices = [x[1] for x in indexed_items]

    def get_ffd_solution():
        bins = []
        for i in range(n):
            placed = False
            w = sorted_weights[i]
            for b in bins:
                if sum(sorted_weights[item] for item in b) + w <= bin_capacity:
                    b.append(i)
                    placed = True
                    break
            if not placed:
                bins.append([i])
        return bins

    best_packing_indices = get_ffd_solution()
    best_bins_count = len(best_packing_indices)

    # ACO Parameters
    alpha = 1.0  # Pheromone importance
    beta = 5.0   # Heuristic importance (weight density)
    rho = 0.05   # Evaporation rate
    
    # MMAS Pheromone initialization
    tau_max = 1.0 / (rho * best_bins_count)
    tau_min = tau_max / (2.0 * n)
    pheromones = [[tau_max for _ in range(n)] for _ in range(n)]

    while time.time() - start_time < time_limit * 0.95:
        num_ants = 4
        iteration_solutions = []

        for _ in range(num_ants):
            unplaced = list(range(n))
            current_sol = []
            
            while unplaced:
                # Open a new bin with the largest remaining item
                bin_items = [unplaced.pop(0)]
                rem_cap = bin_capacity - sorted_weights[bin_items[0]]
                
                while True:
                    candidates = []
                    for idx, item_idx in enumerate(unplaced):
                        if sorted_weights[item_idx] <= rem_cap:
                            candidates.append((idx, item_idx))
                    
                    if not candidates:
                        break
                    
                    # Heuristic: favor items that fill the bin better
                    probs = []
                    for _, c_idx in candidates:
                        # Average pheromone between candidate and items already in bin
                        avg_tau = sum(pheromones[min(c_idx, b_idx)][max(c_idx, b_idx)] for b_idx in bin_items) / len(bin_items)
                        # Heuristic value: item weight relative to capacity
                        eta = sorted_weights[c_idx] / bin_capacity
                        probs.append((avg_tau ** alpha) * (eta ** beta))
                    
                    total_prob = sum(probs)
                    if total_prob == 0:
                        chosen_idx_in_cand = random.randrange(len(candidates))
                    else:
                        pick = random.random() * total_prob
                        cumulative = 0
                        chosen_idx_in_cand = len(candidates) - 1
                        for i, p in enumerate(probs):
                            cumulative += p
                            if cumulative >= pick:
                                chosen_idx_in_cand = i
                                break
                    
                    pop_idx, selected_item = candidates[chosen_idx_in_cand]
                    bin_items.append(selected_item)
                    rem_cap -= sorted_weights[selected_item]
                    unplaced.pop(pop_idx)
                
                current_sol.append(bin_items)
            
            iteration_solutions.append(current_sol)
            if len(current_sol) < best_bins_count:
                best_bins_count = len(current_sol)
                best_packing_indices = current_sol
                tau_max = 1.0 / (rho * best_bins_count)
                tau_min = tau_max / (2.0 * n)

        # Evaporation
        for i in range(n):
            for j in range(i + 1, n):
                pheromones[i][j] = max(tau_min, pheromones[i][j] * (1.0 - rho))
        
        # Global Best Pheromone Update (MMAS)
        delta_tau = 1.0 / best_bins_count
        for b in best_packing_indices:
            for i in range(len(b)):
                for j in range(i + 1, len(b)):
                    u, v = min(b[i], b[j]), max(b[i], b[j])
                    pheromones[u][v] = min(tau_max, pheromones[u][v] + delta_tau)

    final_packing = []
    final_weights = []
    for b in best_packing_indices:
        bin_content = [orig_indices[idx] for idx in b]
        final_packing.append(bin_content)
        final_weights.append(sum(weights[idx] for idx in bin_content))

    return {"packing": final_packing, "bin_weights": final_weights}