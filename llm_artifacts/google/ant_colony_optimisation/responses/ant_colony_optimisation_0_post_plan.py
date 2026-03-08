import time
import random

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

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

    alpha = 1.0
    beta = 2.0
    rho = 0.1
    q0 = 0.5
    
    tau_max = 1.0 / (rho * best_bins_count)
    tau_min = tau_max / (2.0 * n)
    pheromones = [[tau_max for _ in range(n)] for _ in range(n)]

    while time.time() - start_time < time_limit * 0.98:
        iteration_best = None
        num_ants = 5
        
        for _ in range(num_ants):
            unplaced = list(range(n))
            current_sol = []
            
            while unplaced:
                bin_items = [unplaced.pop(0)]
                rem_cap = bin_capacity - sorted_weights[bin_items[0]]
                
                while True:
                    candidates = []
                    count = 0
                    for idx, item_idx in enumerate(unplaced):
                        if sorted_weights[item_idx] <= rem_cap:
                            candidates.append((idx, item_idx))
                            count += 1
                            if count >= 50: break
                    
                    if not candidates: break
                    
                    scores = []
                    for _, c_idx in candidates:
                        avg_tau = sum(pheromones[c_idx][b_idx] for b_idx in bin_items) / len(bin_items)
                        score = (avg_tau ** alpha) * (sorted_weights[c_idx] ** beta)
                        scores.append(score)
                    
                    if random.random() < q0:
                        best_score_idx = 0
                        max_s = -1.0
                        for i, s in enumerate(scores):
                            if s > max_s:
                                max_s = s
                                best_score_idx = i
                        chosen_idx_in_cand = best_score_idx
                    else:
                        total_s = sum(scores)
                        pick = random.random() * total_s
                        acc = 0
                        chosen_idx_in_cand = len(candidates) - 1
                        for i, s in enumerate(scores):
                            acc += s
                            if acc >= pick:
                                chosen_idx_in_cand = i
                                break
                    
                    pop_idx, selected_item = candidates[chosen_idx_in_cand]
                    bin_items.append(selected_item)
                    rem_cap -= sorted_weights[selected_item]
                    unplaced.pop(pop_idx)
                current_sol.append(bin_items)
            
            if len(current_sol) < best_bins_count:
                best_bins_count = len(current_sol)
                best_packing_indices = current_sol
                tau_max = 1.0 / (rho * best_bins_count)
                tau_min = tau_max / (2.0 * n)

        for i in range(n):
            for j in range(n):
                pheromones[i][j] = max(tau_min, pheromones[i][j] * (1.0 - rho))
        
        reward = 1.0 / best_bins_count
        for b in best_packing_indices:
            for i in range(len(b)):
                for j in range(i + 1, len(b)):
                    u, v = b[i], b[j]
                    pheromones[u][v] = min(tau_max, pheromones[u][v] + reward)
                    pheromones[v][u] = pheromones[u][v]

    final_packing = []
    final_weights = []
    for b in best_packing_indices:
        bin_content = [orig_indices[idx] for idx in b]
        final_packing.append(bin_content)
        final_weights.append(sum(weights[idx] for idx in bin_content))

    return {"packing": final_packing, "bin_weights": final_weights}