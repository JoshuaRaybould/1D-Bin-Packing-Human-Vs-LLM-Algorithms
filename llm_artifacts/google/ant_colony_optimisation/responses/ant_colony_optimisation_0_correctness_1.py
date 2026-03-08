import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    num_items = len(weights)
    
    # Sort items by weight in descending order (common heuristic for bin packing)
    # Keep track of original indices
    indexed_weights = sorted([(weights[i], i) for i in range(num_items)], reverse=True)
    sorted_weights = [w for w, i in indexed_weights]
    original_indices = [i for w, i in indexed_weights]

    # ACO parameters
    num_ants = min(50, num_items)
    max_iterations = 1000 
    evaporation_rate = 0.5
    pheromone_deposit_weight = 1.0
    
    initial_pheromone = 1.0
    
    def heuristic_value(item_weight, remaining_capacity):
        if remaining_capacity < item_weight:
            return 0
        return 1.0 / (remaining_capacity - item_weight + 1e-6)

    # Pheromone matrix: pheromones[item_idx_in_sorted_list][bin_idx]
    # The bin_idx can grow up to num_items (worst case: each item in its own bin).
    # We add one extra column for the 'new bin' option.
    max_possible_bins = num_items
    pheromones = [[initial_pheromone for _ in range(max_possible_bins + 1)] for _ in range(num_items)]
    
    best_packing = None
    best_bin_weights = None
    min_bins_used = float('inf')

    iterations_done = 0
    while time.time() - start_time < time_limit and iterations_done < max_iterations:
        iterations_done += 1
        
        all_ant_results = [] # Stores (packing, bin_weights, pheromone_choices) for each ant
        
        for ant_idx in range(num_ants):
            current_packing = [[] for _ in range(max_possible_bins)]
            current_bin_weights = [0 for _ in range(max_possible_bins)]
            num_bins_used_by_ant = 0
            pheromone_choices_for_ant = [0] * num_items # Store chosen_pheromone_idx for each item
            
            for item_sorted_idx in range(num_items):
                item_weight = sorted_weights[item_sorted_idx]
                original_item_idx = original_indices[item_sorted_idx]
                
                available_bins_info = [] # List of (bin_idx, prob, heuristic_val, pheromone_idx)
                
                # Check existing bins
                for bin_idx in range(num_bins_used_by_ant):
                    remaining_capacity = bin_capacity - current_bin_weights[bin_idx]
                    if remaining_capacity >= item_weight:
                        h_val = heuristic_value(item_weight, remaining_capacity)
                        # Pheromone for this item into this existing bin
                        pheromone_level = pheromones[item_sorted_idx][bin_idx]
                        prob = (pheromone_level ** 1.0) * (h_val ** 1.0)
                        available_bins_info.append((bin_idx, prob, h_val, bin_idx))
                
                # Option to open a new bin (represented by pheromone index `max_possible_bins`)
                new_bin_pheromone_level = pheromones[item_sorted_idx][max_possible_bins]
                h_val_new_bin = heuristic_value(item_weight, bin_capacity)
                prob_new_bin = (new_bin_pheromone_level ** 1.0) * (h_val_new_bin ** 1.0)
                available_bins_info.append((num_bins_used_by_ant, prob_new_bin, h_val_new_bin, max_possible_bins))
                
                # Normalize probabilities
                total_prob_sum = sum(p for _, p, _, _ in available_bins_info)
                
                chosen_bin_idx = -1
                chosen_pheromone_idx = -1

                if total_prob_sum == 0:
                    # Fallback: select new bin if no options exist (e.g., item too large, though assumed not to happen with valid inputs).
                    # This is a safeguard. In practice, if an item fits, at least a new bin is an option.
                    chosen_bin_idx = num_bins_used_by_ant
                    chosen_pheromone_idx = max_possible_bins
                else:
                    # FIX: Correctly create normalized_probs with all 4 components for unpacking.
                    # The original code was `[(idx, p / total_prob_sum) for idx, p, _, pher_idx in available_bins_info]`, producing only 2 values.
                    # It should be `[(idx, p / total_prob_sum, h_val, pher_idx) for idx, p, h_val, pher_idx in available_bins_info]` (or similar construction)
                    # The simplest fix is to pass the full tuple and just re-normalize the prob.
                    normalized_probs_full_info = []
                    for bin_idx, prob, h_val, pher_idx in available_bins_info:
                        normalized_probs_full_info.append((bin_idx, prob / total_prob_sum, h_val, pher_idx))
                    
                    rand_val = random.random()
                    cumulative_prob = 0.0
                    for bin_idx, prob, h_val, pher_idx in normalized_probs_full_info:
                        cumulative_prob += prob
                        if rand_val <= cumulative_prob:
                            chosen_bin_idx = bin_idx
                            chosen_pheromone_idx = pher_idx
                            break
                    else:
                        # Fallback for floating point issues: pick the last option
                        chosen_bin_idx, _, _, chosen_pheromone_idx = normalized_probs_full_info[-1]
                    
                # Store the chosen pheromone index for this item
                pheromone_choices_for_ant[item_sorted_idx] = chosen_pheromone_idx

                # Place the item in the chosen bin
                if chosen_bin_idx == num_bins_used_by_ant: # This indicates a new bin is opened
                    # The index `num_bins_used_by_ant` corresponds to the next available slot.
                    # We increment `num_bins_used_by_ant` to reflect the new bin being used.
                    num_bins_used_by_ant += 1
                
                current_packing[chosen_bin_idx].append(original_item_idx)
                current_bin_weights[chosen_bin_idx] += item_weight
            
            # Finalize packing for this ant
            # Ensure we only take bins that were actually used and are not empty.
            final_packing_for_ant = [bin_items for i, bin_items in enumerate(current_packing) if i < num_bins_used_by_ant and bin_items]
            final_bin_weights_for_ant = [current_bin_weights[i] for i in range(num_bins_used_by_ant) if current_packing[i]]
            
            all_ant_results.append((final_packing_for_ant, final_bin_weights_for_ant, pheromone_choices_for_ant))
            
            # Update best solution found so far
            if len(final_packing_for_ant) < min_bins_used:
                min_bins_used = len(final_packing_for_ant)
                best_packing = final_packing_for_ant
                best_bin_weights = final_bin_weights_for_ant
        
        # Pheromone Update (Ant System variant)
        
        # Evaporation
        for i in range(num_items):
            for j in range(max_possible_bins + 1):
                pheromones[i][j] *= (1.0 - evaporation_rate)
        
        # Deposition based on solution quality
        # Use all ants, with better ants depositing more pheromone.
        for packing, bin_weights, choices in all_ant_results:
            num_bins_this_ant = len(packing)
            if num_bins_this_ant > 0:
                # Pheromone deposit amount: higher for solutions with fewer bins.
                # Use a small epsilon to avoid division by zero if num_bins_this_ant is 0 (should not happen if packing is valid).
                deposit = pheromone_deposit_weight / (num_bins_this_ant + 1e-6)
                
                # Apply deposit to the pheromone trails used by this ant
                for item_sorted_idx, chosen_pheromone_idx in enumerate(choices):
                    if 0 <= chosen_pheromone_idx <= max_possible_bins:
                         pheromones[item_sorted_idx][chosen_pheromone_idx] += deposit

    # If no solution was found and items exist, fall back to First Fit Decreasing.
    if best_packing is None and num_items > 0:
        current_packing_ffd = [[] for _ in range(num_items)]
        current_bin_weights_ffd = [0 for _ in range(num_items)]
        num_bins_used_ffd = 0
        for item_sorted_idx in range(num_items):
            item_weight = sorted_weights[item_sorted_idx]
            original_item_idx = original_indices[item_sorted_idx]
            placed = False
            for bin_idx in range(num_bins_used_ffd):
                if current_bin_weights_ffd[bin_idx] + item_weight <= bin_capacity:
                    current_packing_ffd[bin_idx].append(original_item_idx)
                    current_bin_weights_ffd[bin_idx] += item_weight
                    placed = True
                    break
            if not placed:
                current_packing_ffd[num_bins_used_ffd].append(original_item_idx)
                current_bin_weights_ffd[num_bins_used_ffd] += item_weight
                num_bins_used_ffd += 1
        
        best_packing = [bin_items for i, bin_items in enumerate(current_packing_ffd) if i < num_bins_used_ffd and bin_items]
        best_bin_weights = [current_bin_weights_ffd[i] for i in range(num_bins_used_ffd) if current_packing_ffd[i]]

    # Handle case with no items
    if num_items == 0:
        return {"packing": [], "bin_weights": []}

    return {
        "packing": best_packing,
        "bin_weights": best_bin_weights
    }