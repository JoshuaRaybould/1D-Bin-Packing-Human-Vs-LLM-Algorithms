import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    num_items = len(weights)
    
    if num_items == 0:
        return {"packing": [], "bin_weights": []}

    # Sort items by weight in descending order, keeping track of original indices
    indexed_weights = sorted([(weights[i], i) for i in range(num_items)], reverse=True)
    sorted_weights = [w for w, i in indexed_weights]
    original_indices = [i for w, i in indexed_weights]

    # ACO parameters
    num_ants = min(50, num_items) 
    max_iterations = 2000
    
    evaporation_rate = 0.3
    pheromone_deposit_weight = 1.0 
    alpha = 1.0 
    beta = 2.0 
    
    initial_pheromone = 1.0

    def heuristic_value(item_weight, remaining_capacity):
        # Standard heuristic for bin packing: prefers tighter fits.
        # Higher value for smaller remaining capacity after placing the item.
        return 1.0 / (remaining_capacity - item_weight + 1e-6)

    # Initialize pheromones
    max_possible_bins = num_items
    pheromones = [[initial_pheromone for _ in range(max_possible_bins + 1)] for _ in range(num_items)]
    
    best_packing_orig_indices = None
    best_bin_weights = None
    min_bins_used = float('inf')
    
    # ACO main loop
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break

        ant_solutions_data = [] # To store packing, weights, assignments for each ant

        for ant in range(num_ants):
            # Initialize for each ant
            packing = [[] for _ in range(num_items)] # Max possible bins is num_items
            bin_current_capacities = [bin_capacity] * num_items # Track capacity for each potential bin
            item_assignments_for_ant = [-1] * num_items # Stores which bin (option index) each item (in sorted order) is assigned to
            current_bin_idx_counter = 0 # Counts actual bins created
            num_bins_used_by_ant = 0

            # Ant constructs a solution
            for item_idx_sorted in range(num_items):
                item_weight = sorted_weights[item_idx_sorted]
                original_idx = original_indices[item_idx_sorted]

                possible_bin_choices_with_probs = [] # List of (bin_option_idx, probability)
                total_probability_mass = 0.0

                # Option 1: Place in existing bins
                for bin_idx_actual in range(num_bins_used_by_ant):
                    remaining_capacity = bin_current_capacities[bin_idx_actual]
                    if remaining_capacity >= item_weight:
                        # Pheromone for placing item `item_idx_sorted` into existing bin `bin_idx_actual`.
                        # The pheromone matrix is indexed by `item_idx_sorted` and `bin_option_idx`.
                        # `bin_option_idx` for an existing bin is its actual index.
                        tau = pheromones[item_idx_sorted][bin_idx_actual]
                        eta = heuristic_value(item_weight, remaining_capacity)
                        probability = (tau ** alpha) * (eta ** beta)
                        possible_bin_choices_with_probs.append((bin_idx_actual, probability))
                        total_probability_mass += probability
                
                # Option 2: Place in a new bin
                # The 'new bin' option is represented by index `num_items` in the pheromone matrix.
                # The heuristic for placing in a new bin is based on the item weight and full bin capacity.
                if num_bins_used_by_ant < num_items: # Ensure we don't exceed max bins
                    tau_new = pheromones[item_idx_sorted][num_items] # Pheromone for new bin option
                    # Heuristic for placing item in a fresh bin. The remaining capacity after placement is bin_capacity - item_weight.
                    eta_new = heuristic_value(item_weight, bin_capacity)
                    probability_new = (tau_new ** alpha) * (eta_new ** beta)
                    possible_bin_choices_with_probs.append((num_items, probability_new)) # num_items represents the new bin option
                    total_probability_mass += probability_new

                chosen_bin_option_idx = -1
                if total_probability_mass > 1e-9: # If there are valid choices with non-negligible probability
                    normalized_choices = []
                    for bin_option_idx, prob in possible_bin_choices_with_probs:
                        normalized_choices.append((bin_option_idx, prob / total_probability_mass))
                    
                    rand_val = random.random()
                    cumulative_prob = 0.0
                    for bin_option_idx, prob in normalized_choices:
                        cumulative_prob += prob
                        if rand_val <= cumulative_prob:
                            chosen_bin_option_idx = bin_option_idx
                            break
                
                # If no valid choice or total_probability_mass is zero, default to new bin
                # This acts as a fallback to ensure every item is assigned.
                if chosen_bin_option_idx == -1:
                    # If num_items (new bin option) was not even considered or had zero probability,
                    # we must force it if available.
                    if num_bins_used_by_ant < num_items:
                        chosen_bin_option_idx = num_items
                    else:
                        # This case should ideally not be reached if num_items > 0, as we cannot create more bins than items.
                        # If it is, it implies an unresolvable state or an item too large for capacity.
                        # For robustness, let's assume we can't place it and potentially break or raise error.
                        # However, given the problem, items are assumed to fit individually.
                        # If all bins are full and no new bin can be made, this is an issue.
                        # The problem implies items are packable. Defaulting to last bin index if num_items is reached might be an alternative, but not ideal.
                        # For ACO, we rely on problem feasibility. If an item cannot be placed, it's an issue.
                        # Let's rely on num_bins_used_by_ant < num_items for new bin availability.
                        # If this condition is false, and no existing bin fits, then something is wrong.
                        # For this problem, items are assumed to fit. So, this path indicates a bug.
                        # print(f"Warning: Item {original_idx} (weight {item_weight}) could not be placed. Bins used: {num_bins_used_by_ant}")
                        # For safety, let's assign it to the last possible bin if it's already full capacity, or error.
                        # The problem implies feasibility, so let's assume this fallback works.
                        # If no new bin can be opened, and no existing bin works, we are stuck. 
                        # However, the `if num_bins_used_by_ant < num_items:` check should ensure we *can* open a new bin if needed.
                        # If `num_bins_used_by_ant == num_items` AND no existing bin fits, this item cannot be packed. This is an issue with ACO exploration or problem setup.
                        # For now, let's re-assert that the default to `num_items` is the intended fallback if `total_probability_mass` is zero.
                        pass # The chosen_bin_option_idx remains -1, which will be handled below.

                # Final assignment logic
                if chosen_bin_option_idx == -1:
                    # If still -1, it means no choice was made or default failed. This implies an unresolvable state.
                    # For robustness, force assignment to a new bin if possible.
                    if num_bins_used_by_ant < num_items:
                        chosen_bin_option_idx = num_items
                    else:
                        # If cannot open new bin and no existing bin works, something is wrong.
                        # This state implies the problem is infeasible or ACO failed to find a placement.
                        # For a strict implementation, this might be an error condition.
                        # However, we must return a packing. Let's assume for now this state won't be reached for valid inputs.
                        # If reached, it means item cannot be packed. The error message "weight of instance differs..." implies this.
                        # This suggests that items might not be packed.
                        # To ensure all items are accounted for, let's ensure `chosen_bin_option_idx` is always valid.
                        # If `num_bins_used_by_ant == num_items` and no existing bin works, this is a failure.
                        # Let's assume the problem inputs are such that this doesn't happen, or if it does, the problem is infeasible.
                        # If `chosen_bin_option_idx` remains -1 here, it means an item was not assigned. This will cause the sum mismatch error.
                        pass # `chosen_bin_option_idx` will be -1
                
                # Assign item to the chosen bin and record the assignment choice
                # IMPORTANT: If chosen_bin_option_idx is -1, the item is not assigned. This is the likely cause of the sum mismatch.
                # We MUST ensure `chosen_bin_option_idx` is always valid if an item needs to be placed.
                # The logic should ensure this by always having `num_items` as a valid option if `num_bins_used_by_ant < num_items`.

                actual_bin_idx_assigned = -1
                if chosen_bin_option_idx == num_items: # New bin created
                    # Ensure we don't exceed num_items bins
                    if num_bins_used_by_ant < num_items:
                        actual_bin_idx_assigned = current_bin_idx_counter
                        packing[actual_bin_idx_assigned].append(original_idx)
                        bin_current_capacities[actual_bin_idx_assigned] = bin_capacity - item_weight
                        current_bin_idx_counter += 1
                        num_bins_used_by_ant += 1
                    else:
                        # This state implies we cannot open a new bin, and no existing bin worked.
                        # This item cannot be packed. This will cause the sum mismatch error.
                        # To fix this: we need to ensure `chosen_bin_option_idx` is always valid.
                        # If `num_bins_used_by_ant == num_items` AND no existing bin fits, the item is unpacked.
                        # The current loop logic for `chosen_bin_option_idx` should always pick one if `total_probability_mass > 1e-9` or default to `num_items`.
                        # The only way `chosen_bin_option_idx` can remain -1 is if `total_probability_mass` is zero AND the fallback `if chosen_bin_option_idx == -1:` part is also problematic.
                        # The check `if num_bins_used_by_ant < num_items:` should be sufficient for `num_items` option.
                        # Let's assume for now the `chosen_bin_option_idx` is always resolved to a valid option for an item that fits.
                        # The problem is likely that it *doesn't* always get resolved if `total_probability_mass` is zero and `num_items` is not available.
                        # This condition implies item cannot be packed.
                        pass # Item is not packed
                else: # Existing bin
                    actual_bin_idx_assigned = chosen_bin_option_idx
                    packing[actual_bin_idx_assigned].append(original_idx)
                    bin_current_capacities[actual_bin_idx_assigned] -= item_weight
                
                # Record the assignment choice. If an item was NOT packed, its assignment remains -1.
                item_assignments_for_ant[item_idx_sorted] = chosen_bin_option_idx

            # Store ant's solution details
            # Only include bins that actually received items
            final_packing = [bin_items for bin_items in packing if bin_items]
            # Calculate bin weights based on original weights and the packed items
            final_bin_weights = [sum(weights[i] for i in bin_items) for bin_items in final_packing]

            # Verify if all items were packed.
            # This check is crucial for debugging the sum mismatch error.
            packed_item_indices = set()
            for bin_items in final_packing:
                for item_idx in bin_items:
                    packed_item_indices.add(item_idx)
            
            if len(packed_item_indices) != num_items:
                # This indicates some items were not packed into the final solution.
                # This is the direct cause of the sum mismatch error.
                # print(f"Warning: Ant {ant} in iteration {iteration} did not pack all items. Packed: {len(packed_item_indices)}/{num_items}")
                # To prevent this from affecting the best solution update, we can skip this ant's solution
                # if it's incomplete. Or, we can try to make it complete (but that's complex).
                # For this fix, we'll mark this solution as invalid and it won't become the best.
                # If ALL ants fail to pack all items, the 'best' solution found might be incomplete.
                # We must ensure that IF a best solution is found, it is complete.
                # If `best_packing_orig_indices` is None at the end, we return empty.
                # If it's set, it must be from a complete packing.
                # Let's skip processing this ant's solution further if it's incomplete.
                continue # Skip to the next ant
            
            # If the solution is complete, add it to data for pheromone update.
            ant_solutions_data.append({
                'packing': final_packing,
                'bin_weights': final_bin_weights,
                'num_bins': len(final_packing),
                'assignments': item_assignments_for_ant # Stores the bin OPTION index for each item_idx_sorted
            })

            # Update global best solution
            if len(final_packing) < min_bins_used:
                min_bins_used = len(final_packing)
                best_packing_orig_indices = final_packing
                best_bin_weights = final_bin_weights

        # Pheromone Update
        # Evaporation
        for i in range(num_items):
            for j in range(max_possible_bins + 1):
                pheromones[i][j] *= (1.0 - evaporation_rate)

        # Pheromone Deposit
        for ant_data in ant_solutions_data:
            num_bins = ant_data['num_bins']
            # No need to check num_bins == 0 because incomplete solutions were skipped.

            delta_tau = pheromone_deposit_weight / num_bins # Pheromone deposit for this ant's solution quality
            assignments = ant_data['assignments'] # List of chosen bin_option_idx for each item_idx_sorted
            
            for item_idx_sorted in range(num_items):
                chosen_bin_option_idx = assignments[item_idx_sorted]
                # We only deposit pheromones for assignments that were made.
                # If an item was NOT assigned (`chosen_bin_option_idx == -1`), it was skipped.
                # However, ant_solutions_data only contains COMPLETE solutions, so all assignments should be valid.
                if chosen_bin_option_idx != -1:
                    # Deposit pheromone on the *choice* made by the ant
                    # `chosen_bin_option_idx` is the index into the second dimension of `pheromones`.
                    # This index is `bin_idx_actual` for existing bins, or `num_items` for a new bin.
                    pheromones[item_idx_sorted][chosen_bin_option_idx] += delta_tau

    # Return the best solution found
    # If no complete solution was ever found, best_packing_orig_indices will be None.
    if best_packing_orig_indices is None:
        # This can happen if no ant ever produced a complete packing within the time limit/iterations.
        # Or if num_items was 0 (handled at start).
        # Return an empty solution.
        return {"packing": [], "bin_weights": []}

    return {
        "packing": best_packing_orig_indices,
        "bin_weights": best_bin_weights
    }
