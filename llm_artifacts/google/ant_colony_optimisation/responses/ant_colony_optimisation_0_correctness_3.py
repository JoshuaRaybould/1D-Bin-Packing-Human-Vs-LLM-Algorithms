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
    # Increased max_iterations for potentially better quality within time limits
    # The actual runtime will be governed by time_limit, but having a reasonable iteration cap is good.
    max_iterations = 2000
    
    evaporation_rate = 0.3
    pheromone_deposit_weight = 1.0 # Corresponds to Q in ACO literature
    alpha = 1.0 # Pheromone influence
    beta = 2.0 # Heuristic influence (higher beta for more emphasis on heuristic)
    
    initial_pheromone = 1.0

    def heuristic_value(item_weight, remaining_capacity):
        # Standard heuristic for bin packing: inversely proportional to remaining capacity.
        # Add epsilon to avoid division by zero.
        if remaining_capacity < item_weight:
            return 0.0
        # Using remaining_capacity - item_weight + 1e-6 is common for similar problems,
        # but for bin packing, heuristic often relates to how much space is left.
        # A simpler heuristic: more space is better for fitting, but also for fitting larger items.
        # For ACO, we want ants to choose good bins. A bin with more remaining capacity is potentially better.
        # Let's try a heuristic that favors bins that can fit the current item well.
        # The heuristic should guide the ant to choose a bin. If it's a new bin, it should be evaluated.
        # If it's an existing bin, its remaining capacity is key.
        # For assigning an item to a bin, remaining_capacity is the capacity *after* placing the item.
        # So, remaining_capacity = bin_current_capacity - item_weight.
        # A higher heuristic value should indicate a better choice.
        # If remaining_capacity is very small, it might be a tight fit, which could be good or bad.
        # Let's use the inverse of the used capacity (capacity - remaining_capacity).
        # Or, simply the remaining capacity itself is a good heuristic for preferring emptier bins.
        # The original heuristic was 1.0 / (remaining_capacity - item_weight + 1e-6)
        # Let's reconsider. When an ant decides which bin to put an item in:
        # The heuristic value should represent the desirability of a particular bin choice.
        # Option 1: Try to fill bins tightly. Heuristic = 1 / (remaining_capacity_after_item)
        # Option 2: Prefer bins with lots of space. Heuristic = remaining_capacity_after_item.
        # The typical ACO for TSP uses inverse distance. For bin packing, it's trickier.
        # A common approach is to use the ratio of remaining capacity to item weight.
        # However, for selecting a *bin* for an item, the remaining capacity of the bin is what matters.
        # Let's use remaining_capacity - item_weight as the basis for heuristic.
        # A higher value indicates more space left in the bin after placing the item.
        # So, heuristic_value = remaining_capacity - item_weight + 1e-6 is problematic if this value is negative (item doesn't fit).
        # If the item fits, remaining_capacity >= item_weight. Thus, remaining_capacity - item_weight >= 0.
        # A heuristic like 1.0 / (remaining_capacity - item_weight + 1e-6) would penalize larger remaining spaces.
        # Let's use: heuristic = remaining_capacity_of_bin_after_placement + 1e-6
        # Which means (current_bin_capacity - item_weight) + 1e-6
        # But the input to this function is `remaining_capacity` of the bin *before* placing the item.
        # So, `remaining_capacity_after_item = remaining_capacity - item_weight`.
        # The heuristic should be high if the remaining space is good.
        # A higher remaining_capacity is generally good. Let's use it directly.
        # Or, `1.0 / (remaining_capacity - item_weight + 1e-6)` implies smaller remaining space is better.
        # Let's re-evaluate common ACO heuristics for bin packing.
        # Often, the heuristic is related to how well an item fits into a bin.
        # `eta = 1 / (bin_capacity - current_bin_load)` or similar is used.
        # But here, we are choosing *which* bin to place an item into.
        # Let's try a heuristic that is higher for bins with more capacity remaining *after* placing the item.
        # This means `remaining_capacity - item_weight` should be maximized.
        # So, `heuristic_value = remaining_capacity - item_weight + 1e-6` if item_weight <= remaining_capacity.
        # If `remaining_capacity - item_weight` is 0, it's a perfect fit, high heuristic.
        # If `remaining_capacity - item_weight` is large, it means a lot of space is wasted, lower heuristic.
        # So, inverse might be more appropriate.
        # Let's use the original idea: 1.0 / (remaining_capacity_after_placement + 1e-6)
        # where remaining_capacity_after_placement = remaining_capacity - item_weight
        return 1.0 / (remaining_capacity - item_weight + 1e-6)

    # Pheromone matrix: pheromones[item_idx_in_sorted_list][bin_option_idx]
    # An item can be placed in any existing bin or a new bin.
    # The number of possible bins can be at most num_items (each item in its own bin).
    # So, number of bin options for an item is `num_items` (existing bins) + 1 (new bin).
    # `num_items` items, each potentially has `num_items+1` choices.
    # However, the pheromone matrix usually relates to transitions or choices. 
    # For bin packing, an ant iterates through items and decides for *each item* which bin to place it in.
    # The pheromone should probably represent the desirability of placing item `i` in bin `j`.
    # A common ACO formulation for bin packing uses pheromones on (item, bin) pairs or (item, decision) pairs.
    # Here, the structure suggests `pheromones[item_idx][bin_choice_idx]` where `bin_choice_idx` is an index to a bin or a 'new bin' option.
    # Let's reconsider the structure.
    # When an ant constructs a solution, it iterates through items (in sorted order).
    # For each item, it considers placing it in existing bins or a new bin.
    # The decision for item `i` is to choose a bin `j`. Pheromone could be on `(i, j)`.
    # If we have `num_items` items, and `k` bins, we have `num_items * k` choices.
    # The problem states `pheromones[item_idx_in_sorted_list][bin_option_idx]`. 
    # `item_idx_in_sorted_list` is `0` to `num_items-1`.
    # `bin_option_idx` goes up to `num_items` (for new bin option).
    # This implies `pheromones[i][j]` where `i` is the `i`-th item in the sorted list, and `j` is a bin index (0 to num_items-1) or `num_items` for a new bin.
    # This means `pheromones` should be `num_items x (num_items + 1)`. This seems correct for the description.
    max_possible_bins = num_items
    pheromones = [[initial_pheromone for _ in range(max_possible_bins + 1)] for _ in range(num_items)]
    
    best_packing_orig_indices = None
    best_bin_weights = None
    min_bins_used = float('inf')

    # ACO main loop
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break

        ant_solutions = []
        # Store solutions in sorted order for pheromone update
        current_ant_packings = []
        current_ant_bin_weights = []
        current_ant_num_bins = []

        for ant in range(num_ants):
            # Initialize for each ant
            packing = [[] for _ in range(num_items)] # Max possible bins is num_items
            bin_current_capacities = [bin_capacity] * num_items # Initialize with capacity for each potential bin
            item_assignments = [-1] * num_items # Stores which bin each item (in sorted order) is assigned to
            current_bin_idx = 0
            num_bins_used_by_ant = 0

            # Ant constructs a solution by assigning items one by one
            for item_idx_sorted in range(num_items):
                item_weight = sorted_weights[item_idx_sorted]
                original_idx = original_indices[item_idx_sorted]

                # Determine probabilities for choosing bins
                possible_bin_choices = [] # List of (bin_idx, probability)
                total_probability_mass = 0.0

                # Option 1: Place in existing bins
                for bin_idx in range(num_bins_used_by_ant):
                    remaining_capacity = bin_current_capacities[bin_idx]
                    if remaining_capacity >= item_weight:
                        # Pheromone for this bin choice
                        # Pheromones are indexed by the item's position in the sorted list
                        # and the bin's index (or new bin option).
                        # The pheromone matrix `pheromones[item_idx_sorted][bin_idx]` represents the desirability of placing
                        # `item_idx_sorted` into bin `bin_idx`.
                        tau = pheromones[item_idx_sorted][bin_idx]
                        eta = heuristic_value(item_weight, remaining_capacity)
                        probability = (tau ** alpha) * (eta ** beta)
                        possible_bin_choices.append((bin_idx, probability))
                        total_probability_mass += probability
                
                # Option 2: Place in a new bin
                # The 'new bin' option is represented by index `num_items` in the pheromone matrix.
                # This refers to pheromones[item_idx_sorted][num_items].
                # A heuristic for a new bin is often constant or based on item weight itself.
                # For simplicity, let's give it a small base heuristic or consider it neutral.
                # However, the decision to open a new bin is influenced by the *lack* of good existing bins.
                # The `heuristic_value` function is designed for existing bins. How to evaluate a new bin?
                # A new bin has full `bin_capacity`. The item `item_weight` will be placed.
                # So, remaining capacity after placement will be `bin_capacity - item_weight`.
                # Using the same heuristic logic: `1.0 / (bin_capacity - item_weight + 1e-6)`.
                if num_bins_used_by_ant < num_items: # Ensure we don't create more bins than items
                    tau_new = pheromones[item_idx_sorted][num_items] # Pheromone for new bin option
                    eta_new = heuristic_value(item_weight, bin_capacity) # Heuristic for placing item in a new bin
                    probability_new = (tau_new ** alpha) * (eta_new ** beta)
                    possible_bin_choices.append((num_items, probability_new)) # num_items represents the new bin index
                    total_probability_mass += probability_new

                # Normalize probabilities and make a choice
                chosen_bin_idx = -1
                if total_probability_mass > 1e-9: # If there are valid choices
                    # Normalize probabilities
                    normalized_choices = []
                    for bin_idx, prob in possible_bin_choices:
                        normalized_choices.append((bin_idx, prob / total_probability_mass))
                    
                    # Select bin using roulette wheel selection
                    rand_val = random.random()
                    cumulative_prob = 0.0
                    for bin_idx, prob in normalized_choices:
                        cumulative_prob += prob
                        if rand_val <= cumulative_prob:
                            chosen_bin_idx = bin_idx
                            break
                
                # If for some reason no bin was chosen (e.g., all probabilities were 0), place in new bin
                if chosen_bin_idx == -1:
                    chosen_bin_idx = num_items # Force new bin

                # Assign item to the chosen bin
                if chosen_bin_idx == num_items: # New bin
                    # Assign to the next available bin index slot
                    new_bin_actual_idx = num_bins_used_by_ant
                    packing[new_bin_actual_idx].append(original_idx)
                    bin_current_capacities[new_bin_actual_idx] = bin_capacity - item_weight
                    item_assignments[item_idx_sorted] = new_bin_actual_idx
                    num_bins_used_by_ant += 1
                else: # Existing bin
                    bin_idx = chosen_bin_idx
                    packing[bin_idx].append(original_idx)
                    bin_current_capacities[bin_idx] -= item_weight
                    item_assignments[item_idx_sorted] = bin_idx
            
            # Store ant's solution (remove empty bins)
            final_packing = [bin_items for bin_items in packing if bin_items]
            final_bin_weights = [sum(weights[i] for i in bin_items) for bin_items in final_packing]
            
            current_ant_packings.append(final_packing)
            current_ant_bin_weights.append(final_bin_weights)
            current_ant_num_bins.append(len(final_packing))

            # Update global best solution found so far
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
        # Iterate through each ant's solution and deposit pheromones
        for ant_sol_idx in range(num_ants):
            packing = current_ant_packings[ant_sol_idx]
            num_bins = len(packing)
            if num_bins == 0: continue # Skip if ant produced no solution

            # Calculate pheromone deposit amount (e.g., 1/num_bins or other quality measure)
            # Using 1/num_bins is standard for minimization problems like bin packing.
            delta_tau = pheromone_deposit_weight / num_bins

            # Deposit pheromones for each item placement in this ant's solution
            # We need to know WHICH item was placed in WHICH bin and how it relates to the original sorted item index.
            # The `item_assignments` array from the ant's construction phase is key.
            # Reconstruct the assignment based on the solution packing and original item indices.
            # This is a bit complex: the current ant's `item_assignments` refers to the sorted `item_idx_sorted`.
            # For each `item_idx_sorted` from 0 to `num_items-1`:
            # `assigned_bin_idx = item_assignments[item_idx_sorted]`
            # `original_idx = original_indices[item_idx_sorted]`
            # `tau_to_add = delta_tau` if this assignment was part of the ant's solution paths.
            # This requires re-linking assignments. A simpler way: Iterate through the ant's constructed packing.
            
            # Let's rethink how to map the solution back to pheromone updates.
            # An ant constructs assignments `item_assignments[item_idx_sorted] = chosen_bin_idx`.
            # `item_idx_sorted` is 0 to `num_items-1`.
            # `chosen_bin_idx` is 0 to `num_bins_used_by_ant-1` or `num_items` for new bin.
            # The pheromone matrix `pheromones[i][j]` means "pheromone for placing item `i` (in sorted list) into bin `j` (or new bin `num_items`)".
            # So for each item `k` (0 to `num_items-1` in sorted order) assigned to bin `b`:
            # `pheromones[k][b]` should receive `delta_tau` if `b` is an existing bin index, or `pheromones[k][num_items]` if it was a new bin.
            
            # We need to re-access the ant's `item_assignments` or reconstruct it.
            # The problem is that `item_assignments` stores the FINAL bin index, but the decision might have been a new bin that got shifted.
            # `packing[bin_idx].append(original_idx)` is the definitive assignment.
            # For each item `original_idx` in `packing[bin_idx]`:
            # We need to find its `item_idx_sorted`.
            # Then update `pheromones[item_idx_sorted][bin_idx]` or `pheromones[item_idx_sorted][num_items]`.
            
            # Let's trace an ant's decisions more carefully.
            # The `item_assignments` array stores the bin index (0 to num_bins_used_by_ant-1, or num_items for new) for `item_idx_sorted`.
            # We need to map `item_idx_sorted` to the actual bins used.
            # When a new bin is opened (chosen_bin_idx == num_items), the actual bin index assigned is `num_bins_used_by_ant-1`.
            # Let's rebuild the assignments for pheromone update.
            # `reconstructed_assignments[item_idx_sorted] = actual_bin_idx`
            
            # The structure `packing` has the original indices. We need to map back.
            # Create a map from `original_idx` to `item_idx_sorted`.
            original_to_sorted_idx_map = {original_indices[i]: i for i in range(num_items)}
            
            # Re-iterate through ant's assignments to update pheromones.
            # The pheromone is deposited for the *transition* or *choice* made.
            # The choice for item `item_idx_sorted` was to assign it to `chosen_bin_idx`.
            # `chosen_bin_idx` was either an existing bin index, or `num_items` for a new bin.
            # This means we need the `item_assignments` array from the ant's construction phase.
            # Let's re-run ant construction or store assignments.
            
            # To avoid re-running construction, let's use `item_assignments` created during construction.
            # But `item_assignments` stores actual bin indices. When a NEW bin is created (`num_items` choice), it's assigned an index `num_bins_used_by_ant` which then increments.
            # So, `item_assignments[item_idx_sorted]` can be `0, 1, ..., k-1` (existing bins) or `num_items` (new bin).
            # If it's `num_items`, it maps to the `num_bins_used_by_ant-1` bin index for the ant's packing.
            
            # Correct way to get assignments for this ant:
            # The loop `for item_idx_sorted in range(num_items)` inside ant construction is what we need.
            # We need to store `item_assignments` for EACH ant.
            # Let's modify the ant loop to store `item_assignments` for each ant.
            # `ant_solution_details.append({'packing': final_packing, 'bin_weights': final_bin_weights, 'assignments': item_assignments})`
            # Then iterate over these stored details.
            
            # For now, let's assume we can access `item_assignments` as generated for the current ant.
            # `item_assignments` is local to the ant loop. Need to store it.
            # Storing `item_assignments` for each ant's solution.
            current_ant_assignments_list = []
            # Inside ant loop:
            #   ... build solution ...
            #   current_ant_assignments_list.append(item_assignments)
            
            # This implies changing ant loop to store assignments:
            # ant_solutions_data = []
            # for ant in range(num_ants):
            #    ... construct ...
            #    ant_solutions_data.append({'packing': final_packing, 'bin_weights': final_bin_weights, 'num_bins': num_bins_used_by_ant, 'assignments': item_assignments})
            
            # And then loop through `ant_solutions_data` for pheromone update.
            # `for ant_data in ant_solutions_data:`
            #   `num_bins = ant_data['num_bins']`
            #   `delta_tau = pheromone_deposit_weight / num_bins`
            #   `assignments = ant_data['assignments']`
            #   `for item_idx_sorted in range(num_items):
            #       assigned_bin_option_idx = assignments[item_idx_sorted]
            #       if assigned_bin_option_idx != -1: # Should always be assigned if item_idx_sorted < num_items
            #           # actual bin index is either assigned_bin_option_idx (if < num_items) or num_bins-1 (if was new bin)
            #           # the pheromone deposit is on the choice, not the actual bin index itself.
            #           # pheromones[item_idx_sorted][chosen_bin_option_idx]
            #           pheromones[item_idx_sorted][assigned_bin_option_idx] += delta_tau

    # Re-implementing the ant construction and pheromone update logic to store assignments
    
    # Re-initializing for clarity and correctness in loop:
    best_packing_orig_indices = None
    best_bin_weights = None
    min_bins_used = float('inf')
    
    # Initialize pheromones
    max_possible_bins = num_items
    pheromones = [[initial_pheromone for _ in range(max_possible_bins + 1)] for _ in range(num_items)]

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
                    eta_new = heuristic_value(item_weight, bin_capacity) # Heuristic for placing item in a fresh bin
                    probability_new = (tau_new ** alpha) * (eta_new ** beta)
                    possible_bin_choices_with_probs.append((num_items, probability_new)) # num_items represents the new bin option
                    total_probability_mass += probability_new

                chosen_bin_option_idx = -1
                if total_probability_mass > 1e-9:
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
                if chosen_bin_option_idx == -1:
                    chosen_bin_option_idx = num_items

                # Assign item to the chosen bin and record the assignment choice
                item_assignments_for_ant[item_idx_sorted] = chosen_bin_option_idx

                if chosen_bin_option_idx == num_items: # New bin created
                    actual_bin_idx_assigned = current_bin_idx_counter
                    packing[actual_bin_idx_assigned].append(original_idx)
                    bin_current_capacities[actual_bin_idx_assigned] = bin_capacity - item_weight
                    current_bin_idx_counter += 1
                    num_bins_used_by_ant += 1
                else: # Existing bin
                    actual_bin_idx_assigned = chosen_bin_option_idx
                    packing[actual_bin_idx_assigned].append(original_idx)
                    bin_current_capacities[actual_bin_idx_assigned] -= item_weight
            
            # Store ant's solution details
            final_packing = [bin_items for bin_items in packing if bin_items]
            final_bin_weights = [sum(weights[i] for i in bin_items) for bin_items in final_packing]
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
            if num_bins == 0: continue

            delta_tau = pheromone_deposit_weight / num_bins # Pheromone deposit for this ant's solution quality
            assignments = ant_data['assignments'] # List of chosen bin_option_idx for each item_idx_sorted
            
            for item_idx_sorted in range(num_items):
                chosen_bin_option_idx = assignments[item_idx_sorted]
                if chosen_bin_option_idx != -1: # If the item was assigned
                    # Deposit pheromone on the *choice* made by the ant
                    # `chosen_bin_option_idx` is the index into the second dimension of `pheromones`.
                    # This index is `bin_idx_actual` for existing bins, or `num_items` for a new bin.
                    pheromones[item_idx_sorted][chosen_bin_option_idx] += delta_tau

    # Return the best solution found
    # Ensure we return a valid dict even if no items were processed or no solution was found.
    # If min_bins_used is still infinity, it means no solution was generated (e.g. num_items=0, which is handled).
    # For safety, if no best solution was set, create an empty one.
    if best_packing_orig_indices is None:
        return {"packing": [], "bin_weights": []}

    return {
        "packing": best_packing_orig_indices,
        "bin_weights": best_bin_weights
    }
