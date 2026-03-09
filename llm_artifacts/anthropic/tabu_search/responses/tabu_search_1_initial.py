import random
import time

def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start_time = time.time()
    n = len(weights)
    
    if n == 0:
        return {"packing": [], "bin_weights": []}
    
    # --- Initial solution using First Fit Decreasing ---
    indices_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    def first_fit_decreasing():
        bins = []  # list of (set of item indices, total weight)
        bin_items = []
        bin_weights_list = []
        for idx in indices_sorted:
            w = weights[idx]
            placed = False
            for b in range(len(bin_items)):
                if bin_weights_list[b] + w <= bin_capacity:
                    bin_items[b].append(idx)
                    bin_weights_list[b] += w
                    placed = True
                    break
            if not placed:
                bin_items.append([idx])
                bin_weights_list.append(w)
        return bin_items, bin_weights_list
    
    def best_fit_decreasing():
        bin_items = []
        bin_weights_list = []
        for idx in indices_sorted:
            w = weights[idx]
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bin_items)):
                remaining = bin_capacity - bin_weights_list[b]
                if remaining >= w and remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b
            if best_bin >= 0:
                bin_items[best_bin].append(idx)
                bin_weights_list[best_bin] += w
            else:
                bin_items.append([idx])
                bin_weights_list.append(w)
        return bin_items, bin_weights_list
    
    # Try both and pick better
    ffd_items, ffd_weights = first_fit_decreasing()
    bfd_items, bfd_weights = best_fit_decreasing()
    
    if len(bfd_items) < len(ffd_items):
        current_bins = bfd_items
        current_weights = bfd_weights
    else:
        current_bins = ffd_items
        current_weights = ffd_weights
    
    # Convert to lists of sets for faster operations
    # Represent solution as: item_to_bin[i] = bin index, bins[b] = set of items, bin_weight[b] = weight
    
    num_bins = len(current_bins)
    item_to_bin = [0] * n
    bins_sets = [set() for _ in range(num_bins)]
    bin_w = list(current_weights)
    
    for b in range(num_bins):
        for idx in current_bins[b]:
            item_to_bin[idx] = b
            bins_sets[b].add(idx)
    
    def compute_fitness(bw_list):
        """Higher is better - sum of squared fill ratios."""
        s = 0.0
        c2 = bin_capacity * bin_capacity
        for w in bw_list:
            s += w * w
        return s / c2
    
    best_num_bins = num_bins
    best_packing = [list(s) for s in bins_sets]
    best_bin_weights = list(bin_w)
    
    # Tabu list: (item, bin) -> iteration when tabu expires
    # Meaning: item cannot be moved TO this bin until the tabu expires
    tabu = {}
    tabu_tenure = max(7, n // 10)
    
    iteration = 0
    max_iterations = 10000000
    no_improve_count = 0
    
    # Active bins tracking
    active_bins = set(b for b in range(num_bins) if bin_w[b] > 0)
    
    def remove_empty_bins():
        nonlocal num_bins, active_bins
        # Just track active bins
        to_remove = [b for b in active_bins if len(bins_sets[b]) == 0]
        for b in to_remove:
            active_bins.discard(b)
        num_bins = len(active_bins)
    
    def get_sorted_bins():
        """Return bins sorted by weight (ascending) for targeting."""
        return sorted(active_bins, key=lambda b: bin_w[b])
    
    def try_empty_bin(target_bin, iteration):
        """Try to move all items from target_bin to other bins."""
        nonlocal num_bins
        items_in_target = list(bins_sets[target_bin])
        # Sort by weight descending (harder to place first)
        items_in_target.sort(key=lambda i: weights[i], reverse=True)
        
        other_bins = sorted([b for b in active_bins if b != target_bin], 
                          key=lambda b: bin_capacity - bin_w[b])  # ascending remaining
        
        moves = []  # (item, from_bin, to_bin)
        temp_bin_w = dict((b, bin_w[b]) for b in active_bins)
        
        success = True
        for item in items_in_target:
            w = weights[item]
            placed = False
            # Best fit
            best_b = -1
            best_remaining = bin_capacity + 1
            for b in other_bins:
                remaining = bin_capacity - temp_bin_w[b]
                if remaining >= w and remaining < best_remaining:
                    best_remaining = remaining
                    best_b = b
            if best_b >= 0:
                moves.append((item, target_bin, best_b))
                temp_bin_w[best_b] += w
                placed = True
            if not placed:
                success = False
                break
        
        if success:
            # Execute all moves
            for item, from_b, to_b in moves:
                bins_sets[from_b].discard(item)
                bins_sets[to_b].add(item)
                bin_w[from_b] -= weights[item]
                bin_w[to_b] += weights[item]
                item_to_bin[item] = to_b
                # Make moving back tabu
                tabu[(item, from_b)] = iteration + tabu_tenure
            active_bins.discard(target_bin)
            num_bins = len(active_bins)
            return True
        return False
    
    def try_moves_and_swaps(target_bin, iteration):
        """
        Try single item moves from target_bin, or swaps that reduce target_bin weight.
        Returns True if any improving move was made.
        """
        items_in_target = list(bins_sets[target_bin])
        other_bins_list = [b for b in active_bins if b != target_bin]
        
        best_move = None
        best_score = -1  # We want to maximize weight removed from target
        
        # Try moves: move item from target to another bin
        for item in items_in_target:
            w = weights[item]
            is_tabu_item = False
            for b in other_bins_list:
                if bin_w[b] + w <= bin_capacity:
                    # Check tabu
                    tabu_key = (item, b)
                    is_tabu = tabu_key in tabu and tabu[tabu_key] > iteration
                    
                    # Aspiration: if this could lead to emptying the bin
                    aspiration = False
                    
                    score = w  # weight removed from target
                    if not is_tabu or aspiration:
                        if score > best_score:
                            best_score = score
                            best_move = ('move', item, target_bin, b)
        
        # Try swaps: swap item from target with item from other bin
        # Net effect: remove w_target, add w_other to target
        # Want w_target - w_other > 0 (reduce target weight)
        for item_t in items_in_target:
            w_t = weights[item_t]
            for b in other_bins_list:
                remaining_b = bin_capacity - bin_w[b]
                remaining_target = bin_capacity - bin_w[target_bin]
                for item_o in list(bins_sets[b]):
                    w_o = weights[item_o]
                    # After swap: target gets w_o instead of w_t, b gets w_t instead of w_o
                    # target new weight: bin_w[target] - w_t + w_o
                    # b new weight: bin_w[b] - w_o + w_t
                    if (bin_w[target_bin] - w_t + w_o <= bin_capacity and
                        bin_w[b] - w_o + w_t <= bin_capacity):
                        net_reduction = w_t - w_o
                        if net_reduction > 0:
                            # Check tabu
                            tabu_t = (item_t, b) in tabu and tabu[(item_t, b)] > iteration
                            tabu_o = (item_o, target_bin) in tabu and tabu[(item_o, target_bin)] > iteration
                            is_tabu = tabu_t or tabu_o
                            
                            if not is_tabu:
                                if net_reduction > best_score:
                                    best_score = net_reduction
                                    best_move = ('swap', item_t, target_bin, item_o, b)
        
        if best_move is not None:
            if best_move[0] == 'move':
                _, item, from_b, to_b = best_move
                bins_sets[from_b].discard(item)
                bins_sets[to_b].add(item)
                bin_w[from_b] -= weights[item]
                bin_w[to_b] += weights[item]
                item_to_bin[item] = to_b
                tabu[(item, from_b)] = iteration + tabu_tenure
            else:
                _, item_t, bin_t, item_o, bin_o = best_move
                bins_sets[bin_t].discard(item_t)
                bins_sets[bin_o].discard(item_o)
                bins_sets[bin_t].add(item_o)
                bins_sets[bin_o].add(item_t)
                w_t = weights[item_t]
                w_o = weights[item_o]
                bin_w[bin_t] = bin_w[bin_t] - w_t + w_o
                bin_w[bin_o] = bin_w[bin_o] - w_o + w_t
                item_to_bin[item_t] = bin_o
                item_to_bin[item_o] = bin_t
                tabu[(item_t, bin_t)] = iteration + tabu_tenure
                tabu[(item_o, bin_o)] = iteration + tabu_tenure
            
            # Check if target is now empty
            if len(bins_sets[target_bin]) == 0:
                active_bins.discard(target_bin)
                return True
            return True
        return False
    
    def perturbation(iteration):
        """Random perturbation: randomly move some items."""
        active_list = list(active_bins)
        if len(active_list) < 2:
            return
        
        for _ in range(max(3, n // 20)):
            b1 = random.choice(active_list)
            if not bins_sets[b1]:
                continue
            item = random.choice(list(bins_sets[b1]))
            w = weights[item]
            
            candidates = [b for b in active_list if b != b1 and bin_w[b] + w <= bin_capacity]
            if candidates:
                b2 = random.choice(candidates)
                bins_sets[b1].discard(item)
                bins_sets[b2].add(item)
                bin_w[b1] -= w
                bin_w[b2] += w
                item_to_bin[item] = b2
                tabu[(item, b1)] = iteration + tabu_tenure
        
        # Clean up empty bins
        remove_empty_bins()
    
    def save_best():
        nonlocal best_num_bins, best_packing, best_bin_weights
        current_num = len(active_bins)
        if current_num < best_num_bins:
            best_num_bins = current_num
            best_packing = [list(bins_sets[b]) for b in active_bins]
            best_bin_weights = [bin_w[b] for b in active_bins]
            return True
        return False
    
    # Main tabu search loop
    check_interval = 100
    
    while iteration < max_iterations:
        if iteration % check_interval == 0:
            elapsed = time.time() - start_time
            if elapsed >= time_limit * 0.95:
                break
        
        iteration += 1
        
        # Sort bins by weight, target the lightest
        sorted_bins = get_sorted_bins()
        if len(sorted_bins) <= 1:
            break
        
        target = sorted_bins[0]
        
        # First try to completely empty the target bin
        if try_empty_bin(target, iteration):
            num_bins = len(active_bins)
            improved = save_best()
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
        
        # Try the next few lightest bins too
        emptied = False
        for i in range(1, min(3, len(sorted_bins))):
            target = sorted_bins[i]
            if try_empty_bin(target, iteration):
                num_bins = len(active_bins)
                save_best()
                emptied = True
                no_improve_count = 0
                break
        
        if emptied:
            continue
        
        # If can't empty, try moves/swaps to reduce lightest bin
        target = sorted_bins[0]
        moved = try_moves_and_swaps(target, iteration)
        
        if moved:
            remove_empty_bins()
            num_bins = len(active_bins)
            improved = save_best()
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # If stuck, try targeting a different bin
        if no_improve_count > 0 and no_improve_count % 5 == 0:
            # Try a random light bin
            k = min(len(sorted_bins), max(3, len(sorted_bins) // 3))
            target = sorted_bins[random.randint(0, k - 1)]
            try_moves_and_swaps(target, iteration)
            remove_empty_bins()
            save_best()
        
        # Perturbation if stuck for a while
        if no_improve_count > 50:
            perturbation(iteration)
            save_best()
            no_improve_count = 0
            # Adjust tabu tenure
            tabu_tenure = random.randint(max(5, n // 20), max(10, n // 5))
    
    return {
        "packing": best_packing,
        "bin_weights": best_bin_weights
    }