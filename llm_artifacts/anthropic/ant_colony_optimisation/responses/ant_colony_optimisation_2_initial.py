def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    import random

    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Filter out zero-weight items and items exceeding capacity
    # Sort items by weight descending for FFD-style construction
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)

    # Best Fit Decreasing as initial solution
    def best_fit_decreasing():
        bins_rem = []  # remaining capacity
        bins_items = []  # item indices
        for idx in sorted_indices:
            w = weights[idx]
            if w > bin_capacity:
                # Item can't fit; place alone (infeasible but must handle)
                bins_rem.append(bin_capacity - w)
                bins_items.append([idx])
                continue
            if w == 0:
                # Put zero-weight items in first bin or create one
                if bins_items:
                    bins_items[0].append(idx)
                else:
                    bins_rem.append(bin_capacity)
                    bins_items.append([idx])
                continue
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bins_rem)):
                if bins_rem[b] >= w and bins_rem[b] < best_remaining:
                    best_bin = b
                    best_remaining = bins_rem[b]
            if best_bin == -1:
                bins_rem.append(bin_capacity - w)
                bins_items.append([idx])
            else:
                bins_rem[best_bin] -= w
                bins_items[best_bin].append(idx)
        return bins_items

    initial_packing = best_fit_decreasing()
    best_packing = initial_packing
    best_num_bins = len(initial_packing)

    if n <= 1 or best_num_bins <= 1:
        bin_weights = [sum(weights[i] for i in b) for b in best_packing]
        return {"packing": best_packing, "bin_weights": bin_weights}

    # Decide approach based on problem size
    # For large n, pheromone matrix is too big, use sparse
    use_sparse = n > 300

    # ACO parameters
    alpha = 1.0
    beta = 2.5
    rho = 0.1
    tau_min = 0.1
    tau_max = 10.0
    tau_init = 1.0

    if use_sparse:
        tau = {}

        def get_tau(i, j):
            if i == j:
                return 0.0
            if i > j:
                i, j = j, i
            return tau.get((i, j), tau_init)

        def add_tau(i, j, amount):
            if i == j:
                return
            if i > j:
                i, j = j, i
            key = (i, j)
            old = tau.get(key, tau_init)
            tau[key] = min(tau_max, old + amount)

        def evaporate():
            nonlocal tau
            keys_to_delete = []
            for key, val in tau.items():
                new_val = val * (1 - rho)
                if new_val < tau_min + 0.05:
                    keys_to_delete.append(key)
                else:
                    tau[key] = new_val
            for key in keys_to_delete:
                del tau[key]

        def reset_pheromone():
            nonlocal tau
            tau.clear()
    else:
        tau_matrix = [[tau_init] * n for _ in range(n)]

        def get_tau(i, j):
            return tau_matrix[i][j]

        def add_tau(i, j, amount):
            if i == j:
                return
            val = min(tau_max, tau_matrix[i][j] + amount)
            tau_matrix[i][j] = val
            tau_matrix[j][i] = val

        def evaporate():
            for i in range(n):
                for j in range(i + 1, n):
                    new_val = tau_matrix[i][j] * (1 - rho)
                    new_val = max(tau_min, new_val)
                    tau_matrix[i][j] = new_val
                    tau_matrix[j][i] = new_val

        def reset_pheromone():
            for i in range(n):
                for j in range(n):
                    tau_matrix[i][j] = tau_init

    def deposit_pheromone(packing, amount):
        for bin_items in packing:
            nbi = len(bin_items)
            if nbi <= 1:
                continue
            # For large bins, limit deposits to keep it manageable
            if nbi > 50:
                # Sample pairs
                for _ in range(min(200, nbi * (nbi - 1) // 2)):
                    ii = random.randint(0, nbi - 1)
                    jj = random.randint(0, nbi - 2)
                    if jj >= ii:
                        jj += 1
                    add_tau(bin_items[ii], bin_items[jj], amount)
            else:
                for ii in range(nbi):
                    for jj in range(ii + 1, nbi):
                        add_tau(bin_items[ii], bin_items[jj], amount)

    # Deposit from initial solution
    deposit_pheromone(best_packing, 1.0 / best_num_bins)

    # Precompute weight array for fast access
    w_arr = weights  # reference

    # Max candidate bins to evaluate per item for efficiency
    max_candidates = 50

    def construct_solution():
        bins_rem = []
        bins_items = []
        # Track which bins have the most remaining capacity for quick filtering

        for idx in sorted_indices:
            w = w_arr[idx]
            if w > bin_capacity:
                bins_rem.append(bin_capacity - w)
                bins_items.append([idx])
                continue
            if w == 0:
                if bins_items:
                    bins_items[0].append(idx)
                else:
                    bins_rem.append(bin_capacity)
                    bins_items.append([idx])
                continue

            # Find feasible bins
            num_bins = len(bins_rem)
            feasible = []
            for b in range(num_bins):
                if bins_rem[b] >= w:
                    feasible.append(b)

            # Limit candidates for efficiency
            if len(feasible) > max_candidates:
                # Prefer bins with less remaining capacity (better fit)
                feasible.sort(key=lambda b: bins_rem[b])
                feasible = feasible[:max_candidates]

            candidates = []
            for b_idx in feasible:
                rem = bins_rem[b_idx]
                items_in_bin = bins_items[b_idx]

                # Pheromone component
                if items_in_bin:
                    pheromone_sum = 0.0
                    count = min(len(items_in_bin), 20)  # limit for efficiency
                    if count < len(items_in_bin):
                        # Sample
                        for item_in_bin in random.sample(items_in_bin, count):
                            pheromone_sum += get_tau(idx, item_in_bin)
                        pheromone_val = pheromone_sum / count
                    else:
                        for item_in_bin in items_in_bin:
                            pheromone_sum += get_tau(idx, item_in_bin)
                        pheromone_val = pheromone_sum / len(items_in_bin)
                else:
                    pheromone_val = tau_min

                # Heuristic: prefer tighter fit
                fullness = (bin_capacity - rem + w) / bin_capacity
                heuristic_val = fullness

                score = (pheromone_val ** alpha) * (heuristic_val ** beta)
                candidates.append((score, b_idx))

            # Option to open new bin
            new_bin_heuristic = w / bin_capacity
            new_bin_score = (tau_min ** alpha) * (new_bin_heuristic ** beta)
            candidates.append((new_bin_score, -1))

            # Roulette wheel selection
            total_score = sum(s for s, _ in candidates)
            if total_score <= 0:
                bins_rem.append(bin_capacity - w)
                bins_items.append([idx])
                continue

            r = random.random() * total_score
            cumulative = 0.0
            chosen = -1
            for score, b_idx in candidates:
                cumulative += score
                if cumulative >= r:
                    chosen = b_idx
                    break

            if chosen == -1:
                bins_rem.append(bin_capacity - w)
                bins_items.append([idx])
            else:
                bins_rem[chosen] -= w
                bins_items[chosen].append(idx)

        return bins_items

    # Determine number of ants based on problem size
    if n <= 50:
        num_ants = 20
    elif n <= 200:
        num_ants = 15
    elif n <= 500:
        num_ants = 10
    else:
        num_ants = 5

    max_iterations = 10000
    no_improve_count = 0

    for iteration in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.92:
            break

        iteration_best = None
        iteration_best_bins = float('inf')

        for ant in range(num_ants):
            if time.time() - start_time >= time_limit * 0.92:
                break

            packing = construct_solution()
            num_bins = len(packing)

            if num_bins < iteration_best_bins:
                iteration_best_bins = num_bins
                iteration_best = packing

        # Evaporate
        evaporate()

        # Deposit on iteration best
        if iteration_best is not None:
            deposit_amount = 1.0 / iteration_best_bins
            deposit_pheromone(iteration_best, deposit_amount)

        # Update global best
        if iteration_best is not None and iteration_best_bins < best_num_bins:
            best_num_bins = iteration_best_bins
            best_packing = iteration_best
            no_improve_count = 0
            # Extra deposit for global best
            deposit_pheromone(best_packing, 2.0 / best_num_bins)
        else:
            no_improve_count += 1

        # Deposit on global best every few iterations (elitist strategy)
        if iteration % 3 == 0:
            deposit_pheromone(best_packing, 0.5 / best_num_bins)

        # Reset pheromone if stuck
        if no_improve_count > 30:
            reset_pheromone()
            deposit_pheromone(best_packing, 1.0 / best_num_bins)
            no_improve_count = 0

    bin_weights = [sum(weights[i] for i in b) for b in best_packing]
    return {"packing": best_packing, "bin_weights": bin_weights}