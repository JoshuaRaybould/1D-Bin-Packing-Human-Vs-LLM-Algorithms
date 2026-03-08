def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    import random
    import math
    import bisect

    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Preprocessing: separate zero-weight and oversized items
    zero_items = []
    oversized_items = []
    normal_items = []  # (index, weight)
    for i in range(n):
        w = weights[i]
        if w <= 0:
            zero_items.append(i)
        elif w > bin_capacity:
            oversized_items.append(i)
        else:
            normal_items.append((i, w))

    # Sort normal items by weight descending
    normal_items.sort(key=lambda x: x[1], reverse=True)
    normal_indices = [x[0] for x in normal_items]
    normal_weights = [x[1] for x in normal_items]  # aligned with normal_indices
    nn = len(normal_items)  # number of normal items

    # L2 lower bound
    total_weight = sum(w for _, w in normal_items)
    lb = max(1, math.ceil(total_weight / bin_capacity)) if bin_capacity > 0 else nn

    # Map from original index to position in normal_items
    # We'll work with positions 0..nn-1 internally

    def build_result(packing_positions):
        """Convert internal packing (positions) to external packing (original indices)"""
        result_bins = []
        # First handle oversized items
        for oi in oversized_items:
            result_bins.append([oi])
        # Then normal bins
        for bin_pos_list in packing_positions:
            result_bins.append([normal_indices[p] for p in bin_pos_list])
        # Add zero items to first bin
        if zero_items:
            if result_bins:
                result_bins[0].extend(zero_items)
            else:
                result_bins.append(list(zero_items))
        return result_bins

    if nn == 0:
        packing = build_result([])
        bin_weights = [sum(weights[i] for i in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}

    # FFD and BFD on normal items (using positions)
    def first_fit_decreasing():
        bins_rem = []
        bins_items = []
        for p in range(nn):
            w = normal_weights[p]
            placed = False
            for b in range(len(bins_rem)):
                if bins_rem[b] >= w:
                    bins_rem[b] -= w
                    bins_items[b].append(p)
                    placed = True
                    break
            if not placed:
                bins_rem.append(bin_capacity - w)
                bins_items.append([p])
        return bins_items

    def best_fit_decreasing():
        bins_rem = []
        bins_items = []
        for p in range(nn):
            w = normal_weights[p]
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bins_rem)):
                if bins_rem[b] >= w and bins_rem[b] < best_remaining:
                    best_bin = b
                    best_remaining = bins_rem[b]
            if best_bin == -1:
                bins_rem.append(bin_capacity - w)
                bins_items.append([p])
            else:
                bins_rem[best_bin] -= w
                bins_items[best_bin].append(p)
        return bins_items

    ffd = first_fit_decreasing()
    bfd = best_fit_decreasing()
    if len(ffd) <= len(bfd):
        best_packing = ffd
    else:
        best_packing = bfd
    best_num_bins = len(best_packing)

    if best_num_bins <= lb or nn <= 1:
        packing = build_result(best_packing)
        bin_weights = [sum(weights[i] for i in b) for b in packing]
        return {"packing": packing, "bin_weights": bin_weights}

    # ACO parameters
    alpha = 1.0
    beta = 3.0
    rho = 0.08
    q0 = 0.85
    tau_init = 1.0

    # MMAS bounds
    def compute_bounds(best_bins):
        t_max = 1.0 / (rho * best_bins) if best_bins > 0 else 10.0
        t_min = t_max / (2.0 * nn) if nn > 0 else 0.01
        return t_min, t_max

    tau_min, tau_max = compute_bounds(best_num_bins)
    tau_init = min(max(tau_init, tau_min), tau_max)

    use_sparse = nn > 400

    if use_sparse:
        tau = {}

        def get_tau(i, j):
            if i == j:
                return 0.0
            if i > j:
                i, j = j, i
            return tau.get((i, j), tau_init)

        def set_tau(i, j, val):
            if i == j:
                return
            if i > j:
                i, j = j, i
            tau[(i, j)] = val

        def add_tau(i, j, amount):
            if i == j:
                return
            if i > j:
                i, j = j, i
            key = (i, j)
            old = tau.get(key, tau_init)
            tau[key] = min(tau_max, old + amount)

        def evaporate_all():
            nonlocal tau
            keys_to_delete = []
            for key, val in tau.items():
                new_val = val * (1 - rho)
                if new_val <= tau_min:
                    keys_to_delete.append(key)
                else:
                    tau[key] = new_val
            for key in keys_to_delete:
                del tau[key]

        def reset_pheromone():
            nonlocal tau
            tau.clear()
    else:
        # Flat list for n<=400
        tau_flat = [tau_init] * (nn * nn)

        def get_tau(i, j):
            if i == j:
                return 0.0
            return tau_flat[i * nn + j]

        def set_tau(i, j, val):
            if i == j:
                return
            tau_flat[i * nn + j] = val
            tau_flat[j * nn + i] = val

        def add_tau(i, j, amount):
            if i == j:
                return
            val = min(tau_max, tau_flat[i * nn + j] + amount)
            tau_flat[i * nn + j] = val
            tau_flat[j * nn + i] = val

        def evaporate_all():
            factor = 1 - rho
            for i in range(nn):
                base = i * nn
                for j in range(i + 1, nn):
                    new_val = tau_flat[base + j] * factor
                    if new_val < tau_min:
                        new_val = tau_min
                    tau_flat[base + j] = new_val
                    tau_flat[j * nn + i] = new_val

        def reset_pheromone():
            for i in range(nn * nn):
                tau_flat[i] = tau_init

    def deposit_pheromone(packing, amount):
        for bin_items in packing:
            nbi = len(bin_items)
            if nbi <= 1:
                continue
            if nbi > 20:
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

    # Sorted weights for bisect (ascending) - we need weights sorted ascending for bisect
    # normal_weights is descending. We'll keep a separate ascending list for bisect.
    # Actually, for the bin-oriented construction, we maintain unassigned items sorted by weight descending.
    # We use bisect on an ascending-sorted copy to find feasible items.

    def construct_solution(use_q0):
        """Bin-oriented ACO construction."""
        # Maintain unassigned items sorted by weight ascending (for bisect)
        # Each element is (weight, position)
        unassigned_asc = sorted([(normal_weights[p], p) for p in range(nn)], key=lambda x: x[0])
        unassigned_weights_asc = [x[0] for x in unassigned_asc]  # for bisect

        bins_items = []

        while unassigned_asc:
            # Seed: pick heaviest unassigned item (last in ascending list)
            seed_w, seed_p = unassigned_asc.pop()
            unassigned_weights_asc.pop()

            current_bin = [seed_p]
            remaining = bin_capacity - seed_w

            while remaining > 0 and unassigned_asc:
                # Find feasible items: weight <= remaining
                # bisect_right gives insertion point for remaining in ascending weights
                idx_limit = bisect.bisect_right(unassigned_weights_asc, remaining)
                if idx_limit == 0:
                    break

                # Feasible items are unassigned_asc[0:idx_limit]
                # Compute scores
                num_feasible = idx_limit

                # For pheromone, sample from current_bin (limit to 8)
                if len(current_bin) <= 8:
                    bin_sample = current_bin
                else:
                    bin_sample = random.sample(current_bin, 8)

                best_score = -1.0
                best_idx = -1
                scores = []
                total_score = 0.0

                # If too many feasible items, sample a subset for roulette
                if num_feasible > 60:
                    # Always include the heaviest items (near idx_limit) and some random
                    candidate_indices = list(range(max(0, idx_limit - 30), idx_limit))
                    if idx_limit > 30:
                        extra = random.sample(range(0, idx_limit - 30), min(30, idx_limit - 30))
                        candidate_indices.extend(extra)
                else:
                    candidate_indices = list(range(num_feasible))

                for ci in candidate_indices:
                    cw, cp = unassigned_asc[ci]
                    # Pheromone component
                    ph_sum = 0.0
                    for bp in bin_sample:
                        ph_sum += get_tau(cp, bp)
                    pheromone_val = ph_sum / len(bin_sample) if bin_sample else tau_min

                    # Heuristic: candidate_weight / remaining_capacity
                    heuristic_val = cw / remaining

                    score = (pheromone_val ** alpha) * (heuristic_val ** beta)
                    scores.append((score, ci))
                    total_score += score
                    if score > best_score:
                        best_score = score
                        best_idx = ci

                if total_score <= 0 or best_idx == -1:
                    break

                # Pseudo-random proportional rule
                chosen_idx = -1
                if random.random() < use_q0:
                    # Exploitation: pick best
                    chosen_idx = best_idx
                else:
                    # Roulette wheel
                    r = random.random() * total_score
                    cumulative = 0.0
                    for score, ci in scores:
                        cumulative += score
                        if cumulative >= r:
                            chosen_idx = ci
                            break
                    if chosen_idx == -1:
                        chosen_idx = scores[-1][1]

                # Add chosen item to bin
                cw, cp = unassigned_asc[chosen_idx]
                current_bin.append(cp)
                remaining -= cw

                # Remove from unassigned (maintain sorted order)
                del unassigned_asc[chosen_idx]
                del unassigned_weights_asc[chosen_idx]

            bins_items.append(current_bin)

        return bins_items

    # Number of ants
    if nn <= 50:
        num_ants = 20
    elif nn <= 150:
        num_ants = 15
    elif nn <= 400:
        num_ants = 10
    else:
        num_ants = 5

    max_iterations = 100000
    no_improve_count = 0
    current_q0 = q0
    low_q0_remaining = 0

    for iteration in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break

        if best_num_bins <= lb:
            break

        iteration_best = None
        iteration_best_bins = float('inf')

        effective_q0 = current_q0 if low_q0_remaining <= 0 else 0.5
        if low_q0_remaining > 0:
            low_q0_remaining -= 1

        for ant in range(num_ants):
            if time.time() - start_time >= time_limit * 0.95:
                break

            packing = construct_solution(effective_q0)
            num_bins = len(packing)

            if num_bins < iteration_best_bins:
                iteration_best_bins = num_bins
                iteration_best = packing

        # Evaporate
        evaporate_all()

        # Deposit strategy
        if iteration_best is not None:
            if iteration % 5 == 0:
                # Deposit on global best (elitist)
                deposit_amount = 1.0 / best_num_bins
                deposit_pheromone(best_packing, deposit_amount)
            else:
                # Deposit on iteration best
                deposit_amount = 1.0 / iteration_best_bins
                deposit_pheromone(iteration_best, deposit_amount)

        # Update global best
        if iteration_best is not None and iteration_best_bins < best_num_bins:
            best_num_bins = iteration_best_bins
            best_packing = iteration_best
            no_improve_count = 0
            # Recompute MMAS bounds
            tau_min, tau_max = compute_bounds(best_num_bins)
        else:
            no_improve_count += 1

        # Stagnation detection and reset
        if no_improve_count > 40:
            reset_pheromone()
            deposit_pheromone(best_packing, 1.0 / best_num_bins)
            no_improve_count = 0
            low_q0_remaining = 10  # Lower q0 for 10 iterations

    # Final deterministic construction using converged pheromones
    if time.time() - start_time < time_limit * 0.98:
        final = construct_solution(1.0)  # q0=1.0, fully greedy
        if len(final) < best_num_bins:
            best_num_bins = len(final)
            best_packing = final

    # Build result
    packing = build_result(best_packing)
    bin_weights = [sum(weights[i] for i in b) for b in packing]
    return {"packing": packing, "bin_weights": bin_weights}
