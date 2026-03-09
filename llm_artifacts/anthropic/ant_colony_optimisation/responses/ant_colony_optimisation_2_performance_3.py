def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    import time
    import random
    import math

    start_time = time.time()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Preprocessing
    zero_items = []
    oversized_items = []
    normal_items = []
    for i in range(n):
        w = weights[i]
        if w <= 0:
            zero_items.append(i)
        elif w > bin_capacity:
            oversized_items.append(i)
        else:
            normal_items.append((i, w))

    # Sort by weight descending
    normal_items.sort(key=lambda x: x[1], reverse=True)
    normal_indices = [x[0] for x in normal_items]
    normal_weights = [x[1] for x in normal_items]
    nn = len(normal_items)

    total_weight = sum(normal_weights)
    lb = max(1, math.ceil(total_weight / bin_capacity)) if bin_capacity > 0 else nn

    def build_result(packing_positions):
        result_bins = []
        for oi in oversized_items:
            result_bins.append([oi])
        for bin_pos_list in packing_positions:
            result_bins.append([normal_indices[p] for p in bin_pos_list])
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

    # FFD
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

    # BFD
    def best_fit_decreasing():
        bins_rem = []
        bins_items = []
        for p in range(nn):
            w = normal_weights[p]
            best_bin = -1
            best_remaining = bin_capacity + 1
            for b in range(len(bins_rem)):
                if bins_rem[b] >= w and bins_rem[b] - w < best_remaining:
                    best_bin = b
                    best_remaining = bins_rem[b] - w
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
    beta = 2.5
    rho = 0.05
    q0 = 0.9

    def compute_bounds(best_bins):
        t_max = 1.0 / (rho * best_bins) if best_bins > 0 else 10.0
        t_min = t_max / (2.0 * nn) if nn > 0 else 0.01
        return t_min, t_max

    tau_min, tau_max = compute_bounds(best_num_bins)
    tau_init = tau_max

    # For large nn, use sparse pheromone; for small, use flat array
    # Threshold based on memory: nn*nn for nn=500 is 250K entries which is fine
    use_sparse = nn > 500

    if use_sparse:
        tau = {}
        _tau_init = tau_init

        def get_tau_val(i, j):
            if i > j:
                i, j = j, i
            return tau.get(i * nn + j, _tau_init)

        def evaporate_and_deposit(best_pack, best_bins, iter_pack, iter_bins, iteration):
            nonlocal tau, _tau_init, tau_min, tau_max
            factor = 1 - rho
            new_tau = {}
            for key, val in tau.items():
                new_val = val * factor
                if new_val > tau_min:
                    new_tau[key] = new_val
            tau = new_tau
            _tau_init = max(tau_min, min(tau_max, _tau_init * factor))

            # Deposit
            if iteration % 3 == 0:
                dep_pack = best_pack
                dep_amount = 1.0 / best_bins
            else:
                dep_pack = iter_pack
                dep_amount = 1.0 / iter_bins

            for bin_items in dep_pack:
                nbi = len(bin_items)
                if nbi <= 1:
                    continue
                if nbi <= 25:
                    for ii in range(nbi):
                        a = bin_items[ii]
                        for jj in range(ii + 1, nbi):
                            b = bin_items[jj]
                            if a > b:
                                a, b = b, a
                            key = a * nn + b
                            old = tau.get(key, _tau_init)
                            tau[key] = min(tau_max, old + dep_amount)
                else:
                    # Sample pairs
                    for _ in range(min(300, nbi * (nbi - 1) // 2)):
                        ii = random.randint(0, nbi - 1)
                        jj = random.randint(0, nbi - 2)
                        if jj >= ii:
                            jj += 1
                        a, b = bin_items[ii], bin_items[jj]
                        if a > b:
                            a, b = b, a
                        key = a * nn + b
                        old = tau.get(key, _tau_init)
                        tau[key] = min(tau_max, old + dep_amount)

        def reset_pheromone():
            nonlocal tau, _tau_init
            tau.clear()
            _tau_init = tau_max

    else:
        tau_flat = [tau_init] * (nn * nn)

        def get_tau_val(i, j):
            return tau_flat[i * nn + j]

        def evaporate_and_deposit(best_pack, best_bins, iter_pack, iter_bins, iteration):
            nonlocal tau_flat
            factor = 1 - rho
            for i in range(nn):
                base = i * nn
                for j in range(i + 1, nn):
                    new_val = tau_flat[base + j] * factor
                    if new_val < tau_min:
                        new_val = tau_min
                    elif new_val > tau_max:
                        new_val = tau_max
                    tau_flat[base + j] = new_val
                    tau_flat[j * nn + i] = new_val

            if iteration % 3 == 0:
                dep_pack = best_pack
                dep_amount = 1.0 / best_bins
            else:
                dep_pack = iter_pack
                dep_amount = 1.0 / iter_bins

            for bin_items in dep_pack:
                nbi = len(bin_items)
                if nbi <= 1:
                    continue
                if nbi <= 25:
                    for ii in range(nbi):
                        a = bin_items[ii]
                        a_base = a * nn
                        for jj in range(ii + 1, nbi):
                            b = bin_items[jj]
                            val = min(tau_max, tau_flat[a_base + b] + dep_amount)
                            tau_flat[a_base + b] = val
                            tau_flat[b * nn + a] = val
                else:
                    for _ in range(min(300, nbi * (nbi - 1) // 2)):
                        ii = random.randint(0, nbi - 1)
                        jj = random.randint(0, nbi - 2)
                        if jj >= ii:
                            jj += 1
                        a, b = bin_items[ii], bin_items[jj]
                        val = min(tau_max, tau_flat[a * nn + b] + dep_amount)
                        tau_flat[a * nn + b] = val
                        tau_flat[b * nn + a] = val

        def reset_pheromone():
            nonlocal tau_flat
            for i in range(nn * nn):
                tau_flat[i] = tau_max

    # Initial deposit
    dep_amount = 1.0 / best_num_bins
    if not use_sparse:
        for bin_items in best_packing:
            nbi = len(bin_items)
            if nbi <= 1:
                continue
            for ii in range(nbi):
                a = bin_items[ii]
                for jj in range(ii + 1, nbi):
                    b = bin_items[jj]
                    val = min(tau_max, tau_flat[a * nn + b] + dep_amount)
                    tau_flat[a * nn + b] = val
                    tau_flat[b * nn + a] = val

    # Precompute heuristic values: weight/bin_capacity as base
    # We'll compute heuristic dynamically as w/remaining

    # Pre-sort items by weight descending with indices for construction
    # normal_weights[p] is weight of position p, already sorted descending

    # For construction, we iterate items from heaviest to lightest
    # Use a boolean used array and iterate

    _nw = normal_weights  # local reference
    _bc = bin_capacity
    _get_tau = get_tau_val
    _random = random.random
    _randint = random.randint

    def construct_solution(eff_q0):
        used = [False] * nn
        remaining_count = nn
        bins_items = []

        # Precompute sorted indices by weight descending (already 0..nn-1)
        # We iterate in order 0..nn-1 which is heaviest first

        while remaining_count > 0:
            # Find heaviest unused item as seed
            seed = -1
            for p in range(nn):
                if not used[p]:
                    seed = p
                    break
            if seed == -1:
                break

            used[seed] = True
            remaining_count -= 1
            current_bin = [seed]
            remaining = _bc - _nw[seed]

            if remaining <= 0:
                bins_items.append(current_bin)
                continue

            # Try to fill this bin
            while remaining > 0 and remaining_count > 0:
                # Find feasible candidates
                # We need items with weight <= remaining that are not used
                # Iterate from heaviest to lightest, collect candidates
                candidates = []
                cand_scores = []
                total_score = 0.0
                best_score = -1.0
                best_cand = -1

                # For pheromone, sample from current bin
                cb_len = len(current_bin)
                if cb_len <= 6:
                    bin_sample = current_bin
                    bs_len = cb_len
                else:
                    bin_sample = random.sample(current_bin, 6)
                    bs_len = 6

                # Collect feasible candidates
                max_cands = 50
                found = 0
                for p in range(nn):
                    if used[p]:
                        continue
                    w = _nw[p]
                    if w > remaining:
                        continue
                    # Compute score
                    ph_sum = 0.0
                    for bp in bin_sample:
                        ph_sum += _get_tau(p, bp)
                    pheromone_val = ph_sum / bs_len if bs_len > 0 else 1.0

                    heuristic_val = w / remaining
                    score = (pheromone_val ** alpha) * (heuristic_val ** beta)

                    candidates.append(p)
                    cand_scores.append(score)
                    total_score += score
                    if score > best_score:
                        best_score = score
                        best_cand = len(candidates) - 1
                    found += 1
                    if found >= max_cands:
                        break

                if not candidates:
                    break

                if total_score <= 0:
                    break

                # Selection
                if _random() < eff_q0:
                    chosen = best_cand
                else:
                    r = _random() * total_score
                    cumulative = 0.0
                    chosen = len(candidates) - 1
                    for ci in range(len(candidates)):
                        cumulative += cand_scores[ci]
                        if cumulative >= r:
                            chosen = ci
                            break

                cp = candidates[chosen]
                used[cp] = True
                remaining_count -= 1
                current_bin.append(cp)
                remaining -= _nw[cp]

            bins_items.append(current_bin)

        return bins_items

    # Faster construction for large instances - limit pheromone lookups
    def construct_solution_fast(eff_q0):
        used = bytearray(nn)
        remaining_count = nn
        bins_items = []

        while remaining_count > 0:
            seed = -1
            for p in range(nn):
                if not used[p]:
                    seed = p
                    break
            if seed == -1:
                break

            used[seed] = 1
            remaining_count -= 1
            current_bin = [seed]
            remaining = _bc - _nw[seed]

            if remaining <= 0:
                bins_items.append(current_bin)
                continue

            while remaining > 0 and remaining_count > 0:
                cb_len = len(current_bin)
                if cb_len <= 4:
                    bin_sample = current_bin
                    bs_len = cb_len
                else:
                    bin_sample = [current_bin[0], current_bin[-1], current_bin[cb_len >> 1]]
                    bs_len = 3

                best_score = -1.0
                best_p = -1
                candidates = []
                cand_scores = []
                total_score = 0.0
                found = 0
                max_cands = 40

                for p in range(nn):
                    if used[p]:
                        continue
                    w = _nw[p]
                    if w > remaining:
                        continue

                    ph_sum = 0.0
                    for bp in bin_sample:
                        if p < bp:
                            ph_sum += _get_tau(p, bp)
                        else:
                            ph_sum += _get_tau(bp, p)
                    pheromone_val = ph_sum / bs_len

                    heuristic_val = w / remaining
                    score = pheromone_val * (heuristic_val * heuristic_val * math.sqrt(heuristic_val))  # beta~2.5

                    candidates.append(p)
                    cand_scores.append(score)
                    total_score += score
                    if score > best_score:
                        best_score = score
                        best_p = len(candidates) - 1
                    found += 1
                    if found >= max_cands:
                        break

                if not candidates:
                    break

                if total_score <= 0:
                    break

                if _random() < eff_q0:
                    chosen = best_p
                else:
                    r = _random() * total_score
                    cumulative = 0.0
                    chosen = len(candidates) - 1
                    for ci in range(len(candidates)):
                        cumulative += cand_scores[ci]
                        if cumulative >= r:
                            chosen = ci
                            break

                cp = candidates[chosen]
                used[cp] = 1
                remaining_count -= 1
                current_bin.append(cp)
                remaining -= _nw[cp]

            bins_items.append(current_bin)

        return bins_items

    # Choose construction method
    construct = construct_solution_fast if nn > 200 else construct_solution

    # Adaptive ant count
    if nn <= 50:
        num_ants = 25
    elif nn <= 150:
        num_ants = 15
    elif nn <= 400:
        num_ants = 8
    else:
        num_ants = 5

    no_improve_count = 0
    iteration = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit * 0.95:
            break
        if best_num_bins <= lb:
            break

        iteration_best = None
        iteration_best_bins = float('inf')

        # Adaptive q0
        if no_improve_count > 30:
            eff_q0 = 0.5
        elif no_improve_count > 15:
            eff_q0 = 0.7
        else:
            eff_q0 = q0

        for ant in range(num_ants):
            if time.time() - start_time >= time_limit * 0.93:
                break

            packing = construct(eff_q0)
            num_bins = len(packing)

            if num_bins < iteration_best_bins:
                iteration_best_bins = num_bins
                iteration_best = packing

        if iteration_best is None:
            break

        # Evaporate and deposit
        evaporate_and_deposit(best_packing, best_num_bins, iteration_best, iteration_best_bins, iteration)

        # Update global best
        if iteration_best_bins < best_num_bins:
            best_num_bins = iteration_best_bins
            best_packing = iteration_best
            no_improve_count = 0
            tau_min, tau_max = compute_bounds(best_num_bins)
        else:
            no_improve_count += 1

        # Stagnation reset
        if no_improve_count >= 50:
            reset_pheromone()
            no_improve_count = 0

        iteration += 1

    # Final greedy pass
    if time.time() - start_time < time_limit * 0.97:
        final = construct(1.0)
        if len(final) < best_num_bins:
            best_num_bins = len(final)
            best_packing = final

    packing = build_result(best_packing)
    bin_weights = [sum(weights[i] for i in b) for b in packing]
    return {"packing": packing, "bin_weights": bin_weights}
