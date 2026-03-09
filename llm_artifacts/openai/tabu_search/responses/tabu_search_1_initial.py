import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    C = bin_capacity

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # --- Helpers ---
    def time_exceeded() -> bool:
        return (time.time() - start) >= time_limit

    def ffd_initial() -> Tuple[List[List[int]], List[int], List[int]]:
        # First-Fit Decreasing with a best-fit tie among feasible bins.
        order = sorted(range(n), key=lambda i: (-weights[i], i))
        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        for i in order:
            w = weights[i]
            best_b = -1
            best_rem = None
            for b, load in enumerate(loads):
                if load + w <= C:
                    rem = C - (load + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_b = b
            if best_b == -1:
                best_b = len(bins)
                bins.append([])
                loads.append(0)
            bins[best_b].append(i)
            loads[best_b] += w
            assign[i] = best_b
        return bins, loads, assign

    def objective(num_bins: int, loads: List[int]) -> int:
        # Secondary: sum of squared slack to encourage tight packing.
        # Use integer objective: (num_bins * BIG) + slack_penalty
        BIG = 10**9
        slack_pen = 0
        for ld in loads:
            s = C - ld
            slack_pen += s * s
        return num_bins * BIG + slack_pen

    def cleanup_empty(bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        # Remove empty bins and reindex assignments.
        mapping = {}
        new_bins = []
        new_loads = []
        for b, items in enumerate(bins):
            if items:
                mapping[b] = len(new_bins)
                new_bins.append(items)
                new_loads.append(loads[b])
        for i in range(n):
            if assign[i] != -1:
                assign[i] = mapping[assign[i]]
        bins[:] = new_bins
        loads[:] = new_loads

    def apply_reloc(i: int, b_from: int, b_to: int,
                    bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        # Remove i from b_from
        bins[b_from].remove(i)
        loads[b_from] -= weights[i]
        # Add to b_to
        bins[b_to].append(i)
        loads[b_to] += weights[i]
        assign[i] = b_to

    def apply_swap(i: int, j: int, bi: int, bj: int,
                   bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        # Swap items i and j between bins bi and bj
        bins[bi].remove(i)
        bins[bj].remove(j)
        bins[bi].append(j)
        bins[bj].append(i)
        loads[bi] += weights[j] - weights[i]
        loads[bj] += weights[i] - weights[j]
        assign[i], assign[j] = bj, bi

    # --- Initialize ---
    bins, loads, assign = ffd_initial()
    cleanup_empty(bins, loads, assign)

    best_bins = [lst[:] for lst in bins]
    best_loads = loads[:]
    best_assign = assign[:]
    best_obj = objective(len(best_bins), best_loads)

    # Tabu structures
    # reloc tabu: (item, dest_bin) -> expiry
    # swap tabu: (min(i,j), max(i,j)) -> expiry
    tabu_reloc: Dict[Tuple[int, int], int] = {}
    tabu_swap: Dict[Tuple[int, int], int] = {}

    # Parameters
    # Tenure scaled by size, with mild randomness
    base_tenure = max(5, int(0.6 * (n ** 0.5)))

    # Candidate sampling sizes (keep iteration cost bounded)
    max_bins_sample = 25
    max_items_sample = 80

    # Fixed number of iterations (still time-checked)
    max_iter = max(500, 80 * n)

    it = 0
    while it < max_iter and not time_exceeded():
        it += 1
        m = len(bins)
        if m <= 1:
            break

        # Periodically purge expired tabu entries
        if it % 50 == 0:
            tabu_reloc = {k: v for k, v in tabu_reloc.items() if v > it}
            tabu_swap = {k: v for k, v in tabu_swap.items() if v > it}

        curr_obj = objective(m, loads)

        # Sample bins and items for neighborhood exploration
        # Focus more on light bins (more likely to be emptied)
        bin_indices = list(range(m))
        if m > max_bins_sample:
            # bias: pick some smallest-load bins + some random
            by_load = sorted(bin_indices, key=lambda b: loads[b])
            k_small = max_bins_sample // 2
            chosen = set(by_load[:k_small])
            while len(chosen) < max_bins_sample:
                chosen.add(random.randrange(m))
            bin_sample = list(chosen)
        else:
            bin_sample = bin_indices

        # Sample items from sampled bins
        items = []
        for b in bin_sample:
            items.extend(bins[b])
        if len(items) > max_items_sample:
            items_sample = random.sample(items, max_items_sample)
        else:
            items_sample = items

        best_move = None
        best_move_obj = None

        # --- Evaluate relocate moves ---
        # Try moving items out of bins, possibly emptying a bin.
        for i_item in items_sample:
            b_from = assign[i_item]
            w = weights[i_item]
            # Candidate destination bins: sampled + also try best-fit bins globally with some chance
            dest_candidates = bin_sample
            if m > len(bin_sample) and random.random() < 0.25:
                # add a few random bins not in sample
                for _ in range(5):
                    dest_candidates = dest_candidates + [random.randrange(m)]
            # Unique
            if len(dest_candidates) > 1:
                dest_candidates = list(dict.fromkeys(dest_candidates))

            for b_to in dest_candidates:
                if b_to == b_from:
                    continue
                if loads[b_to] + w > C:
                    continue

                # Compute effect
                new_load_from = loads[b_from] - w
                new_load_to = loads[b_to] + w

                new_m = m - 1 if new_load_from == 0 else m

                # New slack penalty delta
                s_from_old = C - loads[b_from]
                s_from_new = C - new_load_from
                s_to_old = C - loads[b_to]
                s_to_new = C - new_load_to
                slack_delta = (s_from_new * s_from_new - s_from_old * s_from_old) + (s_to_new * s_to_new - s_to_old * s_to_old)

                BIG = 10**9
                cand_obj = new_m * BIG + (curr_obj - m * BIG) + slack_delta

                # Tabu check
                is_tabu = tabu_reloc.get((i_item, b_to), 0) > it
                aspiration = cand_obj < best_obj
                if is_tabu and not aspiration:
                    continue

                if best_move_obj is None or cand_obj < best_move_obj:
                    best_move_obj = cand_obj
                    best_move = ("reloc", i_item, b_from, b_to)

        # --- Evaluate swap moves (between sampled bins) ---
        # Swap can help reshuffle without increasing bins.
        # We'll sample some pairs by picking random counterpart items.
        if not time_exceeded() and items_sample:
            num_swap_trials = min(200, 5 * len(items_sample))
            for _ in range(num_swap_trials):
                i = random.choice(items_sample)
                bi = assign[i]
                # choose j from different bin
                bj = random.choice(bin_sample)
                if bj == bi or not bins[bj]:
                    continue
                j = random.choice(bins[bj])
                w_i = weights[i]
                w_j = weights[j]

                # Feasibility
                if loads[bi] - w_i + w_j > C:
                    continue
                if loads[bj] - w_j + w_i > C:
                    continue

                new_load_bi = loads[bi] - w_i + w_j
                new_load_bj = loads[bj] - w_j + w_i

                # bin count unchanged by swap
                s_bi_old = C - loads[bi]
                s_bi_new = C - new_load_bi
                s_bj_old = C - loads[bj]
                s_bj_new = C - new_load_bj
                slack_delta = (s_bi_new * s_bi_new - s_bi_old * s_bi_old) + (s_bj_new * s_bj_new - s_bj_old * s_bj_old)

                cand_obj = curr_obj + slack_delta

                key = (i, j) if i < j else (j, i)
                is_tabu = tabu_swap.get(key, 0) > it
                aspiration = cand_obj < best_obj
                if is_tabu and not aspiration:
                    continue

                if best_move_obj is None or cand_obj < best_move_obj:
                    best_move_obj = cand_obj
                    best_move = ("swap", i, j, bi, bj)

        # If no move found (can happen with restrictive sampling), broaden once
        if best_move is None and not time_exceeded():
            # Try a quick greedy attempt to empty a light bin
            light_bins = sorted(range(m), key=lambda b: loads[b])
            tried = 0
            for b_from in light_bins[: min(5, m)]:
                for i_item in list(bins[b_from]):
                    w = weights[i_item]
                    # best-fit destination
                    best_b = -1
                    best_rem = None
                    for b_to in range(m):
                        if b_to == b_from:
                            continue
                        if loads[b_to] + w <= C:
                            rem = C - (loads[b_to] + w)
                            if best_rem is None or rem < best_rem:
                                best_rem = rem
                                best_b = b_to
                    if best_b != -1:
                        best_move = ("reloc", i_item, b_from, best_b)
                        best_move_obj = None
                        tried = 1
                        break
                if tried:
                    break

        if best_move is None:
            # Nothing to do
            continue

        # --- Apply best move ---
        if best_move[0] == "reloc":
            _, i_item, b_from, b_to = best_move
            apply_reloc(i_item, b_from, b_to, bins, loads, assign)

            # If emptied a bin, cleanup
            if loads[b_from] == 0:
                cleanup_empty(bins, loads, assign)

            # Set tabu
            tenure = base_tenure + random.randint(0, base_tenure)
            # forbid moving this item back to the source bin for a while
            if b_from < len(bins):
                tabu_reloc[(i_item, b_from)] = it + tenure

        else:
            _, i, j, bi, bj = best_move
            apply_swap(i, j, bi, bj, bins, loads, assign)
            tenure = base_tenure + random.randint(0, base_tenure)
            key = (i, j) if i < j else (j, i)
            tabu_swap[key] = it + tenure

        # --- Update best ---
        curr_obj2 = objective(len(bins), loads)
        if curr_obj2 < best_obj:
            best_obj = curr_obj2
            best_bins = [lst[:] for lst in bins]
            best_loads = loads[:]
            best_assign = assign[:]

        # Mild diversification if stuck: random swap within feasibility
        if it % 200 == 0 and not time_exceeded():
            for _ in range(30):
                b1 = random.randrange(len(bins))
                b2 = random.randrange(len(bins))
                if b1 == b2 or not bins[b1] or not bins[b2]:
                    continue
                i = random.choice(bins[b1])
                j = random.choice(bins[b2])
                wi, wj = weights[i], weights[j]
                if loads[b1] - wi + wj <= C and loads[b2] - wj + wi <= C:
                    apply_swap(i, j, b1, b2, bins, loads, assign)
                    break

    # Return best found
    # Ensure alignment and correct weights
    packing = [lst[:] for lst in best_bins]
    bin_weights = best_loads[:]

    # Final sanity: remove any empty bins (shouldn't exist)
    filtered_packing = []
    filtered_weights = []
    for b, items in enumerate(packing):
        if items:
            filtered_packing.append(items)
            filtered_weights.append(sum(weights[i] for i in items))

    return {"packing": filtered_packing, "bin_weights": filtered_weights}
