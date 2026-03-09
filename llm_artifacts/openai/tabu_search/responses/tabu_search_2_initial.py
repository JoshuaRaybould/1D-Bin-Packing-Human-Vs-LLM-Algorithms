import time
import random
from collections import deque


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    # ---------- Helpers ----------
    def now_exceeded() -> bool:
        return time.time() >= deadline

    def objective(num_bins: int, loads: list[int]) -> tuple:
        # Primary: minimize number of bins.
        # Secondary: encourage tight packing (smaller residuals), using sum of squares of residuals.
        # (For fixed num_bins, maximizing sum of squares of loads is equivalent.)
        # Use residuals to keep scale tied to capacity.
        res_sq = 0
        for L in loads:
            r = bin_capacity - L
            res_sq += r * r
        return (num_bins, res_sq)

    def build_packing_from_assign(assign: list[int], loads: list[int]):
        # compress bin ids to 0..k-1
        mapping = {}
        new_id = 0
        new_assign = [-1] * n
        for i, b in enumerate(assign):
            if b not in mapping:
                mapping[b] = new_id
                new_id += 1
            new_assign[i] = mapping[b]
        k = new_id
        pack = [[] for _ in range(k)]
        new_loads = [0] * k
        for i, b in enumerate(new_assign):
            pack[b].append(i)
            new_loads[b] += weights[i]
        return pack, new_loads, new_assign

    # Randomized Best-Fit-Decreasing (RBFD)
    def initial_solution() -> tuple[list[int], list[int]]:
        order = list(range(n))
        # Sort by weight descending; randomize within small blocks to diversify
        order.sort(key=lambda i: weights[i], reverse=True)
        # block shuffle
        block = 8
        for s in range(0, n, block):
            e = min(n, s + block)
            # keep mostly sorted but allow permutations
            if e - s > 1:
                random.shuffle(order[s:e])

        assign = [-1] * n
        loads: list[int] = []
        for i in order:
            w = weights[i]
            best_bin = -1
            best_rem = None
            # Best-fit among existing bins
            for b, L in enumerate(loads):
                if L + w <= bin_capacity:
                    rem = bin_capacity - (L + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_bin = b
            if best_bin == -1:
                best_bin = len(loads)
                loads.append(0)
            loads[best_bin] += w
            assign[i] = best_bin

        # Compress ids (already compact, but keep consistent)
        _, loads2, assign2 = build_packing_from_assign(assign, loads)
        return assign2, loads2

    # ---------- Tabu Search Core ----------
    # Tabu tenure parameters
    base_tenure = 7
    tenure_jitter = 11

    # Iteration and stagnation control
    max_iters = 200000  # fixed cap; will stop earlier by time
    check_period = 200  # time checks
    stagnation_limit = 5000

    # Create initial
    cur_assign, cur_loads = initial_solution()
    cur_k = len(cur_loads)
    cur_obj = objective(cur_k, cur_loads)

    best_assign = cur_assign[:]
    best_loads = cur_loads[:]
    best_obj = cur_obj

    # For efficiency maintain items per bin
    bin_items: list[list[int]] = [[] for _ in range(cur_k)]
    for i, b in enumerate(cur_assign):
        bin_items[b].append(i)

    # Tabu structure: (item, to_bin) -> expiration_iter
    tabu_move = {}
    # For swap we record (i, bj) and (j, bi) as tabu as well.

    # Candidate bin order helper: prefer bins with larger loads (tighter bins)
    def bins_by_preference(exclude_bin: int | None = None):
        idxs = list(range(len(cur_loads)))
        idxs.sort(key=lambda b: cur_loads[b], reverse=True)
        if exclude_bin is not None:
            idxs = [b for b in idxs if b != exclude_bin]
        return idxs

    def remove_empty_bins():
        nonlocal cur_assign, cur_loads, bin_items
        # Compress away empty bins.
        mapping = {}
        new_loads = []
        new_items = []
        for b, items in enumerate(bin_items):
            if items:
                mapping[b] = len(new_loads)
                new_loads.append(cur_loads[b])
                new_items.append(items)
        if len(new_loads) == len(cur_loads):
            return
        new_assign = [-1] * n
        for old_b, new_b in mapping.items():
            for i in bin_items[old_b]:
                new_assign[i] = new_b
        cur_assign = new_assign
        cur_loads = new_loads
        bin_items = new_items

    def compute_delta_res_sq(bin_load: int, new_load: int) -> int:
        # Change in residual squared for a bin when load changes.
        r0 = bin_capacity - bin_load
        r1 = bin_capacity - new_load
        return r1 * r1 - r0 * r0

    def try_best_move(iteration: int):
        """Find and apply the best admissible move (move or swap) from a sampled neighborhood."""
        nonlocal cur_assign, cur_loads, bin_items, cur_obj

        k = len(cur_loads)
        if k == 0:
            return False

        # Sample items biased towards troublesome bins (low residual or very high residual)
        # but keep it simple and fast.
        sample_size = min(n, 60)
        # Build a list of candidate items: from heaviest and from sparsest bins
        # Heavies:
        heavy = sorted(range(n), key=lambda i: weights[i], reverse=True)[: min(n, 30)]
        cand_items = set(heavy)
        # Add random items
        while len(cand_items) < sample_size:
            cand_items.add(random.randrange(n))
        cand_items = list(cand_items)

        best_neighbor = None  # (new_k, new_res_sq, kind, data)
        best_neighbor_obj = None

        cur_res_sq = cur_obj[1]

        # --- Move neighborhood: move item i from bi to bj (existing or new bin) ---
        # Consider moving into existing bins first; occasionally allow opening a new bin.
        allow_new_bin = (random.random() < 0.10)

        for i in cand_items:
            bi = cur_assign[i]
            wi = weights[i]

            # Candidate destination bins: top by load + a few random
            dest_bins = bins_by_preference(exclude_bin=bi)[: min(k - 1, 15)]
            # add random bins
            for _ in range(6):
                if k > 1:
                    b = random.randrange(k)
                    if b != bi:
                        dest_bins.append(b)
            # unique
            if dest_bins:
                dest_bins = list(dict.fromkeys(dest_bins))

            # existing bins
            for bj in dest_bins:
                if bj == bi:
                    continue
                if cur_loads[bj] + wi > bin_capacity:
                    continue

                # Tabu check
                exp = tabu_move.get((i, bj), -1)
                is_tabu = exp >= iteration

                # Compute objective quickly
                # k unchanged unless source becomes empty -> handled approximately by checking bin size
                new_k = k - 1 if (len(bin_items[bi]) == 1 and bi != bj) else k

                # Update residual sum squares for affected bins
                delta = 0
                # source bin
                new_load_bi = cur_loads[bi] - wi
                delta += compute_delta_res_sq(cur_loads[bi], new_load_bi)
                # dest bin
                new_load_bj = cur_loads[bj] + wi
                delta += compute_delta_res_sq(cur_loads[bj], new_load_bj)

                # If source becomes empty and we remove it, objective's residual sum squares should drop
                # its residual term; our delta above includes it (for load 0). Adjust by removing it.
                new_res_sq = cur_res_sq + delta
                if new_k < k:
                    # remove bin bi term where it would be (load 0)
                    new_res_sq -= (bin_capacity - 0) ** 2

                neigh_obj = (new_k, new_res_sq)

                # Aspiration: allow tabu if it improves global best
                if is_tabu and not (neigh_obj < best_obj):
                    continue

                if best_neighbor_obj is None or neigh_obj < best_neighbor_obj:
                    best_neighbor_obj = neigh_obj
                    best_neighbor = ("move", i, bi, bj)

            # optional: move to new bin (usually not helpful, but can escape)
            if allow_new_bin and len(bin_items[bi]) > 1:
                bj = k  # new bin id
                exp = tabu_move.get((i, bj), -1)
                is_tabu = exp >= iteration

                # would increase bins by 1
                new_k = k + 1
                # residual changes: source reduces, add new bin load=wi
                delta = 0
                new_load_bi = cur_loads[bi] - wi
                delta += compute_delta_res_sq(cur_loads[bi], new_load_bi)
                # add new bin term
                new_res_sq = cur_res_sq + delta + (bin_capacity - wi) ** 2
                neigh_obj = (new_k, new_res_sq)

                if is_tabu and not (neigh_obj < best_obj):
                    pass
                else:
                    if best_neighbor_obj is None or neigh_obj < best_neighbor_obj:
                        best_neighbor_obj = neigh_obj
                        best_neighbor = ("newbin", i, bi)

        # --- Swap neighborhood: swap items i (in bi) and j (in bj) ---
        # Sample a few pairs
        pair_trials = 120
        for _ in range(pair_trials):
            i = random.choice(cand_items)
            bi = cur_assign[i]
            if k <= 1:
                break
            bj = random.randrange(k)
            if bj == bi or not bin_items[bj]:
                continue
            j = random.choice(bin_items[bj])
            if i == j:
                continue

            wi = weights[i]
            wj = weights[j]

            Li = cur_loads[bi]
            Lj = cur_loads[bj]

            # Feasibility
            if Li - wi + wj > bin_capacity:
                continue
            if Lj - wj + wi > bin_capacity:
                continue

            # Tabu checks for both implied moves
            exp1 = tabu_move.get((i, bj), -1)
            exp2 = tabu_move.get((j, bi), -1)
            is_tabu = (exp1 >= iteration) or (exp2 >= iteration)

            # k unchanged
            new_k = k
            # residual delta
            delta = 0
            delta += compute_delta_res_sq(Li, Li - wi + wj)
            delta += compute_delta_res_sq(Lj, Lj - wj + wi)
            new_res_sq = cur_res_sq + delta
            neigh_obj = (new_k, new_res_sq)

            if is_tabu and not (neigh_obj < best_obj):
                continue

            if best_neighbor_obj is None or neigh_obj < best_neighbor_obj:
                best_neighbor_obj = neigh_obj
                best_neighbor = ("swap", i, bi, j, bj)

        if best_neighbor is None:
            return False

        # Apply best neighbor
        kind = best_neighbor[0]
        tenure = base_tenure + random.randrange(tenure_jitter)

        if kind == "move":
            _, i, bi, bj = best_neighbor
            # Update structures
            wi = weights[i]
            # remove from bi
            bin_items[bi].remove(i)
            cur_loads[bi] -= wi

            # add to bj
            bin_items[bj].append(i)
            cur_loads[bj] += wi
            cur_assign[i] = bj

            # mark reverse move tabu: forbid going back to bi
            tabu_move[(i, bi)] = iteration + tenure

            # clean empty bins
            remove_empty_bins()

        elif kind == "newbin":
            _, i, bi = best_neighbor
            wi = weights[i]
            # remove from bi
            bin_items[bi].remove(i)
            cur_loads[bi] -= wi
            # create new bin
            bin_items.append([i])
            cur_loads.append(wi)
            cur_assign[i] = len(cur_loads) - 1
            # forbid returning to source
            tabu_move[(i, bi)] = iteration + tenure
            remove_empty_bins()

        else:  # swap
            _, i, bi, j, bj = best_neighbor
            wi, wj = weights[i], weights[j]

            # swap in lists
            bin_items[bi].remove(i)
            bin_items[bj].remove(j)
            bin_items[bi].append(j)
            bin_items[bj].append(i)

            cur_loads[bi] = cur_loads[bi] - wi + wj
            cur_loads[bj] = cur_loads[bj] - wj + wi
            cur_assign[i] = bj
            cur_assign[j] = bi

            # mark tabu to reverse swap (i back to bi, j back to bj)
            tabu_move[(i, bi)] = iteration + tenure
            tabu_move[(j, bj)] = iteration + tenure

        # Update current objective
        cur_obj = objective(len(cur_loads), cur_loads)
        return True

    # Main loop
    stagnation = 0

    for it in range(1, max_iters + 1):
        if it % check_period == 0 and now_exceeded():
            break

        moved = try_best_move(it)
        if not moved:
            # If stuck, restart from a new randomized heuristic solution
            cur_assign, cur_loads = initial_solution()
            cur_obj = objective(len(cur_loads), cur_loads)
            bin_items = [[] for _ in range(len(cur_loads))]
            for i, b in enumerate(cur_assign):
                bin_items[b].append(i)
            stagnation = 0
            continue

        # Update best
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_assign = cur_assign[:]
            best_loads = cur_loads[:]
            stagnation = 0
        else:
            stagnation += 1

        # Diversification if no improvement for a while
        if stagnation >= stagnation_limit:
            cur_assign, cur_loads = initial_solution()
            cur_obj = objective(len(cur_loads), cur_loads)
            bin_items = [[] for _ in range(len(cur_loads))]
            for i, b in enumerate(cur_assign):
                bin_items[b].append(i)
            stagnation = 0

    # Build final packing from best assignment (ensure compact bins)
    packing, bin_wts, _ = build_packing_from_assign(best_assign, best_loads)
    return {"packing": packing, "bin_weights": bin_wts}
