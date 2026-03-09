import time
import random


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    def now_exceeded() -> bool:
        return time.time() >= deadline

    # ---------- Objective ----------
    def objective(loads: list[int]) -> tuple[int, int]:
        # (num_bins, sum residual^2)
        res_sq = 0
        C = bin_capacity
        for L in loads:
            r = C - L
            res_sq += r * r
        return (len(loads), res_sq)

    def build_packing_from_assign(assign: list[int]):
        # compress bin ids
        mapping = {}
        new_id = 0
        new_assign = [-1] * n
        for i, b in enumerate(assign):
            nb = mapping.get(b)
            if nb is None:
                nb = new_id
                mapping[b] = nb
                new_id += 1
            new_assign[i] = nb
        k = new_id
        pack = [[] for _ in range(k)]
        loads = [0] * k
        for i, b in enumerate(new_assign):
            pack[b].append(i)
            loads[b] += weights[i]
        return pack, loads, new_assign

    # ---------- Construction heuristic (RBFD) ----------
    def initial_solution() -> tuple[list[int], list[int], list[list[int]], list[dict[int, int]]]:
        order = list(range(n))
        order.sort(key=lambda i: weights[i], reverse=True)
        block = 8
        for s in range(0, n, block):
            e = min(n, s + block)
            if e - s > 1:
                random.shuffle(order[s:e])

        assign = [-1] * n
        loads: list[int] = []
        # temporary bin contents
        bin_items: list[list[int]] = []
        bin_pos: list[dict[int, int]] = []

        for i in order:
            w = weights[i]
            best_bin = -1
            best_rem = None
            for b, L in enumerate(loads):
                if L + w <= bin_capacity:
                    rem = bin_capacity - (L + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_bin = b
            if best_bin == -1:
                best_bin = len(loads)
                loads.append(0)
                bin_items.append([])
                bin_pos.append({})
            loads[best_bin] += w
            assign[i] = best_bin
            # add to structures
            p = len(bin_items[best_bin])
            bin_items[best_bin].append(i)
            bin_pos[best_bin][i] = p

        # already compact
        return assign, loads, bin_items, bin_pos

    # ---------- Fast bin item removal/addition ----------
    def bin_remove(bin_items: list[list[int]], bin_pos: list[dict[int, int]], b: int, item: int) -> None:
        pos = bin_pos[b].pop(item)
        last = bin_items[b].pop()
        if last != item:
            bin_items[b][pos] = last
            bin_pos[b][last] = pos

    def bin_add(bin_items: list[list[int]], bin_pos: list[dict[int, int]], b: int, item: int) -> None:
        bin_pos[b][item] = len(bin_items[b])
        bin_items[b].append(item)

    # ---------- Tabu Search params ----------
    base_tenure = 7
    tenure_jitter = 11

    # fixed iteration budget, but hard-stop by time anywhere
    max_iters = 200000

    # Create initial
    cur_assign, cur_loads, bin_items, bin_pos = initial_solution()
    cur_obj = objective(cur_loads)

    best_assign = cur_assign[:]
    best_loads = cur_loads[:]
    best_obj = cur_obj

    # Tabu structure: (item, to_bin) -> expiration_iter
    tabu_move: dict[tuple[int, int], int] = {}

    def compute_delta_res_sq(old_load: int, new_load: int) -> int:
        C = bin_capacity
        r0 = C - old_load
        r1 = C - new_load
        return r1 * r1 - r0 * r0

    def remove_empty_bins() -> None:
        nonlocal cur_assign, cur_loads, bin_items, bin_pos
        k = len(cur_loads)
        if k == 0:
            return
        mapping = [-1] * k
        new_loads = []
        new_items = []
        new_pos = []
        for b in range(k):
            if bin_items[b]:
                mapping[b] = len(new_loads)
                new_loads.append(cur_loads[b])
                new_items.append(bin_items[b])
                new_pos.append(bin_pos[b])
        if len(new_loads) == k:
            return
        # remap assignments
        new_assign = [-1] * n
        for old_b, nb in enumerate(mapping):
            if nb != -1:
                for item in bin_items[old_b]:
                    new_assign[item] = nb
        cur_assign = new_assign
        cur_loads = new_loads
        bin_items = new_items
        bin_pos = new_pos

    # Precompute heavy items list once (static weights)
    heavy_items = list(range(n))
    heavy_items.sort(key=lambda i: weights[i], reverse=True)

    def pick_candidate_items(sample_size: int) -> list[int]:
        # Mix of heaviest + random
        s = min(n, sample_size)
        cand = set(heavy_items[: min(n, s // 2 + 1)])
        while len(cand) < s:
            cand.add(random.randrange(n))
        return list(cand)

    def try_best_move(iteration: int) -> bool:
        nonlocal cur_assign, cur_loads, bin_items, bin_pos, cur_obj, best_obj

        if now_exceeded():
            return False

        k = len(cur_loads)
        if k == 0:
            return False

        cur_res_sq = cur_obj[1]

        # candidate items
        cand_items = pick_candidate_items(60)

        # Choose some promising destination bins without sorting all bins:
        # sample a subset and also include a few of the currently fullest bins via a cheap scan.
        # (full scan O(k) but k is number of bins, usually much smaller than n)
        if now_exceeded():
            return False

        # find top bins by load (up to 15) via partial selection
        top_m = min(k, 15)
        # simple O(k * top_m) selection; good enough and avoids full sort
        top_bins: list[int] = []
        for b in range(k):
            L = cur_loads[b]
            inserted = False
            for idx, bb in enumerate(top_bins):
                if L > cur_loads[bb]:
                    top_bins.insert(idx, b)
                    inserted = True
                    break
            if not inserted:
                top_bins.append(b)
            if len(top_bins) > top_m:
                top_bins.pop()

        allow_new_bin = (random.random() < 0.10)

        best_neighbor = None
        best_neighbor_obj = None

        # --- Move neighborhood ---
        for t, i in enumerate(cand_items):
            if (t & 7) == 0 and now_exceeded():
                return False

            bi = cur_assign[i]
            wi = weights[i]

            dest_bins = []
            # top bins first
            for b in top_bins:
                if b != bi:
                    dest_bins.append(b)
            # plus random bins
            for _ in range(6):
                if k > 1:
                    b = random.randrange(k)
                    if b != bi:
                        dest_bins.append(b)
            if dest_bins:
                # unique preserve order
                seen = set()
                uniq = []
                for b in dest_bins:
                    if b not in seen:
                        seen.add(b)
                        uniq.append(b)
                dest_bins = uniq

            for bj in dest_bins:
                if cur_loads[bj] + wi > bin_capacity:
                    continue

                exp = tabu_move.get((i, bj), -1)
                is_tabu = exp >= iteration

                # k changes only if source bin becomes empty
                new_k = k - 1 if (len(bin_items[bi]) == 1 and bi != bj) else k

                delta = 0
                new_load_bi = cur_loads[bi] - wi
                new_load_bj = cur_loads[bj] + wi
                delta += compute_delta_res_sq(cur_loads[bi], new_load_bi)
                delta += compute_delta_res_sq(cur_loads[bj], new_load_bj)

                new_res_sq = cur_res_sq + delta
                if new_k < k:
                    new_res_sq -= (bin_capacity * bin_capacity)  # remove empty bin term

                neigh_obj = (new_k, new_res_sq)

                if is_tabu and not (neigh_obj < best_obj):
                    continue

                if best_neighbor_obj is None or neigh_obj < best_neighbor_obj:
                    best_neighbor_obj = neigh_obj
                    best_neighbor = ("move", i, bi, bj)

            if allow_new_bin and len(bin_items[bi]) > 1:
                bj = k
                exp = tabu_move.get((i, bj), -1)
                is_tabu = exp >= iteration

                new_k = k + 1
                new_load_bi = cur_loads[bi] - wi
                delta = compute_delta_res_sq(cur_loads[bi], new_load_bi)
                new_res_sq = cur_res_sq + delta + (bin_capacity - wi) ** 2
                neigh_obj = (new_k, new_res_sq)

                if not (is_tabu and not (neigh_obj < best_obj)):
                    if best_neighbor_obj is None or neigh_obj < best_neighbor_obj:
                        best_neighbor_obj = neigh_obj
                        best_neighbor = ("newbin", i, bi)

        # --- Swap neighborhood ---
        pair_trials = 120
        for t in range(pair_trials):
            if (t & 15) == 0 and now_exceeded():
                return False

            if k <= 1:
                break
            i = random.choice(cand_items)
            bi = cur_assign[i]
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

            if Li - wi + wj > bin_capacity:
                continue
            if Lj - wj + wi > bin_capacity:
                continue

            exp1 = tabu_move.get((i, bj), -1)
            exp2 = tabu_move.get((j, bi), -1)
            is_tabu = (exp1 >= iteration) or (exp2 >= iteration)

            delta = 0
            delta += compute_delta_res_sq(Li, Li - wi + wj)
            delta += compute_delta_res_sq(Lj, Lj - wj + wi)
            neigh_obj = (k, cur_res_sq + delta)

            if is_tabu and not (neigh_obj < best_obj):
                continue

            if best_neighbor_obj is None or neigh_obj < best_neighbor_obj:
                best_neighbor_obj = neigh_obj
                best_neighbor = ("swap", i, bi, j, bj)

        if best_neighbor is None:
            return False

        kind = best_neighbor[0]
        tenure = base_tenure + random.randrange(tenure_jitter)

        if kind == "move":
            _, i, bi, bj = best_neighbor
            wi = weights[i]

            bin_remove(bin_items, bin_pos, bi, i)
            cur_loads[bi] -= wi

            bin_add(bin_items, bin_pos, bj, i)
            cur_loads[bj] += wi
            cur_assign[i] = bj

            tabu_move[(i, bi)] = iteration + tenure
            if not bin_items[bi]:
                remove_empty_bins()

        elif kind == "newbin":
            _, i, bi = best_neighbor
            wi = weights[i]

            bin_remove(bin_items, bin_pos, bi, i)
            cur_loads[bi] -= wi

            # create new bin
            bin_items.append([i])
            bin_pos.append({i: 0})
            cur_loads.append(wi)
            cur_assign[i] = len(cur_loads) - 1

            tabu_move[(i, bi)] = iteration + tenure
            if not bin_items[bi]:
                remove_empty_bins()

        else:  # swap
            _, i, bi, j, bj = best_neighbor
            wi, wj = weights[i], weights[j]

            # remove both
            bin_remove(bin_items, bin_pos, bi, i)
            bin_remove(bin_items, bin_pos, bj, j)
            # add swapped
            bin_add(bin_items, bin_pos, bi, j)
            bin_add(bin_items, bin_pos, bj, i)

            cur_loads[bi] = cur_loads[bi] - wi + wj
            cur_loads[bj] = cur_loads[bj] - wj + wi
            cur_assign[i] = bj
            cur_assign[j] = bi

            tabu_move[(i, bi)] = iteration + tenure
            tabu_move[(j, bj)] = iteration + tenure

        cur_obj = objective(cur_loads)
        return True

    # ---------- Main loop ----------
    stagnation = 0
    stagnation_limit = 5000

    it = 0
    while it < max_iters and not now_exceeded():
        it += 1

        moved = try_best_move(it)
        if now_exceeded():
            break

        if not moved:
            # restart
            cur_assign, cur_loads, bin_items, bin_pos = initial_solution()
            cur_obj = objective(cur_loads)
            stagnation = 0
            continue

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_assign = cur_assign[:]
            best_loads = cur_loads[:]
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= stagnation_limit:
            cur_assign, cur_loads, bin_items, bin_pos = initial_solution()
            cur_obj = objective(cur_loads)
            stagnation = 0

    # Build final packing from best assignment
    packing, loads, _ = build_packing_from_assign(best_assign)
    return {"packing": packing, "bin_weights": loads}
