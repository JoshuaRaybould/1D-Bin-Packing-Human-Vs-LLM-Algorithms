import time
import math
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    n = len(weights)
    C = int(bin_capacity)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.time()
    hard_cap = 100.0
    deadline = start + max(0.0, float(time_limit))
    deadline = min(deadline, start + hard_cap)

    # ------------------------ Lower bounds ------------------------
    total_w = sum(weights)
    lb1 = (total_w + C - 1) // C
    half = C / 2.0
    lb2 = max(lb1, sum(1 for w in weights if w > half))

    def time_up() -> bool:
        return time.time() >= deadline

    # ------------------------ Representation ------------------------
    # bins: list[list[int]] (items)
    # loads: list[int]
    # item_bin[i], pos_in_bin[i]
    # active_bins: list[int] containing indices of non-empty bins
    # pos_active[b] gives position in active_bins or -1

    def activate_bin(b: int, active_bins: List[int], pos_active: List[int]) -> None:
        if pos_active[b] != -1:
            return
        pos_active[b] = len(active_bins)
        active_bins.append(b)

    def deactivate_bin(b: int, active_bins: List[int], pos_active: List[int]) -> None:
        p = pos_active[b]
        if p == -1:
            return
        last = active_bins[-1]
        active_bins[p] = last
        pos_active[last] = p
        active_bins.pop()
        pos_active[b] = -1

    def append_item(i: int, b: int,
                    bins: List[List[int]], loads: List[int], item_bin: List[int], pos_in_bin: List[int],
                    active_bins: List[int], pos_active: List[int]) -> None:
        if not bins[b]:
            activate_bin(b, active_bins, pos_active)
        pos_in_bin[i] = len(bins[b])
        bins[b].append(i)
        item_bin[i] = b
        loads[b] += weights[i]

    def remove_item(i: int, b: int,
                    bins: List[List[int]], loads: List[int], item_bin: List[int], pos_in_bin: List[int],
                    active_bins: List[int], pos_active: List[int]) -> None:
        p = pos_in_bin[i]
        lst = bins[b]
        last_it = lst[-1]
        lst[p] = last_it
        pos_in_bin[last_it] = p
        lst.pop()
        pos_in_bin[i] = -1
        item_bin[i] = -1
        loads[b] -= weights[i]
        if not lst:
            deactivate_bin(b, active_bins, pos_active)

    def do_move(i: int, a: int, b: int,
                bins: List[List[int]], loads: List[int], item_bin: List[int], pos_in_bin: List[int],
                active_bins: List[int], pos_active: List[int]) -> None:
        remove_item(i, a, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
        append_item(i, b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

    def do_swap(i: int, a: int, j: int, b: int,
                bins: List[List[int]], loads: List[int], item_bin: List[int], pos_in_bin: List[int]) -> None:
        li = bins[a]
        lj = bins[b]
        pi = pos_in_bin[i]
        pj = pos_in_bin[j]
        li[pi], lj[pj] = j, i
        pos_in_bin[i] = pj
        pos_in_bin[j] = pi
        item_bin[i] = b
        item_bin[j] = a
        wi, wj = weights[i], weights[j]
        loads[a] += (wj - wi)
        loads[b] += (wi - wj)

    # ------------------------ Objective ------------------------
    # We use a scalar score but with very separated scales:
    # score = used_bins * A + infeas * B(t) + waste * D
    # waste = sum slack^2 for feasible bins + (overload)^2 for infeasible bins

    def contrib_bin(b: int, bins: List[List[int]], loads: List[int]) -> Tuple[int, int, int]:
        # (active, infeas, waste)
        if not bins[b]:
            return 0, 0, 0
        ld = loads[b]
        if ld <= C:
            s = C - ld
            return 1, 0, s * s
        else:
            ov = ld - C
            return 1, ov, ov * ov

    def full_eval(bins: List[List[int]], loads: List[int], active_bins: List[int]) -> Tuple[int, int, int]:
        used = len(active_bins)
        infeas = 0
        waste = 0
        for b in active_bins:
            ld = loads[b]
            if ld <= C:
                s = C - ld
                waste += s * s
            else:
                ov = ld - C
                infeas += ov
                waste += ov * ov
        return used, infeas, waste

    # ------------------------ Constructions ------------------------
    items_sorted = list(range(n))
    items_sorted.sort(key=lambda i: weights[i], reverse=True)

    def randomized_order(base: List[int]) -> List[int]:
        order = base[:]
        i = 0
        while i < n:
            j = i + 1
            wi = weights[order[i]]
            while j < n and weights[order[j]] == wi:
                j += 1
            if j - i > 1:
                block = order[i:j]
                random.shuffle(block)
                order[i:j] = block
            i = j
        return order

    def construct_bfd(order: List[int], noise: float = 0.0, sample_k: int = 0):
        bins: List[List[int]] = []
        loads: List[int] = []
        item_bin = [-1] * n
        pos_in_bin = [-1] * n
        active_bins: List[int] = []
        pos_active: List[int] = []

        def ensure_bin(b: int) -> None:
            while b >= len(bins):
                bins.append([])
                loads.append(0)
                pos_active.append(-1)

        for it in order:
            w = weights[it]
            best_b = -1
            best_metric = None

            # candidate set: either all active bins or a random subset for speed/diversity
            if sample_k and len(active_bins) > sample_k:
                cand = [active_bins[random.randrange(len(active_bins))] for _ in range(sample_k)]
            else:
                cand = active_bins

            for b in cand:
                ld = loads[b]
                if ld + w <= C:
                    rem = C - (ld + w)
                    metric = rem
                    if noise:
                        metric += noise * random.random()
                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        best_b = b

            if best_b == -1:
                best_b = len(bins)
                ensure_bin(best_b)
            ensure_bin(best_b)
            append_item(it, best_b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

        return bins, loads, item_bin, pos_in_bin, active_bins, pos_active

    # ------------------------ Best bookkeeping ------------------------
    global_best_bins = None
    global_best_waste = None
    global_best_snapshot = None

    def snapshot_from_state(state):
        bins, loads, _, _, _, _ = state
        return ([lst[:] for lst in bins], loads[:])

    def state_from_snapshot(snapshot):
        snap_bins, snap_loads = snapshot
        bins = [lst[:] for lst in snap_bins]
        loads = snap_loads[:]
        item_bin = [-1] * n
        pos_in_bin = [-1] * n
        active_bins: List[int] = []
        pos_active = [-1] * len(bins)
        for b, lst in enumerate(bins):
            if lst:
                pos_active[b] = len(active_bins)
                active_bins.append(b)
            for p, it in enumerate(lst):
                item_bin[it] = b
                pos_in_bin[it] = p
        return bins, loads, item_bin, pos_in_bin, active_bins, pos_active

    def consider_state(state) -> None:
        nonlocal global_best_bins, global_best_waste, global_best_snapshot
        bins, loads, _, _, active_bins, _ = state
        used, infeas, waste = full_eval(bins, loads, active_bins)
        if infeas != 0:
            return
        if (global_best_bins is None or used < global_best_bins or
                (used == global_best_bins and waste < global_best_waste)):
            global_best_bins = used
            global_best_waste = waste
            global_best_snapshot = snapshot_from_state(state)

    # Build initial pool (more diverse)
    init_states = []
    init_states.append(construct_bfd(items_sorted, noise=0.0, sample_k=0))  # pure BFD
    init_states.append(construct_bfd(items_sorted, noise=0.02, sample_k=0))
    for nz in (0.01, 0.03, 0.06, 0.10):
        init_states.append(construct_bfd(randomized_order(items_sorted), noise=nz, sample_k=0))
        init_states.append(construct_bfd(randomized_order(items_sorted), noise=nz, sample_k=25))

    for st in init_states:
        consider_state(st)

    # ------------------------ Sampling helpers ------------------------
    def pick_overloaded_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        m = len(active_bins)
        best = -1
        best_ov = 0
        for _ in range(k):
            b = active_bins[random.randrange(m)]
            ov = loads[b] - C
            if ov > best_ov:
                best_ov = ov
                best = b
        return best

    def pick_light_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        m = len(active_bins)
        best = active_bins[random.randrange(m)]
        best_ld = loads[best]
        for _ in range(k - 1):
            b = active_bins[random.randrange(m)]
            ld = loads[b]
            if ld < best_ld:
                best, best_ld = b, ld
        return best

    def pick_tight_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        m = len(active_bins)
        best = active_bins[random.randrange(m)]
        best_key = (0 if loads[best] <= C else 1, -min(loads[best], C))
        for _ in range(k - 1):
            b = active_bins[random.randrange(m)]
            key = (0 if loads[b] <= C else 1, -min(loads[b], C))
            if key < best_key:
                best, best_key = b, key
        return best

    def choose_dest(i: int, src: int, active_bins: List[int], loads: List[int],
                    probes: int, allow_infeas_p: float, forbid: int = -1) -> int:
        wi = weights[i]
        m = len(active_bins)
        best = -1
        best_metric = None
        for _ in range(probes):
            b = active_bins[random.randrange(m)]
            if b == src or b == forbid:
                continue
            new_ld = loads[b] + wi
            if new_ld <= C or random.random() < allow_infeas_p:
                ov = max(0, new_ld - C)
                slack = max(0, C - new_ld)
                metric = (ov, slack)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best = b
        return best

    # ------------------------ SA core ------------------------
    def sa_run(state, run_budget: float) -> None:
        nonlocal global_best_bins, global_best_waste, global_best_snapshot

        bins, loads, item_bin, pos_in_bin, active_bins, pos_active = state
        if len(pos_active) < len(bins):
            pos_active.extend([-1] * (len(bins) - len(pos_active)))

        used, infeas, waste = full_eval(bins, loads, active_bins)

        # scoring scales
        A = 10**12  # bins
        base_pen = 5 * 10**8  # infeas

        def score(u: int, inf: int, wst: int, T: float, late: float) -> float:
            # late in [0,1]; push feasibility hard late
            pen = base_pen * (1.0 + 4.0 * late) * (1.0 + 0.4 / max(1e-9, T))
            return u * A + inf * pen + wst

        def store_best() -> None:
            nonlocal global_best_bins, global_best_waste, global_best_snapshot
            if infeas != 0:
                return
            u = len(active_bins)
            if (global_best_bins is None or u < global_best_bins or
                    (u == global_best_bins and waste < global_best_waste)):
                global_best_bins = u
                global_best_waste = waste
                global_best_snapshot = snapshot_from_state((bins, loads, item_bin, pos_in_bin, active_bins, pos_active))

        store_best()

        run_deadline = min(deadline, time.time() + max(0.0, run_budget))

        # temperature calibration: sample deltas
        sample = []
        for _ in range(400):
            if len(active_bins) <= 1:
                break
            a = pick_light_bin(active_bins, loads, 4)
            if not bins[a]:
                continue
            i = bins[a][random.randrange(len(bins[a]))]
            b = choose_dest(i, a, active_bins, loads, probes=20, allow_infeas_p=0.05)
            if b == -1:
                continue
            oldA = contrib_bin(a, bins, loads)
            oldB = contrib_bin(b, bins, loads)
            old_used = len(active_bins)
            old_inf = infeas
            old_waste = waste

            do_move(i, a, b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            new_used = len(active_bins)
            newA = contrib_bin(a, bins, loads)
            newB = contrib_bin(b, bins, loads)
            new_inf = old_inf - oldA[1] - oldB[1] + newA[1] + newB[1]
            new_waste = old_waste - oldA[2] - oldB[2] + newA[2] + newB[2]

            d = (new_used - old_used) * A + (new_inf - old_inf) * base_pen + (new_waste - old_waste)
            do_move(i, b, a, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            if d > 0:
                sample.append(d)

        if sample:
            sample.sort()
            med = sample[len(sample) // 2]
            T0 = max(1e-9, med / max(1e-9, -math.log(0.75)))
        else:
            T0 = max(1.0, 0.2 * C)

        # fixed iterations + periodic time checks
        max_iter = 18_000_000
        check_period = 4000

        T = T0
        Tf = max(1e-9, T0 * 5e-6)
        alpha = (Tf / T0) ** (1.0 / max(1, max_iter - 1))

        # operator mix (ruin-recreate used as a standard SA large move)
        p_move = 0.52
        p_swap = 0.20
        p_rr = 0.18   # empty a bin and reinsert
        p_exch = 0.10 # 2-1 exchange

        # parameters
        allow_infeas_early = 0.10
        allow_infeas_late = 0.015

        def ruin_recreate(bin_idx: int, sample_k: int) -> Tuple[bool, list]:
            # remove all items from bin_idx, then reinsert by best-fit into existing bins
            # Returns (success, undo_ops)
            if bin_idx == -1 or pos_active[bin_idx] == -1 or not bins[bin_idx]:
                return False, []

            undo = []
            pool = bins[bin_idx][:]
            # remove
            for it in pool:
                remove_item(it, bin_idx, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                undo.append(("back", it, bin_idx))

            # reinsert
            pool.sort(key=lambda x: weights[x], reverse=True)
            created_bins = []

            for it in pool:
                w = weights[it]
                best_b = -1
                best_rem = None

                if sample_k and len(active_bins) > sample_k:
                    cand = [active_bins[random.randrange(len(active_bins))] for _ in range(sample_k)]
                else:
                    cand = active_bins

                for b in cand:
                    ld = loads[b]
                    if ld + w <= C:
                        rem = C - (ld + w)
                        if best_rem is None or rem < best_rem:
                            best_rem = rem
                            best_b = b

                if best_b == -1:
                    # create new bin (allowed in SA, but will be penalized by used bins)
                    nb = len(bins)
                    bins.append([])
                    loads.append(0)
                    pos_active.append(-1)
                    activate_bin(nb, active_bins, pos_active)
                    created_bins.append(nb)
                    best_b = nb

                append_item(it, best_b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                undo.append(("mv", it, best_b))

            # record created bins for undo
            if created_bins:
                undo.append(("created", created_bins))
            return True, undo

        def undo_rr(undo):
            # undo in reverse
            created_bins = None
            for op in reversed(undo):
                if op[0] == "created":
                    created_bins = op[1]
                    continue
                if op[0] == "mv":
                    _, it, b = op
                    # remove from current bin b and put nowhere for now
                    remove_item(it, b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                else:
                    _, it, b0 = op
                    # put back into original bin
                    # ensure bin exists
                    while b0 >= len(bins):
                        bins.append([])
                        loads.append(0)
                        pos_active.append(-1)
                    append_item(it, b0, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

            # deactivate empty created bins
            if created_bins:
                for b in created_bins:
                    if b < len(bins) and not bins[b] and pos_active[b] != -1:
                        deactivate_bin(b, active_bins, pos_active)

        for it in range(max_iter):
            if (it % check_period) == 0:
                if time.time() >= run_deadline:
                    break

            late = it / max_iter
            allow_infeas = allow_infeas_early * (1.0 - late) + allow_infeas_late * late

            old_used = len(active_bins)
            old_score = score(old_used, infeas, waste, max(1e-12, T), late)

            r = random.random()
            accepted = False

            if r < p_rr and len(active_bins) >= 2:
                # target: light / small bin (easier to eliminate)
                b = pick_light_bin(active_bins, loads, 10)
                if b != -1:
                    ok, undo = ruin_recreate(b, sample_k=35)
                    if ok:
                        new_used, new_inf, new_waste = full_eval(bins, loads, active_bins)
                        new_score = score(new_used, new_inf, new_waste, max(1e-12, T), late)
                        d = new_score - old_score
                        if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                            accepted = True
                            used, infeas, waste = new_used, new_inf, new_waste
                        else:
                            undo_rr(undo)

            elif r < p_rr + p_exch and len(active_bins) >= 2:
                # 2-1 exchange
                A_bin = pick_light_bin(active_bins, loads, 7)
                B_bin = pick_tight_bin(active_bins, loads, 7)
                if A_bin != B_bin and A_bin != -1 and B_bin != -1 and bins[A_bin] and bins[B_bin]:
                    x = bins[A_bin][random.randrange(len(bins[A_bin]))]
                    # move x to B
                    oldA = contrib_bin(A_bin, bins, loads)
                    oldB = contrib_bin(B_bin, bins, loads)
                    do_move(x, A_bin, B_bin, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                    # select y from B and move elsewhere
                    success = False
                    for _ in range(6):
                        y = bins[B_bin][random.randrange(len(bins[B_bin]))]
                        if y == x:
                            continue
                        dest = choose_dest(y, B_bin, active_bins, loads, probes=28, allow_infeas_p=allow_infeas, forbid=B_bin)
                        if dest == -1:
                            continue
                        oldD = contrib_bin(dest, bins, loads)
                        do_move(y, B_bin, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                        newA = contrib_bin(A_bin, bins, loads)
                        newB = contrib_bin(B_bin, bins, loads)
                        newD = contrib_bin(dest, bins, loads)
                        new_inf = infeas - oldA[1] - oldB[1] - oldD[1] + newA[1] + newB[1] + newD[1]
                        new_waste = waste - oldA[2] - oldB[2] - oldD[2] + newA[2] + newB[2] + newD[2]
                        new_used = len(active_bins)
                        new_score = score(new_used, new_inf, new_waste, max(1e-12, T), late)
                        d = new_score - old_score
                        if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                            accepted = True
                            infeas, waste = new_inf, new_waste
                            success = True
                            break
                        else:
                            do_move(y, dest, B_bin, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                    if not success and not accepted:
                        do_move(x, B_bin, A_bin, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

            elif r < p_rr + p_exch + p_swap and len(active_bins) >= 2:
                # swap
                A_bin = pick_light_bin(active_bins, loads, 6)
                B_bin = pick_tight_bin(active_bins, loads, 6)
                if A_bin != B_bin and A_bin != -1 and B_bin != -1 and bins[A_bin] and bins[B_bin]:
                    iitem = bins[A_bin][random.randrange(len(bins[A_bin]))]
                    jitem = bins[B_bin][random.randrange(len(bins[B_bin]))]
                    if iitem != jitem:
                        oldA = contrib_bin(A_bin, bins, loads)
                        oldB = contrib_bin(B_bin, bins, loads)
                        do_swap(iitem, A_bin, jitem, B_bin, bins, loads, item_bin, pos_in_bin)
                        newA = contrib_bin(A_bin, bins, loads)
                        newB = contrib_bin(B_bin, bins, loads)
                        new_inf = infeas - oldA[1] - oldB[1] + newA[1] + newB[1]
                        new_waste = waste - oldA[2] - oldB[2] + newA[2] + newB[2]
                        new_used = len(active_bins)
                        new_score = score(new_used, new_inf, new_waste, max(1e-12, T), late)
                        d = new_score - old_score
                        if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                            accepted = True
                            infeas, waste = new_inf, new_waste
                        else:
                            do_swap(iitem, B_bin, jitem, A_bin, bins, loads, item_bin, pos_in_bin)

            else:
                # relocation move
                # choose source: if infeasible prefer overloaded
                if infeas != 0 and random.random() < 0.75:
                    src = pick_overloaded_bin(active_bins, loads, 14)
                    if src == -1:
                        src = pick_light_bin(active_bins, loads, 8)
                else:
                    src = pick_light_bin(active_bins, loads, 8) if random.random() < 0.6 else pick_tight_bin(active_bins, loads, 8)

                if src != -1 and bins[src]:
                    itx = bins[src][random.randrange(len(bins[src]))]
                    dest = choose_dest(itx, src, active_bins, loads, probes=30, allow_infeas_p=allow_infeas)
                    create_new = (infeas != 0 and random.random() < 0.18 and late < 0.65) or (late < 0.08 and random.random() < 0.04)
                    newbin = -1
                    if dest == -1 and create_new:
                        newbin = len(bins)
                        bins.append([])
                        loads.append(0)
                        pos_active.append(-1)
                        activate_bin(newbin, active_bins, pos_active)
                        dest = newbin

                    if dest != -1:
                        oldS = contrib_bin(src, bins, loads)
                        oldD = contrib_bin(dest, bins, loads) if newbin == -1 else (0, 0, 0)
                        do_move(itx, src, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                        newS = contrib_bin(src, bins, loads)
                        newD = contrib_bin(dest, bins, loads)
                        new_inf = infeas - oldS[1] - oldD[1] + newS[1] + newD[1]
                        new_waste = waste - oldS[2] - oldD[2] + newS[2] + newD[2]
                        new_used = len(active_bins)
                        new_score = score(new_used, new_inf, new_waste, max(1e-12, T), late)
                        d = new_score - old_score
                        if d <= 0 or random.random() < math.exp(-d / max(1e-12, T)):
                            accepted = True
                            infeas, waste = new_inf, new_waste
                        else:
                            do_move(itx, dest, src, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                            if newbin != -1:
                                deactivate_bin(newbin, active_bins, pos_active)

            if accepted:
                store_best()
                # reheat when we match best bins (helps escape plateaus)
                if global_best_bins is not None and infeas == 0 and len(active_bins) <= global_best_bins:
                    if T < 0.12 * T0:
                        T = 0.12 * T0

            T *= alpha

    # ------------------------ Restart schedule ------------------------
    # Always use all time up to deadline.
    remaining = max(0.0, deadline - time.time())
    if remaining <= 0:
        if global_best_snapshot is None:
            global_best_snapshot = snapshot_from_state(init_states[0])
        snap_bins, snap_loads = global_best_snapshot
        packing, bin_weights = [], []
        for b, lst in enumerate(snap_bins):
            if lst:
                packing.append(lst[:])
                bin_weights.append(snap_loads[b])
        return {"packing": packing, "bin_weights": bin_weights}

    # restarts count by time
    tl = remaining
    if tl <= 2.0:
        restarts = 2
    elif tl <= 8.0:
        restarts = 5
    elif tl <= 25.0:
        restarts = 8
    else:
        restarts = 12

    # Seeds: best initial constructions first
    init_states.sort(key=lambda st: full_eval(st[0], st[1], st[4]))
    seeds = init_states[:min(len(init_states), restarts)]

    for r in range(restarts):
        if time_up():
            break
        remaining = max(0.0, deadline - time.time())
        future = max(1, restarts - r)
        budget = remaining / future

        if r < len(seeds):
            st = seeds[r]
        else:
            # perturb best snapshot by rebuilding a couple of light bins
            if global_best_snapshot is not None:
                st = state_from_snapshot(global_best_snapshot)
                bins, loads, item_bin, pos_in_bin, active_bins, pos_active = st
                if len(active_bins) >= 2:
                    chosen = {pick_light_bin(active_bins, loads, 10) for _ in range(2 + (1 if random.random() < 0.35 else 0))}
                    pool = []
                    for b in chosen:
                        if b != -1 and bins[b]:
                            for itx in bins[b][:]:
                                pool.append(itx)
                                remove_item(itx, b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                    pool.sort(key=lambda x: weights[x], reverse=True)
                    for itx in pool:
                        w = weights[itx]
                        best_b = -1
                        best_rem = None
                        for b in active_bins:
                            ld = loads[b]
                            if ld + w <= C:
                                rem = C - (ld + w)
                                if best_rem is None or rem < best_rem:
                                    best_rem = rem
                                    best_b = b
                        if best_b == -1:
                            nb = len(bins)
                            bins.append([])
                            loads.append(0)
                            pos_active.append(-1)
                            activate_bin(nb, active_bins, pos_active)
                            best_b = nb
                        append_item(itx, best_b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            else:
                st = construct_bfd(randomized_order(items_sorted), noise=0.05, sample_k=30)

        sa_run(st, budget)

        # If we hit strong LB, intensify from best more often
        if global_best_bins is not None and global_best_bins <= lb2:
            pass

    # final intensification from best
    if not time_up() and global_best_snapshot is not None:
        rem = max(0.0, deadline - time.time())
        if rem > 0.02:
            st = state_from_snapshot(global_best_snapshot)
            sa_run(st, rem)

    # ------------------------ Export ------------------------
    if global_best_snapshot is None:
        # fallback FFD
        bins = []
        loads = []
        for i in items_sorted:
            w = weights[i]
            best = -1
            best_rem = None
            for b, ld in enumerate(loads):
                if ld + w <= C:
                    rem = C - (ld + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best = b
            if best == -1:
                best = len(bins)
                bins.append([])
                loads.append(0)
            bins[best].append(i)
            loads[best] += w
        return {"packing": bins, "bin_weights": loads}

    snap_bins, snap_loads = global_best_snapshot
    packing: List[List[int]] = []
    bin_weights: List[int] = []
    for b, lst in enumerate(snap_bins):
        if lst:
            packing.append(lst[:])
            bin_weights.append(snap_loads[b])
    return {"packing": packing, "bin_weights": bin_weights}
