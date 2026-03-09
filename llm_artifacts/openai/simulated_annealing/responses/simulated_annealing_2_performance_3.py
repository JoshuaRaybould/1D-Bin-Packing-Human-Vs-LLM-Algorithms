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
    deadline = start + max(0.0, float(time_limit))

    # ------------------------ Lower bounds ------------------------
    total_w = sum(weights)
    lb1 = (total_w + C - 1) // C
    half = C / 2.0
    lb2 = max(lb1, sum(1 for w in weights if w > half))

    def time_up() -> bool:
        return time.time() >= deadline

    # ------------------------ State representation ------------------------
    # bins: list of lists of items (stable indices)
    # loads: load per bin
    # item_bin[i], pos_in_bin[i]
    # active_bins: list of non-empty bin indices; pos_active[bin] in active_bins or -1

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
    BIG = 10**9

    def bin_over(ld: int) -> int:
        return ld - C if ld > C else 0

    def bin_sec(ld: int) -> int:
        # squared slack for feasible bins; 0 if infeasible
        if ld <= C:
            s = C - ld
            return s * s
        return 0

    def full_eval(loads: List[int], active_bins: List[int]) -> Tuple[int, int, int]:
        used = len(active_bins)
        infeas = 0
        sec = 0
        for b in active_bins:
            ld = loads[b]
            infeas += bin_over(ld)
            sec += bin_sec(ld)
        return used, infeas, sec

    def contrib_for_bin(b: int, bins: List[List[int]], loads: List[int]) -> Tuple[int, int, int]:
        # (active, over, sec)
        if bins[b]:
            ld = loads[b]
            return 1, bin_over(ld), bin_sec(ld)
        return 0, 0, 0

    # ------------------------ Constructions ------------------------
    items_sorted = list(range(n))
    items_sorted.sort(key=lambda i: weights[i], reverse=True)

    def construct_bfd(order: List[int], noise: float = 0.0) -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], List[int]]:
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

        for i in order:
            w = weights[i]
            best_b = -1
            best_metric = None
            # best fit with optional tiny noise to diversify tie-breaking
            for b in active_bins:
                ld = loads[b]
                if ld + w <= C:
                    rem = C - (ld + w)
                    metric = rem
                    if noise:
                        metric = metric + noise * random.random()
                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        best_b = b
            if best_b == -1:
                best_b = len(bins)
                ensure_bin(best_b)
            ensure_bin(best_b)
            append_item(i, best_b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

        return bins, loads, item_bin, pos_in_bin, active_bins, pos_active

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

    def quick_repack_one_light_bin(state):
        # take one light bin, remove its items, and reinsert with BFD into existing bins (no new bins unless needed)
        bins, loads, item_bin, pos_in_bin, active_bins, pos_active = state
        if len(active_bins) <= 1:
            return state

        # pick a light bin among few samples
        def pick_light(k: int) -> int:
            ab = active_bins
            best = ab[random.randrange(len(ab))]
            best_ld = loads[best]
            for _ in range(k - 1):
                b = ab[random.randrange(len(ab))]
                ld = loads[b]
                if ld < best_ld:
                    best, best_ld = b, ld
            return best

        b = pick_light(8)
        if b == -1 or not bins[b]:
            return state

        pool = bins[b][:]
        for it in pool[:]:
            remove_item(it, b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

        pool.sort(key=lambda it: weights[it], reverse=True)
        for it in pool:
            w = weights[it]
            best_b = -1
            best_rem = None
            for bb in active_bins:
                ld = loads[bb]
                if ld + w <= C:
                    rem = C - (ld + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_b = bb
            if best_b == -1:
                nb = len(bins)
                bins.append([])
                loads.append(0)
                pos_active.append(-1)
                activate_bin(nb, active_bins, pos_active)
                best_b = nb
            append_item(it, best_b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

        return state

    # Build initial pool
    init_states = []
    init_states.append(construct_bfd(items_sorted, noise=0.0))
    init_states.append(construct_bfd(randomized_order(items_sorted), noise=0.01))
    init_states.append(construct_bfd(randomized_order(items_sorted), noise=0.03))
    init_states.append(construct_bfd(randomized_order(items_sorted), noise=0.06))

    for k in range(min(6, max(0, n // 50))):
        init_states.append(construct_bfd(randomized_order(items_sorted), noise=0.05 + 0.02 * k))

    init_states = [quick_repack_one_light_bin(st) for st in init_states]

    def slack_of(state) -> int:
        bins, loads, _, _, active_bins, _ = state
        s = 0
        for b in active_bins:
            ld = loads[b]
            if ld <= C:
                s += (C - ld)
        return s

    init_states.sort(key=lambda st: (len(st[4]), slack_of(st)))

    # ------------------------ Best snapshot bookkeeping ------------------------
    global_best_bins = None
    global_best_slack = None
    global_best_snapshot = None  # (bins, loads)

    def snapshot_from_state(state):
        bins, loads, _, _, _, _ = state
        return ([lst[:] for lst in bins], loads[:])

    def consider_state(state) -> None:
        nonlocal global_best_bins, global_best_slack, global_best_snapshot
        bins, loads, _, _, active_bins, _ = state
        used, infeas, _ = full_eval(loads, active_bins)
        if infeas != 0:
            return
        sl = slack_of(state)
        if (global_best_bins is None or used < global_best_bins or
                (used == global_best_bins and sl < global_best_slack)):
            global_best_bins = used
            global_best_slack = sl
            global_best_snapshot = snapshot_from_state(state)

    for st in init_states:
        consider_state(st)

    # ------------------------ Sampling helpers ------------------------
    def pick_overloaded_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        ab = active_bins
        m = len(ab)
        best = -1
        best_ov = 0
        for _ in range(k):
            b = ab[random.randrange(m)]
            ov = loads[b] - C
            if ov > best_ov:
                best_ov = ov
                best = b
        return best

    def pick_light_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        ab = active_bins
        m = len(ab)
        best = ab[random.randrange(m)]
        best_ld = loads[best]
        for _ in range(k - 1):
            b = ab[random.randrange(m)]
            ld = loads[b]
            if ld < best_ld:
                best, best_ld = b, ld
        return best

    def pick_tight_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        ab = active_bins
        m = len(ab)
        best = ab[random.randrange(m)]
        best_ld = loads[best]
        for _ in range(k - 1):
            b = ab[random.randrange(m)]
            ld = loads[b]
            if (ld <= C and best_ld > C) or (ld <= C and best_ld <= C and ld > best_ld) or (ld > C and best_ld > C and ld > best_ld):
                best, best_ld = b, ld
        return best

    def choose_destination_for_item(i: int, src: int,
                                    active_bins: List[int], loads: List[int],
                                    k: int, allow_infeas_p: float,
                                    forbid_bin: int = -1) -> int:
        m = len(active_bins)
        wi = weights[i]
        best_b = -1
        best_metric = None
        for _ in range(k):
            b = active_bins[random.randrange(m)]
            if b == src or b == forbid_bin:
                continue
            new_ld = loads[b] + wi
            if new_ld <= C or random.random() < allow_infeas_p:
                ov = max(0, new_ld - C)
                slack = max(0, C - new_ld)
                metric = (ov, slack)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_b = b
        return best_b

    # ------------------------ Bin elimination move (no new bins) ------------------------
    def select_elim_bin(active_bins: List[int], bins: List[List[int]], loads: List[int], probes: int) -> int:
        # Prefer small cardinality and small load (easier to empty)
        best = active_bins[random.randrange(len(active_bins))]
        best_key = (len(bins[best]), loads[best])
        for _ in range(probes - 1):
            b = active_bins[random.randrange(len(active_bins))]
            kb = (len(bins[b]), loads[b])
            if kb < best_key:
                best, best_key = b, kb
        return best

    def try_remove_bin(b: int,
                       bins, loads, item_bin, pos_in_bin, active_bins, pos_active,
                       dest_k: int, allow_infeas_p: float,
                       chain_tries: int = 14) -> Tuple[bool, list]:
        # Attempt to empty bin b without creating a new bin.
        # Uses direct moves, then depth-1 ejection (swap and place ejected elsewhere).
        if pos_active[b] == -1 or not bins[b]:
            return False, []

        undo = []

        def u_move(ii: int, aa: int, bb: int) -> None:
            do_move(ii, aa, bb, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            undo.append(("m", ii, aa, bb))

        def u_swap(ii: int, aa: int, jj: int, bb: int) -> None:
            do_swap(ii, aa, jj, bb, bins, loads, item_bin, pos_in_bin)
            undo.append(("s", ii, aa, jj, bb))

        items = bins[b][:]
        items.sort(key=lambda it: weights[it], reverse=True)

        for x in items:
            if item_bin[x] != b:
                continue

            # direct move
            dest = choose_destination_for_item(x, b, active_bins, loads, dest_k, allow_infeas_p, forbid_bin=b)
            if dest != -1:
                u_move(x, b, dest)
                continue

            # ejection: pick a target bin t and swap with smaller y, then place y elsewhere
            swapped_out = False
            m = len(active_bins)
            for _ in range(chain_tries):
                t = active_bins[random.randrange(m)]
                if t == b or not bins[t]:
                    continue

                # pick y biased to smaller weights
                y = bins[t][random.randrange(len(bins[t]))]
                if weights[y] >= weights[x]:
                    continue

                # check x into t after removing y
                new_ld_t = loads[t] - weights[y] + weights[x]
                if new_ld_t > C and random.random() >= allow_infeas_p:
                    continue

                u_swap(x, b, y, t)

                # now place y out of b
                dest2 = choose_destination_for_item(y, b, active_bins, loads, dest_k, allow_infeas_p, forbid_bin=b)
                if dest2 != -1:
                    u_move(y, b, dest2)
                    swapped_out = True
                    break

                # undo that swap only
                op = undo.pop()
                _, ii, aa, jj, bb = op
                do_swap(ii, bb, jj, aa, bins, loads, item_bin, pos_in_bin)

            if swapped_out:
                continue

            # failure
            return False, undo

        return (not bins[b]), undo

    def undo_ops(undo, bins, loads, item_bin, pos_in_bin, active_bins, pos_active) -> None:
        for op in reversed(undo):
            if op[0] == "m":
                _, i, a, b = op
                do_move(i, b, a, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            else:
                _, i, a, j, b = op
                do_swap(i, b, j, a, bins, loads, item_bin, pos_in_bin)

    # ------------------------ SA run ------------------------
    def sa_run(state, run_budget: float) -> None:
        nonlocal global_best_bins, global_best_slack, global_best_snapshot

        bins, loads, item_bin, pos_in_bin, active_bins, pos_active = state
        if len(pos_active) < len(bins):
            pos_active.extend([-1] * (len(bins) - len(pos_active)))

        used, infeas, sec = full_eval(loads, active_bins)

        def current_slack() -> int:
            s = 0
            for b in active_bins:
                ld = loads[b]
                if ld <= C:
                    s += (C - ld)
            return s

        def store_best() -> None:
            nonlocal global_best_bins, global_best_slack, global_best_snapshot
            if infeas != 0:
                return
            u = len(active_bins)
            sl = current_slack()
            if (global_best_bins is None or u < global_best_bins or
                    (u == global_best_bins and sl < global_best_slack)):
                global_best_bins = u
                global_best_slack = sl
                global_best_snapshot = snapshot_from_state((bins, loads, item_bin, pos_in_bin, active_bins, pos_active))

        store_best()

        # score: bins*BIG + infeas*pen + sec
        # Penalty is adaptive and higher when T is low (push feasibility near the end)
        pen = 2e5

        def score(u: int, inf: int, ssec: int, T: float) -> float:
            # Slightly temperature-coupled penalty for stability
            return u * BIG + inf * (pen * (1.0 + 0.35 / max(1e-9, T))) + ssec

        # temperature calibration from random uphill deltas on sec primarily
        sample_d = []
        warmup = 250
        for _ in range(warmup):
            if len(active_bins) <= 1:
                break
            src = pick_light_bin(active_bins, loads, 4)
            if not bins[src]:
                continue
            i = bins[src][random.randrange(len(bins[src]))]
            dest = choose_destination_for_item(i, src, active_bins, loads, 10, 0.05)
            if dest == -1:
                continue
            old_u, old_inf, old_sec = used, infeas, sec
            old_score = score(old_u, old_inf, old_sec, 1.0)

            oldA = contrib_for_bin(src, bins, loads)
            oldB = contrib_for_bin(dest, bins, loads)
            do_move(i, src, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            new_u = len(active_bins)
            newA = contrib_for_bin(src, bins, loads)
            newB = contrib_for_bin(dest, bins, loads)
            new_inf = old_inf - oldA[1] - oldB[1] + newA[1] + newB[1]
            new_sec = old_sec - oldA[2] - oldB[2] + newA[2] + newB[2]
            new_score = score(new_u, new_inf, new_sec, 1.0)
            d = new_score - old_score
            do_move(i, dest, src, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            if d > 0:
                sample_d.append(d)

        if sample_d:
            sample_d.sort()
            med = sample_d[len(sample_d) // 2]
            T0 = max(1e-9, med / max(1e-9, -math.log(0.70)))
        else:
            T0 = max(1.0, 0.15 * C)

        run_deadline = min(deadline, time.time() + max(0.0, run_budget))

        # Fixed iterations + time checks
        max_iter = 25_000_000
        check_period = 5000

        T = T0
        Tf = max(1e-9, T0 * 2e-5)
        alpha = (Tf / T0) ** (1.0 / max(1, max_iter - 1))

        # Operator mixture
        # More elimination attempts once feasible and near best
        p_reloc = 0.52
        p_swap = 0.20
        p_exch = 0.18
        p_elim = 0.10

        dest_k_A = 14
        dest_k_B = 26
        allow_infeas_base = 0.06

        # infeasibility pressure control
        win = 3000
        infeas_hist = [0] * win
        infeas_sum = 0
        hidx = 0

        target_bins = None
        if global_best_bins is not None:
            target_bins = global_best_bins

        for it in range(max_iter):
            if (it % check_period) == 0:
                if time.time() >= run_deadline:
                    break

            # phase control
            frac = it / max_iter
            if frac < 0.30:
                dest_k = dest_k_A
                allow_infeas = allow_infeas_base
                phase_mult = 1.0
            else:
                dest_k = dest_k_B
                allow_infeas = allow_infeas_base * 0.45
                phase_mult = 0.7

            # update infeas window
            infeas_sum -= infeas_hist[hidx]
            infeas_hist[hidx] = 1 if infeas != 0 else 0
            infeas_sum += infeas_hist[hidx]
            hidx += 1
            if hidx >= win:
                hidx = 0

            if it % 250 == 0 and it >= win:
                frac_inf = infeas_sum / float(win)
                if frac_inf > 0.18:
                    pen *= 1.15
                elif frac_inf < 0.02:
                    pen *= 0.92
                pen = min(max(pen, 5e3), 2e9)

            # intensify bin reduction: if we have a feasible best, spend effort trying to remove one bin
            # by forcing elimination moves more often when feasible and close to target
            if infeas == 0 and global_best_bins is not None:
                if len(active_bins) <= global_best_bins + 1:
                    p_el = min(0.22, p_elim * 2.2)
                else:
                    p_el = p_elim
            else:
                p_el = p_elim * 0.55

            r = random.random()
            do_elim = (r < p_el and len(active_bins) >= 2)
            do_exch = (not do_elim and r < p_el + p_exch and len(active_bins) >= 2)
            do_sw = (not do_elim and not do_exch and r < p_el + p_exch + p_swap and len(active_bins) >= 2)

            old_u = len(active_bins)
            old_score = score(old_u, infeas, sec, max(1e-12, T))
            accepted = False

            if do_elim:
                b = select_elim_bin(active_bins, bins, loads, probes=10)
                if b != -1:
                    # elimination should NOT create bins; either empty b or revert
                    # allow slightly more infeasibility at higher T
                    allow = min(0.12, allow_infeas + 0.12 * (T / max(1e-9, T0)))
                    success, undo = try_remove_bin(b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active,
                                                   dest_k=dest_k, allow_infeas_p=allow,
                                                   chain_tries=16)
                    if success:
                        # incremental update hard; recompute (rare but impactful)
                        new_u, new_inf, new_sec = full_eval(loads, active_bins)
                        new_score = score(new_u, new_inf, new_sec, max(1e-12, T))
                        d = new_score - old_score
                        T_eff = max(1e-12, T * phase_mult)
                        if d <= 0 or random.random() < math.exp(-d / T_eff):
                            accepted = True
                            used, infeas, sec = new_u, new_inf, new_sec
                        else:
                            undo_ops(undo, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                    else:
                        undo_ops(undo, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

            elif do_exch:
                # 2-1 exchange: move x from light A to tight B, then move y from B elsewhere.
                A = pick_light_bin(active_bins, loads, 6)
                B = pick_tight_bin(active_bins, loads, 6)
                if A != -1 and B != -1 and A != B and bins[A] and bins[B]:
                    x = bins[A][random.randrange(len(bins[A]))]
                    # try a few y
                    for _ in range(5):
                        y = bins[B][random.randrange(len(bins[B]))]
                        if x == y:
                            continue

                        oldA = contrib_for_bin(A, bins, loads)
                        oldB = contrib_for_bin(B, bins, loads)
                        do_move(x, A, B, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                        dest = choose_destination_for_item(y, B, active_bins, loads, dest_k, allow_infeas, forbid_bin=B)
                        if dest == -1:
                            # cannot create bins in this operator (keeps pressure on reducing bins)
                            do_move(x, B, A, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                            continue

                        oldD = contrib_for_bin(dest, bins, loads)
                        do_move(y, B, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                        newA = contrib_for_bin(A, bins, loads)
                        newB = contrib_for_bin(B, bins, loads)
                        newD = contrib_for_bin(dest, bins, loads)

                        new_inf = infeas - oldA[1] - oldB[1] - oldD[1] + newA[1] + newB[1] + newD[1]
                        new_sec = sec - oldA[2] - oldB[2] - oldD[2] + newA[2] + newB[2] + newD[2]
                        new_u = len(active_bins)
                        new_score = score(new_u, new_inf, new_sec, max(1e-12, T))
                        d = new_score - old_score
                        T_eff = max(1e-12, T * phase_mult)
                        if d <= 0 or random.random() < math.exp(-d / T_eff):
                            accepted = True
                            infeas, sec = new_inf, new_sec
                            break
                        else:
                            # undo
                            do_move(y, dest, B, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                            do_move(x, B, A, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

            elif do_sw:
                A = pick_light_bin(active_bins, loads, 5)
                B = pick_tight_bin(active_bins, loads, 5)
                if A != -1 and B != -1 and A != B and bins[A] and bins[B]:
                    i = bins[A][random.randrange(len(bins[A]))]
                    j = bins[B][random.randrange(len(bins[B]))]
                    if i != j:
                        oldA = contrib_for_bin(A, bins, loads)
                        oldB = contrib_for_bin(B, bins, loads)
                        do_swap(i, A, j, B, bins, loads, item_bin, pos_in_bin)
                        newA = contrib_for_bin(A, bins, loads)
                        newB = contrib_for_bin(B, bins, loads)
                        new_inf = infeas - oldA[1] - oldB[1] + newA[1] + newB[1]
                        new_sec = sec - oldA[2] - oldB[2] + newA[2] + newB[2]
                        new_u = len(active_bins)
                        new_score = score(new_u, new_inf, new_sec, max(1e-12, T))
                        d = new_score - old_score
                        T_eff = max(1e-12, T * phase_mult)
                        if d <= 0 or random.random() < math.exp(-d / T_eff):
                            accepted = True
                            infeas, sec = new_inf, new_sec
                        else:
                            do_swap(i, B, j, A, bins, loads, item_bin, pos_in_bin)

            else:
                # relocation; allow creating new bins only when infeasible (repair) or early exploration
                create_new = (infeas != 0 and random.random() < 0.25) or (frac < 0.10 and random.random() < 0.03)

                src = -1
                if infeas != 0 and random.random() < 0.65:
                    src = pick_overloaded_bin(active_bins, loads, 10)
                if src == -1:
                    src = pick_light_bin(active_bins, loads, 7) if random.random() < 0.6 else pick_tight_bin(active_bins, loads, 7)

                if src != -1 and bins[src]:
                    i = bins[src][random.randrange(len(bins[src]))]
                    dest = choose_destination_for_item(i, src, active_bins, loads, dest_k, allow_infeas)
                    newbin = -1
                    if dest == -1:
                        if not create_new:
                            dest = None
                        else:
                            newbin = len(bins)
                            bins.append([])
                            loads.append(0)
                            pos_active.append(-1)
                            activate_bin(newbin, active_bins, pos_active)
                            dest = newbin

                    if dest is not None:
                        oldA = contrib_for_bin(src, bins, loads)
                        oldB = contrib_for_bin(dest, bins, loads) if newbin == -1 else (0, 0, 0)
                        do_move(i, src, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                        newA = contrib_for_bin(src, bins, loads)
                        newB = contrib_for_bin(dest, bins, loads)
                        new_inf = infeas - oldA[1] - oldB[1] + newA[1] + newB[1]
                        new_sec = sec - oldA[2] - oldB[2] + newA[2] + newB[2]
                        new_u = len(active_bins)
                        new_score = score(new_u, new_inf, new_sec, max(1e-12, T))
                        d = new_score - old_score
                        T_eff = max(1e-12, T * phase_mult)
                        if d <= 0 or random.random() < math.exp(-d / T_eff):
                            accepted = True
                            infeas, sec = new_inf, new_sec
                        else:
                            do_move(i, dest, src, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                            if newbin != -1:
                                deactivate_bin(newbin, active_bins, pos_active)

            if accepted:
                store_best()

                # If we improved global best bins, slightly reheat to explore the new basin
                if global_best_bins is not None and infeas == 0 and len(active_bins) == global_best_bins:
                    if T < T0 * 0.15:
                        T = T0 * 0.15

            # cooling
            T *= alpha

            # if we hit strong lower bound, keep searching but there's limited room
            # (no early stop; user wants full time usage)

    # ------------------------ Snapshot -> state and perturbation ------------------------
    def state_from_snapshot(snapshot) -> Tuple:
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

    def perturb_snapshot(snapshot, strength: int = 2) -> Tuple:
        state = state_from_snapshot(snapshot)
        bins, loads, item_bin, pos_in_bin, active_bins, pos_active = state
        if len(active_bins) <= 1:
            return state

        # remove items from a few light bins and reinsert
        chosen = []
        for _ in range(strength):
            chosen.append(pick_light_bin(active_bins, loads, 8))
        chosen = list({b for b in chosen if b != -1 and bins[b]})

        pool = []
        for b in chosen:
            for it in bins[b][:]:
                pool.append(it)
                remove_item(it, b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

        pool.sort(key=lambda it: weights[it], reverse=True)
        for it in pool:
            w = weights[it]
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
            append_item(it, best_b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

        return state

    # ------------------------ Multi-start schedule ------------------------
    # Use longer overall time if allowed (up to 100s) as per prompt.
    hard_cap = 100.0
    deadline = min(deadline, start + hard_cap)

    remaining = max(0.0, deadline - time.time())
    if remaining <= 0:
        # export best initial
        if global_best_snapshot is None:
            global_best_snapshot = snapshot_from_state(init_states[0])
        # export
        snap_bins, snap_loads = global_best_snapshot
        packing, bin_weights = [], []
        for b, lst in enumerate(snap_bins):
            if lst:
                packing.append(lst[:])
                bin_weights.append(snap_loads[b])
        return {"packing": packing, "bin_weights": bin_weights}

    # restarts based on available time
    tl = remaining
    if tl <= 2.0:
        restarts = 2
    elif tl <= 8.0:
        restarts = 5
    elif tl <= 25.0:
        restarts = 8
    else:
        restarts = 10

    # seed states
    seeds = init_states[:min(len(init_states), restarts)]

    for r in range(restarts):
        if time_up():
            break
        remaining = max(0.0, deadline - time.time())
        # keep a final intensification chunk
        future_runs = max(1, restarts - r)
        run_budget = remaining / future_runs

        if r < len(seeds):
            st = seeds[r]
        else:
            if global_best_snapshot is not None:
                st = perturb_snapshot(global_best_snapshot, strength=2 if random.random() < 0.7 else 3)
            else:
                st = construct_bfd(randomized_order(items_sorted), noise=0.03)

        sa_run(st, run_budget)

        # if we reached lb2, keep searching but bias to intensify from best
        if global_best_bins is not None and global_best_bins <= lb2 and (deadline - time.time()) > 0.2:
            pass

    # final intensification from best snapshot if time left
    if not time_up() and global_best_snapshot is not None:
        remaining = max(0.0, deadline - time.time())
        if remaining > 0.05:
            st = perturb_snapshot(global_best_snapshot, strength=1)
            sa_run(st, remaining)

    # ------------------------ Export ------------------------
    if global_best_snapshot is None:
        # guaranteed feasible greedy fallback
        bins = []
        loads = []
        for i in items_sorted:
            w = weights[i]
            best_b = -1
            best_rem = None
            for b, ld in enumerate(loads):
                if ld + w <= C:
                    rem = C - (ld + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_b = b
            if best_b == -1:
                best_b = len(bins)
                bins.append([])
                loads.append(0)
            bins[best_b].append(i)
            loads[best_b] += w
        return {"packing": bins, "bin_weights": loads}

    snap_bins, snap_loads = global_best_snapshot
    packing: List[List[int]] = []
    bin_weights: List[int] = []
    for b, lst in enumerate(snap_bins):
        if lst:
            packing.append(lst[:])
            bin_weights.append(snap_loads[b])
    return {"packing": packing, "bin_weights": bin_weights}
