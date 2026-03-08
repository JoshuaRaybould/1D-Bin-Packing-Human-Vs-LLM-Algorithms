import time
import math
import random
from typing import List, Dict, Tuple


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

    # ------------------------ Utilities ------------------------
    def time_up() -> bool:
        return time.time() >= deadline

    # Stable bin indices: bins and loads arrays grow; bins are never deleted during SA.
    # active_bins holds indices with non-empty bins.

    def make_empty_state() -> Tuple[List[List[int]], List[int], List[int], List[int], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []
        item_bin = [-1] * n
        pos_in_bin = [-1] * n
        active_bins: List[int] = []
        pos_active: List[int] = []
        return bins, loads, item_bin, pos_in_bin, active_bins, pos_active

    # ------------------------ Constructive heuristics ------------------------
    def construct(order: List[int], mode: str) -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], List[int]]:
        # mode in {"BFD", "FFD"}
        bins: List[List[int]] = []
        loads: List[int] = []
        item_bin = [-1] * n
        pos_in_bin = [-1] * n
        active_bins: List[int] = []
        pos_active: List[int] = []

        def activate(b: int) -> None:
            pos_active.append(len(active_bins))
            active_bins.append(b)

        for i in order:
            w = weights[i]
            best_b = -1
            best_rem = None

            if mode == "FFD":
                for b in active_bins:
                    ld = loads[b]
                    if ld + w <= C:
                        best_b = b
                        break
            else:  # BFD
                for b in active_bins:
                    ld = loads[b]
                    if ld + w <= C:
                        rem = C - (ld + w)
                        if best_rem is None or rem < best_rem:
                            best_rem = rem
                            best_b = b

            if best_b == -1:
                b = len(bins)
                bins.append([])
                loads.append(0)
                activate(b)
                best_b = b

            # append
            pos_in_bin[i] = len(bins[best_b])
            bins[best_b].append(i)
            loads[best_b] += w
            item_bin[i] = best_b

        return bins, loads, item_bin, pos_in_bin, active_bins, pos_active

    def randomized_bfd_order(items_sorted: List[int]) -> List[int]:
        # Shuffle within equal-weight blocks to diversify.
        order = items_sorted[:]
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

    items_sorted = list(range(n))
    items_sorted.sort(key=lambda i: weights[i], reverse=True)

    init_states = []
    init_states.append(construct(items_sorted, "BFD"))
    init_states.append(construct(items_sorted, "FFD"))
    init_states.append(construct(randomized_bfd_order(items_sorted), "BFD"))

    def state_key(state) -> Tuple[int, int]:
        _, loads, _, _, active_bins, _ = state
        used = len(active_bins)
        slack = 0
        for b in active_bins:
            slack += (C - loads[b])
        return used, slack

    init_states.sort(key=state_key)

    # ------------------------ Core SA machinery ------------------------
    # Contribution functions for incremental objective
    def bin_over(ld: int) -> int:
        return ld - C if ld > C else 0

    def bin_sec(ld: int) -> int:
        # Tightness measure only for feasible bins.
        if ld <= C:
            s = C - ld
            return s * s
        return 0

    # Scalar-encoded lexicographic objective:
    # score = bins_used*BIG + infeas*pen_w + secondary
    BIG = 10**9

    # --- fast active bins operations ---
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

    def append_item(i: int, b: int, bins: List[List[int]], loads: List[int], item_bin: List[int], pos_in_bin: List[int],
                    active_bins: List[int], pos_active: List[int]) -> None:
        if not bins[b]:
            activate_bin(b, active_bins, pos_active)
        pos_in_bin[i] = len(bins[b])
        bins[b].append(i)
        item_bin[i] = b
        loads[b] += weights[i]

    def remove_item(i: int, b: int, bins: List[List[int]], loads: List[int], item_bin: List[int], pos_in_bin: List[int],
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

    # Move and swap are implemented with O(1) position updates.
    def do_move(i: int, a: int, b: int, bins, loads, item_bin, pos_in_bin, active_bins, pos_active) -> None:
        remove_item(i, a, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
        append_item(i, b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

    def do_swap(i: int, a: int, j: int, b: int, bins, loads, item_bin, pos_in_bin) -> None:
        # assumes a != b
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

    # --- objective bookkeeping ---
    def full_eval(loads: List[int], active_bins: List[int]) -> Tuple[int, int, int, int]:
        used = len(active_bins)
        infeas = 0
        sec = 0
        for b in active_bins:
            ld = loads[b]
            ov = bin_over(ld)
            infeas += ov
            sec += bin_sec(ld)
        score = used * BIG + sec
        return used, infeas, sec, score

    # ------------------------ Sampling helpers (biased) ------------------------
    def pick_light_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        # among k random active bins pick min load
        ab = active_bins
        m = len(ab)
        if m == 0:
            return -1
        best = ab[random.randrange(m)]
        best_ld = loads[best]
        for _ in range(k - 1):
            b = ab[random.randrange(m)]
            ld = loads[b]
            if ld < best_ld:
                best, best_ld = b, ld
        return best

    def pick_overloaded_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        ab = active_bins
        m = len(ab)
        if m == 0:
            return -1
        best = -1
        best_ov = 0
        for _ in range(k):
            b = ab[random.randrange(m)]
            ov = loads[b] - C
            if ov > best_ov:
                best_ov = ov
                best = b
        return best

    def pick_tight_bin(active_bins: List[int], loads: List[int], k: int) -> int:
        # pick max load but prefer feasible
        ab = active_bins
        m = len(ab)
        if m == 0:
            return -1
        best = ab[random.randrange(m)]
        best_ld = loads[best]
        for _ in range(k - 1):
            b = ab[random.randrange(m)]
            ld = loads[b]
            # prioritize feasible higher loads
            if (ld <= C and best_ld > C) or (ld <= C and best_ld <= C and ld > best_ld) or (ld > C and best_ld > C and ld > best_ld):
                best, best_ld = b, ld
        return best

    def choose_destination_for_item(i: int, src: int, active_bins: List[int], loads: List[int], k: int,
                                    allow_infeas_p: float) -> int:
        # best-of-k destinations among active bins (excluding src); may also return a new bin index.
        m = len(active_bins)
        wi = weights[i]

        # small chance: try new bin
        best_b = -1
        best_metric = None
        if random.random() < 0.03:
            return -1

        for _ in range(k):
            b = active_bins[random.randrange(m)]
            if b == src:
                continue
            new_ld = loads[b] + wi
            if new_ld <= C or random.random() < allow_infeas_p:
                # metric: prefer smaller overload then smaller slack
                ov = max(0, new_ld - C)
                slack = max(0, C - new_ld)
                metric = (ov, slack)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_b = b

        return best_b

    # ------------------------ Elimination operator (bounded ejection chain) ------------------------
    def select_elim_bin(active_bins: List[int], bins: List[List[int]], loads: List[int], probes: int) -> int:
        # Prefer small #items, then small load, then max item weight (smaller is easier)
        if not active_bins:
            return -1
        best_b = active_bins[random.randrange(len(active_bins))]
        def key(b: int):
            lst = bins[b]
            mx = 0
            for it in lst:
                w = weights[it]
                if w > mx:
                    mx = w
            return (len(lst), loads[b], mx)

        best_k = key(best_b)
        for _ in range(probes - 1):
            b = active_bins[random.randrange(len(active_bins))]
            kb = key(b)
            if kb < best_k:
                best_b, best_k = b, kb
        return best_b

    def attempt_eliminate_bin(b: int,
                             bins, loads, item_bin, pos_in_bin, active_bins, pos_active,
                             allow_infeas_p: float, dest_k: int) -> Tuple[bool, list]:
        # Try to empty bin b by relocating its items with bounded ejection chain (depth 1).
        # Returns (success_applied, undo_stack)
        if pos_active[b] == -1 or not bins[b]:
            return False, []

        items = bins[b][:]
        items.sort(key=lambda it: weights[it], reverse=True)
        undo = []

        # helper to do move with undo
        def apply_move(ii: int, aa: int, bb: int) -> None:
            do_move(ii, aa, bb, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            undo.append(("move", ii, aa, bb))

        def apply_swap(ii: int, aa: int, jj: int, bb: int) -> None:
            do_swap(ii, aa, jj, bb, bins, loads, item_bin, pos_in_bin)
            undo.append(("swap", ii, aa, jj, bb))

        # We'll keep trying to place each item out of b.
        for x in items:
            if item_bin[x] != b:
                continue  # may have moved due to earlier swaps

            # 1) Direct insertion
            dest = choose_destination_for_item(x, b, active_bins, loads, dest_k, allow_infeas_p)
            if dest != -1:
                apply_move(x, b, dest)
                continue

            # 2) Ejection chain depth 1: swap x into some bin by kicking out a smaller item y.
            # Sample some candidate bins and some candidate y in them.
            tried = 0
            swapped = False
            m = len(active_bins)
            for _ in range(min(dest_k, 12)):
                if m == 0:
                    break
                t = active_bins[random.randrange(m)]
                if t == b or not bins[t]:
                    continue
                # choose a random candidate y from t (biased to smaller)
                y = bins[t][random.randrange(len(bins[t]))]
                if weights[y] >= weights[x]:
                    tried += 1
                    if tried > 20:
                        break
                    continue

                # Check if x would fit in t after removing y
                if loads[t] - weights[y] + weights[x] > C and random.random() >= allow_infeas_p:
                    tried += 1
                    if tried > 20:
                        break
                    continue

                # Perform swap x<->y: x goes to t, y comes to b
                apply_swap(x, b, y, t)

                # Now try to move y out of b directly
                dest2 = choose_destination_for_item(y, b, active_bins, loads, dest_k, allow_infeas_p)
                if dest2 != -1:
                    apply_move(y, b, dest2)
                    swapped = True
                    break

                # If can't place y, undo the swap immediately (local undo)
                last = undo.pop()
                # reverse swap
                _, ii, aa, jj, bb = last
                do_swap(ii, bb, jj, aa, bins, loads, item_bin, pos_in_bin)
                swapped = False

                tried += 1
                if tried > 20:
                    break

            if swapped:
                continue

            # 3) Open a new bin for x (this likely defeats elimination, but SA may accept the whole move)
            newb = len(bins)
            bins.append([])
            loads.append(0)
            pos_active.append(-1)
            activate_bin(newb, active_bins, pos_active)
            undo.append(("newbin", newb))
            apply_move(x, b, newb)

        # success means b is empty
        success = (not bins[b])
        return success, undo

    def undo_ops(undo, bins, loads, item_bin, pos_in_bin, active_bins, pos_active) -> None:
        # Undo in reverse
        for op in reversed(undo):
            if op[0] == "move":
                _, i, a, b = op
                # move back (item currently in b)
                do_move(i, b, a, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
            elif op[0] == "swap":
                _, i, a, j, b = op
                # swap back (currently i in b, j in a)
                do_swap(i, b, j, a, bins, loads, item_bin, pos_in_bin)
            else:  # newbin
                # We don't delete bins. Just ensure it's empty and deactivate.
                _, nb = op
                if bins[nb]:
                    # Should not happen if moves were undone; but be safe.
                    for it in bins[nb][:]:
                        remove_item(it, nb, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                deactivate_bin(nb, active_bins, pos_active)

    # ------------------------ SA run ------------------------
    def sa_run(state, run_time_budget: float, global_best) -> Tuple[Tuple, Tuple]:
        bins, loads, item_bin, pos_in_bin, active_bins, pos_active = state

        # ensure pos_active length equals bins length
        if len(pos_active) < len(bins):
            pos_active.extend([-1] * (len(bins) - len(pos_active)))

        used, infeas, sec, _ = full_eval(loads, active_bins)

        best_bins_used, best_slack, best_snapshot = global_best

        def current_slack() -> int:
            s = 0
            for b in active_bins:
                ld = loads[b]
                if ld <= C:
                    s += (C - ld)
            return s

        def store_best_if_feasible() -> None:
            nonlocal best_bins_used, best_slack, best_snapshot
            if infeas != 0:
                return
            u = len(active_bins)
            sl = current_slack()
            if best_bins_used is None or u < best_bins_used or (u == best_bins_used and sl < best_slack):
                best_bins_used = u
                best_slack = sl
                # snapshot bins content and loads for export (keep stable bin indices)
                snap_bins = [lst[:] for lst in bins]
                snap_loads = loads[:]
                best_snapshot = (snap_bins, snap_loads)

        store_best_if_feasible()

        # Adaptive infeasibility penalty weight
        pen_w = max(1.0, 1e5)  # MID-ish base

        # Calibrate initial temperature from sampled positive deltas
        def score_val(u: int, inf: int, ssec: int) -> float:
            return u * BIG + inf * pen_w + ssec

        def bin_contrib(ld: int) -> Tuple[int, int]:
            return bin_over(ld), bin_sec(ld)

        def delta_two_bins(a: int, b: int, old_used: int) -> Tuple[int, int, int, int]:
            # returns (du, dinf, dsec, new_used)
            # old_used given to compute du by active status change
            # Compute old active flags
            a_old_active = (len(bins[a]) > 0)
            b_old_active = (len(bins[b]) > 0)
            a_new_active = a_old_active
            b_new_active = b_old_active
            # Used computed from active_bins, but for incremental we infer changes:
            # This function is used *after* tentative modifications, so we must pass old flags externally.
            raise RuntimeError

        # We instead compute deltas by explicitly computing old contributions for affected bins before move,
        # then new contributions after move, and update used using old/new emptiness.
        def contrib_for_bin(b: int) -> Tuple[int, int, int]:
            # (active, over, sec)
            if bins[b]:
                ld = loads[b]
                return 1, bin_over(ld), bin_sec(ld)
            return 0, 0, 0

        # Warmup to estimate T0
        sample_d = []
        warmup = 300
        for _ in range(warmup):
            if len(active_bins) <= 1:
                break
            # random relocate proposal
            src = pick_light_bin(active_bins, loads, 3)
            if src == -1 or not bins[src]:
                continue
            i = bins[src][random.randrange(len(bins[src]))]
            dest = choose_destination_for_item(i, src, active_bins, loads, 8, 0.05)
            if dest == -1:
                dest = len(bins)
                # simulate new bin
                old_a = contrib_for_bin(src)
                old_used = len(active_bins)
                old_score = score_val(old_used, infeas, sec)

                # apply
                bins.append([])
                loads.append(0)
                pos_active.append(-1)
                activate_bin(dest, active_bins, pos_active)
                do_move(i, src, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                # new
                new_used = len(active_bins)
                new_infeas = infeas
                new_sec = sec
                # update infeas/sec only for src and dest (dest was empty -> now active)
                # recompute via contribs
                new_a = contrib_for_bin(src)
                new_b = contrib_for_bin(dest)
                # old dest was inactive => (0,0,0)
                # update infeas/sec
                new_infeas = infeas - old_a[1] + new_a[1] + new_b[1]
                new_sec = sec - old_a[2] + new_a[2] + new_b[2]
                new_score = score_val(new_used, new_infeas, new_sec)
                d = new_score - old_score

                # undo
                do_move(i, dest, src, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                deactivate_bin(dest, active_bins, pos_active)
                # keep arrays (stable), but ensure empty
                # (bins[dest] should be empty now)

                if d > 0:
                    sample_d.append(d)
            else:
                # simulate move between two existing bins
                old_used = len(active_bins)
                old_score = score_val(old_used, infeas, sec)
                old_a = contrib_for_bin(src)
                old_b = contrib_for_bin(dest)

                do_move(i, src, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                new_used = len(active_bins)
                new_a = contrib_for_bin(src)
                new_b = contrib_for_bin(dest)
                new_infeas = infeas - old_a[1] - old_b[1] + new_a[1] + new_b[1]
                new_sec = sec - old_a[2] - old_b[2] + new_a[2] + new_b[2]
                new_score = score_val(new_used, new_infeas, new_sec)
                d = new_score - old_score

                # undo
                do_move(i, dest, src, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                if d > 0:
                    sample_d.append(d)

        if sample_d:
            sample_d.sort()
            med = sample_d[len(sample_d) // 2]
            # target acceptance for median uphill move ~0.65
            T0 = max(1e-9, med / max(1e-9, -math.log(0.65)))
        else:
            T0 = max(1.0, 0.2 * C)

        # Phases: diversify then intensify
        # Time-driven but still with a fixed iteration cap per run.
        max_iter = 20_000_000  # fixed; time limit will cut it
        time_check_period = 4000

        # Phase split
        phaseA = 0.25
        phaseA_iters = int(max_iter * phaseA)

        T = T0
        Tf = max(1e-9, T0 * 1e-4)
        # default geometric cooling; phase adjustments apply multipliers
        alpha = (Tf / T0) ** (1.0 / max(1, max_iter - 1))

        # Adaptive infeas tracking
        win = 2000
        infeas_hist = [0] * win
        infeas_sum = 0
        hist_idx = 0

        last_best_bins_improve = 0
        last_any_best_update = 0

        # Operator probabilities (tuned toward relocation & exchanges)
        p_reloc = 0.60
        p_swap = 0.20
        p_exch = 0.15
        p_elim = 0.05

        # Destination sampling size
        dest_k_A = 12
        dest_k_B = 22

        # Allow infeasible proposals probability
        allow_infeas_p = 0.06

        run_deadline = min(deadline, time.time() + max(0.0, run_time_budget))

        def update_infeas_window(is_infeas: int) -> None:
            nonlocal infeas_sum, hist_idx
            infeas_sum -= infeas_hist[hist_idx]
            infeas_hist[hist_idx] = is_infeas
            infeas_sum += is_infeas
            hist_idx += 1
            if hist_idx >= win:
                hist_idx = 0

        for it in range(max_iter):
            if (it % time_check_period) == 0:
                if time.time() >= run_deadline:
                    break

            # Phase behavior
            if it < phaseA_iters:
                dest_k = dest_k_A
                # higher temperature effective
                phase_temp_mult = 1.0
                # more exploration
                allow_infeas = allow_infeas_p
                p_el = p_elim * 0.6
            else:
                dest_k = dest_k_B
                phase_temp_mult = 0.6
                allow_infeas = allow_infeas_p * 0.6
                # intensify elimination when close to best/LB
                if best_bins_used is not None and len(active_bins) <= best_bins_used + 1:
                    p_el = min(0.12, p_elim * 2.0)
                else:
                    p_el = p_elim

            # Adaptive infeas penalty
            update_infeas_window(1 if infeas != 0 else 0)
            if it % 200 == 0 and it >= win:
                frac = infeas_sum / float(win)
                if frac > 0.20:
                    pen_w *= 1.12
                elif frac < 0.02 and (it - last_any_best_update) > 5000:
                    pen_w *= 0.93
                pen_w = min(max(pen_w, 1e3), 1e9)

            # Reheat tied to progress
            if best_bins_used is not None:
                if (it - last_best_bins_improve) > 120_000 and it > phaseA_iters:
                    # reheat based on recent median uphill if available
                    T = max(T, T0 * 0.35)
                    last_best_bins_improve = it

            # Choose operator
            r = random.random()
            do_elimination = (r < p_el)
            do_exchange = (not do_elimination) and (r < p_el + p_exch)
            do_swap_op = (not do_elimination) and (not do_exchange) and (r < p_el + p_exch + p_swap)

            old_used = len(active_bins)
            old_score = score_val(old_used, infeas, sec)
            accepted = False

            if do_elimination and len(active_bins) >= 2:
                # Only worth trying when not far from best, and not already at LB2
                if best_bins_used is not None and best_bins_used <= lb2:
                    # already at strong LB; keep searching but reduce elimination frequency implicitly
                    pass
                # choose bin
                b = select_elim_bin(active_bins, bins, loads, probes=8)
                if b != -1:
                    # old contributions for bins that might change: we don't know ahead, but we can use undo and recompute fully
                    # However to stay fast, we do incremental updates for bins touched via undo stack tracking.
                    # We'll just recompute full after elimination attempt since it's rare (5-12%).
                    success, undo = attempt_eliminate_bin(b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active,
                                                         allow_infeas, dest_k)
                    new_used = len(active_bins)
                    # Full recompute: elimination is rare but big
                    new_used2, new_infeas, new_sec, _ = full_eval(loads, active_bins)
                    new_score = score_val(new_used2, new_infeas, new_sec)
                    d = new_score - old_score
                    T_eff = max(1e-12, T * phase_temp_mult)
                    if d <= 0 or random.random() < math.exp(-d / T_eff):
                        accepted = True
                        used, infeas, sec = new_used2, new_infeas, new_sec
                    else:
                        undo_ops(undo, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                        # after undo, (used,infeas,sec) unchanged

            elif do_exchange and len(active_bins) >= 2:
                # 2-1 exchange: move x from light bin A into B, and move y from B into A or elsewhere.
                A = pick_light_bin(active_bins, loads, 5)
                if A != -1 and bins[A]:
                    x = bins[A][random.randrange(len(bins[A]))]
                    # pick a target bin B (tight-ish)
                    B = pick_tight_bin(active_bins, loads, 6)
                    if B != -1 and B != A and bins[B]:
                        # pick candidate y from B (try a few)
                        ok = False
                        for _ in range(4):
                            y = bins[B][random.randrange(len(bins[B]))]
                            if y == x:
                                continue

                            # We'll try: move x A->B and y B->(best dest not B)
                            oldA = contrib_for_bin(A)
                            oldB = contrib_for_bin(B)

                            # apply x to B
                            do_move(x, A, B, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                            # choose destination for y (avoid B)
                            # temporarily treat src=B
                            dest = choose_destination_for_item(y, B, active_bins, loads, dest_k, allow_infeas)
                            if dest == -1:
                                # open new bin
                                dest = len(bins)
                                bins.append([])
                                loads.append(0)
                                pos_active.append(-1)
                                activate_bin(dest, active_bins, pos_active)
                                newbin_created = True
                            else:
                                newbin_created = False

                            do_move(y, B, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                            new_used = len(active_bins)
                            newA = contrib_for_bin(A)
                            newB = contrib_for_bin(B)
                            newD = contrib_for_bin(dest)

                            # old dest contrib
                            # If new bin created, old contrib was (0,0,0). Else old is from before move.
                            if newbin_created:
                                oldD = (0, 0, 0)
                            else:
                                # dest existed; compute old after the fact isn't possible, so we approximate by recomputing full
                                # To keep correctness, just full recompute on this operator (moderate frequency).
                                oldD = None

                            if oldD is None:
                                new_used2, new_infeas, new_sec, _ = full_eval(loads, active_bins)
                                new_score = score_val(new_used2, new_infeas, new_sec)
                                d = new_score - old_score
                                T_eff = max(1e-12, T * phase_temp_mult)
                                if d <= 0 or random.random() < math.exp(-d / T_eff):
                                    accepted = True
                                    used, infeas, sec = new_used2, new_infeas, new_sec
                                    ok = True
                                else:
                                    # undo: move y back, move x back, possibly deactivate new bin
                                    do_move(y, dest, B, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                                    do_move(x, B, A, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                                    if newbin_created:
                                        deactivate_bin(dest, active_bins, pos_active)
                                break
                            else:
                                # (not used)
                                pass
                        # if not ok, nothing accepted

            elif do_swap_op and len(active_bins) >= 2:
                # Swap items between two bins
                A = pick_light_bin(active_bins, loads, 4)
                B = pick_tight_bin(active_bins, loads, 4)
                if A != -1 and B != -1 and A != B and bins[A] and bins[B]:
                    i = bins[A][random.randrange(len(bins[A]))]
                    j = bins[B][random.randrange(len(bins[B]))]
                    if i != j:
                        oldA = contrib_for_bin(A)
                        oldB = contrib_for_bin(B)
                        do_swap(i, A, j, B, bins, loads, item_bin, pos_in_bin)
                        newA = contrib_for_bin(A)
                        newB = contrib_for_bin(B)

                        new_infeas = infeas - oldA[1] - oldB[1] + newA[1] + newB[1]
                        new_sec = sec - oldA[2] - oldB[2] + newA[2] + newB[2]
                        new_used = len(active_bins)
                        new_score = score_val(new_used, new_infeas, new_sec)
                        d = new_score - old_score
                        T_eff = max(1e-12, T * phase_temp_mult)
                        if d <= 0 or random.random() < math.exp(-d / T_eff):
                            accepted = True
                            infeas, sec = new_infeas, new_sec
                        else:
                            do_swap(i, B, j, A, bins, loads, item_bin, pos_in_bin)

            else:
                # Relocate: biased source selection
                src = -1
                if infeas != 0 and random.random() < 0.55:
                    src = pick_overloaded_bin(active_bins, loads, 8)
                if src == -1:
                    src = pick_light_bin(active_bins, loads, 6) if random.random() < 0.65 else pick_tight_bin(active_bins, loads, 6)

                if src != -1 and bins[src]:
                    i = bins[src][random.randrange(len(bins[src]))]

                    dest = choose_destination_for_item(i, src, active_bins, loads, dest_k, allow_infeas)
                    newbin_created = False
                    if dest == -1:
                        dest = len(bins)
                        bins.append([])
                        loads.append(0)
                        pos_active.append(-1)
                        activate_bin(dest, active_bins, pos_active)
                        newbin_created = True

                    oldA = contrib_for_bin(src)
                    oldB = contrib_for_bin(dest) if not newbin_created else (0, 0, 0)

                    do_move(i, src, dest, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

                    newA = contrib_for_bin(src)
                    newB = contrib_for_bin(dest)

                    new_infeas = infeas - oldA[1] - oldB[1] + newA[1] + newB[1]
                    new_sec = sec - oldA[2] - oldB[2] + newA[2] + newB[2]
                    new_used = len(active_bins)
                    new_score = score_val(new_used, new_infeas, new_sec)
                    d = new_score - old_score

                    T_eff = max(1e-12, T * phase_temp_mult)
                    if d <= 0 or random.random() < math.exp(-d / T_eff):
                        accepted = True
                        infeas, sec = new_infeas, new_sec
                    else:
                        do_move(i, dest, src, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)
                        if newbin_created:
                            deactivate_bin(dest, active_bins, pos_active)

            if accepted:
                store_best_if_feasible()
                last_any_best_update = it
                if best_bins_used is not None and best_bins_used == len(active_bins) and infeas == 0:
                    # if we just reached a new best bin count
                    if best_bins_used < old_used:
                        last_best_bins_improve = it

            # cooling
            T *= alpha

            # Optional early stop if we reached strong lower bound (lb2)
            if best_bins_used is not None and best_bins_used <= lb2 and it > phaseA_iters:
                # keep going a bit but not necessary; still return on time
                pass

        return (bins, loads, item_bin, pos_in_bin, active_bins, pos_active), (best_bins_used, best_slack, best_snapshot)

    # ------------------------ Multi-start orchestration ------------------------
    # Global best feasible snapshot stored as (bins_list, loads_list) from some state
    global_best_bins = None
    global_best_slack = None
    global_best_snapshot = None

    def consider_init_for_global(state) -> None:
        nonlocal global_best_bins, global_best_slack, global_best_snapshot
        bins, loads, _, _, active_bins, _ = state
        used = len(active_bins)
        infeas = 0
        slack = 0
        for b in active_bins:
            ld = loads[b]
            if ld > C:
                infeas += (ld - C)
            else:
                slack += (C - ld)
        if infeas != 0:
            return
        if global_best_bins is None or used < global_best_bins or (used == global_best_bins and slack < global_best_slack):
            global_best_bins = used
            global_best_slack = slack
            global_best_snapshot = ([lst[:] for lst in bins], loads[:])

    for st in init_states:
        consider_init_for_global(st)

    # Decide number of restarts based on time
    # Keep fixed iteration cap inside each run; time checks will terminate.
    tl = max(0.0, float(time_limit))
    if tl <= 1.0:
        restarts = 1
    elif tl <= 5.0:
        restarts = 3
    else:
        restarts = 5

    # Run budgets
    remaining = max(0.0, deadline - time.time())
    if remaining <= 0:
        restarts = 0

    # Use initial states as seeds; if more restarts, perturb best.
    seeds = init_states[:]

    def perturb_from_snapshot(snapshot) -> Tuple:
        # perturb by emptying 1-2 light bins and reinserting greedily
        snap_bins, snap_loads = snapshot
        # rebuild full SA state structures
        bins = [lst[:] for lst in snap_bins]
        loads = snap_loads[:]
        item_bin = [-1] * n
        pos_in_bin = [-1] * n
        active_bins = []
        pos_active = [-1] * len(bins)

        for b, lst in enumerate(bins):
            if lst:
                pos_active[b] = len(active_bins)
                active_bins.append(b)
            for p, it in enumerate(lst):
                item_bin[it] = b
                pos_in_bin[it] = p

        if not active_bins:
            # fallback
            return construct(items_sorted, "BFD")

        # choose bins to empty
        k = 1 if random.random() < 0.6 else 2
        chosen = []
        for _ in range(k):
            chosen.append(pick_light_bin(active_bins, loads, 6))
        chosen = list({b for b in chosen if b != -1})

        pool = []
        for b in chosen:
            for it in bins[b][:]:
                pool.append(it)
                remove_item(it, b, bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

        pool.sort(key=lambda it: weights[it], reverse=True)
        for it in pool:
            w = weights[it]
            # best fit
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

        return (bins, loads, item_bin, pos_in_bin, active_bins, pos_active)

    # Main restart loop
    for r in range(restarts):
        if time_up():
            break
        remaining = max(0.0, deadline - time.time())
        run_budget = remaining / max(1, (restarts - r))

        if r < len(seeds):
            state = seeds[r]
        else:
            if global_best_snapshot is not None:
                state = perturb_from_snapshot(global_best_snapshot)
            else:
                state = init_states[0]

        global_best = (global_best_bins, global_best_slack, global_best_snapshot)
        _, global_best = sa_run(state, run_budget, global_best)
        global_best_bins, global_best_slack, global_best_snapshot = global_best

        if global_best_bins is not None and global_best_bins <= lb2:
            # likely optimal or very tight; continue until time but nothing else required
            pass

    # ------------------------ Export best feasible (with repair fallback) ------------------------
    def export_from_snapshot(snapshot) -> dict:
        snap_bins, snap_loads = snapshot
        packing = []
        bin_weights = []
        for b, lst in enumerate(snap_bins):
            if lst:
                packing.append(lst[:])
                bin_weights.append(snap_loads[b])
        return {"packing": packing, "bin_weights": bin_weights}

    if global_best_snapshot is not None:
        return export_from_snapshot(global_best_snapshot)

    # Fallback: repair from best init (should be feasible anyway), else greedy pack
    # Build a guaranteed feasible solution by greedy into new bins
    order = items_sorted
    bins = []
    loads = []
    for i in order:
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
