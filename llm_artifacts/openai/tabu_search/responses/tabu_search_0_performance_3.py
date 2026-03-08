import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)

    # Use up to 100s if allowed; caller may still pass 40s, but statement says execution stops at 100.
    time_limit = float(min(100.0, time_limit))

    start = time.time()

    def elapsed() -> float:
        return time.time() - start

    # ----------------------------
    # Data structure / invariants
    # ----------------------------
    # bins[b] : list of item indices in bin b
    # loads[b]: total weight in bin b
    # assign[i]: bin index containing item i
    # pos[i]: position of item i within bins[assign[i]]
    # active_bins: list of bin indices with non-empty bins
    # active_pos[b]: position of b inside active_bins, or -1 if empty

    def make_empty_solution_structs(num_bins_hint: int = 0):
        bins: List[List[int]] = [[] for _ in range(num_bins_hint)]
        loads: List[int] = [0 for _ in range(num_bins_hint)]
        assign: List[int] = [-1] * n
        pos: List[int] = [-1] * n
        active_bins: List[int] = []
        active_pos: List[int] = [-1] * num_bins_hint
        empty_bins: List[int] = []
        return bins, loads, assign, pos, active_bins, active_pos, empty_bins

    def ensure_bin_exists(b: int, bins: List[List[int]], loads: List[int], active_pos: List[int]):
        if b >= len(bins):
            extra = b + 1 - len(bins)
            bins.extend([[] for _ in range(extra)])
            loads.extend([0 for _ in range(extra)])
            active_pos.extend([-1 for _ in range(extra)])

    def activate_bin(b: int, bins: List[List[int]], active_bins: List[int], active_pos: List[int]):
        if active_pos[b] == -1 and bins[b]:
            active_pos[b] = len(active_bins)
            active_bins.append(b)

    def deactivate_bin(b: int, bins: List[List[int]], loads: List[int], active_bins: List[int], active_pos: List[int], empty_bins: List[int]):
        if active_pos[b] != -1:
            p = active_pos[b]
            lastb = active_bins[-1]
            active_bins[p] = lastb
            active_pos[lastb] = p
            active_bins.pop()
            active_pos[b] = -1
        loads[b] = 0
        empty_bins.append(b)

    def add_item_to_bin(i: int, b: int,
                        bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int],
                        active_bins: List[int], active_pos: List[int]):
        bins[b].append(i)
        pos[i] = len(bins[b]) - 1
        assign[i] = b
        loads[b] += w[i]
        activate_bin(b, bins, active_bins, active_pos)

    def remove_item_from_bin(i: int, b: int,
                             bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int],
                             active_bins: List[int], active_pos: List[int], empty_bins: List[int]) -> None:
        p = pos[i]
        last = bins[b][-1]
        bins[b][p] = last
        pos[last] = p
        bins[b].pop()
        pos[i] = -1
        assign[i] = -1
        loads[b] -= w[i]
        if not bins[b]:
            deactivate_bin(b, bins, loads, active_bins, active_pos, empty_bins)

    # ----------------------------
    # Initial solution generation (multi-start BFD/FFD + randomized)
    # ----------------------------
    def pack_greedy(order: List[int], mode: str) -> Tuple[List[List[int]], List[int]]:
        bins0: List[List[int]] = []
        loads0: List[int] = []
        for i in order:
            wi = w[i]
            chosen = -1
            if mode == "FFD":
                for b in range(len(bins0)):
                    if loads0[b] + wi <= C:
                        chosen = b
                        break
            else:  # BFD
                best_rem = C + 1
                for b in range(len(bins0)):
                    rem = C - loads0[b]
                    if wi <= rem:
                        nrem = rem - wi
                        if nrem < best_rem:
                            best_rem = nrem
                            chosen = b
                            if best_rem == 0:
                                break
            if chosen == -1:
                bins0.append([i])
                loads0.append(wi)
            else:
                bins0[chosen].append(i)
                loads0[chosen] += wi
        return bins0, loads0

    def build_struct_from_bins(bins0: List[List[int]]):
        bins, loads, assign, pos, active_bins, active_pos, empty_bins = make_empty_solution_structs(len(bins0))
        for b, items in enumerate(bins0):
            if not items:
                empty_bins.append(b)
                continue
            s = 0
            for i in items:
                bins[b].append(i)
                pos[i] = len(bins[b]) - 1
                assign[i] = b
                s += w[i]
            loads[b] = s
            activate_bin(b, bins, active_bins, active_pos)
        return bins, loads, assign, pos, active_bins, active_pos, empty_bins

    order_base = sorted(range(n), key=lambda i: w[i], reverse=True)

    # Spend more time on init since we can go up to 100s and initial m matters.
    init_deadline = min(time_limit * 0.06, 1.2)

    best_init_bins: Optional[List[List[int]]] = None
    best_init_m = 10**9

    for mode in ("BFD", "FFD"):
        b0, _l0 = pack_greedy(order_base, mode)
        m0 = sum(1 for x in b0 if x)
        if m0 < best_init_m:
            best_init_m = m0
            best_init_bins = b0

    tries = 0
    block_sizes = [6, 8, 10, 14]
    while elapsed() < init_deadline and tries < 80:
        tries += 1
        order = order_base[:]
        block = random.choice(block_sizes)
        for s in range(0, n, block):
            t = min(n, s + block)
            if t - s > 1 and random.random() < 0.85:
                sub = order[s:t]
                random.shuffle(sub)
                order[s:t] = sub
        if random.random() < 0.10:
            # occasional stronger perturbation
            random.shuffle(order)
            order.sort(key=lambda i: (w[i], random.random()), reverse=True)

        mode = "BFD" if random.random() < 0.8 else "FFD"
        b0, _l0 = pack_greedy(order, mode)
        m0 = sum(1 for x in b0 if x)
        if m0 < best_init_m:
            best_init_m = m0
            best_init_bins = b0

    if best_init_bins is None:
        best_init_bins, _ = pack_greedy(order_base, "BFD")

    bins, loads, assign, pos, active_bins, active_pos, empty_bins = build_struct_from_bins(best_init_bins)

    # ----------------------------
    # Incremental aggregates
    # ----------------------------
    def slack2_of_load(lb: int) -> int:
        s = C - lb
        return s * s

    total_slack2 = 0
    for b in active_bins:
        total_slack2 += slack2_of_load(loads[b])

    def m_active() -> int:
        return len(active_bins)

    # ----------------------------
    # Best solution snapshot
    # ----------------------------
    best_m = m_active()
    best_total_slack2 = total_slack2
    best_bins_snapshot = [lst[:] for lst in bins]
    best_loads_snapshot = loads[:]

    # ----------------------------
    # Tabu + long-term memory (reactive)
    # ----------------------------
    tabu_until: Dict[Tuple[int, int], int] = {}
    freq: Dict[Tuple[int, int], int] = {}

    def freq_penalty(i: int, b: int) -> int:
        return freq.get((i, b), 0)

    def is_tabu(i: int, b: int, it: int) -> bool:
        return tabu_until.get((i, b), -1) > it

    base_tenure = 10
    tenure_span = 20
    min_base_tenure = 6
    max_base_tenure = 70

    def set_tabu(i: int, b: int, it: int) -> None:
        tenure = base_tenure + random.randint(0, tenure_span)
        tabu_until[(i, b)] = it + tenure

    def bump_freq(i: int, b: int) -> None:
        k = (i, b)
        freq[k] = freq.get(k, 0) + 1

    def purge_tabu(it: int) -> None:
        if not tabu_until:
            return
        dead = [k for k, v in tabu_until.items() if v <= it]
        for k in dead:
            del tabu_until[k]

    # ----------------------------
    # Target selection
    # ----------------------------
    target_bin: Optional[int] = None
    phase_start_it = 0

    def pick_target_bin(policy: int = 0) -> Optional[int]:
        if not active_bins:
            return None
        if policy == 0:
            # lightest bin (try to empty)
            bestb = None
            bestload = 10**18
            for b in active_bins:
                lb = loads[b]
                if lb < bestload:
                    bestload = lb
                    bestb = b
            return bestb
        if policy == 1:
            # largest slack (try to tighten / rearrange)
            bestb = None
            bestsl = -1
            for b in active_bins:
                sl = C - loads[b]
                if sl > bestsl:
                    bestsl = sl
                    bestb = b
            return bestb
        # random active bin
        return active_bins[random.randrange(len(active_bins))]

    target_bin = pick_target_bin(0)

    # ----------------------------
    # Sampling helpers
    # ----------------------------
    def sample_bins(k: int) -> List[int]:
        m = len(active_bins)
        if k >= m:
            return active_bins[:]
        return [active_bins[random.randrange(m)] for _ in range(k)]

    def best_fit_dest_for_item(i: int, exclude_bin: int, sample_k: int, heavy_k: int = 6, top: int = 6) -> List[int]:
        wi = w[i]
        cand = []
        for b in sample_bins(sample_k):
            if b == exclude_bin:
                continue
            rem = C - loads[b]
            if rem >= wi:
                cand.append((rem - wi, b))
        if heavy_k > 0 and active_bins:
            # scan-heavy: bins are not huge in count
            heavy = sorted(active_bins, key=lambda bb: loads[bb], reverse=True)
            for b in heavy[:heavy_k]:
                if b == exclude_bin:
                    continue
                rem = C - loads[b]
                if rem >= wi:
                    cand.append((rem - wi, b))
        if not cand:
            return []
        cand.sort()
        out = []
        seen = set()
        for _, b in cand:
            if b not in seen:
                seen.add(b)
                out.append(b)
                if len(out) >= top:
                    break
        return out

    def top_items_in_bin(b: int, k: int) -> List[int]:
        items = bins[b]
        if len(items) <= k:
            return sorted(items, key=lambda i: w[i], reverse=True)
        cand = items[:]
        cand.sort(key=lambda i: w[i], reverse=True)
        return cand[:k]

    # ----------------------------
    # Infeasible search support
    # ----------------------------
    # We allow slight overload temporarily (strategic oscillation).
    # violation = sum(max(0, load[b]-C)) across active bins.
    violation = 0
    for b in active_bins:
        if loads[b] > C:
            violation += loads[b] - C

    def viol_of_load(lb: int) -> int:
        return lb - C if lb > C else 0

    # weights for evaluation (adaptive)
    viol_lambda = 400  # strong penalty: keeps violations small
    freq_lambda = 2

    def score_tuple(m: int, viol: int, R: int, slack2: int) -> Tuple[int, int, int, int]:
        return (m, viol, R, slack2)

    # ----------------------------
    # Move evaluation
    # ----------------------------
    def eval_relocate(i: int, src: int, dst: int, cur_m: int, cur_R: int) -> Optional[Tuple[Tuple[int, int, int, int], int, int, int]]:
        # returns (score, new_slack2, new_violation, new_m)
        if src == dst:
            return None
        wi = w[i]

        # incremental slack2
        s2 = total_slack2
        if active_pos[src] != -1:
            s2 -= slack2_of_load(loads[src])
        if active_pos[dst] != -1:
            s2 -= slack2_of_load(loads[dst])

        new_m = cur_m
        new_load_src = loads[src] - wi
        new_load_dst = loads[dst] + wi

        # bin empty effect
        if len(bins[src]) == 1:
            new_m = cur_m - 1
        else:
            s2 += slack2_of_load(new_load_src)

        s2 += slack2_of_load(new_load_dst)

        # violation update
        v = violation
        v -= viol_of_load(loads[src])
        v -= viol_of_load(loads[dst])
        if len(bins[src]) != 1:
            v += viol_of_load(new_load_src)
        v += viol_of_load(new_load_dst)

        # target residual
        if target_bin is None or active_pos[target_bin] == -1:
            new_R = 0
        else:
            if src == target_bin:
                new_R = new_load_src if len(bins[src]) != 1 else 0
            elif dst == target_bin:
                new_R = new_load_dst
            else:
                new_R = cur_R

        pen = freq_lambda * freq_penalty(i, dst)
        sc = score_tuple(new_m, v, new_R, s2 + pen + viol_lambda * v)
        return sc, s2, v, new_m

    def eval_swap(a: int, bitem: int, ba: int, bb: int, cur_m: int, cur_R: int) -> Optional[Tuple[Tuple[int, int, int, int], int, int]]:
        if ba == bb:
            return None
        wa, wb = w[a], w[bitem]
        la = loads[ba]
        lb = loads[bb]

        nla = la - wa + wb
        nlb = lb - wb + wa

        s2 = total_slack2
        s2 -= slack2_of_load(la)
        s2 -= slack2_of_load(lb)
        s2 += slack2_of_load(nla)
        s2 += slack2_of_load(nlb)

        v = violation
        v -= viol_of_load(la) + viol_of_load(lb)
        v += viol_of_load(nla) + viol_of_load(nlb)

        if target_bin is None or active_pos[target_bin] == -1:
            new_R = 0
        else:
            new_R = cur_R
            if ba == target_bin:
                new_R = nla
            if bb == target_bin:
                new_R = nlb

        pen = freq_lambda * (freq_penalty(a, bb) + freq_penalty(bitem, ba))
        sc = score_tuple(cur_m, v, new_R, s2 + pen + viol_lambda * v)
        return sc, s2, v

    # ----------------------------
    # Move application
    # ----------------------------
    def apply_relocate(i: int, src: int, dst: int) -> None:
        nonlocal total_slack2, violation

        # slack2 update
        if active_pos[src] != -1:
            total_slack2 -= slack2_of_load(loads[src])
        if active_pos[dst] != -1:
            total_slack2 -= slack2_of_load(loads[dst])

        # violation update remove old
        violation -= viol_of_load(loads[src])
        violation -= viol_of_load(loads[dst])

        remove_item_from_bin(i, src, bins, loads, assign, pos, active_bins, active_pos, empty_bins)
        add_item_to_bin(i, dst, bins, loads, assign, pos, active_bins, active_pos)

        # add new
        if active_pos[src] != -1:
            total_slack2 += slack2_of_load(loads[src])
            violation += viol_of_load(loads[src])
        total_slack2 += slack2_of_load(loads[dst])
        violation += viol_of_load(loads[dst])

    def apply_swap(a: int, bitem: int, ba: int, bb: int) -> None:
        nonlocal total_slack2, violation

        la = loads[ba]
        lb = loads[bb]

        total_slack2 -= slack2_of_load(la) + slack2_of_load(lb)
        violation -= viol_of_load(la) + viol_of_load(lb)

        pa = pos[a]
        pb = pos[bitem]

        bins[ba][pa] = bitem
        pos[bitem] = pa
        assign[bitem] = ba

        bins[bb][pb] = a
        pos[a] = pb
        assign[a] = bb

        loads[ba] = la - w[a] + w[bitem]
        loads[bb] = lb - w[bitem] + w[a]

        total_slack2 += slack2_of_load(loads[ba]) + slack2_of_load(loads[bb])
        violation += viol_of_load(loads[ba]) + viol_of_load(loads[bb])

    # ----------------------------
    # Compound moves (bin-empty focused)
    # ----------------------------
    def try_pair_relocate_from_target(it: int, cur_m: int, cur_R: int) -> bool:
        if target_bin is None or active_pos[target_bin] == -1:
            return False
        t = target_bin
        if len(bins[t]) < 2:
            return False

        items = top_items_in_bin(t, 7)
        if len(items) < 2:
            return False

        best_sc = None
        best_plan = None  # (i1,d1,i2,d2)

        combs = []
        for i in range(min(5, len(items))):
            for j in range(i + 1, min(len(items), 7)):
                combs.append((items[i], items[j]))
        random.shuffle(combs)
        combs = combs[:24]

        for i1, i2 in combs:
            wi1, wi2 = w[i1], w[i2]
            d1s = best_fit_dest_for_item(i1, t, sample_k=40, heavy_k=8, top=7)
            if not d1s:
                continue
            d2s = best_fit_dest_for_item(i2, t, sample_k=40, heavy_k=8, top=7)
            if not d2s:
                continue

            for d1 in d1s[:4]:
                # compute provisional load if same destination
                l1 = loads[d1] + wi1
                for d2 in ([d1] + d2s[:5]):
                    if d2 == t:
                        continue
                    if d2 == d1:
                        if l1 + wi2 > C:
                            continue
                    else:
                        if loads[d2] + wi2 > C:
                            continue

                    # fast approximate eval with feasibility enforced here (no violations)
                    # We only consider feasible pair-relocate (strong intensification).
                    new_m = cur_m - (1 if len(bins[t]) == 2 else 0)

                    s2 = total_slack2
                    s2 -= slack2_of_load(loads[t])
                    s2 -= slack2_of_load(loads[d1])
                    if d2 != d1:
                        s2 -= slack2_of_load(loads[d2])

                    new_load_t = loads[t] - wi1 - wi2
                    if len(bins[t]) != 2:
                        s2 += slack2_of_load(new_load_t)

                    if d2 == d1:
                        s2 += slack2_of_load(loads[d1] + wi1 + wi2)
                    else:
                        s2 += slack2_of_load(loads[d1] + wi1)
                        s2 += slack2_of_load(loads[d2] + wi2)

                    new_R = 0 if len(bins[t]) == 2 else new_load_t
                    pen = freq_lambda * (freq_penalty(i1, d1) + freq_penalty(i2, d2))
                    sc = score_tuple(new_m, 0, new_R, s2 + pen)

                    if (is_tabu(i1, d1, it) or is_tabu(i2, d2, it)) and not (new_m < best_m):
                        continue
                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_plan = (i1, d1, i2, d2)

        if best_plan is None:
            return False

        i1, d1, i2, d2 = best_plan
        if w[i2] > w[i1]:
            i1, i2 = i2, i1
            d1, d2 = d2, d1

        if assign[i1] != t or assign[i2] != t:
            return False

        apply_relocate(i1, t, d1)
        set_tabu(i1, t, it)
        bump_freq(i1, d1)

        if target_bin is not None and active_pos[target_bin] != -1 and assign[i2] == target_bin:
            if loads[d2] + w[i2] <= C:
                apply_relocate(i2, target_bin, d2)
                set_tabu(i2, target_bin, it)
                bump_freq(i2, d2)
        return True

    def try_ejection_chain_depth2(it: int) -> bool:
        # i from target to b by ejecting e from b to dst
        if target_bin is None or active_pos[target_bin] == -1:
            return False
        t = target_bin
        if not bins[t]:
            return False

        cur_m = m_active()
        cur_R = loads[t]

        cand_items = top_items_in_bin(t, 5)
        sampled_b = sample_bins(45)
        random.shuffle(sampled_b)

        best_sc = None
        best_plan = None  # (i, b, e, dst)

        for i in cand_items:
            wi = w[i]
            for b in sampled_b[:32]:
                if b == t:
                    continue
                rem = C - loads[b]
                if rem >= wi:
                    continue
                over = wi - rem
                if over <= 0 or over > C // 2:
                    continue

                # pick ejection candidate (prefer small >= over, else largest)
                e = None
                best_we = 10**18
                for cand in bins[b]:
                    wc = w[cand]
                    if wc >= over and wc < best_we:
                        best_we = wc
                        e = cand
                if e is None:
                    # try largest to create room (may still be infeasible for b, but we keep b feasible here)
                    e = max(bins[b], key=lambda x: w[x])
                    if w[e] < over:
                        continue

                # where to put e
                dests_e = best_fit_dest_for_item(e, exclude_bin=b, sample_k=55, heavy_k=10, top=7)
                if not dests_e:
                    continue

                for dst in dests_e[:4]:
                    if dst == t:
                        continue
                    if loads[dst] + w[e] > C:
                        continue

                    # evaluate (feasible chain)
                    s2 = total_slack2
                    s2 -= slack2_of_load(loads[t])
                    s2 -= slack2_of_load(loads[b])
                    s2 -= slack2_of_load(loads[dst])

                    new_m = cur_m
                    new_load_t = loads[t] - wi
                    if len(bins[t]) == 1:
                        new_m -= 1
                    else:
                        s2 += slack2_of_load(new_load_t)

                    new_load_b = loads[b] - w[e] + wi
                    if new_load_b > C:
                        continue
                    s2 += slack2_of_load(new_load_b)

                    new_load_dst = loads[dst] + w[e]
                    s2 += slack2_of_load(new_load_dst)

                    new_R = 0 if len(bins[t]) == 1 else new_load_t
                    pen = freq_lambda * (freq_penalty(i, b) + freq_penalty(e, dst))
                    sc = score_tuple(new_m, 0, new_R, s2 + pen)

                    if (is_tabu(i, b, it) or is_tabu(e, dst, it)) and not (new_m < best_m):
                        continue
                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_plan = (i, b, e, dst)

        if best_plan is None:
            return False

        i, b, e, dst = best_plan
        apply_relocate(e, b, dst)
        set_tabu(e, b, it)
        bump_freq(e, dst)

        if target_bin is not None and active_pos[target_bin] != -1 and assign[i] == target_bin:
            if loads[b] + w[i] <= C:
                apply_relocate(i, target_bin, b)
                set_tabu(i, target_bin, it)
                bump_freq(i, b)
        return True

    def try_empty_bin_intensification(it: int) -> bool:
        # Occasional strong intensification: remove a light bin and reinsert its items (feasible).
        if len(active_bins) <= 1:
            return False
        b = pick_target_bin(0)
        if b is None or active_pos[b] == -1:
            return False
        if loads[b] == 0:
            return False
        if loads[b] > int(0.65 * C):
            return False

        items = bins[b][:]

        # snapshot full (rare call)
        snap_bins = [lst[:] for lst in bins]
        snap_loads = loads[:]
        snap_assign = assign[:]
        snap_pos = pos[:]
        snap_active_bins = active_bins[:]
        snap_active_pos = active_pos[:]
        snap_empty_bins = empty_bins[:]
        snap_s2 = total_slack2
        snap_v = violation

        # remove all items from b
        for i in items:
            if assign[i] == b:
                remove_item_from_bin(i, b, bins, loads, assign, pos, active_bins, active_pos, empty_bins)

        # recompute aggregates
        nonlocal_total_s2 = 0
        nonlocal_v = 0
        for bb in active_bins:
            nonlocal_total_s2 += slack2_of_load(loads[bb])
            nonlocal_v += viol_of_load(loads[bb])
        # assign back
        nonlocal total_slack2, violation
        total_slack2 = nonlocal_total_s2
        violation = nonlocal_v

        items.sort(key=lambda i: w[i], reverse=True)
        for i in items:
            if elapsed() >= time_limit:
                # revert
                bins[:] = snap_bins
                loads[:] = snap_loads
                assign[:] = snap_assign
                pos[:] = snap_pos
                active_bins[:] = snap_active_bins
                active_pos[:] = snap_active_pos
                empty_bins[:] = snap_empty_bins
                total_slack2 = snap_s2
                violation = snap_v
                return False

            dests = best_fit_dest_for_item(i, exclude_bin=-1, sample_k=70, heavy_k=12, top=10)
            placed = False
            for dst in dests:
                if loads[dst] + w[i] <= C:
                    add_item_to_bin(i, dst, bins, loads, assign, pos, active_bins, active_pos)
                    placed = True
                    break
            if not placed:
                # revert
                bins[:] = snap_bins
                loads[:] = snap_loads
                assign[:] = snap_assign
                pos[:] = snap_pos
                active_bins[:] = snap_active_bins
                active_pos[:] = snap_active_pos
                empty_bins[:] = snap_empty_bins
                total_slack2 = snap_s2
                violation = snap_v
                return False

        # success: recompute aggregates precisely
        total_slack2 = 0
        violation = 0
        for bb in active_bins:
            total_slack2 += slack2_of_load(loads[bb])
            violation += viol_of_load(loads[bb])

        for i in items:
            set_tabu(i, b, it)
        return True

    # ----------------------------
    # Main tabu loop
    # ----------------------------
    max_iters = 12_000_000
    check_mask = 0x3FF  # time check every 1024 iters

    phase_iters = min(12000, 800 + 30 * n)

    base_reloc_tries = 1200
    base_swap_tries = 600

    last_best_it = 0
    last_improve_time = 0.0

    it = 0
    while it < max_iters:
        it += 1

        if (it & check_mask) == 0:
            if elapsed() >= time_limit:
                break

        if (it & 0x1FFF) == 0:
            purge_tabu(it)

        # phase / target policy switching
        if target_bin is None or active_pos[target_bin] == -1 or (it - phase_start_it) >= phase_iters:
            r = random.random()
            if r < 0.70:
                target_bin = pick_target_bin(0)
            elif r < 0.92:
                target_bin = pick_target_bin(1)
            else:
                target_bin = pick_target_bin(2)
            phase_start_it = it

        cur_m = m_active()
        cur_R = loads[target_bin] if (target_bin is not None and active_pos[target_bin] != -1) else 0

        # Reactive parameters
        if it - last_best_it > max(900, 25 * n):
            base_tenure = min(max_base_tenure, base_tenure + 2)
            viol_lambda = min(1200, int(viol_lambda * 1.05) + 5)
        elif it - last_best_it < max(140, 4 * n):
            base_tenure = max(min_base_tenure, base_tenure - 1)
            viol_lambda = max(220, int(viol_lambda * 0.98))

        # Periodic strong intensification
        if (it & 0x7FFF) == 0 and elapsed() < time_limit:
            if try_empty_bin_intensification(it):
                target_bin = pick_target_bin(0)
                phase_start_it = it

        # If currently infeasible, prioritize repairs (still via tabu moves)
        infeasible_mode = violation > 0

        progress = elapsed() / max(1e-9, time_limit)
        reloc_tries = base_reloc_tries + (800 if progress < 0.45 else 0)
        swap_tries = base_swap_tries + (400 if progress < 0.45 else 0)

        best_move = None
        best_sc = None

        # --- Candidate generation ---
        if target_bin is not None and active_pos[target_bin] != -1 and bins[target_bin]:
            t = target_bin
            cand_items = top_items_in_bin(t, 7)

            # 1) relocates from target
            for _ in range(reloc_tries):
                i = cand_items[random.randrange(len(cand_items))]
                if assign[i] != t:
                    continue
                dests = best_fit_dest_for_item(i, t, sample_k=55 if not infeasible_mode else 35,
                                               heavy_k=10 if not infeasible_mode else 6,
                                               top=7)
                if not dests:
                    continue
                for dst in dests[:4]:
                    ev = eval_relocate(i, t, dst, cur_m, cur_R)
                    if ev is None:
                        continue
                    sc, _s2, v, new_m = ev

                    # tabu / aspiration
                    tabu = is_tabu(i, t, it) or is_tabu(i, dst, it)
                    if tabu and not (new_m < best_m):
                        # aspiration if improves feasibility a lot or reduces R
                        if not (v < violation or sc[2] + min(4, w[i]) < cur_R):
                            continue

                    # in infeasible mode, prefer moves that reduce violation
                    if infeasible_mode and v > violation:
                        continue

                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_move = ("rel", i, t, dst)

            # 2) swaps involving target
            if len(active_bins) > 1:
                cand_t = top_items_in_bin(t, 6)
                for _ in range(swap_tries):
                    a = cand_t[random.randrange(len(cand_t))]
                    if assign[a] != t:
                        continue
                    bb = active_bins[random.randrange(len(active_bins))]
                    if bb == t or not bins[bb]:
                        continue
                    items_bb = bins[bb]
                    # bias to pick an item that makes bb tighter after swap
                    bitem = items_bb[random.randrange(len(items_bb))]
                    if len(items_bb) > 2 and random.random() < 0.8:
                        cand = [items_bb[random.randrange(len(items_bb))] for _ in range(4)]
                        bitem = min(cand, key=lambda x: w[x])

                    if a == bitem:
                        continue

                    ev = eval_swap(a, bitem, t, bb, cur_m, cur_R)
                    if ev is None:
                        continue
                    sc, _s2, v = ev

                    tabu = is_tabu(a, t, it) or is_tabu(bitem, bb, it)
                    if tabu and not (sc[0] < best_m):
                        if not (v < violation):
                            continue

                    if infeasible_mode and v > violation:
                        continue

                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_move = ("sw", a, bitem, t, bb)

        # 3) If stuck, compound moves
        if best_move is None and target_bin is not None and active_pos[target_bin] != -1 and bins[target_bin]:
            if not infeasible_mode:
                if try_pair_relocate_from_target(it, cur_m, cur_R):
                    pass
                elif try_ejection_chain_depth2(it):
                    pass
                else:
                    # no compound applied
                    pass
            else:
                # in infeasible mode, try ejection chain (often repairs)
                try_ejection_chain_depth2(it)

        # 4) If still no move, global diversification relocates (possibly infeasible but bounded)
        if best_move is None and len(active_bins) > 1:
            for _ in range(260):
                src = active_bins[random.randrange(len(active_bins))]
                if not bins[src]:
                    continue
                i = bins[src][random.randrange(len(bins[src]))]

                # allow infeasible dst occasionally to escape
                if random.random() < 0.12:
                    dst = active_bins[random.randrange(len(active_bins))]
                    if dst == src:
                        continue
                    # allow small overload
                    if loads[dst] + w[i] > C + max(2, C // 20):
                        continue
                    # fabricate an eval by temporarily allowing overload (eval_relocate already supports)
                    ev = eval_relocate(i, src, dst, cur_m, cur_R)
                    if ev is None:
                        continue
                    sc, _s2, v, new_m = ev
                    if v > violation + max(2, C // 20):
                        continue
                    if is_tabu(i, src, it) and not (new_m < best_m):
                        continue
                    best_move = ("rel", i, src, dst)
                    best_sc = sc
                    break

                dests = best_fit_dest_for_item(i, src, sample_k=45, heavy_k=8, top=6)
                if not dests:
                    continue
                dst = dests[0]
                ev = eval_relocate(i, src, dst, cur_m, cur_R)
                if ev is None:
                    continue
                sc, _s2, v, new_m = ev
                if infeasible_mode and v > violation:
                    continue
                if is_tabu(i, src, it) and not (new_m < best_m):
                    continue
                best_move = ("rel", i, src, dst)
                best_sc = sc
                break

        # Apply move if selected
        if best_move is not None:
            if best_move[0] == "rel":
                _, i, src, dst = best_move
                if assign[i] == src:
                    apply_relocate(i, src, dst)
                    set_tabu(i, src, it)
                    bump_freq(i, dst)
            else:
                _, a, bitem, ba, bb = best_move
                if assign[a] == ba and assign[bitem] == bb:
                    apply_swap(a, bitem, ba, bb)
                    set_tabu(a, ba, it)
                    set_tabu(bitem, bb, it)
                    bump_freq(a, bb)
                    bump_freq(bitem, ba)

        # If infeasible, do quick repair pressure by raising viol_lambda a bit
        if violation > 0 and (it & 0x3FF) == 0:
            viol_lambda = min(1500, viol_lambda + 15)

        # Update best only if feasible
        if violation == 0:
            cur_m2 = m_active()
            if cur_m2 < best_m:
                best_m = cur_m2
                best_total_slack2 = total_slack2
                best_bins_snapshot = [lst[:] for lst in bins]
                best_loads_snapshot = loads[:]
                last_best_it = it
                last_improve_time = elapsed()
                target_bin = pick_target_bin(0)
                phase_start_it = it
            elif cur_m2 == best_m and total_slack2 < best_total_slack2:
                best_total_slack2 = total_slack2
                best_bins_snapshot = [lst[:] for lst in bins]
                best_loads_snapshot = loads[:]
                last_best_it = it
                last_improve_time = elapsed()

        # If no improvement for long time, shake target and slightly increase diversification
        if (it & 0x7FFF) == 0:
            if elapsed() - last_improve_time > 0.18 * time_limit:
                target_bin = pick_target_bin(2)
                phase_start_it = it
                base_tenure = min(max_base_tenure, base_tenure + 3)
                freq_lambda = min(6, freq_lambda + 1)

    # ----------------------------
    # Build final packing from best snapshot (compact)
    # ----------------------------
    final_bins: List[List[int]] = []
    final_loads: List[int] = []
    for items in best_bins_snapshot:
        if items:
            final_bins.append(items[:])
            final_loads.append(int(sum(w[i] for i in items)))

    return {"packing": final_bins, "bin_weights": final_loads}
