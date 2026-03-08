import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    C = bin_capacity

    if n == 0:
        return {"packing": [], "bin_weights": []}

    def time_exceeded() -> bool:
        return (time.time() - start) >= time_limit

    # -----------------------
    # Objective
    # -----------------------

    def slack_components(loads: List[int]) -> Tuple[int, int]:
        # (sum_sq_slack, max_slack)
        s2 = 0
        mx = 0
        for ld in loads:
            d = C - ld
            if d > mx:
                mx = d
            s2 += d * d
        return s2, mx

    def obj_from(loads: List[int]) -> Tuple[int, int]:
        # Secondary term: squared slack plus small max-slack weight.
        # Still integer and comparable.
        s2, mx = slack_components(loads)
        return (len(loads), s2 + 7 * mx * mx)

    def better_obj(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

    # -----------------------
    # State utilities
    # -----------------------
    # bins: List[List[int]]
    # loads: List[int]
    # assign: List[int]  item -> bin index
    # pos: List[int]     item -> position within bins[bin]
    # bin_ids: List[int] stable ids for tabu

    def make_state() -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], int]:
        return [], [], [-1] * n, [-1] * n, [], 0  # next_bin_id

    def add_bin(bins, loads, bin_ids, next_bin_id) -> Tuple[int, int]:
        bins.append([])
        loads.append(0)
        bin_ids.append(next_bin_id)
        return len(bins) - 1, next_bin_id + 1

    def remove_item(i: int, b: int, bins, loads, assign, pos) -> None:
        p = pos[i]
        last = bins[b][-1]
        if last != i:
            bins[b][p] = last
            pos[last] = p
        bins[b].pop()
        pos[i] = -1
        assign[i] = -1
        loads[b] -= weights[i]

    def add_item(i: int, b: int, bins, loads, assign, pos) -> None:
        pos[i] = len(bins[b])
        bins[b].append(i)
        assign[i] = b
        loads[b] += weights[i]

    def remove_empty_bin(b: int, bins, loads, assign, pos, bin_ids) -> None:
        if bins[b]:
            return
        last = len(bins) - 1
        if b != last:
            bins[b], bins[last] = bins[last], bins[b]
            loads[b], loads[last] = loads[last], loads[b]
            bin_ids[b], bin_ids[last] = bin_ids[last], bin_ids[b]
            for idx, itm in enumerate(bins[b]):
                assign[itm] = b
                pos[itm] = idx
        bins.pop()
        loads.pop()
        bin_ids.pop()

    def apply_reloc(i: int, b_from: int, b_to: int, bins, loads, assign, pos, bin_ids) -> None:
        remove_item(i, b_from, bins, loads, assign, pos)
        add_item(i, b_to, bins, loads, assign, pos)
        remove_empty_bin(b_from, bins, loads, assign, pos, bin_ids)

    def apply_swap(i: int, j: int, bi: int, bj: int, bins, loads, assign, pos) -> None:
        pi = pos[i]
        pj = pos[j]
        bins[bi][pi] = j
        bins[bj][pj] = i
        pos[i] = pj
        pos[j] = pi
        assign[i] = bj
        assign[j] = bi
        loads[bi] += weights[j] - weights[i]
        loads[bj] += weights[i] - weights[j]

    # -----------------------
    # Construction
    # -----------------------

    def construct(order: List[int], variant: str, rcl_k: int = 3) -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], int]:
        bins, loads, assign, pos, bin_ids, next_id = make_state()

        # variant:
        #  - "BFD": best-fit decreasing
        #  - "FFD": first-fit decreasing
        #  - "GRASP": randomized restricted candidate list on best-fit

        for i in order:
            w = weights[i]
            chosen = -1
            if bins:
                if variant == "FFD":
                    for b in range(len(bins)):
                        if loads[b] + w <= C:
                            chosen = b
                            break
                else:
                    # compute best-fit list (small)
                    best: List[Tuple[int, int]] = []  # (rem_after, b)
                    for b in range(len(bins)):
                        rem = C - loads[b]
                        if rem >= w:
                            ra = rem - w
                            if len(best) < max(1, rcl_k):
                                best.append((ra, b))
                                if len(best) == max(1, rcl_k):
                                    best.sort()
                            else:
                                if ra < best[-1][0]:
                                    best[-1] = (ra, b)
                                    best.sort()
                    if best:
                        if variant == "GRASP" and len(best) > 1:
                            chosen = random.choice(best)[1]
                        else:
                            chosen = best[0][1]

            if chosen == -1:
                chosen, next_id = add_bin(bins, loads, bin_ids, next_id)
            add_item(i, chosen, bins, loads, assign, pos)

        return bins, loads, assign, pos, bin_ids, next_id

    def multi_start_init() -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], int]:
        idx = list(range(n))
        # Main sort: decreasing weights
        base = sorted(idx, key=lambda i: (-weights[i], i))

        starts: List[Tuple[List[int], str]] = [(base, "BFD"), (base, "FFD")]

        # Shuffle ties for diversification
        if n <= 6000:
            # Build weight groups once
            groups: Dict[int, List[int]] = {}
            for i in idx:
                groups.setdefault(weights[i], []).append(i)
            wkeys = sorted(groups.keys(), reverse=True)

            for _ in range(10):
                if time_exceeded():
                    break
                order: List[int] = []
                for w in wkeys:
                    g = groups[w][:]
                    random.shuffle(g)
                    order.extend(g)
                starts.append((order, "BFD"))

            for _ in range(10):
                if time_exceeded():
                    break
                # noisy key keeps global descending but perturbs within band
                order = sorted(idx, key=lambda i: (-weights[i], random.random()))
                starts.append((order, "GRASP"))

        best_state = None
        best_obj = None
        for order, var in starts:
            if time_exceeded():
                break
            st = construct(order, var, rcl_k=4 if var == "GRASP" else 1)
            obj = obj_from(st[1])
            if best_obj is None or better_obj(obj, best_obj):
                best_obj = obj
                best_state = st

        assert best_state is not None
        return best_state

    # -----------------------
    # Tabu / aspiration
    # -----------------------

    def aspiration_ok(cand_obj: Tuple[int, int], best_obj: Tuple[int, int], best2_for_m: Dict[int, int]) -> bool:
        if better_obj(cand_obj, best_obj):
            return True
        m = cand_obj[0]
        v = best2_for_m.get(m)
        return v is None or cand_obj[1] < v

    # -----------------------
    # Candidate bin lists
    # -----------------------

    def pick_light_bins(loads: List[int], k: int) -> List[int]:
        m = len(loads)
        if m <= k:
            return list(range(m))
        return sorted(range(m), key=lambda b: loads[b])[:k]

    def pick_tight_bins(loads: List[int], k: int) -> List[int]:
        m = len(loads)
        if m <= k:
            return list(range(m))
        return sorted(range(m), key=lambda b: loads[b], reverse=True)[:k]

    def pick_dest_bins(loads: List[int], k: int, extra_rand: int) -> List[int]:
        m = len(loads)
        if m == 0:
            return []
        # prefer small remaining (best-fit)
        base = sorted(range(m), key=lambda b: (C - loads[b], loads[b]))[: min(k, m)]
        if m > len(base):
            for _ in range(extra_rand):
                base.append(random.randrange(m))
        # unique
        seen = set()
        out = []
        for b in base:
            if b not in seen:
                seen.add(b)
                out.append(b)
        return out

    def best_fit_r(dest_bins: List[int], loads: List[int], w: int, r: int, forbid: int) -> List[int]:
        best: List[Tuple[int, int]] = []
        for b in dest_bins:
            if b == forbid:
                continue
            rem = C - loads[b]
            if rem >= w:
                ra = rem - w
                if len(best) < r:
                    best.append((ra, b))
                    if len(best) == r:
                        best.sort()
                else:
                    if ra < best[-1][0]:
                        best[-1] = (ra, b)
                        best.sort()
        return [b for _, b in best]

    # -----------------------
    # Extra neighborhood: 2-1 exchange (limited)
    # -----------------------

    def try_two_to_one(b0: int, dest_bins: List[int], bins, loads, assign, pos, bin_ids,
                       tabu_reloc: Dict[Tuple[int, int], int], it: int,
                       best_obj: Tuple[int, int], best2_for_m: Dict[int, int]) -> Optional[Tuple[List[Tuple[int, int, int]], Tuple[int, int]]]:
        # Goal: empty b0 by moving two items out, possibly ejecting one from a target bin.
        # Returns (moves, cand_obj) where moves is list of relocations.
        if not bins[b0]:
            return None
        items0 = sorted(bins[b0], key=lambda i: weights[i], reverse=True)
        if len(items0) < 2:
            return None

        m = len(bins)
        # sample a few pairs (largest with others)
        a = items0[0]
        for b in items0[1: min(10, len(items0))]:
            wa, wb = weights[a], weights[b]
            # try to place a and b into possibly different bins directly first
            candA = best_fit_r(dest_bins, loads, wa, 3, forbid=b0)
            candB = best_fit_r(dest_bins, loads, wb, 3, forbid=b0)
            if candA and candB:
                # pick best combination by slack
                best_local = None
                best_moves = None
                for ba in candA:
                    for bb in candB:
                        # simulate if both go to same bin ensure capacity
                        if ba == bb and loads[ba] + wa + wb > C:
                            continue
                        # tabu checks
                        ida = bin_ids[ba]
                        idb = bin_ids[bb]
                        if (tabu_reloc.get((a, ida), 0) > it) and (not aspiration_ok((m - 1, 0), best_obj, best2_for_m)):
                            continue
                        if (tabu_reloc.get((b, idb), 0) > it) and (not aspiration_ok((m - 1, 0), best_obj, best2_for_m)):
                            continue

                        # compute objective delta using affected bins
                        affected = {b0, ba, bb}
                        old_s2, old_mx = 0, 0
                        new_s2, new_mx = 0, 0
                        for bx in affected:
                            d = C - loads[bx]
                            old_s2 += d * d
                            if d > old_mx:
                                old_mx = d
                        # new loads
                        l0 = loads[b0] - wa - wb
                        la = loads[ba] + wa
                        lb = loads[bb] + wb
                        if ba == bb:
                            la = loads[ba] + wa + wb
                            lb = loads[bb]  # same bin, ignore
                        # compute new slack for affected
                        for bx in affected:
                            if bx == b0:
                                ld = l0
                            elif bx == ba:
                                ld = la
                            elif bx == bb:
                                ld = lb
                            else:
                                ld = loads[bx]
                            if ld < 0 or ld > C:
                                break
                            d = C - ld
                            new_s2 += d * d
                            if d > new_mx:
                                new_mx = d
                        else:
                            # global obj approximation: update only slack term; bins term exact
                            curr_obj = obj_from(loads)
                            s2_curr, mx_curr = slack_components(loads)
                            # Note: mx update is approximate since we only tracked affected; recompute if promising.
                            s2_new = s2_curr - old_s2 + new_s2
                            # keep mx conservative by recomputing (cheap enough for rare call)
                            # apply loads temp list for mx recompute
                            # (avoid allocating: quick recompute)
                            mx = 0
                            for bx, ld in enumerate(loads):
                                if bx == b0:
                                    ld2 = l0
                                elif bx == ba:
                                    ld2 = la
                                elif bx == bb:
                                    ld2 = lb
                                else:
                                    ld2 = ld
                                d2 = C - ld2
                                if d2 > mx:
                                    mx = d2
                            cand_obj = (m - 1, s2_new + 7 * mx * mx)

                            if best_local is None or better_obj(cand_obj, best_local):
                                best_local = cand_obj
                                moves = [(a, b0, ba), (b, b0, bb)]
                                best_moves = moves
                if best_moves is not None:
                    # if ba==bb and second move would use removed b0 still fine, both from b0
                    return best_moves, best_local
        return None

    # -----------------------
    # Init
    # -----------------------

    bins, loads, assign, pos, bin_ids, next_bin_id = multi_start_init()

    best_bins = [b[:] for b in bins]
    best_loads = loads[:]
    best_assign = assign[:]
    best_bin_ids = bin_ids[:]
    best_obj = obj_from(loads)

    best2_for_m: Dict[int, int] = {best_obj[0]: best_obj[1]}

    # Tabu: item->dest_bin_id, and swap pairs
    tabu_reloc: Dict[Tuple[int, int], int] = {}
    tabu_swap: Dict[Tuple[int, int], int] = {}

    # Reactive tenure
    min_tenure = 7
    max_tenure = 60
    tenure = max(min_tenure, min(max_tenure, int(8 + 0.9 * (n ** 0.5))))

    # repetition memory
    def signature(assign: List[int], bin_ids: List[int]) -> int:
        h = 1469598103934665603
        prime = 1099511628211
        for i, b in enumerate(assign):
            bid = bin_ids[b]
            x = (i + 1) * 0x9E3779B1 ^ (bid + 0x85EBCA6B)
            h ^= x & 0xFFFFFFFFFFFFFFFF
            h = (h * prime) & 0xFFFFFFFFFFFFFFFF
        return h

    sig_window = 80
    recent: List[int] = []

    # Elite pool
    elite: List[Tuple[Tuple[int, int], List[List[int]], List[int], List[int], List[int], List[int], int]] = []

    def hamming(a: List[int], b: List[int]) -> int:
        d = 0
        for i in range(n):
            if a[i] != b[i]:
                d += 1
        return d

    def push_elite(obj, bins, loads, assign, pos, bin_ids, next_id):
        nonlocal elite
        for o2, _, _, a2, _, _, _ in elite:
            if obj[0] == o2[0] and hamming(assign, a2) < max(20, n // 60):
                return
        elite.append((obj, [b[:] for b in bins], loads[:], assign[:], pos[:], bin_ids[:], next_id))
        elite.sort(key=lambda x: x[0])
        if len(elite) > 10:
            elite.pop()

    push_elite(best_obj, bins, loads, assign, pos, bin_ids, next_bin_id)

    # -----------------------
    # Main TS loop
    # -----------------------

    max_iter = max(60000, 900 * n)  # fixed iteration cap

    # parameters
    k_light_base = 18 if n < 400 else 28
    k_tight = 14
    k_dest_base = 90 if n < 800 else 120
    extra_rand_dest = 10
    r_bestfit = 6

    swap_trials_per_item = 10

    chain_period = 90
    stagnation_threshold = max(900, 3 * n)

    iters_since_best = 0

    it = 0
    while it < max_iter:
        it += 1
        if it % 7 == 0 and time_exceeded():
            break

        m = len(bins)
        if m <= 1:
            break

        # reactive tenure based on repeats
        if it % 12 == 0:
            sig = signature(assign, bin_ids)
            recent.append(sig)
            if len(recent) > sig_window:
                recent.pop(0)
            rep = recent.count(sig)
            if rep >= 3:
                tenure = min(max_tenure, tenure + 3)
            elif rep == 2:
                tenure = min(max_tenure, tenure + 1)
            else:
                tenure = max(min_tenure, tenure - 1)

        # purge some tabu
        if it % 250 == 0:
            if len(tabu_reloc) > 50000:
                for k in list(tabu_reloc.keys())[:15000]:
                    if tabu_reloc.get(k, 0) <= it:
                        tabu_reloc.pop(k, None)
            if len(tabu_swap) > 50000:
                for k in list(tabu_swap.keys())[:15000]:
                    if tabu_swap.get(k, 0) <= it:
                        tabu_swap.pop(k, None)

        # candidate bins
        k_light = min(k_light_base + (8 if iters_since_best > stagnation_threshold // 2 else 0), m)
        T = pick_light_bins(loads, k_light)
        Tset = set(T)

        D = pick_dest_bins(loads, min(k_dest_base, m), extra_rand_dest)
        tight_bins = pick_tight_bins(loads, min(k_tight, m))

        # items from target bins
        items_T: List[int] = []
        for b in T:
            items_T.extend(bins[b])
        if not items_T:
            continue

        # bias: focus on heavier items in light bins
        if len(items_T) > 120:
            items_T.sort(key=lambda i: -weights[i])
            items_T = items_T[:90] + random.sample(items_T[90:], min(30, len(items_T) - 90))

        curr_obj = obj_from(loads)

        best_kind = None
        best_data = None
        best_obj_move = None
        best_score = None

        # -----------------
        # Relocate
        # -----------------
        for i_item in items_T:
            b_from = assign[i_item]
            if b_from < 0:
                continue
            w = weights[i_item]
            dests = best_fit_r(D, loads, w, r_bestfit, forbid=b_from)
            if not dests:
                continue

            lf_old = loads[b_from]
            for b_to in dests:
                lt_old = loads[b_to]
                lf_new = lf_old - w
                lt_new = lt_old + w
                if lt_new > C:
                    continue
                new_m = m - 1 if lf_new == 0 else m

                # compute cand obj exactly (cheap enough here)
                # update only two loads then recompute components with a fast scan occasionally.
                # We'll approximate slack by delta on squared slack and recompute max-slack via scan.
                s2_curr, mx_curr = slack_components(loads)
                old_s2 = (C - lf_old) ** 2 + (C - lt_old) ** 2
                new_s2 = (C - lf_new) ** 2 + (C - lt_new) ** 2
                s2_new = s2_curr - old_s2 + new_s2

                # recompute max slack cheaply by scan (m is moderate; still OK with limited moves)
                mx = 0
                for b in range(m):
                    if b == b_from:
                        ld = lf_new
                    elif b == b_to:
                        ld = lt_new
                    else:
                        ld = loads[b]
                    d = C - ld
                    if d > mx:
                        mx = d
                cand_obj = (new_m, s2_new + 7 * mx * mx)

                dest_id = bin_ids[b_to]
                is_tabu = tabu_reloc.get((i_item, dest_id), 0) > it
                if is_tabu and not aspiration_ok(cand_obj, best_obj, best2_for_m):
                    continue

                # tie-break: prefer emptying target/light bins and increasing tightness
                score = 0
                if b_from in Tset:
                    score -= 200000
                score -= (lf_old - lf_new) * 20
                if lf_new == 0:
                    score -= 1500000
                # prefer making destination tighter
                score += (C - lt_new)

                if best_obj_move is None or better_obj(cand_obj, best_obj_move) or (cand_obj == best_obj_move and score < best_score):
                    best_kind = "reloc"
                    best_data = (i_item, b_from, b_to)
                    best_obj_move = cand_obj
                    best_score = score

        # -----------------
        # Swap
        # -----------------
        if not time_exceeded():
            partner_bins = list(dict.fromkeys(tight_bins + D[: min(18, len(D))] + random.sample(range(m), min(10, m))))
            for i_item in items_T:
                bi = assign[i_item]
                wi = weights[i_item]
                li_old = loads[bi]
                rem_i = C - li_old
                target_wj = wi + rem_i  # tends to fill bi

                trials = 0
                for bj in partner_bins:
                    if bj == bi or not bins[bj]:
                        continue
                    # sample a few j close to target_wj
                    cand = bins[bj]
                    sample = random.sample(cand, 12) if len(cand) > 12 else cand
                    sample.sort(key=lambda j: abs(weights[j] - target_wj))
                    for j_item in sample[:4]:
                        if j_item == i_item:
                            continue
                        wj = weights[j_item]
                        li_new = li_old - wi + wj
                        lj_old = loads[bj]
                        lj_new = lj_old - wj + wi
                        if li_new > C or lj_new > C:
                            continue

                        # objective exact
                        s2_curr, _ = slack_components(loads)
                        old_s2 = (C - li_old) ** 2 + (C - lj_old) ** 2
                        new_s2 = (C - li_new) ** 2 + (C - lj_new) ** 2
                        s2_new = s2_curr - old_s2 + new_s2
                        mx = 0
                        for b in range(m):
                            if b == bi:
                                ld = li_new
                            elif b == bj:
                                ld = lj_new
                            else:
                                ld = loads[b]
                            d = C - ld
                            if d > mx:
                                mx = d
                        cand_obj = (m, s2_new + 7 * mx * mx)

                        key = (i_item, j_item) if i_item < j_item else (j_item, i_item)
                        is_tabu = tabu_swap.get(key, 0) > it
                        if is_tabu and not aspiration_ok(cand_obj, best_obj, best2_for_m):
                            continue

                        score = 50
                        if bi in Tset:
                            score -= 50000
                        score += (C - li_new)

                        if best_obj_move is None or better_obj(cand_obj, best_obj_move) or (cand_obj == best_obj_move and score < best_score):
                            best_kind = "swap"
                            best_data = (i_item, j_item, bi, bj)
                            best_obj_move = cand_obj
                            best_score = score

                        trials += 1
                        if trials >= swap_trials_per_item:
                            break
                    if trials >= swap_trials_per_item:
                        break

        # -----------------
        # 2-1 emptying attempt
        # -----------------
        do_two1 = (it % 35 == 0) or (iters_since_best > stagnation_threshold // 2)
        if do_two1 and not time_exceeded():
            b0 = T[0]
            res = try_two_to_one(b0, D, bins, loads, assign, pos, bin_ids, tabu_reloc, it, best_obj, best2_for_m)
            if res is not None:
                moves, cand_obj = res
                score = -2500000
                if best_obj_move is None or better_obj(cand_obj, best_obj_move) or (cand_obj == best_obj_move and score < best_score):
                    best_kind = "two1"
                    best_data = moves
                    best_obj_move = cand_obj
                    best_score = score

        # -----------------
        # Apply best move
        # -----------------
        if best_kind is None:
            iters_since_best += 1
        else:
            exp = it + tenure + random.randint(0, tenure)
            if best_kind == "reloc":
                i_item, b_from, b_to = best_data
                src_id = bin_ids[b_from]
                dst_id = bin_ids[b_to]
                apply_reloc(i_item, b_from, b_to, bins, loads, assign, pos, bin_ids)
                tabu_reloc[(i_item, src_id)] = exp
                tabu_reloc[(i_item, dst_id)] = exp

            elif best_kind == "swap":
                i_item, j_item, bi, bj = best_data
                # record old ids before swap for relocation tabu
                id_bi = bin_ids[bi]
                id_bj = bin_ids[bj]
                apply_swap(i_item, j_item, bi, bj, bins, loads, assign, pos)
                key = (i_item, j_item) if i_item < j_item else (j_item, i_item)
                tabu_swap[key] = exp
                tabu_reloc[(i_item, id_bi)] = exp
                tabu_reloc[(j_item, id_bj)] = exp

            else:  # two1
                moves = best_data
                # apply in sequence; b0 might disappear only after second move -> safe
                for (itm, bf, bt) in moves:
                    if bf >= len(bins) or bt >= len(bins):
                        break
                    src_id = bin_ids[bf]
                    dst_id = bin_ids[bt]
                    apply_reloc(itm, bf, bt, bins, loads, assign, pos, bin_ids)
                    tabu_reloc[(itm, src_id)] = exp
                    tabu_reloc[(itm, dst_id)] = exp

            # Update tracking
            curr2 = obj_from(loads)
            mm = curr2[0]
            prev = best2_for_m.get(mm)
            if prev is None or curr2[1] < prev:
                best2_for_m[mm] = curr2[1]

            if better_obj(curr2, best_obj):
                best_obj = curr2
                best_bins = [b[:] for b in bins]
                best_loads = loads[:]
                best_assign = assign[:]
                best_bin_ids = bin_ids[:]
                push_elite(best_obj, bins, loads, assign, pos, bin_ids, next_bin_id)
                iters_since_best = 0
            else:
                iters_since_best += 1

        # -----------------
        # Diversification / restart
        # -----------------
        if iters_since_best > stagnation_threshold and not time_exceeded():
            # perturb: remove items from a few light bins and reinsert with randomized best-fit
            k_pert = 2 if len(bins) < 25 else 3
            light = sorted(range(len(bins)), key=lambda b: loads[b])[: min(k_pert, len(bins))]
            removed: List[int] = []

            # clear tabu to allow movement
            tabu_reloc.clear()
            tabu_swap.clear()

            for b in sorted(light, reverse=True):
                for itm in list(bins[b]):
                    removed.append(itm)
                    remove_item(itm, b, bins, loads, assign, pos)
                remove_empty_bin(b, bins, loads, assign, pos, bin_ids)

            if removed:
                removed.sort(key=lambda i: (-weights[i], random.random()))
                for itm in removed:
                    w = weights[itm]
                    if not bins:
                        bnew, next_bin_id = add_bin(bins, loads, bin_ids, next_bin_id)
                        add_item(itm, bnew, bins, loads, assign, pos)
                        continue
                    D2 = pick_dest_bins(loads, min(140, len(bins)), 12)
                    bf = best_fit_r(D2, loads, w, 7, forbid=-1)
                    if bf:
                        bto = bf[0] if random.random() < 0.6 else random.choice(bf)
                    else:
                        bto, next_bin_id = add_bin(bins, loads, bin_ids, next_bin_id)
                    add_item(itm, bto, bins, loads, assign, pos)

            # restart from elite sometimes
            if elite and random.random() < 0.7:
                pick = random.choice(elite[: min(4, len(elite))])
                _, eb, el, ea, ep, eids, enext = pick
                bins = [x[:] for x in eb]
                loads = el[:]
                assign = ea[:]
                pos = ep[:]
                bin_ids = eids[:]
                next_bin_id = enext

            iters_since_best = 0
            tenure = max(min_tenure, min(max_tenure, int(8 + 0.9 * (n ** 0.5))))

    # -----------------------
    # Output best
    # -----------------------
    packing = [b[:] for b in best_bins if b]
    bin_weights = [sum(weights[i] for i in b) for b in packing]
    return {"packing": packing, "bin_weights": bin_weights}
