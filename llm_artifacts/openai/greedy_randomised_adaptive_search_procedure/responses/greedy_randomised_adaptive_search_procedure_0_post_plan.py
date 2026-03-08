import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # --- Feasibility guard (benchmarks usually feasible) ---
    # Policy: if an overweight item exists, return a trivial packing placing each item alone.
    # (Still infeasible for those items, but avoids crashes.)
    if any(w > bin_capacity for w in weights):
        packing = [[i] for i in range(n)]
        bin_w = [weights[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bin_w}

    def now() -> float:
        return time.perf_counter()

    def elapsed() -> float:
        return now() - start

    # --- Lower bounds ---
    total_w = sum(weights)
    lb1 = (total_w + bin_capacity - 1) // bin_capacity

    # LB2 (cheap large-item counting): for k=2..5
    # If w > C/k then at most (k-1) such items per bin.
    def lb_large_items() -> int:
        best = 0
        C = bin_capacity
        for k in (2, 3, 4, 5):
            thr = C / k
            cnt = 0
            for w in weights:
                if w > thr:
                    cnt += 1
            best = max(best, (cnt + (k - 2)) // (k - 1) if k > 1 else cnt)
        return best

    lb2 = lb_large_items()
    LB = max(lb1, lb2)

    # --- Solution representation utilities ---
    # packing: List[List[int]]
    # bin_w: List[int]
    # where[item] = bin index

    def compute_metrics(bin_w: List[int]) -> Tuple[int, int, int, int]:
        # (bins, -sum_sq_fills, sum_slack, -min_fill)
        # We will compare lexicographically with bins first.
        s_sq = 0
        s_slack = 0
        min_fill = bin_capacity if bin_w else 0
        for bw in bin_w:
            s_sq += bw * bw
            s_slack += (bin_capacity - bw)
            if bw < min_fill:
                min_fill = bw
        return (len(bin_w), -s_sq, s_slack, -min_fill)

    def better_metrics(m_a: Tuple[int, int, int, int], m_b: Optional[Tuple[int, int, int, int]]) -> bool:
        if m_b is None:
            return True
        return m_a < m_b

    # --- Construction (GRASP) ---
    # Multiple placement modes
    MODE_BEST = 0
    MODE_FIRST = 1
    MODE_WORST = 2

    def construct(alpha: float, open_penalty: float, mode: int, k_items: int) -> Tuple[List[List[int]], List[int], List[int]]:
        # Item-RCL: repeatedly pick next item among top-k by weight.
        # Keep items sorted by (-w, noise) for tie diversity.
        items = list(range(n))
        items.sort(key=lambda i: (-weights[i], random.random()))

        packing: List[List[int]] = []
        bin_w: List[int] = []
        rem: List[int] = []  # remaining capacity
        where = [-1] * n

        # For speed: pop from front via index pointer; we still need RCL among first k.
        ptr = 0
        # We will maintain a dynamic list; removal from middle is O(k) bounded by k_items.
        remaining_items = items

        op_counter = 0
        TIME_CHECK_EVERY = 256

        while remaining_items:
            op_counter += 1
            if (op_counter & (TIME_CHECK_EVERY - 1)) == 0 and elapsed() >= time_limit:
                break

            k = k_items if k_items < len(remaining_items) else len(remaining_items)
            # RCL among top-k heavy items
            pick_pos = random.randrange(k)
            it = remaining_items.pop(pick_pos)
            w = weights[it]

            # Build feasible bin candidates
            candidates: List[Tuple[float, int]] = []  # (score, bin_index) ; bin_index == -1 => open new

            if mode == MODE_FIRST:
                # first-fit: choose first feasible, but still GRASP-ified by considering a small prefix
                for b in range(len(rem)):
                    if rem[b] >= w:
                        slack_after = rem[b] - w
                        # score encourages earlier bins with slight slack preference
                        score = float(b) + 0.001 * slack_after
                        candidates.append((score, b))
                        # collect a few to allow random choice
                        if len(candidates) >= 25:
                            break
            else:
                for b, rb in enumerate(rem):
                    if rb >= w:
                        slack_after = rb - w
                        if mode == MODE_BEST:
                            score = slack_after
                        else:
                            # worst-fit: prefer larger slack_after
                            score = -slack_after
                        # tiny noise for tie-breaking
                        candidates.append((score + 1e-9 * random.random(), b))

            # Regret signal: if only one tight bin exists, discourage wasting it unless necessary.
            # We fold it into the decision of opening a new bin.
            best_slack = None
            second_slack = None
            for b, rb in enumerate(rem):
                if rb >= w:
                    s = rb - w
                    if best_slack is None or s < best_slack:
                        second_slack = best_slack
                        best_slack = s
                    elif second_slack is None or s < second_slack:
                        second_slack = s
            regret = 0
            if best_slack is not None:
                if second_slack is None:
                    regret = bin_capacity  # high
                else:
                    regret = second_slack - best_slack

            # Add open-new-bin candidate even if feasible placements exist.
            # Make it competitive mainly for large items or when all placements are "bad".
            # Lower score is better.
            # Base score: remaining slack in new bin + penalty
            new_bin_score = (bin_capacity - w) + open_penalty
            # Encourage opening for very large items
            if w >= int(0.7 * bin_capacity):
                new_bin_score -= 0.35 * open_penalty
            # If regret is high, opening a new bin can avoid consuming a rare tight spot
            if regret >= int(0.15 * bin_capacity):
                new_bin_score -= 0.15 * open_penalty
            candidates.append((new_bin_score + 1e-9 * random.random(), -1))

            # Choose from bin-RCL using top-k then alpha to pick among the best fraction.
            candidates.sort(key=lambda x: x[0])
            k_bins = min(25, len(candidates))
            # alpha in [0.05..0.6]; choose among best m candidates
            m = max(1, int((0.1 + alpha) * k_bins))
            chosen_score, chosen_bin = random.choice(candidates[:m])

            if chosen_bin == -1:
                b = len(packing)
                packing.append([it])
                bin_w.append(w)
                rem.append(bin_capacity - w)
                where[it] = b
            else:
                packing[chosen_bin].append(it)
                bin_w[chosen_bin] += w
                rem[chosen_bin] -= w
                where[it] = chosen_bin

        # If time cut construction short, place any unassigned items greedily (best-fit) to remain feasible.
        if remaining_items:
            # finish deterministically best-fit
            for it in remaining_items:
                w = weights[it]
                best_b = -1
                best_s = None
                for b, rb in enumerate(rem):
                    if rb >= w:
                        s = rb - w
                        if best_s is None or s < best_s:
                            best_s = s
                            best_b = b
                if best_b == -1:
                    b = len(packing)
                    packing.append([it])
                    bin_w.append(w)
                    rem.append(bin_capacity - w)
                    where[it] = b
                else:
                    packing[best_b].append(it)
                    bin_w[best_b] += w
                    rem[best_b] -= w
                    where[it] = best_b

        return packing, bin_w, where

    # --- Local Search (standard GRASP component) ---
    def rebuild_rem(bin_w: List[int]) -> List[int]:
        return [bin_capacity - bw for bw in bin_w]

    def try_empty_bin_plan(
        packing: List[List[int]], bin_w: List[int], rem: List[int], where: List[int],
        bidx: int, order: List[int], insert_mode: int
    ) -> bool:
        # Build a relocation plan without mutating; apply only if all items can be placed.
        tmp_rem = rem[:]  # O(#bins)
        moves: List[Tuple[int, int]] = []  # (item, dest_bin)

        for it in order:
            w = weights[it]
            best_b = -1

            if insert_mode == MODE_FIRST:
                for b in range(len(tmp_rem)):
                    if b == bidx:
                        continue
                    if tmp_rem[b] >= w:
                        best_b = b
                        break
            else:
                best_score = None
                for b, rb in enumerate(tmp_rem):
                    if b == bidx or rb < w:
                        continue
                    slack_after = rb - w
                    score = slack_after if insert_mode == MODE_BEST else -slack_after
                    if best_score is None or score < best_score:
                        best_score = score
                        best_b = b

            if best_b == -1:
                return False

            tmp_rem[best_b] -= w
            moves.append((it, best_b))

        # Apply moves
        # Remove all items from bidx, then insert
        for it, dest in moves:
            # remove from source bin list
            packing[bidx].remove(it)
            bin_w[bidx] -= weights[it]
            rem[bidx] += weights[it]
            where[it] = -1

            packing[dest].append(it)
            bin_w[dest] += weights[it]
            rem[dest] -= weights[it]
            where[it] = dest

        if packing[bidx]:
            return False

        # Remove the emptied bin; update indices in where for bins after bidx
        packing.pop(bidx)
        bin_w.pop(bidx)
        rem.pop(bidx)
        for it in range(n):
            b = where[it]
            if b > bidx:
                where[it] = b - 1
        return True

    def swap_1_1(
        packing: List[List[int]], bin_w: List[int], rem: List[int], where: List[int],
        max_trials: int, time_check_mask: int
    ) -> bool:
        # Limited randomized 1-1 swaps to improve compactness / enable future emptying.
        # Accept only if improves metrics (same bin count).
        if len(packing) < 2:
            return False

        base_m = compute_metrics(bin_w)
        improved = False
        B = len(packing)

        # Focus on bins with large remaining capacity
        idxs = list(range(B))
        idxs.sort(key=lambda b: rem[b], reverse=True)
        focus = idxs[: min(12, B)]

        t = 0
        while t < max_trials:
            t += 1
            if (t & time_check_mask) == 0 and elapsed() >= time_limit:
                break

            a = random.choice(focus)
            b = random.randrange(B)
            if a == b or not packing[a] or not packing[b]:
                continue

            # consider only largest few items from each bin
            la = packing[a]
            lb = packing[b]
            cand_a = sorted(la, key=lambda it: weights[it], reverse=True)[: min(6, len(la))]
            cand_b = sorted(lb, key=lambda it: weights[it], reverse=True)[: min(6, len(lb))]

            ia = random.choice(cand_a)
            ib = random.choice(cand_b)
            wa = weights[ia]
            wb = weights[ib]

            # feasibility after swap
            # a loses wa gains wb => rem[a] + wa - wb must be >=0
            # b loses wb gains wa => rem[b] + wb - wa must be >=0
            if rem[a] + wa - wb < 0 or rem[b] + wb - wa < 0:
                continue

            # apply swap
            la.remove(ia)
            lb.remove(ib)
            la.append(ib)
            lb.append(ia)
            where[ia] = b
            where[ib] = a

            bin_w[a] += (wb - wa)
            bin_w[b] += (wa - wb)
            rem[a] += (wa - wb)
            rem[b] += (wb - wa)

            new_m = compute_metrics(bin_w)
            if new_m < base_m:
                base_m = new_m
                improved = True
            else:
                # revert
                la.remove(ib)
                lb.remove(ia)
                la.append(ia)
                lb.append(ib)
                where[ia] = a
                where[ib] = b
                bin_w[a] -= (wb - wa)
                bin_w[b] -= (wa - wb)
                rem[a] -= (wa - wb)
                rem[b] -= (wb - wa)

        return improved

    def local_search(
        packing: List[List[int]], bin_w: List[int], where: List[int],
        passes: int, empty_tries: int, do_swaps: bool
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        rem = rebuild_rem(bin_w)
        TIME_CHECK_MASK = 63

        for _ in range(passes):
            if elapsed() >= time_limit:
                break

            improved = False

            # Try to empty bins: prioritize light bins and bins with fewer items
            order_bins = list(range(len(packing)))
            order_bins.sort(key=lambda b: (bin_w[b], len(packing[b])))

            # Consider only a prefix to bound work
            max_targets = min(len(order_bins), 18)
            targets = order_bins[:max_targets]

            for idx, bidx in enumerate(targets):
                if elapsed() >= time_limit:
                    break
                if bidx >= len(packing) or len(packing) <= 1:
                    continue

                items_in_bin = packing[bidx][:]
                if not items_in_bin:
                    continue

                # Multiple randomized attempts with different insertion modes
                for t in range(empty_tries):
                    if (t & TIME_CHECK_MASK) == 0 and elapsed() >= time_limit:
                        break

                    # shuffle order; sometimes large-first helps
                    if t % 3 == 0:
                        attempt_order = sorted(items_in_bin, key=lambda it: weights[it], reverse=True)
                    else:
                        attempt_order = items_in_bin[:]
                        random.shuffle(attempt_order)

                    insert_mode = (t % 3)  # cycle best/first/worst
                    if try_empty_bin_plan(packing, bin_w, rem, where, bidx, attempt_order, insert_mode):
                        improved = True
                        # restart passes after structural change
                        break

                if improved:
                    break

            if do_swaps and not improved and elapsed() < time_limit:
                # Swaps to unlock further eliminations
                if swap_1_1(packing, bin_w, rem, where, max_trials=120, time_check_mask=63):
                    improved = True

            if not improved:
                break

        return packing, bin_w, where

    # --- Elite set + diversity ---
    EliteEntry = Tuple[Tuple[int, int, int, int], List[List[int]], List[int]]

    def signature(bin_w: List[int]) -> Tuple[int, ...]:
        # Cheap diversity signature: sorted bin loads
        return tuple(sorted(bin_w, reverse=True))

    def add_to_elite(
        elite: List[Tuple[Tuple[int, int, int, int], Tuple[int, ...], List[List[int]], List[int]]],
        m: Tuple[int, int, int, int],
        packing: List[List[int]],
        bin_w: List[int],
        E: int
    ) -> None:
        sig = signature(bin_w)
        # avoid duplicates
        for _, esig, _, _ in elite:
            if esig == sig:
                return

        entry = (m, sig, [b[:] for b in packing], bin_w[:])
        if len(elite) < E:
            elite.append(entry)
            elite.sort(key=lambda x: x[0])
            return

        # Add if better than worst or if sufficiently different and near-best
        worst_m = elite[-1][0]
        if m < worst_m:
            elite.append(entry)
            elite.sort(key=lambda x: x[0])
            del elite[E:]
            return

        # near-best diversity admission
        # accept if within 1 bin of best and signature far enough (simple Hamming-ish on loads)
        best_m = elite[0][0]
        if m[0] <= best_m[0] + 1:
            # distance: count positions differing among first K loads
            K = min(12, len(sig), len(elite[0][1]))
            dist = 0
            ref = elite[0][1]
            for i in range(K):
                if sig[i] != ref[i]:
                    dist += 1
            if dist >= max(3, K // 3):
                elite.append(entry)
                elite.sort(key=lambda x: x[0])
                del elite[E:]

    # --- Path relinking ---
    def assignment_from_packing(packing: List[List[int]]) -> List[int]:
        where = [-1] * n
        for b, items in enumerate(packing):
            for it in items:
                where[it] = b
        return where

    def relabel_bins_by_load(packing: List[List[int]], bin_w: List[int]) -> Tuple[List[int], List[int]]:
        # Return permutation old_bin -> rank in sorted-by-load order, and inverse.
        order = list(range(len(bin_w)))
        order.sort(key=lambda b: (bin_w[b], len(packing[b])), reverse=True)
        pos = [0] * len(order)
        for r, b in enumerate(order):
            pos[b] = r
        return pos, order

    def path_relink(
        pack_a: List[List[int]], bw_a: List[int],
        pack_b: List[List[int]], bw_b: List[int],
        max_moves: int
    ) -> Optional[Tuple[List[List[int]], List[int]]]:
        # Guided moves from A toward B using approximate bin alignment by load ranks.
        if elapsed() >= time_limit:
            return None

        # Work on a copy of A
        packing = [x[:] for x in pack_a]
        bin_w = bw_a[:]
        rem = rebuild_rem(bin_w)
        where = assignment_from_packing(packing)

        # Align bins by load ranks
        pos_a, order_a = relabel_bins_by_load(packing, bin_w)
        pos_b, order_b = relabel_bins_by_load(pack_b, bw_b)

        # Map each item to target rank in B (approx)
        where_b = assignment_from_packing(pack_b)
        target_rank = [-1] * n
        for it in range(n):
            bb = where_b[it]
            if bb >= 0:
                target_rank[it] = pos_b[bb]

        best_pack = None
        best_bw = None
        best_m = compute_metrics(bin_w)

        # Candidate items: those not already in a bin with desired rank
        candidates = [it for it in range(n) if target_rank[it] != -1]
        random.shuffle(candidates)

        moves_done = 0
        TIME_CHECK_MASK = 31

        while moves_done < max_moves and candidates:
            moves_done += 1
            if (moves_done & TIME_CHECK_MASK) == 0 and elapsed() >= time_limit:
                break

            it = candidates.pop()
            tr = target_rank[it]
            if tr < 0:
                continue

            cur_b = where[it]
            if cur_b < 0:
                continue

            # recompute alignment occasionally (bins may change)
            if moves_done % 25 == 0:
                pos_a, order_a = relabel_bins_by_load(packing, bin_w)

            if pos_a[cur_b] == tr:
                continue

            # Desired bin by rank; if rank out of range use closest
            if not order_a:
                continue
            desired_index = min(tr, len(order_a) - 1)
            desired_bin = order_a[desired_index]

            w = weights[it]
            # attempt move to desired bin; else try nearby ranks
            dest = -1
            if desired_bin != cur_b and rem[desired_bin] >= w:
                dest = desired_bin
            else:
                # try a few nearby bins
                for delta in (1, -1, 2, -2, 3, -3):
                    rr = desired_index + delta
                    if 0 <= rr < len(order_a):
                        b2 = order_a[rr]
                        if b2 != cur_b and rem[b2] >= w:
                            dest = b2
                            break

            if dest == -1:
                continue

            # Apply move
            packing[cur_b].remove(it)
            bin_w[cur_b] -= w
            rem[cur_b] += w
            packing[dest].append(it)
            bin_w[dest] += w
            rem[dest] -= w
            where[it] = dest

            # Remove empty bins immediately (keeps representation compact)
            if not packing[cur_b]:
                packing.pop(cur_b)
                bin_w.pop(cur_b)
                rem.pop(cur_b)
                for j in range(n):
                    if where[j] > cur_b:
                        where[j] -= 1

            m = compute_metrics(bin_w)
            if m < best_m:
                best_m = m
                best_pack = [b[:] for b in packing]
                best_bw = bin_w[:]
                if best_m[0] == LB:
                    break

        if best_pack is None:
            return None
        return best_pack, best_bw

    # --- Main loop controls ---
    MAX_ITER = 1_000_000  # fixed large; time controls actual run
    E = 15

    best_packing: List[List[int]] = []
    best_bin_w: List[int] = []
    best_m: Optional[Tuple[int, int, int, int]] = None

    elite: List[Tuple[Tuple[int, int, int, int], Tuple[int, ...], List[List[int]], List[int]]] = []

    alpha = 0.15
    no_improve = 0

    # construction mode probabilities (adapted mildly)
    modes = [MODE_BEST, MODE_FIRST, MODE_WORST]

    # open penalty baseline: discourages opening bins unless helpful
    base_open_penalty = 0.6 * bin_capacity

    for it in range(MAX_ITER):
        if elapsed() >= time_limit:
            break

        # Adaptive alpha schedule
        if no_improve >= 200:
            alpha = min(0.6, alpha + 0.1)
        
        # Adaptive open penalty: decrease under stagnation
        open_penalty = base_open_penalty * (0.85 ** (no_improve // 200))
        open_penalty = max(0.05 * bin_capacity, open_penalty)

        # Item-RCL size
        k_items = min(25, max(5, int((n ** 0.5))))
        # More diversification under stagnation
        if no_improve >= 200:
            k_items = min(30, max(k_items, 12))

        # Choose construction mode
        if no_improve < 100:
            mode = MODE_BEST if random.random() < 0.75 else random.choice(modes)
        else:
            # more diversification
            r = random.random()
            if r < 0.55:
                mode = MODE_BEST
            elif r < 0.80:
                mode = MODE_FIRST
            else:
                mode = MODE_WORST

        packing, bin_w, where = construct(alpha, open_penalty, mode, k_items)

        # Local search schedule
        # Intensify when close to LB or when we just improved best.
        m0 = compute_metrics(bin_w)
        close_to_lb = (m0[0] <= LB + 1)
        passes = 2 if not close_to_lb else 4
        empty_tries = 5 if not close_to_lb else 12
        do_swaps = True if close_to_lb or no_improve >= 150 else False

        packing, bin_w, where = local_search(packing, bin_w, where, passes, empty_tries, do_swaps)
        m = compute_metrics(bin_w)

        # Update incumbent
        if better_metrics(m, best_m):
            best_m = m
            best_packing = [b[:] for b in packing]
            best_bin_w = bin_w[:]
            no_improve = 0
            alpha = max(0.05, alpha * 0.9)

            # Add to elite
            add_to_elite(elite, m, packing, bin_w, E)

            # Path relinking on new best
            if elite and elapsed() < time_limit:
                # pick up to 2 elites different from current best
                for _ in range(min(2, len(elite))):
                    em, esig, epack, ebw = random.choice(elite)
                    if esig == signature(best_bin_w):
                        continue
                    pr = path_relink(best_packing, best_bin_w, epack, ebw, max_moves=200)
                    if pr is not None:
                        pr_pack, pr_bw = pr
                        pr_where = assignment_from_packing(pr_pack)
                        pr_pack, pr_bw, pr_where = local_search(pr_pack, pr_bw, pr_where, passes=2, empty_tries=8, do_swaps=True)
                        pr_m = compute_metrics(pr_bw)
                        if better_metrics(pr_m, best_m):
                            best_m = pr_m
                            best_packing = [b[:] for b in pr_pack]
                            best_bin_w = pr_bw[:]
                            add_to_elite(elite, pr_m, pr_pack, pr_bw, E)

            # Early exit at LB
            if best_m[0] == LB:
                break
        else:
            no_improve += 1
            # Keep elite updated with good diverse solutions
            add_to_elite(elite, m, packing, bin_w, E)

            # Periodic path relinking during stagnation
            if elite and (it % 60 == 0 or no_improve >= 300) and elapsed() < time_limit:
                em, esig, epack, ebw = random.choice(elite)
                # both directions
                pr1 = path_relink(packing, bin_w, epack, ebw, max_moves=160)
                pr2 = path_relink(epack, ebw, packing, bin_w, max_moves=160)
                for pr in (pr1, pr2):
                    if pr is None or elapsed() >= time_limit:
                        continue
                    pr_pack, pr_bw = pr
                    pr_where = assignment_from_packing(pr_pack)
                    pr_pack, pr_bw, pr_where = local_search(pr_pack, pr_bw, pr_where, passes=2, empty_tries=6, do_swaps=True)
                    pr_m = compute_metrics(pr_bw)
                    add_to_elite(elite, pr_m, pr_pack, pr_bw, E)
                    if better_metrics(pr_m, best_m):
                        best_m = pr_m
                        best_packing = [b[:] for b in pr_pack]
                        best_bin_w = pr_bw[:]
                        no_improve = 0
                        alpha = max(0.05, alpha * 0.9)
                        if best_m[0] == LB:
                            break

    # Fallback (should not happen)
    if best_m is None:
        packing, bin_w, _ = construct(0.15, base_open_penalty, MODE_BEST, min(25, n))
        best_packing, best_bin_w = packing, bin_w

    # Lightweight final alignment: recompute bin_weights to ensure consistency
    # (packing should already be consistent)
    final_bin_w = [sum(weights[i] for i in b) for b in best_packing]

    return {"packing": best_packing, "bin_weights": final_bin_w}
