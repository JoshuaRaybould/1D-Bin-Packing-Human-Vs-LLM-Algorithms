import time
import random
from typing import List, Tuple, Dict, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Allow using up to 100s if caller gives >=40, per statement.
    # Still respect smaller limits.
    hard_limit = min(100.0, float(time_limit))

    if any(w > bin_capacity for w in weights):
        packing = [[i] for i in range(n)]
        bin_w = [weights[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bin_w}

    def elapsed() -> float:
        return time.perf_counter() - start

    C = bin_capacity

    # ---------------- Lower bounds ----------------
    total_w = sum(weights)
    lb1 = (total_w + C - 1) // C

    # LB2: simple large-item counting for k=2..7
    def lb_large_items() -> int:
        best = 0
        for k in (2, 3, 4, 5, 6, 7):
            thr = C / k
            cnt = 0
            for w in weights:
                if w > thr:
                    cnt += 1
            best = max(best, (cnt + (k - 2)) // (k - 1))
        return best

    lb2 = lb_large_items()

    # LB3: pair incompatibility matching on "large" items
    # If w_i + w_j > C, they cannot be together.
    # For items > C/2, any pair is incompatible, so LB is count.
    # More generally, take items > C/3 and compute a greedy maximum matching
    # in the incompatibility graph's complement? Here we need a *lower bound*:
    # build graph where edge means "can be paired" (sum<=C), then maximum pairing
    # gives how many can share bins in pairs; remaining need separate bins.
    # This is a known cheap strengthening.
    def lb_pairing() -> int:
        # consider only items > C/3 to keep it small but meaningful
        idx = [i for i, w in enumerate(weights) if w > C / 3]
        m = len(idx)
        if m == 0:
            return 0
        ws = sorted((weights[i] for i in idx), reverse=True)
        # two-pointer pairing to maximize number of pairs with sum<=C
        # (this maximizes pairs in this special case)
        i, j = 0, m - 1
        pairs = 0
        while i < j:
            if ws[i] + ws[j] <= C:
                pairs += 1
                i += 1
                j -= 1
            else:
                i += 1
        # If we can make `pairs` pairs, then bins needed for these m items is m - pairs
        return m - pairs

    lb3 = lb_pairing()
    LB = max(lb1, lb2, lb3)

    # --------------- Metrics / comparisons ---------------
    # primary: #bins; secondary: maximize sum of squared fills; tertiary: minimize total slack
    def metrics(bin_w: List[int]) -> Tuple[int, int, int]:
        s_sq = 0
        slack = 0
        for bw in bin_w:
            s_sq += bw * bw
            slack += (C - bw)
        return (len(bin_w), -s_sq, slack)

    def better(m1: Tuple[int, int, int], m2: Optional[Tuple[int, int, int]]) -> bool:
        return m2 is None or m1 < m2

    # --------------- Construction (GRASP) ---------------
    # We use item-RCL by weight; bin choice uses candidate list based on best-fit score.

    def construct(alpha: float, item_rcl: int, bin_rcl: int) -> Tuple[List[List[int]], List[int], List[int]]:
        items = list(range(n))
        # random tie-breaker
        items.sort(key=lambda i: (-weights[i], random.random()))

        packing: List[List[int]] = []
        bin_w: List[int] = []
        rem: List[int] = []
        where = [-1] * n

        ops = 0
        TIME_MASK = 511

        # dynamic list for RCL selection among first item_rcl
        remaining = items

        while remaining:
            ops += 1
            if (ops & TIME_MASK) == 0 and elapsed() >= hard_limit:
                break

            k = item_rcl if item_rcl < len(remaining) else len(remaining)
            pos = random.randrange(k)
            it = remaining.pop(pos)
            w = weights[it]

            # build feasible bins list with best-fit score = slack after
            feas: List[Tuple[int, int]] = []  # (slack_after, bin)
            for b, rb in enumerate(rem):
                if rb >= w:
                    feas.append((rb - w, b))

            if feas:
                # restrict to best few by slack
                feas.sort(key=lambda x: x[0])
                k2 = min(bin_rcl, len(feas))
                # GRASP thresholding: allow within alpha of best slack
                best_sl = feas[0][0]
                worst_sl = feas[k2 - 1][0]
                # slack threshold
                thr = best_sl + int(alpha * (worst_sl - best_sl))
                rcl = [b for sl, b in feas[:k2] if sl <= thr]
                if not rcl:
                    rcl = [feas[0][1]]
                chosen = random.choice(rcl)
                packing[chosen].append(it)
                bin_w[chosen] += w
                rem[chosen] -= w
                where[it] = chosen
            else:
                # open new bin
                b = len(packing)
                packing.append([it])
                bin_w.append(w)
                rem.append(C - w)
                where[it] = b

        # finish any unassigned items with deterministic best-fit
        if remaining:
            for it in remaining:
                w = weights[it]
                best_b = -1
                best_sl = None
                for b, rb in enumerate(rem):
                    if rb >= w:
                        sl = rb - w
                        if best_sl is None or sl < best_sl:
                            best_sl = sl
                            best_b = b
                if best_b == -1:
                    b = len(packing)
                    packing.append([it])
                    bin_w.append(w)
                    rem.append(C - w)
                    where[it] = b
                else:
                    packing[best_b].append(it)
                    bin_w[best_b] += w
                    rem[best_b] -= w
                    where[it] = best_b

        return packing, bin_w, where

    # --------------- Local search (GRASP essential) ---------------
    # Neighborhood: try to eliminate bins by reinserting their items (with a few strategies),
    # plus limited "kick" (ejection) moves to create space.

    def rebuild_rem(bin_w: List[int]) -> List[int]:
        return [C - bw for bw in bin_w]

    def remove_bin(packing: List[List[int]], bin_w: List[int], rem: List[int], where: List[int], bidx: int) -> None:
        packing.pop(bidx)
        bin_w.pop(bidx)
        rem.pop(bidx)
        for it in range(n):
            b = where[it]
            if b > bidx:
                where[it] = b - 1

    def try_reinsert_items(
        packing: List[List[int]], bin_w: List[int], rem: List[int], where: List[int],
        bidx: int, order: List[int], attempt_kick: bool
    ) -> bool:
        # Work on a plan first
        tmp_rem = rem[:]
        moves: List[Tuple[int, int]] = []
        kicks: List[Tuple[int, int, int]] = []  # (kicked_item, from_bin, to_bin)

        for it in order:
            w = weights[it]

            # candidate bins (best-fit)
            best_b = -1
            best_sl = None
            for b, rb in enumerate(tmp_rem):
                if b == bidx:
                    continue
                if rb >= w:
                    sl = rb - w
                    if best_sl is None or sl < best_sl:
                        best_sl = sl
                        best_b = b

            if best_b != -1:
                tmp_rem[best_b] -= w
                moves.append((it, best_b))
                continue

            if not attempt_kick:
                return False

            # one-level ejection: find a bin that can accept it after kicking one item to somewhere else
            done = False
            # scan a small set of promising bins: those with smallest slack deficit
            cand_bins = list(range(len(tmp_rem)))
            cand_bins.sort(key=lambda b: (0 if b == bidx else max(0, w - tmp_rem[b])))
            for b in cand_bins[: min(12, len(cand_bins))]:
                if b == bidx:
                    continue
                need = w - tmp_rem[b]
                if need <= 0:
                    continue

                # choose a kick candidate from bin b: smallest item that frees enough
                if not packing[b]:
                    continue
                kick_cands = sorted(packing[b], key=lambda x: weights[x])
                for kt in kick_cands[: min(8, len(kick_cands))]:
                    wk = weights[kt]
                    if wk < need:
                        continue
                    # place kicked item somewhere else (best-fit)
                    dest2 = -1
                    best2 = None
                    for bb, rb2 in enumerate(tmp_rem):
                        if bb == bidx or bb == b:
                            continue
                        if rb2 >= wk:
                            sl2 = rb2 - wk
                            if best2 is None or sl2 < best2:
                                best2 = sl2
                                dest2 = bb
                    if dest2 == -1:
                        continue

                    # simulate
                    tmp_rem[dest2] -= wk
                    tmp_rem[b] += wk
                    tmp_rem[b] -= w
                    kicks.append((kt, b, dest2))
                    moves.append((it, b))
                    done = True
                    break
                if done:
                    break

            if not done:
                return False

        # Apply kicks first (so capacity updates match plan)
        for kt, b_from, b_to in kicks:
            packing[b_from].remove(kt)
            packing[b_to].append(kt)
            bin_w[b_from] -= weights[kt]
            bin_w[b_to] += weights[kt]
            rem[b_from] += weights[kt]
            rem[b_to] -= weights[kt]
            where[kt] = b_to

        # Apply moves
        for it, dest in moves:
            if it in packing[bidx]:
                packing[bidx].remove(it)
                bin_w[bidx] -= weights[it]
                rem[bidx] += weights[it]
            packing[dest].append(it)
            bin_w[dest] += weights[it]
            rem[dest] -= weights[it]
            where[it] = dest

        if packing[bidx]:
            return False

        remove_bin(packing, bin_w, rem, where, bidx)
        return True

    def local_search(packing: List[List[int]], bin_w: List[int], where: List[int], max_passes: int) -> Tuple[List[List[int]], List[int], List[int]]:
        rem = rebuild_rem(bin_w)
        ops = 0
        TIME_MASK = 255

        for _ in range(max_passes):
            # prioritize light bins (easier to empty)
            order_bins = list(range(len(packing)))
            order_bins.sort(key=lambda b: (bin_w[b], len(packing[b])))

            improved = False

            for bidx in order_bins[: min(28, len(order_bins))]:
                ops += 1
                if (ops & TIME_MASK) == 0 and elapsed() >= hard_limit:
                    return packing, bin_w, where
                if bidx >= len(packing) or len(packing) <= 1:
                    continue

                items = packing[bidx][:]
                if not items:
                    continue

                # Several attempts: decreasing difficulty
                # 1) large-first without kicks
                # 2) random order without kicks
                # 3) large-first with kicks
                attempts = [
                    (sorted(items, key=lambda it: weights[it], reverse=True), False),
                    (random.sample(items, len(items)), False),
                    (sorted(items, key=lambda it: weights[it], reverse=True), True),
                    (random.sample(items, len(items)), True),
                ]

                for order, kick in attempts:
                    if elapsed() >= hard_limit:
                        return packing, bin_w, where
                    if try_reinsert_items(packing, bin_w, rem, where, bidx, order, attempt_kick=kick):
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return packing, bin_w, where

    # --------------- Elite set + path relinking (GRASP standard enhancement) ---------------
    # Keep small elite by bins then fill quality; relink using target bin of each item.

    def signature(bin_w: List[int]) -> Tuple[int, ...]:
        return tuple(sorted(bin_w, reverse=True))

    def clone_packing(p: List[List[int]]) -> List[List[int]]:
        return [b[:] for b in p]

    def add_elite(
        elite: List[Tuple[Tuple[int, int, int], Tuple[int, ...], List[List[int]], List[int]]],
        m: Tuple[int, int, int],
        packing: List[List[int]],
        bin_w: List[int],
        E: int,
    ) -> None:
        sig = signature(bin_w)
        for _, esig, _, _ in elite:
            if esig == sig:
                return
        elite.append((m, sig, clone_packing(packing), bin_w[:]))
        elite.sort(key=lambda x: x[0])
        if len(elite) > E:
            elite[:] = elite[:E]

    def assignment_from_packing(packing: List[List[int]]) -> List[int]:
        w = [-1] * n
        for b, items in enumerate(packing):
            for it in items:
                w[it] = b
        return w

    def path_relink(
        pack_a: List[List[int]], bw_a: List[int],
        pack_b: List[List[int]], bw_b: List[int],
        max_moves: int,
    ) -> Optional[Tuple[List[List[int]], List[int]]]:
        if elapsed() >= hard_limit:
            return None

        packing = clone_packing(pack_a)
        bin_w = bw_a[:]
        rem = rebuild_rem(bin_w)
        where = assignment_from_packing(packing)
        where_b = assignment_from_packing(pack_b)

        # Build target bins by grouping items in B by decreasing load order to reduce label mismatch
        # Map bins in B to ranks by load, then targets are rank bins in current solution by load.
        order_b = list(range(len(bw_b)))
        order_b.sort(key=lambda b: bw_b[b], reverse=True)
        rank_b = [0] * len(bw_b)
        for r, b in enumerate(order_b):
            rank_b[b] = r

        # In current A, maintain order by load and update occasionally
        def order_a_bins() -> List[int]:
            order = list(range(len(bin_w)))
            order.sort(key=lambda b: bin_w[b], reverse=True)
            return order

        best_m = metrics(bin_w)
        best_pack = None
        best_bw = None

        candidates = [it for it in range(n) if where_b[it] != -1]
        random.shuffle(candidates)

        ops = 0
        TIME_MASK = 63
        ord_a = order_a_bins()

        while candidates and ops < max_moves:
            ops += 1
            if (ops & TIME_MASK) == 0 and elapsed() >= hard_limit:
                break
            if ops % 20 == 0:
                ord_a = order_a_bins()

            it = candidates.pop()
            cur = where[it]
            if cur < 0:
                continue

            tb = where_b[it]
            if tb < 0:
                continue
            tr = rank_b[tb]
            dest = ord_a[min(tr, len(ord_a) - 1)] if ord_a else -1
            if dest == -1 or dest == cur:
                continue

            w = weights[it]
            if rem[dest] < w:
                continue

            # move
            packing[cur].remove(it)
            packing[dest].append(it)
            bin_w[cur] -= w
            bin_w[dest] += w
            rem[cur] += w
            rem[dest] -= w
            where[it] = dest

            # remove empty bin
            if not packing[cur]:
                remove_bin(packing, bin_w, rem, where, cur)
                ord_a = order_a_bins()

            m = metrics(bin_w)
            if m < best_m:
                best_m = m
                best_pack = clone_packing(packing)
                best_bw = bin_w[:]
                if best_m[0] == LB:
                    break

        if best_pack is None:
            return None
        return best_pack, best_bw

    # --------------- Reactive alpha ---------------
    alphas = [0.05, 0.10, 0.15, 0.22, 0.30, 0.40]
    a_score = [1.0 for _ in alphas]
    a_uses = [1 for _ in alphas]

    def pick_alpha() -> float:
        # probability proportional to score/uses (lower bins gives higher reward later)
        vals = [a_score[i] / a_uses[i] for i in range(len(alphas))]
        s = sum(vals)
        r = random.random() * s
        acc = 0.0
        for i, v in enumerate(vals):
            acc += v
            if acc >= r:
                a_uses[i] += 1
                return alphas[i]
        a_uses[-1] += 1
        return alphas[-1]

    def reward_alpha(alpha: float, bins_used: int) -> None:
        # reward smaller bin count strongly
        i = alphas.index(alpha)
        # scale reward by how close to LB
        gap = max(0, bins_used - LB)
        a_score[i] += 1.0 / (1.0 + gap)

    # --------------- Main GRASP loop ---------------
    MAX_ITER = 2_000_000
    ELITE_SIZE = 18

    best_pack: List[List[int]] = []
    best_bw: List[int] = []
    best_m: Optional[Tuple[int, int, int]] = None

    elite: List[Tuple[Tuple[int, int, int], Tuple[int, ...], List[List[int]], List[int]]] = []

    no_improve = 0
    ops = 0
    TIME_MASK = 127

    # base parameters
    base_item_rcl = max(8, int(n ** 0.5))
    base_bin_rcl = 32

    for it in range(MAX_ITER):
        ops += 1
        if (ops & TIME_MASK) == 0 and elapsed() >= hard_limit:
            break

        alpha = pick_alpha()

        # Diversification/intensification schedule
        if no_improve < 250:
            item_rcl = min(28, base_item_rcl)
            bin_rcl = base_bin_rcl
            ls_passes = 4
        elif no_improve < 700:
            item_rcl = min(34, max(base_item_rcl, 14))
            bin_rcl = 40
            ls_passes = 5
        else:
            item_rcl = min(42, max(base_item_rcl, 18))
            bin_rcl = 48
            ls_passes = 6

        packing, bin_w, where = construct(alpha, item_rcl=item_rcl, bin_rcl=bin_rcl)
        packing, bin_w, where = local_search(packing, bin_w, where, max_passes=ls_passes)
        m = metrics(bin_w)

        reward_alpha(alpha, m[0])

        if better(m, best_m):
            best_m = m
            best_pack = clone_packing(packing)
            best_bw = bin_w[:]
            no_improve = 0
            add_elite(elite, m, packing, bin_w, ELITE_SIZE)

            # path relinking intensification with a couple of elites
            if elite and elapsed() < hard_limit:
                for _ in range(min(3, len(elite))):
                    em, esig, ep, ebw = random.choice(elite)
                    if esig == signature(best_bw):
                        continue
                    pr = path_relink(best_pack, best_bw, ep, ebw, max_moves=260)
                    if pr is None or elapsed() >= hard_limit:
                        continue
                    pr_pack, pr_bw = pr
                    pr_where = assignment_from_packing(pr_pack)
                    pr_pack, pr_bw, pr_where = local_search(pr_pack, pr_bw, pr_where, max_passes=3)
                    pr_m = metrics(pr_bw)
                    add_elite(elite, pr_m, pr_pack, pr_bw, ELITE_SIZE)
                    if better(pr_m, best_m):
                        best_m = pr_m
                        best_pack = clone_packing(pr_pack)
                        best_bw = pr_bw[:]
                        if best_m[0] == LB:
                            break

            if best_m[0] == LB:
                break
        else:
            no_improve += 1
            add_elite(elite, m, packing, bin_w, ELITE_SIZE)

            # occasional relinking during stagnation
            if elite and (no_improve % 120 == 0) and elapsed() < hard_limit:
                em, esig, ep, ebw = random.choice(elite)
                pr1 = path_relink(packing, bin_w, ep, ebw, max_moves=200)
                pr2 = path_relink(ep, ebw, packing, bin_w, max_moves=200)
                for pr in (pr1, pr2):
                    if pr is None or elapsed() >= hard_limit:
                        continue
                    pr_pack, pr_bw = pr
                    pr_where = assignment_from_packing(pr_pack)
                    pr_pack, pr_bw, pr_where = local_search(pr_pack, pr_bw, pr_where, max_passes=2)
                    pr_m = metrics(pr_bw)
                    add_elite(elite, pr_m, pr_pack, pr_bw, ELITE_SIZE)
                    if better(pr_m, best_m):
                        best_m = pr_m
                        best_pack = clone_packing(pr_pack)
                        best_bw = pr_bw[:]
                        no_improve = 0
                        if best_m[0] == LB:
                            break

    # Fallback
    if best_m is None:
        p, bw, _ = construct(0.15, item_rcl=min(25, n), bin_rcl=32)
        best_pack, best_bw = p, bw

    final_bw = [sum(weights[i] for i in b) for b in best_pack]
    return {"packing": best_pack, "bin_weights": final_bw}
