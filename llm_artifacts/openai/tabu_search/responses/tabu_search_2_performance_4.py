import time
import random
import math


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    def time_exceeded() -> bool:
        return time.time() >= deadline

    # --- helpers ---
    order_desc = list(range(n))
    order_desc.sort(key=lambda i: weights[i], reverse=True)

    total_w = sum(weights)
    lb = (total_w + C - 1) // C

    # Secondary objective: encourage tight packing and one very light bin (to empty)
    # sec = sum(slack^2) - gamma*(max_slack)
    # Larger max_slack -> smaller sec -> encourages having an easy-to-empty bin.
    gamma = max(1, C // 8)

    def compute_sec(loads: list[int]) -> int:
        if not loads:
            return 0
        res_sq = 0
        min_load = loads[0]
        for L in loads:
            r = C - L
            res_sq += r * r
            if L < min_load:
                min_load = L
        max_slack = C - min_load
        return res_sq - gamma * max_slack

    def objective(loads: list[int]) -> tuple[int, int]:
        return (len(loads), compute_sec(loads))

    def delta_sq(oldL: int, newL: int) -> int:
        r0 = C - oldL
        r1 = C - newL
        return r1 * r1 - r0 * r0

    # Build packing from assignment and compact bin ids
    def build_packing_from_assign(assign: list[int]):
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

    # --- initial constructors ---
    def construct_bfd(order: list[int]):
        assign = [-1] * n
        loads: list[int] = []
        bin_items: list[list[int]] = []
        for i in order:
            w = weights[i]
            best = -1
            best_rem = None
            for b, L in enumerate(loads):
                if L + w <= C:
                    rem = C - (L + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best = b
            if best == -1:
                best = len(loads)
                loads.append(0)
                bin_items.append([])
            loads[best] += w
            bin_items[best].append(i)
            assign[i] = best
        return assign, loads, bin_items

    def construct_ffd():
        # classic FFD (first fit) over decreasing order
        assign = [-1] * n
        loads: list[int] = []
        bin_items: list[list[int]] = []
        for i in order_desc:
            w = weights[i]
            placed = False
            for b in range(len(loads)):
                if loads[b] + w <= C:
                    loads[b] += w
                    bin_items[b].append(i)
                    assign[i] = b
                    placed = True
                    break
            if not placed:
                b = len(loads)
                loads.append(w)
                bin_items.append([i])
                assign[i] = b
        return assign, loads, bin_items

    def construct_rand_bfd():
        order = order_desc[:]
        # shuffle within blocks to keep heavy-first flavor
        block = 10
        for s in range(0, n, block):
            e = min(n, s + block)
            if e - s > 1:
                random.shuffle(order[s:e])
        return construct_bfd(order)

    def construct_target_residual():
        # Pick a few target residuals from heavy items and try to match them.
        top = [weights[i] for i in order_desc[: min(n, 20)]]
        targets = []
        for _ in range(4):
            if top:
                t = random.choice(top)
                targets.append(max(0, min(C, t)))
        if not targets:
            targets = [0]

        assign = [-1] * n
        loads: list[int] = []
        bin_items: list[list[int]] = []
        for i in order_desc:
            w = weights[i]
            best = -1
            best_score = None
            for b, L in enumerate(loads):
                if L + w <= C:
                    rem = C - (L + w)
                    score = min(abs(rem - t) for t in targets)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = b
            if best == -1:
                best = len(loads)
                loads.append(0)
                bin_items.append([])
            loads[best] += w
            bin_items[best].append(i)
            assign[i] = best
        return assign, loads, bin_items

    # Simple tightening pass: reinsert items from light bins using best-fit.
    # (Still TS family: a deterministic preprocessing improvement, not a separate algorithm.)
    def tighten(assign, loads, bin_items):
        k = len(loads)
        if k <= 1:
            return assign, loads, bin_items
        # try a few lightest bins
        idxs = list(range(k))
        idxs.sort(key=lambda b: loads[b])
        attempts = min(3, k)
        for t in range(attempts):
            b = idxs[t]
            items = bin_items[b][:]
            if not items:
                continue
            items.sort(key=lambda i: weights[i], reverse=True)
            tmp_loads = loads[:]
            tmp_loads[b] = 0
            plan = []
            ok = True
            for it in items:
                w = weights[it]
                best2 = -1
                best_rem = None
                for bb, L in enumerate(tmp_loads):
                    if bb == b:
                        continue
                    if L + w <= C:
                        rem = C - (L + w)
                        if best_rem is None or rem < best_rem:
                            best_rem = rem
                            best2 = bb
                if best2 == -1:
                    ok = False
                    break
                tmp_loads[best2] += w
                plan.append((it, best2))
            if not ok:
                continue
            # apply plan
            for it, bb in plan:
                bin_items[b].remove(it)
                bin_items[bb].append(it)
                assign[it] = bb
            loads[:] = tmp_loads
            # compact
            pack, loads2, assign2 = build_packing_from_assign(assign)
            return assign2, loads2, pack
        return assign, loads, bin_items

    def initial_solution():
        cands = []
        cands.append(construct_ffd())
        cands.append(construct_bfd(order_desc))
        cands.append(construct_rand_bfd())
        cands.append(construct_rand_bfd())
        cands.append(construct_target_residual())
        best = None
        best_obj = None
        for a, L, items in cands:
            a2, L2, items2 = tighten(a[:], L[:], [lst[:] for lst in items])
            obj = objective(L2)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best = (a2, L2, items2)
            if time_exceeded():
                break
        return best

    # --- Tabu Search state structures ---
    # We maintain bin_items as list[list[int]] and positions for O(1) remove.
    def rebuild_positions(bin_items):
        pos = [dict() for _ in range(len(bin_items))]
        for b, lst in enumerate(bin_items):
            for p, it in enumerate(lst):
                pos[b][it] = p
        return pos

    def bin_remove(bin_items, bin_pos, b, it):
        p = bin_pos[b].pop(it)
        last = bin_items[b].pop()
        if last != it:
            bin_items[b][p] = last
            bin_pos[b][last] = p

    def bin_add(bin_items, bin_pos, b, it):
        bin_pos[b][it] = len(bin_items[b])
        bin_items[b].append(it)

    # Slack buckets for fast destination retrieval
    bucket_size = max(1, C // 250)  # finer buckets than before
    nbuckets = C // bucket_size + 2

    def bindex(slack: int) -> int:
        if slack <= 0:
            return 0
        idx = slack // bucket_size
        if idx >= nbuckets:
            return nbuckets - 1
        return idx

    def rebuild_buckets(loads):
        slack = [C - L for L in loads]
        b_of = [bindex(s) for s in slack]
        buckets = [[] for _ in range(nbuckets)]
        for b, idx in enumerate(b_of):
            buckets[idx].append(b)
        return slack, b_of, buckets

    def update_bucket_for_bin(b):
        s = C - cur_loads[b]
        cur_slack[b] = s
        idx = bindex(s)
        if idx != cur_bof[b]:
            cur_bof[b] = idx
            cur_buckets[idx].append(b)  # lazy cleanup

    def propose_bins_for_item(w: int, exclude: int, limit: int):
        need = w
        idx0 = bindex(need)
        cand = []
        for idx in range(idx0, nbuckets):
            if not cur_buckets[idx]:
                continue
            for b in cur_buckets[idx]:
                if b == exclude or b >= len(cur_loads):
                    continue
                if C - cur_loads[b] >= need:
                    cand.append(b)
                    if len(cand) >= limit * 4:
                        break
            if len(cand) >= limit * 4:
                break
        if not cand:
            return []
        cand.sort(key=lambda b: C - (cur_loads[b] + w))  # best-fit
        out = []
        seen = set()
        for b in cand:
            if b not in seen:
                seen.add(b)
                out.append(b)
                if len(out) >= limit:
                    break
        return out

    # Top items cache per bin to focus on hard-to-move items
    TOP = 10

    def rebuild_top(b: int):
        lst = cur_bin_items[b]
        if not lst:
            cur_top[b] = []
            cur_top_dirty[b] = False
            return
        # partial sort via full sort (TOP small, but list may be moderate)
        tmp = sorted(lst, key=lambda i: weights[i], reverse=True)
        cur_top[b] = tmp[:TOP]
        cur_top_dirty[b] = False

    # Tabu attribute: (item, dest_bin) forbidden until iteration t
    tabu = {}

    def is_tabu(it: int, dest: int, iteration: int) -> bool:
        exp = tabu.get((it, dest))
        return exp is not None and exp >= iteration

    # Elite memory for intensification
    ELITE = 8
    elite = {}  # k -> list[(obj, assign)]

    def elite_add(assign, obj):
        k = obj[0]
        pool = elite.get(k)
        if pool is None:
            pool = []
            elite[k] = pool
        pool.append((obj, assign[:]))
        pool.sort(key=lambda x: x[0])
        if len(pool) > ELITE:
            del pool[ELITE:]

    def elite_pick(k: int):
        pool = elite.get(k)
        if not pool:
            return None
        m = min(4, len(pool))
        return random.choice(pool[:m])[1][:]

    # Reactive tenure
    def tenure(iteration: int, k: int, stagn: int):
        base = 7 + int(math.log2(n + 1)) + int(math.log2(k + 2))
        react = min(30, stagn // 600)
        return base + react + random.randrange(8)

    # Candidate bins to try empty: lightest few
    def light_bins(m=3):
        k = len(cur_loads)
        if k <= m:
            return list(range(k))
        # selection without full sort
        best = []
        for b in range(k):
            L = cur_loads[b]
            ins = False
            for j, bb in enumerate(best):
                if L < cur_loads[bb]:
                    best.insert(j, b)
                    ins = True
                    break
            if not ins:
                best.append(b)
            if len(best) > m:
                best.pop()
        return best

    # --- neighborhood: empty a bin (LNS within TS) ---
    def try_empty_bin(b: int, iteration: int, depth: int, dest_limit: int):
        # bounded ejection chains; returns (neigh_obj, seq_moves) or None
        if b >= len(cur_loads) or not cur_bin_items[b]:
            return None

        items = cur_bin_items[b]
        # If too many items, pick a subset first; but attempt full empty when small.
        if cur_top_dirty[b]:
            rebuild_top(b)
        focus = cur_top[b][:]
        # add a few random items too
        if len(focus) < min(14, len(items)):
            for _ in range(min(14, len(items)) - len(focus)):
                focus.append(random.choice(items))
        # unique
        seen = set()
        focus2 = []
        for x in focus:
            if x not in seen:
                seen.add(x)
                focus2.append(x)
        # Prefer larger first
        focus2.sort(key=lambda i: weights[i], reverse=True)

        # If bin has few items, try to move all
        if len(items) <= 12:
            focus2 = sorted(items, key=lambda i: weights[i], reverse=True)
        else:
            focus2 = focus2[:12]

        # simulate on copies (k moderate)
        k = len(cur_loads)
        sim_loads = cur_loads[:]
        sim_res = cur_res_sq
        seq = []

        # local function to simulate a relocation
        def sim_move(it, fr, to):
            nonlocal sim_res
            w = weights[it]
            sim_res += delta_sq(sim_loads[fr], sim_loads[fr] - w)
            sim_res += delta_sq(sim_loads[to], sim_loads[to] + w)
            sim_loads[fr] -= w
            sim_loads[to] += w
            seq.append((it, fr, to))

        # Track which items actually moved out of b in this attempt
        moved_out = set()

        for it0 in focus2:
            if (iteration & 255) == 0 and time_exceeded():
                return None
            if it0 in moved_out:
                continue
            w0 = weights[it0]

            # direct
            dests = propose_bins_for_item(w0, exclude=b, limit=dest_limit)
            placed = False
            for db in dests:
                # avoid taboo strongly but not absolutely (aspiration checked at end)
                if is_tabu(it0, db, iteration) and random.random() < 0.9:
                    continue
                if sim_loads[db] + w0 <= C:
                    sim_move(it0, b, db)
                    moved_out.add(it0)
                    placed = True
                    break
            if placed:
                continue

            if depth <= 1:
                continue

            # one ejection: choose a near-feasible bin and eject one heavy item
            near = propose_bins_for_item(max(1, w0 - 1), exclude=b, limit=min(10, dest_limit + 4))
            ejected_ok = False
            for db in near:
                slack_db = C - sim_loads[db]
                need = w0 - slack_db
                if need <= 0:
                    continue
                if not cur_bin_items[db]:
                    continue
                # pick a candidate eject item from db (heuristic: heaviest items)
                # We use current top cache (approx OK)
                if cur_top_dirty[db]:
                    rebuild_top(db)
                eject_cands = cur_top[db][:min(8, len(cur_top[db]))]
                random.shuffle(eject_cands)
                for x in eject_cands:
                    wx = weights[x]
                    if wx < need:
                        continue
                    # find destination for x
                    dests_x = propose_bins_for_item(wx, exclude=db, limit=8)
                    for dx in dests_x:
                        if dx == b:
                            continue
                        if sim_loads[dx] + wx > C:
                            continue
                        if is_tabu(x, dx, iteration) and random.random() < 0.9:
                            continue
                        # simulate x: db->dx
                        sim_move(x, db, dx)
                        # now place it0 into db
                        if sim_loads[db] + w0 <= C:
                            sim_move(it0, b, db)
                            moved_out.add(it0)
                            ejected_ok = True
                        else:
                            # rollback is complex; abort this ejection branch by resetting simulation fully
                            return None
                        break
                    if ejected_ok:
                        break
                if ejected_ok:
                    break
            if not ejected_ok:
                continue

        # accept only if bin is emptied in simulation
        if sim_loads[b] != 0:
            return None

        # compute neighbor objective
        # remove empty bin's slack^2 term
        sim_res -= C * C
        # compute min load among remaining bins
        minL = None
        for bb, L in enumerate(sim_loads):
            if bb == b:
                continue
            if minL is None or L < minL:
                minL = L
        if minL is None:
            minL = 0
        neigh_sec = sim_res - gamma * (C - minL)
        neigh_obj = (k - 1, neigh_sec)
        return neigh_obj, seq

    # --- neighborhood: best single relocate & swap ---
    heavy_items = order_desc

    def pick_items(count: int):
        if n <= count:
            return list(range(n))
        cand = set()
        # bias: items from light bins
        lbs = light_bins(3)
        for b in lbs:
            if not cur_bin_items[b]:
                continue
            if cur_top_dirty[b]:
                rebuild_top(b)
            for it in cur_top[b]:
                cand.add(it)
                if len(cand) >= int(count * 0.65):
                    break
            while len(cand) < int(count * 0.65) and cur_bin_items[b]:
                cand.add(random.choice(cur_bin_items[b]))
        # add heavy globals
        for it in heavy_items[: max(10, count // 3)]:
            cand.add(it)
            if len(cand) >= count:
                break
        while len(cand) < count:
            cand.add(random.randrange(n))
        return list(cand)

    def best_simple_neighbor(cand_items, iteration: int, move_evals: int, swap_trials: int):
        best = None
        k = len(cur_loads)
        evals = 0
        for it in cand_items:
            if evals >= move_evals:
                break
            if (evals & 127) == 0 and time_exceeded():
                break
            bi = cur_assign[it]
            w = weights[it]
            dests = propose_bins_for_item(w, exclude=bi, limit=12)
            for bj in dests:
                if bj == bi:
                    continue
                empties = (len(cur_bin_items[bi]) == 1)
                newk = k - 1 if empties else k
                # compute res
                res = cur_res_sq
                newLbi = cur_loads[bi] - w
                newLbj = cur_loads[bj] + w
                res += delta_sq(cur_loads[bi], newLbi)
                res += delta_sq(cur_loads[bj], newLbj)
                if empties:
                    res -= C * C

                # compute min load conservatively by local check; occasionally exact scan
                if empties:
                    # source bin removed; min could increase
                    if k <= 2:
                        minL = newLbj
                    else:
                        minL = None
                        for b in range(k):
                            if b == bi:
                                continue
                            Lb = newLbj if b == bj else cur_loads[b]
                            if minL is None or Lb < minL:
                                minL = Lb
                else:
                    minL = None
                    for b in range(k):
                        if b == bi:
                            Lb = newLbi
                        elif b == bj:
                            Lb = newLbj
                        else:
                            Lb = cur_loads[b]
                        if minL is None or Lb < minL:
                            minL = Lb
                sec = res - gamma * (C - (minL if minL is not None else 0))
                neigh = (newk, sec)

                tabu_viol = is_tabu(it, bj, iteration)
                # aspiration: always allow improving best k, or improving best obj
                if tabu_viol and not (neigh < best_obj):
                    continue

                if best is None or neigh < best[0]:
                    best = (neigh, ("move", it, bi, bj))
            evals += 1

        # swaps
        for t in range(swap_trials):
            if (t & 63) == 0 and time_exceeded():
                break
            if k <= 1:
                break
            i = random.choice(cand_items)
            bi = cur_assign[i]
            bj = random.randrange(k)
            if bj == bi or not cur_bin_items[bj]:
                continue
            j = random.choice(cur_bin_items[bj])
            if i == j:
                continue
            wi, wj = weights[i], weights[j]
            Li, Lj = cur_loads[bi], cur_loads[bj]
            newLi = Li - wi + wj
            newLj = Lj - wj + wi
            if newLi > C or newLj > C:
                continue

            res = cur_res_sq
            res += delta_sq(Li, newLi)
            res += delta_sq(Lj, newLj)

            # min
            minL = None
            for b in range(k):
                if b == bi:
                    Lb = newLi
                elif b == bj:
                    Lb = newLj
                else:
                    Lb = cur_loads[b]
                if minL is None or Lb < minL:
                    minL = Lb
            sec = res - gamma * (C - (minL if minL is not None else 0))
            neigh = (k, sec)

            tabu_viol = is_tabu(i, bj, iteration) or is_tabu(j, bi, iteration)
            if tabu_viol and not (neigh < best_obj):
                continue

            if best is None or neigh < best[0]:
                best = (neigh, ("swap", i, bi, j, bj))

        return best

    # --- apply moves ---
    def compact_state():
        nonlocal cur_assign, cur_loads, cur_bin_items, cur_bin_pos
        pack, loads, new_assign = build_packing_from_assign(cur_assign)
        cur_assign = new_assign
        cur_bin_items = pack
        cur_loads = loads
        cur_bin_pos = rebuild_positions(cur_bin_items)

    def apply(move, iteration: int, ten: int):
        nonlocal cur_res_sq
        kind = move[0]
        if kind == "move":
            _, it, bi, bj = move
            w = weights[it]
            cur_res_sq += delta_sq(cur_loads[bi], cur_loads[bi] - w)
            cur_res_sq += delta_sq(cur_loads[bj], cur_loads[bj] + w)

            bin_remove(cur_bin_items, cur_bin_pos, bi, it)
            cur_loads[bi] -= w
            bin_add(cur_bin_items, cur_bin_pos, bj, it)
            cur_loads[bj] += w
            cur_assign[it] = bj

            cur_top_dirty[bi] = True
            cur_top_dirty[bj] = True
            update_bucket_for_bin(bi)
            update_bucket_for_bin(bj)
            tabu[(it, bi)] = iteration + ten

        elif kind == "swap":
            _, i, bi, j, bj = move
            wi, wj = weights[i], weights[j]
            Li, Lj = cur_loads[bi], cur_loads[bj]
            newLi = Li - wi + wj
            newLj = Lj - wj + wi
            cur_res_sq += delta_sq(Li, newLi)
            cur_res_sq += delta_sq(Lj, newLj)

            bin_remove(cur_bin_items, cur_bin_pos, bi, i)
            bin_remove(cur_bin_items, cur_bin_pos, bj, j)
            bin_add(cur_bin_items, cur_bin_pos, bi, j)
            bin_add(cur_bin_items, cur_bin_pos, bj, i)
            cur_loads[bi] = newLi
            cur_loads[bj] = newLj
            cur_assign[i] = bj
            cur_assign[j] = bi

            cur_top_dirty[bi] = True
            cur_top_dirty[bj] = True
            update_bucket_for_bin(bi)
            update_bucket_for_bin(bj)
            tabu[(i, bi)] = iteration + ten
            tabu[(j, bj)] = iteration + ten

        elif kind == "sequence":
            _, seq = move
            for (it, fr, to) in seq:
                w = weights[it]
                cur_res_sq += delta_sq(cur_loads[fr], cur_loads[fr] - w)
                cur_res_sq += delta_sq(cur_loads[to], cur_loads[to] + w)
                bin_remove(cur_bin_items, cur_bin_pos, fr, it)
                cur_loads[fr] -= w
                bin_add(cur_bin_items, cur_bin_pos, to, it)
                cur_loads[to] += w
                cur_assign[it] = to

                cur_top_dirty[fr] = True
                cur_top_dirty[to] = True
                update_bucket_for_bin(fr)
                update_bucket_for_bin(to)
                tabu[(it, fr)] = iteration + ten

        # if any empty bins, compact and rebuild globals
        if any(len(lst) == 0 for lst in cur_bin_items):
            # adjust res_sq by recompute (safe)
            compact_state()
            rebuild_globals()
            # exact recompute
            cur_res_sq = 0
            for L in cur_loads:
                r = C - L
                cur_res_sq += r * r

    def rebuild_globals():
        nonlocal cur_slack, cur_bof, cur_buckets, cur_top, cur_top_dirty
        cur_slack, cur_bof, cur_buckets = rebuild_buckets(cur_loads)
        k = len(cur_loads)
        cur_top = [[] for _ in range(k)]
        cur_top_dirty = [True] * k

    # --- initialize ---
    cur_assign, cur_loads, cur_bin_items = initial_solution()
    # Ensure bin_items are consistent list-of-lists
    # (tighten may have returned packing)
    if len(cur_bin_items) != len(cur_loads):
        pack, loads, new_assign = build_packing_from_assign(cur_assign)
        cur_assign = new_assign
        cur_bin_items = pack
        cur_loads = loads

    cur_bin_pos = rebuild_positions(cur_bin_items)

    cur_res_sq = 0
    for L in cur_loads:
        r = C - L
        cur_res_sq += r * r

    rebuild_globals()

    cur_obj = (len(cur_loads), cur_res_sq - gamma * (C - min(cur_loads) if cur_loads else 0))
    best_obj = cur_obj
    best_assign = cur_assign[:]
    elite_add(best_assign, best_obj)

    # --- main loop ---
    max_iters = 25_000_000  # fixed iteration cap; time-limited by checks
    stagn = 0

    # Effort parameters
    # More aggressive than previous; still bounded.
    for it in range(1, max_iters + 1):
        if (it & 255) == 0 and time_exceeded():
            break

        k = len(cur_loads)
        # If we hit the lower bound, stop early (cannot do better)
        if k == lb:
            best_obj = (k, best_obj[1])
            best_assign = cur_assign[:]
            break

        close = (k <= best_obj[0] + 0)  # at best-k

        # intensify/diversify phases
        phase = (it // 1200) % 4
        # 0,1 intensify; 2 mixed; 3 diversify

        if phase == 3 and stagn > 500:
            # jump to elite of same k sometimes
            if random.random() < 0.55:
                pick = elite_pick(k)
                if pick is not None:
                    cur_assign = pick
                    pack, loads, new_assign = build_packing_from_assign(cur_assign)
                    cur_assign = new_assign
                    cur_bin_items = pack
                    cur_loads = loads
                    cur_bin_pos = rebuild_positions(cur_bin_items)
                    cur_res_sq = 0
                    for L in cur_loads:
                        r = C - L
                        cur_res_sq += r * r
                    rebuild_globals()
                    # light tabu reset
                    if tabu:
                        for kk in list(tabu.keys())[: len(tabu) // 2]:
                            tabu.pop(kk, None)

        # effort scaling
        if phase in (0, 1):
            cand_count = 150 if close else 120
            move_evals = 90 if close else 70
            swap_trials = 140 if close else 100
            dest_limit = 10 if close else 8
            depth = 3 if (close and stagn > 1200) else 2
        elif phase == 2:
            cand_count = 110
            move_evals = 70
            swap_trials = 90
            dest_limit = 8
            depth = 2
        else:
            cand_count = 90
            move_evals = 55
            swap_trials = 70
            dest_limit = 7
            depth = 2

        best_neigh = None

        # 1) try emptying light bins
        # Do this frequently; it is the main mechanism to reduce k.
        lbs = light_bins(3 if phase != 3 else 2)
        for b in lbs:
            if time_exceeded():
                break
            res = try_empty_bin(b, it, depth=depth, dest_limit=dest_limit)
            if res is None:
                continue
            neigh_obj, seq = res
            # tabu check for whole sequence; aspiration if improves best
            tabu_viol = False
            for (itm, _fr, to) in seq:
                if is_tabu(itm, to, it):
                    tabu_viol = True
                    break
            if tabu_viol and not (neigh_obj < best_obj):
                continue
            best_neigh = (neigh_obj, ("sequence", seq))
            # if reduces bins, take immediately
            if neigh_obj[0] < k:
                break

        # 2) otherwise use simple best neighbor
        if best_neigh is None:
            cand_items = pick_items(cand_count)
            best_neigh = best_simple_neighbor(cand_items, it, move_evals, swap_trials)

        if best_neigh is None:
            stagn += 1
            continue

        neigh_obj, mv = best_neigh
        ten = tenure(it, len(cur_loads), stagn)
        apply(mv, it, ten)

        # update objective exactly from maintained res_sq and min load
        if cur_loads:
            minL = min(cur_loads)
            cur_obj = (len(cur_loads), cur_res_sq - gamma * (C - minL))
        else:
            cur_obj = (0, 0)

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_assign = cur_assign[:]
            elite_add(best_assign, best_obj)
            stagn = 0
        else:
            stagn += 1
            if (stagn % 500) == 0:
                elite_add(cur_assign, cur_obj)

        # mild reactive tabu cleanup
        if (it & 8191) == 0 and tabu:
            # remove expired entries (cheap periodic scan)
            # limit work by sampling keys if large
            if len(tabu) <= 4000:
                for kk, exp in list(tabu.items()):
                    if exp < it:
                        tabu.pop(kk, None)
            else:
                keys = list(tabu.keys())
                for kk in random.sample(keys, 1200):
                    exp = tabu.get(kk)
                    if exp is not None and exp < it:
                        tabu.pop(kk, None)

    packing, loads, _ = build_packing_from_assign(best_assign)
    return {"packing": packing, "bin_weights": loads}
