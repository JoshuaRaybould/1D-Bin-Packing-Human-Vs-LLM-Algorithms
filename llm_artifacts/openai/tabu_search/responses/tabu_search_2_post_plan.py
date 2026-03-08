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

    # ---------------- Objective: lexicographic (k, secondary) ----------------
    # secondary = sum(residual^2) - gamma * max_residual
    # max_residual = max(C-L) = C - min_load, encourages having a very light bin
    gamma = max(1, C // 10)  # small-ish vs residual^2 scale

    def objective_from(loads: list[int]) -> tuple[int, int]:
        res_sq = 0
        min_load = None
        for L in loads:
            r = C - L
            res_sq += r * r
            if min_load is None or L < min_load:
                min_load = L
        if min_load is None:
            min_load = 0
        max_residual = C - min_load
        sec = res_sq - gamma * max_residual
        return (len(loads), sec)

    # For delta evaluation we need res_sq and min_load; we keep incremental fields and refresh periodically.

    # ---------------- Utilities to build packing/assignment ----------------
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

    # ---------------- Fast bin item ops ----------------
    def bin_remove(bin_items: list[list[int]], bin_pos: list[dict[int, int]], b: int, item: int) -> None:
        pos = bin_pos[b].pop(item)
        last = bin_items[b].pop()
        if last != item:
            bin_items[b][pos] = last
            bin_pos[b][last] = pos

    def bin_add(bin_items: list[list[int]], bin_pos: list[dict[int, int]], b: int, item: int) -> None:
        bin_pos[b][item] = len(bin_items[b])
        bin_items[b].append(item)

    # ---------------- Slack buckets (plan §4.1) ----------------
    # bucket index = slack // bucket_size, store bins by slack class for fast candidates.
    bucket_size = max(1, C // 200)  # ~200 buckets
    nbuckets = C // bucket_size + 1

    def slack_of(load: int) -> int:
        return C - load

    def bucket_index(slack: int) -> int:
        if slack <= 0:
            return 0
        idx = slack // bucket_size
        if idx >= nbuckets:
            return nbuckets - 1
        return idx

    # We'll maintain: bucket_bins[idx] = list of bin ids (with lazy cleanup)
    # and current bucket id per bin.

    # ---------------- Per-bin top-items cache (plan §4.2) ----------------
    TOP_CACHE = 8

    def rebuild_top_cache_for_bin(b: int) -> None:
        items = bin_items[b]
        if not items:
            bin_top[b] = []
            bin_top_dirty[b] = False
            return
        # partial: sort by weight desc; TOP_CACHE small
        # Python sort of potentially moderate list is fine; we rebuild lazily.
        items_sorted = sorted(items, key=lambda i: weights[i], reverse=True)
        bin_top[b] = items_sorted[:TOP_CACHE]
        bin_top_dirty[b] = False

    # ---------------- Construction heuristics (plan §2.1) ----------------
    order_desc = list(range(n))
    order_desc.sort(key=lambda i: weights[i], reverse=True)

    def construct_ffd() -> tuple[list[int], list[int], list[list[int]], list[dict[int, int]]]:
        assign = [-1] * n
        loads: list[int] = []
        bin_items: list[list[int]] = []
        bin_pos: list[dict[int, int]] = []
        for i in order_desc:
            w = weights[i]
            placed = False
            for b, L in enumerate(loads):
                if L + w <= C:
                    loads[b] = L + w
                    assign[i] = b
                    p = len(bin_items[b])
                    bin_items[b].append(i)
                    bin_pos[b][i] = p
                    placed = True
                    break
            if not placed:
                b = len(loads)
                loads.append(w)
                assign[i] = b
                bin_items.append([i])
                bin_pos.append({i: 0})
        return assign, loads, bin_items, bin_pos

    def construct_bfd(order: list[int]) -> tuple[list[int], list[int], list[list[int]], list[dict[int, int]]]:
        assign = [-1] * n
        loads: list[int] = []
        bin_items: list[list[int]] = []
        bin_pos: list[dict[int, int]] = []
        for i in order:
            w = weights[i]
            best_bin = -1
            best_rem = None
            for b, L in enumerate(loads):
                if L + w <= C:
                    rem = C - (L + w)
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
            p = len(bin_items[best_bin])
            bin_items[best_bin].append(i)
            bin_pos[best_bin][i] = p
        return assign, loads, bin_items, bin_pos

    def construct_randomized_bfd() -> tuple[list[int], list[int], list[list[int]], list[dict[int, int]]]:
        order = order_desc[:]
        block = 8
        for s in range(0, n, block):
            e = min(n, s + block)
            if e - s > 1:
                random.shuffle(order[s:e])
        return construct_bfd(order)

    def construct_complement_target() -> tuple[list[int], list[int], list[list[int]], list[dict[int, int]]]:
        # Prefer tight fits; but also sometimes leave a "target" residual based on common large weights.
        # Choose target residual as a random pick among top few weights complement.
        topw = [weights[i] for i in order_desc[: min(n, 12)]]
        target_res = 0
        if topw:
            wpick = random.choice(topw)
            target_res = max(0, min(C, wpick))
        assign = [-1] * n
        loads: list[int] = []
        bin_items: list[list[int]] = []
        bin_pos: list[dict[int, int]] = []
        for i in order_desc:
            w = weights[i]
            best_bin = -1
            best_score = None
            for b, L in enumerate(loads):
                if L + w <= C:
                    rem = C - (L + w)
                    # Score prefers rem close to 0 OR close to target_res
                    score = min(rem, abs(rem - target_res))
                    if best_score is None or score < best_score:
                        best_score = score
                        best_bin = b
            if best_bin == -1:
                best_bin = len(loads)
                loads.append(0)
                bin_items.append([])
                bin_pos.append({})
            loads[best_bin] += w
            assign[i] = best_bin
            p = len(bin_items[best_bin])
            bin_items[best_bin].append(i)
            bin_pos[best_bin][i] = p
        return assign, loads, bin_items, bin_pos

    # ---------------- Warm-start: try to eliminate lightest bin (plan §2.2) ----------------
    def greedy_reinsert_items(items: list[int], forbid_bin: int, loads: list[int]) -> list[tuple[int, int]] | None:
        # Try to place each item into some other bin; return list of (item, dest_bin) if success.
        # Use best-fit: minimal residual after placement.
        moves: list[tuple[int, int]] = []
        # process largest first
        items_sorted = sorted(items, key=lambda i: weights[i], reverse=True)
        for it in items_sorted:
            w = weights[it]
            best_b = -1
            best_rem = None
            for b, L in enumerate(loads):
                if b == forbid_bin:
                    continue
                if L + w <= C:
                    rem = C - (L + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_b = b
            if best_b == -1:
                return None
            loads[best_b] += w
            moves.append((it, best_b))
        return moves

    def warm_start_merge(assign, loads, bin_items, bin_pos) -> tuple[list[int], list[int], list[list[int]], list[dict[int, int]]]:
        if len(loads) <= 1:
            return assign, loads, bin_items, bin_pos
        # attempt a couple of times on lightest bins
        for _ in range(2):
            # find lightest bin
            lb = min(range(len(loads)), key=lambda b: loads[b])
            if not bin_items[lb]:
                continue
            tmp_loads = loads[:]
            tmp_loads[lb] = 0
            plan_moves = greedy_reinsert_items(bin_items[lb], lb, tmp_loads)
            if plan_moves is None:
                break
            # apply: move all items out
            for (it, dest) in plan_moves:
                # remove from lb
                bin_remove(bin_items, bin_pos, lb, it)
                # add to dest
                bin_add(bin_items, bin_pos, dest, it)
                assign[it] = dest
            # rebuild loads
            loads[:] = tmp_loads
            # lb is empty
            # compact bins
            return compact_state(assign, loads, bin_items, bin_pos)
        return assign, loads, bin_items, bin_pos

    # ---------------- State compaction & rebuild structures ----------------
    def compact_state(assign, loads, bin_items, bin_pos):
        k = len(loads)
        mapping = [-1] * k
        new_loads = []
        new_items = []
        new_pos = []
        for b in range(k):
            if bin_items[b]:
                mapping[b] = len(new_loads)
                new_loads.append(loads[b])
                new_items.append(bin_items[b])
                new_pos.append(bin_pos[b])
        if len(new_loads) == k:
            return assign, loads, bin_items, bin_pos
        new_assign = [-1] * n
        for old_b, nb in enumerate(mapping):
            if nb != -1:
                for it in bin_items[old_b]:
                    new_assign[it] = nb
        return new_assign, new_loads, new_items, new_pos

    def rebuild_slack_buckets():
        nonlocal slack, bin_bucket, bucket_bins
        k = len(cur_loads)
        slack = [0] * k
        bin_bucket = [0] * k
        bucket_bins = [[] for _ in range(nbuckets)]
        for b in range(k):
            s = C - cur_loads[b]
            slack[b] = s
            idx = bucket_index(s)
            bin_bucket[b] = idx
            bucket_bins[idx].append(b)

    def rebuild_all_caches():
        nonlocal bin_top, bin_top_dirty
        k = len(cur_loads)
        bin_top = [[] for _ in range(k)]
        bin_top_dirty = [True] * k

    # ---------------- Multi-start initial solution selection ----------------
    def initial_solution_multistart() -> tuple[list[int], list[int], list[list[int]], list[dict[int, int]]]:
        # Spend tiny time; fixed small number of constructions.
        candidates = []
        # deterministic
        candidates.append(construct_ffd())
        candidates.append(construct_bfd(order_desc))
        # randomized set
        candidates.append(construct_randomized_bfd())
        candidates.append(construct_randomized_bfd())
        candidates.append(construct_complement_target())
        # pick best by objective
        best = None
        best_obj = None
        for (a, L, it, ps) in candidates:
            obj = objective_from(L)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best = (a, L, it, ps)
            if time_exceeded():
                break
        a, L, it, ps = best
        # warm start merge
        a = a[:]  # ensure not shared
        L = L[:]
        it = [lst[:] for lst in it]
        ps = [d.copy() for d in ps]
        a, L, it, ps = warm_start_merge(a, L, it, ps)
        return a, L, it, ps

    # ---------------- Tabu memory (plan §5) ----------------
    # Tabu attribute: (item, forbidden_bin) meaning item cannot be assigned to that bin until expiry.
    tabu_forbid: dict[tuple[int, int], int] = {}

    # aspiration memory: best secondary for each k (plan §1.2)
    best_sec_for_k: dict[int, int] = {}

    # elite pool for same-k solutions (plan §6.3)
    ELITE_M = 6
    elite: dict[int, list[tuple[tuple[int, int], list[int]]]] = {}  # k -> list of (obj, assign)

    def elite_add(assign: list[int], obj: tuple[int, int]):
        k = obj[0]
        pool = elite.get(k)
        if pool is None:
            pool = []
            elite[k] = pool
        # store compactly as list; keep unique-ish by objective only
        pool.append((obj, assign[:]))
        pool.sort(key=lambda x: x[0])
        if len(pool) > ELITE_M:
            del pool[ELITE_M:]

    def elite_pick(k: int) -> list[int] | None:
        pool = elite.get(k)
        if not pool:
            return None
        # pick among top few
        m = min(len(pool), 3)
        return random.choice(pool[:m])[1][:]

    # ---------------- Incremental evaluation helpers ----------------
    def compute_res_sq(loads: list[int]) -> int:
        s = 0
        for L in loads:
            r = C - L
            s += r * r
        return s

    def delta_res_sq(old_load: int, new_load: int) -> int:
        r0 = C - old_load
        r1 = C - new_load
        return r1 * r1 - r0 * r0

    # ---------------- Destination lookup via slack buckets ----------------
    def propose_destination_bins(w: int, exclude: int, limit: int) -> list[int]:
        # Find bins with slack >= w, prioritize best-fit (small residual after placement).
        need = w
        idx = bucket_index(need)
        candidates: list[int] = []
        # search upward buckets until enough candidates
        for bi in range(idx, nbuckets):
            if not bucket_bins[bi]:
                continue
            # gather a few from this bucket (lazy, may include outdated bins)
            for b in bucket_bins[bi]:
                if b == exclude:
                    continue
                if b >= len(cur_loads):
                    continue
                if (C - cur_loads[b]) >= need:
                    candidates.append(b)
                    if len(candidates) >= limit * 3:
                        break
            if len(candidates) >= limit * 3:
                break
        if not candidates:
            return []
        # choose best-fit among candidates
        candidates.sort(key=lambda b: (C - (cur_loads[b] + w)))
        out = []
        seen = set()
        for b in candidates:
            if b not in seen:
                seen.add(b)
                out.append(b)
                if len(out) >= limit:
                    break
        return out

    def update_bin_bucket(b: int):
        s = C - cur_loads[b]
        slack[b] = s
        new_idx = bucket_index(s)
        # lazy: just append to new bucket list; old occurrences remain and are filtered on lookup
        if new_idx != bin_bucket[b]:
            bin_bucket[b] = new_idx
            bucket_bins[new_idx].append(b)

    # ---------------- Target bins refresh (plan §3.1, §8.1) ----------------
    target_bins: list[int] = []
    last_target_refresh = 0

    def refresh_target_bins():
        nonlocal target_bins, last_target_refresh
        k = len(cur_loads)
        if k == 0:
            target_bins = []
            last_target_refresh = it
            return
        m = min(3, k)
        # partial selection of m lightest bins without full sort
        best = []
        for b in range(k):
            L = cur_loads[b]
            inserted = False
            for idx, bb in enumerate(best):
                if L < cur_loads[bb]:
                    best.insert(idx, b)
                    inserted = True
                    break
            if not inserted:
                best.append(b)
            if len(best) > m:
                best.pop()
        target_bins = best
        last_target_refresh = it

    # ---------------- Candidate items (plan §3.4, §8.2) ----------------
    heavy_items = list(range(n))
    heavy_items.sort(key=lambda i: weights[i], reverse=True)

    def pick_candidate_items(count: int) -> list[int]:
        if n <= count:
            return list(range(n))
        cand = set()
        # 70% from target bins
        from_targets = int(count * 0.7)
        if target_bins:
            titems = []
            for b in target_bins:
                if b < len(cur_loads) and bin_items[b]:
                    if bin_top_dirty[b]:
                        rebuild_top_cache_for_bin(b)
                    titems.extend(bin_top[b])
                    # add some random from bin
                    for _ in range(3):
                        titems.append(random.choice(bin_items[b]))
            random.shuffle(titems)
            for x in titems:
                cand.add(x)
                if len(cand) >= from_targets:
                    break
        # 30% global heavy + random
        rest = count - len(cand)
        for x in heavy_items[: max(5, rest // 2)]:
            cand.add(x)
            if len(cand) >= count:
                break
        while len(cand) < count:
            cand.add(random.randrange(n))
        return list(cand)

    # ---------------- Tabu + aspiration checks (plan §1.2, §5.1) ----------------
    def is_move_tabu(item: int, dest_bin: int, iteration: int) -> bool:
        exp = tabu_forbid.get((item, dest_bin))
        return exp is not None and exp >= iteration

    def admissible(neigh_k: int, neigh_sec: int, iteration: int, tabu_violated: bool) -> bool:
        # aspiration by bins: always allow if fewer bins than best
        if neigh_k < best_obj[0]:
            return True
        if not tabu_violated:
            return True
        # aspiration by per-k improvement
        best_sec = best_sec_for_k.get(neigh_k)
        if best_sec is None:
            return True
        # threshold: require some meaningful improvement
        thresh = max(1, abs(best_sec) // 2000)
        return neigh_sec <= best_sec - thresh

    # ---------------- Adaptive tenure (plan §5.3) ----------------
    def current_tenure(k: int, stagn: int) -> int:
        base = 5 + int(math.log2(n + 1)) + int(math.log2(k + 2))
        addon = min(20, stagn // 500)
        jitter = 5 + (n % 7)
        return base + addon + random.randrange(jitter)

    # ---------------- Neighborhood operators ----------------
    def try_pack_out(target_b: int, iteration: int, depth: int, dest_limit: int):
        # Attempt to empty target bin using bounded ejection depth (2-3).
        # Returns (delta_obj, move_sequence) or None.
        if target_b >= len(cur_loads) or not bin_items[target_b]:
            return None
        items_in = bin_items[target_b]
        # take up to m largest items from target bin
        if bin_top_dirty[target_b]:
            rebuild_top_cache_for_bin(target_b)
        candidates = bin_top[target_b][:]
        if len(candidates) < min(6, len(items_in)):
            # add a few more random
            for _ in range(min(6, len(items_in)) - len(candidates)):
                candidates.append(random.choice(items_in))
        # unique
        seen = set()
        items_try = []
        for x in candidates:
            if x not in seen:
                seen.add(x)
                items_try.append(x)
        items_try.sort(key=lambda i: weights[i], reverse=True)
        m = min(len(items_try), 6)
        items_try = items_try[:m]

        # local copies of loads for simulation
        k = len(cur_loads)
        base_res_sq = cur_res_sq

        # try a few orderings
        for _attempt in range(3):
            if time_exceeded():
                return None
            if _attempt > 0:
                random.shuffle(items_try)
            sim_loads = cur_loads[:]  # small k
            sim_res_sq = base_res_sq
            moves = []  # list of (item, from_bin, to_bin)
            ok = True

            # We simulate moves; for ejection we allow moving one blocker item from a chosen bin.
            for it_item in items_try:
                w = weights[it_item]
                fromb = target_b
                # try direct placement
                dests = propose_destination_bins(w, exclude=fromb, limit=dest_limit)
                placed = False
                for db in dests:
                    tabu_viol = is_move_tabu(it_item, db, iteration)
                    # We don't apply aspiration in the middle; only final. Still, avoid hard tabu usually.
                    if tabu_viol and random.random() < 0.85:
                        continue
                    # simulate
                    old_from = sim_loads[fromb]
                    old_to = sim_loads[db]
                    sim_loads[fromb] = old_from - w
                    sim_loads[db] = old_to + w
                    sim_res_sq += delta_res_sq(old_from, sim_loads[fromb])
                    sim_res_sq += delta_res_sq(old_to, sim_loads[db])
                    moves.append((it_item, fromb, db))
                    placed = True
                    break

                if placed:
                    continue

                if depth <= 1:
                    ok = False
                    break

                # one ejection: pick a destination bin db, eject one item x from db to elsewhere to make room.
                # Choose db among bins with some slack but not enough; search a few candidates.
                # We approximate by scanning a few bins from near-needed buckets.
                near = propose_destination_bins(max(1, w - 1), exclude=fromb, limit=8)
                ejected = False
                for db in near:
                    # need create extra slack = w - slack(db)
                    need_extra = w - (C - sim_loads[db])
                    if need_extra <= 0:
                        continue
                    if not bin_items[db]:
                        continue
                    # choose an item x in db with weight >= need_extra to move out
                    # use cache if available
                    # cache based on current real state; but okay as heuristic: choose among real top items
                    if bin_top_dirty[db]:
                        rebuild_top_cache_for_bin(db)
                    cand_x = bin_top[db]
                    if not cand_x:
                        continue
                    for x in cand_x:
                        wx = weights[x]
                        if wx < need_extra:
                            continue
                        # find place for x elsewhere
                        dests_x = propose_destination_bins(wx, exclude=db, limit=6)
                        if not dests_x:
                            continue
                        placed_x = False
                        for dx in dests_x:
                            if dx == fromb:
                                continue
                            # simulate moving x: db->dx
                            old_db = sim_loads[db]
                            old_dx = sim_loads[dx]
                            if old_dx + wx > C:
                                continue
                            sim_loads[db] = old_db - wx
                            sim_loads[dx] = old_dx + wx
                            sim_res_sq += delta_res_sq(old_db, sim_loads[db])
                            sim_res_sq += delta_res_sq(old_dx, sim_loads[dx])
                            moves.append((x, db, dx))
                            placed_x = True
                            break
                        if not placed_x:
                            continue
                        # now place original item into db
                        if sim_loads[db] + w <= C:
                            old_from = sim_loads[fromb]
                            old_db2 = sim_loads[db]
                            sim_loads[fromb] = old_from - w
                            sim_loads[db] = old_db2 + w
                            sim_res_sq += delta_res_sq(old_from, sim_loads[fromb])
                            sim_res_sq += delta_res_sq(old_db2, sim_loads[db])
                            moves.append((it_item, fromb, db))
                            ejected = True
                            break
                        else:
                            # rollback crude: abandon this attempt
                            ok = False
                            break
                    if ejected or not ok:
                        break

                if not ok or not ejected:
                    ok = False
                    break

            if not ok:
                continue

            # Determine resulting k if target bin emptied in simulation
            # In simulation, we only moved subset; but target may not be empty.
            # We treat as candidate if all items in target moved (i.e., items_try equals all items),
            # OR if after moving chosen items, target load becomes 0 (rare unless it had few items).
            new_k = k
            emptied = False
            if sim_loads[target_b] == 0:
                emptied = True
                new_k = k - 1
                sim_res_sq -= (C * C)  # remove empty bin residual^2 term (slack=C)

            # compute secondary including max_residual via min_load
            min_load = min(L for L in sim_loads if L > 0) if emptied else min(sim_loads)
            max_resid = C - min_load
            neigh_sec = sim_res_sq - gamma * max_resid
            return (new_k, neigh_sec), moves, emptied

        return None

    def try_2_for_1_exchange(target_b: int, iteration: int, trials: int):
        if target_b >= len(cur_loads) or not bin_items[target_b]:
            return None
        if bin_top_dirty[target_b]:
            rebuild_top_cache_for_bin(target_b)
        A = target_b
        itemsA = bin_top[A]
        if len(itemsA) < 2:
            return None
        # pick candidate pairs among first few
        pairs = []
        lim = min(6, len(itemsA))
        for i in range(lim):
            for j in range(i + 1, lim):
                pairs.append((itemsA[i], itemsA[j]))
        if not pairs:
            return None
        random.shuffle(pairs)

        best = None
        k = len(cur_loads)
        for t in range(trials):
            if (t & 15) == 0 and time_exceeded():
                return best
            x, y = pairs[t % len(pairs)]
            wx, wy = weights[x], weights[y]
            sumw = wx + wy
            # choose destination bin B that can take x,y after removing one item z from it into A (or elsewhere)
            # pick B by slack close to sumw
            destsB = propose_destination_bins(sumw, exclude=A, limit=8)
            if not destsB:
                continue
            for B in destsB:
                if not bin_items[B]:
                    continue
                # pick z from B (prefer heavy)
                if bin_top_dirty[B]:
                    rebuild_top_cache_for_bin(B)
                for z in bin_top[B][:min(5, len(bin_top[B]))]:
                    wz = weights[z]
                    # after exchange: A loses x,y gains z; B loses z gains x,y
                    newLA = cur_loads[A] - sumw + wz
                    newLB = cur_loads[B] - wz + sumw
                    if newLA < 0 or newLB < 0:
                        continue
                    if newLA > C or newLB > C:
                        continue
                    # tabu check for assignments: x->B, y->B, z->A
                    tabu_viol = is_move_tabu(x, B, iteration) or is_move_tabu(y, B, iteration) or is_move_tabu(z, A, iteration)

                    neigh_res_sq = cur_res_sq
                    neigh_res_sq += delta_res_sq(cur_loads[A], newLA)
                    neigh_res_sq += delta_res_sq(cur_loads[B], newLB)

                    # k unchanged (exchange)
                    sim_loads_min = min(cur_min_load, newLA, newLB)
                    # but min could change if A was min and increases; do conservative recompute of min via occasional refresh elsewhere
                    max_resid = C - sim_loads_min
                    neigh_sec = neigh_res_sq - gamma * max_resid
                    neigh = (k, neigh_sec)

                    if not admissible(neigh[0], neigh[1], iteration, tabu_viol):
                        continue
                    if best is None or neigh < best[0]:
                        best = (neigh, ("2-1", A, x, y, B, z))
                # end z
        return best

    def try_best_simple_moves(cand_items: list[int], iteration: int, move_evals: int, swap_trials: int):
        best = None
        k = len(cur_loads)
        # 1-move biased by slack buckets
        evals = 0
        for i in cand_items:
            if evals >= move_evals:
                break
            if (evals & 63) == 0 and time_exceeded():
                break
            bi = cur_assign[i]
            w = weights[i]
            dests = propose_destination_bins(w, exclude=bi, limit=10)
            if not dests:
                continue
            for bj in dests:
                if bj == bi:
                    continue
                # feasibility already ensured by slack
                new_k = k
                # if move empties source bin
                empties = (len(bin_items[bi]) == 1)
                if empties:
                    new_k = k - 1
                tabu_viol = is_move_tabu(i, bj, iteration)

                neigh_res_sq = cur_res_sq
                newLbi = cur_loads[bi] - w
                newLbj = cur_loads[bj] + w
                neigh_res_sq += delta_res_sq(cur_loads[bi], newLbi)
                neigh_res_sq += delta_res_sq(cur_loads[bj], newLbj)
                if empties:
                    neigh_res_sq -= (C * C)

                # approximate min-load update; if bin eliminated or if bi was min, might change.
                # We'll allow occasional refresh; for evaluation use conservative min among known and modified.
                if empties:
                    # min load among remaining could increase; safe to use min(cur_min_load_except, ...)
                    # quick approximation: min of (cur_min_load if cur_min_bin not emptied else min(new loads))
                    approx_min = cur_min_load
                    if bi == cur_min_bin:
                        approx_min = min(newLbj, *(cur_loads[b] for b in range(k) if b not in (bi, bj) and bin_items[b])) if k > 2 else newLbj
                    else:
                        approx_min = min(cur_min_load, newLbj)
                    max_resid = C - approx_min
                else:
                    approx_min = cur_min_load
                    if bi == cur_min_bin:
                        approx_min = min(newLbi, newLbj, *(cur_loads[b] for b in range(k) if b not in (bi, bj))) if k > 2 else min(newLbi, newLbj)
                    else:
                        approx_min = min(cur_min_load, newLbi, newLbj)
                    max_resid = C - approx_min

                neigh_sec = neigh_res_sq - gamma * max_resid
                neigh = (new_k, neigh_sec)

                if not admissible(neigh[0], neigh[1], iteration, tabu_viol):
                    continue
                if best is None or neigh < best[0]:
                    best = (neigh, ("move", i, bi, bj))
            evals += 1

        # swaps (biased)
        for t in range(swap_trials):
            if (t & 31) == 0 and time_exceeded():
                break
            if k <= 1:
                break
            i = random.choice(cand_items)
            bi = cur_assign[i]
            # choose bj preferably a non-target to spread
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
            if Li - wi + wj > C or Lj - wj + wi > C:
                continue

            tabu_viol = is_move_tabu(i, bj, iteration) or is_move_tabu(j, bi, iteration)

            neigh_res_sq = cur_res_sq
            neigh_res_sq += delta_res_sq(Li, Li - wi + wj)
            neigh_res_sq += delta_res_sq(Lj, Lj - wj + wi)
            approx_min = cur_min_load
            if bi == cur_min_bin or bj == cur_min_bin:
                # fallback conservative
                approx_min = min(Li - wi + wj, Lj - wj + wi, cur_min_load)
            else:
                approx_min = min(cur_min_load, Li - wi + wj, Lj - wj + wi)
            neigh_sec = neigh_res_sq - gamma * (C - approx_min)
            neigh = (k, neigh_sec)

            if not admissible(neigh[0], neigh[1], iteration, tabu_viol):
                continue
            if best is None or neigh < best[0]:
                best = (neigh, ("swap", i, bi, j, bj))

        return best

    # ---------------- Apply moves atomically ----------------
    def apply_move(move, iteration: int, tenure: int):
        nonlocal cur_res_sq, cur_min_bin, cur_min_load
        kind = move[0]
        if kind == "move":
            _, i, bi, bj = move
            wi = weights[i]
            # update res_sq
            cur_res_sq += delta_res_sq(cur_loads[bi], cur_loads[bi] - wi)
            cur_res_sq += delta_res_sq(cur_loads[bj], cur_loads[bj] + wi)

            bin_remove(bin_items, bin_pos, bi, i)
            cur_loads[bi] -= wi
            bin_add(bin_items, bin_pos, bj, i)
            cur_loads[bj] += wi
            cur_assign[i] = bj

            bin_top_dirty[bi] = True
            bin_top_dirty[bj] = True
            update_bin_bucket(bi)
            update_bin_bucket(bj)

            # tabu reverse: forbid returning to from-bin
            tabu_forbid[(i, bi)] = iteration + tenure

        elif kind == "swap":
            _, i, bi, j, bj = move
            wi, wj = weights[i], weights[j]
            Li, Lj = cur_loads[bi], cur_loads[bj]
            cur_res_sq += delta_res_sq(Li, Li - wi + wj)
            cur_res_sq += delta_res_sq(Lj, Lj - wj + wi)

            bin_remove(bin_items, bin_pos, bi, i)
            bin_remove(bin_items, bin_pos, bj, j)
            bin_add(bin_items, bin_pos, bi, j)
            bin_add(bin_items, bin_pos, bj, i)
            cur_loads[bi] = Li - wi + wj
            cur_loads[bj] = Lj - wj + wi
            cur_assign[i] = bj
            cur_assign[j] = bi

            bin_top_dirty[bi] = True
            bin_top_dirty[bj] = True
            update_bin_bucket(bi)
            update_bin_bucket(bj)

            tabu_forbid[(i, bi)] = iteration + tenure
            tabu_forbid[(j, bj)] = iteration + tenure

        elif kind == "2-1":
            _, A, x, y, B, z = move
            wx, wy, wz = weights[x], weights[y], weights[z]
            sumw = wx + wy
            LA, LB = cur_loads[A], cur_loads[B]
            newLA = LA - sumw + wz
            newLB = LB - wz + sumw
            cur_res_sq += delta_res_sq(LA, newLA)
            cur_res_sq += delta_res_sq(LB, newLB)

            # remove x,y from A, z from B
            bin_remove(bin_items, bin_pos, A, x)
            bin_remove(bin_items, bin_pos, A, y)
            bin_remove(bin_items, bin_pos, B, z)
            # add x,y to B, z to A
            bin_add(bin_items, bin_pos, B, x)
            bin_add(bin_items, bin_pos, B, y)
            bin_add(bin_items, bin_pos, A, z)

            cur_loads[A] = newLA
            cur_loads[B] = newLB
            cur_assign[x] = B
            cur_assign[y] = B
            cur_assign[z] = A

            bin_top_dirty[A] = True
            bin_top_dirty[B] = True
            update_bin_bucket(A)
            update_bin_bucket(B)

            tabu_forbid[(x, A)] = iteration + tenure
            tabu_forbid[(y, A)] = iteration + tenure
            tabu_forbid[(z, B)] = iteration + tenure

        elif kind == "sequence":
            # move is ("sequence", list_of_triples)
            _, seq = move
            # apply each relocation in order
            for (it_item, fromb, tob) in seq:
                w = weights[it_item]
                cur_res_sq += delta_res_sq(cur_loads[fromb], cur_loads[fromb] - w)
                cur_res_sq += delta_res_sq(cur_loads[tob], cur_loads[tob] + w)

                bin_remove(bin_items, bin_pos, fromb, it_item)
                cur_loads[fromb] -= w
                bin_add(bin_items, bin_pos, tob, it_item)
                cur_loads[tob] += w
                cur_assign[it_item] = tob

                bin_top_dirty[fromb] = True
                bin_top_dirty[tob] = True
                update_bin_bucket(fromb)
                update_bin_bucket(tob)

                tabu_forbid[(it_item, fromb)] = iteration + tenure

        # remove empty bins if any
        if any(len(bin_items[b]) == 0 for b in range(len(bin_items))):
            # compact and rebuild slack buckets and caches
            nonlocal cur_assign, cur_loads, bin_items, bin_pos
            cur_assign, cur_loads, bin_items, bin_pos = compact_state(cur_assign, cur_loads, bin_items, bin_pos)
            rebuild_slack_buckets()
            rebuild_all_caches()
            # recompute res_sq exactly after compaction (safe)
            nonlocal cur_obj
            cur_res_sq = compute_res_sq(cur_loads)

        # refresh min-load occasionally; cheap full scan on apply
        if cur_loads:
            cur_min_bin = min(range(len(cur_loads)), key=lambda b: cur_loads[b])
            cur_min_load = cur_loads[cur_min_bin]
        else:
            cur_min_bin = -1
            cur_min_load = 0

    # ---------------- Diversification perturbation (plan §6.2) ----------------
    def perturb_fixed_k(steps: int, iteration: int):
        if len(cur_loads) <= 1:
            return
        k = len(cur_loads)
        for s in range(steps):
            if (s & 15) == 0 and time_exceeded():
                break
            # random swap with feasibility, or random move if feasible
            if random.random() < 0.6:
                # swap
                b1 = random.randrange(k)
                b2 = random.randrange(k)
                if b1 == b2 or not bin_items[b1] or not bin_items[b2]:
                    continue
                i = random.choice(bin_items[b1])
                j = random.choice(bin_items[b2])
                wi, wj = weights[i], weights[j]
                if cur_loads[b1] - wi + wj <= C and cur_loads[b2] - wj + wi <= C:
                    apply_move(("swap", i, b1, j, b2), iteration, tenure=1)
            else:
                # move
                b1 = random.randrange(k)
                if not bin_items[b1]:
                    continue
                i = random.choice(bin_items[b1])
                w = weights[i]
                dests = propose_destination_bins(w, exclude=b1, limit=5)
                if not dests:
                    continue
                b2 = random.choice(dests)
                apply_move(("move", i, b1, b2), iteration, tenure=1)
        # decay tabu partially
        if tabu_forbid:
            # drop some random entries
            keys = list(tabu_forbid.keys())
            random.shuffle(keys)
            for kk in keys[: len(keys) // 3]:
                tabu_forbid.pop(kk, None)

    # ---------------- Initialize ----------------
    cur_assign, cur_loads, bin_items, bin_pos = initial_solution_multistart()
    cur_assign, cur_loads, bin_items, bin_pos = compact_state(cur_assign, cur_loads, bin_items, bin_pos)

    cur_res_sq = compute_res_sq(cur_loads)
    cur_min_bin = min(range(len(cur_loads)), key=lambda b: cur_loads[b]) if cur_loads else -1
    cur_min_load = cur_loads[cur_min_bin] if cur_loads else 0
    cur_obj = objective_from(cur_loads)

    best_assign = cur_assign[:]
    best_obj = cur_obj
    best_sec_for_k[best_obj[0]] = best_obj[1]
    elite_add(best_assign, best_obj)

    rebuild_slack_buckets()
    rebuild_all_caches()

    # ---------------- Main tabu loop with phases (plan §6.1) ----------------
    # Large fixed iteration cap; rely on time checks.
    max_iters = 10_000_000
    phase_len = 800

    stagnation = 0
    soft_perturb_every = 5000
    hard_restart_every = 4 * soft_perturb_every

    it = 0
    while it < max_iters:
        it += 1
        if (it & 255) == 0 and time_exceeded():
            break

        if it - last_target_refresh >= 300 or not target_bins:
            refresh_target_bins()

        k = len(cur_loads)
        # Phase selection
        phase = (it // phase_len) % 3
        # 0: intensify-empty, 1: improve-fill, 2: diversify

        if phase == 2 and (stagnation > 300):
            # diversification: jump to elite sometimes
            if random.random() < 0.5:
                pick = elite_pick(k)
                if pick is not None:
                    cur_assign = pick
                    _, cur_loads, cur_assign = build_packing_from_assign(cur_assign)
                    # rebuild bin structures
                    bin_items = [[] for _ in range(len(cur_loads))]
                    bin_pos = [dict() for _ in range(len(cur_loads))]
                    for i, b in enumerate(cur_assign):
                        bin_pos[b][i] = len(bin_items[b])
                        bin_items[b].append(i)
                    cur_res_sq = compute_res_sq(cur_loads)
                    rebuild_slack_buckets()
                    rebuild_all_caches()
                    cur_min_bin = min(range(len(cur_loads)), key=lambda b: cur_loads[b])
                    cur_min_load = cur_loads[cur_min_bin]
            perturb_fixed_k(steps=10 + (stagnation // 1000), iteration=it)

        # Neighborhood effort scaling (plan §7.2)
        close_to_best = (k == best_obj[0])
        depth = 2
        if close_to_best and stagnation > 1500:
            depth = 3

        if phase == 0:
            cand_count = 120 if close_to_best else 90
            move_evals = 70 if close_to_best else 50
            swap_trials = 110 if close_to_best else 70
            ex_trials = 60 if close_to_best else 35
            dest_limit = 8 if close_to_best else 6
        elif phase == 1:
            cand_count = 90
            move_evals = 60
            swap_trials = 80
            ex_trials = 25
            dest_limit = 6
        else:
            cand_count = 70
            move_evals = 40
            swap_trials = 60
            ex_trials = 20
            dest_limit = 5

        cand_items = pick_candidate_items(cand_count)

        best_neighbor = None

        # A) pack-out attempt for lightest target bin(s)
        if phase == 0 and target_bins:
            for tb in target_bins[:2]:
                if time_exceeded():
                    break
                res = try_pack_out(tb, iteration=it, depth=depth, dest_limit=dest_limit)
                if res is None:
                    continue
                neigh_obj, seq, _emptied = res
                # Evaluate tabu for sequence and aspiration
                tabu_viol = False
                for (itm, _fromb, tob) in seq:
                    if is_move_tabu(itm, tob, it):
                        tabu_viol = True
                        break
                if admissible(neigh_obj[0], neigh_obj[1], it, tabu_viol):
                    best_neighbor = (neigh_obj, ("sequence", seq))
                    # If it eliminates a bin, take immediately
                    if neigh_obj[0] < k:
                        break

        # B) 2-1 exchanges involving target bin
        if best_neighbor is None and target_bins and phase in (0, 1):
            for tb in target_bins[:2]:
                res = try_2_for_1_exchange(tb, it, trials=ex_trials)
                if res is not None:
                    if best_neighbor is None or res[0] < best_neighbor[0]:
                        best_neighbor = res

        # C/D) biased 1-move + swaps
        if best_neighbor is None or (best_neighbor[0][0] == k and phase != 0):
            res = try_best_simple_moves(cand_items, it, move_evals=move_evals, swap_trials=swap_trials)
            if res is not None:
                if best_neighbor is None or res[0] < best_neighbor[0]:
                    best_neighbor = res

        if best_neighbor is None:
            # structured mild perturbation rather than full restart
            perturb_fixed_k(steps=6, iteration=it)
            stagnation += 1
            continue

        neigh_obj, move = best_neighbor
        tenure = current_tenure(len(cur_loads), stagnation)
        apply_move(move, it, tenure)

        # update current objective
        # full refresh of objective using current res_sq and min_load
        if cur_loads:
            max_resid = C - cur_min_load
            cur_obj = (len(cur_loads), cur_res_sq - gamma * max_resid)
        else:
            cur_obj = (0, 0)

        # Update bests + memory
        kcur, scur = cur_obj
        prev_best_sec = best_sec_for_k.get(kcur)
        if prev_best_sec is None or scur < prev_best_sec:
            best_sec_for_k[kcur] = scur

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_assign = cur_assign[:]
            elite_add(best_assign, best_obj)
            stagnation = 0
        else:
            stagnation += 1
            if stagnation % 400 == 0:
                elite_add(cur_assign, cur_obj)

        # soft perturbation
        if stagnation >= soft_perturb_every:
            perturb_fixed_k(steps=12, iteration=it)
            stagnation = soft_perturb_every // 2

        # occasional full reconstruction only if very stuck and time remains
        if stagnation >= hard_restart_every and not time_exceeded():
            cur_assign, cur_loads, bin_items, bin_pos = initial_solution_multistart()
            cur_assign, cur_loads, bin_items, bin_pos = compact_state(cur_assign, cur_loads, bin_items, bin_pos)
            cur_res_sq = compute_res_sq(cur_loads)
            cur_min_bin = min(range(len(cur_loads)), key=lambda b: cur_loads[b]) if cur_loads else -1
            cur_min_load = cur_loads[cur_min_bin] if cur_loads else 0
            cur_obj = objective_from(cur_loads)
            rebuild_slack_buckets()
            rebuild_all_caches()
            tabu_forbid.clear()
            stagnation = 0

    packing, loads, _ = build_packing_from_assign(best_assign)
    return {"packing": packing, "bin_weights": loads}
