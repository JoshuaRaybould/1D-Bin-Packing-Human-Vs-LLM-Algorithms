import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.perf_counter()
    deadline = start + min(max(0.0, float(time_limit)), 100.0)

    # ---- periodic time checks ----
    tick = 0
    TICK_MASK = 2047  # check a bit more often than before

    def time_up_hard() -> bool:
        return time.perf_counter() >= deadline

    def time_up() -> bool:
        nonlocal tick
        tick += 1
        if (tick & TICK_MASK) == 0:
            return time.perf_counter() >= deadline
        return False

    # ---- simple lower bound ----
    total_w = sum(weights)
    lb = (total_w + C - 1) // C

    # ---- ordering and weight frequency ----
    order_dec = sorted(range(n), key=lambda i: weights[i], reverse=True)
    wcount: Dict[int, int] = {}
    for w in weights:
        wcount[w] = wcount.get(w, 0) + 1

    def slack_value(sl: int) -> int:
        # heuristic: slack that matches common items is preferable
        v = 0
        v += 8 * wcount.get(sl, 0)
        v += 4 * wcount.get(sl - 1, 0)
        v += 4 * wcount.get(sl + 1, 0)
        v += 2 * wcount.get(sl - 2, 0)
        v += 2 * wcount.get(sl + 2, 0)
        return v

    # ---- state representation ----
    class State:
        __slots__ = ("bin_items", "bin_w", "active", "loc", "pos", "bins_used")

        def __init__(self):
            self.bin_items: List[List[int]] = []
            self.bin_w: List[int] = []
            self.active: List[bool] = []
            self.loc: List[int] = [-1] * n
            self.pos: List[int] = [-1] * n
            self.bins_used: int = 0

        def clone_compact(self) -> "State":
            ns = State()
            for bid, act in enumerate(self.active):
                if not act:
                    continue
                items = self.bin_items[bid]
                if not items:
                    continue
                nbid = len(ns.bin_items)
                ns.bin_items.append(items[:])
                ns.bin_w.append(self.bin_w[bid])
                ns.active.append(True)
                for pi, it in enumerate(items):
                    ns.loc[it] = nbid
                    ns.pos[it] = pi
            ns.bins_used = len(ns.bin_items)
            return ns

        def ensure_bin(self) -> int:
            bid = len(self.bin_items)
            self.bin_items.append([])
            self.bin_w.append(0)
            self.active.append(False)
            return bid

        def activate_bin_with_item(self, it: int) -> int:
            bid = self.ensure_bin()
            self.active[bid] = True
            self.bin_items[bid] = [it]
            self.bin_w[bid] = weights[it]
            self.loc[it] = bid
            self.pos[it] = 0
            self.bins_used += 1
            return bid

        def add_item(self, it: int, bid: int) -> None:
            self.loc[it] = bid
            self.pos[it] = len(self.bin_items[bid])
            self.bin_items[bid].append(it)
            self.bin_w[bid] += weights[it]

        def remove_item(self, it: int) -> int:
            bid = self.loc[it]
            if bid < 0:
                return -1
            idx = self.pos[it]
            items = self.bin_items[bid]
            last = items[-1]
            items[idx] = last
            self.pos[last] = idx
            items.pop()
            self.loc[it] = -1
            self.pos[it] = -1
            self.bin_w[bid] -= weights[it]
            if not items:
                self.active[bid] = False
                self.bins_used -= 1
            return bid

        def move_item(self, it: int, dst: int) -> None:
            src = self.loc[it]
            if src == dst:
                return
            self.remove_item(it)
            self.add_item(it, dst)

        def active_bins(self) -> List[int]:
            return [b for b, a in enumerate(self.active) if a and self.bin_items[b]]

        def obj(self) -> Tuple[int, int]:
            # primary: bins used
            # secondary: sum of squared slack (tighter is better)
            s2 = 0
            for b in range(len(self.bin_items)):
                if self.active[b] and self.bin_items[b]:
                    sl = C - self.bin_w[b]
                    s2 += sl * sl
            return (self.bins_used, s2)

    # ---- candidate bin selection helpers ----
    def active_bins_cached(st: State) -> List[int]:
        # small speed helper: caller may reuse list
        return st.active_bins()

    def best_fit_bin_for_item(st: State, it: int, bins: List[int], exclude: int = -1) -> int:
        w = weights[it]
        best_b = -1
        best_key = None
        # sample if too many
        if len(bins) > 160:
            bins_eval = random.sample(bins, 160)
        else:
            bins_eval = bins
        for b in bins_eval:
            if b == exclude:
                continue
            sl = C - st.bin_w[b]
            if sl >= w:
                left = sl - w
                key = (left, -slack_value(left), -st.bin_w[b])
                if best_key is None or key < best_key:
                    best_key = key
                    best_b = b
        return best_b

    def candidate_bins_for_item(st: State, it: int, bins: List[int], exclude: int = -1, limit: int = 32) -> List[int]:
        w = weights[it]
        if not bins:
            return []
        # sample large sets
        if len(bins) > 220:
            samp = random.sample(bins, 220)
        else:
            samp = bins
        cand = []
        for b in samp:
            if b == exclude:
                continue
            sl = C - st.bin_w[b]
            if sl >= w:
                left = sl - w
                cand.append((left, -slack_value(left), -st.bin_w[b], b))
        cand.sort()
        return [b for _, __, ___, b in cand[:limit]]

    # ---- construction: multiple variants (FFD/BFD + noise) ----
    def construct(mode: int) -> State:
        st = State()
        # Build item order with controlled noise among similar weights
        items = order_dec[:]
        if mode in (1, 2, 3, 4):
            # shuffle within small blocks to diversify
            block = 8 if mode in (1, 2) else 14
            for i in range(0, n, block):
                j = min(n, i + block)
                seg = items[i:j]
                random.shuffle(seg)
                items[i:j] = seg
        # choose policy
        # mode 0: BFD
        # mode 1: BFD with random tie breaks
        # mode 2: FFD-like (first acceptable in sorted by fill)
        # mode 3: BFD but sample fewer bins (speed)
        # mode 4: biased to create fuller bins (use -bin_w tie)
        for it in items:
            w = weights[it]
            bins = active_bins_cached(st)
            if not bins:
                st.activate_bin_with_item(it)
                continue

            if mode == 2:
                # FFD: iterate bins by decreasing fill
                bins2 = bins[:]
                bins2.sort(key=lambda b: st.bin_w[b], reverse=True)
                placed = False
                # sample to cap cost
                if len(bins2) > 200:
                    bins2 = random.sample(bins2, 200)
                    bins2.sort(key=lambda b: st.bin_w[b], reverse=True)
                for b in bins2:
                    if st.bin_w[b] + w <= C:
                        st.add_item(it, b)
                        placed = True
                        break
                if not placed:
                    st.activate_bin_with_item(it)
            else:
                best_b = -1
                if mode == 3:
                    # quicker best-fit from subset
                    subset = bins if len(bins) <= 120 else random.sample(bins, 120)
                    best_b = best_fit_bin_for_item(st, it, subset)
                else:
                    best_b = best_fit_bin_for_item(st, it, bins)

                if best_b == -1:
                    st.activate_bin_with_item(it)
                else:
                    st.add_item(it, best_b)

                if mode == 1 and best_b != -1 and random.random() < 0.08:
                    # small perturbation: sometimes choose alternative close fit
                    cands = candidate_bins_for_item(st, it, bins, exclude=-1, limit=6)
                    if cands:
                        b2 = random.choice(cands)
                        if b2 != best_b and st.bin_w[b2] + w <= C:
                            # undo/redo is expensive; do only if we haven't placed yet.
                            pass

        return st

    # ---- repair: hybrid regret + best-fit ----
    def insertion_options(st: State, it: int, bins: List[int], maxcand: int = 28) -> List[Tuple[Tuple[int, int, int], int]]:
        cand_bins = candidate_bins_for_item(st, it, bins, exclude=-1, limit=maxcand)
        w = weights[it]
        opts: List[Tuple[Tuple[int, int, int], int]] = []
        for b in cand_bins:
            sl = C - st.bin_w[b]
            left = sl - w
            opts.append(((left, -slack_value(left), -(st.bin_w[b] + w)), b))
        opts.sort(key=lambda x: x[0])
        return opts

    def regret_repair(st: State, removed: List[int]) -> None:
        pool = removed[:]
        pool.sort(key=lambda i: weights[i], reverse=True)
        while pool and not time_up_hard():
            bins = active_bins_cached(st)
            scan = pool if len(pool) <= 110 else random.sample(pool, 110)

            best_it = None
            best_score = None
            best_choice = -1

            for it in scan:
                opts = insertion_options(st, it, bins, maxcand=30)
                if not opts:
                    # forced new bin; make it urgent to place large items
                    score = (10**9, weights[it], 0)
                    choice = -1
                else:
                    c1 = opts[0][0]
                    c2 = opts[1][0] if len(opts) > 1 else (c1[0] + 10**6, c1[1], c1[2])
                    c3 = opts[2][0] if len(opts) > 2 else (c1[0] + 10**6, c1[1], c1[2])
                    # regret: difference in leftover (primary) + some usefulness signal
                    r1 = (c2[0] - c1[0])
                    r2 = (c3[0] - c1[0])
                    # also prefer heavier items to reduce future infeasibility
                    score = (r1 + r2, weights[it], slack_value(c1[0]))
                    choice = opts[0][1]

                if best_score is None or score > best_score:
                    best_score = score
                    best_it = it
                    best_choice = choice

            it = best_it
            if it is None:
                break
            pool.remove(it)
            if best_choice == -1:
                st.activate_bin_with_item(it)
            else:
                st.add_item(it, best_choice)

    # ---- shaking (ruin) ----
    def shake(st: State, k: int) -> State:
        ns = st.clone_compact()
        if ns.bins_used <= 1:
            return ns

        bins = ns.active_bins()
        bins.sort(key=lambda b: ns.bin_w[b])
        removed: List[int] = []

        r = random.random()
        if r < 0.50:
            # bin-ruin: remove t light bins
            t = min(len(bins), 1 + (k // 2))
            seg = bins[: max(t + 3, len(bins) // 3)]
            chosen = random.sample(seg, t)
            for b in chosen:
                for it in ns.bin_items[b][:]:
                    ns.remove_item(it)
                    removed.append(it)
        elif r < 0.80:
            # heavy-items ruin (biased)
            rcount = min(n, 10 + 3 * k)
            heavy = order_dec[: min(n, 10 * rcount)]
            picked = set()
            while len(picked) < rcount:
                it = random.choice(heavy)
                picked.add(it)
                if random.random() < 0.35:
                    b = ns.loc[it]
                    if b != -1 and ns.active[b] and ns.bin_items[b]:
                        picked.add(random.choice(ns.bin_items[b]))
            for it in picked:
                if ns.loc[it] != -1:
                    ns.remove_item(it)
                    removed.append(it)
        else:
            # slack-targeted ruin: pick bins with large slack and remove a couple items from them
            # idea: re-pack to improve tightness and enable bin elimination
            big_slack_bins = bins[:]
            big_slack_bins.sort(key=lambda b: (C - ns.bin_w[b]), reverse=True)
            take = min(len(big_slack_bins), 2 + k // 3)
            chosen = random.sample(big_slack_bins[: max(take + 2, len(big_slack_bins) // 2)], take)
            for b in chosen:
                items = ns.bin_items[b]
                if not items:
                    continue
                m = min(len(items), 2 + (k // 2))
                # remove largest few items to give flexibility
                items_sorted = sorted(items, key=lambda x: weights[x], reverse=True)
                for it in items_sorted[:m]:
                    if ns.loc[it] != -1:
                        ns.remove_item(it)
                        removed.append(it)

        if removed:
            regret_repair(ns, removed)
        return ns

    # ---- VND: neighborhood sequence focused on emptying a bin ----
    def try_direct_move(ns: State, it: int, src_bin: int) -> bool:
        bins = ns.active_bins()
        cand = candidate_bins_for_item(ns, it, bins, exclude=src_bin, limit=36)
        w = weights[it]
        for b in cand:
            if ns.bin_w[b] + w <= C:
                ns.move_item(it, b)
                return True
        return False

    def try_swap_move(ns: State, it: int, src_bin: int) -> bool:
        # swap it with a smaller item from another bin to make both feasible
        w_it = weights[it]
        bins = ns.active_bins()
        if len(bins) > 120:
            bins = random.sample(bins, 120)
        # consider tighter destination bins first
        bins.sort(key=lambda b: C - ns.bin_w[b])
        for b in bins:
            if b == src_bin:
                continue
            # need space for it
            need = ns.bin_w[b] + w_it - C
            if need <= 0:
                continue
            items_b = ns.bin_items[b]
            if not items_b:
                continue
            # find a candidate victim v with weight >= need and such that v can go to src
            # sample few candidates
            samp = items_b if len(items_b) <= 10 else random.sample(items_b, 10)
            for v in samp:
                w_v = weights[v]
                if w_v < need:
                    continue
                # after swap:
                if ns.bin_w[src_bin] - w_it + w_v > C:
                    continue
                # execute swap
                ns.remove_item(it)
                ns.remove_item(v)
                # re-add
                if not ns.active[b]:
                    ns.active[b] = True
                    ns.bin_items[b] = []
                    ns.bin_w[b] = 0
                    ns.bins_used += 1
                if not ns.active[src_bin]:
                    ns.active[src_bin] = True
                    ns.bin_items[src_bin] = []
                    ns.bin_w[src_bin] = 0
                    ns.bins_used += 1
                ns.add_item(it, b)
                ns.add_item(v, src_bin)
                return True
        return False

    def ejection_chain(ns: State, it: int, src_bin: int, depth: int = 4, beam: int = 22) -> bool:
        # bounded best-first ejection chain
        w0 = weights[it]

        def node_score(leftover: int, d: int) -> Tuple[int, int, int]:
            return (leftover, d, random.randrange(1_000_000))

        bins = ns.active_bins()
        if len(bins) > 120:
            bins = random.sample(bins, 120)

        frontier = []
        for b in bins:
            if b == src_bin:
                continue
            if ns.bin_w[b] + w0 <= C:
                continue
            need = ns.bin_w[b] + w0 - C
            items_b = ns.bin_items[b]
            if not items_b:
                continue
            samp = items_b if len(items_b) <= 10 else random.sample(items_b, 10)
            for v in samp:
                if weights[v] >= need:
                    neww = ns.bin_w[b] - weights[v] + w0
                    frontier.append((node_score(C - neww, 1), v, b, [(it, b, v)]))

        if not frontier:
            return False
        frontier.sort(key=lambda x: x[0])
        frontier = frontier[:beam]

        seen = set()
        for d in range(2, depth + 1):
            if time_up():
                return False
            new_frontier = []
            for sc, itp, forbid, plan in frontier:
                key = (itp, forbid, d)
                if key in seen:
                    continue
                seen.add(key)
                w = weights[itp]

                # direct placement attempt
                bins2 = ns.active_bins()
                cand = candidate_bins_for_item(ns, itp, bins2, exclude=forbid, limit=34)
                for b2 in cand:
                    if ns.bin_w[b2] + w <= C:
                        full_plan = plan + [(itp, b2, -1)]
                        # commit
                        to_remove = []
                        for pi, pb, ev in full_plan:
                            if ns.loc[pi] != -1:
                                to_remove.append(pi)
                            if ev != -1 and ns.loc[ev] != -1:
                                to_remove.append(ev)
                        uniq = []
                        s = set()
                        for x in to_remove:
                            if x not in s:
                                s.add(x)
                                uniq.append(x)
                        for x in uniq:
                            ns.remove_item(x)
                        for pi, pb, ev in full_plan:
                            if not ns.active[pb]:
                                ns.active[pb] = True
                                ns.bin_items[pb] = []
                                ns.bin_w[pb] = 0
                                ns.bins_used += 1
                            if ns.bin_w[pb] + weights[pi] > C:
                                return False
                            ns.add_item(pi, pb)
                        return True

                # extend with another ejection
                bins3 = ns.active_bins()
                if len(bins3) > 100:
                    bins3 = random.sample(bins3, 100)
                for b3 in bins3:
                    if b3 == forbid:
                        continue
                    if ns.bin_w[b3] + w <= C:
                        continue
                    need = ns.bin_w[b3] + w - C
                    items_b3 = ns.bin_items[b3]
                    if not items_b3:
                        continue
                    samp3 = items_b3 if len(items_b3) <= 8 else random.sample(items_b3, 8)
                    for v3 in samp3:
                        if weights[v3] >= need:
                            neww = ns.bin_w[b3] - weights[v3] + w
                            new_frontier.append((node_score(C - neww, d), v3, b3, plan + [(itp, b3, v3)]))

            if not new_frontier:
                return False
            new_frontier.sort(key=lambda x: x[0])
            frontier = new_frontier[:beam]

        return False

    def empty_bin(ns: State, b: int, max_steps: int = 220) -> bool:
        if not ns.active[b] or not ns.bin_items[b]:
            return True
        steps = 0
        # work largest-first
        while ns.active[b] and ns.bin_items[b] and steps < max_steps and not time_up_hard():
            items = ns.bin_items[b]
            it = max(items, key=lambda x: weights[x])
            if try_direct_move(ns, it, b):
                steps += 1
                continue
            if try_swap_move(ns, it, b):
                steps += 2
                continue
            if ejection_chain(ns, it, b, depth=4, beam=22):
                steps += 4
                continue
            break
        return (not ns.active[b]) or (not ns.bin_items[b])

    def vnd(ns: State, rounds: int = 6) -> None:
        if ns.bins_used <= 1:
            return
        bins = ns.active_bins()
        # focus on light bins (most removable)
        bins.sort(key=lambda b: ns.bin_w[b])
        targets = bins[: min(len(bins), rounds + 2)]
        if len(bins) >= 6:
            targets.append(random.choice(bins))
        seenb = set()
        for b in targets:
            if b in seenb:
                continue
            seenb.add(b)
            if time_up_hard():
                return
            empty_bin(ns, b, max_steps=240)

    # ---- quick post-process merge attempt (very light) ----
    def try_merge_pass(st: State, passes: int = 2) -> None:
        # Attempt to move all items of a light bin into others (like VND but cheaper)
        for _ in range(passes):
            if time_up_hard() or st.bins_used <= 1:
                return
            bins = st.active_bins()
            bins.sort(key=lambda b: st.bin_w[b])
            for b in bins[: min(8, len(bins))]:
                if time_up():
                    return
                if not st.active[b] or not st.bin_items[b]:
                    continue
                items = sorted(st.bin_items[b], key=lambda x: weights[x], reverse=True)
                ok = True
                for it in items:
                    dst = best_fit_bin_for_item(st, it, st.active_bins(), exclude=b)
                    if dst == -1:
                        ok = False
                        break
                    st.move_item(it, dst)
                if ok:
                    # bin should be empty now
                    continue

    # ---- Initialization: spend more time, better diversity ----
    init_end = start + 0.28 * (deadline - start)
    best = None
    best_obj = None

    tries = 0
    while tries < 140 and time.perf_counter() < init_end:
        tries += 1
        mode = tries % 5
        st = construct(mode)
        try_merge_pass(st, passes=1)
        vnd(st, rounds=4)
        o = st.obj()
        if best is None or o < best_obj:
            best = st.clone_compact()
            best_obj = o
        if best_obj[0] == lb:
            break

    if best is None:
        best = construct(0).clone_compact()
        best_obj = best.obj()

    current = best.clone_compact()

    # ---- Main BVNS loop ----
    MAX_ITERS = 900000  # fixed iterations; time checks terminate early
    k_max = 26
    stagn = 0

    for itn in range(MAX_ITERS):
        if (itn & 255) == 0 and time_up_hard():
            break

        base = current
        k = 1

        while k <= k_max and not time_up_hard():
            ns = shake(base, k)
            vnd(ns, rounds=6)

            o = ns.obj()
            if o < best_obj:
                best = ns.clone_compact()
                best_obj = o
                stagn = 0
                if best_obj[0] == lb:
                    current = best.clone_compact()
                    break

            cur_o = current.obj()

            # acceptance
            accept = False
            if o[0] < cur_o[0]:
                accept = True
            elif o[0] == cur_o[0] and o[1] < cur_o[1]:
                accept = True
            elif o[0] == cur_o[0] and random.random() < 0.08:
                accept = True

            if accept:
                current = ns
                base = current
                k = 1
            else:
                k += 1

            if time_up():
                break

        stagn += 1

        # adaptive diversification
        if (itn % 100) == 0:
            if stagn > 260:
                k_max = min(48, k_max + 2)
            elif stagn < 55:
                k_max = max(20, k_max - 1)

        # periodic intensification around best
        if (itn % 240) == 0 and not time_up_hard():
            tmp = best.clone_compact()
            vnd(tmp, rounds=10)
            try_merge_pass(tmp, passes=1)
            o2 = tmp.obj()
            if o2 < best_obj:
                best = tmp.clone_compact()
                best_obj = o2
                current = best.clone_compact()
                stagn = 0

        # keep current near best
        if current.obj()[0] > best_obj[0] + 1:
            current = best.clone_compact()

    # ---- Output best ----
    final = best.clone_compact()
    packing: List[List[int]] = []
    bin_weights: List[int] = []
    for b in range(len(final.bin_items)):
        if final.active[b] and final.bin_items[b]:
            packing.append(final.bin_items[b][:])
            bin_weights.append(final.bin_w[b])

    # Safety check: repair with greedy if needed
    seen = [0] * n
    for bb in packing:
        for it in bb:
            if 0 <= it < n:
                seen[it] += 1
    if any(x != 1 for x in seen):
        st = construct(0).clone_compact()
        packing = []
        bin_weights = []
        for b in range(len(st.bin_items)):
            if st.active[b] and st.bin_items[b]:
                packing.append(st.bin_items[b][:])
                bin_weights.append(st.bin_w[b])

    return {"packing": packing, "bin_weights": bin_weights}
