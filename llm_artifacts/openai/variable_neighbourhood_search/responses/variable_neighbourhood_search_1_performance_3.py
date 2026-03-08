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
    TICK_MASK = 4095

    def time_up_hard() -> bool:
        return time.perf_counter() >= deadline

    def time_up() -> bool:
        nonlocal tick
        tick += 1
        if (tick & TICK_MASK) == 0:
            return time.perf_counter() >= deadline
        return False

    # ---- lower bound (simple but useful) ----
    total_w = sum(weights)
    lb = (total_w + C - 1) // C

    # ---- ordering / helper data ----
    order_dec = sorted(range(n), key=lambda i: weights[i], reverse=True)

    # Frequency-ish slack preference: try to leave slacks that can be filled by remaining items.
    # We approximate by counting weights.
    wcount: Dict[int, int] = {}
    for w in weights:
        wcount[w] = wcount.get(w, 0) + 1

    def slack_value(sl: int) -> int:
        # higher is better (slack matches common item sizes)
        # consider exact match and a couple near matches
        v = 0
        v += 6 * wcount.get(sl, 0)
        v += 3 * wcount.get(sl - 1, 0)
        v += 3 * wcount.get(sl + 1, 0)
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
                bw = self.bin_w[bid]
                ns.bin_w.append(bw)
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
            # assumes feasible and active
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
            # dst must be active
            self.add_item(it, dst)

        def feasible(self, it: int, bid: int) -> bool:
            return self.active[bid] and self.bin_w[bid] + weights[it] <= C

        def active_bins(self) -> List[int]:
            return [b for b, a in enumerate(self.active) if a and self.bin_items[b]]

        def obj(self) -> Tuple[int, int]:
            # Primary: bins used; Secondary: total squared slack (tighter is better)
            s2 = 0
            for b in range(len(self.bin_items)):
                if self.active[b] and self.bin_items[b]:
                    sl = C - self.bin_w[b]
                    s2 += sl * sl
            return (self.bins_used, s2)

    # ---- candidate bin selection ----
    def sample_bins_for_item(st: State, w: int, exclude: int = -1, limit: int = 28) -> List[int]:
        bins = st.active_bins()
        if exclude != -1:
            # filter in loop
            pass
        if not bins:
            return []

        # Prefer tight slacks. We do a partial selection by sampling a subset.
        # For large instances, random sampling is cheaper.
        if len(bins) > 80:
            samp = random.sample(bins, 80)
        else:
            samp = bins

        cand = []
        for b in samp:
            if b == exclude:
                continue
            sl = C - st.bin_w[b]
            if sl >= w:
                cand.append((sl - w, -slack_value(sl - w), b))  # smaller leftover; prefer good leftover
        cand.sort()
        return [b for _, __, b in cand[:limit]]

    # ---- construction (multi-start): BFD with randomized tie-breaking ----
    def construct() -> State:
        st = State()
        for it in order_dec:
            w = weights[it]
            best_b = -1
            best_key = None
            # try a limited candidate set
            bins = st.active_bins()
            if bins:
                # evaluate a sample if many
                if len(bins) > 120:
                    bins_eval = random.sample(bins, 120)
                else:
                    bins_eval = bins
                for b in bins_eval:
                    sl = C - st.bin_w[b]
                    if sl >= w:
                        left = sl - w
                        key = (left, -slack_value(left))
                        if best_key is None or key < best_key:
                            best_key = key
                            best_b = b
            if best_b == -1:
                st.activate_bin_with_item(it)
            else:
                st.add_item(it, best_b)
        return st

    # ---- regret-k reinsertion (k=3) ----
    def insertion_options(st: State, it: int, maxcand: int = 24) -> List[Tuple[Tuple[int, int, int], int]]:
        w = weights[it]
        cand_bins = sample_bins_for_item(st, w, exclude=-1, limit=maxcand)
        opts: List[Tuple[Tuple[int, int, int], int]] = []
        for b in cand_bins:
            sl = C - st.bin_w[b]
            if sl >= w:
                left = sl - w
                # primary: leftover small; secondary: leftover "useful"; tertiary: keep bin weight high
                opts.append(((left, -slack_value(left), -(st.bin_w[b] + w)), b))
        opts.sort(key=lambda x: x[0])
        return opts

    def regret_reinsert(st: State, removed: List[int]) -> None:
        pool = removed[:]
        pool.sort(key=lambda i: weights[i], reverse=True)
        while pool and not time_up_hard():
            scan = pool if len(pool) <= 90 else random.sample(pool, 90)

            best_it = None
            best_score = None
            best_choice = None

            for it in scan:
                opts = insertion_options(st, it, maxcand=26)
                if not opts:
                    # must open new bin
                    score = (10**9, weights[it])
                    choice = -1
                else:
                    c1 = opts[0][0]
                    c2 = opts[1][0] if len(opts) > 1 else (c1[0] + 10**6, c1[1], c1[2])
                    c3 = opts[2][0] if len(opts) > 2 else (c1[0] + 10**6, c1[1], c1[2])
                    # regret based on first vs next bests
                    r1 = (c2[0] - c1[0])
                    r2 = (c3[0] - c1[0])
                    score = (r1 + r2, weights[it])
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

    # ---- Shake operators (bin-oriented + item-oriented) ----
    def shake(st: State, k: int) -> State:
        ns = st.clone_compact()
        if ns.bins_used <= 1:
            return ns

        bins = ns.active_bins()
        # sort by increasing fill (light bins are easier to remove)
        bins.sort(key=lambda b: ns.bin_w[b])

        removed: List[int] = []

        # Operator choice based on k
        r = random.random()
        if r < 0.60:
            # Bin-ruin: remove t light bins
            t = min(len(bins), 1 + (k // 2))
            # allow some randomness among lightest segment
            seg = bins[: max(t + 2, len(bins) // 3)]
            chosen = random.sample(seg, t)
            for b in chosen:
                for it in ns.bin_items[b][:]:
                    ns.remove_item(it)
                    removed.append(it)
        elif r < 0.85:
            # Heavy-items ruin: remove r items (biased to heavy)
            rcount = min(n, 8 + 2 * k)
            heavy = order_dec[: min(n, 6 * rcount)]
            picked = set()
            while len(picked) < rcount:
                picked.add(random.choice(heavy))
                if random.random() < 0.25:
                    it = random.choice(heavy)
                    b = ns.loc[it]
                    if b != -1 and ns.active[b] and ns.bin_items[b]:
                        picked.add(random.choice(ns.bin_items[b]))
            for it in picked:
                if ns.loc[it] != -1:
                    ns.remove_item(it)
                    removed.append(it)
        else:
            # Mixed: remove 1 light bin + a few heavy items
            b = bins[0]
            for it in ns.bin_items[b][:]:
                ns.remove_item(it)
                removed.append(it)
            rcount = min(n, 5 + k)
            heavy = order_dec[: min(n, 8 * rcount)]
            for _ in range(rcount):
                it = random.choice(heavy)
                if ns.loc[it] != -1:
                    ns.remove_item(it)
                    removed.append(it)

        regret_reinsert(ns, removed)
        return ns

    # ---- Targeted VND: try to eliminate bins (standard VNS intensification) ----
    def try_direct_relocate(ns: State, it: int, src_bin: int) -> bool:
        w = weights[it]
        cand = sample_bins_for_item(ns, w, exclude=src_bin, limit=30)
        for b in cand:
            if ns.bin_w[b] + w <= C:
                ns.move_item(it, b)
                return True
        return False

    def ejection_chain(ns: State, it: int, src_bin: int, depth: int = 5, beam: int = 20) -> bool:
        # bounded best-first ejection chain: place it into some bin by ejecting a victim, recursively place victim.
        w0 = weights[it]

        # Each node: (score, item_to_place, forbidden_bin, plan)
        # plan: list of (place_item, dst_bin, ejected_item_or_-1)
        def node_score(leftover: int, d: int) -> Tuple[int, int, int]:
            # prefer tight placement, shallower
            return (leftover, d, random.randrange(1000000))

        frontier = []
        # first level: bins that could take it if we eject one victim
        bins = ns.active_bins()
        if len(bins) > 90:
            bins = random.sample(bins, 90)
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
                    leftover = C - neww
                    frontier.append((node_score(leftover, 1), v, b, [(it, b, v)]))

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

                # try direct placement
                cand = sample_bins_for_item(ns, w, exclude=forbid, limit=26)
                for b2 in cand:
                    if b2 == forbid:
                        continue
                    if ns.bin_w[b2] + w <= C:
                        full_plan = plan + [(itp, b2, -1)]
                        # execute (commit)
                        to_remove = []
                        for pi, pb, ev in full_plan:
                            if ns.loc[pi] != -1:
                                to_remove.append(pi)
                            if ev != -1 and ns.loc[ev] != -1:
                                to_remove.append(ev)
                        # unique
                        uniq = []
                        sr = set()
                        for x in to_remove:
                            if x not in sr:
                                uniq.append(x)
                                sr.add(x)
                        for x in uniq:
                            ns.remove_item(x)
                        # add placements
                        for pi, pb, ev in full_plan:
                            if not ns.active[pb]:
                                # should not happen
                                ns.active[pb] = True
                                ns.bin_items[pb] = []
                                ns.bin_w[pb] = 0
                                ns.bins_used += 1
                            if ns.bin_w[pb] + weights[pi] > C:
                                return False
                            ns.add_item(pi, pb)
                        return True

                # else extend with another ejection
                # pick candidate bins to attempt ejection
                bins2 = ns.active_bins()
                if len(bins2) > 70:
                    bins2 = random.sample(bins2, 70)
                for b3 in bins2:
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
                            leftover = C - neww
                            new_frontier.append((node_score(leftover, d), v3, b3, plan + [(itp, b3, v3)]))

            if not new_frontier:
                return False
            new_frontier.sort(key=lambda x: x[0])
            frontier = new_frontier[:beam]

        return False

    def empty_bin(ns: State, b: int, max_moves: int = 160) -> bool:
        if not ns.active[b] or not ns.bin_items[b]:
            return True
        moves = 0
        while ns.active[b] and ns.bin_items[b] and moves < max_moves and not time_up_hard():
            # take largest item first
            it = max(ns.bin_items[b], key=lambda x: weights[x])
            if try_direct_relocate(ns, it, b):
                moves += 1
                continue
            if ejection_chain(ns, it, b, depth=5, beam=18):
                moves += 3
                continue
            break
        return (not ns.active[b]) or (not ns.bin_items[b])

    def vnd(ns: State, rounds: int = 4) -> None:
        # Try to eliminate several light bins
        if ns.bins_used <= 1:
            return
        bins = ns.active_bins()
        bins.sort(key=lambda b: ns.bin_w[b])
        # focus on lightest + one random
        targets = bins[: min(len(bins), rounds + 2)]
        if len(bins) > 4:
            targets.append(random.choice(bins))
        seenb = set()
        for b in targets:
            if b in seenb:
                continue
            seenb.add(b)
            if time_up_hard():
                return
            empty_bin(ns, b, max_moves=170)

    # ---- Initialization: multi-start then intensify ----
    init_end = start + 0.20 * (deadline - start)
    best = None
    best_obj = None

    tries = 0
    while tries < 60 and time.perf_counter() < init_end:
        tries += 1
        st = construct()
        vnd(st, rounds=3)
        o = st.obj()
        if best is None or o < best_obj:
            best = st.clone_compact()
            best_obj = o
        if best_obj[0] == lb:
            break

    if best is None:
        best = construct().clone_compact()
        best_obj = best.obj()

    current = best.clone_compact()

    # ---- Main BVNS loop ----
    MAX_ITERS = 400000
    k_max = 18
    stagn = 0

    for it in range(MAX_ITERS):
        if (it & 255) == 0 and time_up_hard():
            break

        base = current
        k = 1

        while k <= k_max and not time_up_hard():
            ns = shake(base, k)
            vnd(ns, rounds=4)

            o = ns.obj()
            if o < best_obj:
                best = ns.clone_compact()
                best_obj = o
                stagn = 0
                if best_obj[0] == lb:
                    # cannot beat LB
                    current = best.clone_compact()
                    break

            cur_o = current.obj()

            # acceptance (VNS): accept improving; sometimes accept equal bins with better compactness
            accept = False
            if o[0] < cur_o[0]:
                accept = True
            elif o[0] == cur_o[0] and o[1] <= cur_o[1]:
                accept = True
            elif o[0] == cur_o[0] and random.random() < 0.06:
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

        # adaptive k_max and periodic intensification around best
        if (it % 80) == 0:
            if stagn > 220:
                k_max = min(42, k_max + 2)
            elif stagn < 40:
                k_max = max(14, k_max - 1)

        if (it % 250) == 0 and not time_up_hard():
            # intensify: try harder to remove light bins from best
            tmp = best.clone_compact()
            vnd(tmp, rounds=7)
            o2 = tmp.obj()
            if o2 < best_obj:
                best = tmp.clone_compact()
                best_obj = o2
                current = best.clone_compact()
                stagn = 0

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
        st = construct().clone_compact()
        packing = []
        bin_weights = []
        for b in range(len(st.bin_items)):
            if st.active[b] and st.bin_items[b]:
                packing.append(st.bin_items[b][:])
                bin_weights.append(st.bin_w[b])

    return {"packing": packing, "bin_weights": bin_weights}
