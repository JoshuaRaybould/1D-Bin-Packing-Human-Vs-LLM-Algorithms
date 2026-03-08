import time
import random
from typing import List, Tuple, Dict, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    n = len(weights)

    start = time.perf_counter()
    # Cap extremely large time limits for safety (per plan)
    deadline = start + min(max(0.0, float(time_limit)), 100.0)

    # -------- tick-based time checks --------
    tick = 0
    TICK_MASK = 8191  # check time every 8192 lightweight ops

    def time_up_hard() -> bool:
        return time.perf_counter() >= deadline

    def time_up() -> bool:
        nonlocal tick
        tick += 1
        if (tick & TICK_MASK) == 0:
            return time.perf_counter() >= deadline
        return False

    if n == 0:
        return {"packing": [], "bin_weights": []}

    wmax = max(weights)

    # -------- secondary metric (bin-quality) --------
    # Piecewise penalty on slack: penalize medium slack more than tiny slack.
    # Goal: encourage very full bins and a few sacrificial bins.
    small_thr = 1
    med_thr = min(C, max(3, wmax))  # slacks up to wmax are "blocking" many items

    def slack_penalty(s: int) -> int:
        # 0 for near-full, high for medium, moderate for huge.
        if s <= small_thr:
            return 0
        if s <= med_thr:
            # quadratic-ish growth in the blocking range
            return 10 * s * s
        # large slack: still bad but less informative
        return 10 * med_thr * med_thr + 2 * (s - med_thr)

    # -------- representation --------
    # bin_items[bid] list of items, bin_w[bid] sum weight
    # active[bid] indicates bin is in use. Empty bins may exist but should be inactive.
    # loc[item] = bin id, pos[item] = index in bin_items[loc[item]]

    class State:
        __slots__ = (
            "bin_items",
            "bin_w",
            "active",
            "loc",
            "pos",
            "bins_used",
            "sum_slack2",
            "sum_pen",
            "bucket_size",
            "nbuckets",
            "buckets",
            "bpos",
        )

        def __init__(self):
            self.bin_items: List[List[int]] = []
            self.bin_w: List[int] = []
            self.active: List[bool] = []
            self.loc: List[int] = [-1] * n
            self.pos: List[int] = [-1] * n
            self.bins_used: int = 0
            self.sum_slack2: int = 0
            self.sum_pen: int = 0

            # capacity buckets (coarse if C large)
            if C <= 4000:
                self.bucket_size = 1
            elif C <= 20000:
                self.bucket_size = 5
            else:
                self.bucket_size = 10
            self.nbuckets = C // self.bucket_size + 1
            self.buckets: List[List[int]] = [[] for _ in range(self.nbuckets)]
            self.bpos: List[int] = []  # per bin id: position inside its bucket list

        def clone_compact(self) -> "State":
            # Compact active bins into a new State (used for best snapshot / restart)
            ns = State()
            # rebuild from scratch
            for it in range(n):
                ns.loc[it] = -1
                ns.pos[it] = -1
            # add bins
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
                ns.bpos.append(-1)
                for pi, it in enumerate(items):
                    ns.loc[it] = nbid
                    ns.pos[it] = pi
            ns._recompute_metrics_and_buckets()
            return ns

        def _bucket_id(self, slack: int) -> int:
            if slack < 0:
                slack = 0
            if slack > C:
                slack = C
            return slack // self.bucket_size

        def _bucket_add_bin(self, bid: int) -> None:
            slack = C - self.bin_w[bid]
            b = self._bucket_id(slack)
            self.bpos[bid] = len(self.buckets[b])
            self.buckets[b].append(bid)

        def _bucket_remove_bin(self, bid: int) -> None:
            # Remove bid from its current bucket in O(1)
            slack = C - self.bin_w[bid]
            b = self._bucket_id(slack)
            p = self.bpos[bid]
            if p < 0:
                return
            last = self.buckets[b].pop()
            if last != bid:
                self.buckets[b][p] = last
                self.bpos[last] = p
            self.bpos[bid] = -1

        def _bucket_move_bin(self, bid: int, old_w: int, new_w: int) -> None:
            # update bin bucket after weight change (only if active)
            if not self.active[bid]:
                return
            old_slack = C - old_w
            new_slack = C - new_w
            if self._bucket_id(old_slack) == self._bucket_id(new_slack):
                return
            # remove using old bucket id by temporarily setting weight
            # but easiest: remove with old_w bucket then add with new_w
            # Implement remove by searching old bucket position would require tracking old; we can do:
            # temporarily set weight, remove, restore, then set final and add.
            cur_w = self.bin_w[bid]
            self.bin_w[bid] = old_w
            self._bucket_remove_bin(bid)
            self.bin_w[bid] = new_w
            self._bucket_add_bin(bid)
            self.bin_w[bid] = cur_w  # restore; caller will set final

        def _metrics_delta_for_bin_w_change(self, old_w: int, new_w: int) -> Tuple[int, int]:
            old_sl = C - old_w
            new_sl = C - new_w
            d_s2 = new_sl * new_sl - old_sl * old_sl
            d_pen = slack_penalty(new_sl) - slack_penalty(old_sl)
            return d_s2, d_pen

        def _recompute_metrics_and_buckets(self) -> None:
            self.bins_used = 0
            self.sum_slack2 = 0
            self.sum_pen = 0
            # reset buckets
            self.buckets = [[] for _ in range(self.nbuckets)]
            self.bpos = [-1] * len(self.bin_items)
            for bid in range(len(self.bin_items)):
                if not self.active[bid]:
                    continue
                if not self.bin_items[bid]:
                    self.active[bid] = False
                    continue
                self.bins_used += 1
                sl = C - self.bin_w[bid]
                self.sum_slack2 += sl * sl
                self.sum_pen += slack_penalty(sl)
                self._bucket_add_bin(bid)

        def obj(self) -> Tuple[int, int, int]:
            # lexicographic: bins_used, sum_pen, sum_slack2
            return (self.bins_used, self.sum_pen, self.sum_slack2)

        def ensure_bin(self) -> int:
            bid = len(self.bin_items)
            self.bin_items.append([])
            self.bin_w.append(0)
            self.active.append(False)
            self.bpos.append(-1)
            return bid

        def activate_bin_with_item(self, it: int) -> int:
            bid = self.ensure_bin()
            self.active[bid] = True
            self.bin_items[bid].append(it)
            self.bin_w[bid] = weights[it]
            self.loc[it] = bid
            self.pos[it] = 0
            self.bins_used += 1
            sl = C - self.bin_w[bid]
            self.sum_slack2 += sl * sl
            self.sum_pen += slack_penalty(sl)
            self._bucket_add_bin(bid)
            return bid

        def add_item(self, it: int, bid: int) -> None:
            # assumes feasible and active
            old_w = self.bin_w[bid]
            new_w = old_w + weights[it]
            d_s2, d_pen = self._metrics_delta_for_bin_w_change(old_w, new_w)
            self.sum_slack2 += d_s2
            self.sum_pen += d_pen
            # bucket move
            self._bucket_move_bin(bid, old_w, new_w)
            # apply
            self.bin_w[bid] = new_w
            self.pos[it] = len(self.bin_items[bid])
            self.loc[it] = bid
            self.bin_items[bid].append(it)
            # finalize bucket weight
            # (bucket_move temporarily restored; now ensure bin_w already new and bucket correct)
            # bucket_move already adjusted buckets via temporary set; now we must set weight and keep bucket.
            # Our bucket_move restores cur_w; so we need to re-set and not disturb bucket.
            # To avoid that complexity, we set correct weight after and then recompute bucket placement by:
            # remove/add using current bucket positions.
            # But we handled remove/add while temporarily setting; now bucket contains bid at new slack, but bin_w might differ.
            # Fix: set bin_w to new_w was done; but bucket_move restored to cur_w earlier, so bucket lists are correct but bin_w was wrong during restoration.
            # We set bin_w new_w; buckets are still correct.

        def remove_item(self, it: int) -> None:
            bid = self.loc[it]
            if bid < 0:
                return
            idx = self.pos[it]
            items = self.bin_items[bid]
            last = items[-1]
            # metrics update
            old_w = self.bin_w[bid]
            new_w = old_w - weights[it]
            d_s2, d_pen = self._metrics_delta_for_bin_w_change(old_w, new_w)
            self.sum_slack2 += d_s2
            self.sum_pen += d_pen
            self._bucket_move_bin(bid, old_w, new_w)
            # swap-pop
            items[idx] = last
            self.pos[last] = idx
            items.pop()
            self.loc[it] = -1
            self.pos[it] = -1
            self.bin_w[bid] = new_w
            # if bin becomes empty -> deactivate
            if not items:
                # remove from bucket
                self._bucket_remove_bin(bid)
                self.active[bid] = False
                self.bins_used -= 1
                # metrics already reflect empty bin slack? We currently updated slack change to new_w=0.
                # But an inactive bin should not contribute.
                sl = C - 0
                self.sum_slack2 -= sl * sl
                self.sum_pen -= slack_penalty(sl)

        def move_item(self, it: int, dst: int) -> None:
            # assumes feasible and dst active
            src = self.loc[it]
            if src == dst:
                return
            self.remove_item(it)
            if self.active[dst]:
                self.add_item(it, dst)
            else:
                # shouldn’t happen in normal use
                self.active[dst] = True
                self.bin_items[dst] = [it]
                self.bin_w[dst] = weights[it]
                self.loc[it] = dst
                self.pos[it] = 0
                self.bins_used += 1
                sl = C - self.bin_w[dst]
                self.sum_slack2 += sl * sl
                self.sum_pen += slack_penalty(sl)
                self._bucket_add_bin(dst)

        def feasible(self, it: int, bid: int) -> bool:
            return self.active[bid] and (self.bin_w[bid] + weights[it] <= C)

        def sample_feasible_bins_tight(self, w: int, exclude: int = -1, limit: int = 24) -> List[int]:
            # Use slack buckets: search from tightest feasible slack upward, then random fill.
            need = w
            b0 = self._bucket_id(need)
            cand: List[int] = []
            # tight scan
            for b in range(b0, min(self.nbuckets, b0 + 6)):
                for bid in self.buckets[b]:
                    if bid == exclude:
                        continue
                    if not self.active[bid]:
                        continue
                    if self.bin_w[bid] + w <= C:
                        cand.append(bid)
                        if len(cand) >= limit:
                            return cand
            # broaden a bit
            for b in range(b0 + 6, min(self.nbuckets, b0 + 18)):
                if len(cand) >= limit:
                    break
                bucket_list = self.buckets[b]
                if not bucket_list:
                    continue
                # sample a few from the bucket
                trials = min(4, len(bucket_list))
                for _ in range(trials):
                    bid = random.choice(bucket_list)
                    if bid == exclude or (not self.active[bid]):
                        continue
                    if self.bin_w[bid] + w <= C:
                        cand.append(bid)
                        if len(cand) >= limit:
                            return cand
            # random fallback among active bins
            if len(cand) < limit:
                active_ids = [i for i, a in enumerate(self.active) if a and i != exclude]
                if active_ids:
                    for _ in range(min(limit - len(cand), 12)):
                        bid = random.choice(active_ids)
                        if self.bin_w[bid] + w <= C:
                            cand.append(bid)
            return cand

    # -------- Constructive heuristics (multi-start) --------
    order_dec = sorted(range(n), key=lambda i: weights[i], reverse=True)

    def construct(method: str) -> State:
        st = State()
        for it in range(n):
            st.loc[it] = -1
            st.pos[it] = -1

        for it in order_dec:
            w = weights[it]
            chosen = -1

            if st.bins_used == 0:
                st.activate_bin_with_item(it)
                continue

            # gather candidates
            cand = st.sample_feasible_bins_tight(w, exclude=-1, limit=32)
            if not cand:
                st.activate_bin_with_item(it)
                continue

            if method == "ffd":
                # approximate first-fit: choose among candidates with smallest id
                chosen = min(cand)
            elif method == "bfd":
                # best-fit: tightest slack after insertion
                best_sl = None
                best_list = []
                for bid in cand:
                    sl = C - (st.bin_w[bid] + w)
                    if best_sl is None or sl < best_sl:
                        best_sl = sl
                        best_list = [bid]
                    elif sl == best_sl:
                        best_list.append(bid)
                chosen = random.choice(best_list)
            else:
                # "almost best": choose among top-k tight fits with bias
                scored = []
                for bid in cand:
                    sl = C - (st.bin_w[bid] + w)
                    scored.append((sl, bid))
                scored.sort(key=lambda x: x[0])
                top = scored[: min(5, len(scored))]
                # bias towards tighter
                r = random.random()
                idx = 0 if r < 0.55 else (1 if r < 0.8 else random.randrange(len(top)))
                chosen = top[idx][1]

            st.add_item(it, chosen)

        st._recompute_metrics_and_buckets()
        return st

    # -------- Ruin & recreate (shaking) with regret insertion --------
    def evaluate_insertion_cost(st: State, it: int, bid: int) -> Tuple[int, int]:
        # cost for secondary/tertiary if inserted into bid (bin count constant)
        old_w = st.bin_w[bid]
        new_w = old_w + weights[it]
        old_sl = C - old_w
        new_sl = C - new_w
        d_pen = slack_penalty(new_sl) - slack_penalty(old_sl)
        d_s2 = new_sl * new_sl - old_sl * old_sl
        return (d_pen, d_s2)

    def open_new_bin_cost(it: int) -> Tuple[int, int]:
        sl = C - weights[it]
        return (slack_penalty(sl), sl * sl)

    def regret_reinsert(st: State, removed: List[int], k: int) -> None:
        # Regret-2 insertion with optional new-bin opening probability increasing with k.
        pool = removed[:]
        # place larger first in tie situations
        pool.sort(key=lambda i: weights[i], reverse=True)

        # probability to allow opening a new bin even if feasible placements exist
        p_open = min(0.25, 0.02 * k)

        while pool and not time_up_hard():
            # compute regret for each item
            best_item = None
            best_data = None
            best_regret = None

            # sample if huge pool
            scan = pool if len(pool) <= 80 else random.sample(pool, 80)

            for it in scan:
                w = weights[it]
                cand_bins = st.sample_feasible_bins_tight(w, exclude=-1, limit=28)
                options: List[Tuple[Tuple[int, int], int]] = []
                for bid in cand_bins:
                    if st.bin_w[bid] + w <= C:
                        options.append((evaluate_insertion_cost(st, it, bid), bid))
                options.sort(key=lambda x: x[0])

                # consider opening new bin as an option
                new_cost = open_new_bin_cost(it)
                if options:
                    best_cost, best_bid = options[0]
                    second_cost = options[1][0] if len(options) >= 2 else (best_cost[0] + 999999, best_cost[1] + 999999)
                    regret = (second_cost[0] - best_cost[0], second_cost[1] - best_cost[1])
                    # if opening new bin is better than best (rare), treat as best
                    if new_cost < best_cost and random.random() < 0.8:
                        best_cost, best_bid = new_cost, -1
                        second_cost = options[0][0]
                        regret = (second_cost[0] - best_cost[0], second_cost[1] - best_cost[1])
                else:
                    best_cost, best_bid = new_cost, -1
                    regret = (10**9, 10**9)

                key = (regret[0], regret[1], weights[it])
                if best_regret is None or key > best_regret:
                    best_regret = key
                    best_item = it
                    best_data = (best_bid,)

            it = best_item
            if it is None:
                break
            pool.remove(it)
            w = weights[it]

            # choose placement for selected item
            cand_bins = st.sample_feasible_bins_tight(w, exclude=-1, limit=40)
            options = []
            for bid in cand_bins:
                if st.bin_w[bid] + w <= C:
                    options.append((evaluate_insertion_cost(st, it, bid), bid))
            options.sort(key=lambda x: x[0])

            if options:
                # maybe open new bin for diversification
                if random.random() < p_open and st.bins_used > 1:
                    # open if not too harmful: only if best option creates medium slack penalties
                    best_cost = options[0][0]
                    if best_cost[0] > 0:
                        st.activate_bin_with_item(it)
                        continue
                # pick among top few
                top = options[: min(4, len(options))]
                chosen = random.choice(top)[1]
                st.add_item(it, chosen)
            else:
                st.activate_bin_with_item(it)

    def shake(st: State, k: int) -> State:
        # Choose shake operator based on k (A/B/C) and do ruin&recreate.
        ns = st.clone_compact()

        if ns.bins_used <= 1:
            return ns

        # select active bins list
        active_bins = [bid for bid, a in enumerate(ns.active) if a]
        # compute bin ordering by weight (light first)
        active_bins.sort(key=lambda b: ns.bin_w[b])

        removed_items: List[int] = []

        op = "A" if k >= 3 and random.random() < 0.55 else ("C" if random.random() < 0.35 else "B")

        if op == "A":
            # remove t whole bins
            t = min(len(active_bins), 1 + (k // 2))
            # prefer light bins but with randomness
            cand = active_bins[: max(t + 2, len(active_bins) // 3)]
            chosen_bins = random.sample(cand, t)
            for b in chosen_bins:
                # remove all items
                items = ns.bin_items[b][:]
                for it in items:
                    ns.remove_item(it)
                    removed_items.append(it)

        elif op == "B":
            # remove critical large items + companions
            r = min(n, 6 + k)
            # weight-biased sampling without external libs
            # build a small candidate set of heavy items
            heavy = order_dec[: min(n, 3 * r)]
            removed = set()
            while len(removed) < r:
                it = random.choice(heavy)
                removed.add(it)
                # sometimes add a random companion from same bin
                if random.random() < 0.35:
                    b = ns.loc[it]
                    if b >= 0 and ns.active[b] and ns.bin_items[b]:
                        removed.add(random.choice(ns.bin_items[b]))
                if len(removed) >= r:
                    break
            for it in removed:
                if ns.loc[it] != -1:
                    ns.remove_item(it)
                    removed_items.append(it)

        else:  # "C"
            # related/conflict removal
            # pick random bin and item
            b0 = random.choice(active_bins)
            it0 = random.choice(ns.bin_items[b0])
            w0 = weights[it0]
            removed = {it0}
            # also remove some items from bins whose slack is just below w0 (blocking)
            target_slack = max(0, w0 - 1)
            bstart = ns._bucket_id(max(0, target_slack - 2 * ns.bucket_size))
            bend = ns._bucket_id(min(C, target_slack + 2 * ns.bucket_size))
            bins_blocking = []
            for b in range(bstart, bend + 1):
                bins_blocking.extend(ns.buckets[b])
            random.shuffle(bins_blocking)
            for b in bins_blocking[: 3 + k // 2]:
                if not ns.active[b] or not ns.bin_items[b]:
                    continue
                removed.add(random.choice(ns.bin_items[b]))
                if random.random() < 0.3 and len(ns.bin_items[b]) >= 2:
                    removed.add(random.choice(ns.bin_items[b]))
            # add some from same bin as it0
            for _ in range(1 + k // 3):
                removed.add(random.choice(ns.bin_items[b0]))

            for it in removed:
                if ns.loc[it] != -1:
                    ns.remove_item(it)
                    removed_items.append(it)

        # recreate via regret insertion
        regret_reinsert(ns, removed_items, k)
        # ensure consistency
        ns._recompute_metrics_and_buckets()
        return ns

    # -------- Empty-bin procedure with targeted neighborhoods & ejection chains --------
    def try_direct_relocate(ns: State, it: int, target_bin: int) -> bool:
        w = weights[it]
        cand = ns.sample_feasible_bins_tight(w, exclude=target_bin, limit=28)
        # try tightest first (implicitly via bucket ordering)
        for b in cand:
            if b == target_bin or not ns.active[b]:
                continue
            if ns.bin_w[b] + w <= C:
                ns.move_item(it, b)
                return True
        return False

    def try_swap_1_1(ns: State, it: int, target_bin: int) -> bool:
        # Swap item it in target with some item in another bin so target weight decreases
        w = weights[it]
        # choose candidate bins that could accept w after swap
        cand_bins = ns.sample_feasible_bins_tight(w, exclude=target_bin, limit=18)
        random.shuffle(cand_bins)
        for b in cand_bins:
            if not ns.active[b] or b == target_bin:
                continue
            # sample items from b
            items_b = ns.bin_items[b]
            if not items_b:
                continue
            samp = items_b if len(items_b) <= 8 else random.sample(items_b, 8)
            for jb in samp:
                wb = weights[jb]
                # after swap: target loses it gains jb; b loses jb gains it
                new_wt = ns.bin_w[target_bin] - w + wb
                new_wb = ns.bin_w[b] - wb + w
                if new_wt > C or new_wb > C:
                    continue
                # Prefer swaps that reduce target weight or item count difficulty
                if new_wt >= ns.bin_w[target_bin] and random.random() > 0.15:
                    continue

                # Apply swap
                # remove both then add
                ns.remove_item(it)
                ns.remove_item(jb)
                # re-add
                if not ns.active[target_bin]:
                    ns.active[target_bin] = True
                    ns.bin_items[target_bin] = []
                    ns.bin_w[target_bin] = 0
                    ns.bins_used += 1
                    sl = C
                    ns.sum_slack2 += sl * sl
                    ns.sum_pen += slack_penalty(sl)
                    ns._bucket_add_bin(target_bin)
                if not ns.active[b]:
                    ns.active[b] = True
                    ns.bin_items[b] = []
                    ns.bin_w[b] = 0
                    ns.bins_used += 1
                    sl = C
                    ns.sum_slack2 += sl * sl
                    ns.sum_pen += slack_penalty(sl)
                    ns._bucket_add_bin(b)

                ns.add_item(jb, target_bin)
                ns.add_item(it, b)
                return True
        return False

    def try_swap_2_2(ns: State, target_bin: int) -> bool:
        items_t = ns.bin_items[target_bin]
        if len(items_t) < 2:
            return False
        # sample 2 items from target
        samp_t = items_t if len(items_t) <= 8 else random.sample(items_t, 8)
        pairs = []
        for i in range(len(samp_t)):
            for j in range(i + 1, len(samp_t)):
                a, b = samp_t[i], samp_t[j]
                pairs.append((a, b, weights[a] + weights[b]))
        random.shuffle(pairs)
        pairs = pairs[:20]

        # choose other bins to try
        other_bins = [bid for bid, a in enumerate(ns.active) if a and bid != target_bin]
        if not other_bins:
            return False
        random.shuffle(other_bins)
        other_bins = other_bins[:12]

        for (a1, a2, wa) in pairs:
            for ob in other_bins:
                items_o = ns.bin_items[ob]
                if len(items_o) < 2:
                    continue
                samp_o = items_o if len(items_o) <= 8 else random.sample(items_o, 8)
                # choose pair from other
                for _ in range(min(10, len(samp_o) * (len(samp_o) - 1) // 2)):
                    b1, b2 = random.sample(samp_o, 2)
                    wb = weights[b1] + weights[b2]
                    new_wt = ns.bin_w[target_bin] - wa + wb
                    new_wo = ns.bin_w[ob] - wb + wa
                    if new_wt > C or new_wo > C:
                        continue
                    # Apply 2-2 swap
                    ns.remove_item(a1)
                    ns.remove_item(a2)
                    ns.remove_item(b1)
                    ns.remove_item(b2)

                    # Ensure bins still active containers if emptied during removals
                    if ns.loc[a1] == -1 and not ns.active[target_bin]:
                        # should not happen because target still has other items; but keep safe
                        ns.active[target_bin] = True
                        ns.bin_items[target_bin] = []
                        ns.bin_w[target_bin] = 0
                        ns.bins_used += 1
                        sl = C
                        ns.sum_slack2 += sl * sl
                        ns.sum_pen += slack_penalty(sl)
                        ns._bucket_add_bin(target_bin)
                    if not ns.active[ob]:
                        ns.active[ob] = True
                        ns.bin_items[ob] = []
                        ns.bin_w[ob] = 0
                        ns.bins_used += 1
                        sl = C
                        ns.sum_slack2 += sl * sl
                        ns.sum_pen += slack_penalty(sl)
                        ns._bucket_add_bin(ob)

                    ns.add_item(b1, target_bin)
                    ns.add_item(b2, target_bin)
                    ns.add_item(a1, ob)
                    ns.add_item(a2, ob)
                    return True
        return False

    def ejection_chain_place(ns: State, item: int, src_bin: int, depth: int = 4, beam: int = 18) -> bool:
        # Attempt to move `item` out of src_bin by bounded ejection chain.
        # State in search: (current_item_to_place, forbidden_bin, moved_items list)
        # We apply moves on success only; search uses lightweight simulation via planned moves.

        w0 = weights[item]

        # Node: (score_tuple, item_to_place, exclude_bin, plan)
        # plan is list of (move_type, a, b):
        #  - ("move", it, dst)
        #  - ("eject", victim, dst_of_victim) meaning victim will be moved later; here we just record removal.
        # For simplicity, we record a sequence of (place_item, into_bin, ejected_item_or_-1)

        def plan_score(slack_after: int) -> int:
            # lower is better (prefer tight fit)
            return slack_after

        # initial candidates: try bins that can take item directly (should have been tried already)
        # chain candidates: bins that can take item if one victim is ejected.

        seen = set()
        frontier: List[Tuple[Tuple[int, int, int], int, int, List[Tuple[int, int, int]]]] = []

        def push(sc: Tuple[int, int, int], itp: int, excl: int, plan: List[Tuple[int, int, int]]):
            frontier.append((sc, itp, excl, plan))

        # start with placing item with one ejection
        cand_bins = ns.sample_feasible_bins_tight(w0, exclude=src_bin, limit=30)
        for b in cand_bins:
            if b == src_bin or not ns.active[b]:
                continue
            if ns.bin_w[b] + w0 <= C:
                continue
            # need to eject some victim from b
            # choose victims whose removal makes room
            need = ns.bin_w[b] + w0 - C
            items_b = ns.bin_items[b]
            if not items_b:
                continue
            samp = items_b if len(items_b) <= 10 else random.sample(items_b, 10)
            for v in samp:
                if weights[v] >= need:
                    slack_after = C - (ns.bin_w[b] - weights[v] + w0)
                    sc = (plan_score(slack_after), 0, 0)
                    push(sc, v, b, [(item, b, v)])

        if not frontier:
            return False

        # best-first by score (small beam)
        frontier.sort(key=lambda x: x[0])
        frontier = frontier[:beam]

        for d in range(1, depth + 1):
            if time_up():
                return False
            new_frontier = []
            for sc, itp, excl_bin, plan in frontier:
                key = (itp, excl_bin, d)
                if key in seen:
                    continue
                seen.add(key)

                w = weights[itp]
                # try to place itp directly somewhere (excluding excl_bin to avoid immediate undo)
                cand2 = ns.sample_feasible_bins_tight(w, exclude=excl_bin, limit=26)
                for b2 in cand2:
                    if not ns.active[b2] or b2 == excl_bin:
                        continue
                    if ns.bin_w[b2] + w <= C:
                        # success: we can execute plan: move item->bin with victim ejected, then place victim chain ...
                        full_plan = plan + [(itp, b2, -1)]
                        # execute
                        # remove src item and all ejected victims first to avoid conflicts
                        to_remove = []
                        # collect all placed items and ejected victims
                        for (pi, pb, ev) in full_plan:
                            if ns.loc[pi] != -1:
                                to_remove.append(pi)
                            if ev != -1 and ns.loc[ev] != -1:
                                to_remove.append(ev)
                        # unique while preserving order
                        uniq = []
                        seenr = set()
                        for x in to_remove:
                            if x not in seenr:
                                uniq.append(x)
                                seenr.add(x)
                        # capture their original bins for possible issues (but we commit only on success)
                        for x in uniq:
                            ns.remove_item(x)

                        # place according to plan in order
                        # ensure destination bins are active (should be)
                        ok = True
                        for (pi, pb, ev) in full_plan:
                            if not ns.active[pb]:
                                ok = False
                                break
                            if ns.bin_w[pb] + weights[pi] > C:
                                ok = False
                                break
                            ns.add_item(pi, pb)
                        if ok:
                            return True
                        # if something went wrong, abort by returning False (rare); caller can recompute state via compaction elsewhere
                        return False

                # else extend by ejecting from some bin to place itp
                cand3 = ns.sample_feasible_bins_tight(w, exclude=excl_bin, limit=18)
                for b3 in cand3:
                    if not ns.active[b3] or b3 == excl_bin:
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
                            slack_after = C - (ns.bin_w[b3] - weights[v3] + w)
                            sc2 = (plan_score(slack_after), d, random.randrange(1000))
                            new_frontier.append((sc2, v3, b3, plan + [(itp, b3, v3)]))

            if not new_frontier:
                return False
            new_frontier.sort(key=lambda x: x[0])
            frontier = new_frontier[:beam]

        return False

    def empty_bin(ns: State, bid: int, max_moves: int = 120) -> bool:
        # Attempt to empty a specific bin using N1..N5.
        if not ns.active[bid] or not ns.bin_items[bid]:
            return True

        moves = 0
        # try largest-first from this bin
        while ns.active[bid] and ns.bin_items[bid] and moves < max_moves and not time_up_hard():
            items = ns.bin_items[bid]
            # choose a difficult item: largest
            it = max(items, key=lambda x: weights[x])

            # N1 direct relocate
            if try_direct_relocate(ns, it, bid):
                moves += 1
                continue
            # N2 1-1 swap
            if try_swap_1_1(ns, it, bid):
                moves += 1
                continue
            # N4 2-2 swap occasionally
            if random.random() < 0.35 and try_swap_2_2(ns, bid):
                moves += 2
                continue
            # N5 ejection chain
            if ejection_chain_place(ns, it, bid, depth=4, beam=16):
                moves += 3
                continue

            # no progress
            break

        # if emptied, bin will be inactive
        return (not ns.active[bid]) or (not ns.bin_items[bid])

    # -------- VND: bin-empty attempts + brief global quality pass --------
    def vnd(ns: State, attempts: int = 3) -> None:
        # 1) attempt to eliminate a few candidate bins
        if ns.bins_used <= 1:
            return
        active_bins = [bid for bid, a in enumerate(ns.active) if a]
        active_bins.sort(key=lambda b: ns.bin_w[b])
        cand = active_bins[: min(len(active_bins), attempts + 2)]
        # include one random bin too
        if len(active_bins) > 3:
            cand.append(random.choice(active_bins))
        # unique
        seenb = set()
        cand2 = []
        for b in cand:
            if b not in seenb:
                cand2.append(b)
                seenb.add(b)
        for b in cand2:
            if time_up_hard():
                return
            empty_bin(ns, b, max_moves=120)

        # 2) short global improvement: a few random relocates to reduce penalties
        active_bins = [bid for bid, a in enumerate(ns.active) if a]
        if len(active_bins) <= 1:
            return
        for _ in range(80):
            if time_up():
                return
            b = random.choice(active_bins)
            if not ns.active[b] or not ns.bin_items[b]:
                continue
            it = random.choice(ns.bin_items[b])
            w = weights[it]
            cand = ns.sample_feasible_bins_tight(w, exclude=b, limit=12)
            if not cand:
                continue
            # accept if improves secondary/tertiary without increasing bin count
            base_obj = ns.obj()
            for dst in cand:
                if dst == b or not ns.active[dst]:
                    continue
                if ns.bin_w[dst] + w > C:
                    continue
                # compute delta on (sum_pen, sum_s2)
                d1 = evaluate_insertion_cost(ns, it, dst)
                # removal delta from src
                old_w = ns.bin_w[b]
                new_w = old_w - w
                old_sl = C - old_w
                new_sl = C - new_w
                dpen_src = slack_penalty(new_sl) - slack_penalty(old_sl)
                ds2_src = new_sl * new_sl - old_sl * old_sl
                new_pen = ns.sum_pen + d1[0] + dpen_src
                new_s2 = ns.sum_slack2 + d1[1] + ds2_src
                if (ns.bins_used, new_pen, new_s2) <= base_obj:
                    ns.move_item(it, dst)
                    break

    # -------- Initialization: multi-start constructive + post emptying --------
    init_deadline = start + 0.18 * (deadline - start)
    best = None
    best_obj = None

    methods = ["bfd", "ffd", "abf"]
    tries = 0
    while tries < 40 and time.perf_counter() < init_deadline:
        tries += 1
        m = methods[tries % len(methods)]
        st = construct(m)
        # quick bin-empty attempt on a couple light bins
        vnd(st, attempts=2)
        o = st.obj()
        if best is None or o < best_obj:
            best = st.clone_compact()
            best_obj = o

    if best is None:
        best = construct("bfd")
        best_obj = best.obj()

    current = best.clone_compact()

    # -------- Main VNS loop with adaptive k --------
    MAX_OUTER = 200000  # fixed iteration cap
    k_max = 12
    no_bin_improve = 0

    for outer in range(MAX_OUTER):
        if (outer & 127) == 0 and time_up_hard():
            break

        base = current
        # bounded inner steps (fixed per outer, but k_max adaptive)
        k = 1
        inner_steps = 0
        inner_step_cap = 28

        while k <= k_max and inner_steps < inner_step_cap and not time_up_hard():
            inner_steps += 1
            ns = shake(base, k)
            vnd(ns, attempts=3)

            o = ns.obj()
            if o < best_obj:
                best = ns.clone_compact()
                best_obj = o
                # improvement event
                if o[0] < current.obj()[0]:
                    no_bin_improve = 0

            # Acceptance focusing on bin count
            cur_o = current.obj()
            accept = False
            if o[0] < cur_o[0]:
                accept = True
            elif o[0] == cur_o[0]:
                # accept if better secondary, or with small probability if slightly worse
                if (o[1], o[2]) < (cur_o[1], cur_o[2]):
                    accept = True
                else:
                    # SA-like mild acceptance to diversify
                    if random.random() < 0.08:
                        accept = True

            if accept:
                current = ns
                base = current
                k = 1
            else:
                k += 1

        # stagnation control & occasional restart-like heavy shake from best
        if current.obj()[0] > best_obj[0]:
            current = best.clone_compact()

        if best_obj[0] == current.obj()[0]:
            no_bin_improve += 1
        else:
            no_bin_improve = 0

        if (outer % 50) == 0:
            # adapt k_max based on stagnation
            if no_bin_improve > 200:
                k_max = min(40, k_max + 2)
            elif no_bin_improve == 0:
                k_max = max(10, k_max - 1)

        if (outer % 300) == 0 and no_bin_improve > 350 and not time_up_hard():
            # heavy diversification: bin-ruin shake from best
            current = shake(best, k=min(k_max, 20))
            vnd(current, attempts=3)

    # -------- Final compaction for output --------
    final = best.clone_compact()
    packing: List[List[int]] = []
    bin_weights: List[int] = []
    for bid, act in enumerate(final.active):
        if not act:
            continue
        items = final.bin_items[bid]
        if not items:
            continue
        packing.append(items[:])
        bin_weights.append(final.bin_w[bid])

    # Safety: ensure each item appears exactly once
    # (If something went wrong due to rare chain failure, repair by reconstructing)
    seen = [0] * n
    for b in packing:
        for it in b:
            if 0 <= it < n:
                seen[it] += 1
    if any(x != 1 for x in seen):
        # fallback: simple BFD rebuild from current best ordering
        st = construct("bfd")
        st = st.clone_compact()
        packing = []
        bin_weights = []
        for bid, act in enumerate(st.active):
            if act and st.bin_items[bid]:
                packing.append(st.bin_items[bid][:])
                bin_weights.append(st.bin_w[bid])

    return {"packing": packing, "bin_weights": bin_weights}
