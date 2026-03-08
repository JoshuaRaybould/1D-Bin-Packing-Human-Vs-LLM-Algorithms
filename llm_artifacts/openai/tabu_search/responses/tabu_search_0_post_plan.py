import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)

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
        # caller ensures bins[b] is empty
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
        # O(1) swap-with-last removal
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
    # Initial solution generation
    # ----------------------------
    def pack_greedy(order: List[int], mode: str) -> Tuple[List[List[int]], List[int]]:
        # mode in {"FFD", "BFD"}
        bins: List[List[int]] = []
        loads: List[int] = []
        for i in order:
            wi = w[i]
            chosen = -1
            if mode == "FFD":
                for b in range(len(bins)):
                    if loads[b] + wi <= C:
                        chosen = b
                        break
            else:  # BFD
                best_rem = C + 1
                for b in range(len(bins)):
                    rem = C - loads[b]
                    if wi <= rem:
                        nrem = rem - wi
                        if nrem < best_rem:
                            best_rem = nrem
                            chosen = b
                            if best_rem == 0:
                                break
            if chosen == -1:
                bins.append([i])
                loads.append(wi)
            else:
                bins[chosen].append(i)
                loads[chosen] += wi
        return bins, loads

    def build_struct_from_bins(bins0: List[List[int]]) -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], List[int], List[int]]:
        bins, loads, assign, pos, active_bins, active_pos, empty_bins = make_empty_solution_structs(len(bins0))
        # compute loads and fill
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

    def packing_score_bins_only(bins0: List[List[int]]) -> int:
        return sum(1 for items in bins0 if items)

    # Multi-start initialization within small time slice
    order_base = sorted(range(n), key=lambda i: w[i], reverse=True)
    init_deadline = min(time_limit * 0.03, 0.25)  # small fixed-ish cap

    best_init_bins: Optional[List[List[int]]] = None
    best_init_m = 10**9

    # Deterministic FFD/BFD
    for mode in ("BFD", "FFD"):
        b0, _l0 = pack_greedy(order_base, mode)
        m0 = packing_score_bins_only(b0)
        if m0 < best_init_m:
            best_init_m = m0
            best_init_bins = b0

    # Randomized tie-breaking / perturbations
    # Shuffle within small windows to create different packings
    tries = 0
    while elapsed() < init_deadline and tries < 30:
        tries += 1
        order = order_base[:]
        # shuffle within blocks (preserves near-decreasing)
        block = 8
        for s in range(0, n, block):
            t = min(n, s + block)
            if t - s > 1:
                # slight chance to shuffle block
                if random.random() < 0.7:
                    sub = order[s:t]
                    random.shuffle(sub)
                    order[s:t] = sub
        mode = "BFD" if random.random() < 0.7 else "FFD"
        b0, _l0 = pack_greedy(order, mode)
        m0 = packing_score_bins_only(b0)
        if m0 < best_init_m:
            best_init_m = m0
            best_init_bins = b0

    if best_init_bins is None:
        best_init_bins, _ = pack_greedy(order_base, "BFD")

    bins, loads, assign, pos, active_bins, active_pos, empty_bins = build_struct_from_bins(best_init_bins)

    # ----------------------------
    # Incremental aggregates
    # ----------------------------
    # total_slack2 across active bins (empty bins contribute 0, but if kept around, ignore)
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
    # Tabu + frequency memory
    # ----------------------------
    # tabu_until[(item, bin)] = iteration index when allowed again.
    tabu_until: Dict[Tuple[int, int], int] = {}
    # lightweight frequency: count per (item, bin)
    freq: Dict[Tuple[int, int], int] = {}

    def freq_penalty(i: int, b: int) -> int:
        return freq.get((i, b), 0)

    def is_tabu(i: int, b: int, it: int) -> bool:
        return tabu_until.get((i, b), -1) > it

    # Adaptive tenure
    default_base_tenure = 7
    base_tenure = default_base_tenure
    tenure_span = 10
    max_base_tenure = 45

    def set_tabu(i: int, b: int, it: int) -> None:
        nonlocal base_tenure
        tenure = base_tenure + random.randint(0, tenure_span)
        tabu_until[(i, b)] = it + tenure

    def bump_freq(i: int, b: int) -> None:
        k = (i, b)
        freq[k] = freq.get(k, 0) + 1

    # Periodic tabu purge
    def purge_tabu(it: int) -> None:
        if not tabu_until:
            return
        # delete expired
        dead = [k for k, v in tabu_until.items() if v <= it]
        for k in dead:
            del tabu_until[k]

    # ----------------------------
    # Phase / target bin selection
    # ----------------------------
    target_bin: Optional[int] = None
    phase_start_it = 0

    def pick_target_bin() -> Optional[int]:
        # choose among lightest active bins
        if not active_bins:
            return None
        # sample a few light bins quickly
        # exact min is fine: active_bins size is number of bins (small)
        bestb = None
        bestload = 10**18
        for b in active_bins:
            lb = loads[b]
            if lb < bestload:
                bestload = lb
                bestb = b
        return bestb

    def current_R() -> int:
        if target_bin is None:
            return 0
        return loads[target_bin]

    def score_tuple(m: int, R: int, slack2: int) -> Tuple[int, int, int]:
        return (m, R, slack2)

    # Initialize target
    target_bin = pick_target_bin()

    # ----------------------------
    # Candidate helpers
    # ----------------------------
    def sample_bins(k: int) -> List[int]:
        if k >= len(active_bins):
            return active_bins[:]
        # random sample without importing extra libs
        res = []
        for _ in range(k):
            res.append(active_bins[random.randrange(len(active_bins))])
        return res

    def best_fit_dest_for_item(i: int, exclude_bin: int, sample_k: int, extra_heavy_k: int = 3) -> List[int]:
        # returns a small ordered list of destination bins (best-fit among sampled)
        wi = w[i]
        candidates = []
        # sample bins
        for b in sample_bins(sample_k):
            if b == exclude_bin:
                continue
            rem = C - loads[b]
            if rem >= wi:
                candidates.append((rem - wi, b))
        # also include a few most loaded bins (tight packing tends to help)
        if extra_heavy_k > 0 and len(active_bins) > 0:
            # find heavy bins by scanning
            heavy = sorted(active_bins, key=lambda b: loads[b], reverse=True)
            for b in heavy[:extra_heavy_k]:
                if b == exclude_bin:
                    continue
                rem = C - loads[b]
                if rem >= wi:
                    candidates.append((rem - wi, b))
        if not candidates:
            return []
        candidates.sort()
        # return top few unique
        out = []
        seen = set()
        for _, b in candidates:
            if b not in seen:
                seen.add(b)
                out.append(b)
                if len(out) >= 5:
                    break
        return out

    def top_items_in_bin(b: int, k: int) -> List[int]:
        items = bins[b]
        if len(items) <= k:
            return sorted(items, key=lambda i: w[i], reverse=True)
        # partial selection: sample and then sort; bin sizes are usually small
        cand = items[:]
        cand.sort(key=lambda i: w[i], reverse=True)
        return cand[:k]

    # ----------------------------
    # Move evaluation (incremental)
    # ----------------------------
    # All eval functions return:
    #   new_m, new_R, new_slack2, penalty, and feasibility
    # plus the tabu-attribute (item, forbidden_bin) decisions happen outside.

    freq_lambda = 1  # small tertiary penalty weight

    def eval_relocate(i: int, src: int, dst: int, cur_m: int, cur_slack2: int, cur_R: int) -> Optional[Tuple[Tuple[int, int, int], int]]:
        # returns (score_tuple, new_slack2)
        wi = w[i]
        if src == dst:
            return None
        if loads[dst] + wi > C:
            return None

        new_m = cur_m
        # slack2 delta for src, dst; if src becomes empty: remove its contribution
        s2 = cur_slack2
        s2 -= slack2_of_load(loads[src])
        s2 -= slack2_of_load(loads[dst])
        new_load_src = loads[src] - wi
        new_load_dst = loads[dst] + wi
        if len(bins[src]) == 1:
            # bin becomes empty => it will be deactivated, no slack2 contribution
            new_m -= 1
            s2 += slack2_of_load(new_load_dst)
        else:
            s2 += slack2_of_load(new_load_src)
            s2 += slack2_of_load(new_load_dst)

        # target-bin residual
        if target_bin is None:
            new_R = 0
        else:
            if src == target_bin:
                new_R = loads[src] - wi
                if len(bins[src]) == 1:
                    new_R = 0
            elif dst == target_bin:
                new_R = loads[dst] + wi
            else:
                new_R = cur_R

        # frequency penalty embedded into slack2 as small add-on
        pen = freq_lambda * freq_penalty(i, dst)
        return (score_tuple(new_m, new_R, s2 + pen), s2)

    def eval_swap(a: int, bitem: int, ba: int, bb: int, cur_m: int, cur_slack2: int, cur_R: int) -> Optional[Tuple[Tuple[int, int, int], int]]:
        if ba == bb:
            return None
        wa, wb = w[a], w[bitem]
        la = loads[ba]
        lb = loads[bb]
        if la - wa + wb > C:
            return None
        if lb - wb + wa > C:
            return None

        s2 = cur_slack2
        s2 -= slack2_of_load(la)
        s2 -= slack2_of_load(lb)
        nla = la - wa + wb
        nlb = lb - wb + wa
        s2 += slack2_of_load(nla)
        s2 += slack2_of_load(nlb)

        # update R if target involved
        if target_bin is None:
            new_R = 0
        else:
            new_R = cur_R
            if ba == target_bin:
                new_R = loads[ba] - wa + wb
            if bb == target_bin:
                new_R = loads[bb] - wb + wa

        pen = freq_lambda * (freq_penalty(a, bb) + freq_penalty(bitem, ba))
        return (score_tuple(cur_m, new_R, s2 + pen), s2)

    # ----------------------------
    # Move application (updates slack2, active sets)
    # ----------------------------
    def apply_relocate(i: int, src: int, dst: int) -> None:
        nonlocal total_slack2
        # update slack2 removing old contributions
        total_slack2 -= slack2_of_load(loads[src])
        total_slack2 -= slack2_of_load(loads[dst])

        remove_item_from_bin(i, src, bins, loads, assign, pos, active_bins, active_pos, empty_bins)
        add_item_to_bin(i, dst, bins, loads, assign, pos, active_bins, active_pos)

        # add new contributions
        if active_pos[src] != -1:
            total_slack2 += slack2_of_load(loads[src])
        total_slack2 += slack2_of_load(loads[dst])

    def apply_swap(a: int, bitem: int, ba: int, bb: int) -> None:
        nonlocal total_slack2
        # remove slack2 contributions
        total_slack2 -= slack2_of_load(loads[ba])
        total_slack2 -= slack2_of_load(loads[bb])

        # swap in-place using positions
        pa = pos[a]
        pb = pos[bitem]

        bins[ba][pa] = bitem
        pos[bitem] = pa
        assign[bitem] = ba

        bins[bb][pb] = a
        pos[a] = pb
        assign[a] = bb

        loads[ba] += w[bitem] - w[a]
        loads[bb] += w[a] - w[bitem]

        # add slack2
        total_slack2 += slack2_of_load(loads[ba])
        total_slack2 += slack2_of_load(loads[bb])

    # ----------------------------
    # Pair relocate out of target (2-0)
    # ----------------------------
    def try_pair_relocate_from_target(it: int, cur_m: int, cur_R: int) -> bool:
        # Attempt to move two items from target into (possibly same) other bins.
        # Apply best found improving (m,R,slack2) within limited trials.
        if target_bin is None or active_pos[target_bin] == -1:
            return False
        t = target_bin
        items = top_items_in_bin(t, 6)
        if len(items) < 2:
            return False

        best_plan = None
        best_sc = None

        # few combinations biased: large+medium
        combs = []
        for i in range(min(len(items), 4)):
            for j in range(i + 1, min(len(items), 6)):
                combs.append((items[i], items[j]))
        random.shuffle(combs)
        combs = combs[:20]

        for (i1, i2) in combs:
            if elapsed() >= time_limit:
                break
            wi1, wi2 = w[i1], w[i2]
            # candidates for i1
            d1s = best_fit_dest_for_item(i1, t, sample_k=24, extra_heavy_k=3)
            if not d1s:
                continue
            for d1 in d1s[:3]:
                # temporarily consider loads after placing i1
                l_d1 = loads[d1] + wi1
                if l_d1 > C:
                    continue
                # candidates for i2 (allow same bin if fits)
                # if same, check with updated load
                d2s = best_fit_dest_for_item(i2, t, sample_k=24, extra_heavy_k=3)
                # also allow d1 even if not in list
                if d1 != t:
                    d2s = ([d1] + d2s)
                seen2 = set()
                for d2 in d2s:
                    if d2 == t or d2 in seen2:
                        continue
                    seen2.add(d2)
                    if d2 == d1:
                        if l_d1 + wi2 > C:
                            continue
                    else:
                        if loads[d2] + wi2 > C:
                            continue

                    # Evaluate compound move approximately by sequential deltas on slack2 and R
                    # Compute new_m (may drop if target empties): only possible if target has exactly 2 items
                    new_m = cur_m
                    target_size = len(bins[t])
                    if target_size == 2:
                        new_m = cur_m - 1

                    # slack2 delta: remove old of t,d1,d2; add new.
                    s2 = total_slack2
                    s2 -= slack2_of_load(loads[t])
                    s2 -= slack2_of_load(loads[d1])
                    if d2 != d1:
                        s2 -= slack2_of_load(loads[d2])

                    new_load_t = loads[t] - wi1 - wi2
                    new_load_d1 = loads[d1] + wi1
                    if d2 == d1:
                        new_load_d1 += wi2
                        # t might become empty
                        if target_size != 2:
                            s2 += slack2_of_load(new_load_t)
                        s2 += slack2_of_load(new_load_d1)
                    else:
                        new_load_d2 = loads[d2] + wi2
                        if target_size != 2:
                            s2 += slack2_of_load(new_load_t)
                        s2 += slack2_of_load(new_load_d1)
                        s2 += slack2_of_load(new_load_d2)

                    new_R = new_load_t
                    if target_size == 2:
                        new_R = 0

                    pen = freq_lambda * (freq_penalty(i1, d1) + freq_penalty(i2, d2))
                    sc = score_tuple(new_m, new_R, s2 + pen)

                    # tabu check / aspiration:
                    # forbid returning items to their previous bins; here src is target, so tabu is (i, target)
                    # but we are moving out, so check tabu on destination assignment.
                    if (is_tabu(i1, d1, it) or is_tabu(i2, d2, it)) and not (new_m < best_m):
                        continue

                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_plan = (i1, d1, i2, d2)

        if best_plan is None:
            return False

        i1, d1, i2, d2 = best_plan
        # Apply: move heavier first to reduce feasibility surprises
        if w[i2] > w[i1]:
            i1, i2 = i2, i1
            d1, d2 = d2, d1

        # Ensure still in target
        if assign[i1] != target_bin or assign[i2] != target_bin:
            return False

        # apply
        apply_relocate(i1, target_bin, d1)
        set_tabu(i1, target_bin, it)  # prevent immediate return
        bump_freq(i1, d1)

        # target bin might have changed content/emptied
        src2 = assign[i2]
        if src2 != -1:
            # src2 should still be target (unless target emptied by first move when size==1, but we had 2 items or more)
            if src2 == target_bin and (d2 == d1):
                if loads[d1] + w[i2] <= C:
                    apply_relocate(i2, target_bin, d2)
                    set_tabu(i2, target_bin, it)
                    bump_freq(i2, d2)
            elif src2 == target_bin:
                if loads[d2] + w[i2] <= C:
                    apply_relocate(i2, target_bin, d2)
                    set_tabu(i2, target_bin, it)
                    bump_freq(i2, d2)
        return True

    # ----------------------------
    # Ejection chain lite (depth 2)
    # ----------------------------
    def try_ejection_chain_from_target(it: int) -> bool:
        # Attempt: move i from target -> bin b (requires ejecting one item e from b) then place e elsewhere.
        if target_bin is None or active_pos[target_bin] == -1:
            return False
        t = target_bin
        if not bins[t]:
            return False

        # focus on a few largest items in target
        cand_items = top_items_in_bin(t, 4)
        # pick some destination bins that are close-to-feasible
        sampled = sample_bins(32)
        random.shuffle(sampled)

        cur_m = m_active()
        cur_R = loads[t]

        best_sc = None
        best_plan = None  # (i, b, e, dst_e)

        for i in cand_items:
            wi = w[i]
            for b in sampled[:24]:
                if b == t:
                    continue
                rem = C - loads[b]
                if rem >= wi:
                    continue  # direct relocate should handle; ejection reserved for hard cases
                over = wi - rem
                # only consider small overfill
                if over <= 0 or over > C // 3:
                    continue
                # choose an ejection item e from b: smallest item with weight >= over (to make feasible)
                e = None
                best_we = 10**9
                for cand in bins[b]:
                    wc = w[cand]
                    if wc >= over and wc < best_we:
                        best_we = wc
                        e = cand
                if e is None:
                    continue

                # after ejecting e, i fits into b
                # now relocate e to somewhere else
                dests_e = best_fit_dest_for_item(e, exclude_bin=b, sample_k=28, extra_heavy_k=3)
                if not dests_e:
                    continue

                for dst_e in dests_e[:3]:
                    if dst_e == t:
                        # allow placing into target only if it helps R (usually hurts); skip
                        continue
                    # check feasibility after b receives i and ejects e (b load decreases by we then increases by wi)
                    if loads[dst_e] + w[e] > C:
                        continue

                    # tabu check
                    if (is_tabu(i, b, it) or is_tabu(e, dst_e, it)) and not (cur_m - (1 if len(bins[t]) == 1 else 0) < best_m):
                        continue

                    # Evaluate compound slack2 quickly
                    s2 = total_slack2
                    # bins involved: t, b, dst_e (dst_e might equal t? excluded)
                    s2 -= slack2_of_load(loads[t])
                    s2 -= slack2_of_load(loads[b])
                    s2 -= slack2_of_load(loads[dst_e])

                    new_m = cur_m
                    # t after removing i
                    new_load_t = loads[t] - wi
                    if len(bins[t]) == 1:
                        new_m -= 1
                    else:
                        s2 += slack2_of_load(new_load_t)

                    # b: remove e, add i
                    new_load_b = loads[b] - w[e] + wi
                    s2 += slack2_of_load(new_load_b)

                    # dst_e: add e
                    new_load_dst = loads[dst_e] + w[e]
                    s2 += slack2_of_load(new_load_dst)

                    new_R = new_load_t if (t == target_bin and len(bins[t]) != 1) else (0 if len(bins[t]) == 1 else cur_R)
                    if t == target_bin:
                        if len(bins[t]) == 1:
                            new_R = 0
                        else:
                            new_R = new_load_t

                    pen = freq_lambda * (freq_penalty(i, b) + freq_penalty(e, dst_e))
                    sc = score_tuple(new_m, new_R, s2 + pen)

                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_plan = (i, b, e, dst_e)

        if best_plan is None:
            return False

        i, b, e, dst_e = best_plan
        # Apply compound move: eject e from b -> dst_e, then move i from t -> b
        # First eject e
        apply_relocate(e, b, dst_e)
        set_tabu(e, b, it)
        bump_freq(e, dst_e)
        # Then move i
        # ensure i still in target
        if target_bin is None or assign[i] != target_bin:
            return True
        # b now has space
        if loads[b] + w[i] <= C:
            apply_relocate(i, target_bin, b)
            set_tabu(i, target_bin, it)
            bump_freq(i, b)
        return True

    # ----------------------------
    # Bin elimination trial (rare)
    # ----------------------------
    def try_eliminate_bin_trial(it: int) -> bool:
        nonlocal total_slack2
        # pick a very light bin and try to reinsert its items greedily into others (with one-step ejection)
        if not active_bins or len(active_bins) <= 1:
            return False
        # choose lightest bin
        b = pick_target_bin()
        if b is None or active_pos[b] == -1:
            return False
        if loads[b] == 0:
            return False
        # only attempt if reasonably light
        if loads[b] > C * 0.55:
            return False

        items = bins[b][:]
        # save state snapshot for bins and loads and assignments of involved items only
        involved_bins = set(active_bins)
        # snapshot full state is expensive; but bin counts are small, so accept copying sometimes
        snap_bins = [lst[:] for lst in bins]
        snap_loads = loads[:]
        snap_assign = assign[:]
        snap_pos = pos[:]
        snap_active_bins = active_bins[:]
        snap_active_pos = active_pos[:]
        snap_empty_bins = empty_bins[:]
        snap_slack2 = total_slack2

        # remove all items from b
        for i in items:
            if assign[i] == b:
                remove_item_from_bin(i, b, bins, loads, assign, pos, active_bins, active_pos, empty_bins)
        # recompute slack2 accurately after bulk removals (safe)
        total_slack2 = 0
        for bb in active_bins:
            total_slack2 += slack2_of_load(loads[bb])

        # try reinsert
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
                total_slack2 = snap_slack2
                return False

            dests = best_fit_dest_for_item(i, exclude_bin=-1, sample_k=min(40, max(10, len(active_bins))), extra_heavy_k=5)
            placed = False
            for dst in dests:
                if loads[dst] + w[i] <= C:
                    add_item_to_bin(i, dst, bins, loads, assign, pos, active_bins, active_pos)
                    placed = True
                    break
            if not placed:
                # one-step ejection attempt
                ok = False
                for dst in sample_bins(min(30, len(active_bins))):
                    rem = C - loads[dst]
                    if rem >= w[i]:
                        add_item_to_bin(i, dst, bins, loads, assign, pos, active_bins, active_pos)
                        ok = True
                        break
                    over = w[i] - rem
                    if over <= 0 or over > C // 3:
                        continue
                    # choose ejection
                    e = None
                    best_we = 10**9
                    for cand in bins[dst]:
                        wc = w[cand]
                        if wc >= over and wc < best_we:
                            best_we = wc
                            e = cand
                    if e is None:
                        continue
                    # find place for e
                    dests_e = best_fit_dest_for_item(e, exclude_bin=dst, sample_k=35, extra_heavy_k=5)
                    for dst_e in dests_e:
                        if loads[dst_e] + w[e] <= C:
                            # apply eject then insert i
                            apply_relocate(e, dst, dst_e)
                            add_item_to_bin(i, dst, bins, loads, assign, pos, active_bins, active_pos)
                            ok = True
                            break
                    if ok:
                        break
                if not ok:
                    # revert
                    bins[:] = snap_bins
                    loads[:] = snap_loads
                    assign[:] = snap_assign
                    pos[:] = snap_pos
                    active_bins[:] = snap_active_bins
                    active_pos[:] = snap_active_pos
                    empty_bins[:] = snap_empty_bins
                    total_slack2 = snap_slack2
                    return False

        # success: recompute slack2 precisely
        total_slack2 = 0
        for bb in active_bins:
            total_slack2 += slack2_of_load(loads[bb])

        # mark some tabu to prevent immediate undo: items that were in b cannot go back to b
        for i in items:
            set_tabu(i, b, it)

        return True

    # ----------------------------
    # Main tabu search loop
    # ----------------------------
    # Large iters, time-controlled
    max_iters = 5_000_000
    check_mask = 0xFF

    # Phase length
    phase_iters = min(5000, 500 + 20 * n)

    # Candidate counts (cheap eval now)
    base_reloc_tries = 600
    base_swap_tries = 250

    last_best_it = 0

    it = 0
    while it < max_iters:
        it += 1

        if (it & check_mask) == 0 and elapsed() >= time_limit:
            break

        if (it & 0x7FF) == 0:
            purge_tabu(it)

        # phase control
        if target_bin is None or active_pos[target_bin] == -1 or (it - phase_start_it) >= phase_iters:
            target_bin = pick_target_bin()
            phase_start_it = it

        cur_m = m_active()
        cur_R = loads[target_bin] if (target_bin is not None and active_pos[target_bin] != -1) else 0
        cur_sc = score_tuple(cur_m, cur_R, total_slack2)

        # occasional bin elimination trial
        if (it & 0x3FF) == 0 and elapsed() < time_limit:
            if try_eliminate_bin_trial(it):
                # after big move, retarget
                target_bin = pick_target_bin()
                phase_start_it = it
                cur_m = m_active()
                if cur_m < best_m:
                    best_m = cur_m
                    best_total_slack2 = total_slack2
                    best_bins_snapshot = [lst[:] for lst in bins]
                    best_loads_snapshot = loads[:]
                    last_best_it = it
                continue

        # adaptive tenure
        if it - last_best_it > max(400, 15 * n):
            base_tenure = min(max_base_tenure, base_tenure + 1)
        elif it - last_best_it < max(80, 3 * n):
            if base_tenure > default_base_tenure:
                base_tenure -= 1

        # neighborhood sizing (slightly larger early)
        progress = elapsed() / max(1e-9, time_limit)
        reloc_tries = base_reloc_tries + (300 if progress < 0.35 else 0)
        swap_tries = base_swap_tries + (150 if progress < 0.35 else 0)

        best_move = None  # (type, data...)
        best_move_sc = None
        best_move_new_slack2 = None

        # 1) Targeted relocate from target bin
        if target_bin is not None and active_pos[target_bin] != -1 and bins[target_bin]:
            t = target_bin
            cand_items = top_items_in_bin(t, 6)
            # try many best-fit placements
            for _ in range(reloc_tries):
                if not cand_items:
                    break
                i = cand_items[random.randrange(len(cand_items))]
                if assign[i] != t:
                    continue
                dests = best_fit_dest_for_item(i, t, sample_k=30, extra_heavy_k=4)
                if not dests:
                    continue
                # pick best among first few
                for dst in dests[:3]:
                    ev = eval_relocate(i, t, dst, cur_m, total_slack2, cur_R)
                    if ev is None:
                        continue
                    sc, new_s2 = ev
                    # tabu: prevent moving i back to t; attribute is (i, t)
                    # and also tabu on (i, dst) to prevent immediate reversal patterns
                    tabu = is_tabu(i, t, it) or is_tabu(i, dst, it)
                    # aspiration: if reduces best bin count, allow
                    if tabu and not (sc[0] < best_m):
                        # also allow if strongly reduces R
                        if not (sc[1] + min(3, w[i]) < cur_R):
                            continue
                    if best_move_sc is None or sc < best_move_sc:
                        best_move_sc = sc
                        best_move_new_slack2 = new_s2
                        best_move = ("rel", i, t, dst)

        # 2) Targeted swap: item in target with item outside
        if target_bin is not None and active_pos[target_bin] != -1 and len(active_bins) > 1 and bins[target_bin]:
            t = target_bin
            cand_t = top_items_in_bin(t, 5)
            for _ in range(swap_tries):
                if not cand_t:
                    break
                a = cand_t[random.randrange(len(cand_t))]
                if assign[a] != t:
                    continue
                bb = active_bins[random.randrange(len(active_bins))]
                if bb == t or not bins[bb]:
                    continue
                # choose a small item from bb to create room
                # sample a few and take the smallest
                items_bb = bins[bb]
                bitem = items_bb[random.randrange(len(items_bb))]
                if len(items_bb) > 2 and random.random() < 0.7:
                    # bias to smaller in bb
                    cand = [items_bb[random.randrange(len(items_bb))] for _ in range(3)]
                    bitem = min(cand, key=lambda x: w[x])
                if a == bitem:
                    continue
                ev = eval_swap(a, bitem, t, bb, cur_m, total_slack2, cur_R)
                if ev is None:
                    continue
                sc, new_s2 = ev
                tabu = is_tabu(a, t, it) or is_tabu(bitem, bb, it)
                if tabu and not (sc[0] < best_m):
                    continue
                if best_move_sc is None or sc < best_move_sc:
                    best_move_sc = sc
                    best_move_new_slack2 = new_s2
                    best_move = ("sw", a, bitem, t, bb)

        # 3) If not finding progress, attempt pair relocate or ejection chain from target
        if best_move is None and target_bin is not None and active_pos[target_bin] != -1 and bins[target_bin]:
            # try pair relocate
            if try_pair_relocate_from_target(it, cur_m, cur_R):
                # update best
                cur_m2 = m_active()
                if cur_m2 < best_m or (cur_m2 == best_m and total_slack2 < best_total_slack2):
                    best_m = cur_m2
                    best_total_slack2 = total_slack2
                    best_bins_snapshot = [lst[:] for lst in bins]
                    best_loads_snapshot = loads[:]
                    last_best_it = it
                continue
            # try ejection chain
            if try_ejection_chain_from_target(it):
                cur_m2 = m_active()
                if cur_m2 < best_m or (cur_m2 == best_m and total_slack2 < best_total_slack2):
                    best_m = cur_m2
                    best_total_slack2 = total_slack2
                    best_bins_snapshot = [lst[:] for lst in bins]
                    best_loads_snapshot = loads[:]
                    last_best_it = it
                continue

        # 4) Small global relocate for diversification (when still no move)
        if best_move is None and len(active_bins) > 1:
            for _ in range(150):
                src = active_bins[random.randrange(len(active_bins))]
                if not bins[src]:
                    continue
                i = bins[src][random.randrange(len(bins[src]))]
                dests = best_fit_dest_for_item(i, src, sample_k=25, extra_heavy_k=3)
                if not dests:
                    continue
                dst = dests[0]
                ev = eval_relocate(i, src, dst, cur_m, total_slack2, cur_R)
                if ev is None:
                    continue
                sc, _new_s2 = ev
                tabu = is_tabu(i, src, it)
                if tabu and not (sc[0] < best_m):
                    continue
                best_move = ("rel", i, src, dst)
                best_move_sc = sc
                break

        # If still nothing, structured diversification: reshuffle a large item from a slacky bin
        if best_move is None:
            if len(active_bins) > 1:
                # pick bin with largest slack
                slacky = max(active_bins, key=lambda b: (C - loads[b]))
                if bins[slacky]:
                    # pick a large item there
                    i = max(bins[slacky], key=lambda x: w[x])
                    # try push into tight bins
                    dests = best_fit_dest_for_item(i, slacky, sample_k=40, extra_heavy_k=8)
                    for dst in dests[:3]:
                        if loads[dst] + w[i] <= C:
                            apply_relocate(i, slacky, dst)
                            set_tabu(i, slacky, it)
                            bump_freq(i, dst)
                            break
            continue

        # Apply chosen best move
        if best_move[0] == "rel":
            _, i, src, dst = best_move
            if assign[i] != src:
                continue
            if loads[dst] + w[i] > C:
                continue
            apply_relocate(i, src, dst)
            # tabu on returning to src
            set_tabu(i, src, it)
            bump_freq(i, dst)
        else:
            _, a, bitem, ba, bb = best_move
            if assign[a] != ba or assign[bitem] != bb:
                continue
            # feasibility recheck
            if loads[ba] - w[a] + w[bitem] > C or loads[bb] - w[bitem] + w[a] > C:
                continue
            apply_swap(a, bitem, ba, bb)
            set_tabu(a, ba, it)
            set_tabu(bitem, bb, it)
            bump_freq(a, bb)
            bump_freq(bitem, ba)

        # update best
        cur_m2 = m_active()
        # update target residual quickly
        if target_bin is None or active_pos[target_bin] == -1:
            cur_R2 = 0
        else:
            cur_R2 = loads[target_bin]

        # primary: bins
        if cur_m2 < best_m:
            best_m = cur_m2
            best_total_slack2 = total_slack2
            best_bins_snapshot = [lst[:] for lst in bins]
            best_loads_snapshot = loads[:]
            last_best_it = it
            # new target
            target_bin = pick_target_bin()
            phase_start_it = it
        elif cur_m2 == best_m and total_slack2 < best_total_slack2:
            best_total_slack2 = total_slack2
            best_bins_snapshot = [lst[:] for lst in bins]
            best_loads_snapshot = loads[:]
            last_best_it = it

    # ----------------------------
    # Build final packing from best snapshot
    # Compact bins (renumber) only at the end
    # ----------------------------
    final_bins: List[List[int]] = []
    final_loads: List[int] = []
    for b, items in enumerate(best_bins_snapshot):
        if items:
            final_bins.append(items[:])
            final_loads.append(int(sum(w[i] for i in items)))

    return {"packing": final_bins, "bin_weights": final_loads}
