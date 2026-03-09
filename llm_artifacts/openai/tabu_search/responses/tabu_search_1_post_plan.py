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
    # Representation utilities
    # -----------------------

    # State:
    # bins: List[List[int]]
    # loads: List[int]
    # assign: List[int]
    # pos_in_bin: List[int]
    # bin_ids: List[int] immutable ids aligned with bins/loads
    # slack_sq_sum: int

    def compute_slack_sq_sum(loads: List[int]) -> int:
        s = 0
        for ld in loads:
            d = C - ld
            s += d * d
        return s

    def better_obj(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        # lexicographic: fewer bins, then smaller slack
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

    def make_empty_state() -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], int, int]:
        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        pos_in_bin = [-1] * n
        bin_ids: List[int] = []
        slack_sq_sum = 0
        next_bin_id = 0
        return bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum, next_bin_id

    def add_bin(
        bins: List[List[int]], loads: List[int], bin_ids: List[int], slack_sq_sum: int, next_bin_id: int
    ) -> Tuple[int, int, int]:
        # adds empty bin and updates slack
        bins.append([])
        loads.append(0)
        bin_ids.append(next_bin_id)
        # slack term for empty bin
        slack_sq_sum += C * C
        return len(bins) - 1, slack_sq_sum, next_bin_id + 1

    def remove_item_from_bin(
        i: int,
        b: int,
        bins: List[List[int]],
        loads: List[int],
        assign: List[int],
        pos_in_bin: List[int],
    ) -> None:
        # O(1) swap-pop
        p = pos_in_bin[i]
        last = bins[b][-1]
        if last != i:
            bins[b][p] = last
            pos_in_bin[last] = p
        bins[b].pop()
        pos_in_bin[i] = -1
        assign[i] = -1
        loads[b] -= weights[i]

    def add_item_to_bin(
        i: int,
        b: int,
        bins: List[List[int]],
        loads: List[int],
        assign: List[int],
        pos_in_bin: List[int],
    ) -> None:
        pos_in_bin[i] = len(bins[b])
        bins[b].append(i)
        assign[i] = b
        loads[b] += weights[i]

    def maybe_remove_empty_bin(
        b: int,
        bins: List[List[int]],
        loads: List[int],
        assign: List[int],
        pos_in_bin: List[int],
        bin_ids: List[int],
        slack_sq_sum: int,
    ) -> Tuple[bool, int]:
        # If bin b is empty, remove it by swapping with last bin.
        # Returns (removed, new_slack_sq_sum)
        if bins[b]:
            return False, slack_sq_sum

        last = len(bins) - 1
        # slack remove for empty bin is (C-0)^2 = C^2
        slack_sq_sum -= C * C

        if b != last:
            # swap arrays b <-> last
            bins[b], bins[last] = bins[last], bins[b]
            loads[b], loads[last] = loads[last], loads[b]
            bin_ids[b], bin_ids[last] = bin_ids[last], bin_ids[b]
            # Update assignments/positions for items moved to index b
            for idx, item in enumerate(bins[b]):
                assign[item] = b
                pos_in_bin[item] = idx

        bins.pop()
        loads.pop()
        bin_ids.pop()
        return True, slack_sq_sum

    def apply_reloc_move(
        i: int,
        b_from: int,
        b_to: int,
        bins: List[List[int]],
        loads: List[int],
        assign: List[int],
        pos_in_bin: List[int],
        bin_ids: List[int],
        slack_sq_sum: int,
    ) -> Tuple[int, bool]:
        # returns (new_slack_sq_sum, removed_bin)
        # update slack for affected bins
        lf_old = loads[b_from]
        lt_old = loads[b_to]
        slack_sq_sum -= (C - lf_old) ** 2
        slack_sq_sum -= (C - lt_old) ** 2

        remove_item_from_bin(i, b_from, bins, loads, assign, pos_in_bin)
        add_item_to_bin(i, b_to, bins, loads, assign, pos_in_bin)

        lf_new = loads[b_from]
        lt_new = loads[b_to]
        slack_sq_sum += (C - lf_new) ** 2
        slack_sq_sum += (C - lt_new) ** 2

        removed, slack_sq_sum = maybe_remove_empty_bin(
            b_from, bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum
        )
        return slack_sq_sum, removed

    def apply_swap_move(
        i: int,
        j: int,
        bi: int,
        bj: int,
        bins: List[List[int]],
        loads: List[int],
        assign: List[int],
        pos_in_bin: List[int],
        slack_sq_sum: int,
    ) -> int:
        # swap i and j between bins bi and bj using positions
        li_old = loads[bi]
        lj_old = loads[bj]
        slack_sq_sum -= (C - li_old) ** 2
        slack_sq_sum -= (C - lj_old) ** 2

        pi = pos_in_bin[i]
        pj = pos_in_bin[j]
        bins[bi][pi] = j
        bins[bj][pj] = i
        pos_in_bin[i] = pj
        pos_in_bin[j] = pi
        assign[i] = bj
        assign[j] = bi
        loads[bi] += weights[j] - weights[i]
        loads[bj] += weights[i] - weights[j]

        li_new = loads[bi]
        lj_new = loads[bj]
        slack_sq_sum += (C - li_new) ** 2
        slack_sq_sum += (C - lj_new) ** 2
        return slack_sq_sum

    # -----------------------
    # Construction heuristics
    # -----------------------

    def construct_packing(order: List[int]) -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], int, int]:
        bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum, next_bin_id = make_empty_state()

        for i in order:
            w = weights[i]
            # best-fit among existing
            best_b = -1
            best_rem = None
            for b in range(len(bins)):
                rem = C - loads[b]
                if rem >= w:
                    new_rem = rem - w
                    if best_rem is None or new_rem < best_rem:
                        best_rem = new_rem
                        best_b = b
            if best_b == -1:
                best_b, slack_sq_sum, next_bin_id = add_bin(bins, loads, bin_ids, slack_sq_sum, next_bin_id)
            # update slack for bin receiving item
            old = loads[best_b]
            slack_sq_sum -= (C - old) ** 2
            add_item_to_bin(i, best_b, bins, loads, assign, pos_in_bin)
            new = loads[best_b]
            slack_sq_sum += (C - new) ** 2

        return bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum, next_bin_id

    def multi_start_initial() -> Tuple[List[List[int]], List[int], List[int], List[int], List[int], int, int]:
        idx = list(range(n))
        # base deterministic decreasing
        base = sorted(idx, key=lambda i: (-weights[i], i))

        # determine number of starts quickly based on n and time
        # (fixed list; we still time-check)
        starts: List[List[int]] = []
        starts.append(base)

        # best-fit decreasing is same insertion rule here; diversify by order
        # randomized tie shuffles among equal weights
        if n <= 4000:
            for _ in range(6):
                if time_exceeded():
                    break
                # shuffle within equal weight groups
                groups: Dict[int, List[int]] = {}
                for i in idx:
                    groups.setdefault(weights[i], []).append(i)
                for g in groups.values():
                    random.shuffle(g)
                wkeys = sorted(groups.keys(), reverse=True)
                order = []
                for w in wkeys:
                    order.extend(groups[w])
                starts.append(order)

            # noisy sort key variant
            for _ in range(4):
                if time_exceeded():
                    break
                order = sorted(idx, key=lambda i: (-weights[i], random.random()))
                starts.append(order)

        best_state = None
        best_obj = None
        for order in starts:
            if time_exceeded():
                break
            st = construct_packing(order)
            m = len(st[0])
            obj = (m, st[5])
            if best_obj is None or better_obj(obj, best_obj):
                best_obj = obj
                best_state = st

        assert best_state is not None
        return best_state

    def greedy_pre_tighten(
        bins: List[List[int]],
        loads: List[int],
        assign: List[int],
        pos_in_bin: List[int],
        bin_ids: List[int],
        slack_sq_sum: int,
        max_attempts: int,
    ) -> int:
        # Attempt to empty the lightest bin by direct best-fit relocations only.
        for _ in range(max_attempts):
            if len(bins) <= 1:
                break
            # lightest bin
            b0 = min(range(len(bins)), key=lambda b: loads[b])
            if not bins[b0]:
                removed, slack_sq_sum = maybe_remove_empty_bin(b0, bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum)
                if not removed:
                    break
                continue

            moved_any = False
            # move largest-first from b0
            items0 = sorted(bins[b0], key=lambda i: weights[i], reverse=True)
            for i in items0:
                w = weights[i]
                best_b = -1
                best_rem = None
                for b in range(len(bins)):
                    if b == b0:
                        continue
                    rem = C - loads[b]
                    if rem >= w:
                        new_rem = rem - w
                        if best_rem is None or new_rem < best_rem:
                            best_rem = new_rem
                            best_b = b
                if best_b != -1:
                    slack_sq_sum, _ = apply_reloc_move(i, b0, best_b, bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum)
                    moved_any = True
                    if b0 >= len(bins):
                        # b0 removed; stop this tighten pass
                        break
            if not moved_any:
                break
        return slack_sq_sum

    # -----------------------
    # Tabu and aspiration
    # -----------------------

    def aspiration_ok(cand_obj: Tuple[int, int], best_obj: Tuple[int, int], best_slack_for_m: Dict[int, int]) -> bool:
        if better_obj(cand_obj, best_obj):
            return True
        m = cand_obj[0]
        bs = best_slack_for_m.get(m)
        return bs is None or cand_obj[1] < bs

    # -----------------------
    # Neighborhood helpers
    # -----------------------

    def pick_target_bins(loads: List[int], k_light: int) -> List[int]:
        m = len(loads)
        if m <= k_light:
            return list(range(m))
        # partial selection by sorting (k small)
        return sorted(range(m), key=lambda b: loads[b])[:k_light]

    def pick_dest_bins(loads: List[int], k_dest: int, extra_rand: int) -> List[int]:
        m = len(loads)
        if m == 0:
            return []
        rem_sorted = sorted(range(m), key=lambda b: (C - loads[b], loads[b]))
        D = rem_sorted[: min(k_dest, m)]
        # add a few random bins for diversification
        if m > len(D):
            for _ in range(extra_rand):
                D.append(random.randrange(m))
        # unique preserving order
        seen = set()
        out = []
        for b in D:
            if b not in seen:
                seen.add(b)
                out.append(b)
        return out

    def best_fit_r(dest_bins: List[int], loads: List[int], w: int, r: int, forbidden_bin: int) -> List[int]:
        # Return up to r bins that best-fit item of weight w among dest_bins.
        best: List[Tuple[int, int]] = []  # (rem_after, bin)
        for b in dest_bins:
            if b == forbidden_bin:
                continue
            rem = C - loads[b]
            if rem >= w:
                rem_after = rem - w
                if len(best) < r:
                    best.append((rem_after, b))
                    if len(best) == r:
                        best.sort()
                else:
                    if rem_after < best[-1][0]:
                        best[-1] = (rem_after, b)
                        best.sort()
        return [b for _, b in best]

    def tie_break_score(move_kind: str, b_from: int, load_from_before: int, load_from_after: int, target_set: set) -> int:
        # Lower is better.
        # Prefer reducing load of one of lightest target bins; prefer emptying.
        score = 0
        if b_from in target_set:
            score -= 100000
        # how much we reduced source load (more reduction is better)
        score -= (load_from_before - load_from_after) * 10
        # emptying is best
        if load_from_after == 0:
            score -= 1000000
        # swaps are slightly less direct for emptying
        if move_kind == "swap":
            score += 50
        return score

    # -----------------------
    # Ejection chain (depth 2)
    # -----------------------

    def find_depth2_chain(
        b0: int,
        target_bins: List[int],
        dest_bins: List[int],
        bins: List[List[int]],
        loads: List[int],
        assign: List[int],
        bin_ids: List[int],
        tabu_reloc: Dict[Tuple[int, int], int],
        it: int,
        tenure_block_best_obj: Tuple[int, int],
        best_slack_for_m: Dict[int, int],
        r_bestfit: int,
        deficit_bins_scan: int,
    ) -> Optional[List[Tuple[int, int, int]]]:
        # Attempt: move an item a from b0 to b1 (needs space). If not, eject x from b1 to b2 to make space.
        # Return a chain [(x, b1, b2), (a, b0, b1)] in that order, or None.
        if not bins[b0]:
            return None

        # hardest-first
        items0 = sorted(bins[b0], key=lambda i: weights[i], reverse=True)
        m = len(bins)
        for a in items0:
            w_a = weights[a]
            # direct fits first (if exists, not a chain)
            for b1 in best_fit_r(dest_bins, loads, w_a, r_bestfit, forbidden_bin=b0):
                # if direct possible, we don't return chain here
                # chain search used when direct fails; continue
                pass

            # find candidate b1 where it almost fits (small deficit)
            # scan a small set of bins (dest_bins) for smallest positive deficit
            best_def: List[Tuple[int, int]] = []  # (deficit, b1)
            for b1 in dest_bins:
                if b1 == b0:
                    continue
                rem1 = C - loads[b1]
                if rem1 < w_a:
                    deficit = w_a - rem1
                    # keep a few smallest deficits
                    if len(best_def) < deficit_bins_scan:
                        best_def.append((deficit, b1))
                        if len(best_def) == deficit_bins_scan:
                            best_def.sort()
                    else:
                        if deficit < best_def[-1][0]:
                            best_def[-1] = (deficit, b1)
                            best_def.sort()

            for deficit, b1 in best_def:
                # choose an item x in b1 with weight >= deficit (to free enough)
                # sample a few candidate x by taking heavier items
                if not bins[b1]:
                    continue
                # sort a small prefix of heaviest items
                items1 = sorted(bins[b1], key=lambda i: weights[i], reverse=True)
                for x in items1[: min(8, len(items1))]:
                    if weights[x] < deficit:
                        continue
                    # find destination b2 for x
                    w_x = weights[x]
                    b2_list = best_fit_r(dest_bins, loads, w_x, r_bestfit, forbidden_bin=b1)
                    if not b2_list:
                        continue
                    for b2 in b2_list:
                        if b2 == b0:
                            # allowed, but usually not helpful; skip for stability
                            continue

                        # Tabu checks for atomic moves (x -> b2) and (a -> b1)
                        # Use dest bin IDs for tabu keys
                        x_to_id = bin_ids[b2]
                        a_to_id = bin_ids[b1]
                        is_tabu1 = tabu_reloc.get((x, x_to_id), 0) > it
                        is_tabu2 = tabu_reloc.get((a, a_to_id), 0) > it

                        # compute candidate objective if applying chain
                        # simulate load changes only for b0, b1, b2
                        lf0 = loads[b0]
                        lf1 = loads[b1]
                        lf2 = loads[b2]

                        # after move x: b1 loses w_x, b2 gains w_x
                        lf1a = lf1 - w_x
                        lf2a = lf2 + w_x
                        # then move a: b0 loses w_a, b1 gains w_a
                        lf0b = lf0 - w_a
                        lf1b = lf1a + w_a

                        if lf2a > C or lf1b > C or lf0b < 0 or lf1a < 0:
                            continue

                        new_m = m - 1 if lf0b == 0 else m

                        # slack update delta
                        old_slack = (C - lf0) ** 2 + (C - lf1) ** 2 + (C - lf2) ** 2
                        new_slack = (C - lf0b) ** 2 + (C - lf1b) ** 2 + (C - lf2a) ** 2
                        cand_slack = slack_sq_sum_curr - old_slack + new_slack
                        cand_obj = (new_m, cand_slack)

                        if (is_tabu1 or is_tabu2) and (not aspiration_ok(cand_obj, tenure_block_best_obj, best_slack_for_m)):
                            continue

                        return [(x, b1, b2), (a, b0, b1)]

        return None

    # -----------------------
    # Elite pool
    # -----------------------

    def assignment_signature(assign: List[int], bin_ids: List[int]) -> int:
        # Hash of item -> bin_id mapping to detect cycles/repeats
        # (avoid Python tuple of length n)
        h = 1469598103934665603
        fnv_prime = 1099511628211
        for i, b in enumerate(assign):
            # b should be valid
            bid = bin_ids[b]
            x = (i + 1) * 1315423911 ^ (bid + 0x9e3779b9)
            h ^= x & 0xFFFFFFFFFFFFFFFF
            h = (h * fnv_prime) & 0xFFFFFFFFFFFFFFFF
        return h

    def hamming_distance_assign(a: List[int], b: List[int]) -> int:
        # approximate diversity measure
        d = 0
        for i in range(n):
            if a[i] != b[i]:
                d += 1
        return d

    # -----------------------
    # Initialize (multi-start)
    # -----------------------

    bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum, next_bin_id = multi_start_initial()

    # brief tighten
    slack_sq_sum = greedy_pre_tighten(
        bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum,
        max_attempts=10
    )

    best_bins = [b[:] for b in bins]
    best_loads = loads[:]
    best_assign = assign[:]
    best_bin_ids = bin_ids[:]
    best_obj = (len(bins), slack_sq_sum)

    best_slack_for_m: Dict[int, int] = {len(bins): slack_sq_sum}

    # Tabu structures using persistent bin_ids
    tabu_reloc: Dict[Tuple[int, int], int] = {}   # (item, dest_bin_id) -> expiry
    tabu_swap: Dict[Tuple[int, int], int] = {}    # (min(i,j), max(i,j)) -> expiry

    # adaptive tenure
    min_tenure = 5
    max_tenure = 40
    tenure = int(5 + 0.7 * (n ** 0.5))
    tenure = max(min_tenure, min(max_tenure, tenure))

    # cycle detection memory
    sig_window = 60
    recent_sigs: List[int] = []

    # long-term stats
    item_move_count = [0] * n
    bin_used_count: Dict[int, int] = {}

    # elite pool
    elite: List[Tuple[Tuple[int, int], List[List[int]], List[int], List[int], List[int], List[int], int]] = []
    # store: (obj, bins, loads, assign, pos_in_bin, bin_ids, slack)

    def push_elite(obj, bins, loads, assign, pos_in_bin, bin_ids, slack):
        nonlocal elite
        # keep up to 8; ensure diversity
        cand_assign = assign
        # reject near-duplicates
        for _, _, _, a2, _, _, _ in elite:
            if hamming_distance_assign(cand_assign, a2) < max(10, n // 50):
                return
        elite.append((obj, [b[:] for b in bins], loads[:], assign[:], pos_in_bin[:], bin_ids[:], slack))
        elite.sort(key=lambda x: x[0])
        if len(elite) > 8:
            elite.pop()

    push_elite(best_obj, bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum)

    # -----------------------
    # Iteration budgeting (fixed)
    # -----------------------

    max_iter = max(20000, 400 * n)

    # Phase scheduling
    phaseA_end = int(0.3 * max_iter)

    # Neighborhood parameters
    k_light_base = 12 if n < 200 else 20
    k_dest_base = 50 if n < 500 else 70
    r_bestfit = 5
    extra_rand_dest = 6

    swap_trials_per_item = 6

    chain_period = 120
    stagnation_threshold = max(600, 2 * n)

    iters_since_best = 0

    # For chain function: needs access to current slack
    slack_sq_sum_curr = slack_sq_sum

    # -----------------------
    # Main TS loop
    # -----------------------

    it = 0
    while it < max_iter:
        it += 1
        if it % 5 == 0 and time_exceeded():
            break

        m = len(bins)
        if m <= 1:
            break

        # update cycle memory and reactive tenure
        if it % 10 == 0:
            sig = assignment_signature(assign, bin_ids)
            recent_sigs.append(sig)
            if len(recent_sigs) > sig_window:
                recent_sigs.pop(0)
            repeats = recent_sigs.count(sig)
            if repeats >= 3:
                tenure = min(max_tenure, tenure + 2)
            else:
                tenure = max(min_tenure, tenure - 1 if tenure > min_tenure else tenure)

        # housekeeping: cap tabu size, purge some expired
        if it % 200 == 0:
            if len(tabu_reloc) > 20000:
                # purge expired
                for k in list(tabu_reloc.keys())[:5000]:
                    if tabu_reloc.get(k, 0) <= it:
                        tabu_reloc.pop(k, None)
            if len(tabu_swap) > 20000:
                for k in list(tabu_swap.keys())[:5000]:
                    if tabu_swap.get(k, 0) <= it:
                        tabu_swap.pop(k, None)

        # phase-dependent focus
        in_phaseA = it <= phaseA_end
        k_light = k_light_base if in_phaseA else k_light_base + 8
        k_dest = k_dest_base

        # deterministic candidates
        T = pick_target_bins(loads, min(k_light, m))
        Tset = set(T)
        D = pick_dest_bins(loads, min(k_dest, m), extra_rand_dest)

        # Build item list from T with bias to rarely moved items during diversification
        items_T: List[int] = []
        for b in T:
            items_T.extend(bins[b])
        if not items_T:
            continue

        if not in_phaseA and len(items_T) > 60:
            # pick 60 items biased to low move count
            items_T.sort(key=lambda i: item_move_count[i])
            # take a mix: mostly low-move, plus some random
            take = items_T[:45]
            take += random.sample(items_T[45:], min(15, len(items_T) - 45))
            items_T = take

        # Current objective
        slack_sq_sum_curr = slack_sq_sum
        curr_obj = (m, slack_sq_sum_curr)

        # Track best move
        best_move_kind = None
        best_move_data = None
        best_move_obj = None
        best_move_score = None

        # --------------
        # Relocate moves
        # --------------
        # Evaluate focused relocates from light bins.
        for i_item in items_T:
            b_from = assign[i_item]
            if b_from < 0:
                continue
            w = weights[i_item]
            # best-fit top-r among D
            dests = best_fit_r(D, loads, w, r_bestfit, forbidden_bin=b_from)
            if not dests:
                continue

            load_from_before = loads[b_from]
            for b_to in dests:
                # objective delta
                lf_old = loads[b_from]
                lt_old = loads[b_to]
                lf_new = lf_old - w
                lt_new = lt_old + w

                new_m = m - 1 if lf_new == 0 else m
                old_slack = (C - lf_old) ** 2 + (C - lt_old) ** 2
                new_slack = (C - lf_new) ** 2 + (C - lt_new) ** 2
                cand_slack = slack_sq_sum_curr - old_slack + new_slack
                cand_obj = (new_m, cand_slack)

                # tabu check
                dest_id = bin_ids[b_to]
                is_tabu = tabu_reloc.get((i_item, dest_id), 0) > it
                if is_tabu and not aspiration_ok(cand_obj, best_obj, best_slack_for_m):
                    continue

                score = tie_break_score("reloc", b_from, load_from_before, lf_new, Tset)

                if best_move_obj is None or better_obj(cand_obj, best_move_obj) or (
                    cand_obj == best_move_obj and score < best_move_score
                ):
                    best_move_kind = "reloc"
                    best_move_data = (i_item, b_from, b_to)
                    best_move_obj = cand_obj
                    best_move_score = score

        # -----------
        # Swap moves
        # -----------
        # Targeted: choose i from light bins and try a few j from tight and slack bins.
        # Build a few candidate partner bins
        # tight bins: high load; slack bins: high remaining
        if not time_exceeded():
            m = len(bins)
            if m > 1:
                # preselect partner bins lists
                tight = sorted(range(m), key=lambda b: loads[b], reverse=True)[: min(12, m)]
                slacky = sorted(range(m), key=lambda b: (C - loads[b]), reverse=True)[: min(12, m)]
                partner_bins = list(dict.fromkeys(tight + slacky + random.sample(range(m), min(8, m))))

                for i_item in items_T:
                    bi = assign[i_item]
                    if bi < 0:
                        continue
                    wi = weights[i_item]
                    rem_i = C - loads[bi]

                    # choose candidate bins different from bi
                    trials = 0
                    for bj in partner_bins:
                        if bj == bi or not bins[bj]:
                            continue
                        # choose j in bj close to wi+rem_i to increase fill of bi
                        target_wj = wi + rem_i
                        # sample few items from bj
                        cand_items = bins[bj]
                        if len(cand_items) > 10:
                            sample_j = random.sample(cand_items, 10)
                        else:
                            sample_j = cand_items
                        # pick best j by closeness
                        sample_j_sorted = sorted(sample_j, key=lambda j: abs(weights[j] - target_wj))
                        for j_item in sample_j_sorted[:3]:
                            if j_item == i_item:
                                continue
                            wj = weights[j_item]
                            if loads[bi] - wi + wj > C:
                                continue
                            if loads[bj] - wj + wi > C:
                                continue

                            li_old = loads[bi]
                            lj_old = loads[bj]
                            li_new = li_old - wi + wj
                            lj_new = lj_old - wj + wi
                            old_slack = (C - li_old) ** 2 + (C - lj_old) ** 2
                            new_slack = (C - li_new) ** 2 + (C - lj_new) ** 2
                            cand_slack = slack_sq_sum_curr - old_slack + new_slack
                            cand_obj = (m, cand_slack)

                            key = (i_item, j_item) if i_item < j_item else (j_item, i_item)
                            is_tabu = tabu_swap.get(key, 0) > it
                            if is_tabu and not aspiration_ok(cand_obj, best_obj, best_slack_for_m):
                                continue

                            score = tie_break_score("swap", bi, li_old, li_new, Tset)
                            if best_move_obj is None or better_obj(cand_obj, best_move_obj) or (
                                cand_obj == best_move_obj and score < best_move_score
                            ):
                                best_move_kind = "swap"
                                best_move_data = (i_item, j_item, bi, bj)
                                best_move_obj = cand_obj
                                best_move_score = score

                            trials += 1
                            if trials >= swap_trials_per_item:
                                break
                        if trials >= swap_trials_per_item:
                            break

        # -----------------
        # Ejection chain try
        # -----------------
        do_chain = (it % chain_period == 0) or (iters_since_best > stagnation_threshold)
        chain_move = None
        if do_chain and not time_exceeded():
            # pick lightest bin as target to empty
            b0 = T[0] if T else min(range(len(bins)), key=lambda b: loads[b])
            # Depth-2 chain attempt
            # Provide global slack in a variable for closure
            slack_sq_sum_curr = slack_sq_sum
            chain_move = find_depth2_chain(
                b0=b0,
                target_bins=T,
                dest_bins=D,
                bins=bins,
                loads=loads,
                assign=assign,
                bin_ids=bin_ids,
                tabu_reloc=tabu_reloc,
                it=it,
                tenure_block_best_obj=best_obj,
                best_slack_for_m=best_slack_for_m,
                r_bestfit=4,
                deficit_bins_scan=6,
            )

            if chain_move is not None:
                # evaluate chain objective again (for comparison with best_move)
                # chain is [(x,b1,b2),(a,b0,b1)]
                (x, b1, b2), (a, b0, b1b) = chain_move
                w_x = weights[x]
                w_a = weights[a]

                lf0 = loads[b0]
                lf1 = loads[b1]
                lf2 = loads[b2]
                lf1a = lf1 - w_x
                lf2a = lf2 + w_x
                lf0b = lf0 - w_a
                lf1b = lf1a + w_a
                new_m = len(bins) - 1 if lf0b == 0 else len(bins)

                old_slack = (C - lf0) ** 2 + (C - lf1) ** 2 + (C - lf2) ** 2
                new_slack = (C - lf0b) ** 2 + (C - lf1b) ** 2 + (C - lf2a) ** 2
                cand_slack = slack_sq_sum - old_slack + new_slack
                cand_obj = (new_m, cand_slack)
                score = -2000000  # prioritize chain if equal objective

                if best_move_obj is None or better_obj(cand_obj, best_move_obj) or (
                    cand_obj == best_move_obj and score < best_move_score
                ):
                    best_move_kind = "chain"
                    best_move_data = chain_move
                    best_move_obj = cand_obj
                    best_move_score = score

        # If no move found, trigger perturbation/restart policies
        if best_move_kind is None:
            iters_since_best += 1
        else:
            # --------------
            # Apply best move
            # --------------
            if best_move_kind == "reloc":
                i_item, b_from, b_to = best_move_data
                # set tabu to destination bin id AND forbid returning to source bin id
                source_id = bin_ids[b_from]
                dest_id = bin_ids[b_to]

                # apply
                slack_sq_sum, _ = apply_reloc_move(
                    i_item, b_from, b_to,
                    bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum
                )

                # tabu
                exp = it + tenure + random.randint(0, tenure)
                tabu_reloc[(i_item, source_id)] = exp  # forbid return
                tabu_reloc[(i_item, dest_id)] = exp    # forbid immediate re-move to same dest

                item_move_count[i_item] += 1
                bin_used_count[dest_id] = bin_used_count.get(dest_id, 0) + 1

            elif best_move_kind == "swap":
                i_item, j_item, bi, bj = best_move_data
                # apply
                slack_sq_sum = apply_swap_move(
                    i_item, j_item, bi, bj,
                    bins, loads, assign, pos_in_bin, slack_sq_sum
                )
                exp = it + tenure + random.randint(0, tenure)
                key = (i_item, j_item) if i_item < j_item else (j_item, i_item)
                tabu_swap[key] = exp
                # also restrict each item from returning immediately to its previous bin id
                # previous: i was in bi, now in bj; j was in bj, now in bi
                tabu_reloc[(i_item, bin_ids[bi])] = exp
                tabu_reloc[(j_item, bin_ids[bj])] = exp

                item_move_count[i_item] += 1
                item_move_count[j_item] += 1

            else:  # chain
                chain = best_move_data
                exp = it + tenure + random.randint(0, tenure)
                for (itm, b_from, b_to) in chain:
                    src_id = bin_ids[b_from]
                    dst_id = bin_ids[b_to]
                    slack_sq_sum, _ = apply_reloc_move(
                        itm, b_from, b_to,
                        bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum
                    )
                    # tabu both source and dest
                    tabu_reloc[(itm, src_id)] = exp
                    tabu_reloc[(itm, dst_id)] = exp
                    item_move_count[itm] += 1
                    bin_used_count[dst_id] = bin_used_count.get(dst_id, 0) + 1

            # Update best tracking
            curr_obj2 = (len(bins), slack_sq_sum)
            m2 = curr_obj2[0]
            prev = best_slack_for_m.get(m2)
            if prev is None or curr_obj2[1] < prev:
                best_slack_for_m[m2] = curr_obj2[1]

            if better_obj(curr_obj2, best_obj):
                best_obj = curr_obj2
                best_bins = [b[:] for b in bins]
                best_loads = loads[:]
                best_assign = assign[:]
                best_bin_ids = bin_ids[:]
                iters_since_best = 0
                push_elite(best_obj, bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum)
            else:
                iters_since_best += 1

        # -------------------------
        # Stagnation diversification
        # -------------------------
        if iters_since_best > stagnation_threshold and not time_exceeded():
            # Structured perturbation: select 1-3 lightest bins, remove their items and reinsert with randomized best-fit
            k_pert = 1 if len(bins) < 10 else (2 if len(bins) < 30 else 3)
            light = sorted(range(len(bins)), key=lambda b: loads[b])[: min(k_pert, len(bins))]
            removed_items: List[int] = []

            # clear short-term tabu
            tabu_reloc.clear()
            tabu_swap.clear()

            # remove items from selected bins
            # removing affects indices: process bins in descending index order to keep stable positions
            for b in sorted(light, reverse=True):
                for i_item in list(bins[b]):
                    removed_items.append(i_item)
                    # slack update for bin b
                    old = loads[b]
                    slack_sq_sum -= (C - old) ** 2
                    remove_item_from_bin(i_item, b, bins, loads, assign, pos_in_bin)
                    new = loads[b]
                    slack_sq_sum += (C - new) ** 2
                # remove empty bin
                _, slack_sq_sum = maybe_remove_empty_bin(b, bins, loads, assign, pos_in_bin, bin_ids, slack_sq_sum)

            # reinsert items: randomized order, best-fit with slight randomness
            if removed_items:
                removed_items.sort(key=lambda i: (-weights[i], random.random()))
                for i_item in removed_items:
                    w = weights[i_item]
                    # candidate bins: best few by fit
                    if len(bins) == 0:
                        bnew, slack_sq_sum, next_bin_id = add_bin(bins, loads, bin_ids, slack_sq_sum, next_bin_id)
                        # update slack for insert
                        old = loads[bnew]
                        slack_sq_sum -= (C - old) ** 2
                        add_item_to_bin(i_item, bnew, bins, loads, assign, pos_in_bin)
                        new = loads[bnew]
                        slack_sq_sum += (C - new) ** 2
                        continue

                    # build small candidate list
                    D2 = pick_dest_bins(loads, min(60, len(bins)), 4)
                    bf = best_fit_r(D2, loads, w, 6, forbidden_bin=-1)
                    if bf:
                        # pick among best fits with randomness
                        b_to = bf[0] if random.random() < 0.7 else random.choice(bf)
                    else:
                        b_to, slack_sq_sum, next_bin_id = add_bin(bins, loads, bin_ids, slack_sq_sum, next_bin_id)

                    old = loads[b_to]
                    slack_sq_sum -= (C - old) ** 2
                    add_item_to_bin(i_item, b_to, bins, loads, assign, pos_in_bin)
                    new = loads[b_to]
                    slack_sq_sum += (C - new) ** 2

            # restart from elite occasionally (intensification)
            if elite and random.random() < 0.6:
                # pick one of top 3
                pick = random.choice(elite[: min(3, len(elite))])
                _, ebins, eloads, eassign, epos, ebids, eslack = pick
                bins = [b[:] for b in ebins]
                loads = eloads[:]
                assign = eassign[:]
                pos_in_bin = epos[:]
                bin_ids = ebids[:]
                slack_sq_sum = eslack

            iters_since_best = 0
            # reset tenure toward baseline
            tenure = max(min_tenure, min(max_tenure, int(5 + 0.7 * (n ** 0.5))))

    # -----------------------
    # Build output from best
    # -----------------------

    packing = [b[:] for b in best_bins if b]
    # recompute weights once for safety
    bin_weights = [sum(weights[i] for i in b) for b in packing]

    return {"packing": packing, "bin_weights": bin_weights}
