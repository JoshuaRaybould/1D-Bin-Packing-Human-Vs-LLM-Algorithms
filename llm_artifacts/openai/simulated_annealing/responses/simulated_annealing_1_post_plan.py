import time
import math
import random
from typing import List, Dict


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)

    start = time.perf_counter()
    deadline = start + float(time_limit)

    def time_up() -> bool:
        return time.perf_counter() >= deadline

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---------------- Bucket index (A.1) ----------------
    # Use exact buckets for moderate C, else coarse buckets.
    # We maintain bucket lists of bin ids keyed by rem (or rem//g) and positions for O(1) remove.
    if C <= 5000:
        g = 1
        bucket_count = C + 1
    else:
        g = max(1, C // 256)
        bucket_count = (C // g) + 1

    def bucket_id(rem: int) -> int:
        bid = rem // g
        if bid < 0:
            return 0
        if bid >= bucket_count:
            return bucket_count - 1
        return bid

    # ---------------- Initial construction heuristics (E.1) ----------------
    def pack_ffd(order: List[int], best_fit: bool) -> tuple[list[list[int]], list[int], list[int]]:
        bins: list[list[int]] = []
        bw: list[int] = []
        item_bin = [-1] * n
        for i in order:
            wi = w[i]
            chosen = -1
            chosen_metric = None
            for b in range(len(bins)):
                rem = C - bw[b]
                if wi <= rem:
                    if best_fit:
                        metric = rem - wi  # minimize leftover
                        if chosen == -1 or metric < chosen_metric:
                            chosen, chosen_metric = b, metric
                            if metric == 0:
                                break
                    else:
                        chosen = b
                        break
            if chosen == -1:
                chosen = len(bins)
                bins.append([i])
                bw.append(wi)
            else:
                bins[chosen].append(i)
                bw[chosen] += wi
            item_bin[i] = chosen
        return bins, bw, item_bin

    def randomized_decreasing_order() -> List[int]:
        # Sort by weight decreasing, but shuffle ties (small random noise within groups)
        order = list(range(n))
        order.sort(key=lambda i: (-w[i], random.random()))
        return order

    # ---------------- Core state + helpers (G.1) ----------------
    bins: list[list[int]]
    bw: list[int]
    rem: list[int]
    item_bin: list[int]

    # Bucket structures over current bins
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    bucket_pos: list[int] = []  # per bin id: position within its bucket list

    # For optional L term: small-load indicator
    small_load: list[int] = []

    # Energy components (B,S,L) (B.1)
    # M should dominate typical changes in S (~O(C^2)).
    M = 10 * C * C
    lambda_L = C * C  # modest
    small_thresh = 0.2 * C

    B = 0
    S = 0
    L = 0

    def compute_components_from_scratch() -> tuple[int, int, int]:
        _B = len(bins)
        _S = 0
        _L = 0
        for b in range(_B):
            r = C - bw[b]
            _S += r * r
            if bw[b] < small_thresh:
                _L += 1
        return _B, _S, _L

    def energy_val(_B: int, _S: int, _L: int) -> int:
        return _B * M + _S + int(lambda_L) * _L

    def rebuild_buckets_and_smallload() -> None:
        nonlocal buckets, bucket_pos, small_load
        buckets = [[] for _ in range(bucket_count)]
        bucket_pos = [0] * len(bins)
        small_load = [0] * len(bins)
        for b in range(len(bins)):
            r = rem[b]
            bid = bucket_id(r)
            bucket_pos[b] = len(buckets[bid])
            buckets[bid].append(b)
            small_load[b] = 1 if bw[b] < small_thresh else 0

    def bucket_remove_bin(b: int) -> None:
        bid = bucket_id(rem[b])
        pos = bucket_pos[b]
        lst = buckets[bid]
        last = lst[-1]
        lst[pos] = last
        lst.pop()
        bucket_pos[last] = pos

    def bucket_add_bin(b: int) -> None:
        bid = bucket_id(rem[b])
        bucket_pos[b] = len(buckets[bid])
        buckets[bid].append(b)

    def update_bin_bucket(b: int, old_rem: int) -> None:
        old_bid = bucket_id(old_rem)
        new_bid = bucket_id(rem[b])
        if old_bid == new_bid:
            return
        # remove from old
        pos = bucket_pos[b]
        lst = buckets[old_bid]
        last = lst[-1]
        lst[pos] = last
        lst.pop()
        bucket_pos[last] = pos
        # add to new
        bucket_pos[b] = len(buckets[new_bid])
        buckets[new_bid].append(b)

    # O(1) unordered remove from bin list
    def remove_item_from_bin(bin_list: list[int], idx_in_bin: int) -> int:
        last_item = bin_list[-1]
        removed = bin_list[idx_in_bin]
        bin_list[idx_in_bin] = last_item
        bin_list.pop()
        return removed

    def normalize_remove_empty_bin(b: int, change_log: list[tuple]) -> None:
        """Remove empty bin b by swapping with last. Records enough to undo.
        change_log entries are tuples; this records ('remove_bin', b, last_bin_id, last_items_copy, last_bw, last_rem, last_small)
        """
        nonlocal B
        last = len(bins) - 1
        if b != last:
            # record last bin snapshot for undo
            change_log.append((
                'remove_bin', b, last,
                bins[last][:], bw[last], rem[last], small_load[last]
            ))
            # buckets: remove both bins then add back swapped-in (simplest)
            bucket_remove_bin(last)
            bucket_remove_bin(b)

            bins[b] = bins[last]
            bw[b] = bw[last]
            rem[b] = rem[last]
            small_load[b] = small_load[last]
            for it in bins[b]:
                item_bin[it] = b

            bins.pop(); bw.pop(); rem.pop(); small_load.pop(); bucket_pos.pop()

            # add swapped-in b
            bucket_add_bin(b)
        else:
            change_log.append(('remove_bin_last', last, bins[last][:], bw[last], rem[last], small_load[last]))
            bucket_remove_bin(last)
            bins.pop(); bw.pop(); rem.pop(); small_load.pop(); bucket_pos.pop()
        B -= 1

    def open_new_bin_with_item(i: int, change_log: list[tuple]) -> int:
        """Open a new bin with item i."""
        nonlocal B
        change_log.append(('open_bin',))
        b = len(bins)
        bins.append([i])
        bw.append(w[i])
        rem.append(C - w[i])
        small_load.append(1 if bw[b] < small_thresh else 0)
        bucket_pos.append(0)
        bucket_add_bin(b)
        item_bin[i] = b
        B += 1
        return b

    def move_item(i: int, src: int, src_pos: int, dst: int, change_log: list[tuple]) -> None:
        """Move item i from src (at src_pos) to dst. Records reversible operations.
        Updates bw/rem/buckets/small_load and item_bin.
        """
        wi = w[i]
        # record enough to undo: ('move', i, src, dst, removed_pos, src_old_rem, dst_old_rem, src_old_small, dst_old_small)
        src_old_rem = rem[src]
        dst_old_rem = rem[dst]
        src_old_small = small_load[src]
        dst_old_small = small_load[dst]
        change_log.append(('move', i, src, dst, src_pos, src_old_rem, dst_old_rem, src_old_small, dst_old_small))

        # remove from src in O(1) at src_pos
        remove_item_from_bin(bins[src], src_pos)
        bw[src] -= wi
        rem[src] += wi

        # add to dst
        bins[dst].append(i)
        bw[dst] += wi
        rem[dst] -= wi
        item_bin[i] = dst

        # update buckets and small flags
        update_bin_bucket(src, src_old_rem)
        update_bin_bucket(dst, dst_old_rem)

        small_load[src] = 1 if bw[src] < small_thresh else 0
        small_load[dst] = 1 if bw[dst] < small_thresh else 0

    def swap_items(b1: int, p1: int, b2: int, p2: int, change_log: list[tuple]) -> None:
        i = bins[b1][p1]
        j = bins[b2][p2]
        wi, wj = w[i], w[j]

        b1_old_rem = rem[b1]
        b2_old_rem = rem[b2]
        b1_old_small = small_load[b1]
        b2_old_small = small_load[b2]
        change_log.append(('swap', b1, p1, b2, p2, i, j, b1_old_rem, b2_old_rem, b1_old_small, b2_old_small))

        bins[b1][p1] = j
        bins[b2][p2] = i
        item_bin[i] = b2
        item_bin[j] = b1

        bw[b1] = bw[b1] - wi + wj
        bw[b2] = bw[b2] - wj + wi
        rem[b1] = C - bw[b1]
        rem[b2] = C - bw[b2]

        update_bin_bucket(b1, b1_old_rem)
        update_bin_bucket(b2, b2_old_rem)
        small_load[b1] = 1 if bw[b1] < small_thresh else 0
        small_load[b2] = 1 if bw[b2] < small_thresh else 0

    def undo(change_log: list[tuple]) -> None:
        """Undo in reverse order."""
        nonlocal B
        for entry in reversed(change_log):
            tag = entry[0]
            if tag == 'move':
                _, i, src, dst, src_pos, src_old_rem, dst_old_rem, src_old_small, dst_old_small = entry
                wi = w[i]
                # remove i from dst (last position)
                bins[dst].pop()
                bw[dst] -= wi
                rem[dst] = dst_old_rem

                # put i back into src at src_pos by appending and swapping into position
                bins[src].append(i)
                back_pos = len(bins[src]) - 1
                bins[src][back_pos], bins[src][src_pos] = bins[src][src_pos], bins[src][back_pos]
                bw[src] += wi
                rem[src] = src_old_rem

                item_bin[i] = src

                # bucket updates: easiest by direct fix via update with known old rems
                # But rem already restored; still need to ensure membership consistent.
                # We can do remove/add by using old_bid/new_bid; call update_bin_bucket with current=old so no-op.
                # Instead, force reinsert by removing and adding based on restored rem.
                # Lightweight: remove from current bucket using stored positions would require tracking; avoid.
                # So we do: bucket_remove_bin then bucket_add_bin for src/dst.
                # However bucket_pos may have shifted due to other operations; remove_bin expects correct bucket_pos.
                # To keep undo simple and correct, we avoid such forced operations and rely on symmetric updates
                # being undone by restoring rem and then calling update with old_rem captured at apply-time.
                # At apply-time we updated via update_bin_bucket(old_rem). Undo restores rem, so we must update
                # from *current bucket (after apply)* to *old bucket*. We don't have the apply-time rem.
                # Therefore we record old rems (done) AND we can compute apply-time rems:
                # src_apply_rem = old_rem + wi; dst_apply_rem = old_rem - wi.
                src_apply_rem = src_old_rem + wi
                dst_apply_rem = dst_old_rem - wi
                # move membership back
                update_bin_bucket(src, src_apply_rem)
                update_bin_bucket(dst, dst_apply_rem)

                small_load[src] = src_old_small
                small_load[dst] = dst_old_small

            elif tag == 'swap':
                (_, b1, p1, b2, p2, i, j,
                 b1_old_rem, b2_old_rem, b1_old_small, b2_old_small) = entry
                # after swap, bins[b1][p1]==j and bins[b2][p2]==i
                wi, wj = w[i], w[j]
                # restore arrays
                bins[b1][p1] = i
                bins[b2][p2] = j
                item_bin[i] = b1
                item_bin[j] = b2

                # compute apply-time rems to update bucket back
                b1_apply_rem = C - (C - b1_old_rem - wi + wj)  # not needed
                b2_apply_rem = C - (C - b2_old_rem - wj + wi)
                # simpler: apply-time rem can be derived:
                # rem = old_rem + wi - wj for b1; old_rem + wj - wi for b2
                b1_apply_rem = b1_old_rem + wi - wj
                b2_apply_rem = b2_old_rem + wj - wi

                bw[b1] = C - b1_old_rem
                bw[b2] = C - b2_old_rem
                rem[b1] = b1_old_rem
                rem[b2] = b2_old_rem

                update_bin_bucket(b1, b1_apply_rem)
                update_bin_bucket(b2, b2_apply_rem)
                small_load[b1] = b1_old_small
                small_load[b2] = b2_old_small

            elif tag == 'open_bin':
                # remove last bin
                b = len(bins) - 1
                bucket_remove_bin(b)
                for it in bins[b]:
                    item_bin[it] = -1
                bins.pop(); bw.pop(); rem.pop(); small_load.pop(); bucket_pos.pop()
                B -= 1

            elif tag == 'remove_bin':
                # ('remove_bin', b, last, last_items_copy, last_bw, last_rem, last_small)
                _, b, last, last_items, last_bw, last_rem, last_small = entry
                # We removed last and moved it into b. Undo by restoring last and emptying b.
                # Remove b from its current bucket
                bucket_remove_bin(b)

                # append placeholder for last
                bins.append([])
                bw.append(0)
                rem.append(0)
                small_load.append(0)
                bucket_pos.append(0)
                B += 1

                # move current b contents to last (these are last's contents originally)
                bins[last] = bins[b]
                bw[last] = bw[b]
                rem[last] = rem[b]
                small_load[last] = small_load[b]
                for it in bins[last]:
                    item_bin[it] = last

                # restore recorded last into b
                bins[b] = last_items
                bw[b] = last_bw
                rem[b] = last_rem
                small_load[b] = last_small
                for it in bins[b]:
                    item_bin[it] = b

                # re-add buckets
                bucket_add_bin(b)
                bucket_add_bin(last)

            elif tag == 'remove_bin_last':
                _, last, last_items, last_bw, last_rem, last_small = entry
                # we popped last; undo by appending it back
                bins.append(last_items)
                bw.append(last_bw)
                rem.append(last_rem)
                small_load.append(last_small)
                bucket_pos.append(0)
                bucket_add_bin(last)
                for it in last_items:
                    item_bin[it] = last
                B += 1

    # ---------------- Destination sampling using buckets (A.1, C.1) ----------------
    def sample_destination_bins(wi: int, exclude: int, samples_per_bucket: int = 3) -> List[int]:
        """Return a small list of candidate bins with rem >= wi, biased near rem≈wi."""
        if not bins:
            return []
        need = wi
        # choose buckets around target rem=need..need+2g.. and some larger slack buckets
        candidates: list[int] = []
        # start bucket at need
        start_bid = bucket_id(need)
        # Consider a few buckets: start, start+1, start+2, and a high-slack bucket near max
        bid_list = [start_bid]
        if start_bid + 1 < bucket_count:
            bid_list.append(start_bid + 1)
        if start_bid + 2 < bucket_count:
            bid_list.append(start_bid + 2)
        bid_list.append(bucket_count - 1)

        for bid in bid_list:
            lst = buckets[bid]
            if not lst:
                continue
            for _ in range(samples_per_bucket):
                b = lst[random.randrange(len(lst))]
                if b != exclude and rem[b] >= need:
                    candidates.append(b)
        return candidates

    def choose_best_fit_dest(wi: int, exclude: int) -> int:
        best = -1
        best_rem_after = C + 1
        for b in sample_destination_bins(wi, exclude, samples_per_bucket=4):
            ra = rem[b] - wi
            if 0 <= ra < best_rem_after:
                best = b
                best_rem_after = ra
                if ra == 0:
                    break
        return best

    # ---------------- Bin sampling lists (A.2) ----------------
    light_bins: list[int] = []
    tight_bins: list[int] = []
    candidate_victim_bins: list[int] = []

    def rebuild_sampling_lists() -> None:
        nonlocal light_bins, tight_bins, candidate_victim_bins
        Bn = len(bins)
        if Bn == 0:
            light_bins = tight_bins = candidate_victim_bins = []
            return

        # classify by remaining capacity quantiles
        # threshold: light if rem >= 0.5C, tight if rem <= 0.15C
        light_bins = [b for b in range(Bn) if rem[b] >= 0.5 * C and bins[b]]
        tight_bins = [b for b in range(Bn) if rem[b] <= 0.15 * C and bins[b]]

        # victims: small item count 1..6 OR large slack
        candidate_victim_bins = [b for b in range(Bn) if bins[b] and (len(bins[b]) <= 6 or rem[b] >= 0.35 * C)]

    # ---------------- Move evaluation (B.2) ----------------
    def delta_S_for_bin_change(old_rem: int, new_rem: int) -> int:
        return new_rem * new_rem - old_rem * old_rem

    def delta_L_for_bin_change(old_small: int, new_small: int) -> int:
        return new_small - old_small

    # ---------------- Neighborhood moves (C) ----------------
    def try_relocate(T: float) -> bool:
        nonlocal S, L
        if not bins:
            return False

        # pick source bin; bias towards tight bins sometimes
        if tight_bins and random.random() < 0.55:
            b1 = tight_bins[random.randrange(len(tight_bins))]
        else:
            b1 = random.randrange(len(bins))
        if not bins[b1]:
            return False

        pos = random.randrange(len(bins[b1]))
        i = bins[b1][pos]
        wi = w[i]

        b2 = choose_best_fit_dest(wi, exclude=b1)
        open_new = (b2 == -1)

        # compute dE for simple relocate (with possible open/close)
        b1_old_rem = rem[b1]
        b1_old_small = small_load[b1]
        b1_new_rem = b1_old_rem + wi
        b1_new_small = 1 if (bw[b1] - wi) < small_thresh else 0

        dB = 0
        dS = delta_S_for_bin_change(b1_old_rem, b1_new_rem)
        dL = delta_L_for_bin_change(b1_old_small, b1_new_small)

        if open_new:
            # if b1 empties, B unchanged; else B+1
            b1_will_empty = (len(bins[b1]) == 1)
            if not b1_will_empty:
                dB += 1
                # new bin rem = C-wi
                dS += (C - wi) * (C - wi)
                new_small = 1 if wi < small_thresh else 0
                dL += new_small
            # else: we remove b1, but also open bin with same item: net B unchanged
        else:
            b2_old_rem = rem[b2]
            b2_old_small = small_load[b2]
            b2_new_rem = b2_old_rem - wi
            b2_new_small = 1 if (bw[b2] + wi) < small_thresh else 0
            dS += delta_S_for_bin_change(b2_old_rem, b2_new_rem)
            dL += delta_L_for_bin_change(b2_old_small, b2_new_small)

            # if b1 becomes empty, B-1 and remove its rem^2 (which would be 0^2 after removal, but we remove bin)
            if len(bins[b1]) == 1:
                dB -= 1
                # removing empty bin removes its contribution after move: it would be rem=C (since bw=0) if left empty,
                # but we normalize-remove, so bin disappears. Our dS currently assumes b1 remains with new_rem=b1_old_rem+wi.
                # When b1 empties, its new_rem would be C (bw=0). But bin removed => subtract that would-be contribution.
                # Net correction: replace b1 contribution with 0 (removed). So subtract (C^2).
                dS -= C * C
                # L term: removed bin would be small (empty), but removed so subtract its would-be small indicator (=1)
                dL -= 1

        dE = dB * M + dS + int(lambda_L) * dL

        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            change_log: list[tuple] = []
            # apply
            if open_new:
                if len(bins[b1]) == 1:
                    # move item into a new bin, then remove empty b1 -> effectively rename bin; simpler:
                    # remove item from b1 and just keep bin b1 with item (no net change). So do nothing.
                    return False
                # remove from b1, open new bin
                move_item(i, b1, pos, open_new_bin_with_item(i, change_log), change_log)  # note: this double-moves i
                # The above is wrong logically (i already placed). Undo approach too complex.
            # We'll implement open_new properly without using move_item.
            # Re-apply cleanly by undoing if reached here.
            undo(change_log)

        return False

    # Fix relocate implementation with correct open_new handling
    def try_relocate(T: float) -> bool:
        nonlocal S, L
        if not bins:
            return False

        if tight_bins and random.random() < 0.55:
            b1 = tight_bins[random.randrange(len(tight_bins))]
        else:
            b1 = random.randrange(len(bins))
        if not bins[b1]:
            return False

        pos = random.randrange(len(bins[b1]))
        i = bins[b1][pos]
        wi = w[i]

        b2 = choose_best_fit_dest(wi, exclude=b1)
        open_new = (b2 == -1)

        b1_old_rem = rem[b1]
        b1_old_small = small_load[b1]

        # precompute dE
        dB = 0
        dS = 0
        dL = 0

        # source bin after removal
        b1_new_rem = b1_old_rem + wi
        b1_new_small = 1 if (bw[b1] - wi) < small_thresh else 0

        if open_new:
            b1_will_empty = (len(bins[b1]) == 1)
            if b1_will_empty:
                # moving item from b1 to new bin is a no-op up to renaming; reject to avoid churn.
                return False
            # b1 changes, plus add new bin
            dB += 1
            dS += delta_S_for_bin_change(b1_old_rem, b1_new_rem)
            dS += (C - wi) * (C - wi)
            dL += delta_L_for_bin_change(b1_old_small, b1_new_small)
            dL += (1 if wi < small_thresh else 0)
        else:
            b2_old_rem = rem[b2]
            b2_old_small = small_load[b2]
            b2_new_rem = b2_old_rem - wi
            b2_new_small = 1 if (bw[b2] + wi) < small_thresh else 0

            dS += delta_S_for_bin_change(b1_old_rem, b1_new_rem)
            dS += delta_S_for_bin_change(b2_old_rem, b2_new_rem)
            dL += delta_L_for_bin_change(b1_old_small, b1_new_small)
            dL += delta_L_for_bin_change(b2_old_small, b2_new_small)

            if len(bins[b1]) == 1:
                # bin removed
                dB -= 1
                # remove would-be empty bin contribution (C^2) and L(+1)
                dS -= C * C
                dL -= 1

        dE = dB * M + dS + int(lambda_L) * dL

        if dE > 0 and random.random() >= math.exp(-dE / max(T, 1e-12)):
            return False

        change_log: list[tuple] = []
        old_S, old_L, old_B = S, L, B

        # apply
        if open_new:
            # remove item from b1
            src_old_rem = rem[b1]
            src_old_small = small_load[b1]
            change_log.append(('reloc_open', i, b1, pos, src_old_rem, src_old_small))
            remove_item_from_bin(bins[b1], pos)
            bw[b1] -= wi
            rem[b1] += wi
            update_bin_bucket(b1, src_old_rem)
            small_load[b1] = 1 if bw[b1] < small_thresh else 0

            # open new bin with i
            open_new_bin_with_item(i, change_log)

        else:
            move_item(i, b1, pos, b2, change_log)
            if not bins[b1]:
                normalize_remove_empty_bin(b1, change_log)

        # update energy components
        S = old_S + dS
        L = old_L + dL
        # B already updated by open/close; but dB must match
        if B != old_B + dB:
            # safeguard: recompute components if mismatch
            B2, S2, L2 = compute_components_from_scratch()
            B = B2
            S = S2
            L = L2
            rebuild_buckets_and_smallload()
            rebuild_sampling_lists()
        return True

    def undo_reloc_open(change_log_entry: tuple) -> None:
        # ('reloc_open', i, b1, pos, src_old_rem, src_old_small)
        _, i, b1, pos, src_old_rem, src_old_small = change_log_entry
        wi = w[i]
        # remove i from last bin (opened bin)
        bnew = len(bins) - 1
        bucket_remove_bin(bnew)
        bins.pop(); bw.pop(); rem.pop(); small_load.pop(); bucket_pos.pop()
        # restore item_bin later
        # put i back into b1 at pos
        bins[b1].append(i)
        back_pos = len(bins[b1]) - 1
        bins[b1][back_pos], bins[b1][pos] = bins[b1][pos], bins[b1][back_pos]
        bw[b1] += wi
        # buckets: b1 rem currently = src_old_rem + wi, restore to src_old_rem
        b1_apply_rem = src_old_rem + wi
        rem[b1] = src_old_rem
        update_bin_bucket(b1, b1_apply_rem)
        small_load[b1] = src_old_small
        item_bin[i] = b1

    # Extend undo to handle reloc_open tag
    def undo(change_log: list[tuple]) -> None:
        nonlocal B
        for entry in reversed(change_log):
            tag = entry[0]
            if tag == 'reloc_open':
                # undo opening bin is handled together with 'open_bin'; here we already popped open_bin in open_new_bin_with_item
                undo_reloc_open(entry)
                B -= 1  # open_new_bin_with_item incremented B; remove it
            elif tag == 'move':
                _, i, src, dst, src_pos, src_old_rem, dst_old_rem, src_old_small, dst_old_small = entry
                wi = w[i]
                # remove i from dst
                bins[dst].pop()
                bw[dst] -= wi
                dst_apply_rem = dst_old_rem - wi
                rem[dst] = dst_old_rem

                # restore i into src at src_pos
                bins[src].append(i)
                back_pos = len(bins[src]) - 1
                bins[src][back_pos], bins[src][src_pos] = bins[src][src_pos], bins[src][back_pos]
                bw[src] += wi
                src_apply_rem = src_old_rem + wi
                rem[src] = src_old_rem

                item_bin[i] = src

                update_bin_bucket(src, src_apply_rem)
                update_bin_bucket(dst, dst_apply_rem)
                small_load[src] = src_old_small
                small_load[dst] = dst_old_small

            elif tag == 'swap':
                (_, b1, p1, b2, p2, i, j,
                 b1_old_rem, b2_old_rem, b1_old_small, b2_old_small) = entry
                wi, wj = w[i], w[j]
                # restore
                bins[b1][p1] = i
                bins[b2][p2] = j
                item_bin[i] = b1
                item_bin[j] = b2
                b1_apply_rem = b1_old_rem + wi - wj
                b2_apply_rem = b2_old_rem + wj - wi
                bw[b1] = C - b1_old_rem
                bw[b2] = C - b2_old_rem
                rem[b1] = b1_old_rem
                rem[b2] = b2_old_rem
                update_bin_bucket(b1, b1_apply_rem)
                update_bin_bucket(b2, b2_apply_rem)
                small_load[b1] = b1_old_small
                small_load[b2] = b2_old_small

            elif tag == 'open_bin':
                # not used directly; reloc_open handles it
                pass

            elif tag == 'remove_bin':
                _, b, last, last_items, last_bw, last_rem, last_small = entry
                bucket_remove_bin(b)

                bins.append([])
                bw.append(0)
                rem.append(0)
                small_load.append(0)
                bucket_pos.append(0)
                B += 1

                bins[last] = bins[b]
                bw[last] = bw[b]
                rem[last] = rem[b]
                small_load[last] = small_load[b]
                for it in bins[last]:
                    item_bin[it] = last

                bins[b] = last_items
                bw[b] = last_bw
                rem[b] = last_rem
                small_load[b] = last_small
                for it in bins[b]:
                    item_bin[it] = b

                bucket_add_bin(b)
                bucket_add_bin(last)

            elif tag == 'remove_bin_last':
                _, last, last_items, last_bw, last_rem, last_small = entry
                bins.append(last_items)
                bw.append(last_bw)
                rem.append(last_rem)
                small_load.append(last_small)
                bucket_pos.append(0)
                bucket_add_bin(last)
                for it in last_items:
                    item_bin[it] = last
                B += 1

    def try_swap(T: float) -> bool:
        nonlocal S, L
        if len(bins) < 2:
            return False
        b1 = random.randrange(len(bins))
        b2 = random.randrange(len(bins) - 1)
        if b2 >= b1:
            b2 += 1
        if not bins[b1] or not bins[b2]:
            return False
        p1 = random.randrange(len(bins[b1]))
        p2 = random.randrange(len(bins[b2]))
        i = bins[b1][p1]
        j = bins[b2][p2]
        wi, wj = w[i], w[j]

        load1 = bw[b1] - wi + wj
        load2 = bw[b2] - wj + wi
        if load1 > C or load2 > C:
            return False

        b1_old_rem = rem[b1]
        b2_old_rem = rem[b2]
        b1_old_small = small_load[b1]
        b2_old_small = small_load[b2]

        b1_new_rem = C - load1
        b2_new_rem = C - load2
        b1_new_small = 1 if load1 < small_thresh else 0
        b2_new_small = 1 if load2 < small_thresh else 0

        dS = delta_S_for_bin_change(b1_old_rem, b1_new_rem) + delta_S_for_bin_change(b2_old_rem, b2_new_rem)
        dL = delta_L_for_bin_change(b1_old_small, b1_new_small) + delta_L_for_bin_change(b2_old_small, b2_new_small)
        dE = dS + int(lambda_L) * dL

        if dE > 0 and random.random() >= math.exp(-dE / max(T, 1e-12)):
            return False

        change_log: list[tuple] = []
        swap_items(b1, p1, b2, p2, change_log)
        S += dS
        L += dL
        return True

    def try_ejection_chain(T: float) -> bool:
        """Swap+relocate (ejection chain length 2) (C.4)."""
        nonlocal S, L
        if len(bins) < 2:
            return False

        b1 = random.randrange(len(bins))
        if not bins[b1]:
            return False
        p1 = random.randrange(len(bins[b1]))
        i = bins[b1][p1]
        wi = w[i]

        # choose b2 where we'd like to place i (close fit), even if infeasible; then try eject
        # sample a few candidates by proximity
        cand_b2 = sample_destination_bins(max(1, wi // 2), exclude=b1, samples_per_bucket=4)
        if not cand_b2:
            b2 = random.randrange(len(bins))
            if b2 == b1:
                return False
        else:
            b2 = cand_b2[random.randrange(len(cand_b2))]
            if b2 == b1:
                return False

        if not bins[b2]:
            return False

        # If direct relocate feasible, prefer standard relocate.
        if rem[b2] >= wi:
            return False

        # Try eject one item j from b2 to some b3 to make room.
        p2 = random.randrange(len(bins[b2]))
        j = bins[b2][p2]
        wj = w[j]

        # after ejecting j, room in b2 is rem[b2] + wj
        if rem[b2] + wj < wi:
            return False

        b3 = choose_best_fit_dest(wj, exclude=b2)
        if b3 == -1 or b3 == b1:
            # allow open new bin for j only at high T
            if T < 0.5:
                return False
            b3 = -1

        # compute dE for compound move as one acceptance
        # changes: b1 remove i; b2 remove j add i; b3 add j (or open)
        # If b1 empties, close it.
        dB = 0
        dS = 0
        dL = 0

        # b1
        b1_old_rem = rem[b1]
        b1_old_small = small_load[b1]
        b1_new_rem = b1_old_rem + wi
        b1_new_small = 1 if (bw[b1] - wi) < small_thresh else 0
        dS += delta_S_for_bin_change(b1_old_rem, b1_new_rem)
        dL += delta_L_for_bin_change(b1_old_small, b1_new_small)

        # b2
        b2_old_rem = rem[b2]
        b2_old_small = small_load[b2]
        # b2 load: -wj + wi
        b2_new_rem = b2_old_rem + wj - wi
        b2_new_small = 1 if (bw[b2] - wj + wi) < small_thresh else 0
        dS += delta_S_for_bin_change(b2_old_rem, b2_new_rem)
        dL += delta_L_for_bin_change(b2_old_small, b2_new_small)

        if b3 == -1:
            # open new bin with j
            if len(bins[b2]) == 1:
                return False
            dB += 1
            dS += (C - wj) * (C - wj)
            dL += (1 if wj < small_thresh else 0)
        else:
            b3_old_rem = rem[b3]
            b3_old_small = small_load[b3]
            b3_new_rem = b3_old_rem - wj
            b3_new_small = 1 if (bw[b3] + wj) < small_thresh else 0
            dS += delta_S_for_bin_change(b3_old_rem, b3_new_rem)
            dL += delta_L_for_bin_change(b3_old_small, b3_new_small)

        # close b1 if empties
        if len(bins[b1]) == 1:
            dB -= 1
            dS -= C * C
            dL -= 1

        dE = dB * M + dS + int(lambda_L) * dL
        if dE > 0 and random.random() >= math.exp(-dE / max(T, 1e-12)):
            return False

        change_log: list[tuple] = []
        old_S, old_L, old_B = S, L, B

        # Apply: move j out of b2 first (careful with indices), then move i b1->b2
        # find updated position of j if b2 list changes: we remove at p2 then later append i.
        if b3 == -1:
            # remove j from b2
            b2_rem_old = rem[b2]
            b2_small_old = small_load[b2]
            change_log.append(('eject_open_remove', j, b2, p2, b2_rem_old, b2_small_old))
            remove_item_from_bin(bins[b2], p2)
            bw[b2] -= wj
            rem[b2] += wj
            update_bin_bucket(b2, b2_rem_old)
            small_load[b2] = 1 if bw[b2] < small_thresh else 0

            # open new bin with j
            open_new_bin_with_item(j, change_log)
        else:
            move_item(j, b2, p2, b3, change_log)

        # now move i into b2
        # i is in b1 at p1; but if b1==b2 (should not) already excluded by rem infeasible sampling.
        move_item(i, b1, p1, b2, change_log)
        if not bins[b1]:
            normalize_remove_empty_bin(b1, change_log)

        S = old_S + dS
        L = old_L + dL
        if B != old_B + dB:
            B2, S2, L2 = compute_components_from_scratch()
            B = B2
            S = S2
            L = L2
            rebuild_buckets_and_smallload()
            rebuild_sampling_lists()
        return True

    def undo_eject_open_remove(entry: tuple) -> None:
        # ('eject_open_remove', j, b2, p2, old_rem, old_small)
        _, j, b2, p2, old_rem, old_small = entry
        wj = w[j]
        # remove j from last bin
        bnew = len(bins) - 1
        bucket_remove_bin(bnew)
        bins.pop(); bw.pop(); rem.pop(); small_load.pop(); bucket_pos.pop()
        # put back into b2 at p2
        bins[b2].append(j)
        back_pos = len(bins[b2]) - 1
        bins[b2][back_pos], bins[b2][p2] = bins[b2][p2], bins[b2][back_pos]
        bw[b2] += wj
        b2_apply_rem = old_rem + wj
        rem[b2] = old_rem
        update_bin_bucket(b2, b2_apply_rem)
        small_load[b2] = old_small
        item_bin[j] = b2

    # Patch undo to handle eject_open_remove
    def undo(change_log: list[tuple]) -> None:
        nonlocal B
        for entry in reversed(change_log):
            tag = entry[0]
            if tag == 'eject_open_remove':
                undo_eject_open_remove(entry)
                B -= 1
            elif tag == 'reloc_open':
                undo_reloc_open(entry)
                B -= 1
            elif tag == 'move':
                _, i, src, dst, src_pos, src_old_rem, dst_old_rem, src_old_small, dst_old_small = entry
                wi = w[i]
                bins[dst].pop()
                bw[dst] -= wi
                dst_apply_rem = dst_old_rem - wi
                rem[dst] = dst_old_rem

                bins[src].append(i)
                back_pos = len(bins[src]) - 1
                bins[src][back_pos], bins[src][src_pos] = bins[src][src_pos], bins[src][back_pos]
                bw[src] += wi
                src_apply_rem = src_old_rem + wi
                rem[src] = src_old_rem
                item_bin[i] = src

                update_bin_bucket(src, src_apply_rem)
                update_bin_bucket(dst, dst_apply_rem)
                small_load[src] = src_old_small
                small_load[dst] = dst_old_small

            elif tag == 'swap':
                (_, b1, p1, b2, p2, i, j,
                 b1_old_rem, b2_old_rem, b1_old_small, b2_old_small) = entry
                wi, wj = w[i], w[j]
                bins[b1][p1] = i
                bins[b2][p2] = j
                item_bin[i] = b1
                item_bin[j] = b2
                b1_apply_rem = b1_old_rem + wi - wj
                b2_apply_rem = b2_old_rem + wj - wi
                bw[b1] = C - b1_old_rem
                bw[b2] = C - b2_old_rem
                rem[b1] = b1_old_rem
                rem[b2] = b2_old_rem
                update_bin_bucket(b1, b1_apply_rem)
                update_bin_bucket(b2, b2_apply_rem)
                small_load[b1] = b1_old_small
                small_load[b2] = b2_old_small

            elif tag == 'remove_bin':
                _, b, last, last_items, last_bw, last_rem, last_small = entry
                bucket_remove_bin(b)

                bins.append([])
                bw.append(0)
                rem.append(0)
                small_load.append(0)
                bucket_pos.append(0)
                B += 1

                bins[last] = bins[b]
                bw[last] = bw[b]
                rem[last] = rem[b]
                small_load[last] = small_load[b]
                for it in bins[last]:
                    item_bin[it] = last

                bins[b] = last_items
                bw[b] = last_bw
                rem[b] = last_rem
                small_load[b] = last_small
                for it in bins[b]:
                    item_bin[it] = b

                bucket_add_bin(b)
                bucket_add_bin(last)

            elif tag == 'remove_bin_last':
                _, last, last_items, last_bw, last_rem, last_small = entry
                bins.append(last_items)
                bw.append(last_bw)
                rem.append(last_rem)
                small_load.append(last_small)
                bucket_pos.append(0)
                bucket_add_bin(last)
                for it in last_items:
                    item_bin[it] = last
                B += 1

    def try_victim_empty(T: float, allow_partial: bool) -> bool:
        """Attempt to empty a victim bin as one compound SA move (C.2)."""
        nonlocal S, L
        if not candidate_victim_bins:
            return False
        v = candidate_victim_bins[random.randrange(len(candidate_victim_bins))]
        if not bins[v] or len(bins) == 1:
            return False

        items = bins[v][:]
        items.sort(key=lambda i: w[i], reverse=True)

        placements: list[tuple[int, int]] = []  # (item, dest_bin)

        # Greedy place into other bins
        for i in items:
            wi = w[i]
            d = choose_best_fit_dest(wi, exclude=v)
            if d == -1:
                if allow_partial:
                    break
                return False
            placements.append((i, d))

        if not placements:
            return False

        # If partial, require we moved all but 1 item to set up near-empty
        if len(placements) < len(items) and (len(items) - len(placements) > 1):
            return False

        # Compute dE by simulating rem changes on involved bins
        involved = {v}
        for _, d in placements:
            involved.add(d)

        # Current rem/small snapshots
        old_rems = {b: rem[b] for b in involved}
        old_smalls = {b: small_load[b] for b in involved}
        old_bws = {b: bw[b] for b in involved}

        # Apply deltas to compute new loads
        v_move_weight = sum(w[i] for i, _ in placements)
        new_bws = dict(old_bws)
        new_bws[v] = old_bws[v] - v_move_weight
        for i, d in placements:
            new_bws[d] = new_bws.get(d, bw[d]) + w[i]
            if new_bws[d] > C:
                return False

        # bins count change: if v becomes empty => -1 else 0
        dB = -1 if new_bws[v] == 0 else 0

        dS = 0
        dL = 0
        for b in involved:
            b_old_rem = old_rems[b]
            b_old_small = old_smalls[b]
            b_new_rem = C - new_bws[b]
            b_new_small = 1 if new_bws[b] < small_thresh else 0
            dS += delta_S_for_bin_change(b_old_rem, b_new_rem)
            dL += delta_L_for_bin_change(b_old_small, b_new_small)

        if dB == -1:
            # remove empty bin would remove rem=C contribution and small indicator
            dS -= C * C
            dL -= 1

        dE = dB * M + dS + int(lambda_L) * dL
        if dE > 0 and random.random() >= math.exp(-dE / max(T, 1e-12)):
            return False

        # Apply with change log; must locate each item position in v at time of move.
        change_log: list[tuple] = []
        old_S, old_L, old_B = S, L, B

        # Build a position map for victim bin (updated as we remove in O(1))
        pos_in_v = {bins[v][k]: k for k in range(len(bins[v]))}
        for i, d in placements:
            p = pos_in_v.get(i)
            if p is None:
                # should not happen
                undo(change_log)
                return False
            # moving i out will swap some last element into p; update map
            last_item = bins[v][-1]
            # apply move
            move_item(i, v, p, d, change_log)
            if i != last_item:
                pos_in_v[last_item] = p
            pos_in_v.pop(i, None)

        if not bins[v]:
            normalize_remove_empty_bin(v, change_log)

        S = old_S + dS
        L = old_L + dL
        if B != old_B + dB:
            B2, S2, L2 = compute_components_from_scratch()
            B = B2
            S = S2
            L = L2
            rebuild_buckets_and_smallload()
            rebuild_sampling_lists()
        return True

    def try_merge_2for1(T: float, allow_partial: bool) -> bool:
        """2-for-1 merge: try to empty a donor bin by redistributing its items (C.3)."""
        # Implemented as victim emptying but donor picked differently and with a preferred receiver.
        if not candidate_victim_bins or len(bins) < 2:
            return False
        donor = candidate_victim_bins[random.randrange(len(candidate_victim_bins))]
        if not bins[donor] or len(bins[donor]) > 10:
            # keep it cheap
            return False
        return try_victim_empty(T, allow_partial=allow_partial)

    # ---------------- Optional validation (G.2) ----------------
    def validate_solution() -> bool:
        seen = [0] * n
        for b in range(len(bins)):
            s = 0
            for it in bins[b]:
                if it < 0 or it >= n:
                    return False
                seen[it] += 1
                s += w[it]
            if s != bw[b] or s > C:
                return False
        return all(x == 1 for x in seen)

    # ---------------- Build multiple starts (E.1) ----------------
    init_budget = min(0.5, 0.02 * max(0.0, time_limit))
    init_end = start + init_budget

    candidates: list[tuple[list[list[int]], list[int], list[int]]] = []

    # FFD
    order0 = list(range(n))
    order0.sort(key=lambda i: w[i], reverse=True)
    candidates.append(pack_ffd(order0, best_fit=False))

    # BFD
    candidates.append(pack_ffd(order0, best_fit=True))

    # randomized variants
    k_rand = 6
    while len(candidates) < 2 + k_rand and time.perf_counter() < init_end:
        candidates.append(pack_ffd(randomized_decreasing_order(), best_fit=True if random.random() < 0.7 else False))

    # Choose best by (B, S)
    best_pack = None
    best_key = None
    for cbins, cbw, citem_bin in candidates:
        # compute rem and S
        cB = len(cbw)
        cS = 0
        for load in cbw:
            r = C - load
            cS += r * r
        key = (cB, cS)
        if best_key is None or key < best_key:
            best_key = key
            best_pack = (cbins, cbw, citem_bin)

    bins, bw, item_bin = best_pack  # type: ignore
    rem = [C - x for x in bw]

    B, S, L = compute_components_from_scratch()
    rebuild_buckets_and_smallload()
    rebuild_sampling_lists()

    best_bins = [b[:] for b in bins]
    best_bw = bw[:]
    best_B = B
    best_E = energy_val(B, S, L)

    # ---------------- Warmup + T0 calibration (E.2) ----------------
    # Sample positive dE from attempted moves.
    pos_dEs: list[int] = []

    def attempt_random_move_collect(T: float) -> bool:
        beforeE = energy_val(B, S, L)
        r = random.random()
        moved = False
        if r < 0.55:
            moved = try_relocate(T)
        elif r < 0.70:
            moved = try_swap(T)
        elif r < 0.85:
            moved = try_ejection_chain(T)
        else:
            moved = try_victim_empty(T, allow_partial=True)
        afterE = energy_val(B, S, L)
        dE = afterE - beforeE
        if dE > 0:
            pos_dEs.append(dE)
        return moved

    warmup_end = min(deadline, time.perf_counter() + min(0.15, 0.05 * max(0.0, time_limit)))
    T_tmp = 1.0
    while time.perf_counter() < warmup_end and len(pos_dEs) < 200:
        attempt_random_move_collect(T_tmp)

    if pos_dEs:
        pos_dEs.sort()
        med = pos_dEs[len(pos_dEs) // 2]
        # T0 so that exp(-med/T0) ~ 0.6
        T0 = max(1e-6, -med / math.log(0.6))
    else:
        avg_w = sum(w) / n
        T0 = max(1.0, 0.5 * avg_w)

    Tmin = 1e-6
    Tmax = T0

    # ---------------- Iteration budgeting (F) ----------------
    now = time.perf_counter()
    remaining = deadline - now
    if remaining <= 0:
        return {"packing": best_bins, "bin_weights": best_bw}

    calib_time = min(0.25, 0.10 * remaining)
    calib_end = now + calib_time
    calib_iters = 0
    T_cal = min(T0, 1.0 * T0)

    while time.perf_counter() < calib_end:
        attempt_random_move_collect(T_cal)
        calib_iters += 1
        curE = energy_val(B, S, L)
        if curE < best_E or (B < best_B):
            best_E = curE
            best_B = min(best_B, B)
            best_bins = [b[:] for b in bins]
            best_bw = bw[:]

    elapsed_cal = max(1e-6, time.perf_counter() - now)
    iters_per_sec = calib_iters / elapsed_cal

    now2 = time.perf_counter()
    remaining2 = max(0.0, deadline - now2)
    target = 0.99 * remaining2

    min_iters = 10_000 + 80 * n
    max_iters_cap = 50_000_000
    max_iters = int(max(min_iters, min(max_iters_cap, target * iters_per_sec)))

    # ---------------- SA main loop with epoch cooling (D) ----------------
    T = T0
    epoch = min(5000, max(700, 500 + 5 * n))

    accepted = 0
    attempted = 0

    epochs_since_bestB = 0
    boost_compound_epochs = 0

    # stats for adaptive mix (H)
    victim_attempts = victim_success = 0

    # move mix schedule (C.5) base probabilities per phase
    def move_probs(progress: float) -> tuple[float, float, float, float]:
        if progress < 0.3:
            pr, ps, pe, pv = 0.45, 0.15, 0.20, 0.20
        elif progress < 0.8:
            pr, ps, pe, pv = 0.35, 0.10, 0.20, 0.35
        else:
            pr, ps, pe, pv = 0.25, 0.05, 0.10, 0.60
        if boost_compound_epochs > 0:
            # temporary increase compound moves
            pv = min(0.75, pv + 0.15)
            pe = min(0.25, pe + 0.05)
            # renormalize by reducing relocate
            total = pr + ps + pe + pv
            pr = max(0.05, pr - (total - 1.0))
        return pr, ps, pe, pv

    # periodic rebuild of sampling lists (A.2)
    rebuild_sampling_lists()

    # optional validation cadence
    next_validate = time.perf_counter() + 1.5

    for it in range(max_iters):
        if (it & 1023) == 0 and time_up():
            break

        if (it % epoch) == 0:
            rebuild_sampling_lists()
            if time_up():
                break

        progress = it / max(1, max_iters)
        pr, ps, pe, pv = move_probs(progress)

        # adaptive tweak (H.2)
        if attempted >= epoch and (it % epoch) == 0:
            if victim_attempts >= 30:
                rate = victim_success / max(1, victim_attempts)
                if rate < 0.01:
                    pe = min(0.30, pe + 0.05)
                elif rate > 0.05:
                    pv = min(0.80, pv + 0.05)
            victim_attempts = victim_success = 0

        r = random.random()
        moved = False
        beforeE = energy_val(B, S, L)

        if r < pr:
            moved = try_relocate(T)
        elif r < pr + ps:
            moved = try_swap(T)
        elif r < pr + ps + pe:
            moved = try_ejection_chain(T)
        else:
            victim_attempts += 1
            moved = try_victim_empty(T, allow_partial=(T > 0.4 * T0))
            if moved:
                victim_success += 1

        afterE = energy_val(B, S, L)
        attempted += 1
        if moved:
            accepted += 1

        # update best
        if B < best_B or (B == best_B and afterE < best_E):
            best_B = B
            best_E = afterE
            best_bins = [b[:] for b in bins]
            best_bw = bw[:]
            epochs_since_bestB = 0
        
        # epoch temperature control (D.1)
        if (it + 1) % epoch == 0:
            acc = accepted / max(1, attempted)
            if acc < 0.12:
                T *= 1.25
            elif acc > 0.45:
                T *= 0.85
            else:
                T *= 0.95
            T = min(Tmax, max(Tmin, T))
            accepted = 0
            attempted = 0

            # reheating tied to bin-count improvement (D.2)
            epochs_since_bestB += 1
            if epochs_since_bestB >= 5:
                T = min(T0, T * 1.8)
                boost_compound_epochs = 1
                epochs_since_bestB = 0
            else:
                if boost_compound_epochs > 0:
                    boost_compound_epochs -= 1

        # optional validation (G.2)
        if time.perf_counter() >= next_validate:
            next_validate = time.perf_counter() + 1.5
            if not validate_solution():
                # fall back to best known state and continue
                bins = [b[:] for b in best_bins]
                bw = best_bw[:]
                item_bin = [-1] * n
                for b in range(len(bins)):
                    for it2 in bins[b]:
                        item_bin[it2] = b
                rem = [C - x for x in bw]
                B, S, L = compute_components_from_scratch()
                rebuild_buckets_and_smallload()
                rebuild_sampling_lists()

    return {"packing": best_bins, "bin_weights": best_bw}
