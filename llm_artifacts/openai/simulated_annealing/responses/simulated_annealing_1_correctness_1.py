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

    # ---------- Helpers ----------
    def time_up() -> bool:
        return time.perf_counter() >= deadline

    def ffd_initial() -> tuple[list[list[int]], list[int], list[int]]:
        order = list(range(n))
        order.sort(key=lambda i: w[i], reverse=True)
        bins: list[list[int]] = []
        bw: list[int] = []
        item_bin = [-1] * n
        for i in order:
            wi = w[i]
            best_b = -1
            best_rem = C + 1
            for b in range(len(bins)):
                rem = C - bw[b]
                if wi <= rem:
                    new_rem = rem - wi
                    if new_rem < best_rem:
                        best_rem = new_rem
                        best_b = b
                        if best_rem == 0:
                            break
            if best_b == -1:
                best_b = len(bins)
                bins.append([i])
                bw.append(wi)
            else:
                bins[best_b].append(i)
                bw[best_b] += wi
            item_bin[i] = best_b
        return bins, bw, item_bin

    def normalize(bins: list[list[int]], bw: list[int], item_bin: list[int]) -> None:
        b = 0
        while b < len(bins):
            if bins[b]:
                b += 1
                continue
            last = len(bins) - 1
            if b == last:
                bins.pop()
                bw.pop()
                break
            # move last into b
            bins[b] = bins[last]
            bw[b] = bw[last]
            for it in bins[b]:
                item_bin[it] = b
            bins.pop()
            bw.pop()

    # remove item from a bin in O(1) (unordered)
    def remove_item_from_bin(bin_list: list[int], idx_in_bin: int) -> int:
        # returns the removed item
        last_item = bin_list[-1]
        removed = bin_list[idx_in_bin]
        bin_list[idx_in_bin] = last_item
        bin_list.pop()
        return removed

    BIG = C * (n + 1) + 10_000

    def energy(bw: list[int]) -> int:
        waste = 0
        for load in bw:
            waste += (C - load)
        return len(bw) * BIG + waste

    # ---------- Initial solution ----------
    bins, bw, item_bin = ffd_initial()
    normalize(bins, bw, item_bin)

    best_bins = [b[:] for b in bins]
    best_bw = bw[:]
    best_E = energy(best_bw)
    cur_E = best_E

    # ---------- SA parameters ----------
    avg_w = (sum(w) / n) if n else 1.0
    T0 = max(1.0, 0.5 * avg_w)
    Tmin = 1e-4

    # Neighborhood move probabilities
    p_relocate = 0.72

    stagnation = 0
    stag_limit = max(2_000, 20 * n)
    reheat_factor = 1.8

    # We will set max_iters dynamically to consume (most of) the time budget,
    # but it is still a fixed iteration count once determined.
    # Cooling alpha depends on max_iters, so we compute it after calibration.

    def try_relocate(T: float) -> bool:
        nonlocal cur_E

        if not bins:
            return False

        b1 = random.randrange(len(bins))
        for _ in range(3):
            if bins[b1]:
                break
            b1 = random.randrange(len(bins))
        if not bins[b1]:
            return False

        # choose an item position to remove in O(1)
        pos_i = random.randrange(len(bins[b1]))
        i = bins[b1][pos_i]
        wi = w[i]

        best_b2 = -1
        best_rem = C + 1

        k = 10 if len(bins) > 10 else len(bins)
        for _ in range(k):
            b2 = random.randrange(len(bins))
            if b2 == b1:
                continue
            rem = C - bw[b2]
            if wi <= rem:
                new_rem = rem - wi
                if new_rem < best_rem:
                    best_rem = new_rem
                    best_b2 = b2
                    if best_rem == 0:
                        break

        open_new = (best_b2 == -1)

        before_bins = len(bins)
        before_waste = (C - bw[b1]) + (0 if open_new else (C - bw[best_b2]))
        after_waste = (C - (bw[b1] - wi)) + (0 if open_new else (C - (bw[best_b2] + wi)))
        if open_new:
            after_waste += (C - wi)

        b1_will_empty = (len(bins[b1]) == 1)
        after_bins = before_bins
        if open_new:
            if not b1_will_empty:
                after_bins = before_bins + 1
        else:
            if b1_will_empty:
                after_bins = before_bins - 1

        dE = (after_bins - before_bins) * BIG + (after_waste - before_waste)

        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            # apply: remove from b1 in O(1)
            remove_item_from_bin(bins[b1], pos_i)
            bw[b1] -= wi

            if open_new:
                b2 = len(bins)
                bins.append([i])
                bw.append(wi)
                item_bin[i] = b2
            else:
                b2 = best_b2
                bins[b2].append(i)
                bw[b2] += wi
                item_bin[i] = b2

            if not bins[b1]:
                normalize(bins, bw, item_bin)

            cur_E += dE
            return True
        return False

    def try_swap(T: float) -> bool:
        nonlocal cur_E

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

        before_waste = (C - bw[b1]) + (C - bw[b2])
        after_waste = (C - load1) + (C - load2)
        dE = after_waste - before_waste

        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            # apply swap in-place
            bins[b1][p1] = j
            bins[b2][p2] = i
            bw[b1] = load1
            bw[b2] = load2
            item_bin[i] = b2
            item_bin[j] = b1
            cur_E += dE
            return True
        return False

    # ---------- Calibrate iteration budget to use the time limit ----------
    # Spend a small fraction of the budget to estimate moves/sec.
    now = time.perf_counter()
    remaining = deadline - now
    if remaining <= 0:
        return {"packing": best_bins, "bin_weights": best_bw}

    calib_time = min(0.25, 0.10 * remaining)  # up to 0.25s, or 10% of remaining
    calib_end = now + calib_time

    # Run a quick SA-like loop at constant temperature for calibration
    T_cal = T0
    calib_iters = 0
    while time.perf_counter() < calib_end:
        if random.random() < p_relocate:
            try_relocate(T_cal)
        else:
            try_swap(T_cal)
        calib_iters += 1

        if cur_E < best_E:
            best_E = cur_E
            best_bins = [b[:] for b in bins]
            best_bw = bw[:]

    elapsed_cal = max(1e-6, time.perf_counter() - now)
    iters_per_sec = calib_iters / elapsed_cal

    # Determine fixed iteration budget for the main loop to target ~95% usage.
    now2 = time.perf_counter()
    remaining2 = max(0.0, deadline - now2)
    target = 0.95 * remaining2

    # Safety caps/floors (still “fixed number of iterations” once computed)
    min_iters = 5_000 + 50 * n
    max_iters_cap = 5_000_000
    max_iters = int(max(min_iters, min(max_iters_cap, target * iters_per_sec)))

    # Cooling schedule based on chosen max_iters
    alpha = math.exp(math.log(Tmin / T0) / max(1, max_iters))

    # ---------- Main SA loop ----------
    T = T0
    for it in range(max_iters):
        if (it & 2047) == 0 and time_up():
            break

        moved = False
        if random.random() < p_relocate:
            moved = try_relocate(T)
        else:
            moved = try_swap(T)

        if moved:
            stagnation = 0
        else:
            stagnation += 1

        if cur_E < best_E:
            best_E = cur_E
            best_bins = [b[:] for b in bins]
            best_bw = bw[:]

        T *= alpha
        if T < Tmin:
            T = Tmin

        if stagnation >= stag_limit:
            T = min(T0, T * reheat_factor)
            stagnation = 0

    return {"packing": best_bins, "bin_weights": best_bw}
