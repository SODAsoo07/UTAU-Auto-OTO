from __future__ import annotations


def build_monotonic_index_plan(score_rows):
    """
    score_rows[i][j]:
    target i를 candidate j에 배정했을 때의 점수.

    반환값은 strictly increasing candidate index sequence.
    유효한 계획이 없으면 None.
    """
    if not score_rows:
        return []
    rows = [list(row or []) for row in score_rows]
    if not rows or not rows[0]:
        return None

    target_count = len(rows)
    cand_count = len(rows[0])
    if cand_count < target_count:
        return None
    if any(len(row) != cand_count for row in rows):
        return None

    neg_inf = float("-inf")
    dp = [[neg_inf] * cand_count for _ in range(target_count)]
    prev = [[-1] * cand_count for _ in range(target_count)]

    for i in range(target_count):
        lo = i
        hi = cand_count - (target_count - i)
        for j in range(lo, hi + 1):
            try:
                score = float(rows[i][j])
            except Exception:
                continue
            if i == 0:
                dp[i][j] = score
                continue
            best_prev = neg_inf
            best_prev_idx = -1
            for k in range(i - 1, j):
                val = dp[i - 1][k]
                if val > best_prev:
                    best_prev = val
                    best_prev_idx = k
            if best_prev_idx < 0:
                continue
            dp[i][j] = best_prev + score
            prev[i][j] = best_prev_idx

    best_last = -1
    best_score = neg_inf
    for j in range(target_count - 1, cand_count):
        if dp[target_count - 1][j] > best_score:
            best_score = dp[target_count - 1][j]
            best_last = j
    if best_last < 0:
        return None

    plan = [0] * target_count
    cur = best_last
    for i in range(target_count - 1, -1, -1):
        if cur < 0:
            return None
        plan[i] = int(cur)
        cur = prev[i][cur] if i > 0 else -1
    return plan


__all__ = ["build_monotonic_index_plan"]
