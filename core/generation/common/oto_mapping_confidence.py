from __future__ import annotations


def evaluate_index_plan(score_rows, planned_indices):
    if not score_rows or not planned_indices:
        return {
            "available": False,
            "total_score": 0.0,
            "mean_score": 0.0,
            "min_score": 0.0,
            "margin": 0.0,
        }

    rows = [list(row or []) for row in score_rows]
    plan = [int(idx) for idx in planned_indices]
    if len(rows) != len(plan):
        return {
            "available": False,
            "total_score": 0.0,
            "mean_score": 0.0,
            "min_score": 0.0,
            "margin": 0.0,
        }

    chosen_scores = []
    margins = []
    for row, idx in zip(rows, plan):
        if not (0 <= idx < len(row)):
            return {
                "available": False,
                "total_score": 0.0,
                "mean_score": 0.0,
                "min_score": 0.0,
                "margin": 0.0,
            }
        chosen = float(row[idx])
        chosen_scores.append(chosen)
        best_other = None
        for j, score in enumerate(row):
            if j == idx:
                continue
            value = float(score)
            if best_other is None or value > best_other:
                best_other = value
        margins.append(chosen - (best_other if best_other is not None else chosen))

    total_score = float(sum(chosen_scores))
    mean_score = total_score / float(max(len(chosen_scores), 1))
    min_score = float(min(chosen_scores)) if chosen_scores else 0.0
    margin = float(sum(margins) / float(max(len(margins), 1))) if margins else 0.0
    return {
        "available": True,
        "total_score": total_score,
        "mean_score": mean_score,
        "min_score": min_score,
        "margin": margin,
    }


__all__ = ["evaluate_index_plan"]
