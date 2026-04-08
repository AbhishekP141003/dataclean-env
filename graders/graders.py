"""
Agent graders — deterministic programmatic scoring for each task.
Scores are strictly between 0 and 1: range (0.01, 0.99)
"""
from typing import Any, Dict
from env.models import EnvState


def clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (not 0.0 or 1.0)."""
    return round(max(0.01, min(0.99, score)), 4)


def grade_task_easy(state: EnvState) -> Dict[str, Any]:
    if state.total_errors_initial == 0:
        return {"score": 0.99, "grade": "excellent",
                "reason": "No errors to fix", "task_id": "task_easy", "breakdown": {}}

    raw = state.errors_fixed / state.total_errors_initial
    score = clamp(raw)

    breakdown = {
        "errors_fixed": state.errors_fixed,
        "total_errors": state.total_errors_initial,
        "errors_remaining": state.errors_remaining,
        "steps_used": state.step_count,
    }

    if score >= 0.90:   grade = "excellent"
    elif score >= 0.70: grade = "good"
    elif score >= 0.40: grade = "partial"
    else:               grade = "poor"

    return {
        "score": score,
        "grade": grade,
        "reason": f"Fixed {state.errors_fixed}/{state.total_errors_initial} errors",
        "breakdown": breakdown,
        "task_id": "task_easy",
    }


def grade_task_medium(state: EnvState) -> Dict[str, Any]:
    dataset = state.dataset_snapshot

    seen_ids = {}
    dup_remaining = 0
    for row in dataset:
        cid = row.get("customer_id")
        if cid in seen_ids:
            dup_remaining += 1
        else:
            seen_ids[cid] = True

    outlier_remaining = sum(
        1 for row in dataset
        if row.get("purchase_amount") is not None
        and float(row["purchase_amount"]) > 10000
    )

    EXPECTED_DUPS     = 5
    EXPECTED_OUTLIERS = 5
    dup_score     = max(0.0, 1.0 - dup_remaining / EXPECTED_DUPS)
    outlier_score = max(0.0, 1.0 - outlier_remaining / EXPECTED_OUTLIERS)
    raw   = 0.4 * dup_score + 0.6 * outlier_score
    score = clamp(raw)

    if score >= 0.90:   grade = "excellent"
    elif score >= 0.70: grade = "good"
    elif score >= 0.40: grade = "partial"
    else:               grade = "poor"

    return {
        "score": score,
        "grade": grade,
        "reason": f"Dup score={dup_score:.2f}, Outlier score={outlier_score:.2f}",
        "breakdown": {
            "duplicates_remaining": dup_remaining,
            "outliers_remaining": outlier_remaining,
            "dup_score": round(dup_score, 4),
            "outlier_score": round(outlier_score, 4),
        },
        "task_id": "task_medium",
    }


def grade_task_hard(state: EnvState) -> Dict[str, Any]:
    from datetime import date as date_cls

    dataset = state.dataset_snapshot
    VALID_STATUSES = {"pending", "shipped", "delivered", "cancelled"}

    total_errors = 0
    date_errors  = 0
    enum_errors  = 0

    for row in dataset:
        try:
            expected = round(float(row["qty"]) * float(row["unit_price"]), 2)
            actual   = float(row["total"])
            if abs(expected - actual) > 0.02:
                total_errors += 1
        except (KeyError, TypeError, ValueError):
            total_errors += 1

        try:
            od = date_cls.fromisoformat(str(row["order_date"])[:10])
            sd = date_cls.fromisoformat(str(row["ship_date"])[:10])
            if sd < od:
                date_errors += 1
        except (KeyError, TypeError, ValueError):
            date_errors += 1

        if row.get("status") not in VALID_STATUSES:
            enum_errors += 1

    EXPECTED_TOTAL  = 8
    EXPECTED_DATE   = 6
    EXPECTED_ENUM   = 4

    total_score = max(0.0, 1.0 - total_errors / EXPECTED_TOTAL)
    date_score  = max(0.0, 1.0 - date_errors  / EXPECTED_DATE)
    enum_score  = max(0.0, 1.0 - enum_errors  / EXPECTED_ENUM)

    raw   = (total_score + date_score + enum_score) / 3.0
    score = clamp(raw)

    if score >= 0.90:   grade = "excellent"
    elif score >= 0.70: grade = "good"
    elif score >= 0.40: grade = "partial"
    else:               grade = "poor"

    return {
        "score": score,
        "grade": grade,
        "reason": f"Total={total_score:.2f}, Date={date_score:.2f}, Enum={enum_score:.2f}",
        "breakdown": {
            "total_consistency_errors": total_errors,
            "date_order_errors": date_errors,
            "enum_errors": enum_errors,
            "total_score": round(total_score, 4),
            "date_score": round(date_score, 4),
            "enum_score": round(enum_score, 4),
        },
        "task_id": "task_hard",
    }


GRADERS = {
    "task_easy":   grade_task_easy,
    "task_medium": grade_task_medium,
    "task_hard":   grade_task_hard,
}


def run_grader(task_id: str, state: EnvState) -> Dict[str, Any]:
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task '{task_id}'")
    return GRADERS[task_id](state)
