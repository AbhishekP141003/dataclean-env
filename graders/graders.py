"""
Agent graders — deterministic programmatic scoring (0.0–1.0) for each task.
Called after an episode ends to produce a final reproducible score.
"""
from typing import Any, Dict, List
from env.models import EnvState


def grade_task_easy(state: EnvState) -> Dict[str, Any]:
    """
    Easy task grader: proportion of original errors fixed without collateral damage.
    Score = errors_fixed / total_errors_initial  (capped at 1.0)
    """
    if state.total_errors_initial == 0:
        return {"score": 1.0, "reason": "No errors to fix (perfect dataset)", "task_id": "task_easy"}

    score = min(1.0, state.errors_fixed / state.total_errors_initial)

    breakdown = {
        "errors_fixed": state.errors_fixed,
        "total_errors": state.total_errors_initial,
        "errors_remaining": state.errors_remaining,
        "steps_used": state.step_count,
    }

    if score >= 0.9:
        grade = "excellent"
    elif score >= 0.7:
        grade = "good"
    elif score >= 0.4:
        grade = "partial"
    else:
        grade = "poor"

    return {
        "score": round(score, 4),
        "grade": grade,
        "reason": f"Fixed {state.errors_fixed}/{state.total_errors_initial} errors",
        "breakdown": breakdown,
        "task_id": "task_easy",
    }


def grade_task_medium(state: EnvState) -> Dict[str, Any]:
    """
    Medium grader: weighted score — duplicates worth 40%, outliers worth 60%.
    Checks dataset_snapshot for remaining issues.
    """
    dataset = state.dataset_snapshot

    # Count remaining duplicates
    seen_ids = {}
    dup_remaining = 0
    for row in dataset:
        cid = row.get("customer_id")
        if cid in seen_ids:
            dup_remaining += 1
        else:
            seen_ids[cid] = True

    # Count remaining outliers (purchase_amount > 10000)
    outlier_remaining = sum(
        1 for row in dataset
        if row.get("purchase_amount") is not None and float(row["purchase_amount"]) > 10000
    )

    # Score components (we expect ~5 dups and ~5 outliers injected)
    EXPECTED_DUPS = 5
    EXPECTED_OUTLIERS = 5
    dup_score = max(0.0, 1.0 - dup_remaining / EXPECTED_DUPS)
    outlier_score = max(0.0, 1.0 - outlier_remaining / EXPECTED_OUTLIERS)

    score = round(0.4 * dup_score + 0.6 * outlier_score, 4)

    return {
        "score": score,
        "grade": "excellent" if score >= 0.9 else "good" if score >= 0.7 else "partial" if score >= 0.4 else "poor",
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
    """
    Hard grader: three equal-weight components.
    1. Total consistency errors (total != qty * price)
    2. Date order errors (ship_date < order_date)
    3. Invalid status enum values
    """
    from datetime import date as date_cls

    dataset = state.dataset_snapshot
    VALID_STATUSES = {"pending", "shipped", "delivered", "cancelled"}

    total_errors = 0
    date_errors = 0
    enum_errors = 0

    for row in dataset:
        # Check total consistency
        try:
            expected = round(float(row["qty"]) * float(row["unit_price"]), 2)
            actual = float(row["total"])
            if abs(expected - actual) > 0.02:
                total_errors += 1
        except (KeyError, TypeError, ValueError):
            total_errors += 1

        # Check date order
        try:
            od = date_cls.fromisoformat(str(row["order_date"])[:10])
            sd = date_cls.fromisoformat(str(row["ship_date"])[:10])
            if sd < od:
                date_errors += 1
        except (KeyError, TypeError, ValueError):
            date_errors += 1

        # Check enum
        if row.get("status") not in VALID_STATUSES:
            enum_errors += 1

    EXPECTED_TOTAL_ERRORS = 8
    EXPECTED_DATE_ERRORS = 6
    EXPECTED_ENUM_ERRORS = 4

    total_score = max(0.0, 1.0 - total_errors / EXPECTED_TOTAL_ERRORS)
    date_score = max(0.0, 1.0 - date_errors / EXPECTED_DATE_ERRORS)
    enum_score = max(0.0, 1.0 - enum_errors / EXPECTED_ENUM_ERRORS)

    score = round((total_score + date_score + enum_score) / 3.0, 4)

    return {
        "score": score,
        "grade": "excellent" if score >= 0.9 else "good" if score >= 0.7 else "partial" if score >= 0.4 else "poor",
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
    "task_easy": grade_task_easy,
    "task_medium": grade_task_medium,
    "task_hard": grade_task_hard,
}


def run_grader(task_id: str, state: EnvState) -> Dict[str, Any]:
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task '{task_id}'")
    return GRADERS[task_id](state)
