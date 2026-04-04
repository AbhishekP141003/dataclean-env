import copy
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action, ActionType, ColumnTransform, EnvState, Observation,
    Reward, SchemaField, StepResult, ValidationError,
)
from env.datasets import TASK_FACTORIES


TASK_META = {
    "task_easy": {
        "name": "Fix Missing Values & Type Errors",
        "difficulty": "easy",
        "max_steps": 30,
        "description": (
            "A 20-row employee dataset has missing values (None) and wrong data types "
            "(e.g. age stored as '30yrs'). Fix all nulls with sensible defaults and coerce "
            "each column to its correct type. Use mark_done when finished."
        ),
    },
    "task_medium": {
        "name": "Detect & Repair Duplicates + Outliers",
        "difficulty": "medium",
        "max_steps": 50,
        "description": (
            "A 45-50-row customer purchase dataset contains exact duplicate rows and "
            "purchase_amount outliers (> 10,000). Drop duplicate rows and fix outlier "
            "amounts to a reasonable value (≤ 10,000). Use mark_done when finished."
        ),
    },
    "task_hard": {
        "name": "Multi-column Consistency & Referential Integrity",
        "difficulty": "hard",
        "max_steps": 80,
        "description": (
            "An 80-row order dataset has three kinds of errors: "
            "(1) total ≠ qty × unit_price, "
            "(2) ship_date < order_date, "
            "(3) status not in {pending, shipped, delivered, cancelled}. "
            "Fix all violations. Use mark_done when finished."
        ),
    },
}


class DataCleaningEnv:
    """OpenEnv-compliant data cleaning and validation environment."""

    def __init__(self):
        self._task_id: Optional[str] = None
        self._dataset: List[Dict[str, Any]] = []
        self._clean_ref: List[Dict[str, Any]] = []
        self._injected_errors: List[Dict] = []
        self._schema: List[SchemaField] = []
        self._step_count: int = 0
        self._done: bool = False
        self._errors_fixed: int = 0
        self._collateral: int = 0
        self._max_steps: int = 30
        self._initial_error_count: int = 0

    # ── Public OpenEnv API ──────────────────────────────────────────────────

    def reset(self, task_id: str = "task_easy") -> Observation:
        if task_id not in TASK_FACTORIES:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASK_FACTORIES)}")

        factory_fn, schema = TASK_FACTORIES[task_id]
        self._task_id = task_id
        self._schema = schema
        self._dataset, self._clean_ref, self._injected_errors = factory_fn()
        self._step_count = 0
        self._done = False
        self._errors_fixed = 0
        self._collateral = 0
        self._max_steps = TASK_META[task_id]["max_steps"]

        errors = self._detect_errors()
        self._initial_error_count = len(errors)

        return self._make_observation(errors)

    def step(self, action: Action) -> StepResult:
        if self._done:
            obs = self._make_observation(self._detect_errors())
            return StepResult(
                observation=obs,
                reward=Reward(value=0.0, reason="Episode already done"),
                done=True,
                info={"warning": "Episode is finished. Call reset()."},
            )

        self._step_count += 1
        errors_before = self._detect_errors()
        fixed_this_step = 0
        damage_this_step = 0
        info: Dict[str, Any] = {}

        try:
            fixed_this_step, damage_this_step, info = self._apply_action(action)
        except Exception as e:
            info["error"] = str(e)

        errors_after = self._detect_errors()
        self._errors_fixed = self._initial_error_count - len(errors_after)

        # Determine done
        no_errors = len(errors_after) == 0
        max_steps_hit = self._step_count >= self._max_steps
        agent_done = action.action_type == ActionType.mark_done
        self._done = no_errors or max_steps_hit or agent_done

        reward = self._compute_reward(
            errors_before=len(errors_before),
            errors_after=len(errors_after),
            damage=damage_this_step,
            done=self._done,
        )

        obs = self._make_observation(errors_after)
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> EnvState:
        errors = self._detect_errors()
        score = self._errors_fixed / max(self._initial_error_count, 1)
        return EnvState(
            task_id=self._task_id or "",
            step_count=self._step_count,
            done=self._done,
            errors_remaining=len(errors),
            errors_fixed=self._errors_fixed,
            total_errors_initial=self._initial_error_count,
            score=round(score, 4),
            dataset_snapshot=copy.deepcopy(self._dataset),
        )

    # ── Action application ──────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> Tuple[int, int, Dict]:
        info = {}
        fixed = 0
        damage = 0

        if action.action_type == ActionType.mark_done:
            return 0, 0, {"message": "Agent marked episode done"}

        if action.action_type == ActionType.fix_value:
            row, col, val = action.row_index, action.column, action.new_value
            if row is None or col is None or val is None:
                raise ValueError("fix_value requires row_index, column, and new_value")
            if 0 <= row < len(self._dataset):
                old = self._dataset[row].get(col)
                self._dataset[row][col] = val
                # Did we fix an actual error?
                if self._was_error(row, col, old):
                    fixed = 1
                    info["fixed"] = f"row={row} col={col}"
                elif self._broke_clean(row, col, val):
                    damage = 1
                    info["damage"] = f"Overwrote correct value at row={row} col={col}"

        elif action.action_type == ActionType.drop_row:
            row = action.row_index
            if row is None:
                raise ValueError("drop_row requires row_index")
            if 0 <= row < len(self._dataset):
                removed = self._dataset.pop(row)
                # Check if it was a duplicate
                cid = removed.get("customer_id")
                seen = [r.get("customer_id") for r in self._dataset]
                if any(e["kind"] == "duplicate" and e["row"] == row for e in self._injected_errors):
                    fixed = 1
                    info["fixed"] = f"Dropped duplicate row={row}"
                else:
                    damage = 1
                    info["damage"] = f"Dropped non-duplicate row={row}"
                # Adjust injected error row indices
                self._injected_errors = [
                    e for e in self._injected_errors if e.get("row") != row
                ]
                for e in self._injected_errors:
                    if e.get("row", 0) > row:
                        e["row"] -= 1

        elif action.action_type == ActionType.fill_missing:
            col = action.column
            val = action.new_value
            if col is None or val is None:
                raise ValueError("fill_missing requires column and new_value")
            count = 0
            for i, row in enumerate(self._dataset):
                if row.get(col) is None:
                    self._dataset[i][col] = val
                    count += 1
            fixed = count
            info["filled"] = f"{count} nulls in '{col}'"

        elif action.action_type == ActionType.normalize_column:
            col = action.column
            transform = action.column_transform
            if col is None or transform is None:
                raise ValueError("normalize_column requires column and column_transform")
            count = 0
            for i, row in enumerate(self._dataset):
                old_val = row.get(col)
                new_val = self._apply_transform(old_val, transform)
                if new_val != old_val:
                    self._dataset[i][col] = new_val
                    count += 1
            fixed = min(count, len(self._injected_errors))
            info["normalized"] = f"{count} values in '{col}' via {transform}"

        return fixed, damage, info

    def _apply_transform(self, val: Any, transform: ColumnTransform) -> Any:
        if val is None:
            return val
        if transform == ColumnTransform.lowercase:
            return str(val).lower()
        if transform == ColumnTransform.strip:
            return str(val).strip()
        if transform == ColumnTransform.to_int:
            cleaned = re.sub(r"[^\d\-]", "", str(val))
            return int(cleaned) if cleaned else val
        if transform == ColumnTransform.to_float:
            cleaned = re.sub(r"[^\d\.\-]", "", str(val))
            return float(cleaned) if cleaned else val
        if transform == ColumnTransform.to_date:
            return str(val)[:10]
        return val

    # ── Error detection ─────────────────────────────────────────────────────

    def _detect_errors(self) -> List[ValidationError]:
        errors: List[ValidationError] = []
        schema_map = {f.name: f for f in self._schema}

        if self._task_id == "task_easy":
            errors += self._check_nulls_and_types(schema_map)

        elif self._task_id == "task_medium":
            errors += self._check_nulls_and_types(schema_map)
            errors += self._check_duplicates()
            errors += self._check_outliers(schema_map)

        elif self._task_id == "task_hard":
            errors += self._check_nulls_and_types(schema_map)
            errors += self._check_total_consistency()
            errors += self._check_date_order()
            errors += self._check_enum("status", {"pending", "shipped", "delivered", "cancelled"})

        return errors

    def _check_nulls_and_types(self, schema_map: Dict) -> List[ValidationError]:
        errors = []
        for i, row in enumerate(self._dataset):
            for col, field in schema_map.items():
                val = row.get(col)
                if val is None and not field.nullable:
                    errors.append(ValidationError(
                        row_index=i, column=col,
                        error_type="missing_value",
                        description=f"Row {i}: '{col}' is null but required",
                        severity="high",
                    ))
                elif val is not None:
                    if field.expected_type == "int":
                        try:
                            int(str(val).replace(",", ""))
                            if not isinstance(val, (int,)):
                                raise ValueError
                        except (ValueError, TypeError):
                            errors.append(ValidationError(
                                row_index=i, column=col,
                                error_type="type_error",
                                description=f"Row {i}: '{col}'={val!r} cannot be cast to int",
                                severity="medium",
                            ))
                    elif field.expected_type == "float":
                        try:
                            float(str(val).replace(",", ""))
                        except (ValueError, TypeError):
                            errors.append(ValidationError(
                                row_index=i, column=col,
                                error_type="type_error",
                                description=f"Row {i}: '{col}'={val!r} cannot be cast to float",
                            ))
        return errors

    def _check_duplicates(self) -> List[ValidationError]:
        errors = []
        seen = {}
        for i, row in enumerate(self._dataset):
            key = row.get("customer_id")
            if key in seen:
                errors.append(ValidationError(
                    row_index=i, column="customer_id",
                    error_type="duplicate",
                    description=f"Row {i}: duplicate customer_id '{key}' (first at row {seen[key]})",
                    severity="high",
                ))
            else:
                seen[key] = i
        return errors

    def _check_outliers(self, schema_map: Dict) -> List[ValidationError]:
        errors = []
        for i, row in enumerate(self._dataset):
            for col, field in schema_map.items():
                if field.constraints and "max" in field.constraints:
                    val = row.get(col)
                    if val is not None:
                        try:
                            if float(val) > field.constraints["max"]:
                                errors.append(ValidationError(
                                    row_index=i, column=col,
                                    error_type="outlier",
                                    description=f"Row {i}: '{col}'={val} exceeds max {field.constraints['max']}",
                                    severity="medium",
                                ))
                        except (ValueError, TypeError):
                            pass
        return errors

    def _check_total_consistency(self) -> List[ValidationError]:
        errors = []
        for i, row in enumerate(self._dataset):
            try:
                expected = round(float(row["qty"]) * float(row["unit_price"]), 2)
                actual = float(row["total"])
                if abs(expected - actual) > 0.02:
                    errors.append(ValidationError(
                        row_index=i, column="total",
                        error_type="consistency_total",
                        description=f"Row {i}: total={actual} but qty*unit_price={expected}",
                        severity="high",
                    ))
            except (KeyError, TypeError, ValueError):
                pass
        return errors

    def _check_date_order(self) -> List[ValidationError]:
        errors = []
        for i, row in enumerate(self._dataset):
            try:
                od = date.fromisoformat(str(row["order_date"])[:10])
                sd = date.fromisoformat(str(row["ship_date"])[:10])
                if sd < od:
                    errors.append(ValidationError(
                        row_index=i, column="ship_date",
                        error_type="date_order",
                        description=f"Row {i}: ship_date={sd} is before order_date={od}",
                        severity="high",
                    ))
            except (KeyError, TypeError, ValueError):
                pass
        return errors

    def _check_enum(self, col: str, valid_values: set) -> List[ValidationError]:
        errors = []
        for i, row in enumerate(self._dataset):
            val = row.get(col)
            if val not in valid_values:
                errors.append(ValidationError(
                    row_index=i, column=col,
                    error_type="invalid_enum",
                    description=f"Row {i}: '{col}'={val!r} not in {valid_values}",
                    severity="medium",
                ))
        return errors

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _was_error(self, row: int, col: str, old_val: Any) -> bool:
        """Check if the cell that was changed was indeed an error."""
        for e in self._injected_errors:
            if e.get("row") == row and e.get("col") == col:
                return True
        if old_val is None:
            return True
        return False

    def _broke_clean(self, row: int, col: str, new_val: Any) -> bool:
        """Check if agent overwrote a previously correct value."""
        if row < len(self._clean_ref):
            ref = self._clean_ref[row].get(col)
            if ref is not None and new_val != ref:
                for e in self._injected_errors:
                    if e.get("row") == row and e.get("col") == col:
                        return False
                return True
        return False

    def _compute_reward(self, errors_before: int, errors_after: int,
                        damage: int, done: bool) -> Reward:
        progress = max(0, errors_before - errors_after)
        partial = progress / max(self._initial_error_count, 1)

        base = self._errors_fixed / max(self._initial_error_count, 1)
        penalty = damage * 0.05
        step_penalty = (self._step_count / self._max_steps) * 0.05
        value = max(0.0, min(1.0, base - penalty - step_penalty))

        return Reward(
            value=round(value, 4),
            errors_fixed=self._errors_fixed,
            collateral_damage=damage,
            partial_progress=round(partial, 4),
            reason=f"Fixed {progress} error(s) this step. Total fixed: {self._errors_fixed}/{self._initial_error_count}",
        )

    def _make_observation(self, errors: List[ValidationError]) -> Observation:
        meta = TASK_META[self._task_id]
        return Observation(
            dataset=copy.deepcopy(self._dataset),
            schema_fields=self._schema,
            errors=errors,
            step_count=self._step_count,
            task_id=self._task_id,
            task_description=meta["description"],
            total_errors_initial=self._initial_error_count,
            errors_fixed_so_far=self._errors_fixed,
            done=self._done,
        )
