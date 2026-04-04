import random
import copy
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple
from env.models import SchemaField, ValidationError


# ── TASK EASY: Missing values & type errors ──────────────────────────────────

EASY_SCHEMA = [
    SchemaField(name="id", expected_type="int", nullable=False),
    SchemaField(name="name", expected_type="str", nullable=False),
    SchemaField(name="age", expected_type="int", nullable=False, constraints={"min": 0, "max": 120}),
    SchemaField(name="salary", expected_type="float", nullable=False, constraints={"min": 0}),
    SchemaField(name="email", expected_type="str", nullable=False),
]

def make_easy_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (dirty_dataset, clean_reference)"""
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
             "Hank", "Iris", "Jack", "Karen", "Leo", "Mia", "Ned", "Olivia",
             "Paul", "Quinn", "Rose", "Sam", "Tina"]
    clean = []
    for i, name in enumerate(names):
        clean.append({
            "id": i + 1,
            "name": name,
            "age": random.randint(22, 60),
            "salary": round(random.uniform(30000, 120000), 2),
            "email": f"{name.lower()}@example.com",
        })

    dirty = copy.deepcopy(clean)
    injected_errors = []

    # Inject ~8 errors
    error_rows = random.sample(range(20), 8)
    for idx in error_rows:
        error_type = random.choice(["missing_age", "missing_salary", "bad_age_type", "missing_email"])
        if error_type == "missing_age":
            dirty[idx]["age"] = None
            injected_errors.append({"row": idx, "col": "age", "kind": "missing"})
        elif error_type == "missing_salary":
            dirty[idx]["salary"] = None
            injected_errors.append({"row": idx, "col": "salary", "kind": "missing"})
        elif error_type == "bad_age_type":
            dirty[idx]["age"] = str(dirty[idx]["age"]) + "yrs"
            injected_errors.append({"row": idx, "col": "age", "kind": "type_error"})
        elif error_type == "missing_email":
            dirty[idx]["email"] = None
            injected_errors.append({"row": idx, "col": "email", "kind": "missing"})

    return dirty, clean, injected_errors


# ── TASK MEDIUM: Duplicates & outliers ───────────────────────────────────────

MEDIUM_SCHEMA = [
    SchemaField(name="customer_id", expected_type="str", nullable=False),
    SchemaField(name="first_name", expected_type="str", nullable=False),
    SchemaField(name="last_name", expected_type="str", nullable=False),
    SchemaField(name="purchase_amount", expected_type="float", nullable=False,
                constraints={"min": 0, "max": 10000}),
    SchemaField(name="purchase_date", expected_type="str", nullable=False),
]

def make_medium_dataset() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    first_names = ["Anna","Ben","Cara","Dan","Ella","Finn","Gina","Hugo",
                   "Ida","Jake","Kira","Liam","Maya","Nick","Opal","Pete",
                   "Rita","Seth","Tara","Uma","Vince","Wren","Xena","Yara","Zara"]
    base_date = date(2024, 1, 1)
    clean = []
    for i in range(40):
        fn = first_names[i % len(first_names)]
        clean.append({
            "customer_id": f"C{i+1:04d}",
            "first_name": fn,
            "last_name": f"Smith{i}",
            "purchase_amount": round(random.uniform(10, 2000), 2),
            "purchase_date": str(base_date + timedelta(days=random.randint(0, 365))),
        })

    dirty = copy.deepcopy(clean)
    injected_errors = []

    # Inject 5 duplicates
    dup_src = random.sample(range(40), 5)
    for src in dup_src:
        dup = copy.deepcopy(dirty[src])
        dirty.append(dup)
        injected_errors.append({"row": len(dirty) - 1, "col": "customer_id", "kind": "duplicate"})

    # Inject 5 outliers in purchase_amount
    outlier_rows = random.sample(range(40), 5)
    for idx in outlier_rows:
        dirty[idx]["purchase_amount"] = round(random.uniform(50000, 999999), 2)
        injected_errors.append({"row": idx, "col": "purchase_amount", "kind": "outlier"})

    return dirty, clean, injected_errors


# ── TASK HARD: Cross-column consistency ──────────────────────────────────────

HARD_SCHEMA = [
    SchemaField(name="order_id", expected_type="str", nullable=False),
    SchemaField(name="product", expected_type="str", nullable=False),
    SchemaField(name="qty", expected_type="int", nullable=False, constraints={"min": 1}),
    SchemaField(name="unit_price", expected_type="float", nullable=False, constraints={"min": 0}),
    SchemaField(name="total", expected_type="float", nullable=False),
    SchemaField(name="order_date", expected_type="str", nullable=False),
    SchemaField(name="ship_date", expected_type="str", nullable=False),
    SchemaField(name="status", expected_type="str", nullable=False,
                constraints={"enum": ["pending", "shipped", "delivered", "cancelled"]}),
]

PRODUCTS = [("Widget A", 9.99), ("Widget B", 14.99), ("Gadget X", 49.99),
            ("Gadget Y", 79.99), ("Tool Z", 24.99)]

def make_hard_dataset() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    base_date = date(2024, 3, 1)
    clean = []
    for i in range(80):
        product, price = random.choice(PRODUCTS)
        qty = random.randint(1, 20)
        order_dt = base_date + timedelta(days=random.randint(0, 180))
        ship_dt = order_dt + timedelta(days=random.randint(1, 14))
        clean.append({
            "order_id": f"ORD-{i+1:04d}",
            "product": product,
            "qty": qty,
            "unit_price": price,
            "total": round(qty * price, 2),
            "order_date": str(order_dt),
            "ship_date": str(ship_dt),
            "status": random.choice(["pending", "shipped", "delivered"]),
        })

    dirty = copy.deepcopy(clean)
    injected_errors = []

    # Wrong total (qty * price mismatch)
    total_err_rows = random.sample(range(80), 8)
    for idx in total_err_rows:
        dirty[idx]["total"] = round(dirty[idx]["total"] * random.uniform(1.5, 3.0), 2)
        injected_errors.append({"row": idx, "col": "total", "kind": "consistency_total"})

    # ship_date < order_date
    date_err_rows = random.sample([r for r in range(80) if r not in total_err_rows], 6)
    for idx in date_err_rows:
        od = date.fromisoformat(dirty[idx]["order_date"])
        dirty[idx]["ship_date"] = str(od - timedelta(days=random.randint(1, 10)))
        injected_errors.append({"row": idx, "col": "ship_date", "kind": "date_order"})

    # Invalid status
    status_err_rows = random.sample(
        [r for r in range(80) if r not in total_err_rows and r not in date_err_rows], 4)
    for idx in status_err_rows:
        dirty[idx]["status"] = random.choice(["unknown", "processing", "N/A"])
        injected_errors.append({"row": idx, "col": "status", "kind": "invalid_enum"})

    return dirty, clean, injected_errors


TASK_FACTORIES = {
    "task_easy": (make_easy_dataset, EASY_SCHEMA),
    "task_medium": (make_medium_dataset, MEDIUM_SCHEMA),
    "task_hard": (make_hard_dataset, HARD_SCHEMA),
}
