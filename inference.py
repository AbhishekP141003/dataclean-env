"""
inference.py — Baseline inference script for DataClean OpenEnv
Follows OpenEnv structured log format: START / STEP / END
"""
import os
import json
import time
import requests
from openai import OpenAI

# ── Required environment variables ───────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")          # No default — must be set explicitly
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """You are a data cleaning agent. Fix dataset errors using ONE JSON action per turn.

Actions (pick the right one):
- {"action_type":"fix_value","row_index":N,"column":"col","new_value":X}
- {"action_type":"drop_row","row_index":N}
- {"action_type":"fill_missing","column":"col","new_value":X}
- {"action_type":"normalize_column","column":"col","column_transform":"to_int|to_float|to_date|lowercase|strip"}
- {"action_type":"mark_done"}

RULES FOR EACH ERROR TYPE:
- missing_value: use fill_missing with a sensible default (0 for numbers, "unknown" for strings)
- type_error: use normalize_column with to_int or to_float to fix the column
- duplicate: use drop_row with the duplicate row_index
- outlier: use fix_value to set purchase_amount to a value <= 10000
- consistency_total: use fix_value to set total = qty * unit_price (calculate it yourself)
- date_order: use fix_value to set ship_date to a date AFTER order_date (add 3 days)
- invalid_enum: use fix_value to set status to one of: pending, shipped, delivered, cancelled

Reply with ONE JSON object only. No explanation."""


def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    if method == "POST":
        resp = requests.post(url, json=payload, timeout=30)
    else:
        resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_user_message(obs: dict) -> str:
    errors = obs.get("errors", [])
    # Ultra compact — only what's needed to fix the first error
    slim_errors = [
        {"row": e.get("row_index"), "col": e.get("column"),
         "type": e.get("error_type"), "desc": e.get("description", "")[:80]}
        for e in errors[:3]
    ]
    # Get the rows that have errors to help the LLM
    error_rows = list({e.get("row_index") for e in errors[:3] if e.get("row_index") is not None})
    relevant_rows = [obs["dataset"][i] for i in error_rows if i < len(obs["dataset"])][:2]

    schema_str = ", ".join(
        f"{f.get('name')}:{f.get('expected_type')}"
        for f in obs.get("schema_fields", [])
    )
    return (
        f"Fixed:{obs['errors_fixed_so_far']}/{obs['total_errors_initial']} "
        f"Remaining:{len(errors)}\n"
        f"Schema:{schema_str}\n"
        f"Errors:{json.dumps(slim_errors)}\n"
        f"Relevant rows:{json.dumps(relevant_rows)}\n"
        f"Fix the first error. Reply ONE JSON action."
    )


def call_llm_with_retry(messages, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=120,
            )
            return response.choices[0].message.content.strip(), None
        except Exception as e:
            err_str = str(e)
            if "rate_limit_exceeded" in err_str or "429" in err_str or "413" in err_str:
                wait = 65 * (attempt + 1)
                print(json.dumps({
                    "type": "STEP", "step": 0,
                    "info": f"Rate limit, waiting {wait}s (retry {attempt+1}/{retries})"
                }))
                time.sleep(wait)
            else:
                return None, err_str
    return None, "Max retries exceeded"


def run_episode(task_id: str) -> dict:
    obs = call_env("POST", "/reset", {"task_id": task_id})
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    history = []
    done = obs.get("done", False)
    step = 0
    last_errors_remaining = 9999
    stuck_count = 0

    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "total_errors": obs.get("total_errors_initial", 0),
        "max_steps": {"task_easy": 30, "task_medium": 50, "task_hard": 80}.get(task_id, 50),
    }))

    while not done:
        user_msg = build_user_message(obs)
        # Keep only last 2 turns to save tokens
        recent = history[-2:] if len(history) > 2 else history
        messages = [system_msg] + recent + [{"role": "user", "content": user_msg}]

        raw_action, err = call_llm_with_retry(messages)

        if err:
            print(json.dumps({"type": "STEP", "step": step, "error": err}))
            break

        # Parse JSON
        try:
            if "```" in raw_action:
                raw_action = raw_action.split("```")[1]
                if raw_action.startswith("json"):
                    raw_action = raw_action[4:]
            action = json.loads(raw_action.strip())
        except json.JSONDecodeError:
            action = {"action_type": "mark_done"}

        # If stuck (no progress for 5 steps), clear history to reset LLM context
        errors_now = len(obs.get("errors", []))
        if errors_now >= last_errors_remaining:
            stuck_count += 1
            if stuck_count >= 5:
                history = []  # reset context so LLM gets fresh view
                stuck_count = 0
        else:
            stuck_count = 0
            last_errors_remaining = errors_now

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": json.dumps(action)})

        try:
            result = call_env("POST", "/step", action)
        except Exception as e:
            print(json.dumps({"type": "STEP", "step": step, "error": str(e)}))
            break

        reward_val = result["reward"]["value"]
        done       = result["done"]
        obs        = result["observation"]
        step      += 1

        print(json.dumps({
            "type": "STEP",
            "step": step,
            "action": action.get("action_type"),
            "reward": round(reward_val, 4),
            "errors_remaining": len(obs.get("errors", [])),
            "errors_fixed": obs.get("errors_fixed_so_far", 0),
            "done": done,
        }))

        if step >= 100:
            break

    try:
        grade_result = call_env("GET", "/grade")
        score     = grade_result.get("score", 0.0)
        grade     = grade_result.get("grade", "unknown")
        breakdown = grade_result.get("breakdown", {})
    except Exception:
        score, grade, breakdown = 0.0, "error", {}

    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "score": round(score, 4),
        "grade": grade,
        "total_steps": step,
        "breakdown": breakdown,
    }))

    return {"task_id": task_id, "score": score, "grade": grade, "steps": step}


def main():
    try:
        health = call_env("GET", "/health")
        assert health.get("openenv") is True
    except Exception as e:
        print(json.dumps({"type": "ERROR", "message": f"Environment not reachable: {e}"}))
        exit(1)

    results = []
    for task_id in TASKS:
        result = run_episode(task_id)
        results.append(result)

    avg = sum(r["score"] for r in results) / len(results)

    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "average_score": round(avg, 4),
        }, f, indent=2)


if __name__ == "__main__":
    main()
