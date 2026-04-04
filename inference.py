"""
inference.py — Baseline inference script for DataClean OpenEnv
Follows OpenEnv structured log format: START / STEP / END
"""
import os
import json
import requests
from openai import OpenAI

# ── Required environment variables (checklist items 2 & 3) ───────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")          # No default — must be set explicitly
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

# ── OpenAI client configured via the required variables (checklist item 4) ───
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """You are an expert data cleaning agent. You will receive a dataset with errors and must fix them.

Available action_types:
- fix_value: Fix a specific cell. Requires row_index (int), column (str), new_value (any)
- drop_row: Drop a duplicate row. Requires row_index (int)
- fill_missing: Fill all nulls in a column. Requires column (str), new_value (any)
- normalize_column: Transform a whole column. Requires column (str), column_transform (one of: lowercase, strip, to_int, to_float, to_date)
- mark_done: Signal that you are finished

Respond with a single valid JSON object only:
{
  "action_type": "<action_type>",
  "row_index": <int or null>,
  "column": "<str or null>",
  "new_value": <value or null>,
  "column_transform": "<transform or null>"
}

No explanation outside the JSON."""


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
    dataset_preview = obs["dataset"][:5]
    return f"""Task: {obs['task_description']}

Step: {obs['step_count']}
Initial errors: {obs['total_errors_initial']}
Errors fixed so far: {obs['errors_fixed_so_far']}
Errors remaining: {len(errors)}

Current errors (first 10):
{json.dumps(errors[:10], indent=2)}

Dataset preview (first 5 rows):
{json.dumps(dataset_preview, indent=2)}

Schema:
{json.dumps(obs['schema_fields'], indent=2)}

What action will you take next?"""


def run_episode(task_id: str) -> dict:
    obs = call_env("POST", "/reset", {"task_id": task_id})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = obs.get("done", False)
    step = 0

    # ── START log (required structured format) ───────────────────────────────
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "total_errors": obs.get("total_errors_initial", 0),
        "max_steps": {"task_easy": 30, "task_medium": 50, "task_hard": 80}.get(task_id, 50),
    }))

    while not done:
        user_msg = build_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

        # LLM call via OpenAI client
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            raw_action = response.choices[0].message.content.strip()
        except Exception as e:
            print(json.dumps({"type": "STEP", "step": step, "error": str(e)}))
            break

        # Parse action JSON
        try:
            if raw_action.startswith("```"):
                raw_action = raw_action.split("```")[1]
                if raw_action.startswith("json"):
                    raw_action = raw_action[4:]
            action = json.loads(raw_action.strip())
        except json.JSONDecodeError:
            action = {"action_type": "mark_done"}

        messages.append({"role": "assistant", "content": json.dumps(action)})

        # Apply action to environment
        try:
            result = call_env("POST", "/step", action)
        except Exception as e:
            print(json.dumps({"type": "STEP", "step": step, "error": str(e)}))
            break

        reward_val = result["reward"]["value"]
        done       = result["done"]
        obs        = result["observation"]
        step      += 1

        # ── STEP log (required structured format) ────────────────────────────
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

    # Final grade from deterministic grader
    try:
        grade_result = call_env("GET", "/grade")
        score = grade_result.get("score", 0.0)
        grade = grade_result.get("grade", "unknown")
        breakdown = grade_result.get("breakdown", {})
    except Exception:
        score = 0.0
        grade = "error"
        breakdown = {}

    # ── END log (required structured format) ─────────────────────────────────
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
    # Verify environment is reachable
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

    # Save baseline results
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "average_score": round(avg, 4),
        }, f, indent=2)


if __name__ == "__main__":
    main()
