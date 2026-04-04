"""
gradio_ui.py — Interactive Gradio interface for DataClean OpenEnv
Runs alongside FastAPI via subprocess; displayed on HF Spaces.
"""
import gradio as gr
import requests
import json

ENV_URL = "http://localhost:7860"


def call_env(method, endpoint, payload=None):
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "POST":
            r = requests.post(url, json=payload, timeout=10)
        else:
            r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def reset_env(task_id):
    data, err = call_env("POST", "/reset", {"task_id": task_id})
    if err:
        return f"Error: {err}", "", "", ""
    errors = data.get("errors", [])
    dataset = data.get("dataset", [])
    desc = data.get("task_description", "")
    errors_txt = json.dumps(errors[:15], indent=2)
    dataset_txt = json.dumps(dataset[:10], indent=2)
    return (
        f"✅ Reset to **{task_id}** | Initial errors: {data['total_errors_initial']}",
        desc,
        errors_txt,
        dataset_txt,
    )


def take_step(action_type, row_index, column, new_value, column_transform):
    action = {"action_type": action_type}
    if row_index.strip():
        try:
            action["row_index"] = int(row_index)
        except ValueError:
            pass
    if column.strip():
        action["column"] = column.strip()
    if new_value.strip():
        # Try to infer type
        val = new_value.strip()
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        action["new_value"] = val
    if column_transform and column_transform != "none":
        action["column_transform"] = column_transform

    data, err = call_env("POST", "/step", action)
    if err:
        return f"Error: {err}", "", ""

    obs = data.get("observation", {})
    reward = data.get("reward", {})
    done = data.get("done", False)

    status = (
        f"Step {obs.get('step_count')} | "
        f"Reward: {reward.get('value', 0):.4f} | "
        f"Errors fixed: {obs.get('errors_fixed_so_far')} | "
        f"Done: {done}"
    )
    errors_txt = json.dumps(obs.get("errors", [])[:15], indent=2)
    dataset_txt = json.dumps(obs.get("dataset", [])[:10], indent=2)
    return status, errors_txt, dataset_txt


def get_grade():
    data, err = call_env("GET", "/grade")
    if err:
        return f"Error: {err}"
    return json.dumps(data, indent=2)


def get_state():
    data, err = call_env("GET", "/state")
    if err:
        return f"Error: {err}"
    return json.dumps({k: v for k, v in data.items() if k != "dataset_snapshot"}, indent=2)


with gr.Blocks(title="DataClean OpenEnv") as demo:
    gr.Markdown("# 🧹 DataClean OpenEnv\n**Real-world data cleaning environment for AI agents**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Task Control")
            task_dropdown = gr.Dropdown(
                choices=["task_easy", "task_medium", "task_hard"],
                value="task_easy",
                label="Select Task",
            )
            reset_btn = gr.Button("Reset Environment", variant="primary")
            status_box = gr.Markdown("*Not started. Click Reset.*")
            task_desc = gr.Textbox(label="Task Description", lines=4, interactive=False)

            gr.Markdown("### ⚡ Manual Action")
            action_type = gr.Dropdown(
                choices=["fix_value", "drop_row", "fill_missing", "normalize_column", "mark_done"],
                value="fix_value",
                label="Action Type",
            )
            row_idx = gr.Textbox(label="Row Index (int, optional)", placeholder="e.g. 3")
            col_name = gr.Textbox(label="Column (optional)", placeholder="e.g. age")
            new_val = gr.Textbox(label="New Value (optional)", placeholder="e.g. 30")
            col_transform = gr.Dropdown(
                choices=["none", "lowercase", "strip", "to_int", "to_float", "to_date"],
                value="none",
                label="Column Transform (for normalize_column)",
            )
            step_btn = gr.Button("Take Step", variant="secondary")
            step_status = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Current Errors")
            errors_box = gr.Code(language="json", label="Validation Errors", lines=20)

            gr.Markdown("### 📊 Dataset Preview (first 10 rows)")
            dataset_box = gr.Code(language="json", label="Dataset", lines=15)

            with gr.Row():
                grade_btn = gr.Button("📈 Get Grade", variant="primary")
                state_btn = gr.Button("📋 Get State")
            grade_box = gr.Code(language="json", label="Grade Result", lines=10)

    reset_btn.click(
        reset_env,
        inputs=[task_dropdown],
        outputs=[status_box, task_desc, errors_box, dataset_box],
    )
    step_btn.click(
        take_step,
        inputs=[action_type, row_idx, col_name, new_val, col_transform],
        outputs=[step_status, errors_box, dataset_box],
    )
    grade_btn.click(get_grade, outputs=[grade_box])
    state_btn.click(get_state, outputs=[grade_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)