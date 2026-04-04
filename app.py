import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from env.environment import DataCleaningEnv, TASK_META
from env.models import Action, StepResult, Observation, EnvState, TaskInfo
from graders.graders import run_grader

app = FastAPI(
    title="DataClean OpenEnv",
    description="Real-world data cleaning environment for AI agent training",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global env instance (sufficient for hackathon / single-session use)
env = DataCleaningEnv()


class ResetRequest(BaseModel):
    task_id: str = "task_easy"


# ── OpenEnv Core Endpoints ──────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    """Reset the environment and return the initial observation."""
    try:
        obs = env.reset(task_id=request.task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """Apply an action and return (observation, reward, done, info)."""
    result = env.step(action)
    return result


@app.get("/state", response_model=EnvState)
def state():
    """Return current environment state."""
    return env.state()


# ── Supplementary Endpoints ─────────────────────────────────────────────────

@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks():
    """List all available tasks."""
    return [
        TaskInfo(
            task_id=tid,
            name=meta["name"],
            difficulty=meta["difficulty"],
            max_steps=meta["max_steps"],
            description=meta["description"],
        )
        for tid, meta in TASK_META.items()
    ]


@app.get("/grade")
def grade():
    """Run the deterministic grader on the current episode state."""
    current_state = env.state()
    if not current_state.task_id:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")
    result = run_grader(current_state.task_id, current_state)
    return result


@app.get("/health")
def health():
    """Health check — used by OpenEnv validator."""
    return {
        "status": "ok",
        "env": "data-cleaning-env",
        "version": "1.0.0",
        "openenv": True,
    }


@app.get("/")
def root():
    return {
        "message": "DataClean OpenEnv is running",
        "docs": "/docs",
        "tasks": "/tasks",
        "reset": "POST /reset",
        "step": "POST /step",
        "state": "GET /state",
    }
