
"""Live SSE API for the IEEE-CIS fraud detection LangGraph workflow."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sklearn.model_selection import train_test_split

router = APIRouter(prefix="/ieee-fraud", tags=["IEEE-CIS Fraud"])

REPO = Path("/repo")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.langgraph.ieee_fraud_graph import FEATURE_CANDIDATES, build_graph  # noqa: E402


def export_data() -> Path:
    """Read processed records from PostgreSQL and export a stratified sample."""
    conn = psycopg2.connect(
        host=os.getenv("ANALYTICS_DB_HOST", "analytics-db"),
        dbname=os.getenv("ANALYTICS_DB_NAME", "fraud_analytics"),
        user=os.getenv("ANALYTICS_DB_USER", "analytics"),
        password=os.getenv("ANALYTICS_DB_PASSWORD", "analytics_dev"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT features FROM ieee_fraud_transactions")
            rows = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    target = "isFraud" if "isFraud" in df else "is_fraud"

    if target not in df:
        raise ValueError("Fraud label missing from PostgreSQL features")

    features = [col for col in FEATURE_CANDIDATES if col in df.columns]
    if not features:
        raise ValueError("No supported features found in PostgreSQL")

    df = df[[target, *features]]

    if len(df) > 12000:
        df, _ = train_test_split(
            df,
            train_size=12000,
            stratify=df[target],
            random_state=42,
        )

    output = REPO / "artifacts/ieee_fraud/live_agent_input.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def sse(event_type: str, payload: dict) -> str:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, default=str)}\n\n"
    )


@router.get("/stream")
def stream_ieee_fraud() -> StreamingResponse:
    """Stream live LangGraph node execution events."""

    def generate():
        try:
            data_path = export_data()
            graph = build_graph()

            state = {
                "data_path": str(data_path),
                "max_cycles": 3,
                "target_f1": 0.3,
                "seed": 42,
                "output_path": str(
                    REPO / "artifacts/ieee_fraud/live_agent_results.json"
                ),
                "steps": [],
            }

            yield sse("run_started", {"type": "run_started"})


            for mode, event in graph.stream(
                state,
                stream_mode=["tasks", "updates"],
            ):
                if mode == "tasks":
                    # LangGraph emits task-start and task-finish events.
                    node_name = event.get("name")
                    if not node_name:
                        continue

                    if "result" not in event and "error" not in event:
                        yield sse(
                            "node_started",
                            {
                                "type": "node_started",
                                "node": node_name,
                            },
                        )

                    elif event.get("error"):
                        yield sse(
                            "node_failed",
                            {
                                "type": "node_failed",
                                "node": node_name,
                                "error": str(event["error"]),
                            },
                        )

                elif mode == "updates":
                    for node_name, update in event.items():
                        if node_name == "__interrupt__":
                            continue

                        yield sse(
                            "node_completed",
                            {
                                "type": "node_completed",
                                "node": node_name,
                                "update": update,
                            },
                        )


            yield sse("run_completed", {"type": "run_completed"})

        except Exception as exc:
            yield sse(
                "error",
                {"type": "error", "message": str(exc)},
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
