# IEEE-CIS Fraud Detection: Airflow and LangGraph

## Project Overview

This project implements a data engineering and machine learning workflow using the IEEE-CIS Fraud Detection dataset.

It consists of two components:

- **Part 1 — Apache Airflow:** An end-to-end data pipeline for ingestion, cleaning, transformation, PostgreSQL storage, fraud analysis, and metrics publication.
- **Part 2 — LangGraph:** An adaptive machine learning workflow that selects models, evaluates performance, analyzes feedback, and iteratively improves fraud detection results.

Both components are implemented within the AutoML Research Lab environment.

## Dataset

**Source:** IEEE-CIS Fraud Detection (Kaggle)

The project uses two related datasets:

- `train_transaction.csv` — Transaction records, fraud labels, and transaction-related features.
- `train_identity.csv` — Identity-related information associated with transactions.

The datasets are joined using `TransactionID`.

For local experimentation, a stratified sample of 50,000 transactions is used for the Airflow pipeline. The LangGraph workflow uses a separate 12,000-row sample exported from PostgreSQL.

Raw and sampled datasets are excluded from Git version control.

## Part 1: Apache Airflow Pipeline

The Airflow DAG is implemented in:

`lab/airflow/dags/ieee_fraud_pipeline.py`

### Pipeline Tasks

1. `ingest_transactions` — Load transaction records.
2. `clean_transactions` — Validate and clean transaction data.
3. `ingest_identity` — Load identity records.
4. `clean_identity` — Clean identity-related fields.
5. `merge_sources` — Join transaction and identity data.
6. `engineer_features` — Generate features for fraud analysis.
7. `load_postgres` — Store processed records in PostgreSQL.
8. `read_postgres` — Retrieve stored records for downstream analysis.
9. `analyze_fraud` — Train and evaluate a baseline fraud detection model.
10. `publish_metrics` — Publish pipeline metrics.

### Pipeline Results

| Metric | Value |
|---|---|
| Processed transactions | 50,000 |
| Fraud rate | 3.50% |
| Identity match rate | 24.71% |
| Training records | 40,000 |
| Test records | 10,000 |
| Baseline model | Logistic Regression |
| Accuracy | 0.7594 |
| Precision | 0.0783 |
| Recall | 0.5457 |
| F1 Score | 0.1370 |

The pipeline completed successfully, with all ten tasks executed.

## Part 2: Adaptive LangGraph Workflow

The LangGraph implementation is located in:

`lab/langgraph/ieee_fraud_graph.py`

The workflow includes the following nodes:

- **Data Profiler:** Inspects the fraud dataset.
- **Experiment Planner:** Selects models and experiment configurations.
- **Model Trainer:** Trains candidate models.
- **Performance Evaluator:** Evaluates validation performance.
- **Feedback Analyzer:** Determines how to improve subsequent experiments.
- **Holdout Evaluation:** Evaluates the selected model on held-out data.
- **Results Publisher:** Saves experiment results and decision traces.

### Adaptive Model Selection

The workflow begins with Logistic Regression and adjusts its strategy based on validation results.

| Experiment | Model | Validation F1 |
|---|---|---:|
| 1 | Logistic Regression | 0.0835 |
| 2 | Random Forest | 0.2624 |
| 3 | Histogram Gradient Boosting | 0.3455 |

The workflow stops when the target validation F1 score is reached or the configured maximum number of cycles is exhausted.

### Final Holdout Evaluation

| Metric | Value |
|---|---|
| Selected model | Histogram Gradient Boosting |
| Best validation F1 | 0.3455 |
| Holdout F1 | 0.4404 |
| Holdout accuracy | 0.9746 |
| Holdout records | 2,400 |
| Stop reason | Target reached |

The workflow records model selections, evaluation outcomes, feedback decisions, and stopping conditions.

## Technology Stack

- Python
- Apache Airflow
- LangGraph
- PostgreSQL
- pandas and NumPy
- scikit-learn
- Docker and Docker Compose

## Key Files

| File | Purpose |
|---|---|
| `lab/airflow/dags/ieee_fraud_pipeline.py` | Airflow pipeline |
| `lab/langgraph/ieee_fraud_graph.py` | Adaptive LangGraph workflow |
| `sample_ieee_data.py` | Dataset sampling |
| `test_ieee_agent.py` | LangGraph execution and testing |
| `loop_runtime.py` | Model training utilities |
| `artifacts/ieee_fraud/latest_metrics.json` | Airflow metrics |
| `artifacts/ieee_fraud/agent_results.json` | LangGraph experiment results |
| `artifacts/ieee_fraud/ieee_agent_graph.mmd` | Workflow graph |

## Reproducibility Notes

The project runs in Docker-based environments.

The Gateway image includes pandas, scikit-learn, LangGraph, and the PostgreSQL driver `psycopg2-binary`.

Raw datasets are not included in the repository. Users must obtain the IEEE-CIS Fraud Detection dataset separately and prepare the required sample files.

The Airflow and LangGraph components use different sample sizes and evaluation procedures. Their reported F1 scores should therefore not be interpreted as a controlled head-to-head comparison.

## Future Work

The workflow can be extended with multiple specialized agents, adaptive task routing, failure recovery, and an evaluation suite comparing adaptive workflows against fixed Airflow and simple agent baselines.