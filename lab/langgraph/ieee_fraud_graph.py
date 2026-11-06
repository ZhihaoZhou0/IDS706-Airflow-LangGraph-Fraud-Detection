"""Independent IEEE-CIS fraud experiment graph; uses Part 1 PostgreSQL output.

Graph: profile -> plan -> train -> evaluate -> (feedback -> plan | holdout -> publish).
Run inside the existing gateway container with /repo and loop_runtime.py mounted.
"""
from __future__ import annotations

import json
import operator
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO, Path('/app')):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from loop_runtime import _fit_score  # noqa: E402

# Avoid high-cardinality identifiers, timestamps, and leaked/proxy labels.
FEATURE_CANDIDATES = [
    'TransactionAmt', 'TransactionDT', 'ProductCD', 'card1', 'card2',
    'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
    'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D2', 'D3',
    'DeviceType', 'has_identity', 'transaction_amt', 'product_cd',
    'device_type', 'card_type', 'hour', 'transaction_hour',
]

class FraudState(TypedDict, total=False):
    data_path: str
    max_cycles: int
    target_f1: float
    seed: int
    cycle: int
    profile: dict
    plan: dict
    current_result: dict
    best_spec: dict
    best_val_f1: float
    history: list
    verdict: dict
    feedback: dict
    holdout_metrics: dict
    decision_trace: list
    stop_reason: str
    output_path: str
    steps: Annotated[list, operator.add]


def node_profile(state: FraudState) -> dict:
    df = pd.read_csv(state['data_path'])
    target = 'isFraud' if 'isFraud' in df else 'is_fraud'
    if target not in df:
        raise ValueError(f'No fraud label found in columns: {list(df.columns)}')
    cols = [c for c in FEATURE_CANDIDATES if c in df.columns and c != target]
    if not cols:
        raise ValueError('No expected fraud features found; inspect the PostgreSQL JSONB schema')
    profile = {'target': target, 'rows': len(df), 'fraud_rate': float(pd.to_numeric(df[target]).mean()), 'features': cols}
    return {'profile': profile, 'cycle': 0, 'best_val_f1': -1.0, 'history': [], 'steps': [{'node': 'fraud_data_profiler', **profile}]}


def node_plan(state: FraudState) -> dict:
    """Choose a model using the previous evaluation feedback, not a fixed cycle index."""
    cycle = state.get('cycle', 0)
    feedback = state.get('feedback') or {}
    if cycle == 0:
        family, hp = 'logistic_regression', {'C': 0.5, 'max_iter': 500}
        reason = 'start_with_interpretable_baseline'
    elif feedback.get('action') == 'switch_to_tree':
        family, hp = 'random_forest', {'n_estimators': 100, 'max_depth': 8}
        reason = 'weak_linear_baseline_try_nonlinear_model'
    elif feedback.get('action') == 'try_boosting':
        family, hp = 'hist_gradient_boosting', {'learning_rate': 0.08, 'max_iter': 80}
        reason = 'tree_model_below_target_try_boosting'
    elif feedback.get('action') == 'tune_best':
        best = state.get('best_spec') or {}
        family = best.get('model_family', 'random_forest')
        hp = dict(best.get('hyperparameters') or {})
        if family == 'random_forest':
            hp['n_estimators'] = min(200, int(hp.get('n_estimators', 100)) + 50)
            hp['max_depth'] = min(14, int(hp.get('max_depth', 8)) + 2)
        elif family == 'hist_gradient_boosting':
            hp['learning_rate'] = max(0.03, float(hp.get('learning_rate', 0.08)) * 0.75)
            hp['max_iter'] = min(150, int(hp.get('max_iter', 80)) + 30)
        else:
            hp['C'] = max(0.05, float(hp.get('C', 0.5)) * 0.5)
            hp['max_iter'] = 700
        reason = 'performance_regressed_tune_best_model'
    else:
        family, hp = 'hist_gradient_boosting', {'learning_rate': 0.08, 'max_iter': 80}
        reason = 'explore_boosting'
    spec = {'experiment_id': f'fraud-exp-{cycle+1}', 'model_family': family,
            'hyperparameters': hp, 'drop_columns': [], 'reason': reason}
    return {'plan': spec, 'steps': [{'node': 'experiment_planner',
            'experiment_id': spec['experiment_id'], 'model': family, 'reason': reason}]}


def node_train(state: FraudState) -> dict:
    df = pd.read_csv(state['data_path'])
    target = state['profile']['target']
    keep = state['profile']['features'] + [target]
    df = df[keep].copy()
    df[target] = pd.to_numeric(df[target], errors='raise').astype(int)
    result = _fit_score(df, target, state['plan'], state.get('seed', 42), 0.2, on_holdout=False)
    return {'current_result': result, 'steps': [{'node': 'model_trainer', 'experiment_id': result['experiment_id'], 'validation_metrics': result['validation_metrics']}]}


def node_evaluate(state: FraudState) -> dict:
    result = state['current_result']
    f1 = float(result['validation_metrics']['f1'])
    best = float(state.get('best_val_f1', -1))
    improved = f1 > best
    history = [*state.get('history', []), {'experiment_id': result['experiment_id'], 'model_family': result['model_family'], **result['validation_metrics']}]
    cycle = state.get('cycle', 0) + 1
    stop = cycle >= state.get('max_cycles', 3) or max(f1, best) >= state.get('target_f1', 0.3)
    verdict = {'improved': improved, 'continue': not stop, 'cycle': cycle}
    return {'history': history, 'cycle': cycle, 'verdict': verdict,
            'best_val_f1': f1 if improved else best,
            'best_spec': state['plan'] if improved else state.get('best_spec', {}),
            'stop_reason': ('target_reached' if max(f1, best) >= state.get('target_f1', 0.3) else 'max_cycles') if stop else '',
            'steps': [{'node': 'performance_evaluator', **verdict, 'f1': f1}]}


def route_evaluation(state: FraudState) -> str:
    return 'feedback' if state['verdict']['continue'] else 'holdout'


def node_feedback(state: FraudState) -> dict:
    """Turn measured F1 and improvement into a concrete next-step decision."""
    last = state['history'][-1]
    previous = state['history'][-2]['f1'] if len(state['history']) > 1 else None
    current_f1 = float(last['f1'])
    target = float(state.get('target_f1', 0.3))
    improvement = None if previous is None else current_f1 - float(previous)
    if last['model_family'] == 'logistic_regression' and current_f1 < target:
        action = 'switch_to_tree'
    elif improvement is not None and improvement < 0:
        action = 'tune_best'
    elif current_f1 < target:
        action = 'try_boosting'
    else:
        action = 'tune_best'
    feedback = {'previous_model': last['model_family'],
                'previous_f1': current_f1, 'f1_delta': improvement,
                'target_gap': max(0.0, target - current_f1), 'action': action}
    return {'feedback': feedback, 'steps': [{'node': 'feedback_analyzer', **feedback}]}


def node_holdout(state: FraudState) -> dict:
    df = pd.read_csv(state['data_path'])
    target = state['profile']['target']
    df = df[state['profile']['features'] + [target]].copy()
    df[target] = pd.to_numeric(df[target], errors='raise').astype(int)
    metrics = _fit_score(df, target, state['best_spec'], state.get('seed', 42), 0.2, on_holdout=True)
    return {'holdout_metrics': metrics, 'steps': [{'node': 'holdout_evaluation', **metrics}]}


def node_publish(state: FraudState) -> dict:
    payload = {
    'dataset': 'IEEE-CIS Fraud Detection',
    'source': 'PostgreSQL',
    'profile': state['profile'],
    'history': state['history'],    
    'best_spec': state['best_spec'],
    'best_validation_f1': state['best_val_f1'],
    'holdout_metrics': state['holdout_metrics'],
    'cycles': state['cycle'],
    'stop_reason': state['stop_reason'],
    'decision_trace': [
        step for step in state.get('steps', [])
        if step.get('node') in (
            'experiment_planner',
            'performance_evaluator',
            'feedback_analyzer',
        )
    ],
}
    out = Path(state.get('output_path', '/repo/artifacts/ieee_fraud/agent_results.json'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return {
    'steps': [{'node': 'publish_results', 'path': str(out)}],
    'decision_trace': payload['decision_trace'],
    'history': payload['history'],
    'holdout_metrics': payload['holdout_metrics'],
}


def build_graph():
    graph = StateGraph(FraudState)
    for name, fn in [('fraud_data_profiler', node_profile), ('experiment_planner', node_plan),
                     ('model_trainer', node_train), ('performance_evaluator', node_evaluate),
                     ('feedback_analyzer', node_feedback), ('holdout_evaluation', node_holdout),
                     ('publish_results', node_publish)]:
        graph.add_node(name, fn)
    graph.add_edge(START, 'fraud_data_profiler')
    graph.add_edge('fraud_data_profiler', 'experiment_planner')
    graph.add_edge('experiment_planner', 'model_trainer')
    graph.add_edge('model_trainer', 'performance_evaluator')
    graph.add_conditional_edges('performance_evaluator', route_evaluation,
                                {'feedback': 'feedback_analyzer', 'holdout': 'holdout_evaluation'})
    graph.add_edge('feedback_analyzer', 'experiment_planner')
    graph.add_edge('holdout_evaluation', 'publish_results')
    graph.add_edge('publish_results', END)
    return graph.compile()
