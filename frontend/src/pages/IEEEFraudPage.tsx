
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

type NodeStatus = 'pending' | 'running' | 'completed' | 'failed'

type Experiment = {
  experiment_id: string
  model_family: string
  f1: number
  accuracy?: number
}

type Decision = {
  node: string
  experiment_id?: string
  model?: string
  reason?: string
  action?: string
  cycle?: number
  f1?: number
  continue?: boolean
}

type HoldoutMetrics = {
  f1: number
  accuracy: number
  n_holdout: number
}

type FraudNodeData = {
  title: string
  description: string
  status: NodeStatus
  [key: string]: unknown
}

const WORKFLOW = [
  { id: 'fraud_data_profiler', title: 'Data Profiler', description: 'Inspect cleaned fraud data', x: 80, y: 180 },
  { id: 'experiment_planner', title: 'Experiment Planner', description: 'Select model and parameters', x: 360, y: 180 },
  { id: 'model_trainer', title: 'Model Trainer', description: 'Train candidate models', x: 640, y: 180 },
  { id: 'performance_evaluator', title: 'Performance Evaluator', description: 'Evaluate validation metrics', x: 920, y: 180 },
  { id: 'feedback_analyzer', title: 'Feedback Analyzer', description: 'Improve next experiment', x: 640, y: 390 },
  { id: 'holdout_evaluation', title: 'Holdout Evaluation', description: 'Evaluate unseen test data', x: 1200, y: 180 },
  { id: 'publish_results', title: 'Publish Results', description: 'Save final model metrics', x: 1480, y: 180 },
] as const

const CONNECTIONS = [
  ['fraud_data_profiler', 'experiment_planner'],
  ['experiment_planner', 'model_trainer'],
  ['model_trainer', 'performance_evaluator'],
  ['performance_evaluator', 'feedback_analyzer'],
  ['feedback_analyzer', 'experiment_planner'],
  ['performance_evaluator', 'holdout_evaluation'],
  ['holdout_evaluation', 'publish_results'],
] as const

const STATUS_COLORS: Record<NodeStatus, string> = {
  pending: '#94a3b8',
  running: '#2563eb',
  completed: '#16a34a',
  failed: '#dc2626',
}

function FraudFlowNode({ data }: NodeProps<Node<FraudNodeData>>) {
  const color = STATUS_COLORS[data.status]

  return (
    <div
      style={{
        width: 220,
        padding: 16,
        borderRadius: 12,
        background: '#ffffff',
        border: `2px solid ${color}`,
        boxShadow: data.status === 'running'
          ? '0 0 18px rgba(37, 99, 235, 0.25)'
          : '0 3px 12px rgba(0, 0, 0, 0.08)',
        color: '#0f172a',
      }}
    >
      <Handle type="target" position={Position.Left} />
      <div style={{ fontWeight: 700, fontSize: 14 }}>{data.title}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>
        {data.description}
      </div>
      <div style={{ fontSize: 11, color, marginTop: 10, fontWeight: 700 }}>
        {data.status.toUpperCase()}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

const nodeTypes = { fraud: FraudFlowNode }


export function IEEEFraudPage() {
  const [statuses, setStatuses] = useState<Record<string, NodeStatus>>({})
  const [running, setRunning] = useState(false)
  const [message, setMessage] = useState('Ready to run')
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [decisions, setDecisions] = useState<Decision[]>([])
  const [holdout, setHoldout] = useState<HoldoutMetrics | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close()
    }
  }, [])

  function startRun() {
    if (running) return

    eventSourceRef.current?.close()
    setStatuses({})
    setExperiments([])
    setDecisions([])
    setHoldout(null)
    setRunning(true)
    setMessage('Connecting to LangGraph...')

    const source = new EventSource('/api/ieee-fraud/stream')
    eventSourceRef.current = source

    source.addEventListener('run_started', () => {
      setMessage('LangGraph workflow started')
    })

    source.addEventListener('node_started', (event) => {
      const payload = JSON.parse(event.data)
      setStatuses((previous) => ({
        ...previous,
        [payload.node]: 'running',
      }))
      setMessage(`Running: ${payload.node}`)
    })

    source.addEventListener('node_completed', (event) => {
      const payload = JSON.parse(event.data)
      const update = payload.update ?? {}

      setStatuses((previous) => ({
        ...previous,
        [payload.node]: 'completed',
      }))
      setMessage(`Completed: ${payload.node}`)

      if (payload.node === 'performance_evaluator') {
        if (Array.isArray(update.history)) {
          setExperiments(update.history)
        }

        if (Array.isArray(update.decision_trace)) {
          setDecisions(update.decision_trace)
        }
      }

      if (payload.node === 'holdout_evaluation') {
        if (update.holdout_metrics) {
          setHoldout(update.holdout_metrics)
        }
      }

      if (payload.node === 'publish_results') {
        if (Array.isArray(update.history)) {
          setExperiments(update.history)
        }

        if (Array.isArray(update.decision_trace)) {
          setDecisions(update.decision_trace)
        }

        if (update.holdout_metrics) {
          setHoldout(update.holdout_metrics)
        }
      }
    })

    source.addEventListener('node_failed', (event) => {
      const payload = JSON.parse(event.data)
      setStatuses((previous) => ({
        ...previous,
        [payload.node]: 'failed',
      }))
      setMessage(`Failed: ${payload.node}`)
    })

    source.addEventListener('run_completed', () => {
      setRunning(false)
      setMessage('Workflow completed successfully')
      source.close()
      eventSourceRef.current = null
    })

    source.addEventListener('error', (event) => {
      if (event instanceof MessageEvent) {
        try {
          const payload = JSON.parse(event.data)
          setMessage(`Error: ${payload.message}`)
        } catch {
          setMessage('Workflow failed')
        }
      } else {
        setMessage('SSE connection lost')
      }
      setRunning(false)
      source.close()
      eventSourceRef.current = null
    })
  }


  const nodes: Node<FraudNodeData>[] = useMemo(
    () =>
      WORKFLOW.map((item) => ({
        id: item.id,
        type: 'fraud',
        position: { x: item.x, y: item.y },
        draggable: false,
        data: {
          title: item.title,
          description: item.description,
          status: statuses[item.id] ?? 'pending',
        },
      })),
    [statuses],
  )

  const edges: Edge[] = useMemo(
    () =>
      CONNECTIONS.map(([source, target]) => ({
        id: `${source}-${target}`,
        source,
        target,
        type: 'smoothstep',
        animated: statuses[source] === 'running',
        style: {
          stroke: statuses[source] === 'running' ? '#2563eb' : '#94a3b8',
          strokeWidth: 2,
        },
      })),
    [statuses],
  )

  return (
    <div style={{ padding: 24, minHeight: '100vh', background: '#f8fafc' }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
        IEEE-CIS Fraud Detection
      </h1>
      <p style={{ color: '#475569', marginTop: 8 }}>
        Live LangGraph Agent Dashboard — Adaptive Fraud Detection Workflow
      </p>



      <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', gap: 16 }}>
        <button
          type="button"
          onClick={startRun}
          disabled={running}
          style={{
            padding: '12px 20px',
            background: running ? '#94a3b8' : '#2563eb',
            color: '#ffffff',
            border: 'none',
            borderRadius: 8,
            fontWeight: 700,
            cursor: running ? 'not-allowed' : 'pointer',
          }}
        >
          {running ? 'Running...' : 'Run Fraud Detection'}
        </button>
        <span style={{ color: '#475569', fontSize: 14 }}>{message}</span>
      </div>


      <div
        style={{
          marginTop: 24,
          height: 580,
          border: '1px solid #cbd5e1',
          borderRadius: 14,
          overflow: 'hidden',
          background: '#ffffff',
        }}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          nodesDraggable={false}
          nodesConnectable={false}
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>


      <section style={{ marginTop: 28 }}>
        <h2 style={{ fontSize: 22, color: '#0f172a' }}>
          Model Performance
        </h2>

        {experiments.length === 0 ? (
          <p style={{ color: '#64748b' }}>
            Run the workflow to view model performance.
          </p>
        ) : (
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            background: '#ffffff',
            marginTop: 12,
          }}>
            <thead>
              <tr>
                {['Experiment', 'Model', 'Validation F1', 'Accuracy'].map(
                  (heading) => (
                    <th key={heading} style={{
                      padding: 12,
                      borderBottom: '1px solid #cbd5e1',
                      textAlign: 'left',
                    }}>
                      {heading}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => (
                <tr key={exp.experiment_id}>
                  <td style={{ padding: 12 }}>{exp.experiment_id}</td>
                  <td style={{ padding: 12 }}>{exp.model_family}</td>
                  <td style={{ padding: 12 }}>{exp.f1.toFixed(4)}</td>
                  <td style={{ padding: 12 }}>
                    {exp.accuracy == null
                      ? '—'
                      : `${(exp.accuracy * 100).toFixed(2)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section style={{ marginTop: 28 }}>
        <h2 style={{ fontSize: 22, color: '#0f172a' }}>
          Holdout Evaluation
        </h2>

        {holdout ? (
          <div style={{
            display: 'flex',
            gap: 24,
            flexWrap: 'wrap',
            marginTop: 12,
          }}>
            <div><strong>F1:</strong> {holdout.f1.toFixed(4)}</div>
            <div>
              <strong>Accuracy:</strong>{' '}
              {(holdout.accuracy * 100).toFixed(2)}%
            </div>
            <div>
              <strong>Test samples:</strong> {holdout.n_holdout}
            </div>
          </div>
        ) : (
          <p style={{ color: '#64748b' }}>
            Holdout metrics will appear after evaluation.
          </p>
        )}
      </section>

      <section style={{ marginTop: 28 }}>
        <h2 style={{ fontSize: 22, color: '#0f172a' }}>
          Agent Decision Trace
        </h2>

        {decisions.length === 0 ? (
          <p style={{ color: '#64748b' }}>
            Agent decisions will appear during execution.
          </p>
        ) : (
          <ol style={{ paddingLeft: 24 }}>
            {decisions.map((decision, index) => (
              <li key={index} style={{
                padding: '8px 0',
                color: '#334155',
              }}>
                <strong>{decision.node}</strong>
                {decision.model && ` — ${decision.model}`}
                {decision.reason && ` — ${decision.reason}`}
                {decision.action && ` — ${decision.action}`}
                {decision.f1 != null &&
                  ` — F1: ${decision.f1.toFixed(4)}`}
              </li>
            ))}
          </ol>
        )}
      </section>

    </div>
  )
}
