import { useState, useEffect } from 'react'
import { BarChart3, Clock, Cpu, CheckCircle2, XCircle, RefreshCw, ExternalLink, Activity, Database, Users } from 'lucide-react'

interface MetricsSummary {
    enabled: boolean
    runs?: number
    experiment_name?: string
    tracking_uri?: string
}

interface StatusData {
    status: string
    indexed: boolean
    chunks?: number
    directory?: string
    mlflow?: MetricsSummary
}

export default function MetricsPage() {
    const [status, setStatus] = useState<StatusData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    const fetchStatus = async () => {
        setLoading(true)
        setError(null)
        try {
            const res = await fetch('http://localhost:8000/api/status')
            if (!res.ok) throw new Error('Failed to fetch status')
            const data = await res.json()
            setStatus(data)
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Unknown error')
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchStatus()
    }, [])

    const StatCard = ({ 
        title, 
        value, 
        subtitle, 
        icon: Icon,
        status: cardStatus 
    }: { 
        title: string
        value: string | number
        subtitle?: string
        icon: React.ElementType
        status?: 'success' | 'warning' | 'error' | 'neutral'
    }) => {
        const statusColors = {
            success: 'text-success',
            warning: 'text-warning',
            error: 'text-danger',
            neutral: 'text-text-secondary'
        }
        
        return (
            <div className="bg-surface rounded-xl border border-border p-5">
                <div className="flex items-start justify-between">
                    <div>
                        <p className="text-sm font-medium text-text-secondary">{title}</p>
                        <p className={`text-2xl font-semibold mt-1 ${statusColors[cardStatus || 'neutral']}`}>
                            {value}
                        </p>
                        {subtitle && (
                            <p className="text-xs text-text-muted mt-1">{subtitle}</p>
                        )}
                    </div>
                    <div className="p-2.5 bg-surface-secondary rounded-lg">
                        <Icon className="w-5 h-5 text-text-secondary" />
                    </div>
                </div>
            </div>
        )
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <RefreshCw className="w-6 h-6 text-text-muted animate-spin" />
            </div>
        )
    }

    if (error) {
        return (
            <div className="p-6">
                <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                    <p className="text-danger font-medium">Failed to load metrics</p>
                    <p className="text-sm text-danger/70 mt-1">{error}</p>
                    <button 
                        onClick={fetchStatus}
                        className="btn btn-secondary mt-3"
                    >
                        Retry
                    </button>
                </div>
            </div>
        )
    }

    const mlflow = status?.mlflow

    return (
        <div className="max-w-4xl mx-auto px-6 py-8">
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h1 className="text-2xl font-semibold text-text-primary">Metrics Dashboard</h1>
                    <p className="text-text-secondary mt-1">MLflow tracking and system observability</p>
                </div>
                <button 
                    onClick={fetchStatus}
                    className="btn btn-secondary"
                >
                    <RefreshCw className="w-4 h-4" />
                    Refresh
                </button>
            </div>

            {/* System Status */}
            <section className="mb-8">
                <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">System Status</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <StatCard
                        title="Backend"
                        value={status?.status === 'ok' ? 'Online' : 'Offline'}
                        icon={Activity}
                        status={status?.status === 'ok' ? 'success' : 'error'}
                    />
                    <StatCard
                        title="Codebase"
                        value={status?.indexed ? 'Indexed' : 'Not Indexed'}
                        subtitle={status?.indexed ? `${status.chunks} chunks` : undefined}
                        icon={Database}
                        status={status?.indexed ? 'success' : 'warning'}
                    />
                    <StatCard
                        title="MLflow"
                        value={mlflow?.enabled ? 'Enabled' : 'Disabled'}
                        subtitle={mlflow?.enabled ? `${mlflow.runs || 0} runs logged` : 'Install: pip install mlflow'}
                        icon={BarChart3}
                        status={mlflow?.enabled ? 'success' : 'neutral'}
                    />
                </div>
            </section>

            {/* MLflow Details */}
            {mlflow?.enabled && (
                <section className="mb-8">
                    <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">MLflow Tracking</h2>
                    <div className="bg-surface rounded-xl border border-border overflow-hidden">
                        <div className="px-5 py-4 border-b border-border">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="font-medium text-text-primary">{mlflow.experiment_name}</p>
                                    <p className="text-sm text-text-muted mt-0.5">{mlflow.tracking_uri}</p>
                                </div>
                                <a
                                    href="http://localhost:5000"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="btn btn-secondary text-sm"
                                >
                                    <ExternalLink className="w-4 h-4" />
                                    Open Dashboard
                                </a>
                            </div>
                        </div>
                        <div className="px-5 py-4">
                            <div className="grid grid-cols-3 gap-6">
                                <div>
                                    <p className="text-sm text-text-muted">Total Runs</p>
                                    <p className="text-xl font-semibold text-text-primary mt-1">{mlflow.runs || 0}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-text-muted">Tracked Metrics</p>
                                    <p className="text-xl font-semibold text-text-primary mt-1">6</p>
                                </div>
                                <div>
                                    <p className="text-sm text-text-muted">Status</p>
                                    <p className="text-xl font-semibold text-success mt-1">Active</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            )}

            {/* What's Tracked */}
            <section className="mb-8">
                <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">What's Being Tracked</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-surface rounded-xl border border-border p-5">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="p-2 bg-blue-50 rounded-lg">
                                <Clock className="w-4 h-4 text-blue-600" />
                            </div>
                            <h3 className="font-medium text-text-primary">Query Metrics</h3>
                        </div>
                        <ul className="space-y-2 text-sm text-text-secondary">
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Response latency (ms)
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Tokens generated
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Intent classification
                            </li>
                        </ul>
                    </div>

                    <div className="bg-surface rounded-xl border border-border p-5">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="p-2 bg-green-50 rounded-lg">
                                <Database className="w-4 h-4 text-green-600" />
                            </div>
                            <h3 className="font-medium text-text-primary">RAG Metrics</h3>
                        </div>
                        <ul className="space-y-2 text-sm text-text-secondary">
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Retrieval time (ms)
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Chunks returned
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Relevance scores
                            </li>
                        </ul>
                    </div>

                    <div className="bg-surface rounded-xl border border-border p-5">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="p-2 bg-purple-50 rounded-lg">
                                <Users className="w-4 h-4 text-purple-600" />
                            </div>
                            <h3 className="font-medium text-text-primary">Crew Metrics</h3>
                        </div>
                        <ul className="space-y-2 text-sm text-text-secondary">
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Execution time (s)
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Tasks completed
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Agent count
                            </li>
                        </ul>
                    </div>

                    <div className="bg-surface rounded-xl border border-border p-5">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="p-2 bg-orange-50 rounded-lg">
                                <Cpu className="w-4 h-4 text-orange-600" />
                            </div>
                            <h3 className="font-medium text-text-primary">Model Info</h3>
                        </div>
                        <ul className="space-y-2 text-sm text-text-secondary">
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                orchestrator-ft (fine-tuned)
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                deepseek-coder:6.7b
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                                Local Ollama inference
                            </li>
                        </ul>
                    </div>
                </div>
            </section>

            {/* Instructions */}
            {!mlflow?.enabled && (
                <section>
                    <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">Enable MLflow</h2>
                    <div className="bg-surface rounded-xl border border-border p-5">
                        <p className="text-text-secondary mb-4">
                            MLflow is not currently enabled. To enable tracking:
                        </p>
                        <div className="bg-surface-secondary rounded-lg p-4 font-mono text-sm">
                            <p className="text-text-muted"># Install MLflow</p>
                            <p className="text-text-primary">pip install mlflow</p>
                            <p className="text-text-muted mt-3"># Start the dashboard</p>
                            <p className="text-text-primary">mlflow ui --backend-store-uri ./data/mlruns</p>
                        </div>
                    </div>
                </section>
            )}
        </div>
    )
}
