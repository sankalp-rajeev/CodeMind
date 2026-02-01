import { AgentState, CrewState } from '../../hooks/useCrewWebSocket'
import { X, Play, Loader2, CheckCircle2, Circle, ArrowRight, Search, Shield, Zap, TestTube, FileText } from 'lucide-react'

interface CrewProgressProps {
    crewState: CrewState
    onClose: () => void
    onCancel: () => void
}

// Map agent names to icons
const agentIcons: Record<string, React.ElementType> = {
    'Code Explorer': Search,
    'Security Analyst': Shield,
    'Algorithm Optimizer': Zap,
    'Test Engineer': TestTube,
    'Documentation Writer': FileText,
}

export default function CrewProgress({ crewState, onClose, onCancel }: CrewProgressProps) {
    const { isRunning, target, focus, agents, currentAgentIndex, finalResult, agentOutputs, error } = crewState

    const getAgentIcon = (agentName: string) => {
        const Icon = agentIcons[agentName] || Circle
        return Icon
    }

    const getAgentStatusIcon = (agent: AgentState) => {
        if (agent.status === 'done') {
            return <CheckCircle2 className="w-5 h-5 text-success" />
        } else if (agent.status === 'active') {
            return <Loader2 className="w-5 h-5 text-brand animate-spin" />
        } else {
            return <Circle className="w-5 h-5 text-text-muted" />
        }
    }

    const getAgentBgClass = (agent: AgentState) => {
        if (agent.status === 'done') return 'bg-green-50 border-green-200'
        if (agent.status === 'active') return 'bg-brand-light border-brand/30 animate-pulse-subtle'
        return 'bg-surface-secondary border-border'
    }

    const currentAgent = agents[currentAgentIndex]

    return (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-surface rounded-2xl shadow-lg max-w-4xl w-full max-h-[90vh] overflow-hidden border border-border animate-fadeIn">
                {/* Header */}
                <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface-secondary">
                    <div>
                        <h2 className="text-lg font-semibold text-text-primary flex items-center gap-2">
                            <Play className="w-5 h-5 text-brand" />
                            RefactoringCrew
                        </h2>
                        <p className="text-sm text-text-secondary mt-0.5">
                            Target: <span className="text-text-primary font-medium">{target}</span>
                            {focus && <> · Focus: <span className="text-brand font-medium">{focus}</span></>}
                        </p>
                    </div>
                    <button
                        onClick={isRunning ? onCancel : onClose}
                        className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-surface-tertiary transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Agent Pipeline */}
                <div className="px-6 py-5 border-b border-border bg-surface">
                    <div className="flex items-center justify-center gap-2 overflow-x-auto pb-2">
                        {agents.map((agent, index) => {
                            const AgentIcon = getAgentIcon(agent.name)
                            return (
                                <div key={agent.name} className="flex items-center">
                                    <div
                                        className={`flex flex-col items-center p-3 rounded-xl border transition-all duration-300 min-w-[90px] ${getAgentBgClass(agent)}`}
                                    >
                                        <AgentIcon className="w-5 h-5 text-text-secondary mb-1" />
                                        {getAgentStatusIcon(agent)}
                                        <span className="text-xs text-text-primary mt-2 text-center font-medium">
                                            {agent.name.split(' ')[0]}
                                        </span>
                                    </div>
                                    {index < agents.length - 1 && (
                                        <ArrowRight className={`w-5 h-5 mx-1 ${agent.status === 'done' ? 'text-success' : 'text-border-dark'}`} />
                                    )}
                                </div>
                            )
                        })}
                    </div>
                </div>

                {/* Output Area */}
                <div className="px-6 py-4 max-h-[60vh] overflow-y-auto bg-surface-secondary">
                    {error ? (
                        <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                            <p className="text-danger font-medium">Error</p>
                            <p className="text-danger/80 text-sm mt-1">{error}</p>
                        </div>
                    ) : finalResult ? (
                        <div>
                            <p className="text-success font-medium mb-4 flex items-center gap-2">
                                <CheckCircle2 className="w-5 h-5" />
                                Complete ({agentOutputs.length} agent{agentOutputs.length !== 1 ? 's' : ''})
                            </p>
                            
                            {agentOutputs.length > 0 ? (
                                <div className="space-y-3">
                                    {agentOutputs.map((ao, idx) => (
                                        <details key={idx} className="group" open={idx === 0}>
                                            <summary className="cursor-pointer p-3 bg-surface rounded-xl border border-border hover:border-brand/40 transition-colors font-medium text-text-primary">
                                                {ao.agent}
                                            </summary>
                                            <div className="mt-2 p-4 bg-surface rounded-xl border border-border">
                                                <pre className="text-sm text-text-secondary whitespace-pre-wrap font-mono overflow-x-auto">
                                                    {ao.output}
                                                </pre>
                                            </div>
                                        </details>
                                    ))}
                                </div>
                            ) : (
                                <div className="p-4 bg-surface rounded-xl border border-border">
                                    <pre className="text-sm text-text-secondary whitespace-pre-wrap font-mono">
                                        {finalResult}
                                    </pre>
                                </div>
                            )}
                        </div>
                    ) : currentAgent ? (
                        <div>
                            {(() => {
                                const CurrentAgentIcon = getAgentIcon(currentAgent.name)
                                return (
                                    <p className="text-brand font-medium mb-3 flex items-center gap-2">
                                        <CurrentAgentIcon className="w-5 h-5" />
                                        {currentAgent.name} is working...
                                    </p>
                                )
                            })()}
                            <div className="space-y-2">
                                {currentAgent.output.map((line, i) => (
                                    <div
                                        key={i}
                                        className="flex items-start gap-2 text-sm animate-fadeIn"
                                    >
                                        <span className="text-text-muted font-mono text-xs mt-1">
                                            {String(i + 1).padStart(2, '0')}
                                        </span>
                                        <span className="text-text-secondary">{line}</span>
                                    </div>
                                ))}
                                {isRunning && (
                                    <div className="flex items-center gap-2 text-brand mt-4">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span className="text-sm">Processing...</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="flex items-center justify-center h-32 text-text-muted">
                            {isRunning ? (
                                <div className="flex items-center gap-2">
                                    <Loader2 className="w-5 h-5 animate-spin text-brand" />
                                    <span>Initializing crew...</span>
                                </div>
                            ) : (
                                <span>Waiting to start...</span>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-border bg-surface flex items-center justify-between">
                    <div className="text-sm text-text-secondary">
                        {isRunning ? (
                            <span className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-brand rounded-full animate-pulse" />
                                {currentAgentIndex + 1} of {agents.length} agents
                            </span>
                        ) : agents.length > 0 ? (
                            <span className="flex items-center gap-2">
                                <CheckCircle2 className="w-4 h-4 text-success" />
                                {agents.filter(a => a.status === 'done').length} agents completed
                            </span>
                        ) : null}
                    </div>
                    <button
                        onClick={isRunning ? onCancel : onClose}
                        className={`btn ${isRunning ? 'bg-red-50 border-red-200 text-danger hover:bg-red-100' : 'btn-secondary'}`}
                    >
                        {isRunning ? 'Cancel' : 'Close'}
                    </button>
                </div>
            </div>
        </div>
    )
}
