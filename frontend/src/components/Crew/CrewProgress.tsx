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
            return <Loader2 className="w-5 h-5 text-primary animate-spin" />
        } else {
            return <Circle className="w-5 h-5 text-muted-foreground" />
        }
    }

    const getAgentBgClass = (agent: AgentState) => {
        if (agent.status === 'done') return 'bg-success/10 border-success/30'
        if (agent.status === 'active') return 'bg-primary/10 border-primary/30 animate-pulse'
        return 'bg-secondary border-border'
    }

    const currentAgent = agents[currentAgentIndex]

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-background rounded-xl shadow-lg max-w-4xl w-full max-h-[90vh] overflow-hidden border border-border">
                {/* Header */}
                <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-secondary/50">
                    <div>
                        <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
                            <Play className="w-5 h-5 text-primary" />
                            RefactoringCrew
                        </h2>
                        <p className="text-sm text-muted-foreground mt-0.5">
                            Target: <span className="text-foreground font-medium">{target}</span>
                            {focus && <> · Focus: <span className="text-primary font-medium">{focus}</span></>}
                        </p>
                    </div>
                    <button
                        onClick={isRunning ? onCancel : onClose}
                        className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Agent Pipeline */}
                <div className="px-6 py-5 border-b border-border">
                    <div className="flex items-center justify-center gap-2 overflow-x-auto pb-2">
                        {agents.map((agent, index) => {
                            const AgentIcon = getAgentIcon(agent.name)
                            return (
                                <div key={agent.name} className="flex items-center">
                                    <div
                                        className={`flex flex-col items-center p-3 rounded-lg border transition-all duration-300 min-w-[90px] ${getAgentBgClass(agent)}`}
                                    >
                                        <AgentIcon className="w-5 h-5 text-muted-foreground mb-1" />
                                        {getAgentStatusIcon(agent)}
                                        <span className="text-xs text-foreground mt-2 text-center font-medium">
                                            {agent.name.split(' ')[0]}
                                        </span>
                                    </div>
                                    {index < agents.length - 1 && (
                                        <ArrowRight className={`w-5 h-5 mx-1 ${agent.status === 'done' ? 'text-success' : 'text-border'}`} />
                                    )}
                                </div>
                            )
                        })}
                    </div>
                </div>

                {/* Output Area */}
                <div className="px-6 py-4 max-h-[60vh] overflow-y-auto bg-secondary/30">
                    {error ? (
                        <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
                            <p className="text-destructive font-medium">Error</p>
                            <p className="text-destructive/80 text-sm mt-1">{error}</p>
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
                                            <summary className="cursor-pointer p-3 bg-background rounded-lg border border-border hover:border-primary/40 transition-colors font-medium text-foreground">
                                                {ao.agent}
                                            </summary>
                                            <div className="mt-2 p-4 bg-background rounded-lg border border-border">
                                                <pre className="text-sm text-muted-foreground whitespace-pre-wrap font-mono overflow-x-auto">
                                                    {ao.output}
                                                </pre>
                                            </div>
                                        </details>
                                    ))}
                                </div>
                            ) : (
                                <div className="p-4 bg-background rounded-lg border border-border">
                                    <pre className="text-sm text-muted-foreground whitespace-pre-wrap font-mono">
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
                                    <p className="text-primary font-medium mb-3 flex items-center gap-2">
                                        <CurrentAgentIcon className="w-5 h-5" />
                                        {currentAgent.name} is working...
                                    </p>
                                )
                            })()}
                            <div className="space-y-2">
                                {currentAgent.output.map((line, i) => (
                                    <div
                                        key={i}
                                        className="flex items-start gap-2 text-sm"
                                    >
                                        <span className="text-muted-foreground font-mono text-xs mt-1">
                                            {String(i + 1).padStart(2, '0')}
                                        </span>
                                        <span className="text-muted-foreground">{line}</span>
                                    </div>
                                ))}
                                {isRunning && (
                                    <div className="flex items-center gap-2 text-primary mt-4">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span className="text-sm">Processing...</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="flex items-center justify-center h-32 text-muted-foreground">
                            {isRunning ? (
                                <div className="flex items-center gap-2">
                                    <Loader2 className="w-5 h-5 animate-spin text-primary" />
                                    <span>Initializing crew...</span>
                                </div>
                            ) : (
                                <span>Waiting to start...</span>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-border flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">
                        {isRunning ? (
                            <span className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-primary rounded-full animate-pulse" />
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
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                            isRunning 
                                ? 'bg-destructive/10 border border-destructive/30 text-destructive hover:bg-destructive/20' 
                                : 'border border-border bg-background text-foreground hover:bg-secondary'
                        }`}
                    >
                        {isRunning ? 'Cancel' : 'Close'}
                    </button>
                </div>
            </div>
        </div>
    )
}
