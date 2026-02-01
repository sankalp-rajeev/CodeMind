import { ChevronDown, GitBranch, Inbox, Brain, Target, GitMerge, Search, MessageSquare, Settings, Rocket } from 'lucide-react'

interface RoutingStep {
    step: string
    icon?: string
    title: string
    detail: string
    timestamp: Date
}

interface RoutingFlowProps {
    steps: RoutingStep[]
    query?: string
}

// Map step types to icons
const stepIcons: Record<string, React.ElementType> = {
    'received': Inbox,
    'classifying': Brain,
    'classified': Target,
    'routing': GitMerge,
    'rag': Search,
    'generating': MessageSquare,
    'working': Settings,
    'crew_start': Rocket,
}

export default function RoutingFlow({ steps, query }: RoutingFlowProps) {
    if (steps.length === 0) return null

    const isDecisionStep = (s: string) => ['classifying', 'classified', 'routing'].includes(s)
    const isActive = (i: number) => i === steps.length - 1
    
    const getStepIcon = (step: string) => stepIcons[step] || Settings

    return (
        <div className="mb-6 p-5 bg-surface rounded-2xl border border-border shadow-sm animate-fadeIn overflow-x-auto">
            {/* Header */}
            <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border">
                <GitBranch className="w-4 h-4 text-brand" />
                <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
                    Query Pipeline
                </span>
            </div>

            <div className="flex flex-col items-center min-w-[280px]">
                {/* Query node (root) */}
                {query && (
                    <div className="flex flex-col items-center w-full">
                        <div className="px-4 py-3 rounded-xl bg-brand-muted border border-brand/20 text-center max-w-[320px]">
                            <span className="text-xs font-medium text-brand uppercase tracking-wider block mb-1">
                                Your Query
                            </span>
                            <p className="text-sm text-text-primary line-clamp-2">{query}</p>
                        </div>
                        <div className="w-0.5 h-5 bg-gradient-to-b from-brand/30 to-border my-0.5" />
                    </div>
                )}

                {/* Flow nodes */}
                {steps.map((step, index) => (
                    <div
                        key={`${step.step}-${index}`}
                        className="flex flex-col items-center w-full"
                        style={{ animationDelay: `${index * 60}ms` }}
                    >
                        <div
                            className={`flex flex-col items-center w-full animate-fadeIn ${
                                isDecisionStep(step.step) ? 'decision-node' : ''
                            }`}
                            style={{ animationDelay: `${index * 60}ms` }}
                        >
                            {/* Connector line */}
                            {index > 0 && (
                                <div className="w-0.5 h-4 bg-gradient-to-b from-border to-border-light flex-shrink-0" />
                            )}

                            {/* Node */}
                            {(() => {
                                const StepIcon = getStepIcon(step.step)
                                return (
                                    <div
                                        className={`flex items-center gap-3 px-4 py-3 min-w-[240px] max-w-[320px] transition-all ${
                                            isDecisionStep(step.step)
                                                ? 'rounded-lg border-2 border-dashed border-blue-300 bg-blue-50'
                                                : 'rounded-xl border'
                                        } ${
                                            isActive(index)
                                                ? 'bg-brand-muted border-brand/40 shadow-sm ring-1 ring-brand/10'
                                                : 'bg-surface-secondary border-border'
                                        }`}
                                    >
                                        <div className="p-1.5 rounded-lg bg-surface-tertiary flex-shrink-0">
                                            <StepIcon className={`w-4 h-4 ${isActive(index) ? 'text-brand' : 'text-text-secondary'}`} />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <span className={`block font-medium text-sm ${isActive(index) ? 'text-brand' : 'text-text-primary'}`}>
                                                {step.title}
                                            </span>
                                            {step.detail && (
                                                <span className="block text-xs text-text-muted truncate mt-0.5">
                                                    {step.detail}
                                                </span>
                                            )}
                                        </div>
                                        {isActive(index) && (
                                            <span className="w-2 h-2 rounded-full bg-brand animate-pulse flex-shrink-0" />
                                        )}
                                    </div>
                                )
                            })()}

                            {/* Down arrow between nodes */}
                            {index < steps.length - 1 && (
                                <div className="flex flex-col items-center py-1">
                                    <ChevronDown className="w-5 h-5 text-border-dark" strokeWidth={2.5} />
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {/* Progress indicator */}
                {steps.length > 0 && (
                    <div className="mt-3 flex items-center gap-2 text-xs text-text-muted">
                        <span className="w-1.5 h-1.5 rounded-full bg-brand animate-pulse" />
                        <span>Processing pipeline</span>
                    </div>
                )}
            </div>
        </div>
    )
}
