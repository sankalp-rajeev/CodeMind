import { useState } from 'react'
import { Wrench, X, Shield, Zap, TestTube, FileText, Play } from 'lucide-react'

interface RefactorPanelProps {
    onStart: (target: string, focus: string | null) => void
    onClose: () => void
    isIndexed: boolean
}

const focusOptions = [
    { value: null, label: 'All Agents', icon: Wrench, description: 'Run all 5 agents' },
    { value: 'security', label: 'Security', icon: Shield, description: 'Focus on vulnerabilities' },
    { value: 'performance', label: 'Performance', icon: Zap, description: 'Optimize algorithms' },
    { value: 'tests', label: 'Tests', icon: TestTube, description: 'Generate test cases' },
    { value: 'docs', label: 'Documentation', icon: FileText, description: 'Write docstrings' },
]

export default function RefactorPanel({ onStart, onClose, isIndexed }: RefactorPanelProps) {
    const [target, setTarget] = useState('')
    const [focus, setFocus] = useState<string | null>(null)

    const handleStart = () => {
        if (target.trim()) {
            onStart(target.trim(), focus)
        }
    }

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-background rounded-xl shadow-lg max-w-md w-full border border-border">
                <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                    <div>
                        <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
                            <Wrench className="w-5 h-5 text-primary" />
                            RefactoringCrew
                        </h2>
                        <p className="text-sm text-muted-foreground mt-0.5">
                            Multi-agent code analysis
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="px-6 py-5 space-y-5">
                    {!isIndexed && (
                        <div className="p-3 bg-warning/10 border border-warning/30 rounded-lg">
                            <p className="text-warning text-sm">
                                Index a codebase first before running the crew.
                            </p>
                        </div>
                    )}

                    <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                            Target
                        </label>
                        <input
                            type="text"
                            value={target}
                            onChange={(e) => setTarget(e.target.value)}
                            placeholder="e.g., src/rag/retriever.py, BaseAgent, authentication"
                            className="w-full rounded-md border border-input bg-secondary px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                            disabled={!isIndexed}
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                            Focus Area
                        </label>
                        <p className="text-xs text-muted-foreground mb-3">
                            Select <strong className="text-foreground">Performance</strong> for actual code changes.
                        </p>
                        <div className="grid grid-cols-2 gap-2">
                            {focusOptions.map((option) => {
                                const Icon = option.icon
                                const isSelected = focus === option.value
                                return (
                                    <button
                                        key={option.label}
                                        onClick={() => setFocus(option.value)}
                                        disabled={!isIndexed}
                                        className={`p-3 rounded-lg border text-left transition-all ${isSelected
                                            ? 'bg-primary/10 border-primary/40 text-primary'
                                            : 'bg-secondary border-border text-foreground hover:border-primary/30'
                                        } ${!isIndexed ? 'opacity-50 cursor-not-allowed' : ''}`}
                                    >
                                        <div className="flex items-center gap-2">
                                            <Icon className="w-4 h-4" />
                                            <span className="font-medium text-sm">{option.label}</span>
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1">{option.description}</p>
                                    </button>
                                )
                            })}
                        </div>
                    </div>
                </div>

                <div className="px-6 py-4 border-t border-border flex items-center justify-end gap-3 bg-secondary/50">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 rounded-md text-sm font-medium border border-border bg-background text-foreground hover:bg-secondary transition-colors"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleStart}
                        disabled={!target.trim() || !isIndexed}
                        className="px-4 py-2 rounded-md text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <Play className="w-4 h-4" />
                        Start Analysis
                    </button>
                </div>
            </div>
        </div>
    )
}
