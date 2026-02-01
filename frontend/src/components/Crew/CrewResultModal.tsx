import { X, Loader2, CheckCircle2 } from 'lucide-react'

interface CrewResultModalProps {
    title: string
    isRunning: boolean
    result: string | null
    error: string | null
    onClose: () => void
    onCancel?: () => void
}

export default function CrewResultModal({
    title,
    isRunning,
    result,
    error,
    onClose,
    onCancel
}: CrewResultModalProps) {
    return (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-surface rounded-2xl shadow-lg max-w-4xl w-full max-h-[90vh] overflow-hidden border border-border animate-fadeIn">
                <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface-secondary">
                    <h2 className="text-lg font-semibold text-text-primary">{title}</h2>
                    <button
                        onClick={isRunning && onCancel ? onCancel : onClose}
                        className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-surface-tertiary transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="px-6 py-6 max-h-[60vh] overflow-y-auto">
                    {error ? (
                        <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                            <p className="text-danger font-medium">Error</p>
                            <p className="text-danger/80 text-sm mt-1">{error}</p>
                        </div>
                    ) : result ? (
                        <div>
                            <p className="text-success font-medium mb-3 flex items-center gap-2">
                                <CheckCircle2 className="w-5 h-5" />
                                Complete
                            </p>
                            <div className="p-4 bg-surface-secondary rounded-xl border border-border">
                                <pre className="text-sm text-text-secondary whitespace-pre-wrap font-mono overflow-x-auto">
                                    {result}
                                </pre>
                            </div>
                        </div>
                    ) : isRunning ? (
                        <div className="flex flex-col items-center justify-center py-12 gap-4">
                            <Loader2 className="w-12 h-12 text-brand animate-spin" />
                            <p className="text-text-secondary">Running crew...</p>
                            {onCancel && (
                                <button
                                    onClick={onCancel}
                                    className="btn bg-red-50 border-red-200 text-danger hover:bg-red-100"
                                >
                                    Cancel
                                </button>
                            )}
                        </div>
                    ) : (
                        <p className="text-text-muted text-center py-8">Waiting...</p>
                    )}
                </div>

                <div className="px-6 py-4 border-t border-border bg-surface-secondary">
                    <button
                        onClick={onClose}
                        className="btn btn-secondary"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    )
}
