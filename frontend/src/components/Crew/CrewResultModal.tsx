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
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-background rounded-xl shadow-lg max-w-4xl w-full max-h-[90vh] overflow-hidden border border-border">
                <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-secondary/50">
                    <h2 className="text-lg font-semibold text-foreground">{title}</h2>
                    <button
                        onClick={isRunning && onCancel ? onCancel : onClose}
                        className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="px-6 py-6 max-h-[60vh] overflow-y-auto">
                    {error ? (
                        <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
                            <p className="text-destructive font-medium">Error</p>
                            <p className="text-destructive/80 text-sm mt-1">{error}</p>
                        </div>
                    ) : result ? (
                        <div>
                            <p className="text-success font-medium mb-3 flex items-center gap-2">
                                <CheckCircle2 className="w-5 h-5" />
                                Complete
                            </p>
                            <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                                <pre className="text-sm text-muted-foreground whitespace-pre-wrap font-mono overflow-x-auto">
                                    {result}
                                </pre>
                            </div>
                        </div>
                    ) : isRunning ? (
                        <div className="flex flex-col items-center justify-center py-12 gap-4">
                            <Loader2 className="w-12 h-12 text-primary animate-spin" />
                            <p className="text-muted-foreground">Running crew...</p>
                            {onCancel && (
                                <button
                                    onClick={onCancel}
                                    className="px-4 py-2 rounded-md text-sm font-medium bg-destructive/10 border border-destructive/30 text-destructive hover:bg-destructive/20 transition-colors"
                                >
                                    Cancel
                                </button>
                            )}
                        </div>
                    ) : (
                        <p className="text-muted-foreground text-center py-8">Waiting...</p>
                    )}
                </div>

                <div className="px-6 py-4 border-t border-border bg-secondary/50">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 rounded-md text-sm font-medium border border-border bg-background text-foreground hover:bg-secondary transition-colors"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    )
}
