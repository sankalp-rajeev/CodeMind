import { useState } from 'react'
import { TestTube, X, Play, Loader2 } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

interface TestPanelProps {
    onStart: (code: string, file: string) => void
    onClose: () => void
}

export default function TestPanel({ onStart, onClose }: TestPanelProps) {
    const [filePath, setFilePath] = useState('src/rag/retriever.py')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleStart = async () => {
        if (!filePath.trim()) return
        setLoading(true)
        setError(null)
        try {
            const res = await fetch(`${API_BASE}/api/read-file?path=${encodeURIComponent(filePath.trim())}`)
            if (!res.ok) throw new Error('File not found')
            const data = await res.json()
            onStart(data.content, data.path || filePath)
            onClose()
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to read file')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-surface rounded-2xl shadow-lg max-w-md w-full border border-border animate-fadeIn">
                <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                    <div>
                        <h2 className="text-lg font-semibold text-text-primary flex items-center gap-2">
                            <TestTube className="w-5 h-5 text-amber-500" />
                            TestingCrew
                        </h2>
                        <p className="text-sm text-text-secondary mt-0.5">
                            Iterative test generation until coverage targets met
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-surface-tertiary transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="px-6 py-5 space-y-4">
                    {error && (
                        <div className="p-3 bg-red-50 border border-red-200 rounded-xl">
                            <p className="text-danger text-sm">{error}</p>
                        </div>
                    )}

                    <div>
                        <label className="block text-sm font-medium text-text-primary mb-2">
                            File path to generate tests for
                        </label>
                        <input
                            type="text"
                            value={filePath}
                            onChange={(e) => setFilePath(e.target.value)}
                            placeholder="e.g., src/rag/retriever.py"
                            className="input"
                        />
                    </div>
                </div>

                <div className="px-6 py-4 border-t border-border flex items-center justify-end gap-3 bg-surface-secondary">
                    <button
                        onClick={onClose}
                        className="btn btn-secondary"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleStart}
                        disabled={!filePath.trim() || loading}
                        className="btn btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? (
                            <><Loader2 className="w-4 h-4 animate-spin" /> Loading...</>
                        ) : (
                            <><Play className="w-4 h-4" /> Start Crew</>
                        )}
                    </button>
                </div>
            </div>
        </div>
    )
}
