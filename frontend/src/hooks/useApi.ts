import { useState, useEffect } from 'react'

interface IndexStatus {
    indexed: boolean
    chunks: number
    directory: string | null
}

interface IndexResult {
    status: string
    chunks_indexed: number
    files_processed: number
    duration_seconds: number
}

const API_BASE = 'http://localhost:8000'

export function useApi() {
    const [status, setStatus] = useState<IndexStatus | null>(null)
    const [isIndexing, setIsIndexing] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Fetch status on mount
    useEffect(() => {
        fetchStatus()
    }, [])

    const fetchStatus = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/status`)
            if (response.ok) {
                const data = await response.json()
                setStatus(data)
            }
        } catch (e) {
            console.error('Failed to fetch status:', e)
        }
    }

    const indexCodebase = async (directory: string, forceReindex: boolean = false): Promise<IndexResult | null> => {
        setIsIndexing(true)
        setError(null)

        try {
            const response = await fetch(`${API_BASE}/api/index`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ directory, force_reindex: forceReindex })
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || 'Indexing failed')
            }

            const result = await response.json()
            await fetchStatus() // Refresh status
            return result
        } catch (e) {
            const errorMessage = e instanceof Error ? e.message : 'Unknown error'
            setError(errorMessage)
            return null
        } finally {
            setIsIndexing(false)
        }
    }

    const cloneAndIndex = async (githubUrl: string, forceReindex: boolean = false): Promise<IndexResult | null> => {
        setIsIndexing(true)
        setError(null)

        try {
            const response = await fetch(`${API_BASE}/api/clone-and-index`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ github_url: githubUrl, force_reindex: forceReindex })
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || 'Clone and index failed')
            }

            const result = await response.json()
            await fetchStatus() // Refresh status
            return result
        } catch (e) {
            const errorMessage = e instanceof Error ? e.message : 'Unknown error'
            setError(errorMessage)
            return null
        } finally {
            setIsIndexing(false)
        }
    }

    const clearIndex = async (): Promise<boolean> => {
        try {
            const response = await fetch(`${API_BASE}/api/clear-index`, {
                method: 'POST'
            })
            if (response.ok) {
                setStatus({ indexed: false, chunks: 0, directory: null })
                return true
            }
            return false
        } catch (e) {
            console.error('Failed to clear index:', e)
            return false
        }
    }

    return {
        status,
        isIndexing,
        error,
        indexCodebase,
        cloneAndIndex,
        clearIndex,
        fetchStatus
    }
}
