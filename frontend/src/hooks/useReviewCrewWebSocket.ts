import { useState, useCallback, useRef } from 'react'

const WS_URL = 'ws://localhost:8000/ws/review'

export interface ReviewCrewState {
    isRunning: boolean
    result: string | null
    error: string | null
}

export function useReviewCrewWebSocket() {
    const [state, setState] = useState<ReviewCrewState>({
        isRunning: false,
        result: null,
        error: null
    })

    const wsRef = useRef<WebSocket | null>(null)

    const startReview = useCallback((code: string, file: string) => {
        setState({ isRunning: true, result: null, error: null })
        const ws = new WebSocket(WS_URL)

        ws.onopen = () => {
            ws.send(JSON.stringify({ code, file }))
        }

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data)
                if (data.type === 'crew_complete' || data.type === 'crew_done') {
                    setState(prev => ({
                        ...prev,
                        isRunning: false,
                        result: data.synthesis || data.result || ''
                    }))
                    ws.close()
                } else if (data.type === 'error') {
                    setState(prev => ({
                        ...prev,
                        isRunning: false,
                        error: data.content || 'Unknown error'
                    }))
                    ws.close()
                }
            } catch (e) {
                console.error('Failed to parse message:', e)
            }
        }

        ws.onerror = () => {
            setState(prev => ({ ...prev, isRunning: false, error: 'Connection failed' }))
        }

        wsRef.current = ws
    }, [])

    const cancel = useCallback(() => {
        wsRef.current?.close()
        setState(prev => ({ ...prev, isRunning: false }))
    }, [])

    const reset = useCallback(() => {
        setState({ isRunning: false, result: null, error: null })
    }, [])

    return { state, startReview, cancel, reset }
}
