import { useState, useEffect, useCallback, useRef } from 'react'

export interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: Date
    intent?: string
}

interface WebSocketMessage {
    type: 'token' | 'done' | 'error' | 'intent'
    content?: string
}

export function useWebSocket(url: string) {
    const [messages, setMessages] = useState<Message[]>([])
    const [isConnected, setIsConnected] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [currentIntent, setCurrentIntent] = useState<string | null>(null)

    const wsRef = useRef<WebSocket | null>(null)
    const currentMessageRef = useRef<string>('')
    const messageIdRef = useRef<number>(0)

    // Connect to WebSocket
    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return

        const ws = new WebSocket(url)

        ws.onopen = () => {
            console.log('WebSocket connected')
            setIsConnected(true)
        }

        ws.onclose = () => {
            console.log('WebSocket disconnected')
            setIsConnected(false)
            // Attempt reconnection after 3 seconds
            setTimeout(connect, 3000)
        }

        ws.onerror = (error) => {
            console.error('WebSocket error:', error)
        }

        ws.onmessage = (event) => {
            try {
                const data: WebSocketMessage = JSON.parse(event.data)

                switch (data.type) {
                    case 'intent':
                        setCurrentIntent(data.content || null)
                        break

                    case 'token':
                        // Append token to current message
                        currentMessageRef.current += data.content || ''

                        // Update the last assistant message
                        setMessages(prev => {
                            const lastMsg = prev[prev.length - 1]
                            if (lastMsg && lastMsg.role === 'assistant') {
                                return [
                                    ...prev.slice(0, -1),
                                    { ...lastMsg, content: currentMessageRef.current }
                                ]
                            }
                            return prev
                        })
                        break

                    case 'done':
                        setIsLoading(false)
                        currentMessageRef.current = ''
                        setCurrentIntent(null)
                        break

                    case 'error':
                        setIsLoading(false)
                        setMessages(prev => [
                            ...prev,
                            {
                                id: `error-${Date.now()}`,
                                role: 'assistant',
                                content: `Error: ${data.content}`,
                                timestamp: new Date()
                            }
                        ])
                        currentMessageRef.current = ''
                        break
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e)
            }
        }

        wsRef.current = ws
    }, [url])

    // Send message
    const sendMessage = useCallback((content: string) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected')
            return
        }

        // Add user message
        const userMessage: Message = {
            id: `user-${++messageIdRef.current}`,
            role: 'user',
            content,
            timestamp: new Date()
        }

        // Add placeholder for assistant response
        const assistantMessage: Message = {
            id: `assistant-${++messageIdRef.current}`,
            role: 'assistant',
            content: '',
            timestamp: new Date()
        }

        setMessages(prev => [...prev, userMessage, assistantMessage])
        setIsLoading(true)
        currentMessageRef.current = ''

        // Send to WebSocket
        wsRef.current.send(JSON.stringify({
            message: content,
            use_rag: true
        }))
    }, [])

    // Connect on mount
    useEffect(() => {
        connect()
        return () => {
            wsRef.current?.close()
        }
    }, [connect])

    return {
        messages,
        setMessages,
        isConnected,
        isLoading,
        currentIntent,
        sendMessage,
        connect
    }
}
