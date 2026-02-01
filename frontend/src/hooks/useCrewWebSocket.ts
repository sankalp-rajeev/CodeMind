import { useState, useCallback, useRef } from 'react'

export interface AgentState {
    name: string
    icon: string
    status: 'pending' | 'active' | 'done'
    output: string[]
    summary?: string
}

export interface CrewState {
    isRunning: boolean
    target: string
    focus: string | null
    agents: AgentState[]
    currentAgentIndex: number
    finalResult: string | null
    agentOutputs: { agent: string; output: string }[]
    error: string | null
}

interface AgentOutput {
    agent: string
    output: string
}

interface CrewMessage {
    type: 'crew_start' | 'agent_start' | 'agent_output' | 'agent_done' | 'agent_result' | 'crew_done' | 'error'
    agent?: string
    icon?: string
    index?: number
    total?: number
    content?: string
    summary?: string
    result?: string
    output?: string
    agents?: string[]
    agent_outputs?: AgentOutput[]
    target?: string
    focus?: string
    tasks_completed?: number
}

const WS_URL = 'ws://localhost:8000/ws/refactor'

export function useCrewWebSocket() {
    const [crewState, setCrewState] = useState<CrewState>({
        isRunning: false,
        target: '',
        focus: null,
        agents: [],
        currentAgentIndex: -1,
        finalResult: null,
        agentOutputs: [],
        error: null
    })

    const wsRef = useRef<WebSocket | null>(null)

    const startRefactoring = useCallback((target: string, focus: string | null) => {
        // Reset state
        setCrewState({
            isRunning: true,
            target,
            focus,
            agents: [],
            currentAgentIndex: -1,
            finalResult: null,
            agentOutputs: [],
            error: null
        })

        // Connect WebSocket
        const ws = new WebSocket(WS_URL)

        ws.onopen = () => {
            console.log('Crew WebSocket connected')
            ws.send(JSON.stringify({ target, focus }))
        }

        ws.onmessage = (event) => {
            try {
                const data: CrewMessage = JSON.parse(event.data)

                switch (data.type) {
                    case 'crew_start':
                        // Initialize agents
                        const agentNames = data.agents || []
                        setCrewState(prev => ({
                            ...prev,
                            agents: agentNames.map(name => ({
                                name,
                                icon: 'pending',
                                status: 'pending' as const,
                                output: []
                            }))
                        }))
                        break

                    case 'agent_start':
                        setCrewState(prev => ({
                            ...prev,
                            currentAgentIndex: data.index ?? prev.currentAgentIndex,
                            agents: prev.agents.map((agent, i) =>
                                i === data.index
                                    ? { ...agent, status: 'active' as const, icon: 'active' }
                                    : agent
                            )
                        }))
                        break

                    case 'agent_output':
                        setCrewState(prev => ({
                            ...prev,
                            agents: prev.agents.map(agent =>
                                agent.name === data.agent
                                    ? { ...agent, output: [...agent.output, data.content || ''] }
                                    : agent
                            )
                        }))
                        break

                    case 'agent_done':
                        setCrewState(prev => ({
                            ...prev,
                            agents: prev.agents.map((agent, i) =>
                                i === data.index
                                    ? { ...agent, status: 'done' as const, icon: 'done', summary: data.summary }
                                    : agent
                            )
                        }))
                        break

                    case 'agent_result':
                        // Real agent output from the crew
                        setCrewState(prev => ({
                            ...prev,
                            agentOutputs: [...prev.agentOutputs, {
                                agent: data.agent || 'Unknown',
                                output: data.output || ''
                            }]
                        }))
                        break

                    case 'crew_done':
                        setCrewState(prev => ({
                            ...prev,
                            isRunning: false,
                            finalResult: data.result || '',
                            agentOutputs: data.agent_outputs || prev.agentOutputs
                        }))
                        ws.close()
                        break

                    case 'error':
                        setCrewState(prev => ({
                            ...prev,
                            isRunning: false,
                            error: data.content || 'Unknown error'
                        }))
                        ws.close()
                        break
                }
            } catch (e) {
                console.error('Failed to parse crew message:', e)
            }
        }

        ws.onerror = (error) => {
            console.error('Crew WebSocket error:', error)
            setCrewState(prev => ({
                ...prev,
                isRunning: false,
                error: 'WebSocket connection failed'
            }))
        }

        ws.onclose = () => {
            console.log('Crew WebSocket closed')
        }

        wsRef.current = ws
    }, [])

    const cancelRefactoring = useCallback(() => {
        wsRef.current?.close()
        setCrewState(prev => ({
            ...prev,
            isRunning: false,
            error: 'Cancelled by user'
        }))
    }, [])

    const reset = useCallback(() => {
        setCrewState({
            isRunning: false,
            target: '',
            focus: null,
            agents: [],
            currentAgentIndex: -1,
            finalResult: null,
            agentOutputs: [],
            error: null
        })
    }, [])

    return {
        crewState,
        startRefactoring,
        cancelRefactoring,
        reset
    }
}
