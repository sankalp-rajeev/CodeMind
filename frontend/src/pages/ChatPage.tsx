import { useState, useRef, useEffect } from 'react'
import { Bot, Wifi, WifiOff, FolderOpen, Loader2, Zap } from 'lucide-react'
import ChatMessage from '../components/Chat/ChatMessage'
import ChatInput from '../components/Chat/ChatInput'
import { useWebSocket, Message } from '../hooks/useWebSocket'
import { useApi } from '../hooks/useApi'

const WS_URL = 'ws://localhost:8000/ws/chat'

export default function ChatPage() {
    const {
        messages,
        setMessages,
        isConnected,
        isLoading,
        currentIntent,
        sendMessage
    } = useWebSocket(WS_URL)

    const { status, isIndexing, indexCodebase } = useApi()
    const [showIndexPanel, setShowIndexPanel] = useState(false)
    const [indexPath, setIndexPath] = useState('./data/test-repo')

    const messagesEndRef = useRef<HTMLDivElement>(null)

    // Add welcome message on mount
    useEffect(() => {
        if (messages.length === 0) {
            const welcomeMessage: Message = {
                id: 'welcome',
                role: 'assistant',
                content: `Welcome to **CodeMind AI**

I'm your multi-agent codebase assistant. I can help you:
- **Explore** - Understand code structure and functionality
- **Refactor** - Improve code quality and performance
- **Test** - Generate unit tests
- **Security** - Find vulnerabilities
- **Document** - Add documentation

${status?.indexed ? `Codebase indexed: **${status.chunks}** chunks ready` : 'No codebase indexed yet. Click the folder icon to index one.'}`,
                timestamp: new Date()
            }
            setMessages([welcomeMessage])
        }
    }, [status])

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const handleSendMessage = async (content: string) => {
        if (!status?.indexed) {
            // Add error message if not indexed
            setMessages(prev => [...prev, {
                id: `error-${Date.now()}`,
                role: 'assistant',
                content: 'Please index a codebase first. Click the folder icon in the header.',
                timestamp: new Date()
            }])
            return
        }
        sendMessage(content)
    }

    const handleIndex = async () => {
        const result = await indexCodebase(indexPath, true)
        if (result) {
            setMessages(prev => [...prev, {
                id: `system-${Date.now()}`,
                role: 'assistant',
                content: `**Indexed successfully**
- Chunks: ${result.chunks_indexed}
- Files: ${result.files_processed}
- Time: ${result.duration_seconds}s

You can now ask questions about the code.`,
                timestamp: new Date()
            }])
            setShowIndexPanel(false)
        }
    }

    return (
        <div className="flex flex-col h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
            {/* Header */}
            <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm">
                <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                            <Bot className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-lg font-semibold text-white">CodeMind AI</h1>
                            <div className="flex items-center gap-2 text-xs">
                                {isConnected ? (
                                    <span className="flex items-center gap-1 text-emerald-400">
                                        <Wifi className="w-3 h-3" /> Connected
                                    </span>
                                ) : (
                                    <span className="flex items-center gap-1 text-red-400">
                                        <WifiOff className="w-3 h-3" /> Disconnected
                                    </span>
                                )}
                                {status?.indexed && (
                                    <span className="text-slate-400">• {status.chunks} chunks</span>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        {currentIntent && (
                            <span className="px-2 py-1 text-xs bg-blue-500/20 text-blue-300 rounded-full flex items-center gap-1">
                                <Zap className="w-3 h-3" /> {currentIntent}
                            </span>
                        )}
                        <button
                            onClick={() => setShowIndexPanel(!showIndexPanel)}
                            className="p-2 rounded-lg bg-slate-700/50 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
                            title="Index codebase"
                        >
                            <FolderOpen className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                {/* Index Panel */}
                {showIndexPanel && (
                    <div className="border-t border-slate-700/50 bg-slate-800/50 backdrop-blur-sm">
                        <div className="max-w-4xl mx-auto px-4 py-3">
                            <div className="flex items-center gap-3">
                                <input
                                    type="text"
                                    value={indexPath}
                                    onChange={(e) => setIndexPath(e.target.value)}
                                    placeholder="Path to codebase..."
                                    className="flex-1 px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                                <button
                                    onClick={handleIndex}
                                    disabled={isIndexing}
                                    className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 rounded-lg text-white font-medium flex items-center gap-2 transition-colors"
                                >
                                    {isIndexing ? (
                                        <>
                                            <Loader2 className="w-4 h-4 animate-spin" /> Indexing...
                                        </>
                                    ) : (
                                        'Index'
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </header>

            {/* Messages */}
            <main className="flex-1 overflow-y-auto">
                <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
                    {messages.map((message) => (
                        <ChatMessage key={message.id} message={message} />
                    ))}

                    {isLoading && (
                        <div className="flex items-center gap-3 text-slate-400">
                            <div className="flex gap-1">
                                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                            <span className="text-sm">Thinking...</span>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>
            </main>

            {/* Input */}
            <footer className="border-t border-slate-700/50 bg-slate-900/80 backdrop-blur-sm">
                <div className="max-w-4xl mx-auto px-4 py-4">
                    <ChatInput
                        onSend={handleSendMessage}
                        isLoading={isLoading}
                        disabled={!isConnected}
                    />
                </div>
            </footer>
        </div>
    )
}
