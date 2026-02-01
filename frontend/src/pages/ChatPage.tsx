import { useState, useRef, useEffect } from 'react'
import { WifiOff, FolderOpen, Loader2, Wrench, TestTube, FileSearch, BookOpen, MessageSquare, PanelLeftClose, PanelLeft, Code2, BarChart3 } from 'lucide-react'
import ChatMessage from '../components/Chat/ChatMessage'
import ChatInput from '../components/Chat/ChatInput'
import RoutingFlow from '../components/Chat/RoutingFlow'
import { useWebSocket, Message } from '../hooks/useWebSocket'
import { useApi } from '../hooks/useApi'
import { useCrewWebSocket } from '../hooks/useCrewWebSocket'
import { useTestCrewWebSocket } from '../hooks/useTestCrewWebSocket'
import { useReviewCrewWebSocket } from '../hooks/useReviewCrewWebSocket'
import CrewProgress from '../components/Crew/CrewProgress'
import CrewResultModal from '../components/Crew/CrewResultModal'
import RefactorPanel from '../components/Crew/RefactorPanel'
import TestPanel from '../components/Crew/TestPanel'
import ReviewPanel from '../components/Crew/ReviewPanel'
import DocsPage from '../components/Docs/DocsPage'
import MetricsPage from '../components/Metrics/MetricsPage'
import FileExplorer from '../components/Explorer/FileExplorer'

const WS_URL = 'ws://localhost:8000/ws/chat'

export default function ChatPage() {
    const {
        messages,
        setMessages,
        isConnected,
        isLoading,
        currentIntent,
        routingSteps,
        sendMessage
    } = useWebSocket(WS_URL)

    const { status, isIndexing, error, indexCodebase } = useApi()
    const { crewState, startRefactoring, cancelRefactoring, reset } = useCrewWebSocket()
    const { state: testState, startTesting, cancel: cancelTest, reset: resetTest } = useTestCrewWebSocket()
    const { state: reviewState, startReview, cancel: cancelReview, reset: resetReview } = useReviewCrewWebSocket()

    const [activeTab, setActiveTab] = useState<'chat' | 'docs' | 'metrics'>('chat')
    const [showIndexPanel, setShowIndexPanel] = useState(false)
    const [showRefactorPanel, setShowRefactorPanel] = useState(false)
    const [showTestPanel, setShowTestPanel] = useState(false)
    const [showReviewPanel, setShowReviewPanel] = useState(false)
    const [showCrewProgress, setShowCrewProgress] = useState(false)
    const [indexPath, setIndexPath] = useState('./data/test-repo')
    const [showExplorer, setShowExplorer] = useState(true)
    const [insertPath, setInsertPath] = useState<string | null>(null)

    const messagesEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (messages.length === 0) {
            const welcomeMessage: Message = {
                id: 'welcome',
                role: 'assistant',
                content: `**CodeMind** — AI-Powered Code Intelligence

Capabilities:
- **Explore** — Understand code structure and dependencies
- **Refactor** — Optimize performance and code quality  
- **Test** — Generate comprehensive test suites
- **Security** — Identify vulnerabilities and fixes
- **Document** — Create clear documentation

${status?.indexed ? `Index Status: **${status.chunks}** chunks ready for analysis` : 'To begin, index a codebase using the folder icon in the toolbar.'}`,
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

    const lastUserQuery = messages.filter(m => m.role === 'user').pop()?.content

    return (
        <div className="flex flex-col h-screen bg-surface-secondary">
            {/* Header */}
            <header className="flex-shrink-0 bg-surface border-b border-border">
                <div className="max-w-5xl mx-auto px-6 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-5">
                        {/* Logo */}
                        <div className="flex items-center gap-3">
                            <div className="w-9 h-9 rounded-lg bg-brand flex items-center justify-center shadow-sm">
                                <Code2 className="w-5 h-5 text-white" strokeWidth={2} />
                            </div>
                            <div>
                                <h1 className="text-lg font-semibold text-text-primary tracking-tight">CodeMind</h1>
                                <div className="flex items-center gap-2 text-xs">
                                    {isConnected ? (
                                        <span className="flex items-center gap-1.5 text-success">
                                            <span className="w-1.5 h-1.5 rounded-full bg-success" />
                                            Online
                                        </span>
                                    ) : (
                                        <span className="flex items-center gap-1.5 text-danger">
                                            <WifiOff className="w-3 h-3" />
                                            Offline
                                        </span>
                                    )}
                                    {status?.indexed && (
                                        <span className="text-text-muted">· {status.chunks} chunks</span>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Tabs */}
                        <div className="flex rounded-lg border border-border p-1 bg-surface-secondary">
                            <button
                                onClick={() => setActiveTab('chat')}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${activeTab === 'chat' ? 'bg-surface text-text-primary shadow-sm' : 'text-text-secondary hover:text-text-primary'}`}
                            >
                                <MessageSquare className="w-4 h-4" />
                                Chat
                            </button>
                            <button
                                onClick={() => setActiveTab('docs')}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${activeTab === 'docs' ? 'bg-surface text-text-primary shadow-sm' : 'text-text-secondary hover:text-text-primary'}`}
                            >
                                <BookOpen className="w-4 h-4" />
                                Docs
                            </button>
                            <button
                                onClick={() => setActiveTab('metrics')}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${activeTab === 'metrics' ? 'bg-surface text-text-primary shadow-sm' : 'text-text-secondary hover:text-text-primary'}`}
                            >
                                <BarChart3 className="w-4 h-4" />
                                Metrics
                            </button>
                        </div>
                    </div>

                    <div className="flex items-center gap-1">
                        {currentIntent && (
                            <span className="badge badge-brand mr-2">
                                {currentIntent}
                            </span>
                        )}
                        
                        {/* Tool buttons */}
                        <button
                            onClick={() => setShowRefactorPanel(true)}
                            className="p-2.5 rounded-lg text-text-secondary hover:text-brand hover:bg-brand-muted transition-colors"
                            title="RefactoringCrew"
                        >
                            <Wrench className="w-5 h-5" />
                        </button>
                        <button
                            onClick={() => setShowTestPanel(true)}
                            className="p-2.5 rounded-lg text-text-secondary hover:text-brand hover:bg-brand-muted transition-colors"
                            title="TestingCrew"
                        >
                            <TestTube className="w-5 h-5" />
                        </button>
                        <button
                            onClick={() => setShowReviewPanel(true)}
                            className="p-2.5 rounded-lg text-text-secondary hover:text-brand hover:bg-brand-muted transition-colors"
                            title="CodeReviewCrew"
                        >
                            <FileSearch className="w-5 h-5" />
                        </button>
                        
                        <div className="w-px h-6 bg-border mx-1" />
                        
                        <button
                            onClick={() => setShowIndexPanel(!showIndexPanel)}
                            className={`p-2.5 rounded-lg transition-colors ${showIndexPanel ? 'text-brand bg-brand-muted' : 'text-text-secondary hover:text-brand hover:bg-brand-muted'}`}
                            title="Index codebase"
                        >
                            <FolderOpen className="w-5 h-5" />
                        </button>
                        {status?.indexed && (
                            <button
                                onClick={() => setShowExplorer(!showExplorer)}
                                className={`p-2.5 rounded-lg transition-colors ${showExplorer ? 'text-brand bg-brand-muted' : 'text-text-secondary hover:text-brand hover:bg-brand-muted'}`}
                                title="File explorer"
                            >
                                {showExplorer ? <PanelLeftClose className="w-5 h-5" /> : <PanelLeft className="w-5 h-5" />}
                            </button>
                        )}
                    </div>
                </div>

                {/* Index Panel */}
                {showIndexPanel && (
                    <div className="border-t border-border bg-surface-secondary">
                        <div className="max-w-5xl mx-auto px-6 py-4">
                            <div className="flex items-center gap-3">
                                <input
                                    type="text"
                                    value={indexPath}
                                    onChange={(e) => setIndexPath(e.target.value)}
                                    placeholder="Path to codebase (e.g. ./src or ./data/test-repo)"
                                    className="input flex-1"
                                />
                                <button
                                    onClick={handleIndex}
                                    disabled={isIndexing}
                                    className="btn btn-primary"
                                >
                                    {isIndexing ? (
                                        <>
                                            <Loader2 className="w-4 h-4 animate-spin" /> Indexing...
                                        </>
                                    ) : (
                                        'Index Codebase'
                                    )}
                                </button>
                            </div>
                            {error && (
                                <p className="mt-2 text-sm text-danger">{error}</p>
                            )}
                        </div>
                    </div>
                )}
            </header>

            {/* Main content */}
            <div className="flex flex-1 min-h-0">
                {/* Sidebar */}
                {showExplorer && status?.indexed && (
                    <aside className="w-60 flex-shrink-0 border-r border-border bg-surface flex flex-col overflow-hidden">
                        <FileExplorer
                            isIndexed={status?.indexed || false}
                            indexedDirectory={status?.directory}
                            onSelectFile={(path) => setInsertPath(path)}
                            className="flex-1 min-h-0"
                        />
                    </aside>
                )}
                
                {/* Chat area */}
                <div className="flex flex-col flex-1 min-w-0">
                    <main className="flex-1 overflow-y-auto">
                        {activeTab === 'chat' ? (
                            <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
                                {messages.map((message) => (
                                    <ChatMessage key={message.id} message={message} />
                                ))}
                                {routingSteps.length > 0 && (
                                    <RoutingFlow steps={routingSteps} query={lastUserQuery} />
                                )}
                                {isLoading && (
                                    <div className="flex items-center gap-3 text-text-muted py-2">
                                        <div className="flex gap-1">
                                            <span className="w-2 h-2 bg-brand rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                            <span className="w-2 h-2 bg-brand rounded-full animate-bounce" style={{ animationDelay: '120ms' }} />
                                            <span className="w-2 h-2 bg-brand rounded-full animate-bounce" style={{ animationDelay: '240ms' }} />
                                        </div>
                                        <span className="text-sm">Processing...</span>
                                    </div>
                                )}
                                <div ref={messagesEndRef} />
                            </div>
                        ) : activeTab === 'docs' ? (
                            <DocsPage />
                        ) : (
                            <MetricsPage />
                        )}
                    </main>

                    {/* Input */}
                    {activeTab === 'chat' && (
                        <footer className="flex-shrink-0 border-t border-border bg-surface">
                            <div className="max-w-3xl mx-auto px-6 py-4">
                                <ChatInput
                                    onSend={handleSendMessage}
                                    isLoading={isLoading}
                                    disabled={!isConnected}
                                    insertText={insertPath}
                                    onInsertConsumed={() => setInsertPath(null)}
                                />
                            </div>
                        </footer>
                    )}
                </div>
            </div>

            {/* Modals */}
            {showRefactorPanel && (
                <RefactorPanel
                    onStart={(target, focus) => {
                        setShowRefactorPanel(false)
                        setShowCrewProgress(true)
                        startRefactoring(target, focus)
                    }}
                    onClose={() => setShowRefactorPanel(false)}
                    isIndexed={status?.indexed || false}
                />
            )}

            {showCrewProgress && (
                <CrewProgress
                    crewState={crewState}
                    onClose={() => {
                        // Save results to chat before closing (check !isRunning instead of status)
                        if (!crewState.isRunning && crewState.agentOutputs.length > 0) {
                            const target = crewState.target || 'target'
                            const outputSummary = crewState.agentOutputs
                                .map(o => `### ${o.agent}\n${o.output.slice(0, 1500)}${o.output.length > 1500 ? '...' : ''}`)
                                .join('\n\n---\n\n')
                            
                            setMessages(prev => [...prev, {
                                id: `crew-${Date.now()}`,
                                role: 'assistant',
                                content: `**RefactoringCrew Analysis** — \`${target}\`\n\n${outputSummary}`,
                                timestamp: new Date()
                            }])
                        }
                        setShowCrewProgress(false)
                        reset()
                    }}
                    onCancel={() => cancelRefactoring()}
                />
            )}

            {showTestPanel && (
                <TestPanel
                    onStart={(code, file) => {
                        setShowTestPanel(false)
                        startTesting(code, file)
                    }}
                    onClose={() => setShowTestPanel(false)}
                />
            )}

            {showReviewPanel && (
                <ReviewPanel
                    onStart={(code, file) => {
                        setShowReviewPanel(false)
                        startReview(code, file)
                    }}
                    onClose={() => setShowReviewPanel(false)}
                />
            )}

            {(testState.isRunning || testState.result || testState.error) && (
                <CrewResultModal
                    title="TestingCrew"
                    isRunning={testState.isRunning}
                    result={testState.result}
                    error={testState.error}
                    onClose={() => {
                        if (testState.result) {
                            setMessages(prev => [...prev, {
                                id: `test-${Date.now()}`,
                                role: 'assistant',
                                content: `**TestingCrew Result**\n\n${testState.result.slice(0, 3000)}${testState.result.length > 3000 ? '...' : ''}`,
                                timestamp: new Date()
                            }])
                        }
                        resetTest()
                    }}
                    onCancel={() => cancelTest()}
                />
            )}

            {(reviewState.isRunning || reviewState.result || reviewState.error) && (
                <CrewResultModal
                    title="CodeReviewCrew"
                    isRunning={reviewState.isRunning}
                    result={reviewState.result}
                    error={reviewState.error}
                    onClose={() => {
                        if (reviewState.result) {
                            setMessages(prev => [...prev, {
                                id: `review-${Date.now()}`,
                                role: 'assistant',
                                content: `**CodeReviewCrew Result**\n\n${reviewState.result.slice(0, 3000)}${reviewState.result.length > 3000 ? '...' : ''}`,
                                timestamp: new Date()
                            }])
                        }
                        resetReview()
                    }}
                    onCancel={() => cancelReview()}
                />
            )}
        </div>
    )
}
