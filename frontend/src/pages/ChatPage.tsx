import { useState, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import {
    Sheet,
    SheetContent,
    SheetHeader,
    SheetTitle,
} from '@/components/ui/sheet'
import {
    Code2,
    PanelLeftClose,
    PanelLeft,
    Search,
    RefreshCw,
    TestTube,
    FileSearch,
    FileText,
    MessageSquare,
    BarChart3,
    BookOpen,
    Send,
    Loader2,
    FolderOpen,
    Bot,
    Check,
    Circle,
    ChevronDown,
    ChevronUp,
    X,
    Home,
} from 'lucide-react'
import ChatMessage from '../components/Chat/ChatMessage'
import RoutingFlow from '../components/Chat/RoutingFlow'
import { useWebSocket, Message } from '../hooks/useWebSocket'
import { useApi } from '../hooks/useApi'
import { useCrewWebSocket } from '../hooks/useCrewWebSocket'
import { useTestCrewWebSocket } from '../hooks/useTestCrewWebSocket'
import { useReviewCrewWebSocket } from '../hooks/useReviewCrewWebSocket'
import RefactorPanel from '../components/Crew/RefactorPanel'
import TestPanel from '../components/Crew/TestPanel'
import ReviewPanel from '../components/Crew/ReviewPanel'
import DocsPage from '../components/Docs/DocsPage'
import MetricsPage from '../components/Metrics/MetricsPage'
import FileExplorer from '../components/Explorer/FileExplorer'
import HeroPage from '../components/Hero/HeroPage'

const WS_URL = 'ws://localhost:8000/ws/chat'

const quickActions = [
    { id: 'index', label: 'Index', icon: Search },
    { id: 'refactor', label: 'Refactor', icon: RefreshCw },
    { id: 'test', label: 'Test', icon: TestTube },
    { id: 'review', label: 'Review', icon: FileSearch },
    { id: 'docs', label: 'Docs', icon: FileText },
]

interface AgentCardProps {
    agent: {
        name: string
        status: 'pending' | 'active' | 'done'
        output?: string
    }
}

function AgentCard({ agent }: AgentCardProps) {
    const [expanded, setExpanded] = useState(false)

    return (
        <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-3">
                    <div
                        className={cn(
                            "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
                            agent.status === "done" && "bg-success/10 text-success",
                            agent.status === "active" && "bg-primary/10 text-primary",
                            agent.status === "pending" && "bg-secondary text-muted-foreground"
                        )}
                    >
                        <Bot className="h-5 w-5" />
                    </div>
                    <div className="space-y-1">
                        <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">{agent.name}</span>
                            <Badge
                                variant="secondary"
                                className={cn(
                                    "text-xs",
                                    agent.status === "done" && "bg-success/10 text-success border-success/20",
                                    agent.status === "active" && "bg-primary/10 text-primary border-primary/20",
                                    agent.status === "pending" && "bg-secondary text-muted-foreground"
                                )}
                            >
                                {agent.status === "done" && <Check className="mr-1 h-3 w-3" />}
                                {agent.status === "active" && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
                                {agent.status === "pending" && <Circle className="mr-1 h-3 w-3" />}
                                {agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}
                            </Badge>
                        </div>
                    </div>
                </div>
                {agent.output && (
                    <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0 text-muted-foreground"
                        onClick={() => setExpanded(!expanded)}
                    >
                        {expanded ? (
                            <ChevronUp className="h-4 w-4" />
                        ) : (
                            <ChevronDown className="h-4 w-4" />
                        )}
                    </Button>
                )}
            </div>
            {agent.status === "active" && (
                <div className="mt-3">
                    <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                        <div className="h-full bg-primary animate-pulse w-2/3" />
                    </div>
                    <p className="mt-1.5 text-xs text-muted-foreground">Processing...</p>
                </div>
            )}
            {expanded && agent.output && (
                <div className="mt-3 rounded-md bg-secondary/50 p-3">
                    <p className="font-mono text-xs text-foreground/80 whitespace-pre-wrap">{agent.output.slice(0, 500)}{agent.output.length > 500 ? '...' : ''}</p>
                </div>
            )}
        </div>
    )
}

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

    const { status, isIndexing, error, indexCodebase, cloneAndIndex, clearIndex } = useApi()
    const { crewState, startRefactoring, cancelRefactoring, reset } = useCrewWebSocket()
    const { state: testState, startTesting, cancel: cancelTest, reset: resetTest } = useTestCrewWebSocket()
    const { state: reviewState, startReview, cancel: cancelReview, reset: resetReview } = useReviewCrewWebSocket()

    const [activeTab, setActiveTab] = useState<string>('chat')
    const [sidebarOpen, setSidebarOpen] = useState(true)
    const [showIndexPanel, setShowIndexPanel] = useState(false)
    const [showRefactorPanel, setShowRefactorPanel] = useState(false)
    const [showTestPanel, setShowTestPanel] = useState(false)
    const [showReviewPanel, setShowReviewPanel] = useState(false)
    const [agentProgressOpen, setAgentProgressOpen] = useState(false)
    const [indexPath, setIndexPath] = useState('./data/test-repo')
    const [insertPath, setInsertPath] = useState<string | null>(null)
    const [inputValue, setInputValue] = useState('')

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

${status?.indexed ? `Index Status: **${status.chunks}** chunks ready for analysis` : 'To begin, index a codebase using the Index button in the sidebar.'}`,
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

    // Handle insertPath from file explorer
    useEffect(() => {
        if (insertPath) {
            setInputValue(prev => prev + (prev ? ' ' : '') + insertPath)
            setInsertPath(null)
        }
    }, [insertPath])

    const handleSendMessage = async (e?: React.FormEvent) => {
        e?.preventDefault()
        if (!inputValue.trim() || isLoading) return

        if (!status?.indexed) {
            setMessages(prev => [...prev, {
                id: `error-${Date.now()}`,
                role: 'assistant',
                content: 'Please index a codebase first. Use the Index button in the sidebar.',
                timestamp: new Date()
            }])
            return
        }
        sendMessage(inputValue.trim())
        setInputValue('')
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

    const handleQuickAction = (actionId: string) => {
        switch (actionId) {
            case 'index':
                setShowIndexPanel(true)
                break
            case 'refactor':
                setShowRefactorPanel(true)
                break
            case 'test':
                setShowTestPanel(true)
                break
            case 'review':
                setShowReviewPanel(true)
                break
            case 'docs':
                setActiveTab('docs')
                break
        }
    }

    const lastUserQuery = messages.filter(m => m.role === 'user').pop()?.content

    // Calculate crew progress
    const totalAgents = crewState.agents.length
    const completedAgents = crewState.agents.filter(a => a.status === 'done').length

    // Handle index from hero page
    const handleHeroIndex = async (path: string) => {
        const result = await indexCodebase(path, true)
        if (result) {
            setMessages([{
                id: 'welcome',
                role: 'assistant',
                content: `**CodeMind** — Ready to analyze your code

Indexed **${result.chunks_indexed}** chunks from **${result.files_processed}** files in ${result.duration_seconds}s.

Ask me anything about your codebase, or use the quick actions in the sidebar for:
- **Refactor** — Multi-agent code improvement
- **Test** — Generate test suites
- **Review** — Security & performance analysis`,
                timestamp: new Date()
            }])
        }
    }

    // Handle GitHub clone from hero page
    const handleCloneAndIndex = async (githubUrl: string) => {
        const result = await cloneAndIndex(githubUrl, true)
        if (result) {
            setMessages([{
                id: 'welcome',
                role: 'assistant',
                content: `**CodeMind** — Ready to analyze your code

Cloned and indexed **${result.repo}**
- **${result.chunks_indexed}** chunks from **${result.files_processed}** files
- Completed in ${result.duration_seconds}s

Ask me anything about this codebase, or use the quick actions in the sidebar for:
- **Refactor** — Multi-agent code improvement
- **Test** — Generate test suites
- **Review** — Security & performance analysis`,
                timestamp: new Date()
            }])
        }
    }

    // Show hero page if not indexed
    if (!status?.indexed) {
        return <HeroPage onIndex={handleHeroIndex} onCloneAndIndex={handleCloneAndIndex} isIndexing={isIndexing} error={error} />
    }

    return (
        <div className="flex h-screen flex-col bg-background">
            {/* Header */}
            <header className="flex h-14 items-center justify-between border-b border-border bg-card/50 px-4">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                            <Code2 className="h-5 w-5 text-primary-foreground" />
                        </div>
                        <span className="text-lg font-semibold text-foreground">CodeMind</span>
                    </div>
                    <div className="hidden items-center gap-2 md:flex">
                        <span className={cn(
                            "h-1.5 w-1.5 rounded-full",
                            isConnected ? "bg-success" : "bg-destructive"
                        )} />
                        <span className="text-sm text-muted-foreground">
                            {isConnected ? 'Connected' : 'Disconnected'}
                        </span>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    {currentIntent && (
                        <Badge variant="outline" className="font-mono text-xs">
                            {currentIntent}
                        </Badge>
                    )}
                    {status?.indexed && (
                        <Badge variant="secondary" className="hidden font-mono text-xs sm:inline-flex">
                            {status.chunks} chunks
                        </Badge>
                    )}
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={async () => {
                            await clearIndex()
                            setMessages([])
                        }}
                        className="h-8 gap-1.5 text-muted-foreground hover:text-foreground"
                        title="Back to Home"
                    >
                        <Home className="h-4 w-4" />
                    </Button>
                </div>
            </header>

            <div className="flex flex-1 overflow-hidden">
                {/* Sidebar */}
                <aside
                    className={cn(
                        "flex flex-col border-r border-border bg-sidebar transition-all duration-300",
                        sidebarOpen ? "w-64" : "w-0"
                    )}
                >
                    {sidebarOpen && (
                        <>
                            {/* Quick Actions */}
                            <div className="border-b border-sidebar-border p-3">
                                <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                    Quick Actions
                                </p>
                                <div className="flex flex-wrap gap-1.5">
                                    {quickActions.map((action) => {
                                        const Icon = action.icon
                                        return (
                                            <Button
                                                key={action.id}
                                                variant="secondary"
                                                size="sm"
                                                className="h-8 gap-1.5 text-xs"
                                                onClick={() => handleQuickAction(action.id)}
                                            >
                                                <Icon className="h-4 w-4" />
                                                {action.label}
                                            </Button>
                                        )
                                    })}
                                </div>
                            </div>

                            {/* Index Panel */}
                            {showIndexPanel && (
                                <div className="border-b border-sidebar-border p-3 space-y-2">
                                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                        Index Codebase
                                    </p>
                                    <input
                                        type="text"
                                        value={indexPath}
                                        onChange={(e) => setIndexPath(e.target.value)}
                                        placeholder="Path to codebase"
                                        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                    />
                                    <div className="flex gap-2">
                                        <Button
                                            size="sm"
                                            onClick={handleIndex}
                                            disabled={isIndexing}
                                            className="flex-1"
                                        >
                                            {isIndexing ? (
                                                <>
                                                    <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                                    Indexing
                                                </>
                                            ) : (
                                                <>
                                                    <FolderOpen className="mr-1 h-3 w-3" />
                                                    Index
                                                </>
                                            )}
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="ghost"
                                            onClick={() => setShowIndexPanel(false)}
                                        >
                                            <X className="h-3 w-3" />
                                        </Button>
                                    </div>
                                    {error && (
                                        <p className="text-xs text-destructive">{error}</p>
                                    )}
                                </div>
                            )}

                            {/* File Explorer */}
                            {status?.indexed && (
                                <div className="flex-1 overflow-hidden">
                                    <div className="flex items-center justify-between px-3 py-2">
                                        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                            Explorer
                                        </p>
                                    </div>
                                    <ScrollArea className="h-[calc(100vh-280px)]">
                                        <FileExplorer
                                            isIndexed={status?.indexed || false}
                                            indexedDirectory={status?.directory}
                                            onSelectFile={(path) => setInsertPath(path)}
                                            className="px-1"
                                        />
                                    </ScrollArea>
                                </div>
                            )}
                        </>
                    )}
                </aside>

                {/* Sidebar Toggle */}
                <Button
                    variant="ghost"
                    size="sm"
                    className="absolute left-0 top-16 z-10 h-8 w-8 rounded-l-none rounded-r-md border border-l-0 border-border bg-card p-0 transition-all hover:bg-secondary"
                    style={{ left: sidebarOpen ? '256px' : '0' }}
                    onClick={() => setSidebarOpen(!sidebarOpen)}
                >
                    {sidebarOpen ? (
                        <PanelLeftClose className="h-4 w-4" />
                    ) : (
                        <PanelLeft className="h-4 w-4" />
                    )}
                </Button>

                {/* Main Content */}
                <main className="flex-1 flex flex-col min-h-0">
                    {/* Tab Navigation */}
                    <div className="flex-shrink-0 border-b border-border bg-card/30 px-4">
                        <div className="flex h-12 items-center gap-1">
                            <button
                                onClick={() => setActiveTab('chat')}
                                className={cn(
                                    "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors",
                                    activeTab === 'chat'
                                        ? "bg-secondary text-foreground"
                                        : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                                )}
                            >
                                <MessageSquare className="h-4 w-4" />
                                Chat
                            </button>
                            <button
                                onClick={() => setActiveTab('metrics')}
                                className={cn(
                                    "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors",
                                    activeTab === 'metrics'
                                        ? "bg-secondary text-foreground"
                                        : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                                )}
                            >
                                <BarChart3 className="h-4 w-4" />
                                Metrics
                            </button>
                            <button
                                onClick={() => setActiveTab('docs')}
                                className={cn(
                                    "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors",
                                    activeTab === 'docs'
                                        ? "bg-secondary text-foreground"
                                        : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                                )}
                            >
                                <BookOpen className="h-4 w-4" />
                                Documentation
                            </button>
                        </div>
                    </div>

                    {/* Tab Content */}
                    <div className="flex-1 min-h-0 overflow-hidden">
                        {activeTab === 'chat' && (
                            <div className="h-full flex flex-col">
                                {/* Messages */}
                                <div className="flex-1 overflow-y-auto p-6">
                                    <div className="max-w-3xl mx-auto space-y-6 pb-4">
                                        {messages.map((message) => (
                                            <ChatMessage key={message.id} message={message} />
                                        ))}
                                        {routingSteps.length > 0 && (
                                            <RoutingFlow steps={routingSteps} query={lastUserQuery} />
                                        )}
                                        {isLoading && (
                                            <div className="flex gap-3">
                                                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
                                                    <Loader2 className="h-4 w-4 animate-spin" />
                                                </div>
                                                <div className="flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-3">
                                                    <span className="text-sm text-muted-foreground">Thinking...</span>
                                                </div>
                                            </div>
                                        )}
                                        <div ref={messagesEndRef} />
                                    </div>
                                </div>

                                {/* Input Area */}
                                <div className="flex-shrink-0 border-t border-border p-4">
                                    <form onSubmit={handleSendMessage} className="max-w-3xl mx-auto flex gap-3">
                                        <div className="relative flex-1">
                                            <Textarea
                                                value={inputValue}
                                                onChange={(e) => setInputValue(e.target.value)}
                                                placeholder="Ask about your codebase..."
                                                className="min-h-[80px] resize-none bg-secondary/50 pr-12 text-foreground placeholder:text-muted-foreground focus:bg-secondary/70"
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter' && !e.shiftKey) {
                                                        e.preventDefault()
                                                        handleSendMessage()
                                                    }
                                                }}
                                            />
                                        </div>
                                        <Button
                                            type="submit"
                                            disabled={!inputValue.trim() || isLoading || !isConnected}
                                            className="h-auto px-6"
                                        >
                                            {isLoading ? (
                                                <Loader2 className="h-4 w-4 animate-spin" />
                                            ) : (
                                                <Send className="h-4 w-4" />
                                            )}
                                        </Button>
                                    </form>
                                    <p className="max-w-3xl mx-auto mt-2 text-xs text-muted-foreground">
                                        Press Enter to send, Shift+Enter for new line
                                    </p>
                                </div>
                            </div>
                        )}

                        {activeTab === 'metrics' && (
                            <div className="h-full overflow-y-auto">
                                <MetricsPage />
                            </div>
                        )}

                        {activeTab === 'docs' && (
                            <div className="h-full overflow-y-auto">
                                <DocsPage />
                            </div>
                        )}
                    </div>
                </main>
            </div>

            {/* Agent Progress Sheet */}
            <Sheet open={agentProgressOpen} onOpenChange={setAgentProgressOpen}>
                <SheetContent className="w-full sm:max-w-lg bg-background border-border">
                    <SheetHeader className="space-y-3">
                        <div className="flex items-center justify-between">
                            <SheetTitle className="text-foreground">Agent Pipeline</SheetTitle>
                            <Badge variant="outline" className="font-mono text-xs">
                                {completedAgents}/{totalAgents} Complete
                            </Badge>
                        </div>
                        <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                            <div 
                                className="h-full bg-primary transition-all duration-300" 
                                style={{ width: totalAgents ? `${(completedAgents / totalAgents) * 100}%` : '0%' }}
                            />
                        </div>
                    </SheetHeader>
                    <div className="mt-6 space-y-3">
                        {crewState.agents.map((agent, idx) => (
                            <AgentCard key={idx} agent={agent} />
                        ))}
                    </div>
                    <div className="mt-6 flex gap-3">
                        <Button
                            variant="outline"
                            className="flex-1 bg-transparent"
                            onClick={() => {
                                cancelRefactoring()
                                setAgentProgressOpen(false)
                            }}
                        >
                            <X className="mr-2 h-4 w-4" />
                            Cancel
                        </Button>
                        {!crewState.isRunning && crewState.agentOutputs.length > 0 && (
                            <Button 
                                className="flex-1"
                                onClick={() => {
                                    // Save to chat
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
                                    setAgentProgressOpen(false)
                                    reset()
                                }}
                            >
                                Save to Chat
                            </Button>
                        )}
                    </div>
                </SheetContent>
            </Sheet>

            {/* Crew Panels */}
            {showRefactorPanel && (
                <RefactorPanel
                    onStart={(target, focus) => {
                        setShowRefactorPanel(false)
                        setAgentProgressOpen(true)
                        startRefactoring(target, focus)
                    }}
                    onClose={() => setShowRefactorPanel(false)}
                    isIndexed={status?.indexed || false}
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

            {/* Test Crew Result Modal */}
            {(testState.isRunning || testState.result || testState.error) && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
                    <div className="w-full max-w-2xl rounded-lg border border-border bg-background p-6 shadow-lg">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold">TestingCrew</h3>
                            {!testState.isRunning && (
                                <Button variant="ghost" size="sm" onClick={() => {
                                    if (testState.result) {
                                        setMessages(prev => [...prev, {
                                            id: `test-${Date.now()}`,
                                            role: 'assistant',
                                            content: `**TestingCrew Result**\n\n${testState.result.slice(0, 3000)}${testState.result.length > 3000 ? '...' : ''}`,
                                            timestamp: new Date()
                                        }])
                                    }
                                    resetTest()
                                }}>
                                    <X className="h-4 w-4" />
                                </Button>
                            )}
                        </div>
                        {testState.isRunning ? (
                            <div className="flex items-center gap-3 py-8 justify-center">
                                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                                <span className="text-muted-foreground">Running tests...</span>
                            </div>
                        ) : testState.error ? (
                            <p className="text-destructive">{testState.error}</p>
                        ) : (
                            <ScrollArea className="h-[400px]">
                                <pre className="text-sm whitespace-pre-wrap">{testState.result}</pre>
                            </ScrollArea>
                        )}
                        <div className="mt-4 flex justify-end gap-2">
                            {testState.isRunning && (
                                <Button variant="outline" onClick={cancelTest}>Cancel</Button>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Review Crew Result Modal */}
            {(reviewState.isRunning || reviewState.result || reviewState.error) && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
                    <div className="w-full max-w-2xl rounded-lg border border-border bg-background p-6 shadow-lg">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold">CodeReviewCrew</h3>
                            {!reviewState.isRunning && (
                                <Button variant="ghost" size="sm" onClick={() => {
                                    if (reviewState.result) {
                                        setMessages(prev => [...prev, {
                                            id: `review-${Date.now()}`,
                                            role: 'assistant',
                                            content: `**CodeReviewCrew Result**\n\n${reviewState.result.slice(0, 3000)}${reviewState.result.length > 3000 ? '...' : ''}`,
                                            timestamp: new Date()
                                        }])
                                    }
                                    resetReview()
                                }}>
                                    <X className="h-4 w-4" />
                                </Button>
                            )}
                        </div>
                        {reviewState.isRunning ? (
                            <div className="flex items-center gap-3 py-8 justify-center">
                                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                                <span className="text-muted-foreground">Reviewing code...</span>
                            </div>
                        ) : reviewState.error ? (
                            <p className="text-destructive">{reviewState.error}</p>
                        ) : (
                            <ScrollArea className="h-[400px]">
                                <pre className="text-sm whitespace-pre-wrap">{reviewState.result}</pre>
                            </ScrollArea>
                        )}
                        <div className="mt-4 flex justify-end gap-2">
                            {reviewState.isRunning && (
                                <Button variant="outline" onClick={cancelReview}>Cancel</Button>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
