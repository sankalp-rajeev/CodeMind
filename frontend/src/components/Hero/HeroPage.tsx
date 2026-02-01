import { useState } from 'react'
import { 
    Code2, 
    Search, 
    Shield, 
    Zap, 
    TestTube, 
    FileText, 
    ArrowRight,
    Loader2,
    FolderOpen,
    Brain,
    GitBranch,
    Sparkles,
    Github
} from 'lucide-react'

interface HeroPageProps {
    onIndex: (path: string) => Promise<void>
    onCloneAndIndex: (githubUrl: string) => Promise<void>
    isIndexing: boolean
    error: string | null
}

const features = [
    {
        icon: Search,
        title: 'Explore',
        description: 'Understand code structure and dependencies with semantic search'
    },
    {
        icon: Shield,
        title: 'Security',
        description: 'Identify vulnerabilities and get actionable fixes'
    },
    {
        icon: Zap,
        title: 'Performance',
        description: 'Optimize algorithms and improve code efficiency'
    },
    {
        icon: TestTube,
        title: 'Testing',
        description: 'Generate comprehensive test suites automatically'
    },
    {
        icon: FileText,
        title: 'Documentation',
        description: 'Create clear, maintainable documentation'
    },
    {
        icon: GitBranch,
        title: 'Refactor',
        description: 'Multi-agent analysis for code improvement'
    },
]

export default function HeroPage({ onIndex, onCloneAndIndex, isIndexing, error }: HeroPageProps) {
    const [inputValue, setInputValue] = useState('')
    const [inputType, setInputType] = useState<'local' | 'github'>('github')

    // Auto-detect if input is a GitHub URL
    const isGitHubUrl = (value: string) => {
        return value.includes('github.com/') || value.match(/^[\w-]+\/[\w-]+$/)
    }

    const handleInputChange = (value: string) => {
        setInputValue(value)
        // Auto-detect input type
        if (isGitHubUrl(value)) {
            setInputType('github')
        } else if (value.startsWith('./') || value.startsWith('/') || value.includes(':\\')) {
            setInputType('local')
        }
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!inputValue.trim()) return

        if (inputType === 'github' || isGitHubUrl(inputValue)) {
            // Convert shorthand (owner/repo) to full URL
            let url = inputValue.trim()
            if (url.match(/^[\w-]+\/[\w-]+$/)) {
                url = `https://github.com/${url}`
            }
            await onCloneAndIndex(url)
        } else {
            await onIndex(inputValue.trim())
        }
    }

    return (
        <div className="min-h-screen bg-background flex flex-col">
            {/* Header */}
            <header className="border-b border-border/50 bg-background/80 backdrop-blur-sm sticky top-0 z-10">
                <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary shadow-lg shadow-primary/20">
                            <Code2 className="h-6 w-6 text-primary-foreground" />
                        </div>
                        <span className="text-xl font-bold text-foreground">CodeMind</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Brain className="h-4 w-4" />
                        <span>AI-Powered Code Intelligence</span>
                    </div>
                </div>
            </header>

            {/* Hero Section */}
            <main className="flex-1 flex flex-col items-center px-6 py-8">
                <div className="max-w-4xl mx-auto text-center space-y-6">
                    {/* Logo & Title */}
                    <div className="space-y-3">
                        <div className="flex justify-center">
                            <div className="relative">
                                <div className="h-20 w-20 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shadow-2xl shadow-primary/30">
                                    <Code2 className="h-10 w-10 text-primary-foreground" />
                                </div>
                                <div className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-success flex items-center justify-center">
                                    <Sparkles className="h-2.5 w-2.5 text-success-foreground" />
                                </div>
                            </div>
                        </div>
                        <h1 className="text-4xl font-bold text-foreground tracking-tight">
                            CodeMind
                        </h1>
                        <p className="text-lg text-muted-foreground max-w-xl mx-auto">
                            AI-powered code intelligence. Explore, analyze, and improve 
                            your codebase with multi-agent workflows.
                        </p>
                    </div>

                    {/* Index Form */}
                    <div className="max-w-md mx-auto w-full">
                        {/* Input Type Toggle */}
                        <div className="flex justify-center gap-2 mb-4">
                            <button
                                type="button"
                                onClick={() => {
                                    setInputType('github')
                                    setInputValue('')
                                }}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                    inputType === 'github'
                                        ? 'bg-primary text-primary-foreground'
                                        : 'bg-secondary text-muted-foreground hover:text-foreground'
                                }`}
                            >
                                <Github className="h-4 w-4" />
                                GitHub
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setInputType('local')
                                    setInputValue('./data/test-repo')
                                }}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                    inputType === 'local'
                                        ? 'bg-primary text-primary-foreground'
                                        : 'bg-secondary text-muted-foreground hover:text-foreground'
                                }`}
                            >
                                <FolderOpen className="h-4 w-4" />
                                Local Path
                            </button>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-3">
                            <div className="relative">
                                {inputType === 'github' ? (
                                    <Github className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                                ) : (
                                    <FolderOpen className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                                )}
                                <input
                                    type="text"
                                    value={inputValue}
                                    onChange={(e) => handleInputChange(e.target.value)}
                                    placeholder={inputType === 'github' 
                                        ? "https://github.com/owner/repo" 
                                        : "Enter path to your codebase..."
                                    }
                                    className="w-full h-12 pl-12 pr-4 rounded-lg border border-border bg-card text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                                    disabled={isIndexing}
                                />
                            </div>
                            <button
                                type="submit"
                                disabled={!inputValue.trim() || isIndexing}
                                className="w-full h-12 rounded-lg bg-primary text-primary-foreground font-semibold flex items-center justify-center gap-2 hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-primary/20"
                            >
                                {isIndexing ? (
                                    <>
                                        <Loader2 className="h-5 w-5 animate-spin" />
                                        {inputType === 'github' ? 'Cloning & Indexing...' : 'Indexing...'}
                                    </>
                                ) : (
                                    <>
                                        Get Started
                                        <ArrowRight className="h-5 w-5" />
                                    </>
                                )}
                            </button>
                        </form>
                        {error && (
                            <p className="mt-2 text-sm text-destructive text-center">{error}</p>
                        )}
                    </div>
                </div>

                {/* Features Grid */}
                <div className="max-w-4xl mx-auto mt-10 w-full">
                    <h2 className="text-center text-xs font-medium uppercase tracking-wider text-muted-foreground mb-5">
                        Capabilities
                    </h2>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {features.map((feature) => {
                            const Icon = feature.icon
                            return (
                                <div
                                    key={feature.title}
                                    className="p-4 rounded-lg border border-border bg-card/50 hover:bg-card hover:border-primary/30 transition-all group"
                                >
                                    <div className="flex items-start gap-3">
                                        <div className="h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors flex-shrink-0">
                                            <Icon className="h-4 w-4 text-primary" />
                                        </div>
                                        <div>
                                            <h3 className="font-medium text-sm text-foreground">{feature.title}</h3>
                                            <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">
                                                {feature.description}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-border/50 py-6">
                <div className="max-w-6xl mx-auto px-6 flex items-center justify-between text-sm text-muted-foreground">
                    <p>Built with CrewAI, RAG, and Local LLMs</p>
                    <p>University of Michigan</p>
                </div>
            </footer>
        </div>
    )
}
