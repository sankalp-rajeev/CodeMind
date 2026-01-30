import { Bot, User, Code, FileCode } from 'lucide-react'
import { Message } from '../../hooks/useWebSocket'

interface ChatMessageProps {
    message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === 'user'

    // Simple markdown-like rendering for code blocks
    const renderContent = (content: string) => {
        // Split by code blocks
        const parts = content.split(/(```[\s\S]*?```)/g)

        return parts.map((part, index) => {
            if (part.startsWith('```')) {
                // Extract language and code
                const match = part.match(/```(\w+)?\n?([\s\S]*?)```/)
                if (match) {
                    const [, lang, code] = match
                    return (
                        <div key={index} className="my-3">
                            <div className="flex items-center gap-2 px-3 py-1 bg-slate-800 rounded-t-lg border border-slate-700 border-b-0">
                                <FileCode className="w-4 h-4 text-slate-400" />
                                <span className="text-xs text-slate-400">{lang || 'code'}</span>
                            </div>
                            <pre className="p-3 bg-slate-900 rounded-b-lg border border-slate-700 overflow-x-auto">
                                <code className="text-sm text-slate-300 font-mono">{code.trim()}</code>
                            </pre>
                        </div>
                    )
                }
            }

            // Regular text - handle bold and inline code
            return (
                <span key={index} className="whitespace-pre-wrap">
                    {part.split(/(\*\*.*?\*\*)/g).map((segment, i) => {
                        if (segment.startsWith('**') && segment.endsWith('**')) {
                            return <strong key={i} className="font-semibold text-white">{segment.slice(2, -2)}</strong>
                        }
                        // Handle inline code
                        return segment.split(/(`[^`]+`)/g).map((s, j) => {
                            if (s.startsWith('`') && s.endsWith('`')) {
                                return (
                                    <code key={`${i}-${j}`} className="px-1.5 py-0.5 bg-slate-700/50 text-blue-300 rounded text-sm font-mono">
                                        {s.slice(1, -1)}
                                    </code>
                                )
                            }
                            return s
                        })
                    })}
                </span>
            )
        })
    }

    return (
        <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
            {/* Avatar */}
            <div className={`flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center shadow-lg ${isUser
                    ? 'bg-gradient-to-br from-emerald-500 to-teal-600 shadow-emerald-500/20'
                    : 'bg-gradient-to-br from-blue-500 to-purple-600 shadow-blue-500/20'
                }`}>
                {isUser ? (
                    <User className="w-5 h-5 text-white" />
                ) : (
                    <Bot className="w-5 h-5 text-white" />
                )}
            </div>

            {/* Message Content */}
            <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : ''}`}>
                <div className={`inline-block rounded-2xl px-4 py-3 ${isUser
                        ? 'bg-blue-600 text-white rounded-tr-sm'
                        : 'bg-slate-700/50 text-slate-100 rounded-tl-sm border border-slate-600/30'
                    }`}>
                    <div className="text-left text-[15px] leading-relaxed">
                        {renderContent(message.content)}
                    </div>
                </div>
                <p className="text-xs text-slate-500 mt-1.5">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    {message.intent && (
                        <span className="ml-2 text-blue-400">• {message.intent}</span>
                    )}
                </p>
            </div>
        </div>
    )
}
