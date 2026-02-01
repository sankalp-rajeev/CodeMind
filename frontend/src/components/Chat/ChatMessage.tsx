import { Bot, User, FileCode } from 'lucide-react'
import { Message } from '../../hooks/useWebSocket'

interface ChatMessageProps {
    message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === 'user'

    const renderContent = (content: string) => {
        const parts = content.split(/(```[\s\S]*?```)/g)

        return parts.map((part, index) => {
            if (part.startsWith('```')) {
                const match = part.match(/```(\w+)?\n?([\s\S]*?)```/)
                if (match) {
                    const [, lang, code] = match
                    return (
                        <div key={index} className="my-4">
                            <div className="flex items-center gap-2 px-3 py-2 bg-surface-tertiary rounded-t-lg border border-border border-b-0">
                                <FileCode className="w-4 h-4 text-text-muted" />
                                <span className="text-xs text-text-muted font-mono">{lang || 'code'}</span>
                            </div>
                            <pre className="p-4 bg-surface-tertiary rounded-b-lg border border-border overflow-x-auto text-text-primary text-[13px] leading-relaxed">
                                <code>{code.trim()}</code>
                            </pre>
                        </div>
                    )
                }
            }

            return (
                <span key={index} className="whitespace-pre-wrap">
                    {part.split(/(\*\*.*?\*\*)/g).map((segment, i) => {
                        if (segment.startsWith('**') && segment.endsWith('**')) {
                            return <strong key={i} className="font-semibold text-text-primary">{segment.slice(2, -2)}</strong>
                        }
                        return segment.split(/(`[^`]+`)/g).map((s, j) => {
                            if (s.startsWith('`') && s.endsWith('`')) {
                                return (
                                    <code key={`${i}-${j}`} className="px-1.5 py-0.5 bg-brand-muted text-brand rounded text-[13px] font-mono">
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
        <div className={`flex gap-4 animate-fadeIn ${isUser ? 'flex-row-reverse' : ''}`}>
            <div className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center ${isUser
                ? 'bg-brand'
                : 'bg-surface-tertiary border border-border'
            }`}>
                {isUser ? (
                    <User className="w-4 h-4 text-white" strokeWidth={2} />
                ) : (
                    <Bot className="w-4 h-4 text-text-secondary" strokeWidth={2} />
                )}
            </div>

            <div className={`flex-1 min-w-0 ${isUser ? 'text-right' : ''}`}>
                <div className={`inline-block max-w-full rounded-2xl px-4 py-3 ${isUser
                    ? 'bg-brand text-white rounded-tr-md'
                    : 'bg-surface border border-border rounded-tl-md shadow-sm'
                }`}>
                    <div className={`text-left text-[15px] leading-relaxed ${isUser ? 'text-white' : 'text-text-secondary'}`}>
                        {renderContent(message.content)}
                    </div>
                </div>
                <p className="text-xs text-text-muted mt-1.5">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    {message.intent && (
                        <span className="ml-2 text-brand">· {message.intent}</span>
                    )}
                </p>
            </div>
        </div>
    )
}
