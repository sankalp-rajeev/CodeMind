import { Bot, User } from 'lucide-react'

interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: Date
}

interface ChatMessageProps {
    message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === 'user'

    return (
        <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
            {/* Avatar */}
            <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${isUser
                    ? 'bg-gradient-to-br from-emerald-500 to-teal-600'
                    : 'bg-gradient-to-br from-blue-500 to-purple-600'
                }`}>
                {isUser ? (
                    <User className="w-5 h-5 text-white" />
                ) : (
                    <Bot className="w-5 h-5 text-white" />
                )}
            </div>

            {/* Message Content */}
            <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
                <div className={`inline-block rounded-2xl px-4 py-3 ${isUser
                        ? 'bg-blue-600 text-white rounded-tr-sm'
                        : 'bg-slate-700/50 text-slate-100 rounded-tl-sm'
                    }`}>
                    <p className="whitespace-pre-wrap text-left">{message.content}</p>
                </div>
                <p className="text-xs text-slate-500 mt-1">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
            </div>
        </div>
    )
}
