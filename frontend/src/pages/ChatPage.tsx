import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2 } from 'lucide-react'
import ChatMessage from '../components/Chat/ChatMessage'
import ChatInput from '../components/Chat/ChatInput'

interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: Date
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            role: 'assistant',
            content: 'Hello! I\'m CodeMind AI. I can help you understand, refactor, and improve your codebase. Start by indexing a repository, then ask me anything about your code!',
            timestamp: new Date()
        }
    ])
    const [isLoading, setIsLoading] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const handleSendMessage = async (content: string) => {
        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content,
            timestamp: new Date()
        }

        setMessages(prev => [...prev, userMessage])
        setIsLoading(true)

        // TODO: Connect to WebSocket backend
        // For now, simulate a response
        setTimeout(() => {
            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: `I received your message: "${content}"\n\nThis is a placeholder response. Connect the WebSocket backend to get real AI responses!`,
                timestamp: new Date()
            }
            setMessages(prev => [...prev, assistantMessage])
            setIsLoading(false)
        }, 1000)
    }

    return (
        <div className="flex flex-col h-screen">
            {/* Header */}
            <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
                <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                        <Bot className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-semibold text-white">CodeMind AI</h1>
                        <p className="text-sm text-slate-400">Multi-agent codebase assistant</p>
                    </div>
                </div>
            </header>

            {/* Messages */}
            <main className="flex-1 overflow-y-auto">
                <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
                    {messages.map((message) => (
                        <ChatMessage key={message.id} message={message} />
                    ))}

                    {isLoading && (
                        <div className="flex items-center gap-3 text-slate-400">
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Thinking...</span>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>
            </main>

            {/* Input */}
            <footer className="border-t border-slate-700 bg-slate-900/50 backdrop-blur-sm">
                <div className="max-w-4xl mx-auto px-4 py-4">
                    <ChatInput onSend={handleSendMessage} isLoading={isLoading} />
                </div>
            </footer>
        </div>
    )
}
