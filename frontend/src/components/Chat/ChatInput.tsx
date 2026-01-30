import { useState, KeyboardEvent } from 'react'
import { Send } from 'lucide-react'

interface ChatInputProps {
    onSend: (message: string) => void
    isLoading: boolean
}

export default function ChatInput({ onSend, isLoading }: ChatInputProps) {
    const [input, setInput] = useState('')

    const handleSend = () => {
        if (input.trim() && !isLoading) {
            onSend(input.trim())
            setInput('')
        }
    }

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="flex gap-3">
            <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your codebase..."
                disabled={isLoading}
                rows={1}
                className="flex-1 resize-none rounded-xl bg-slate-800 border border-slate-600 px-4 py-3 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 transition-all"
                style={{ minHeight: '48px', maxHeight: '200px' }}
            />
            <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="flex-shrink-0 w-12 h-12 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
            >
                <Send className="w-5 h-5 text-white" />
            </button>
        </div>
    )
}
