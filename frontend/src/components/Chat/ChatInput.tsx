import { useState, KeyboardEvent, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'

interface ChatInputProps {
    onSend: (message: string) => void
    isLoading: boolean
    disabled?: boolean
}

export default function ChatInput({ onSend, isLoading, disabled }: ChatInputProps) {
    const [input, setInput] = useState('')
    const textareaRef = useRef<HTMLTextAreaElement>(null)

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto'
            textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 150) + 'px'
        }
    }, [input])

    const handleSend = () => {
        if (input.trim() && !isLoading && !disabled) {
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

    const placeholders = [
        "What does the training function do?",
        "Explain the data preprocessing pipeline",
        "Find functions that handle user input",
        "How is the model evaluated?",
    ]
    const [placeholder] = useState(placeholders[Math.floor(Math.random() * placeholders.length)])

    return (
        <div className="flex gap-3 items-end">
            <div className="flex-1 relative">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    disabled={isLoading || disabled}
                    rows={1}
                    className="w-full resize-none rounded-xl bg-slate-800/80 border border-slate-600/50 px-4 py-3 pr-12 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 disabled:opacity-50 transition-all"
                    style={{ minHeight: '48px' }}
                />
                {input.length > 0 && (
                    <span className="absolute right-3 bottom-3 text-xs text-slate-500">
                        {input.length}/2000
                    </span>
                )}
            </div>
            <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading || disabled}
                className="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed flex items-center justify-center transition-all shadow-lg shadow-blue-500/20 disabled:shadow-none"
            >
                {isLoading ? (
                    <Loader2 className="w-5 h-5 text-white animate-spin" />
                ) : (
                    <Send className="w-5 h-5 text-white" />
                )}
            </button>
        </div>
    )
}
