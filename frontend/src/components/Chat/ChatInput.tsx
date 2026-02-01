import { useState, KeyboardEvent, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'

interface ChatInputProps {
    onSend: (message: string) => void
    isLoading: boolean
    disabled?: boolean
    insertText?: string | null
    onInsertConsumed?: () => void
}

export default function ChatInput({ onSend, isLoading, disabled, insertText, onInsertConsumed }: ChatInputProps) {
    const [input, setInput] = useState('')
    const textareaRef = useRef<HTMLTextAreaElement>(null)

    useEffect(() => {
        if (insertText && insertText.trim()) {
            setInput((prev) => (prev.trim() ? prev + ' ' + insertText : insertText))
            textareaRef.current?.focus()
            onInsertConsumed?.()
        }
    }, [insertText])

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
                    className="w-full resize-none rounded-xl bg-surface-secondary border border-border px-4 py-3 pr-14 text-text-primary placeholder-text-muted text-[15px] focus:outline-none focus:ring-2 focus:ring-brand/20 focus:border-brand disabled:opacity-50 transition-all"
                    style={{ minHeight: '48px' }}
                />
                {input.length > 0 && (
                    <span className="absolute right-3 bottom-3 text-xs text-text-muted font-mono">
                        {input.length}/2000
                    </span>
                )}
            </div>
            <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading || disabled}
                className="flex-shrink-0 w-12 h-12 rounded-xl bg-brand hover:bg-brand-hover disabled:bg-surface-tertiary disabled:text-text-muted flex items-center justify-center transition-colors disabled:cursor-not-allowed shadow-sm"
            >
                {isLoading ? (
                    <Loader2 className="w-5 h-5 text-white animate-spin" />
                ) : (
                    <Send className="w-5 h-5 text-white" strokeWidth={2} />
                )}
            </button>
        </div>
    )
}
