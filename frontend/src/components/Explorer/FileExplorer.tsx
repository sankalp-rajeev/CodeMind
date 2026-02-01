import { useState, useEffect } from 'react'
import { ChevronRight, ChevronDown, FileCode, Folder, FolderOpen } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

interface TreeNode {
    name: string
    path: string
    type: 'file' | 'dir'
    children?: TreeNode[]
}

interface FileTreeResponse {
    root: string | null
    tree: TreeNode | null
    chunks?: number
    error?: string
}

interface FileExplorerProps {
    isIndexed: boolean
    indexedDirectory?: string | null
    onSelectFile: (path: string) => void
    className?: string
}

function TreeItem({
    node,
    level,
    onSelectFile,
    expandedDirs,
    toggleDir
}: {
    node: TreeNode
    level: number
    onSelectFile: (path: string) => void
    expandedDirs: Set<string>
    toggleDir: (path: string) => void
}) {
    const isDir = node.type === 'dir'
    const hasChildren = isDir && node.children && node.children.length > 0
    const isExpanded = expandedDirs.has(node.path)

    if (isDir && !hasChildren) return null

    return (
        <div className="select-none">
            {isDir ? (
                <>
                    <button
                        onClick={() => toggleDir(node.path)}
                        className="flex items-center gap-1.5 w-full px-2 py-1.5 rounded-lg text-left hover:bg-surface-tertiary transition-colors text-text-primary text-sm"
                        style={{ paddingLeft: `${level * 12 + 8}px` }}
                    >
                        {isExpanded ? (
                            <ChevronDown className="w-4 h-4 flex-shrink-0 text-text-muted" />
                        ) : (
                            <ChevronRight className="w-4 h-4 flex-shrink-0 text-text-muted" />
                        )}
                        {isExpanded ? (
                            <FolderOpen className="w-4 h-4 flex-shrink-0 text-amber-500" />
                        ) : (
                            <Folder className="w-4 h-4 flex-shrink-0 text-amber-500" />
                        )}
                        <span className="truncate">{node.name}</span>
                    </button>
                    {isExpanded && hasChildren && (
                        <div>
                            {node.children!.map((child) => (
                                <TreeItem
                                    key={child.path || child.name}
                                    node={child}
                                    level={level + 1}
                                    onSelectFile={onSelectFile}
                                    expandedDirs={expandedDirs}
                                    toggleDir={toggleDir}
                                />
                            ))}
                        </div>
                    )}
                </>
            ) : (
                <button
                    onClick={() => onSelectFile(node.path)}
                    className="flex items-center gap-1.5 w-full px-2 py-1.5 rounded-lg text-left hover:bg-brand-muted transition-colors text-text-secondary text-sm group"
                    style={{ paddingLeft: `${level * 12 + 8}px` }}
                >
                    <span className="w-4 flex-shrink-0" />
                    <FileCode className="w-4 h-4 flex-shrink-0 text-brand" />
                    <span className="truncate group-hover:text-brand">{node.name}</span>
                </button>
            )}
        </div>
    )
}

export default function FileExplorer({ isIndexed, indexedDirectory, onSelectFile, className = '' }: FileExplorerProps) {
    const [tree, setTree] = useState<TreeNode | null>(null)
    const [root, setRoot] = useState<string | null>(null)
    const [chunks, setChunks] = useState<number>(0)
    const [loading, setLoading] = useState(false)
    const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set())

    useEffect(() => {
        if (!isIndexed) {
            setTree(null)
            setRoot(null)
            return
        }
        setLoading(true)
        fetch(`${API_BASE}/api/file-tree`)
            .then((res) => res.json())
            .then((data: FileTreeResponse) => {
                setTree(data.tree || null)
                setRoot(data.root || null)
                setChunks(data.chunks || 0)
                if (data.tree?.path) {
                    setExpandedDirs((prev) => new Set(prev).add(data.tree!.path))
                }
            })
            .catch(() => setTree(null))
            .finally(() => setLoading(false))
    }, [isIndexed, indexedDirectory])

    const toggleDir = (path: string) => {
        setExpandedDirs((prev) => {
            const next = new Set(prev)
            if (next.has(path)) next.delete(path)
            else next.add(path)
            return next
        })
    }

    if (!isIndexed) {
        return (
            <div className={`p-4 text-center text-text-muted text-sm ${className}`}>
                <Folder className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>Index a codebase to browse files</p>
            </div>
        )
    }

    if (loading) {
        return (
            <div className={`p-4 text-center text-text-muted text-sm ${className}`}>
                <div className="animate-pulse">Loading...</div>
            </div>
        )
    }

    if (!tree) {
        return (
            <div className={`p-4 text-center text-text-muted text-sm ${className}`}>
                <p>No file tree available</p>
            </div>
        )
    }

    return (
        <div className={`flex flex-col ${className}`}>
            <div className="px-3 py-2.5 border-b border-border flex items-center justify-between bg-surface-secondary">
                <span className="text-xs font-medium text-text-secondary truncate" title={root || ''}>
                    {tree.name}
                </span>
                {chunks > 0 && (
                    <span className="text-xs text-brand font-medium">{chunks} chunks</span>
                )}
            </div>
            <div className="flex-1 overflow-y-auto py-2 px-1">
                <TreeItem
                    node={tree}
                    level={0}
                    onSelectFile={onSelectFile}
                    expandedDirs={expandedDirs}
                    toggleDir={toggleDir}
                />
            </div>
        </div>
    )
}
