"use client";

import { useEffect, useState, useRef } from 'react';
import { useAuth } from '@/lib/auth-context';
import { getProjects, sendMessage, getRAGConfig, getDocuments, getSessions, deleteSession, getHistory, Project, RAGConfig, Document } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Send, Bot, User as UserIcon, FileText, Settings, Folder, ChevronRight, Menu, MessageSquare, Trash2, Plus, Info, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription } from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface Source {
    source: string;
    doc_id: number;
}

interface Message {
    role: 'user' | 'assistant';
    content: string;
    sources?: Source[] | string;
    usage_metadata?: any;
}

export default function ChatPage() {
    const { user, logout, isLoading } = useAuth();
    const router = useRouter();
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);
    const [sessionId, setSessionId] = useState<number | null>(null);
    const [projectConfig, setProjectConfig] = useState<RAGConfig | null>(null);
    const [documents, setDocuments] = useState<Document[]>([]);

    // Project & Settings State
    const [projects, setProjects] = useState<Project[]>([]);
    const [selectedProject, setSelectedProject] = useState<Project | null>(null);

    // Model Config State - Default to Google (Gemini) for better tool support
    const [modelProvider, setModelProvider] = useState<string>('groq');
    const [modelName, setModelName] = useState<string>('llama-3.3-70b-versatile');
    const [temperature, setTemperature] = useState<number[]>([0.1]);
    const [historyLimit, setHistoryLimit] = useState<number[]>([5]);
    // const [projectContextLimit, setProjectContextLimit] = useState<number[]>([2]); // Replaced by manual selection
    const [selectedContextSessions, setSelectedContextSessions] = useState<number[]>([]);

    // Sessions State
    const [sessions, setSessions] = useState<{ id: number, title: string, created_at: string, settings?: any }[]>([]);

    // UI State
    useEffect(() => {
        if (!isLoading && !user) router.push('/login');
    }, [user, isLoading, router]);


    useEffect(() => {
        if (scrollRef.current) {
            // Scroll the container to bottom
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isTyping]);

    useEffect(() => {
        if (user) loadProjects();
    }, [user]);

    useEffect(() => {
        if (selectedProject) {
            loadConfig(selectedProject.id);
            getDocuments(selectedProject.id).then(setDocuments).catch(() => { });
            loadSessions(selectedProject.id); // Load sessions for this project
        } else {
            setSessions([]);
        }
    }, [selectedProject]);

    const loadProjects = async () => {
        try {
            const data = await getProjects();
            setProjects(data);
            if (data.length > 0 && !selectedProject) setSelectedProject(data[0]);
        } catch (e) { toast.error("Failed to load projects"); }
    };

    const loadSessions = async (pid: number) => {
        try {
            console.log("Loading sessions for project:", pid);
            const data = await getSessions(pid);
            console.log("Sessions loaded:", data);
            setSessions(data);
            // toast.info(`Loaded ${data.length} sessions for Project ${pid}`);
        } catch (e) {
            console.error("Failed to load sessions:", e);
            toast.error("Failed to load sessions: " + String(e));
        }
    };

    const loadConfig = async (pid: number) => {
        try {
            const data = await getRAGConfig(pid);
            setProjectConfig(data);
        } catch (e) { }
    };

    const [isNewChatOpen, setIsNewChatOpen] = useState(false);
    const [newChatTitle, setNewChatTitle] = useState('');
    const [pendingChatTitle, setPendingChatTitle] = useState<string | null>(null); // Title to use for next created session

    const handleNewChat = () => {
        setSessionId(null);
        setMessages([]);
        setInput('');
        setPendingChatTitle(null);
    }

    const handleCreateNamedChat = () => {
        if (!newChatTitle.trim()) return;
        handleNewChat();
        setPendingChatTitle(newChatTitle);
        setIsNewChatOpen(false);
        setNewChatTitle('');
        toast.info(`Starting new section: "${newChatTitle}". Send a message to save.`);
    }

    const handleDeleteSession = async (sid: number, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm("Delete this chat permanently?")) return;
        try {
            await deleteSession(sid);
            setSessions(prev => prev.filter(s => s.id !== sid));
            if (sessionId === sid) {
                handleNewChat();
            }
            toast.success("Chat deleted");
        } catch (e) { toast.error("Failed to delete chat"); }
    }

    const handleLoadSession = async (sid: number) => {
        if (sessionId === sid) return;
        try {
            const msgs = await getHistory(sid);
            // Convert to UI format
            const uiMsgs: Message[] = msgs.map((m: any) => ({
                role: m.role,
                content: m.content,
                usage_metadata: m.usage_metadata
            }));
            setMessages(uiMsgs);
            setSessionId(sid);

            // Restore Settings from Session if available
            const session = sessions.find(s => s.id === sid);
            if (session?.settings) {
                // Handle both dict and stringified JSON
                let settings = session.settings;
                if (typeof settings === 'string') {
                    try { settings = JSON.parse(settings); } catch { }
                }

                if (settings.model_provider) setModelProvider(settings.model_provider);
                if (settings.model_name) setModelName(settings.model_name);
                if (settings.temperature) setTemperature([settings.temperature]);
                if (settings.history_limit) setHistoryLimit([settings.history_limit]);
                // if (settings.project_context_limit) setProjectContextLimit([settings.project_context_limit]);
            }

        } catch (e) { toast.error("Failed to load chat"); }
    }

    const handleSend = async () => {
        if (!input.trim() || !selectedProject) return;

        const userMsg: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsTyping(true);

        try {
            // New Session created with Project ID if sessionId is null
            const res = await sendMessage(
                userMsg.content,
                selectedProject.id,
                sessionId || undefined,
                temperature[0],
                modelProvider,
                modelName,
                historyLimit[0],
                0, // projectContextLimit deprecated
                selectedContextSessions,
                pendingChatTitle || undefined
            );

            let sources = res.sources;
            if (typeof sources === 'string') {
                try { sources = JSON.parse(sources); } catch { }
            }

            const botMsg: Message = {
                role: 'assistant',
                role: 'assistant',
                content: res.content,
                sources: sources,
                usage_metadata: res.usage_metadata
            };

            setMessages(prev => [...prev, botMsg]);
            if (!sessionId) {
                setSessionId(res.session_id);
                loadSessions(selectedProject.id); // Refresh list to show new session
            }
        } catch (e) {
            toast.error("Failed to send message");
        } finally {
            setIsTyping(false);
        }
    };

    const projectList = (
        <div className="space-y-4">
            <div className="space-y-1">
                <h3 className="mb-2 px-2 text-xs font-semibold text-slate-500 uppercase">Projects</h3>
                {projects.map(p => (
                    <div key={p.id} className="space-y-1">
                        <Button
                            variant={selectedProject?.id === p.id ? "secondary" : "ghost"}
                            className="w-full justify-start gap-2"
                            onClick={() => {
                                if (selectedProject?.id !== p.id) {
                                    setSelectedProject(p);
                                    handleNewChat();
                                }
                            }}
                        >
                            <Folder className="w-4 h-4" />
                            <span className="truncate flex-1 text-left">{p.name}</span>
                            {selectedProject?.id === p.id && <ChevronRight className="w-3 h-3 text-slate-400 rotate-90 transition-transform" />}
                        </Button>

                        <AnimatePresence>
                            {selectedProject?.id === p.id && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className="ml-4 border-l pl-2 space-y-1 overflow-hidden"
                                >
                                    <div className="flex items-center justify-between px-2 py-1">
                                        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Sections</span>
                                        <Dialog open={isNewChatOpen} onOpenChange={setIsNewChatOpen}>
                                            <DialogTrigger asChild>
                                                <Button variant="ghost" size="icon" className="h-5 w-5" title="New Chat Section">
                                                    <Plus className="w-3 h-3" />
                                                </Button>
                                            </DialogTrigger>
                                            <DialogContent>
                                                <DialogHeader>
                                                    <DialogTitle>New Chat Section</DialogTitle>
                                                    <DialogDescription>Create a new topic or thread for this project.</DialogDescription>
                                                </DialogHeader>
                                                <div className="space-y-4 py-4">
                                                    <div className="space-y-2">
                                                        <Label>Section Name</Label>
                                                        <Input
                                                            value={newChatTitle}
                                                            onChange={(e) => setNewChatTitle(e.target.value)}
                                                            placeholder="e.g., Resume Analysis, Achievements, Cover Letter"
                                                        />
                                                    </div>
                                                    <Button onClick={handleCreateNamedChat} className="w-full">Create Section</Button>
                                                </div>
                                            </DialogContent>
                                        </Dialog>
                                    </div>

                                    <div className="space-y-0.5">
                                        {sessions.map(s => (
                                            <div key={s.id} className={`group flex items-center gap-1 rounded-md px-2 py-1 text-sm transition-colors hover:bg-slate-100 dark:hover:bg-slate-800 ${sessionId === s.id ? 'bg-slate-100 dark:bg-slate-800 font-medium text-indigo-600' : 'text-slate-600 dark:text-slate-400'}`}>
                                                <button
                                                    className="flex-1 text-left truncate flex items-center gap-2"
                                                    onClick={() => handleLoadSession(s.id)}
                                                >
                                                    <MessageSquare className="w-3 h-3" />
                                                    <span className="truncate w-[110px]">{s.title || "Untitled Section"}</span>
                                                </button>
                                                <button
                                                    className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-opacity"
                                                    onClick={(e) => handleDeleteSession(s.id, e)}
                                                >
                                                    <Trash2 className="w-3 h-3" />
                                                </button>
                                            </div>
                                        ))}
                                        {sessions.length === 0 && <div className="text-[10px] text-muted-foreground px-2 py-1">No sections yet.</div>}
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                ))}
            </div>
        </div>
    );

    if (isLoading || !user) return null;

    return (
        <div className="flex h-[100dvh] bg-slate-50 dark:bg-slate-950 overflow-hidden">
            {/* Desktop Sidebar */}
            <div className="w-64 bg-white dark:bg-slate-900 border-r hidden md:flex flex-col">
                <div className="p-4 border-b font-bold text-xl flex items-center gap-2 text-indigo-600">
                    <Bot className="w-6 h-6" /> RAGOps
                </div>
                <div className="flex-1 p-4 overflow-y-auto">
                    {projectList}
                </div>
                <div className="p-4 border-t space-y-2">
                    <div className="flex items-center gap-2 mb-2">
                        <Avatar className="w-8 h-8">
                            <AvatarFallback>{user.email[0].toUpperCase()}</AvatarFallback>
                        </Avatar>
                        <div className="text-sm truncate font-medium">{user.email}</div>
                    </div>
                    <Button variant="outline" className="w-full text-xs" onClick={logout}>Logout</Button>
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col relative h-full">
                {/* Header */}
                <header className="h-16 border-b bg-white/80 dark:bg-slate-900/80 backdrop-blur flex items-center px-6 justify-between shrink-0">
                    <div className="flex items-center gap-2">
                        <Sheet>
                            <SheetTrigger asChild>
                                <Button variant="ghost" size="icon" className="md:hidden">
                                    <Menu className="w-5 h-5" />
                                </Button>
                            </SheetTrigger>
                            <SheetContent side="left" className="w-64 p-0 pt-10">
                                <div className="p-4">{projectList}</div>
                            </SheetContent>
                        </Sheet>
                        <div className="flex flex-col">
                            <span className="font-bold flex items-center gap-2">
                                {selectedProject?.name || "Select Project"}
                            </span>
                            <span className="text-xs text-muted-foreground">
                                {sessionId
                                    ? (sessions.find(s => s.id === sessionId)?.title || "Untitled Session")
                                    : (selectedProject?.description || "Select a project to view details")
                                }
                            </span>
                        </div>
                    </div>

                    <Popover>
                        <PopoverTrigger asChild>
                            <Button variant="ghost" size="sm" className="gap-2">
                                <Settings className="w-4 h-4" />
                                <span className="hidden sm:inline">Settings</span>
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-80">
                            <div className="grid gap-4">
                                <div className="space-y-2">
                                    <h4 className="font-medium leading-none">Model Settings</h4>
                                    <p className="text-sm text-muted-foreground">Configure LLM parameters.</p>
                                </div>
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <Label>Provider</Label>
                                        <select
                                            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                                            value={modelProvider}
                                            onChange={(e) => {
                                                setModelProvider(e.target.value);
                                                // Set default model for provider
                                                if (e.target.value === 'google') setModelName('gemini-1.5-flash');
                                                else setModelName('llama-3.3-70b-versatile');
                                            }}
                                        >
                                            <option value="google">Google Gemini</option>
                                            <option value="groq">Groq</option>
                                        </select>
                                    </div>

                                    <div className="space-y-2">
                                        <Label>Model</Label>
                                        <select
                                            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                                            value={modelName}
                                            onChange={(e) => setModelName(e.target.value)}
                                        >
                                            {modelProvider === 'google' ? (
                                                <>
                                                    <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
                                                    <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                                                </>
                                            ) : (
                                                <>
                                                    <option value="llama-3.3-70b-versatile">Llama 3.3 70B</option>
                                                    <option value="llama-3.1-8b-instant">Llama 3.1 8B</option>
                                                </>
                                            )}
                                        </select>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <Label>Temperature</Label>
                                        <span className="text-sm text-muted-foreground">{temperature[0]}</span>
                                    </div>
                                    <Slider
                                        defaultValue={[0.7]}
                                        max={1}
                                        step={0.1}
                                        value={temperature}
                                        onValueChange={setTemperature}
                                    />

                                    <div className="flex items-center justify-between">
                                        <Label>Context Window</Label>
                                        <span className="text-sm text-muted-foreground">{historyLimit[0]} msgs</span>
                                    </div>
                                    <Slider
                                        defaultValue={[5]}
                                        max={20}
                                        step={1}
                                        value={historyLimit}
                                        onValueChange={setHistoryLimit}
                                    />
                                    <p className="text-[10px] text-muted-foreground">Number of previous messages to remember.</p>

                                    <div className="flex items-center justify-between pt-2 border-t">
                                        <Label>Context Selection</Label>
                                        <span className="text-xs text-muted-foreground">{selectedContextSessions.length} selected</span>
                                    </div>
                                    <div className="max-h-32 overflow-y-auto space-y-1 border rounded-md p-2 bg-muted/20">
                                        {sessions.filter(s => s.id !== sessionId).map(s => (
                                            <div key={s.id} className="flex items-center space-x-2">
                                                <Checkbox
                                                    id={`ctx-${s.id}`}
                                                    checked={selectedContextSessions.includes(s.id)}
                                                    onCheckedChange={(checked) => {
                                                        if (checked) setSelectedContextSessions(prev => [...prev, s.id]);
                                                        else setSelectedContextSessions(prev => prev.filter(id => id !== s.id));
                                                    }}
                                                />
                                                <label
                                                    htmlFor={`ctx-${s.id}`}
                                                    className="text-xs font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 truncate"
                                                >
                                                    {s.title || `Session ${s.id}`}
                                                </label>
                                            </div>
                                        ))}
                                        {sessions.filter(s => s.id !== sessionId).length === 0 && (
                                            <p className="text-[10px] text-muted-foreground">No other chats to use as context.</p>
                                        )}
                                    </div>
                                    <p className="text-[10px] text-muted-foreground">Select other chats to influence this answer.</p>
                                </div>
                                {projectConfig && (
                                    <>
                                        <div className="h-px bg-border conversation-drag" />
                                        <div className="space-y-2">
                                            <h4 className="font-medium leading-none">Project Config</h4>
                                            <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground mt-2">
                                                <div>Chunk Size: <span className="font-mono text-foreground">{projectConfig.chunk_size}</span></div>
                                                <div>Overlap: <span className="font-mono text-foreground">{projectConfig.chunk_overlap}</span></div>
                                                <div>Max Tokens: <span className="font-mono text-foreground">{projectConfig.max_tokens}</span></div>
                                            </div>
                                        </div>
                                    </>
                                )}
                                {documents.length > 0 && (
                                    <>
                                        <div className="h-px bg-border cancel-drag" />
                                        <div className="space-y-2">
                                            <h4 className="font-medium leading-none">Knowledge Base</h4>
                                            <div className="max-h-24 overflow-y-auto space-y-1 mt-2">
                                                {documents.map(d => (
                                                    <div key={d.id} className="text-xs text-muted-foreground flex items-center gap-2">
                                                        <FileText className="w-3 h-3" />
                                                        <span className="truncate">{d.filename}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        </PopoverContent>
                    </Popover>
                </header>

                <div className="flex-1 overflow-y-auto p-4 md:p-8 scroll-smooth" ref={scrollRef}>
                    <div className="max-w-3xl mx-auto space-y-6 pb-4">
                        <AnimatePresence initial={false}>
                            {messages.map((msg, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    {msg.role === 'assistant' && (
                                        <Avatar className="w-8 h-8 bg-indigo-100 dark:bg-indigo-900 mt-1">
                                            <AvatarFallback><Bot className="w-5 h-5 text-indigo-600 dark:text-indigo-300" /></AvatarFallback>
                                        </Avatar>
                                    )}

                                    <div className={`space-y-2 max-w-[85%]`}>
                                        <div className={`p-4 rounded-3xl shadow-sm text-sm ${msg.role === 'user'
                                            ? 'bg-indigo-600 text-white rounded-tr-none'
                                            : 'bg-white dark:bg-slate-800 border dark:border-slate-700 rounded-tl-none'
                                            }`}>
                                            {msg.content}

                                            {msg.role === 'assistant' && msg.usage_metadata && (
                                                <div className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-700 flex justify-end">
                                                    <Popover>
                                                        <PopoverTrigger asChild>
                                                            <Button variant="ghost" size="sm" className="h-5 rounded-full text-[10px] px-2 gap-1 text-slate-500 hover:text-indigo-600">
                                                                <Info className="w-3 h-3" />
                                                                Gen Info
                                                            </Button>
                                                        </PopoverTrigger>
                                                        <PopoverContent className="w-64 p-3">
                                                            <div className="space-y-2">
                                                                <h4 className="font-semibold text-xs border-b pb-1 mb-1">Generation Settings</h4>
                                                                <div className="grid grid-cols-2 gap-y-1 text-xs">
                                                                    <span className="text-muted-foreground">Model:</span>
                                                                    <span className="font-mono">{msg.usage_metadata.model}</span>

                                                                    <span className="text-muted-foreground">Temp:</span>
                                                                    <span className="font-mono">{msg.usage_metadata.temperature}</span>

                                                                    <span className="text-muted-foreground">Provider:</span>
                                                                    <span className="font-mono">{msg.usage_metadata.provider}</span>

                                                                    <span className="text-muted-foreground">Embeddings:</span>
                                                                    <span className="font-mono truncate" title={msg.usage_metadata.embeddings}>{msg.usage_metadata.embeddings}</span>
                                                                </div>
                                                                <div className="space-y-1 pt-2 border-t">
                                                                    <span className="text-xs font-semibold text-muted-foreground">RAG Config:</span>
                                                                    <div className="grid grid-cols-2 gap-y-1 text-xs pl-2">
                                                                        <span className="text-muted-foreground">Chunk Size:</span>
                                                                        <span>{msg.usage_metadata.rag_config?.chunk_size}</span>
                                                                        <span className="text-muted-foreground">Overlap:</span>
                                                                        <span>{msg.usage_metadata.rag_config?.chunk_overlap}</span>
                                                                        <span className="text-muted-foreground">Top K:</span>
                                                                        <span>{msg.usage_metadata.rag_config?.similarity_threshold ? `> ${msg.usage_metadata.rag_config.similarity_threshold}` : "Default"}</span>
                                                                    </div>
                                                                </div>
                                                                {msg.usage_metadata.context_used && msg.usage_metadata.context_used !== "None" && (
                                                                    <div className="space-y-1 pt-2 border-t">
                                                                        <span className="text-xs font-semibold text-muted-foreground">Context Used:</span>
                                                                        <div className="text-[10px] text-slate-600 pl-2">
                                                                            {Array.isArray(msg.usage_metadata.context_used) ? msg.usage_metadata.context_used.join(", ") : msg.usage_metadata.context_used}
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </PopoverContent>
                                                    </Popover>
                                                </div>
                                            )}
                                        </div>

                                        {msg.role === 'assistant' && msg.sources && (Array.isArray(msg.sources) ? msg.sources : []).length > 0 && (
                                            <div className="flex gap-2 flex-wrap ml-2">
                                                {(msg.sources as Source[]).map((src, idx) => (
                                                    <div key={idx} className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-slate-500 bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded-full border">
                                                        <FileText className="w-3 h-3" />
                                                        <span className="truncate max-w-[120px]">{src.source}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    {msg.role === 'user' && (
                                        <Avatar className="w-8 h-8 bg-slate-200 dark:bg-slate-700 mt-1">
                                            <AvatarFallback><UserIcon className="w-5 h-5" /></AvatarFallback>
                                        </Avatar>
                                    )}
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        {isTyping && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                                <Avatar className="w-8 h-8 bg-indigo-100 dark:bg-indigo-900">
                                    <AvatarFallback><Bot className="w-5 h-5 text-indigo-600" /></AvatarFallback>
                                </Avatar>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-3xl rounded-tl-none border shadow-sm flex gap-1 items-center">
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-75" />
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-150" />
                                </div>
                            </motion.div>
                        )}
                        <div ref={scrollRef} />
                    </div>
                </div>

                {/* Input Area */}
                <div className="p-4 bg-white dark:bg-slate-900 border-t shrink-0">
                    <div className="max-w-3xl mx-auto flex gap-2">
                        <Input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                            placeholder={selectedProject ? `Ask about "${selectedProject.name}"...` : "Select a project to start chatting..."}
                            disabled={!selectedProject}
                            className="flex-1 rounded-full px-6 h-12 border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 focus-visible:ring-indigo-500 text-base"
                        />
                        <Button
                            onClick={handleSend}
                            disabled={!selectedProject || !input.trim()}
                            size="icon"
                            className="shrink-0 w-12 h-12 rounded-full bg-indigo-600 hover:bg-indigo-700 transition-all shadow-lg hover:shadow-indigo-500/25"
                        >
                            <Send className="w-5 h-5" />
                        </Button>
                    </div>
                </div>
            </div >
        </div >
    );
}
