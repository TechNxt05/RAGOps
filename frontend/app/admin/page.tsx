"use client";

import { useEffect, useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import {
    getProjects, createProject, getRAGConfig, updateRAGConfig, uploadDocument, getDocuments,
    getDocumentChunks, debugSearch,
    Project, RAGConfig, Document, Chunk, DebugSearchResult
} from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import { Plus, Folder, Settings, Upload, FileText, Info, Search, Database, Layers, Trash2 } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog";
import { deleteProject } from '@/lib/api';

export default function AdminDashboard() {
    const { user, logout, isLoading } = useAuth();
    const router = useRouter();
    const [projects, setProjects] = useState<Project[]>([]);
    const [selectedProject, setSelectedProject] = useState<Project | null>(null);
    const [config, setConfig] = useState<Partial<RAGConfig>>({
        chunk_size: 1000, chunk_overlap: 200, max_tokens: 2000,
        temperature: 0.7, top_p: 0.9, top_k: 4, similarity_threshold: 0.0,
        max_output_tokens: 1024, max_context_tokens: 2048,
        response_style: "Concise", answer_only_from_docs: false, hallucination_guard: false
    });
    const [documents, setDocuments] = useState<Document[]>([]);

    // Inspector State
    const [inspectDocId, setInspectDocId] = useState<number | null>(null);
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [debugQuery, setDebugQuery] = useState("");
    const [debugResults, setDebugResults] = useState<DebugSearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);

    const [newProjectName, setNewProjectName] = useState("");
    const [isCreateOpen, setIsCreateOpen] = useState(false);

    useEffect(() => {
        if (!isLoading && (!user || user.role !== 'admin')) {
            if (user && user.role !== 'admin') router.push('/chat');
            else router.push('/login');
        }
    }, [user, isLoading, router]);

    useEffect(() => {
        if (user?.role === 'admin') loadProjects();
    }, [user]);

    useEffect(() => {
        if (selectedProject) {
            loadConfig(selectedProject.id);
            refreshDocuments();
        }
    }, [selectedProject]);

    const refreshDocuments = () => {
        if (selectedProject) {
            getDocuments(selectedProject.id).then(setDocuments).catch(() => { });
        }
    }

    const loadProjects = async () => {
        try {
            const data = await getProjects();
            setProjects(data);
            if (data.length > 0 && !selectedProject) setSelectedProject(data[0]);
        } catch (e) {
            toast.error("Failed to load projects");
        }
    };

    const handleCreateProject = async () => {
        if (!newProjectName) return;
        try {
            const newProj = await createProject(newProjectName, "");
            toast.success("Project created!");
            setProjects([...projects, newProj]);
            setSelectedProject(newProj);
            setNewProjectName("");
            setIsCreateOpen(false);
        } catch (e) {
            toast.error("Failed to create project");
        }
    };

    const handleDeleteProject = async () => {
        if (!selectedProject) return;
        try {
            await deleteProject(selectedProject.id);
            toast.success("Project deleted");
            const updatedProjects = projects.filter(p => p.id !== selectedProject.id);
            setProjects(updatedProjects);
            setSelectedProject(updatedProjects.length > 0 ? updatedProjects[0] : null);
        } catch (e) {
            toast.error("Failed to delete project");
        }
    }

    const loadConfig = async (pid: number) => {
        try {
            const data = await getRAGConfig(pid);
            setConfig(data);
        } catch (e) { toast.error("Failed to load config"); }
    };

    const handleSaveConfig = async () => {
        if (!selectedProject) return;
        try {
            await updateRAGConfig({ ...config, project_id: selectedProject.id });
            toast.success("Configuration saved for " + selectedProject.name);
        } catch (e) { toast.error("Failed to save config"); }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!selectedProject || !e.target.files) return;
        const file = e.target.files[0];
        try {
            toast.info("Uploading to " + selectedProject.name + "...");
            await uploadDocument(selectedProject.id, file);
            toast.success("Document processed!");
            refreshDocuments(); // Fix: Reload documents after upload
        } catch (e) { toast.error("Upload failed."); }
    };

    const handleFetchChunks = async (docId: string) => {
        if (!docId) return;
        setInspectDocId(parseInt(docId));
        try {
            const data = await getDocumentChunks(parseInt(docId));
            setChunks(data);
        } catch (e) { toast.error("Failed to fetch chunks"); }
    };

    const handleDebugSearch = async () => {
        if (!selectedProject || !debugQuery) return;
        setIsSearching(true);
        try {
            const results = await debugSearch({
                project_id: selectedProject.id,
                query: debugQuery,
                top_k: config.top_k || 4,
                similarity_threshold: config.similarity_threshold || 0
            });
            setDebugResults(results);
        } catch (e) {
            toast.error("Search failed");
        } finally {
            setIsSearching(false);
        }
    }

    if (isLoading || !user) return <div className="flex h-screen items-center justify-center">Loading...</div>;

    const SettingInfo = ({ text, recommend }: { text: string, recommend: string }) => (
        <div className="text-xs text-muted-foreground mt-1">
            <span className="block mb-1">{text}</span>
            <span className="font-medium text-blue-600 dark:text-blue-400">Recommendation: {recommend}</span>
        </div>
    );

    return (
        <div className="flex h-screen bg-slate-50 dark:bg-slate-950 overflow-hidden">
            {/* Sidebar */}
            <div className="w-64 bg-white dark:bg-slate-900 border-r flex flex-col p-4">
                <div className="flex items-center justify-between mb-8">
                    <h1 className="font-bold text-xl tracking-tight">RAGOps Admin</h1>
                    <Button variant="ghost" size="icon" onClick={logout} title="Logout">
                        <span className="sr-only">Logout</span>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-log-out w-5 h-5"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" x2="9" y1="12" y2="12" /></svg>
                    </Button>
                </div>

                <div className="flex items-center justify-between mb-4">
                    <span className="text-sm font-semibold text-slate-500">PROJECTS</span>
                    <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
                        <DialogTrigger asChild>
                            <Button variant="ghost" size="sm"><Plus className="w-4 h-4" /></Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Create New Project</DialogTitle>
                                <DialogDescription>Enter a name for the new project folder.</DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                                <Label>Project Name</Label>
                                <Input value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} />
                                <Button onClick={handleCreateProject}>Create</Button>
                            </div>
                        </DialogContent>
                    </Dialog>
                </div>

                <div className="space-y-1 flex-1 overflow-y-auto">
                    {projects.map(p => (
                        <Button
                            key={p.id}
                            variant={selectedProject?.id === p.id ? "secondary" : "ghost"}
                            className="w-full justify-start gap-2"
                            onClick={() => setSelectedProject(p)}
                        >
                            <Folder className="w-4 h-4" />
                            {p.name}
                        </Button>
                    ))}
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 p-8 overflow-y-auto">
                {selectedProject ? (
                    <div className="max-w-6xl mx-auto space-y-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <h2 className="text-3xl font-bold">{selectedProject.name}</h2>
                                <p className="text-muted-foreground">{selectedProject.description || "Manage project settings and documents"}</p>
                            </div>
                            <Dialog>
                                <DialogTrigger asChild>
                                    <Button variant="destructive" className="gap-2">
                                        <Trash2 className="w-4 h-4" /> Delete Project
                                    </Button>
                                </DialogTrigger>
                                <DialogContent>
                                    <DialogHeader>
                                        <DialogTitle>Delete Project?</DialogTitle>
                                        <DialogDescription>
                                            This action cannot be undone. It will permanently delete the project "<strong>{selectedProject.name}</strong>" and all associated documents, indexes, and chat sessions.
                                        </DialogDescription>
                                    </DialogHeader>
                                    <DialogFooter>
                                        <DialogClose asChild>
                                            <Button variant="outline">Cancel</Button>
                                        </DialogClose>
                                        <Button variant="destructive" onClick={handleDeleteProject}>Confirm Delete</Button>
                                    </DialogFooter>
                                </DialogContent>
                            </Dialog>
                        </div>

                        <Tabs defaultValue="inspector" className="w-full">
                            <TabsList>
                                <TabsTrigger value="config" className="gap-2"><Settings className="w-4 h-4" /> RAG Configuration</TabsTrigger>
                                <TabsTrigger value="files" className="gap-2"><FileText className="w-4 h-4" /> Knowledge Base</TabsTrigger>
                                <TabsTrigger value="inspector" className="gap-2"><Database className="w-4 h-4" /> Knowledge Inspector</TabsTrigger>
                            </TabsList>

                            {/* CONFIG TAB */}
                            <TabsContent value="config">
                                <Card>
                                    <CardHeader>
                                        <CardTitle>RAG Settings</CardTitle>
                                        <CardDescription>Configure Generation, Retrieval, and Safety parameters.</CardDescription>
                                    </CardHeader>
                                    <CardContent className="space-y-8">
                                        {/* Retrieval Section */}
                                        <div className="space-y-4">
                                            <div className="flex items-center gap-2 border-b pb-2">
                                                <h3 className="text-lg font-semibold">Retrieval Configuration</h3>
                                                <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300">Core</span>
                                            </div>

                                            <div className="grid gap-8 md:grid-cols-2">
                                                <div className="space-y-2">
                                                    <Label>Chunk Size (Tokens)</Label>
                                                    <Input type="number" value={config.chunk_size} onChange={(e) => setConfig({ ...config, chunk_size: parseInt(e.target.value) })} />
                                                    <SettingInfo text="Characters per chunk. Smaller chunks are more precise, larger capture more context." recommend="1000 for standard docs." />
                                                </div>
                                                <div className="space-y-2">
                                                    <Label>Chunk Overlap</Label>
                                                    <Input type="number" value={config.chunk_overlap} onChange={(e) => setConfig({ ...config, chunk_overlap: parseInt(e.target.value) })} />
                                                    <SettingInfo text="Duplicate characters at edges to preserve context between chunks." recommend="10-20% of Chunk Size (e.g. 200)." />
                                                </div>
                                                <div className="space-y-2">
                                                    <div className="flex justify-between"><Label>Top K: {config.top_k}</Label></div>
                                                    <Slider value={[config.top_k || 4]} max={20} step={1} onValueChange={(v) => setConfig({ ...config, top_k: v[0] })} />
                                                    <SettingInfo text="Number of document chunks to retrieve and feed to the AI." recommend="3-5. Too many = expensive/confusing." />
                                                </div>
                                                <div className="space-y-2">
                                                    <div className="flex justify-between"><Label>Similarity Threshold: {config.similarity_threshold}</Label></div>
                                                    <Slider value={[config.similarity_threshold || 0]} max={1} step={0.05} onValueChange={(v) => setConfig({ ...config, similarity_threshold: v[0] })} />
                                                    <SettingInfo text="Minimum relevance score (0.0=Any, 1.0=Exact Match). Filters out noise." recommend="0.5 for strict, 0.0 for broad." />
                                                </div>
                                            </div>
                                        </div>

                                        {/* Generation Section */}
                                        <div className="space-y-4">
                                            <div className="flex items-center gap-2 border-b pb-2">
                                                <h3 className="text-lg font-semibold">LLM Generation</h3>
                                                <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300">Model</span>
                                            </div>

                                            <div className="grid gap-8 md:grid-cols-2">
                                                <div className="space-y-2">
                                                    <div className="flex justify-between"><Label>Temperature: {config.temperature}</Label></div>
                                                    <Slider value={[config.temperature || 0.7]} max={1} step={0.1} onValueChange={(v) => setConfig({ ...config, temperature: v[0] })} />
                                                    <SettingInfo text="Controls randomness. 0.0 is deterministic/factual, 1.0 is creative." recommend="0.0 - 0.3 for RAG/Fact-finding." />
                                                </div>
                                                <div className="space-y-2">
                                                    <div className="flex justify-between"><Label>Top P: {config.top_p}</Label></div>
                                                    <Slider value={[config.top_p || 0.9]} max={1} step={0.05} onValueChange={(v) => setConfig({ ...config, top_p: v[0] })} />
                                                    <SettingInfo text="Nucleus sampling. limits vocabulary to top probability mass." recommend="0.9 is standard." />
                                                </div>
                                                <div className="space-y-2">
                                                    <Label>Response Style</Label>
                                                    <select className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                                                        value={config.response_style || "Concise"}
                                                        onChange={(e) => setConfig({ ...config, response_style: e.target.value })}
                                                    >
                                                        <option value="Concise">Concise - Short & Direct</option>
                                                        <option value="Detailed">Detailed - Deep explanation</option>
                                                        <option value="Step-by-step">Step-by-step - Good for tutorials</option>
                                                        <option value="Academic">Academic - Formal tone</option>
                                                    </select>
                                                    <SettingInfo text="Injects specific instructions into the System Prompt." recommend="Concise for quick answers." />
                                                </div>
                                                <div className="space-y-2">
                                                    <Label>Max Output Tokens</Label>
                                                    <Input type="number" value={config.max_output_tokens} onChange={(e) => setConfig({ ...config, max_output_tokens: parseInt(e.target.value) })} />
                                                    <SettingInfo text="Maximum length of the generated response." recommend="1024 (approx 750 words)." />
                                                </div>
                                            </div>
                                        </div>

                                        {/* Safety Section */}
                                        <div className="space-y-4">
                                            <div className="flex items-center gap-2 border-b pb-2">
                                                <h3 className="text-lg font-semibold">Safety & Governance</h3>
                                                <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300">Guardrails</span>
                                            </div>
                                            <div className="grid gap-6 md:grid-cols-2">
                                                <div className="border p-4 rounded-lg space-y-3">
                                                    <div className="flex items-center justify-between"><Label className="font-semibold">Strict Retrieval Mode</Label><Switch checked={config.answer_only_from_docs} onCheckedChange={(c) => setConfig({ ...config, answer_only_from_docs: c })} /></div>
                                                    <SettingInfo text="If checked, AI will refuse to answer if information is not found in docs." recommend="ON for compliance/enterprise use." />
                                                </div>
                                                <div className="border p-4 rounded-lg space-y-3">
                                                    <div className="flex items-center justify-between"><Label className="font-semibold">Hallucination Guard</Label><Switch checked={config.hallucination_guard} onCheckedChange={(c) => setConfig({ ...config, hallucination_guard: c })} /></div>
                                                    <SettingInfo text="Checks response against context to ensure factual accuracy (Extra latency)." recommend="OFF unless accuracy is critical." />
                                                </div>
                                            </div>
                                        </div>

                                        <div className="pt-4">
                                            <Button size="lg" onClick={handleSaveConfig} className="w-full font-semibold">Save All Configurations</Button>
                                            <p className="text-center text-xs text-muted-foreground mt-2">Changes apply immediately to new chat sessions.</p>
                                        </div>
                                    </CardContent>
                                </Card>
                            </TabsContent>

                            {/* FILES TAB */}
                            <TabsContent value="files">
                                <Card>
                                    <CardHeader>
                                        <CardTitle>Upload Documents</CardTitle>
                                        <CardDescription>Upload PDF or Text files to be indexed for this project.</CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="p-12 border-2 border-dashed rounded-lg flex flex-col items-center justify-center text-center space-y-4 hover:bg-slate-50 dark:hover:bg-slate-800 transition">
                                            <Upload className="w-10 h-10 text-slate-400" />
                                            <div className="text-muted-foreground">
                                                <p>Drag & drop or Click to Upload</p>
                                                <p className="text-sm">Supported: PDF, TXT</p>
                                            </div>
                                            <Input type="file" onChange={handleFileUpload} className="max-w-xs cursor-pointer" />
                                        </div>

                                        <div className="mt-6 space-y-2">
                                            <h3 className="font-semibold text-sm text-slate-500">Indexed Files</h3>
                                            <div className="space-y-2">
                                                {documents.map(doc => (
                                                    <div key={doc.id} className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900 rounded border">
                                                        <div className="flex items-center gap-3">
                                                            <div className={`p-2 rounded ${doc.processed ? 'bg-green-100 text-green-600' : 'bg-yellow-100 text-yellow-600'}`}>
                                                                <FileText className="w-4 h-4" />
                                                            </div>
                                                            <div>
                                                                <div className="font-medium text-sm">{doc.filename}</div>
                                                                <div className="text-xs text-muted-foreground">{new Date(doc.uploaded_at).toLocaleDateString()}</div>
                                                            </div>
                                                        </div>
                                                        <span className="text-xs px-2 py-1 bg-white dark:bg-slate-800 border rounded">{doc.processed ? 'Ready' : 'Processing...'}</span>
                                                    </div>
                                                ))}
                                                {documents.length === 0 && <div className="text-sm text-muted-foreground text-center py-4">No documents yet.</div>}
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>
                            </TabsContent>

                            {/* INSPECTOR TAB */}
                            <TabsContent value="inspector">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Left: Chunk Browser */}
                                    <Card>
                                        <CardHeader>
                                            <CardTitle className="flex items-center gap-2"><Layers className="w-5 h-5" /> Chunk Explorer</CardTitle>
                                            <CardDescription>See how your documents are split.</CardDescription>
                                        </CardHeader>
                                        <CardContent className="space-y-4">
                                            <div className="space-y-2">
                                                <Label>Select Document</Label>
                                                <select
                                                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                                                    onChange={(e) => handleFetchChunks(e.target.value)}
                                                >
                                                    <option value="">-- Select a file --</option>
                                                    {documents.map(d => (
                                                        <option key={d.id} value={d.id}>{d.filename}</option>
                                                    ))}
                                                </select>
                                            </div>

                                            <div className="h-[500px] overflow-y-auto border rounded-md p-2 space-y-2 bg-slate-50 dark:bg-slate-900">
                                                {chunks.length === 0 && <div className="text-center text-muted-foreground pt-10">Select a document to view chunks.</div>}
                                                {chunks.map((chunk, i) => (
                                                    <div key={chunk.id} className="p-3 bg-white dark:bg-slate-800 rounded border text-sm shadow-sm">
                                                        <div className="flex justify-between items-center mb-1 border-b pb-1">
                                                            <span className="font-bold text-xs">Chunk #{i + 1}</span>
                                                            <span className="text-xs text-muted-foreground">ID: {chunk.id}</span>
                                                        </div>
                                                        <p className="whitespace-pre-wrap text-slate-700 dark:text-slate-300 font-mono text-xs">{chunk.content}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        </CardContent>
                                    </Card>

                                    {/* Right: Retrieval Playground */}
                                    <Card className="flex flex-col h-full">
                                        <CardHeader>
                                            <CardTitle className="flex items-center gap-2"><Search className="w-5 h-5" /> Retrieval Playground</CardTitle>
                                            <CardDescription>Test your vector search logic.</CardDescription>
                                        </CardHeader>
                                        <CardContent className="flex-1 flex flex-col space-y-4">
                                            <div className="flex gap-2">
                                                <Input
                                                    placeholder="Enter a test query..."
                                                    value={debugQuery}
                                                    onChange={(e) => setDebugQuery(e.target.value)}
                                                    onKeyDown={(e) => e.key === 'Enter' && handleDebugSearch()}
                                                />
                                                <Button onClick={handleDebugSearch} disabled={isSearching}>
                                                    {isSearching ? "..." : "Search"}
                                                </Button>
                                            </div>

                                            <div className="flex items-center gap-4 text-xs text-muted-foreground bg-slate-100 dark:bg-slate-800 p-2 rounded">
                                                <span>Current Settings:</span>
                                                <strong>Top K: {config.top_k}</strong>
                                                <strong>Threshold: {config.similarity_threshold}</strong>
                                            </div>

                                            <div className="flex-1 overflow-y-auto border rounded-md p-2 bg-slate-50 dark:bg-slate-900 space-y-2">
                                                {debugResults.length === 0 && !isSearching && <div className="text-center text-muted-foreground pt-10">No results. Try a query.</div>}
                                                {debugResults.map((res, i) => (
                                                    <div key={i} className="p-3 bg-white dark:bg-slate-800 rounded border shadow-sm">
                                                        <div className="flex justify-between items-start mb-2">
                                                            <div className="text-xs font-semibold text-blue-600 truncate max-w-[200px]">{res.document_name}</div>
                                                            <div className="flex items-center gap-2">
                                                                <div className="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
                                                                    <div
                                                                        className={`h-full ${res.score > 0.7 ? 'bg-green-500' : res.score > 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                                                        style={{ width: `${Math.max(0, Math.min(100, res.score * 100))}%` }}
                                                                    />
                                                                </div>
                                                                <span className="text-xs font-mono">{(res.score * 100).toFixed(1)}%</span>
                                                            </div>
                                                        </div>
                                                        <p className="text-xs text-slate-600 dark:text-slate-400 line-clamp-4">{res.content}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        </CardContent>
                                    </Card>
                                </div>
                            </TabsContent>
                        </Tabs>
                    </div>
                ) : (
                    <div className="flex h-full items-center justify-center text-muted-foreground">Select a project to manage</div>
                )}
            </div>
        </div>
    );
}
