"use client";

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';
import {
    getProjects, createProject, getRAGConfig, updateRAGConfig, uploadDocument, getDocuments,
    getDocumentChunks, debugSearch, deleteProject, compareModels,
    Project, RAGConfig, Document, Chunk, DebugSearchResult, type CompareModelsResponse,
} from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import { Plus, Folder, Settings, Upload, FileText, Info, Search, Database, Layers, Trash2, BarChart3, Zap, Menu } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { ModelOrchestrator } from '@/components/ModelOrchestrator';
import { DocumentManagerPanel } from '@/components/DocumentManagerPanel';

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
    const [debugQueryAnalysis, setDebugQueryAnalysis] = useState<any>(null);
    const [debugPipelineTrace, setDebugPipelineTrace] = useState<any>(null);
    const [isSearching, setIsSearching] = useState(false);

    const [newProjectName, setNewProjectName] = useState("");
    const [isCreateOpen, setIsCreateOpen] = useState(false);
    const [lastUploadedDocId, setLastUploadedDocId] = useState<number | null>(null);
    const [compareOpen, setCompareOpen] = useState(false);
    const [compareQuery, setCompareQuery] = useState("");
    const [compareLoading, setCompareLoading] = useState(false);
    const [compareData, setCompareData] = useState<CompareModelsResponse | null>(null);

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
            const res = await uploadDocument(selectedProject.id, file);
            setLastUploadedDocId(res.doc_id);
            toast.success("Document queued for processing");
            refreshDocuments();
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
            const res = await debugSearch({
                project_id: selectedProject.id,
                query: debugQuery,
                top_k: config.top_k || 4,
                similarity_threshold: config.similarity_threshold || 0
            });
            setDebugResults(res.results);
            setDebugQueryAnalysis(res.query_analysis);
            setDebugPipelineTrace(res.pipeline_trace);
        } catch (e) {
            toast.error("Search failed");
        } finally {
            setIsSearching(false);
        }
    }

    const runCompare = async () => {
        if (!selectedProject || !compareQuery.trim()) return;
        setCompareLoading(true);
        try {
            const data = await compareModels(selectedProject.id, compareQuery.trim());
            setCompareData(data);
        } catch {
            toast.error("Model comparison failed");
        } finally {
            setCompareLoading(false);
        }
    };

    if (isLoading || !user) return (
        <div className="flex h-screen items-center justify-center bg-slate-50 dark:bg-slate-950">
            <div className="h-32 w-64 animate-pulse rounded-xl bg-muted" />
        </div>
    );

    const SettingInfo = ({ text, recommend }: { text: string, recommend: string }) => (
        <div className="text-xs text-muted-foreground mt-1">
            <span className="block mb-1">{text}</span>
            <span className="font-medium text-blue-600 dark:text-blue-400">Recommendation: {recommend}</span>
        </div>
    );

    const projectList = (
        <>
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
                        className="w-full justify-start gap-2 text-sm"
                        onClick={() => setSelectedProject(p)}
                    >
                        <Folder className="w-4 h-4 shrink-0" />
                        <span className="truncate">{p.name}</span>
                    </Button>
                ))}
                {projects.length === 0 && (
                    <div className="text-xs text-muted-foreground text-center py-4">No projects yet</div>
                )}
            </div>
            {selectedProject && (
                <div className="mt-4 space-y-2">
                    <Button variant="outline" className="w-full gap-2 text-xs" asChild>
                        <Link href={`/analytics/${selectedProject.id}`}>
                            <BarChart3 className="h-4 w-4" />
                            Analytics
                        </Link>
                    </Button>
                </div>
            )}
        </>
    );

    return (
        <div className="flex h-screen bg-slate-50 dark:bg-slate-950 overflow-hidden">
            {/* Desktop Sidebar */}
            <div className="w-64 bg-white dark:bg-slate-900 border-r hidden md:flex flex-col p-4">
                <div className="flex items-center justify-between mb-8">
                    <h1 className="font-bold text-xl tracking-tight text-indigo-600">RAGOps</h1>
                    <Button variant="ghost" size="icon" onClick={logout} title="Logout">
                        <span className="sr-only">Logout</span>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-log-out w-5 h-5"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" x2="9" y1="12" y2="12" /></svg>
                    </Button>
                </div>
                {projectList}
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Header for Mobile */}
                <header className="h-14 border-b bg-white dark:bg-slate-900 flex items-center px-4 md:hidden">
                    <Sheet>
                        <SheetTrigger asChild>
                            <Button variant="ghost" size="icon">
                                <Menu className="w-5 h-5" />
                            </Button>
                        </SheetTrigger>
                        <SheetContent side="left" className="w-64 p-4 flex flex-col">
                            <div className="mb-8 font-bold text-xl text-indigo-600">RAGOps Admin</div>
                            {projectList}
                        </SheetContent>
                    </Sheet>
                    <span className="ml-3 font-semibold truncate">{selectedProject?.name || "Admin Panel"}</span>
                </header>

                <div className="flex-1 p-4 md:p-8 overflow-y-auto">
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
                                <ErrorBoundary section="RAG configuration">
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
                                <div className="mt-6 space-y-6">
                                    <ErrorBoundary section="Model orchestration">
                                        {selectedProject && <ModelOrchestrator projectId={selectedProject.id} />}
                                    </ErrorBoundary>
                                    <Card>
                                        <CardHeader>
                                            <CardTitle className="flex items-center gap-2"><Zap className="w-5 h-5" />Live model comparison</CardTitle>
                                            <CardDescription>Primary vs fallback models on the same retrieved context (admin).</CardDescription>
                                        </CardHeader>
                                        <CardContent className="space-y-4">
                                            <Textarea
                                                placeholder="e.g. What is our refund policy?"
                                                value={compareQuery}
                                                onChange={(e) => setCompareQuery(e.target.value)}
                                                rows={3}
                                            />
                                            <Button type="button" disabled={compareLoading} onClick={() => void runCompare()}>
                                                {compareLoading ? "Running…" : "Run comparison"}
                                            </Button>
                                            {compareData && (
                                                <div className="space-y-3">
                                                    <div className="grid gap-4 md:grid-cols-2">
                                                        <div className="rounded-lg border bg-card p-3 text-sm shadow-sm">
                                                            <div className="mb-2 font-semibold text-indigo-600">{compareData.left.model}</div>
                                                            <p className="max-h-48 overflow-y-auto whitespace-pre-wrap text-xs text-muted-foreground">{String(compareData.left.content)}</p>
                                                            <div className="mt-2 space-y-1 border-t pt-2 text-[11px] text-muted-foreground">
                                                                <div>Latency: {Math.round(compareData.left.latency_ms)} ms</div>
                                                                <div>Hallucination risk: {compareData.left.hallucination_score.toFixed(2)}</div>
                                                                <div>Faithfulness: {compareData.left.faithfulness_score.toFixed(2)}</div>
                                                            </div>
                                                        </div>
                                                        <div className="rounded-lg border bg-card p-3 text-sm shadow-sm">
                                                            <div className="mb-2 font-semibold text-violet-600">{compareData.right.model}</div>
                                                            <p className="max-h-48 overflow-y-auto whitespace-pre-wrap text-xs text-muted-foreground">{String(compareData.right.content)}</p>
                                                            <div className="mt-2 space-y-1 border-t pt-2 text-[11px] text-muted-foreground">
                                                                <div>Latency: {Math.round(compareData.right.latency_ms)} ms</div>
                                                                <div>Hallucination risk: {compareData.right.hallucination_score.toFixed(2)}</div>
                                                                <div>Faithfulness: {compareData.right.faithfulness_score.toFixed(2)}</div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <p className="text-xs text-muted-foreground">Winner: <span className="font-medium text-foreground">{compareData.winner}</span></p>
                                                </div>
                                            )}
                                        </CardContent>
                                    </Card>
                                </div>
                                </ErrorBoundary>
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

                                        <ErrorBoundary section="Documents">
                                            {selectedProject && (
                                                <DocumentManagerPanel
                                                    projectId={selectedProject.id}
                                                    topK={config.top_k || 4}
                                                    similarityThreshold={config.similarity_threshold || 0}
                                                    lastUploadedDocId={lastUploadedDocId}
                                                />
                                            )}
                                        </ErrorBoundary>
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
                                        <CardContent className="flex-1 flex flex-col space-y-4 text-left">
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

                                            <Tabs defaultValue="results" className="w-full flex-1 flex flex-col min-h-0 text-left">
                                                <TabsList className="grid w-full grid-cols-3">
                                                    <TabsTrigger value="results">Retrieved Chunks</TabsTrigger>
                                                    <TabsTrigger value="understanding">Query Analysis</TabsTrigger>
                                                    <TabsTrigger value="trace">Pipeline Trace</TabsTrigger>
                                                </TabsList>

                                                <TabsContent value="results" className="flex-1 flex flex-col min-h-0 mt-2">
                                                    <div className="flex-1 overflow-y-auto border rounded-md p-2 bg-slate-50 dark:bg-slate-900 space-y-2 max-h-[350px]">
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
                                                </TabsContent>

                                                <TabsContent value="understanding" className="flex-1 flex flex-col min-h-0 mt-2 text-left">
                                                    <div className="flex-1 border rounded-md p-3 bg-slate-50 dark:bg-slate-900 text-xs text-left max-h-[350px] overflow-y-auto">
                                                        {debugQueryAnalysis ? (
                                                            <div className="space-y-4 text-left">
                                                                <div className="flex items-center justify-between p-2 rounded bg-white dark:bg-slate-800 border">
                                                                    <span className="font-semibold text-muted-foreground uppercase text-[10px] text-left">Complexity Classification</span>
                                                                    <span className={`px-2 py-0.5 rounded-full font-bold uppercase text-[9px] ${
                                                                        debugQueryAnalysis.complexity === 'factoid' ? 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300' :
                                                                        debugQueryAnalysis.complexity === 'analytical' ? 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300' :
                                                                        'bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300'
                                                                    }`}>
                                                                        {debugQueryAnalysis.complexity} ({(debugQueryAnalysis.confidence * 100).toFixed(0)}% confidence)
                                                                    </span>
                                                                </div>

                                                                <div className="space-y-1 text-left">
                                                                    <span className="font-bold text-muted-foreground uppercase text-[9px] block text-left">Decomposed Sub-Queries</span>
                                                                    <div className="space-y-1 text-left">
                                                                        {debugQueryAnalysis.sub_queries?.map((sub: string, idx: number) => (
                                                                            <div key={idx} className="p-2 rounded bg-white dark:bg-slate-800 border text-slate-700 dark:text-slate-300 font-mono text-[11px] text-left">
                                                                                {idx + 1}. {sub}
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                </div>

                                                                <div className="space-y-1 text-left">
                                                                    <span className="font-bold text-muted-foreground uppercase text-[9px] block text-left">Expanded Query String</span>
                                                                    <div className="p-2 rounded bg-white dark:bg-slate-800 border text-slate-700 dark:text-slate-300 font-mono text-[11px] break-words text-left">
                                                                        {debugQueryAnalysis.expanded_query || "No expansion applied"}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <div className="text-center text-muted-foreground py-10">Run a search query to view semantic analysis.</div>
                                                        )}
                                                    </div>
                                                </TabsContent>

                                                <TabsContent value="trace" className="flex-1 flex flex-col min-h-0 mt-2 text-left">
                                                    <div className="flex-1 border rounded-md p-3 bg-slate-50 dark:bg-slate-900 text-xs text-left max-h-[350px] overflow-y-auto">
                                                        {debugPipelineTrace ? (
                                                            <div className="space-y-4 text-left">
                                                                <div className="flex justify-between items-center border-b pb-2 text-left">
                                                                    <div>
                                                                        <span className="font-bold text-slate-700 dark:text-slate-300">Pipeline Status</span>
                                                                        <span className={`ml-2 px-1.5 py-0.5 rounded font-mono uppercase text-[9px] ${
                                                                            debugPipelineTrace.status === 'success' ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                                                                        }`}>
                                                                            {debugPipelineTrace.status}
                                                                        </span>
                                                                    </div>
                                                        <div className="text-right font-mono text-muted-foreground">
                                                                        Total: {debugPipelineTrace.total_duration_ms?.toFixed(1)}ms
                                                                    </div>
                                                                </div>

                                                                {debugPipelineTrace.turn_type && (
                                                                 <div className="grid grid-cols-2 gap-3 mb-4">
                                                                     {debugPipelineTrace.turn_type && (
                                                                         <div className="col-span-2 flex space-x-2">
                                                                             <div className="flex-1 bg-white dark:bg-slate-800 p-2.5 rounded border shadow-sm">
                                                                                 <span className="text-[10px] uppercase text-muted-foreground font-semibold">Turn Classification</span>
                                                                                 <div className="mt-1 font-mono text-xs">{debugPipelineTrace.turn_type}</div>
                                                                             </div>
                                                                             {debugPipelineTrace.route_decision && (
                                                                                 <div className="flex-1 bg-white dark:bg-slate-800 p-2.5 rounded border shadow-sm">
                                                                                     <span className="text-[10px] uppercase text-muted-foreground font-semibold">Route Decision</span>
                                                                                     <div className="mt-1 font-mono text-xs text-indigo-600 dark:text-indigo-400">
                                                                                         {debugPipelineTrace.route_decision.route} 
                                                                                         {debugPipelineTrace.route_decision.tier > 0 && ` (Tier ${debugPipelineTrace.route_decision.tier})`}
                                                                                     </div>
                                                                                 </div>
                                                                             )}
                                                                         </div>
                                                                     )}

                                                                     {/* Semantic Routing card */}
                                                                     {debugPipelineTrace.semantic_routing && (
                                                                         <div className="bg-white dark:bg-slate-800 p-3 rounded border shadow-sm space-y-2 border-l-4 border-l-emerald-500">
                                                                             <span className="text-[10px] uppercase text-muted-foreground font-bold flex items-center gap-1.5">
                                                                                 <Layers className="w-3.5 h-3.5 text-emerald-500" />
                                                                                 Semantic Routing
                                                                             </span>
                                                                             <div className="space-y-1">
                                                                                 <div className="flex justify-between">
                                                                                     <span className="text-slate-400">Category:</span>
                                                                                     <span className="font-semibold">{debugPipelineTrace.semantic_routing.category}</span>
                                                                                 </div>
                                                                                 <div className="flex justify-between">
                                                                                     <span className="text-slate-400">Strategy:</span>
                                                                                     <span className={`font-mono px-1 rounded text-[10px] ${
                                                                                         debugPipelineTrace.semantic_routing.strategy === 'fast' ? 'bg-green-100 text-green-800 dark:bg-green-950/20 dark:text-green-400' :
                                                                                         debugPipelineTrace.semantic_routing.strategy === 'deep' ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/20 dark:text-amber-400' :
                                                                                         debugPipelineTrace.semantic_routing.strategy === 'parallel' ? 'bg-purple-100 text-purple-800 dark:bg-purple-950/20 dark:text-purple-400' :
                                                                                         'bg-blue-100 text-blue-800 dark:bg-blue-950/20 dark:text-blue-400'
                                                                                     }`}>
                                                                                         {debugPipelineTrace.semantic_routing.strategy?.toUpperCase()}
                                                                                     </span>
                                                                                 </div>
                                                                                 <div className="flex justify-between text-[10px] text-slate-500 pt-1">
                                                                                     <span>Top K: {debugPipelineTrace.semantic_routing.top_k}</span>
                                                                                     <span>Rerank N: {debugPipelineTrace.semantic_routing.rerank_top_n}</span>
                                                                                 </div>
                                                                             </div>
                                                                         </div>
                                                                     )}

                                                                     {/* Output Contract card */}
                                                                     {debugPipelineTrace.output_contract && (
                                                                         <div className="bg-white dark:bg-slate-800 p-3 rounded border shadow-sm space-y-2 border-l-4 border-l-blue-500">
                                                                             <span className="text-[10px] uppercase text-muted-foreground font-bold flex items-center gap-1.5">
                                                                                 <Settings className="w-3.5 h-3.5 text-blue-500" />
                                                                                 Output Contract
                                                                             </span>
                                                                             <div className="space-y-1">
                                                                                 <div className="flex justify-between">
                                                                                     <span className="text-slate-400">Format:</span>
                                                                                     <span className="font-semibold uppercase font-mono text-[10px]">{debugPipelineTrace.output_contract.format || debugPipelineTrace.output_contract.format_used}</span>
                                                                                 </div>
                                                                                 <div className="flex justify-between">
                                                                                     <span className="text-slate-400">Compliance:</span>
                                                                                     <span className={`font-mono px-1 rounded text-[10px] ${
                                                                                         debugPipelineTrace.output_contract.contract_compliant ? 'bg-green-100 text-green-800 dark:bg-green-950/20 dark:text-green-400' : 'bg-red-100 text-red-800 dark:bg-red-950/20 dark:text-red-400'
                                                                                     }`}>
                                                                                         {debugPipelineTrace.output_contract.contract_compliant ? 'PASSED' : 'FAILED'}
                                                                                     </span>
                                                                                 </div>
                                                                                 <div className="text-[9px] text-slate-500 space-y-0.5 border-t pt-1 mt-1">
                                                                                     <div>Length Check: {debugPipelineTrace.output_contract.checks?.length_compliant ? '✅' : '❌'} ({debugPipelineTrace.output_contract.approximate_tokens} tokens)</div>
                                                                                     <div>Citations: {debugPipelineTrace.output_contract.checks?.has_citations ? '✅' : '❌'}</div>
                                                                                     <div>Bullet structure: {debugPipelineTrace.output_contract.checks?.has_bullet_structure ? '✅' : '❌'}</div>
                                                                                 </div>
                                                                             </div>
                                                                         </div>
                                                                     )}

                                                                     {/* Compression Stats card */}
                                                                     {debugPipelineTrace.compression_stats && (
                                                                         <div className="bg-white dark:bg-slate-800 p-3 rounded border shadow-sm space-y-2 border-l-4 border-l-purple-500">
                                                                             <span className="text-[10px] uppercase text-muted-foreground font-bold flex items-center gap-1.5">
                                                                                 <Zap className="w-3.5 h-3.5 text-purple-500" />
                                                                                 Compression Stats
                                                                             </span>
                                                                             <div className="space-y-1">
                                                                                 <div className="flex justify-between">
                                                                                     <span className="text-slate-400">Ratio:</span>
                                                                                     <span className="font-semibold text-purple-600 dark:text-purple-400">{(debugPipelineTrace.compression_stats.compression_ratio * 100).toFixed(1)}%</span>
                                                                                 </div>
                                                                                 <div className="flex justify-between">
                                                                                     <span className="text-slate-400">Sentences kept:</span>
                                                                                     <span className="font-mono">{debugPipelineTrace.compression_stats.sentences_kept} / {debugPipelineTrace.compression_stats.sentences_kept + debugPipelineTrace.compression_stats.sentences_dropped}</span>
                                                                                 </div>
                                                                                 <div className="text-[10px] text-slate-550 dark:text-slate-400 text-right pt-1">
                                                                                     Dropped {debugPipelineTrace.compression_stats.sentences_dropped} sentences
                                                                                 </div>
                                                                             </div>
                                                                         </div>
                                                                     )}

                                                                     {/* Prompt Cache card */}
                                                                     {debugPipelineTrace.prompt_cache_info && (
                                                                         <div className="bg-white dark:bg-slate-800 p-3 rounded border shadow-sm space-y-2 border-l-4 border-l-indigo-500">
                                                                             <span className="text-[10px] uppercase text-muted-foreground font-bold flex items-center gap-1.5">
                                                                                 <Database className="w-3.5 h-3.5 text-indigo-500" />
                                                                                 Prompt Caching
                                                                             </span>
                                                                             <div className="space-y-1">
                                                                                 <div className="flex justify-between">
                                                                                     <span className="text-slate-400">Provider:</span>
                                                                                     <span className="font-semibold capitalize">{debugPipelineTrace.prompt_cache_info.provider}</span>
                                                                                 </div>
                                                                                 <div className="flex justify-between">
                                                                                     <span className="text-slate-400">Savings Rate:</span>
                                                                                     <span className="font-semibold text-emerald-600 dark:text-emerald-400">{debugPipelineTrace.prompt_cache_info.estimated_cache_savings_rate}</span>
                                                                                 </div>
                                                                                 <div className="flex justify-between text-[10px] text-slate-500 pt-1">
                                                                                     <span>Est. monthly:</span>
                                                                                     <span className="font-mono text-foreground font-semibold">${debugPipelineTrace.prompt_cache_info.estimated_monthly_savings_usd?.toFixed(2)}</span>
                                                                                 </div>
                                                                             </div>
                                                                         </div>
                                                                     )}

                                                                     {/* RAGAS Metrics card */}
                                                                     {debugPipelineTrace.ragas_metrics && (
                                                                         <div className="bg-white dark:bg-slate-800 p-3 rounded border shadow-sm space-y-2 border-l-4 border-l-amber-500 col-span-2">
                                                                             <span className="text-[10px] uppercase text-muted-foreground font-bold flex items-center gap-1.5">
                                                                                 <BarChart3 className="w-3.5 h-3.5 text-amber-500" />
                                                                                 RAGAS Inline Scores
                                                                             </span>
                                                                             <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-1">
                                                                                 {Object.entries(debugPipelineTrace.ragas_metrics).map(([key, val]: [string, any]) => {
                                                                                     if (key === 'overall_score' || key === 'hallucination_risk' || key === 'evaluation_method') return null;
                                                                                     return (
                                                                                         <div key={key} className="flex justify-between text-[11px]">
                                                                                             <span className="text-slate-500 capitalize">{key.replace('_', ' ')}:</span>
                                                                                             <span className="font-mono font-semibold">{val.score?.toFixed(2)} ({val.label})</span>
                                                                                         </div>
                                                                                     );
                                                                                 })}
                                                                                 <div className="col-span-2 border-t pt-1 mt-1 flex justify-between text-xs font-bold">
                                                                                     <span>Overall Score:</span>
                                                                                     <span className="text-indigo-600 dark:text-indigo-400">{debugPipelineTrace.ragas_metrics.overall_score?.toFixed(2)}</span>
                                                                                 </div>
                                                                             </div>
                                                                         </div>
                                                                     )}

                                                                     {/* History-Aware Rewrite */}
                                                                     {debugPipelineTrace.rewrite_info?.was_rewritten && (
                                                                         <div className="col-span-2 bg-white dark:bg-slate-800 p-2.5 rounded border shadow-sm border-l-4 border-l-blue-400">
                                                                             <span className="text-[10px] uppercase text-muted-foreground font-semibold">History-Aware Rewrite</span>
                                                                             <div className="mt-1 space-y-1">
                                                                                 <div className="text-[10px] text-slate-500 line-through">{debugPipelineTrace.rewrite_info.original_query}</div>
                                                                                 <div className="font-mono text-xs text-blue-700 dark:text-blue-400">{debugPipelineTrace.rewrite_info.rewritten_query}</div>
                                                                             </div>
                                                                         </div>
                                                                     )}

                                                                     {/* Hard Constraints */}
                                                                     {debugPipelineTrace.constraints_applied?.has_constraints && (
                                                                         <div className="col-span-2 bg-white dark:bg-slate-800 p-2.5 rounded border shadow-sm border-l-4 border-l-amber-450">
                                                                             <span className="text-[10px] uppercase text-muted-foreground font-semibold">Hard Constraints</span>
                                                                             <div className="mt-1 flex flex-wrap gap-1">
                                                                                 {debugPipelineTrace.constraints_applied.excluded_terms?.map((t: string, i: number) => (
                                                                                     <span key={`ex-${i}`} className="bg-rose-105 text-rose-800 px-1 rounded text-[10px] font-mono">NOT {t}</span>
                                                                                 ))}
                                                                                 {debugPipelineTrace.constraints_applied.doc_types?.map((t: string, i: number) => (
                                                                                     <span key={`dt-${i}`} className="bg-amber-105 text-amber-800 px-1 rounded text-[10px] font-mono">TYPE: {t}</span>
                                                                                 ))}
                                                                             </div>
                                                                         </div>
                                                                     )}
                                                                 </div>
                                                                )}

                                                                <div className="relative border-l-2 border-slate-200 dark:border-slate-700 ml-3 pl-4 space-y-4 text-left">
                                                                    {Object.entries(debugPipelineTrace.stages || {}).map(([sName, sTrace]: [string, any]) => (
                                                                        <div key={sName} className="relative text-left">
                                                                            {/* Stage Dot Indicator */}
                                                                            <div className={`absolute -left-[23px] top-1.5 w-3.5 h-3.5 rounded-full border-2 bg-white dark:bg-slate-900 ${
                                                                                sTrace.status === 'success' ? 'border-emerald-500' :
                                                                                sTrace.status === 'failed' ? 'border-rose-500' :
                                                                                sTrace.status === 'skipped' ? 'border-amber-400' :
                                                                                'border-slate-300'
                                                                            }`} />

                                                                            <div className="space-y-1 bg-white dark:bg-slate-800 p-2.5 rounded border shadow-sm text-left">
                                                                                <div className="flex justify-between items-center text-left">
                                                                                    <span className="font-bold capitalize text-slate-700 dark:text-slate-300 text-left">{sName.replace(/_/g, ' ')}</span>
                                                                                    <span className="font-mono text-muted-foreground text-[10px]">
                                                                                        {sTrace.duration_ms?.toFixed(1) || '0.0'}ms
                                                                                    </span>
                                                                                </div>
                                                                                <span className="text-[10px] text-muted-foreground uppercase block text-left">Status: {sTrace.status}</span>
                                                                                
                                                                                {sTrace.error_message && (
                                                                                    <p className="text-[10px] text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-950/20 p-1.5 rounded border border-rose-100 dark:border-rose-900/40 mt-1 font-mono break-words leading-tight text-left">
                                                                                        {sTrace.error_message}
                                                                                    </p>
                                                                                )}
                                                                                
                                                                                {sTrace.metadata && Object.keys(sTrace.metadata).length > 0 && (
                                                                                    <div className="mt-2 text-[10px] text-slate-500 font-mono bg-slate-50 dark:bg-slate-900/60 p-1.5 rounded border border-dashed border-slate-200 dark:border-slate-800 max-h-32 overflow-y-auto text-left">
                                                                                        {Object.entries(sTrace.metadata).map(([k, v]: [string, any]) => (
                                                                                            <div key={k} className="flex justify-between gap-2 border-b border-slate-100 dark:border-slate-800/40 last:border-0 py-0.5 text-left">
                                                                                                <span className="text-slate-400 text-left">{k}:</span>
                                                                                                <span className="text-right text-slate-600 dark:text-slate-400 truncate max-w-[200px]">{JSON.stringify(v)}</span>
                                                                                            </div>
                                                                                        ))}
                                                                                    </div>
                                                                                )}
                                                                            </div>
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <div className="text-center text-muted-foreground py-10">Run a search query to inspect stage timings.</div>
                                                        )}
                                                    </div>
                                                </TabsContent>
                                            </Tabs>
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
        </div>
    );
}
