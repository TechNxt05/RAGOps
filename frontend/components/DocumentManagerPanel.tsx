"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  getDocuments,
  getDocumentStatus,
  deleteDocument,
  rechunkDocument,
  getDocumentChunksPaged,
  debugSearch,
  type Document,
  type Chunk,
  type DebugSearchResult,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { toast } from "sonner";
import { Eye, FileText, RefreshCw, Trash2 } from "lucide-react";

function formatBytes(n?: number | null) {
  if (n == null || n <= 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocumentManagerPanel({
  projectId,
  topK,
  similarityThreshold,
  lastUploadedDocId,
}: {
  projectId: number;
  topK: number;
  similarityThreshold: number;
  lastUploadedDocId?: number | null;
}) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [previewDoc, setPreviewDoc] = useState<Document | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [chunkPage, setChunkPage] = useState(1);
  const [chunkTotalPages, setChunkTotalPages] = useState(1);
  const [retrievalQuery, setRetrievalQuery] = useState("");
  const [retrievalResults, setRetrievalResults] = useState<DebugSearchResult[]>([]);
  const [rechunkTarget, setRechunkTarget] = useState<Document | null>(null);
  const [rechunkSize, setRechunkSize] = useState(512);
  const [rechunkOverlap, setRechunkOverlap] = useState(50);
  const [deleteTarget, setDeleteTarget] = useState<Document | null>(null);
  const [polling, setPolling] = useState<Record<number, string>>({});

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getDocuments(projectId, true);
      setDocuments(data);
    } catch {
      toast.error("Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (lastUploadedDocId != null && lastUploadedDocId > 0) {
      setPolling((p) => ({ ...p, [lastUploadedDocId]: "pending" }));
    }
  }, [lastUploadedDocId]);

  useEffect(() => {
    const timers: number[] = [];
    Object.entries(polling).forEach(([idStr, _]) => {
      const id = Number(idStr);
      const t = window.setInterval(async () => {
        try {
          const st = await getDocumentStatus(id);
          if (st.status === "complete" || st.status === "failed") {
            setPolling((p) => {
              const next = { ...p };
              delete next[id];
              return next;
            });
            window.clearInterval(t);
            void refresh();
            if (st.status === "failed") toast.error(st.error || "Document processing failed");
            else toast.success("Document ready");
          }
        } catch {
          /* ignore */
        }
      }, 2000);
      timers.push(t);
    });
    return () => timers.forEach((t) => window.clearInterval(t));
  }, [polling, refresh]);

  const toggleSelect = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const openPreview = async (doc: Document) => {
    setPreviewDoc(doc);
    setChunkPage(1);
    try {
      const page = await getDocumentChunksPaged(doc.id, 1, 15);
      setChunks(page.chunks);
      setChunkTotalPages(page.pages);
    } catch {
      toast.error("Could not load chunks");
    }
  };

  const loadChunkPage = async (docId: number, page: number) => {
    try {
      const data = await getDocumentChunksPaged(docId, page, 15);
      setChunks(data.chunks);
      setChunkPage(data.page);
      setChunkTotalPages(data.pages);
    } catch {
      toast.error("Chunk page failed");
    }
  };

  const runRetrievalTest = async () => {
    if (!retrievalQuery.trim()) return;
    try {
      const res = await debugSearch({
        project_id: projectId,
        query: retrievalQuery,
        top_k: topK,
        similarity_threshold: similarityThreshold,
      });
      setRetrievalResults(res.results);
    } catch {
      toast.error("Retrieval test failed");
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteDocument(deleteTarget.id);
      toast.success("Document removed from knowledge base");
      setDeleteTarget(null);
      void refresh();
    } catch {
      toast.error("Delete failed");
    }
  };

  const handleRechunk = async () => {
    if (!rechunkTarget) return;
    const id = rechunkTarget.id;
    try {
      await rechunkDocument(id, rechunkSize, rechunkOverlap);
      toast.info("Re-chunking started…");
      setRechunkTarget(null);
      setPolling((p) => ({ ...p, [id]: "processing" }));
      void refresh();
    } catch {
      toast.error("Re-chunk failed");
    }
  };

  const bulkDelete = async () => {
    if (selected.size === 0) return;
    for (const id of selected) {
      try {
        await deleteDocument(id);
      } catch {
        toast.error(`Failed to delete document ${id}`);
      }
    }
    setSelected(new Set());
    void refresh();
    toast.success("Bulk delete completed");
  };

  if (loading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-28 animate-pulse rounded-lg bg-muted" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {selected.size > 0 && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg border bg-muted/40 px-4 py-2 text-sm">
          <span>{selected.size} selected</span>
          <Button size="sm" variant="destructive" onClick={() => void bulkDelete()}>
            Delete selected
          </Button>
        </div>
      )}

      {documents.length === 0 ? (
        <div className="rounded-xl border border-dashed py-16 text-center text-muted-foreground">
          <FileText className="mx-auto mb-3 h-12 w-12 opacity-40" />
          <h3 className="text-lg font-medium text-foreground">No documents yet</h3>
          <p className="mt-1 text-sm">Upload PDF or TXT files to start building your knowledge base.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {documents.map((doc) => {
            const active = doc.is_active !== false;
            const busy = polling[doc.id] != null;
            return (
              <div
                key={doc.id}
                className="flex flex-col gap-3 rounded-xl border bg-card p-4 shadow-sm sm:flex-row sm:items-start"
              >
                <div className="flex items-start gap-3">
                  <Checkbox
                    checked={selected.has(doc.id)}
                    onCheckedChange={() => toggleSelect(doc.id)}
                    aria-label={`Select ${doc.filename}`}
                  />
                  <div
                    className={`mt-0.5 rounded-md p-2 ${
                      doc.processing_status === "complete" || doc.processed
                        ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                        : "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200"
                    }`}
                  >
                    <FileText className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium">{doc.filename}</span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] uppercase ${
                          active ? "bg-emerald-100 text-emerald-800" : "bg-slate-200 text-slate-600"
                        }`}
                      >
                        {active ? "Active" : "Inactive"}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {new Date(doc.uploaded_at).toLocaleDateString()} · {formatBytes(doc.file_size_bytes)}
                      {doc.page_count != null ? ` · ${doc.page_count} pages` : ""}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {doc.chunk_count != null ? `${doc.chunk_count} chunks` : "—"} · chunk_size=
                      {doc.chunk_size_used ?? "—"} · {doc.embedding_model_used ?? "embeddings —"}
                      {doc.chunking_strategy ? ` · strategy=${doc.chunking_strategy}` : ""}
                    </p>
                    {busy && (
                      <div className="mt-2 space-y-1">
                        <div className="flex justify-between text-[10px] text-muted-foreground">
                          <span>Processing</span>
                          <span>{doc.processing_status}</span>
                        </div>
                        <motion.div
                          className="h-1.5 overflow-hidden rounded-full bg-muted"
                          initial={false}
                          animate={{ opacity: [0.6, 1, 0.6] }}
                          transition={{ repeat: Infinity, duration: 1.6 }}
                        >
                          <motion.div
                            className="h-full rounded-full bg-indigo-500"
                            initial={{ width: "15%" }}
                            animate={{ width: ["15%", "85%", "40%"] }}
                            transition={{ repeat: Infinity, duration: 2.4 }}
                          />
                        </motion.div>
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 sm:ml-auto">
                  <Button type="button" size="sm" variant="outline" onClick={() => void openPreview(doc)}>
                    <Eye className="mr-1 h-3.5 w-3.5" />
                    Preview
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={!active || busy}
                    onClick={() => setRechunkTarget(doc)}
                  >
                    <RefreshCw className="mr-1 h-3.5 w-3.5" />
                    Re-chunk
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    disabled={!active}
                    onClick={() => setDeleteTarget(doc)}
                  >
                    <Trash2 className="mr-1 h-3.5 w-3.5" />
                    Delete
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <Dialog open={!!previewDoc} onOpenChange={(o) => !o && setPreviewDoc(null)}>
        <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{previewDoc?.filename}</DialogTitle>
            <DialogDescription>Preview, chunks, and retrieval test for this document.</DialogDescription>
          </DialogHeader>
          <Tabs defaultValue="text">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="text">Full text</TabsTrigger>
              <TabsTrigger value="chunks">Chunks</TabsTrigger>
              <TabsTrigger value="adaptive">Adaptive Quality</TabsTrigger>
              <TabsTrigger value="retrieval">Retrieval test</TabsTrigger>
            </TabsList>
            <TabsContent value="text">
              <ScrollArea className="mt-2 h-64 rounded-md border p-3 text-xs font-mono whitespace-pre-wrap">
                {previewDoc?.content || "No text stored."}
              </ScrollArea>
            </TabsContent>
            <TabsContent value="chunks">
              <div className="mt-2 flex items-center justify-between gap-2">
                <span className="text-xs text-muted-foreground">
                  Page {chunkPage} / {chunkTotalPages}
                </span>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={chunkPage <= 1 || !previewDoc}
                    onClick={() => previewDoc && void loadChunkPage(previewDoc.id, chunkPage - 1)}
                  >
                    Prev
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={chunkPage >= chunkTotalPages || !previewDoc}
                    onClick={() => previewDoc && void loadChunkPage(previewDoc.id, chunkPage + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
              <ScrollArea className="mt-2 h-64 space-y-2">
                {chunks.map((c, i) => (
                  <div key={c.id} className="mb-2 rounded border p-2 text-xs">
                    <div className="mb-1 font-semibold text-muted-foreground">Chunk #{i + 1}</div>
                    <p className="whitespace-pre-wrap">{c.content}</p>
                  </div>
                ))}
              </ScrollArea>
            </TabsContent>
            <TabsContent value="adaptive">
              <ScrollArea className="mt-2 h-64 space-y-4">
                {previewDoc?.chunking_strategy ? (
                  <div className="space-y-4 p-2 text-left">
                    <div className="flex justify-between items-center bg-indigo-50 dark:bg-indigo-950/40 p-3 rounded-lg border border-indigo-100 dark:border-indigo-900/60">
                      <div>
                        <span className="text-xs uppercase text-indigo-500 font-semibold tracking-wider block text-left">Selected Strategy</span>
                        <span className="font-bold text-sm text-indigo-700 dark:text-indigo-300 block text-left capitalize">{previewDoc.chunking_strategy.replace(/_/g, ' ')}</span>
                      </div>
                      <div className="text-right">
                        <span className="text-xs uppercase text-indigo-500 font-semibold tracking-wider block text-right">Composite Quality</span>
                        <span className="font-extrabold text-lg text-indigo-700 dark:text-indigo-300">
                          {((previewDoc.chunking_metrics?.composite_score ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div className="space-y-3 text-left">
                      <h4 className="font-semibold text-xs text-muted-foreground uppercase tracking-wider text-left">Ekimetrics Intrinsic Metrics</h4>
                      
                      {[
                        { label: "Size Compliance (SC)", value: previewDoc.chunking_metrics?.size_compliance, desc: "Fraction of chunks within embedder token limits" },
                        { label: "Intrachunk Cohesion (ICC)", value: previewDoc.chunking_metrics?.intrachunk_cohesion, desc: "Semantic theme focus within each individual chunk" },
                        { label: "Contextual Coherence (DCC)", value: previewDoc.chunking_metrics?.contextual_coherence, desc: "Smoothness of flow/overlap between adjacent chunks" },
                        { label: "Block Integrity (BI)", value: previewDoc.chunking_metrics?.block_integrity, desc: "Ensuring tables, lists, and code blocks are kept intact" },
                        { label: "Reference Completeness (RC)", value: previewDoc.chunking_metrics?.reference_completeness, desc: "Keeping pronoun-antecedent entity pairs in same chunk" }
                      ].map((m, idx) => (
                        <div key={idx} className="space-y-1 text-left">
                          <div className="flex justify-between text-xs font-medium">
                            <span className="text-foreground">{m.label}</span>
                            <span className="font-mono text-muted-foreground">{((m.value ?? 0) * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full" 
                              style={{ width: `${Math.max(0, Math.min(100, (m.value ?? 0) * 100))}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-muted-foreground leading-none block text-left">{m.desc}</span>
                        </div>
                      ))}
                    </div>

                    {previewDoc.chunking_metrics?.all_strategy_scores && (
                      <div className="space-y-2 pt-2 text-left">
                        <h4 className="font-semibold text-xs text-muted-foreground uppercase tracking-wider text-left">Candidate Strategies Comparison</h4>
                        <div className="border rounded-md overflow-hidden text-xs">
                          <table className="w-full text-left border-collapse">
                            <thead>
                              <tr className="bg-muted text-muted-foreground font-medium text-[10px] uppercase border-b">
                                <th className="p-2">Strategy</th>
                                <th className="p-2 text-center">Chunks</th>
                                <th className="p-2 text-right">Composite Score</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(previewDoc.chunking_metrics.all_strategy_scores).map(([sName, sData], sIdx) => (
                                <tr key={sIdx} className={`border-b last:border-0 ${sName === previewDoc.chunking_strategy ? 'bg-indigo-50/30 dark:bg-indigo-950/20 font-semibold text-indigo-600 dark:text-indigo-400' : 'text-muted-foreground'}`}>
                                  <td className="p-2 capitalize">{sName.replace(/_/g, ' ')}</td>
                                  <td className="p-2 text-center">{sData.chunk_count}</td>
                                  <td className="p-2 text-right font-mono">{(sData.composite * 100).toFixed(1)}%</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-10 text-muted-foreground text-xs space-y-2">
                    <p className="font-medium text-foreground">Legacy Index Segment</p>
                    <p>This document was indexed before Adaptive Chunking was deployed. Click "Re-chunk" to apply the new multi-candidate structural evaluator!</p>
                  </div>
                )}
              </ScrollArea>
            </TabsContent>
            <TabsContent value="retrieval">
              <div className="mt-2 flex gap-2">
                <Input
                  placeholder="Test query…"
                  value={retrievalQuery}
                  onChange={(e) => setRetrievalQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && void runRetrievalTest()}
                />
                <Button type="button" onClick={() => void runRetrievalTest()}>
                  Run
                </Button>
              </div>
              <div className="mt-3 space-y-2">
                {retrievalResults.map((r, idx) => (
                  <div key={idx} className="rounded border p-2 text-xs">
                    <div className="mb-1 flex justify-between font-medium">
                      <span>{r.document_name}</span>
                      <span>{(r.score * 100).toFixed(1)}%</span>
                    </div>
                    <p className="line-clamp-4 text-muted-foreground">{r.content}</p>
                  </div>
                ))}
                {retrievalResults.length === 0 && (
                  <p className="text-center text-xs text-muted-foreground">Run a query to see retrieved chunks.</p>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>

      <Dialog open={!!rechunkTarget} onOpenChange={(o) => !o && setRechunkTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Re-chunk document</DialogTitle>
            <DialogDescription>
              Current: {rechunkTarget?.chunk_count ?? "?"} chunks (size={rechunkTarget?.chunk_size_used ?? "—"}).
              This rebuilds vectors for the whole index segment and may take a minute.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div>
              <Label>Chunk size: {rechunkSize}</Label>
              <Slider
                value={[rechunkSize]}
                min={256}
                max={1536}
                step={64}
                onValueChange={(v) => setRechunkSize(v[0])}
              />
            </div>
            <div>
              <Label>Overlap: {rechunkOverlap}</Label>
              <Slider
                value={[rechunkOverlap]}
                min={0}
                max={256}
                step={8}
                onValueChange={(v) => setRechunkOverlap(v[0])}
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setRechunkTarget(null)}>
              Cancel
            </Button>
            <Button type="button" onClick={() => void handleRechunk()}>
              Re-chunk
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={!!deleteTarget} onOpenChange={(o) => !o && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {deleteTarget?.filename}?</AlertDialogTitle>
            <AlertDialogDescription>
              This soft-deletes the document and rebuilds the vector index so it no longer appears in retrieval.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => void handleDelete()}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
