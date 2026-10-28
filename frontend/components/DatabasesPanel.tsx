"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Database, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  addDatabase,
  askDatabase,
  deleteDatabase,
  errorMessage,
  getDatabases,
  refreshDatabaseSchema,
  updateDatabase,
  type DbAnswer,
  type ExternalDb,
} from "@/lib/api";
import { DbAnswerView } from "@/components/DbAnswerView";

export function DatabasesPanel({ projectId }: { projectId: number }) {
  const [dbs, setDbs] = useState<ExternalDb[]>([]);
  const [form, setForm] = useState({ name: "", description: "", connection_string: "" });
  const [connecting, setConnecting] = useState(false);
  const [testQ, setTestQ] = useState("");
  const [testing, setTesting] = useState<number | null>(null);
  const [answer, setAnswer] = useState<DbAnswer | null>(null);

  const load = useCallback(async () => {
    try {
      setDbs(await getDatabases(projectId));
    } catch (e) {
      toast.error(errorMessage(e, "Could not load databases"));
    }
  }, [projectId]);

  useEffect(() => {
    void load();
  }, [load]);

  const connect = async () => {
    setConnecting(true);
    try {
      await addDatabase(projectId, form);
      toast.success("Connected. Now choose which tables employees may query.");
      setForm({ name: "", description: "", connection_string: "" });
      void load();
    } catch (e) {
      toast.error(errorMessage(e, "Could not connect"));
    } finally {
      setConnecting(false);
    }
  };

  const toggleTable = async (db: ExternalDb, table: string, on: boolean) => {
    const allowed = on ? [...db.allowed_tables, table] : db.allowed_tables.filter((t) => t !== table);
    try {
      const updated = await updateDatabase(projectId, db.id, { allowed_tables: allowed });
      setDbs((list) => list.map((d) => (d.id === db.id ? updated : d)));
    } catch (e) {
      toast.error(errorMessage(e, "Could not update tables"));
    }
  };

  const test = async (db: ExternalDb) => {
    if (!testQ.trim()) return;
    setTesting(db.id);
    setAnswer(null);
    try {
      setAnswer(await askDatabase(projectId, db.id, testQ));
    } catch (e) {
      toast.error(errorMessage(e, "Query failed"));
    } finally {
      setTesting(null);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Database className="h-5 w-5" />Connected databases</CardTitle>
        <CardDescription>
          Connect a company PostgreSQL or MySQL database. Employees with access to this project can then ask questions in
          plain language. Use a <strong>read-only</strong> database user; RAGOps also enforces read-only queries, a table
          allow-list, timeouts and row limits.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2 rounded-lg border p-3">
          <Label>New connection</Label>
          <div className="grid gap-2 sm:grid-cols-2">
            <Input placeholder="Name, e.g. Sales DB" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            <Input placeholder="postgresql://readonly_user:password@host:5432/dbname" value={form.connection_string}
              onChange={(e) => setForm({ ...form, connection_string: e.target.value })} />
          </div>
          <Textarea rows={2} placeholder="What's in this database? (helps the AI write correct queries, e.g. 'amount is in USD; status is paid/refunded/pending')"
            value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
          <Button disabled={!form.connection_string || connecting} onClick={() => void connect()}>
            {connecting ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : null}Test &amp; connect
          </Button>
        </div>

        {dbs.map((db) => (
          <div key={db.id} className="space-y-3 rounded-lg border p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="font-medium">{db.name} <span className="text-xs text-muted-foreground">({db.db_type} · {db.host}/{db.db_name})</span></p>
                {db.description && <p className="text-xs text-muted-foreground">{db.description}</p>}
              </div>
              <div className="flex gap-1">
                <Button size="sm" variant="ghost" onClick={async () => {
                  try { await refreshDatabaseSchema(projectId, db.id); toast.success("Schema refreshed"); void load(); }
                  catch (e) { toast.error(errorMessage(e, "Refresh failed")); }
                }}><RefreshCw className="mr-1 h-3.5 w-3.5" />Refresh schema</Button>
                <Button size="sm" variant="ghost" onClick={async () => { await deleteDatabase(projectId, db.id); void load(); }}>
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div>
              <p className="mb-1 text-xs font-medium">Tables employees may query ({db.allowed_tables.length} of {Object.keys(db.tables || {}).length})</p>
              <div className="grid max-h-56 gap-1 overflow-y-auto sm:grid-cols-2">
                {Object.entries(db.tables || {}).map(([table, cols]) => (
                  <label key={table} className="flex items-start gap-2 rounded px-2 py-1 text-sm hover:bg-muted/50">
                    <Checkbox checked={db.allowed_tables.includes(table)} onCheckedChange={(c) => void toggleTable(db, table, c === true)} />
                    <span className="min-w-0">
                      <span className="font-mono">{table}</span>
                      <span className="block truncate text-[11px] text-muted-foreground">{cols.map((c) => c.column).join(", ")}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <div className="flex gap-2">
              <Input placeholder="Try a question, e.g. total revenue by month this year" value={testQ} onChange={(e) => setTestQ(e.target.value)} />
              <Button variant="outline" disabled={testing === db.id || !db.allowed_tables.length} onClick={() => void test(db)}>
                {testing === db.id ? <Loader2 className="h-4 w-4 animate-spin" /> : "Ask"}
              </Button>
            </div>
            {answer && testing === null && <DbAnswerView answer={answer} />}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
