"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Copy, KeyRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { createApiKey, errorMessage, getApiKeys, revokeApiKey, type ApiKeyInfo } from "@/lib/api";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function ApiKeysPanel({ projectId }: { projectId: number }) {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [name, setName] = useState("");
  const [fresh, setFresh] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setKeys(await getApiKeys(projectId));
    } catch (e) {
      toast.error(errorMessage(e, "Could not load API keys"));
    }
  }, [projectId]);

  useEffect(() => {
    let cancelled = false;
    getApiKeys(projectId)
      .then((k) => { if (!cancelled) { setKeys(k); setFresh(null); } })
      .catch((e) => toast.error(errorMessage(e, "Could not load API keys")));
    return () => { cancelled = true; };
  }, [projectId]);

  const create = async () => {
    try {
      const res = await createApiKey(projectId, name.trim() || "MCP client");
      setFresh(res.key);
      setName("");
      void load();
    } catch (e) {
      toast.error(errorMessage(e, "Could not create key"));
    }
  };

  const mcpUrl = `${API_URL.replace(/\/$/, "")}/mcp/`;
  const config = JSON.stringify(
    { mcpServers: { ragops: { type: "http", url: mcpUrl, headers: { Authorization: `Bearer ${fresh ?? "rgo_YOUR_KEY"}` } } } },
    null,
    2
  );
  const copy = (text: string) => navigator.clipboard.writeText(text).then(() => toast.success("Copied"));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><KeyRound className="h-5 w-5" />MCP &amp; API access</CardTitle>
        <CardDescription>
          Let Claude Desktop, Claude Code, Cursor or any MCP client use this project&apos;s knowledge base and databases.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Input placeholder="Key name, e.g. Claude Desktop" value={name} onChange={(e) => setName(e.target.value)} />
          <Button onClick={() => void create()}>Create key</Button>
        </div>

        {fresh && (
          <div className="space-y-2 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm dark:border-amber-800 dark:bg-amber-950/30">
            <p className="font-medium">Copy this key now. It won&apos;t be shown again.</p>
            <div className="flex items-center gap-2">
              <code className="min-w-0 flex-1 truncate rounded bg-background px-2 py-1 text-xs">{fresh}</code>
              <Button size="sm" variant="outline" onClick={() => void copy(fresh)}><Copy className="h-3.5 w-3.5" /></Button>
            </div>
          </div>
        )}

        <div className="space-y-1.5">
          <p className="text-xs text-muted-foreground">MCP client configuration (Streamable HTTP):</p>
          <pre className="overflow-x-auto rounded-lg bg-muted p-3 text-xs">{config}</pre>
          <Button size="sm" variant="ghost" onClick={() => void copy(config)}><Copy className="mr-1 h-3.5 w-3.5" />Copy config</Button>
          <p className="text-xs text-muted-foreground">
            Tools exposed: <code>ask</code>, <code>search_documents</code>, <code>list_documents</code>, <code>list_databases</code>, <code>query_database</code>.
          </p>
        </div>

        <ul className="divide-y rounded-lg border text-sm">
          {keys.length === 0 && <li className="px-3 py-2 text-muted-foreground">No keys yet.</li>}
          {keys.map((k) => (
            <li key={k.id} className="flex items-center justify-between gap-2 px-3 py-2">
              <span className="min-w-0 truncate">
                {k.name} <code className="text-xs text-muted-foreground">{k.prefix}…</code>
              </span>
              <span className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground">
                {k.revoked ? "revoked" : k.last_used_at ? `used ${new Date(k.last_used_at).toLocaleDateString()}` : "never used"}
                {!k.revoked && (
                  <Button size="sm" variant="ghost" onClick={async () => { await revokeApiKey(projectId, k.id); void load(); }}>
                    Revoke
                  </Button>
                )}
              </span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
