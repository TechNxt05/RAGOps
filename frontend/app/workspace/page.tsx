"use client";

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { ArrowLeft, Copy, KeyRound, Link2, Trash2, UserPlus, Users } from 'lucide-react';
import { useAuth } from '@/lib/auth-context';
import {
    createInvite, createUser, deleteInvite, deleteUser, errorMessage, getWorkspace, inviteLink,
    regenerateInviteLink, renameWorkspace, setMemberProjects, updateWorkspaceKeys,
    type LlmProvider, type WorkspaceInfo,
} from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

const selectCls = "flex h-9 rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm";

const PROVIDERS: { id: LlmProvider; label: string; hint: string }[] = [
    { id: 'groq', label: 'Groq', hint: 'console.groq.com/keys' },
    { id: 'google', label: 'Gemini', hint: 'aistudio.google.com/apikey' },
    { id: 'openrouter', label: 'OpenRouter', hint: 'openrouter.ai/keys' },
];

function ProjectPicker({ projects, value, onChange }: {
    projects: WorkspaceInfo['projects'];
    value: number[];
    onChange: (ids: number[]) => void;
}) {
    if (projects.length === 0) return <span className="text-xs text-muted-foreground">No projects yet</span>;
    return (
        <div className="flex flex-wrap gap-x-4 gap-y-1">
            {projects.map((p) => (
                <label key={p.id} className="flex items-center gap-1.5 text-sm">
                    <input
                        type="checkbox"
                        className="h-4 w-4 accent-indigo-600"
                        checked={value.includes(p.id)}
                        onChange={(e) => onChange(e.target.checked ? [...value, p.id] : value.filter((x) => x !== p.id))}
                    />
                    {p.name}
                    {p.visibility === 'all_users' && <span className="text-xs text-muted-foreground">(everyone)</span>}
                </label>
            ))}
        </div>
    );
}

function CopyLink({ token }: { token: string }) {
    const url = inviteLink(token);
    return (
        <div className="flex gap-2">
            <Input readOnly value={url} className="font-mono text-xs" />
            <Button size="icon" variant="outline" aria-label="Copy invite link"
                onClick={() => { void navigator.clipboard.writeText(url); toast.success('Link copied'); }}>
                <Copy className="h-4 w-4" />
            </Button>
        </div>
    );
}

export default function WorkspacePage() {
    const { user, isLoading } = useAuth();
    const router = useRouter();
    const [ws, setWs] = useState<WorkspaceInfo | null>(null);
    const [name, setName] = useState('');
    const [keyDraft, setKeyDraft] = useState<Record<LlmProvider, string>>({ groq: '', google: '', openrouter: '' });
    const [savingKey, setSavingKey] = useState<LlmProvider | null>(null);
    const [invite, setInvite] = useState({ email: '', role: 'client' as 'client' | 'admin', projectIds: [] as number[] });
    const [freshLinks, setFreshLinks] = useState<Record<string, string>>({});
    const [account, setAccount] = useState({ email: '', password: '', role: 'client' as 'client' | 'admin' });

    useEffect(() => {
        if (!isLoading && (!user || user.role !== 'admin')) router.push(user ? '/chat' : '/login');
    }, [user, isLoading, router]);

    const load = useCallback(async () => {
        try {
            const data = await getWorkspace();
            setWs(data);
            setName(data.name);
        } catch (e) {
            toast.error(errorMessage(e, 'Could not load the workspace'));
        }
    }, []);

    useEffect(() => {
        if (user?.role === 'admin') void load();
    }, [user, load]);

    if (!ws) return (
        <div className="flex h-screen items-center justify-center bg-slate-50 dark:bg-slate-950">
            <div className="h-32 w-64 animate-pulse rounded-xl bg-muted" />
        </div>
    );

    const saveKey = async (provider: LlmProvider, value: string) => {
        setSavingKey(provider);
        try {
            setWs(await updateWorkspaceKeys({ [provider]: value }));
            setKeyDraft({ ...keyDraft, [provider]: '' });
            toast.success(value ? 'Key saved' : 'Key removed');
        } catch (e) {
            toast.error(errorMessage(e, 'Could not save the key'));
        } finally {
            setSavingKey(null);
        }
    };

    const sendInvite = async () => {
        try {
            const res = await createInvite(invite.email.trim(), invite.role, invite.projectIds);
            setFreshLinks({ ...freshLinks, [res.email]: res.token });
            setInvite({ email: '', role: 'client', projectIds: [] });
            toast.success('Invite created: copy the link and send it');
            void load();
        } catch (e) {
            toast.error(errorMessage(e, 'Could not create the invite'));
        }
    };

    const memberProjects = async (userId: number, ids: number[]) => {
        try {
            setWs(await setMemberProjects(userId, ids));
        } catch (e) {
            toast.error(errorMessage(e, 'Could not update access'));
        }
    };

    const membersOnly = ws.projects.filter((p) => p.visibility === 'members_only').length;

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
            <div className="mx-auto max-w-4xl space-y-6 px-4 py-6 md:py-10">
                <div className="flex items-center gap-3">
                    <Button variant="ghost" size="icon" asChild aria-label="Back to projects">
                        <Link href="/admin"><ArrowLeft className="h-5 w-5" /></Link>
                    </Button>
                    <div className="min-w-0">
                        <h1 className="truncate text-2xl font-bold">{ws.name}</h1>
                        <p className="text-sm text-muted-foreground">Workspace settings</p>
                    </div>
                </div>

                {ws.is_demo && (
                    <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
                        This is the shared public demo. It runs on the site owner&apos;s API keys, so keys, invites and
                        account changes are turned off here. Create your own workspace from the sign-in page to use them.
                    </div>
                )}

                <Card>
                    <CardHeader>
                        <CardTitle>General</CardTitle>
                    </CardHeader>
                    <CardContent className="flex flex-col gap-2 sm:flex-row">
                        <Input value={name} onChange={(e) => setName(e.target.value)} disabled={ws.is_demo} />
                        <Button variant="outline" disabled={ws.is_demo || !name.trim() || name === ws.name}
                            onClick={async () => {
                                try { setWs(await renameWorkspace(name.trim())); toast.success('Renamed'); }
                                catch (e) { toast.error(errorMessage(e, 'Rename failed')); }
                            }}>Rename</Button>
                    </CardContent>
                    {ws.limits && (
                        <CardContent className="pt-0 text-xs text-muted-foreground">
                            Free plan limits: {ws.limits.projects} projects, {ws.limits.documents} documents, {ws.limits.members} members.
                        </CardContent>
                    )}
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2"><KeyRound className="h-5 w-5" /> LLM API keys</CardTitle>
                        <CardDescription>
                            {ws.uses_platform_keys
                                ? "This workspace uses the server's own keys, set in the hosting environment."
                                : 'Every question your team asks uses these keys. One provider is enough; add more for automatic fallback when one is rate-limited. Keys are encrypted at rest and never shown again in full.'}
                        </CardDescription>
                    </CardHeader>
                    {!ws.uses_platform_keys && (
                        <CardContent className="space-y-4">
                            {PROVIDERS.map((p) => {
                                const current = ws.llm_keys[p.id];
                                return (
                                    <div key={p.id} className="space-y-1">
                                        <Label className="text-sm">
                                            {p.label} <span className="text-xs text-muted-foreground">({p.hint})</span>
                                            {current
                                                ? <span className="ml-2 font-mono text-xs text-emerald-600">{current}</span>
                                                : <span className="ml-2 text-xs text-muted-foreground">not set</span>}
                                        </Label>
                                        <div className="flex flex-col gap-2 sm:flex-row">
                                            <Input type="password" autoComplete="off" placeholder={current ? 'Paste a new key to replace' : 'Paste key'}
                                                value={keyDraft[p.id]} onChange={(e) => setKeyDraft({ ...keyDraft, [p.id]: e.target.value })} />
                                            <div className="flex gap-2">
                                                <Button disabled={!keyDraft[p.id].trim() || savingKey === p.id}
                                                    onClick={() => void saveKey(p.id, keyDraft[p.id].trim())}>
                                                    {savingKey === p.id ? 'Checking…' : 'Save'}
                                                </Button>
                                                {current && (
                                                    <Button variant="outline" disabled={savingKey === p.id}
                                                        onClick={() => void saveKey(p.id, '')}>Remove</Button>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </CardContent>
                    )}
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2"><Users className="h-5 w-5" /> Team &amp; project access</CardTitle>
                        <CardDescription>
                            Clients see only the projects ticked for them. Projects shared with everyone (set per project
                            under Access &amp; Integrations) are visible to the whole workspace.
                            {membersOnly === 0 && ws.projects.length > 0 && ' All your projects are currently shared with everyone.'}
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        {!ws.is_demo && (
                            <div className="space-y-3 rounded-lg border p-3">
                                <Label className="flex items-center gap-2"><UserPlus className="h-4 w-4" /> Invite someone</Label>
                                <div className="flex flex-col gap-2 sm:flex-row">
                                    <Input type="email" placeholder="sales@company.com" value={invite.email}
                                        onChange={(e) => setInvite({ ...invite, email: e.target.value })} />
                                    <select className={selectCls} value={invite.role}
                                        onChange={(e) => setInvite({ ...invite, role: e.target.value as 'client' | 'admin' })}>
                                        <option value="client">Client</option>
                                        <option value="admin">Admin</option>
                                    </select>
                                    <Button disabled={!invite.email.includes('@')} onClick={() => void sendInvite()}>Create invite</Button>
                                </div>
                                {invite.role === 'client' && (
                                    <div className="space-y-1">
                                        <p className="text-xs text-muted-foreground">Projects they can open:</p>
                                        <ProjectPicker projects={ws.projects} value={invite.projectIds}
                                            onChange={(ids) => setInvite({ ...invite, projectIds: ids })} />
                                    </div>
                                )}
                            </div>
                        )}

                        <div className="space-y-2">
                            <Label>Members</Label>
                            <ul className="divide-y rounded-lg border">
                                {ws.members.map((m) => (
                                    <li key={m.id} className="space-y-2 px-3 py-2.5">
                                        <div className="flex items-center justify-between gap-2">
                                            <span className="min-w-0 truncate text-sm">
                                                {m.email}
                                                <span className="ml-2 text-xs capitalize text-muted-foreground">
                                                    {m.is_owner ? 'owner' : m.role}{m.id === user?.id ? ' (you)' : ''}
                                                </span>
                                            </span>
                                            {!ws.is_demo && !m.is_owner && m.id !== user?.id && (
                                                <Button size="icon" variant="ghost" aria-label={`Remove ${m.email}`} onClick={async () => {
                                                    if (!window.confirm(`Delete the account ${m.email}? Their chat history stays in analytics.`)) return;
                                                    try { await deleteUser(m.id); void load(); }
                                                    catch (e) { toast.error(errorMessage(e, 'Delete failed')); }
                                                }}>
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            )}
                                        </div>
                                        {m.role === 'client' && (
                                            <ProjectPicker projects={ws.projects} value={m.project_ids}
                                                onChange={(ids) => void memberProjects(m.id, ids)} />
                                        )}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {ws.invites.length > 0 && (
                            <div className="space-y-2">
                                <Label>Pending invites</Label>
                                <ul className="divide-y rounded-lg border">
                                    {ws.invites.map((i) => (
                                        <li key={i.id} className="space-y-2 px-3 py-2.5">
                                            <div className="flex items-center justify-between gap-2">
                                                <span className="min-w-0 truncate text-sm">
                                                    {i.email}
                                                    <span className="ml-2 text-xs text-muted-foreground">
                                                        {i.role} · {i.project_ids.length
                                                            ? ws.projects.filter((p) => i.project_ids.includes(p.id)).map((p) => p.name).join(', ')
                                                            : 'no projects yet'} · expires in {i.expires_in_days}d
                                                    </span>
                                                </span>
                                                <span className="flex shrink-0 gap-1">
                                                    <Button size="icon" variant="ghost" aria-label={`New link for ${i.email}`} title="New link"
                                                        onClick={async () => {
                                                            try {
                                                                const r = await regenerateInviteLink(i.id);
                                                                setFreshLinks({ ...freshLinks, [r.email]: r.token });
                                                                void load();
                                                            } catch (e) { toast.error(errorMessage(e, 'Could not create a link')); }
                                                        }}>
                                                        <Link2 className="h-4 w-4" />
                                                    </Button>
                                                    <Button size="icon" variant="ghost" aria-label={`Cancel invite for ${i.email}`}
                                                        onClick={async () => { await deleteInvite(i.id); void load(); }}>
                                                        <Trash2 className="h-4 w-4" />
                                                    </Button>
                                                </span>
                                            </div>
                                            {freshLinks[i.email] && <CopyLink token={freshLinks[i.email]} />}
                                        </li>
                                    ))}
                                </ul>
                                <p className="text-xs text-muted-foreground">
                                    Links are shown once. Lost one? Click the link icon to issue a new one (the old link stops working).
                                </p>
                            </div>
                        )}

                        {!ws.is_demo && (
                            <details className="rounded-lg border p-3">
                                <summary className="cursor-pointer text-sm font-medium">Create an account with a password instead</summary>
                                <p className="mt-2 text-xs text-muted-foreground">
                                    For people who can&apos;t use an invite link. Share the temporary password with them securely.
                                </p>
                                <div className="mt-2 grid gap-2 sm:grid-cols-[1fr_1fr_auto_auto]">
                                    <Input placeholder="email" value={account.email} onChange={(e) => setAccount({ ...account, email: e.target.value })} />
                                    <Input type="password" placeholder="temporary password (8+ chars)" value={account.password}
                                        onChange={(e) => setAccount({ ...account, password: e.target.value })} />
                                    <select className={selectCls} value={account.role}
                                        onChange={(e) => setAccount({ ...account, role: e.target.value as 'client' | 'admin' })}>
                                        <option value="client">Client</option>
                                        <option value="admin">Admin</option>
                                    </select>
                                    <Button variant="outline" disabled={!account.email || account.password.length < 8} onClick={async () => {
                                        try {
                                            await createUser(account.email.trim(), account.password, account.role);
                                            toast.success(`Account created for ${account.email.trim()}`);
                                            setAccount({ email: '', password: '', role: 'client' });
                                            void load();
                                        } catch (e) { toast.error(errorMessage(e, 'Could not create account')); }
                                    }}>Create</Button>
                                </div>
                            </details>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
