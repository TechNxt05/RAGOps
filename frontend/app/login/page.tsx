"use client";

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import api, { demoLogin, errorMessage, getDemoStatus, getInviteInfo, inviteLink, register, type RegisterResult } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Copy, KeyRound, ShieldCheck, UserRound } from 'lucide-react';
import { toast } from 'sonner';

const KEY_FIELDS = [
    { id: 'groq', label: 'Groq API key', hint: 'console.groq.com/keys', placeholder: 'gsk_…' },
    { id: 'google', label: 'Gemini API key', hint: 'aistudio.google.com/apikey', placeholder: 'AIza…' },
    { id: 'openrouter', label: 'OpenRouter API key', hint: 'openrouter.ai/keys', placeholder: 'sk-or-…' },
] as const;

function LoginInner() {
    const { login } = useAuth();
    const params = useSearchParams();
    const inviteToken = params.get('invite') || '';

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [busy, setBusy] = useState(false);
    const [demo, setDemo] = useState<{ enabled: boolean; reset_hours: number } | null>(null);

    // Create workspace
    const [wsName, setWsName] = useState('');
    const [keys, setKeys] = useState<Record<string, string>>({ groq: '', google: '', openrouter: '' });
    const [clientEmails, setClientEmails] = useState('');
    const [created, setCreated] = useState<RegisterResult | null>(null);

    // Join via invite
    const [invite, setInvite] = useState<{ email: string; workspace_name: string | null } | null>(null);
    const [inviteError, setInviteError] = useState('');

    useEffect(() => {
        getDemoStatus().then(setDemo).catch(() => setDemo(null));
    }, []);

    useEffect(() => {
        if (!inviteToken) return;
        getInviteInfo(inviteToken)
            .then((info) => { setInvite(info); setEmail(info.email); })
            .catch((e) => setInviteError(errorMessage(e, 'This invite link is invalid or has expired.')));
    }, [inviteToken]);

    const signIn = async (mail: string, pass: string) => {
        const form = new FormData();
        form.append('username', mail);
        form.append('password', pass);
        const res = await api.post('/auth/token', form, { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } });
        login(res.data.access_token);
    };

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setBusy(true);
        try {
            await signIn(email, password);
            toast.success('Logged in');
        } catch (err) {
            const e = err as { code?: string; response?: { status?: number } };
            if (e.code === 'ERR_NETWORK') toast.error('Cannot connect to server. Is the backend running?');
            else toast.error(errorMessage(err, e.response?.status === 401 ? 'Invalid credentials' : 'Login failed. Please try again.'));
        } finally {
            setBusy(false);
        }
    };

    const handleCreate = async (e: React.FormEvent) => {
        e.preventDefault();
        setBusy(true);
        try {
            const res = await register({
                email, password,
                workspace_name: wsName.trim() || undefined,
                llm_keys: Object.fromEntries(Object.entries(keys).filter(([, v]) => v.trim())),
                client_emails: clientEmails.split(/[\s,;]+/).map((x) => x.trim()).filter(Boolean),
            });
            if (res.invites.length) setCreated(res);  // show invite links before moving on
            else await signIn(email, password);
            toast.success(`Workspace "${res.workspace_name}" created`);
        } catch (err) {
            toast.error(errorMessage(err, 'Could not create the workspace.'));
        } finally {
            setBusy(false);
        }
    };

    const handleJoin = async (e: React.FormEvent) => {
        e.preventDefault();
        setBusy(true);
        try {
            await register({ email, password, invite_token: inviteToken });
            toast.success(`Joined ${invite?.workspace_name || 'the workspace'}`);
            await signIn(email, password);
        } catch (err) {
            toast.error(errorMessage(err, 'Could not join the workspace.'));
        } finally {
            setBusy(false);
        }
    };

    const tryDemo = async (role: 'admin' | 'client') => {
        setBusy(true);
        try {
            login(await demoLogin(role));
        } catch (err) {
            toast.error(errorMessage(err, 'The demo is unavailable right now.'));
        } finally {
            setBusy(false);
        }
    };

    const credentialFields = (prefix: string, lockedEmail = false) => (
        <>
            <div className="space-y-2">
                <Label htmlFor={`${prefix}-email`}>Email</Label>
                <Input id={`${prefix}-email`} type="email" value={email} readOnly={lockedEmail}
                    onChange={(e) => setEmail(e.target.value)} required suppressHydrationWarning />
            </div>
            <div className="space-y-2">
                <Label htmlFor={`${prefix}-password`}>Password</Label>
                <Input id={`${prefix}-password`} type="password" minLength={prefix === 'login' ? undefined : 8}
                    value={password} onChange={(e) => setPassword(e.target.value)} required suppressHydrationWarning />
            </div>
        </>
    );

    if (created) {
        return (
            <Shell>
                <CardHeader>
                    <CardTitle>Send these invite links</CardTitle>
                    <CardDescription>
                        Each person opens their link and sets a password to join <strong>{created.workspace_name}</strong>.
                        Links work once and expire in 14 days. You can create new ones later under Workspace settings.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    {created.invites.map((i) => (
                        <div key={i.email} className="space-y-1">
                            <Label className="text-xs">{i.email}</Label>
                            <div className="flex gap-2">
                                <Input readOnly value={inviteLink(i.token)} className="font-mono text-xs" />
                                <Button type="button" size="icon" variant="outline" aria-label={`Copy link for ${i.email}`}
                                    onClick={() => { void navigator.clipboard.writeText(inviteLink(i.token)); toast.success('Link copied'); }}>
                                    <Copy className="h-4 w-4" />
                                </Button>
                            </div>
                        </div>
                    ))}
                    <Button className="w-full" disabled={busy} onClick={() => void signIn(email, password)}>Continue to my workspace</Button>
                </CardContent>
            </Shell>
        );
    }

    if (inviteToken) {
        return (
            <Shell>
                <CardHeader>
                    <CardTitle>Join {invite?.workspace_name || 'a workspace'}</CardTitle>
                    <CardDescription>
                        {inviteError || 'You were invited to RAGOps. Choose a password to create your account.'}
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {invite ? (
                        <form onSubmit={handleJoin} className="space-y-4">
                            {credentialFields('join', true)}
                            <Button type="submit" className="w-full" disabled={busy}>Join workspace</Button>
                        </form>
                    ) : (
                        <Button variant="outline" className="w-full" onClick={() => { window.location.href = '/login'; }}>Go to sign in</Button>
                    )}
                </CardContent>
            </Shell>
        );
    }

    return (
        <Shell>
            <CardHeader>
                <CardTitle>Welcome to RAGOps</CardTitle>
                <CardDescription>Sign in, or create a workspace for your team.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                {demo?.enabled && (
                    <div className="space-y-2 rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3">
                        <p className="text-sm font-medium">Try it without an account</p>
                        <div className="grid grid-cols-2 gap-2">
                            <Button variant="outline" disabled={busy} onClick={() => void tryDemo('admin')} className="gap-2">
                                <ShieldCheck className="h-4 w-4" /> Demo admin
                            </Button>
                            <Button variant="outline" disabled={busy} onClick={() => void tryDemo('client')} className="gap-2">
                                <UserRound className="h-4 w-4" /> Demo client
                            </Button>
                        </div>
                        <p className="text-xs text-muted-foreground">
                            A shared sandbox: other visitors see the same projects and chats, so don&apos;t upload anything private.
                            Demo projects are deleted after {demo.reset_hours} hours.
                        </p>
                    </div>
                )}

                <Tabs defaultValue="login">
                    <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="login">Sign in</TabsTrigger>
                        <TabsTrigger value="create">Create workspace</TabsTrigger>
                    </TabsList>

                    <TabsContent value="login">
                        <form onSubmit={handleLogin} className="space-y-4 mt-4">
                            {credentialFields('login')}
                            <Button type="submit" className="w-full" disabled={busy} suppressHydrationWarning>Sign in</Button>
                            <p className="text-xs text-muted-foreground">
                                Joining a team? Open the invite link your admin sent you.
                            </p>
                        </form>
                    </TabsContent>

                    <TabsContent value="create">
                        <form onSubmit={handleCreate} className="space-y-4 mt-4">
                            <div className="space-y-2">
                                <Label htmlFor="ws-name">Workspace name</Label>
                                <Input id="ws-name" placeholder="Acme Corp" value={wsName} onChange={(e) => setWsName(e.target.value)} />
                            </div>
                            {credentialFields('create')}

                            <div className="space-y-3 rounded-lg border p-3">
                                <div className="flex items-center gap-2 text-sm font-medium"><KeyRound className="h-4 w-4" /> LLM API keys</div>
                                <p className="text-xs text-muted-foreground">
                                    Your workspace uses its own keys. All three providers have free tiers; one is enough.
                                    You can add or change them later.
                                </p>
                                {KEY_FIELDS.map((k) => (
                                    <div key={k.id} className="space-y-1">
                                        <Label htmlFor={`key-${k.id}`} className="text-xs">{k.label} <span className="text-muted-foreground">({k.hint})</span></Label>
                                        <Input id={`key-${k.id}`} type="password" autoComplete="off" placeholder={k.placeholder}
                                            value={keys[k.id]} onChange={(e) => setKeys({ ...keys, [k.id]: e.target.value })} />
                                    </div>
                                ))}
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="clients">Client emails <span className="text-muted-foreground">(optional)</span></Label>
                                <Textarea id="clients" rows={3} placeholder={"sales@acme.com\ntech@acme.com"}
                                    value={clientEmails} onChange={(e) => setClientEmails(e.target.value)} />
                                <p className="text-xs text-muted-foreground">
                                    You&apos;ll get an invite link for each. Choose which projects they can open in Workspace settings.
                                </p>
                            </div>
                            <Button type="submit" className="w-full" disabled={busy}>Create workspace</Button>
                        </form>
                    </TabsContent>
                </Tabs>
            </CardContent>
        </Shell>
    );
}

function Shell({ children }: { children: React.ReactNode }) {
    return (
        <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4 py-8 dark:bg-slate-950">
            <Card className="w-full max-w-[440px]">{children}</Card>
        </div>
    );
}

export default function LoginPage() {
    return (
        <Suspense fallback={null}>
            <LoginInner />
        </Suspense>
    );
}
