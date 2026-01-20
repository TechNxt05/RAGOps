"use client";

import { useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import api from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';

export default function LoginPage() {
    const { login } = useAuth();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [role, setRole] = useState<'ADMIN' | 'CLIENT'>('CLIENT');

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const formData = new FormData();
            formData.append('username', email);
            formData.append('password', password);

            const res = await api.post('/auth/token', formData, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
            });
            login(res.data.access_token);
            toast.success('Logged in successfully');
        } catch (err: any) {
            if (err.response && err.response.status === 401) {
                toast.error('Invalid credentials');
            } else if (err.code === 'ERR_NETWORK') {
                toast.error('Cannot connect to server. Is the backend running?');
            } else {
                toast.error('Login failed. Please try again.');
            }
        }
    };

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            await api.post('/auth/register', { email, password, role });
            toast.success('Registration successful! Please login.');
        } catch (err) {
            toast.error('Registration failed. Email might be taken.');
        }
    };

    const handleQuickLogin = async (userRole: 'ADMIN' | 'CLIENT') => {
        try {
            const credentials = userRole === 'ADMIN'
                ? { email: 'admin@ragops.com', password: 'admin123' }
                : { email: 'client@ragops.com', password: 'client123' };

            const formData = new FormData();
            formData.append('username', credentials.email);
            formData.append('password', credentials.password);

            const res = await api.post('/auth/token', formData, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
            });
            login(res.data.access_token);
            toast.success(`Logged in as ${userRole}`);
        } catch (err: any) {
            toast.error(`Quick login failed. Please run the seed script first.`);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-slate-50 dark:bg-slate-950">
            <Card className="w-[400px]">
                <CardHeader>
                    <CardTitle>Welcome to RAGOps</CardTitle>
                    <CardDescription>Enter your credentials to access the platform.</CardDescription>
                </CardHeader>
                <div className="px-6 pb-4 space-y-2">
                    <p className="text-xs text-muted-foreground text-center">Quick Demo Access:</p>
                    <div className="grid grid-cols-2 gap-2">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleQuickLogin('ADMIN')}
                            className="w-full"
                        >
                            Login as Admin
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleQuickLogin('CLIENT')}
                            className="w-full"
                        >
                            Login as Client
                        </Button>
                    </div>
                </div>
                <CardContent>
                    <Tabs defaultValue="login">
                        <TabsList className="grid w-full grid-cols-2">
                            <TabsTrigger value="login">Login</TabsTrigger>
                            <TabsTrigger value="register">Register</TabsTrigger>
                        </TabsList>

                        <TabsContent value="login">
                            <form onSubmit={handleLogin} className="space-y-4 mt-4">
                                <div className="space-y-2">
                                    <Label htmlFor="email">Email</Label>
                                    <Input id="email" type="email" placeholder="admin@example.com" value={email} onChange={(e) => setEmail(e.target.value)} required suppressHydrationWarning />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="password">Password</Label>
                                    <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required suppressHydrationWarning />
                                </div>
                                <Button type="submit" className="w-full" suppressHydrationWarning>Login</Button>
                            </form>
                        </TabsContent>

                        <TabsContent value="register">
                            <form onSubmit={handleRegister} className="space-y-4 mt-4">
                                <div className="space-y-2">
                                    <Label htmlFor="reg-email">Email</Label>
                                    <Input id="reg-email" type="email" placeholder="user@example.com" value={email} onChange={(e) => setEmail(e.target.value)} required />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="reg-password">Password</Label>
                                    <Input id="reg-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="reg-role">Role</Label>
                                    <select
                                        id="reg-role"
                                        value={role}
                                        onChange={(e) => setRole(e.target.value as 'ADMIN' | 'CLIENT')}
                                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                                    >
                                        <option value="CLIENT">Client</option>
                                        <option value="ADMIN">Admin</option>
                                    </select>
                                    <p className="text-xs text-muted-foreground">⚠️ Admin role is for demo purposes only</p>
                                </div>
                                <Button type="submit" variant="secondary" className="w-full">Create Account</Button>
                            </form>
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>
        </div>
    );
}
