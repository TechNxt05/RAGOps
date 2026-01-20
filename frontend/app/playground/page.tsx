"use client";

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { Send, Bot, User as UserIcon, Trash2, Zap, Settings2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';
import { api } from '@/lib/api';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

export default function PlaygroundPage() {
    // Config State
    const [systemPrompt, setSystemPrompt] = useState("You are a helpful AI assistant.");
    const [modelProvider, setModelProvider] = useState("groq");
    const [modelName, setModelName] = useState("llama-3.3-70b-versatile");
    const [temperature, setTemperature] = useState([0.7]);
    const [maxTokens, setMaxTokens] = useState([1000]);

    // Chat State
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isTyping]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const newMsgs = [...messages, { role: 'user', content: input } as Message];
        setMessages(newMsgs);
        setInput("");
        setIsTyping(true);

        try {
            const res = await api.post('/playground/generate', {
                messages: newMsgs,
                model_provider: modelProvider,
                model_name: modelName,
                temperature: temperature[0],
                max_tokens: maxTokens[0],
                system_prompt: systemPrompt
            });

            setMessages(prev => [...prev, { role: 'assistant', content: res.data.content }]);

        } catch (e: any) {
            toast.error("Generation failed: " + (e.response?.data?.detail || e.message));
        } finally {
            setIsTyping(false);
        }
    };

    const clearChat = () => setMessages([]);

    return (
        <div className="flex h-[100dvh] bg-slate-50 dark:bg-slate-950 overflow-hidden">
            {/* Sidebar: Config */}
            <div className="w-80 bg-white dark:bg-slate-900 border-r flex flex-col overflow-y-auto p-6 space-y-6">
                <div className="flex items-center gap-2 font-bold text-xl text-indigo-600">
                    <Zap className="w-6 h-6" /> Playground
                </div>

                <div className="space-y-4">
                    <div className="space-y-2">
                        <Label>System Prompt</Label>
                        <Textarea
                            value={systemPrompt}
                            onChange={e => setSystemPrompt(e.target.value)}
                            className="h-32 text-sm font-mono resize-none bg-slate-50 dark:bg-slate-800"
                            placeholder="Define the AI's persona..."
                        />
                    </div>

                    <div className="space-y-2">
                        <Label>Provider</Label>
                        <select
                            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                            value={modelProvider}
                            onChange={(e) => {
                                setModelProvider(e.target.value);
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
                            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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

                    <div className="space-y-4 pt-4 border-t">
                        <div className="flex items-center justify-between">
                            <Label>Temperature</Label>
                            <span className="text-sm text-slate-500">{temperature[0]}</span>
                        </div>
                        <Slider value={temperature} max={1} step={0.1} onValueChange={setTemperature} />
                    </div>

                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <Label>Max Tokens</Label>
                            <span className="text-sm text-slate-500">{maxTokens[0]}</span>
                        </div>
                        <Slider value={maxTokens} max={4000} step={100} onValueChange={setMaxTokens} />
                    </div>
                </div>

                <div className="mt-auto pt-6 border-t text-xs text-slate-400">
                    Use this mode to test prompts and models without RAG context.
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col h-full relative">
                <header className="h-16 border-b flex items-center justify-between px-6 bg-white/50 backdrop-blur dark:bg-slate-900/50">
                    <div className="font-semibold text-slate-700 dark:text-slate-200">Interactive Session</div>
                    <Button variant="ghost" size="sm" onClick={clearChat} className="text-red-500 hover:text-red-600 hover:bg-red-50">
                        <Trash2 className="w-4 h-4 mr-2" />
                        Clear
                    </Button>
                </header>

                <div className="flex-1 overflow-y-auto p-6 scroll-smooth" ref={scrollRef}>
                    <div className="max-w-3xl mx-auto space-y-6">
                        {messages.length === 0 && (
                            <div className="flex flex-col items-center justify-center h-[50vh] text-slate-400">
                                <Settings2 className="w-12 h-12 mb-4 opacity-50" />
                                <p>Adjust settings on the left and start typing to test.</p>
                            </div>
                        )}
                        <AnimatePresence initial={false}>
                            {messages.map((msg, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    {msg.role !== 'user' && (
                                        <div className="w-8 h-8 rounded-full bg-indigo-100 dark:bg-indigo-900 flex items-center justify-center shrink-0">
                                            <Bot className="w-5 h-5 text-indigo-600 dark:text-indigo-300" />
                                        </div>
                                    )}

                                    <div className={`p-4 rounded-2xl shadow-sm text-sm whitespace-pre-wrap max-w-[80%] ${msg.role === 'user'
                                            ? 'bg-indigo-600 text-white rounded-tr-none'
                                            : 'bg-white dark:bg-slate-800 border rounded-tl-none'
                                        }`}>
                                        {msg.content}
                                    </div>

                                    {msg.role === 'user' && (
                                        <div className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center shrink-0">
                                            <UserIcon className="w-5 h-5" />
                                        </div>
                                    )}
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        {isTyping && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                                <div className="w-8 h-8 rounded-full bg-indigo-100 dark:bg-indigo-900 flex items-center justify-center">
                                    <Bot className="w-5 h-5 text-indigo-600" />
                                </div>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-2xl border shadow-sm flex gap-1 items-center">
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-75" />
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-150" />
                                </div>
                            </motion.div>
                        )}
                        <div ref={scrollRef} />
                    </div>
                </div>

                <div className="p-4 bg-white dark:bg-slate-900 border-t">
                    <div className="max-w-3xl mx-auto flex gap-2">
                        <Input
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && handleSend()}
                            placeholder="Type a message..."
                            className="rounded-full shadow-sm"
                        />
                        <Button onClick={handleSend} disabled={!input.trim() || isTyping} className="rounded-full w-10 h-10 p-0">
                            <Send className="w-4 h-4" />
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
}
