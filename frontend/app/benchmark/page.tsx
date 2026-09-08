"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { getProjects, Project } from "@/lib/api";
import { BenchmarkStudio } from "@/components/BenchmarkStudio";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Sparkles, Folder } from "lucide-react";
import { useRouter } from "next/navigation";

export default function BenchmarkPage() {
  const { user, isLoading } = useAuth();
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number>(1);

  useEffect(() => {
    if (!isLoading && (!user || user.role !== "admin")) {
      if (user && user.role !== "admin") router.push("/chat");
      else router.push("/login");
    }
  }, [user, isLoading, router]);

  useEffect(() => {
    getProjects()
      .then((data) => {
        setProjects(data);
        if (data.length > 0) setSelectedProjectId(data[0].id);
      })
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Header Bar */}
      <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30">
        <div className="flex items-center gap-4">
          <Link href="/admin">
            <Button variant="ghost" size="sm" className="gap-1.5 text-xs text-slate-300 hover:text-white">
              <ArrowLeft className="w-4 h-4" /> Back to Admin
            </Button>
          </Link>
          <div className="h-4 w-px bg-slate-800" />
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-indigo-400" />
            <h1 className="font-bold text-base text-white tracking-tight">AutoRAGBench Evaluation Studio</h1>
          </div>
        </div>

        {projects.length > 1 && (
          <div className="flex items-center gap-2">
            <Folder className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-xs text-slate-400">Target Corpus:</span>
            <select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(Number(e.target.value))}
              className="bg-slate-950 border border-slate-800 rounded-md text-xs px-2.5 py-1 text-slate-200"
            >
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        )}
      </header>

      {/* Main Studio View */}
      <main className="flex-1 p-6 md:p-10 overflow-y-auto">
        <BenchmarkStudio projectId={selectedProjectId} />
      </main>
    </div>
  );
}
