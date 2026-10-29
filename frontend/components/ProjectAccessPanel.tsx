"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Trash2, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import Link from "next/link";
import {
  addProjectMember,
  errorMessage,
  getUsers,
  getProjectMembers,
  removeProjectMember,
  updateProject,
  type Project,
  type ProjectMember,
  type UserInfo,
} from "@/lib/api";

const selectCls = "flex h-9 rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm";

export function ProjectAccessPanel({ project, onProjectChange }: { project: Project; onProjectChange: (p: Project) => void }) {
  const [members, setMembers] = useState<ProjectMember[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<"viewer" | "editor">("viewer");
  const [users, setUsers] = useState<UserInfo[]>([]);

  useEffect(() => {
    let cancelled = false;
    getUsers().then((u) => { if (!cancelled) setUsers(u); }).catch(() => undefined);
    return () => { cancelled = true; };
  }, []);

  const load = useCallback(async () => {
    try {
      setMembers(await getProjectMembers(project.id));
    } catch (e) {
      toast.error(errorMessage(e, "Could not load members"));
    }
  }, [project.id]);

  useEffect(() => {
    let cancelled = false;
    getProjectMembers(project.id)
      .then((m) => { if (!cancelled) setMembers(m); })
      .catch((e) => toast.error(errorMessage(e, "Could not load members")));
    return () => { cancelled = true; };
  }, [project.id]);

  const setVisibility = async (visibility: "all_users" | "members_only") => {
    try {
      onProjectChange(await updateProject(project.id, { visibility }));
      toast.success(visibility === "members_only" ? "Only assigned members can open this project now" : "Everyone in the workspace can open this project");
    } catch (e) {
      toast.error(errorMessage(e, "Could not update visibility"));
    }
  };

  const add = async () => {
    if (!email.trim()) return;
    try {
      await addProjectMember(project.id, email.trim(), role);
      setEmail("");
      toast.success("Member added");
      void load();
    } catch (e) {
      toast.error(errorMessage(e, "Could not add member"));
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Access control</CardTitle>
        <CardDescription>
          Decide who in your workspace can chat with this project&apos;s documents and connected databases.
          Invite people from <Link href="/workspace" className="underline">Workspace settings</Link>.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label>Visibility</Label>
          <select
            className={selectCls + " w-full"}
            value={project.visibility || "all_users"}
            onChange={(e) => void setVisibility(e.target.value as "all_users" | "members_only")}
          >
            <option value="members_only">Only assigned members</option>
            <option value="all_users">Everyone in this workspace</option>
          </select>
        </div>

        <div className="space-y-2">
          <Label>Members</Label>
          <p className="text-xs text-muted-foreground">Viewers can ask questions. Editors can also upload and manage documents.</p>
          <div className="flex flex-col gap-2 sm:flex-row">
            <select className={selectCls + " flex-1"} value={email} onChange={(e) => setEmail(e.target.value)}>
              <option value="">Choose a workspace member…</option>
              {users.filter((u) => u.role === "client" && !members.some((m) => m.user_id === u.id)).map((u) => (
                <option key={u.id} value={u.email}>{u.email}</option>
              ))}
            </select>
            <select className={selectCls} value={role} onChange={(e) => setRole(e.target.value as "viewer" | "editor")}>
              <option value="viewer">Viewer</option>
              <option value="editor">Editor</option>
            </select>
            <Button onClick={() => void add()}><UserPlus className="mr-1 h-4 w-4" />Add</Button>
          </div>
          <ul className="divide-y rounded-lg border text-sm">
            {members.length === 0 && <li className="px-3 py-2 text-muted-foreground">No members yet.</li>}
            {members.map((m) => (
              <li key={m.user_id} className="flex items-center justify-between px-3 py-2">
                <span className="truncate">{m.email}</span>
                <span className="flex items-center gap-2">
                  <span className="text-xs capitalize text-muted-foreground">{m.role}</span>
                  <Button
                    size="icon"
                    variant="ghost"
                    aria-label={`Remove ${m.email}`}
                    onClick={async () => {
                      await removeProjectMember(project.id, m.user_id);
                      void load();
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </span>
              </li>
            ))}
          </ul>
        </div>

      </CardContent>
    </Card>
  );
}
