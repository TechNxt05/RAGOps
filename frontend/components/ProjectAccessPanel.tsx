"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Trash2, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  addProjectMember,
  createUser,
  deleteUser,
  errorMessage,
  getUsers,
  getProjectMembers,
  removeProjectMember,
  updateProject,
  type Project,
  type ProjectMember,
  type UserInfo,
} from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

const selectCls = "flex h-9 rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm";

export function ProjectAccessPanel({ project, onProjectChange }: { project: Project; onProjectChange: (p: Project) => void }) {
  const [members, setMembers] = useState<ProjectMember[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<"viewer" | "editor">("viewer");
  const [newUser, setNewUser] = useState({ email: "", password: "", role: "client" as "client" | "admin" });
  const [users, setUsers] = useState<UserInfo[]>([]);
  const { user: me } = useAuth();

  useEffect(() => {
    let cancelled = false;
    getUsers().then((u) => { if (!cancelled) setUsers(u); }).catch(() => undefined);
    return () => { cancelled = true; };
  }, []);
  const reloadUsers = () => getUsers().then(setUsers).catch(() => undefined);

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
      toast.success(visibility === "members_only" ? "Only members can access this project now" : "All signed-in users can access this project");
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

  const create = async () => {
    try {
      const u = await createUser(newUser.email.trim(), newUser.password, newUser.role);
      toast.success(`Account created for ${u.email}`);
      setNewUser({ email: "", password: "", role: "client" });
      void reloadUsers();
    } catch (e) {
      toast.error(errorMessage(e, "Could not create account"));
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Access control</CardTitle>
        <CardDescription>Decide who can chat with this project&apos;s documents and connected databases.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label>Visibility</Label>
          <select
            className={selectCls + " w-full"}
            value={project.visibility || "all_users"}
            onChange={(e) => void setVisibility(e.target.value as "all_users" | "members_only")}
          >
            <option value="all_users">Everyone with an account</option>
            <option value="members_only">Members only</option>
          </select>
        </div>

        <div className="space-y-2">
          <Label>Members</Label>
          <p className="text-xs text-muted-foreground">Viewers can ask questions. Editors can also upload and manage documents.</p>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input placeholder="employee@company.com" value={email} onChange={(e) => setEmail(e.target.value)} />
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

        <div className="space-y-2 rounded-lg border p-3">
          <Label>Accounts</Label>
          <p className="text-xs text-muted-foreground">For employees when self-registration is turned off. Share the password with them securely.</p>
          <div className="grid gap-2 sm:grid-cols-[1fr_1fr_auto_auto]">
            <Input placeholder="email" value={newUser.email} onChange={(e) => setNewUser({ ...newUser, email: e.target.value })} />
            <Input type="password" placeholder="temporary password (8+ chars)" value={newUser.password} onChange={(e) => setNewUser({ ...newUser, password: e.target.value })} />
            <select className={selectCls} value={newUser.role} onChange={(e) => setNewUser({ ...newUser, role: e.target.value as "client" | "admin" })}>
              <option value="client">Employee</option>
              <option value="admin">Admin</option>
            </select>
            <Button variant="outline" disabled={!newUser.email || newUser.password.length < 8} onClick={() => void create()}>Create</Button>
          </div>
          <ul className="max-h-56 divide-y overflow-y-auto rounded-lg border text-sm">
            {users.map((u) => (
              <li key={u.id} className="flex items-center justify-between px-3 py-1.5">
                <span className="truncate">{u.email} <span className="text-xs text-muted-foreground">({u.role})</span></span>
                {u.id !== me?.id && (
                  <Button size="icon" variant="ghost" aria-label={`Delete ${u.email}`} onClick={async () => {
                    if (!window.confirm(`Delete the account ${u.email}? Their chat history stays in analytics.`)) return;
                    try { await deleteUser(u.id); void reloadUsers(); } catch (e) { toast.error(errorMessage(e, "Delete failed")); }
                  }}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
