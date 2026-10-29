"""Multi-tenant isolation: two independent workspaces, invite links, per-client
project access, bring-your-own LLM keys, and the restricted demo."""

import pytest

from conftest import _login, join_with_invite


@pytest.fixture(scope="session")
def other_admin(app_client, admin):
    r = app_client.post("/auth/register", json={
        "email": "boss@globex.io", "password": "password123", "workspace_name": "Globex",
        "client_emails": ["sales@globex.io", "tech@globex.io"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["role"] == "admin" and body["workspace_name"] == "Globex"
    assert {i["email"] for i in body["invites"]} == {"sales@globex.io", "tech@globex.io"}
    headers = _login(app_client, "boss@globex.io")
    headers["_invites"] = {i["email"]: i["token"] for i in body["invites"]}
    return headers


def _h(headers):
    return {k: v for k, v in headers.items() if not k.startswith("_")}


def test_workspaces_are_isolated(app_client, admin, employee, project, other_admin):
    boss = _h(other_admin)
    # The other admin sees none of the first workspace: projects, users, analytics, docs.
    assert app_client.get("/rag/projects/", headers=boss).json() == []
    assert app_client.get(f"/rag/projects/{project}", headers=boss).status_code == 404
    assert app_client.delete(f"/rag/projects/{project}", headers=boss).status_code == 404
    assert app_client.get(f"/api/analytics/{project}", headers=boss).status_code == 404
    assert app_client.get(f"/api/analytics/{project}/executions", headers=boss).status_code == 404
    assert app_client.get(f"/rag/ingest/?project_id={project}", headers=boss).status_code == 404
    emails = {u["email"] for u in app_client.get("/auth/users", headers=boss).json()}
    assert emails == {"boss@globex.io"}
    emp_id = next(u["id"] for u in app_client.get("/auth/users", headers=admin).json() if u["email"] == "emp@test.io")
    assert app_client.delete(f"/auth/users/{emp_id}", headers=boss).status_code == 404
    assert app_client.patch(f"/auth/users/{emp_id}", json={"role": "admin"}, headers=boss).status_code == 404
    # Members can't be pulled across workspaces either.
    own = app_client.post("/rag/projects/", json={"name": "Globex KB"}, headers=boss).json()["id"]
    r = app_client.post(f"/rag/projects/{own}/members", json={"email": "emp@test.io"}, headers=boss)
    assert r.status_code == 404
    # And the first workspace can't see Globex's project.
    assert own not in [p["id"] for p in app_client.get("/rag/projects/", headers=admin).json()]
    assert app_client.get(f"/rag/projects/{own}", headers=employee).status_code == 404


def test_invite_link_controls_join_and_project_access(app_client, other_admin):
    boss = _h(other_admin)
    tech = app_client.post("/rag/projects/", json={"name": "Tech"}, headers=boss).json()
    sales = app_client.post("/rag/projects/", json={"name": "Sales"}, headers=boss).json()
    assert tech["visibility"] == "members_only"  # new projects are private by default

    token = other_admin["_invites"]["tech@globex.io"]
    # The link is needed, and only for the invited email.
    bad = app_client.post("/auth/register", json={"email": "tech@globex.io", "password": "password123", "invite_token": "nope"})
    assert bad.status_code == 400
    wrong = app_client.post("/auth/register", json={"email": "intruder@x.io", "password": "password123", "invite_token": token})
    assert wrong.status_code == 400
    assert app_client.get(f"/auth/invite/{token}").json()["workspace_name"] == "Globex"

    # Re-inviting with projects refreshes the link (the old token stops working).
    r = app_client.post("/workspace/invites", json={"email": "tech@globex.io", "project_ids": [tech["id"]]}, headers=boss)
    new_token = r.json()["token"]
    assert app_client.get(f"/auth/invite/{token}").status_code == 404
    r = app_client.post("/auth/register", json={"email": "tech@globex.io", "password": "password123", "invite_token": new_token})
    assert r.status_code == 200 and r.json()["role"] == "client", r.text
    client = _login(app_client, "tech@globex.io")
    assert [p["name"] for p in app_client.get("/rag/projects/", headers=client).json()] == ["Tech"]
    assert app_client.get(f"/rag/projects/{sales['id']}", headers=client).status_code == 404
    assert app_client.get("/auth/me", headers=client).json()["workspace"]["name"] == "Globex"

    # The admin reassigns the client from Tech to Sales.
    uid = next(u["id"] for u in app_client.get("/auth/users", headers=boss).json() if u["email"] == "tech@globex.io")
    r = app_client.put(f"/workspace/members/{uid}/projects", json={"project_ids": [sales["id"]]}, headers=boss)
    assert r.status_code == 200
    assert [p["name"] for p in app_client.get("/rag/projects/", headers=client).json()] == ["Sales"]
    # A used invite is gone.
    assert all(i["email"] != "tech@globex.io" for i in app_client.get("/workspace", headers=boss).json()["invites"])


def test_byok_workspace_never_uses_server_keys(app_client, other_admin):
    """Globex has no keys of its own: LLM calls must fail with a clear message,
    not silently run on the server's GROQ_API_KEY."""
    boss = _h(other_admin)
    pid = app_client.post("/rag/projects/", json={"name": "Policies"}, headers=boss).json()["id"]
    app_client.patch(f"/rag/projects/{pid}/config", json={"confidence_threshold": 0.0}, headers=boss)
    app_client.post("/rag/ingest/upload", data={"project_id": pid},
                    files=[("files", ("p.txt", b"Remote work is allowed two days per week.", "text/plain"))], headers=boss)
    from app.jobs import queue
    queue.drain()

    r = app_client.post("/chat/message", json={"content": "How many remote days?", "project_id": pid}, headers=boss)
    assert r.status_code == 400 and "API key" in r.json()["detail"]
    assert app_client.get("/models/catalog", headers=boss).json()["providers"] == {}

    info = app_client.put("/workspace/keys", json={"groq": "gsk_workspace_key_123456", "verify": False}, headers=boss).json()
    assert info["llm_keys"]["groq"].startswith("gsk_") and "123456" not in info["llm_keys"]["groq"].replace("3456", "")
    r = app_client.post("/chat/message", json={"content": "How many remote days?", "project_id": pid}, headers=boss)
    assert r.status_code == 200, r.text

    # The key really is the workspace's own: the scope resolves to it, not the env.
    from sqlmodel import Session
    from app.core import llm_keys
    from app.db import engine
    with Session(engine) as db:
        assert llm_keys.scope_for_project(db, pid) == {"groq": "gsk_workspace_key_123456"}
    # Removing it brings the error back.
    app_client.put("/workspace/keys", json={"groq": "", "verify": False}, headers=boss)
    r = app_client.post("/chat/message", json={"content": "How many remote days?", "project_id": pid}, headers=boss)
    assert r.status_code == 400


def test_streaming_uses_workspace_scope(app_client, other_admin):
    boss = _h(other_admin)
    pid = app_client.get("/rag/projects/", headers=boss).json()[0]["id"]
    with app_client.stream("POST", "/chat/message/stream", json={"content": "hello", "project_id": pid}, headers=boss) as r:
        body = "".join(r.iter_text())
    assert "event: error" in body and "API key" in body


def test_demo_login_and_restrictions(app_client, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "demo_mode", True)
    assert app_client.get("/auth/demo").json()["enabled"] is True
    admin = {"Authorization": f"Bearer {app_client.post('/auth/demo', json={'role': 'admin'}).json()['access_token']}"}
    client = {"Authorization": f"Bearer {app_client.post('/auth/demo', json={'role': 'client'}).json()['access_token']}"}
    me = app_client.get("/auth/me", headers=admin).json()
    assert me["role"] == "admin" and me["workspace"]["is_demo"] is True

    pid = app_client.post("/rag/projects/", json={"name": "Demo KB", "visibility": "members_only"}, headers=admin).json()["id"]
    # The demo client sees the demo admin's projects (and nothing else).
    assert [p["name"] for p in app_client.get("/rag/projects/", headers=client).json()] == ["Demo KB"]

    assert app_client.put("/workspace/keys", json={"groq": "x", "verify": False}, headers=admin).status_code == 403
    assert app_client.post("/workspace/invites", json={"email": "a@b.io"}, headers=admin).status_code == 403
    assert app_client.post("/auth/users", json={"email": "a@b.io", "password": "password123"}, headers=admin).status_code == 403
    assert app_client.post(f"/rag/projects/{pid}/api-keys", json={"name": "k"}, headers=admin).status_code == 403
    r = app_client.post(f"/rag/projects/{pid}/databases", json={
        "name": "x", "db_type": "postgresql", "host": "h", "port": 5432, "db_name": "d", "username": "u", "password": "p"},
        headers=admin)
    assert r.status_code == 403
    assert app_client.post("/api/benchmark/start", json={"project_id": pid, "variants": ["naive"]}, headers=admin).status_code == 403

    # Demo projects expire.
    from datetime import datetime, timedelta
    from sqlmodel import Session
    from app.db import engine
    from app.models.rag import Project
    with Session(engine) as db:
        p = db.get(Project, pid)
        p.created_at = datetime.utcnow() - timedelta(hours=settings.demo_reset_hours + 1)
        db.add(p)
        db.commit()
    app_client.post("/auth/demo", json={"role": "client"})
    assert app_client.get("/rag/projects/", headers=admin).json() == []


def test_demo_disabled_by_default(app_client):
    assert app_client.post("/auth/demo", json={"role": "admin"}).status_code == 404


def test_retired_public_passwords_are_refused(app_client, admin):
    app_client.post("/auth/users", json={"email": "admin@ragops.com", "password": "admin123", "role": "admin"}, headers=admin)
    r = app_client.post("/auth/token", data={"username": "admin@ragops.com", "password": "admin123"})
    assert r.status_code == 403


def test_legacy_data_moves_into_one_workspace(tmp_path):
    """Upgrade path: a pre-workspace database gets a 'Main workspace' holding
    every existing user and project, running on the server keys."""
    from sqlalchemy import create_engine, text
    from sqlmodel import SQLModel
    from app import db as dbmod

    eng = create_engine(f"sqlite:///{tmp_path}/legacy.db")
    SQLModel.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO \"user\" (email, hashed_password, role) VALUES ('old@a.io', 'x', 'ADMIN'), ('c@a.io', 'x', 'CLIENT')"))
        conn.execute(text("INSERT INTO project (name, created_at, kb_version, visibility) VALUES ('Old', '2026-01-01', 1, 'all_users')"))
        dbmod._m004_default_workspace(conn)
        ws = conn.execute(text("SELECT id, name, owner_id, uses_platform_keys FROM workspace")).fetchall()
        assert len(ws) == 1 and ws[0][1] == "Main workspace" and ws[0][3]
        owner = conn.execute(text("SELECT id FROM \"user\" WHERE email = 'old@a.io'")).scalar()
        assert ws[0][2] == owner
        assert conn.execute(text("SELECT COUNT(*) FROM \"user\" WHERE workspace_id = :w"), {"w": ws[0][0]}).scalar() == 2
        assert conn.execute(text("SELECT workspace_id FROM project")).scalar() == ws[0][0]
