def test_first_user_is_admin_and_no_self_escalation(app_client, admin, employee):
    me = app_client.get("/auth/me", headers=employee).json()
    assert me["role"] == "client"            # asked for admin, got client
    assert "hashed_password" not in me


def test_unauthenticated_endpoints_are_closed(app_client):
    assert app_client.post("/playground/generate", json={"messages": []}).status_code == 401
    assert app_client.post("/api/benchmark/start", json={}).status_code == 401
    assert app_client.get("/rag/projects/").status_code == 401


def test_members_only_project_is_hidden(app_client, admin, employee):
    pid = app_client.post("/rag/projects/", json={"name": "Secret", "visibility": "members_only"}, headers=admin).json()["id"]
    assert pid not in [p["id"] for p in app_client.get("/rag/projects/", headers=employee).json()]
    assert app_client.get(f"/rag/projects/{pid}", headers=employee).status_code == 404
    assert app_client.post("/chat/message", json={"content": "hi", "project_id": pid}, headers=employee).status_code == 404

    app_client.post(f"/rag/projects/{pid}/members", json={"email": "emp@test.io", "role": "viewer"}, headers=admin)
    assert app_client.get(f"/rag/projects/{pid}", headers=employee).status_code == 200
    up = app_client.post("/rag/ingest/upload", data={"project_id": pid},
                         files=[("files", ("a.txt", b"hello world text", "text/plain"))], headers=employee)
    assert up.status_code == 403             # viewers can't upload

    app_client.post(f"/rag/projects/{pid}/members", json={"email": "emp@test.io", "role": "editor"}, headers=admin)
    up = app_client.post("/rag/ingest/upload", data={"project_id": pid},
                         files=[("files", ("a.txt", b"hello world text", "text/plain"))], headers=employee)
    assert up.status_code == 200


def test_clients_cannot_manage_users(app_client, employee):
    assert app_client.get("/auth/users", headers=employee).status_code == 403
    assert app_client.post("/auth/users", json={"email": "x@y.z", "password": "password123", "role": "admin"},
                           headers=employee).status_code == 403


def test_admin_can_delete_user_with_history(app_client, admin, project):
    app_client.post("/auth/users", json={"email": "temp@test.io", "password": "password123", "role": "client"}, headers=admin)
    tok = app_client.post("/auth/token", data={"username": "temp@test.io", "password": "password123"}).json()["access_token"]
    app_client.post("/chat/message", json={"content": "How many leave days?", "project_id": project},
                    headers={"Authorization": f"Bearer {tok}"})
    uid = next(u["id"] for u in app_client.get("/auth/users", headers=admin).json() if u["email"] == "temp@test.io")
    assert app_client.delete(f"/auth/users/{uid}", headers=admin).status_code == 200
    assert app_client.post("/auth/token", data={"username": "temp@test.io", "password": "password123"}).status_code == 401
