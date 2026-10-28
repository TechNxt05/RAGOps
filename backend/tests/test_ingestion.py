from app.jobs import queue


def _status(client, headers, pid, ids):
    r = client.get(f"/rag/ingest/status?project_id={pid}&doc_ids={','.join(map(str, ids))}", headers=headers)
    return {d["filename"]: d for d in r.json()["documents"]}


def test_multi_file_upload_validation_dedupe_and_quarantine(app_client, admin):
    pid = app_client.post("/rag/projects/", json={"name": "Uploads"}, headers=admin).json()["id"]
    files = [
        ("files", ("a.txt", b"Alpha document about quarterly revenue targets and growth.", "text/plain")),
        ("files", ("b.md", b"# Beta\nBeta document describing the onboarding checklist.", "text/markdown")),
        ("files", ("c.html", b"<html><body><h1>Gamma</h1><p>Gamma covers the security policy.</p></body></html>", "text/html")),
        ("files", ("evil.exe", b"MZ", "application/octet-stream")),
        ("files", ("empty.txt", b"", "text/plain")),
        ("files", ("keys.txt", b"aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "text/plain")),
    ]
    res = app_client.post("/rag/ingest/upload", data={"project_id": pid}, files=files, headers=admin).json()["results"]
    by = {r["filename"]: r for r in res}
    assert by["evil.exe"]["status"] == "rejected"
    assert by["empty.txt"]["status"] == "rejected"
    assert all(by[f]["status"] == "queued" for f in ("a.txt", "b.md", "c.html", "keys.txt"))

    assert queue.drain() == 4
    st = _status(app_client, admin, pid, [r["doc_id"] for r in res if r.get("doc_id")])
    assert st["a.txt"]["status"] == "complete" and st["a.txt"]["chunk_count"] >= 1
    assert st["c.html"]["status"] == "complete"
    assert st["keys.txt"]["status"] == "quarantined"

    dup = app_client.post("/rag/ingest/upload", data={"project_id": pid}, files=[files[0]], headers=admin).json()["results"][0]
    assert dup["status"] == "duplicate"


def test_concurrent_projects_do_not_overwrite_each_other(app_client, admin, project):
    """The old single FAISS file lost other projects' vectors on rebuild."""
    other = app_client.post("/rag/projects/", json={"name": "Other"}, headers=admin).json()["id"]
    app_client.post("/rag/ingest/upload", data={"project_id": other},
                    files=[("files", ("x.txt", b"Completely different content about rockets and fuel.", "text/plain"))], headers=admin)
    queue.drain()
    app_client.post(f"/rag/ingest/projects/{other}/reindex", headers=admin)
    queue.drain()
    hits = app_client.post("/rag/inspector/search", json={"project_id": project, "query": "annual leave days", "top_k": 3},
                           headers=admin).json()["results"]
    assert hits and hits[0]["document_name"] == "leave.txt"


def test_delete_removes_document_from_retrieval(app_client, admin):
    pid = app_client.post("/rag/projects/", json={"name": "Deletes"}, headers=admin).json()["id"]
    r = app_client.post("/rag/ingest/upload", data={"project_id": pid},
                        files=[("files", ("gone.txt", b"The submarine budget is eleven million.", "text/plain"))], headers=admin)
    queue.drain()
    doc_id = r.json()["results"][0]["doc_id"]
    before = app_client.post("/rag/inspector/search", json={"project_id": pid, "query": "submarine budget"}, headers=admin).json()
    assert before["results"]
    assert app_client.post("/rag/ingest/documents/bulk-delete", json={"doc_ids": [doc_id]}, headers=admin).json()["deleted"] == [doc_id]
    after = app_client.post("/rag/inspector/search", json={"project_id": pid, "query": "submarine budget"}, headers=admin).json()
    assert after["results"] == []


def test_background_worker_picks_up_uploads_promptly(app_client, admin):
    """Real worker thread (not drain): uploads committed after the wake signal must
    still be processed within seconds, although the idle poll is an hour."""
    import time
    queue.start_workers(1)
    try:
        pid = app_client.post("/rag/projects/", json={"name": "Worker"}, headers=admin).json()["id"]
        r = app_client.post("/rag/ingest/upload", data={"project_id": pid},
                            files=[("files", ("w.txt", b"The worker processes this file in the background.", "text/plain"))],
                            headers=admin)
        doc_id = r.json()["results"][0]["doc_id"]
        deadline = time.time() + 20
        status = None
        while time.time() < deadline:
            status = app_client.get(f"/rag/ingest/documents/{doc_id}/status", headers=admin).json()["status"]
            if status == "complete":
                break
            time.sleep(0.3)
        assert status == "complete"
    finally:
        queue.stop_workers()
