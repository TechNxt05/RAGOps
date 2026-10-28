import json

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_mcp_requires_key_and_answers(app_client, admin, project):
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    assert app_client.post("/mcp/", json=rpc, headers=MCP_HEADERS).status_code == 401
    assert app_client.post("/mcp/", json=rpc, headers={**MCP_HEADERS, "Authorization": "Bearer rgo_bogus"}).status_code == 401

    key = app_client.post(f"/rag/projects/{project}/api-keys", json={"name": "ci"}, headers=admin).json()["key"]
    h = {**MCP_HEADERS, "Authorization": f"Bearer {key}"}
    tools = app_client.post("/mcp/", json=rpc, headers=h).json()["result"]["tools"]
    assert {"ask", "search_documents", "list_documents", "query_database"} <= {t["name"] for t in tools}

    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "ask", "arguments": {"question": "How many days of annual leave?"}}}
    result = json.loads(app_client.post("/mcp/", json=call, headers=h).json()["result"]["content"][0]["text"])
    assert "24 days" in result["answer"] and result["sources"][0]["source"] == "leave.txt"

    key_id = app_client.get(f"/rag/projects/{project}/api-keys", headers=admin).json()[0]["id"]
    app_client.delete(f"/rag/projects/{project}/api-keys/{key_id}", headers=admin)
    assert app_client.post("/mcp/", json=rpc, headers=h).status_code == 401
