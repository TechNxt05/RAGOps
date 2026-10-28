import json


def test_answer_is_grounded_and_cited(app_client, employee, project):
    j = app_client.post("/chat/message", json={"content": "How many vacation days do I get?", "project_id": project},
                        headers=employee).json()
    assert "24 days" in j["content"] and "[leave.txt]" in j["content"]
    assert j["sources"][0]["source"] == "leave.txt"
    # the chunks shown as sources are the chunks the model was given
    assert any("24 days" in s["snippet"] for s in j["sources"])
    assert j["usage_metadata"]["tokens"]["input_tokens"] > 0


def test_semantic_cache_hits_same_question_but_not_different_entity(app_client, employee, project):
    q = {"content": "How many vacation days do I get each year?", "project_id": project}
    first = app_client.post("/chat/message", json=q, headers=employee).json()
    again = app_client.post("/chat/message", json=q, headers=employee).json()
    assert first["cached"] is False and again["cached"] is True
    other = app_client.post("/chat/message", json={"content": "How many sick days do I get each year?", "project_id": project},
                            headers=employee).json()
    assert other["cached"] is False


def test_off_topic_question_is_refused(app_client, employee, project):
    j = app_client.post("/chat/message", json={"content": "What is the boiling point of mercury?", "project_id": project},
                        headers=employee).json()
    assert j.get("refused") is True and j["sources"] == []


def test_streaming_emits_meta_tokens_done(app_client, employee, project):
    events, done = [], None
    with app_client.stream("POST", "/chat/message/stream",
                           json={"content": "Who approves international travel?", "project_id": project}, headers=employee) as r:
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:") and events[-1] == "done":
                done = json.loads(line[5:])
    assert events[0] == "meta" and "token" in events and events[-1] == "done"
    assert "director" in done["content"].lower()


def test_cross_user_chats_never_leak(app_client, admin, employee, project):
    app_client.post("/chat/message", json={"content": "My secret code is 4242. How many leave days?", "project_id": project},
                    headers=admin)
    j = app_client.post("/chat/message", json={"content": "Who approves travel?", "project_id": project,
                                               "project_context_limit": 5}, headers=employee).json()
    assert "4242" not in json.dumps(j["pipeline_trace"])
    assert j["usage_metadata"]["context_used"] in ("None", []) or all(
        "secret" not in t.lower() for t in j["usage_metadata"]["context_used"])
