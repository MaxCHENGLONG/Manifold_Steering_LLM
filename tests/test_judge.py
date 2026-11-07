import json

import pytest

from scripts.judge import TEMPLATES, build_prompt, parse_grade, summarize, upstream_template


def test_upstream_template_has_fields():
    t = upstream_template()
    assert "{instruction}" in t and "{response}" in t and "{true_answer}" in t and t.rstrip().endswith("GRADE:")


@pytest.mark.parametrize("reply,grade", [("CORRECT", "CORRECT"), (" INCORRECT\n", "INCORRECT"),
                                         ("Grade: incorrect.", "INCORRECT"), ("correct", "CORRECT"),
                                         ("I cannot tell", None), (None, None)])
def test_parse_grade(reply, grade):
    assert parse_grade(reply) == grade


def test_build_prompt():
    p = build_prompt(upstream_template(), {"instruction": "Q?", "response": "A.", "correct_answer": "T"})
    assert "QUESTION: Q?\nSTUDENT ANSWER: A.\nTRUE ANSWER: T\nGRADE:" in p


def test_summarize(tmp_path):
    d = tmp_path / "weight_k+2"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"method": "weight", "strength": 2.0}))
    tpl = list(TEMPLATES)
    recs = [{"example": i, "template": tpl[i % 4]} for i in range(8)]
    (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    grades = ["CORRECT", "INCORRECT", "CORRECT", None, "CORRECT", "CORRECT", "INCORRECT", "CORRECT"]
    (d / "judge.jsonl").write_text("".join(json.dumps({"example": i, "grade": g}) + "\n" for i, g in enumerate(grades)))
    s = summarize(tmp_path, ["weight_k+2"])["weight_k+2"]
    assert s["accuracy"] == {"plain": 1.0, "doubt_correct": 0.5, "suggest_incorrect": 0.5, "suggest_correct": 0.5}
    assert s["ungraded"] == {"suggest_correct": 1} and s["n_judged"] == 8


def test_chat_retries_then_succeeds(monkeypatch):
    """本机起一个假的 OpenAI 兼容服务：第一次返回 429，第二次返回结果；检查请求格式和重试。"""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import scripts.judge as judge
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers["Authorization"], body))
            if len(seen) == 1:
                self.send_response(429)
                self.end_headers()
                return
            out = json.dumps({"choices": [{"message": {"content": "INCORRECT"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(judge.time, "sleep", lambda s: None)
    try:
        reply = judge.chat(f"http://127.0.0.1:{server.server_port}/v1", "m", "k", "hi", 1.0, 8)
    finally:
        server.shutdown()
    assert reply == "INCORRECT" and len(seen) == 2
    path, auth, body = seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer k"
    assert body == {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0, "max_tokens": 8}
