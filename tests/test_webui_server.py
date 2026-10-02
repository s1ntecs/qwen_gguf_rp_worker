import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "webui"))

import server  # noqa: E402


class FakeClient:
    def __init__(self):
        self.runs = []
        self.cancelled = []

    def health(self):
        return {"workers": {"idle": 1}, "jobs": {"inQueue": 0}}

    def run(self, payload):
        self.runs.append(payload)
        return {"id": "job-1", "status": "IN_QUEUE"}

    def status(self, job_id):
        if job_id == "conflict":
            raise server.RunPodError(409, {"detail": "paused"})
        return {"id": job_id, "status": "COMPLETED", "output": {"images_base64": ["AAAA"]}}

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        return {"id": job_id, "status": "CANCELLED"}


class LoadEnvTests(unittest.TestCase):
    def test_parses_quotes_comments_and_blanks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('# comment\n\nA=1\nB = "two"\nC=\'three\'\nD=x=y\n', encoding="utf-8")
            self.assertEqual(server.load_env(path), {"A": "1", "B": "two", "C": "three", "D": "x=y"})

    def test_missing_file_is_empty(self):
        self.assertEqual(server.load_env(Path("definitely-missing.env")), {})


class JobIdTests(unittest.TestCase):
    def test_accepts_runpod_ids(self):
        self.assertTrue(server.valid_job_id("d9e0474e-45a0-4e1a-bdda-f55ffe6217d5-e1"))

    def test_rejects_path_tricks(self):
        for bad in ["", "../health", "a/b", "a b", "x" * 200]:
            self.assertFalse(server.valid_job_id(bad), bad)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        Path(self.tmp.name, "index.html").write_text("<html>ui</html>", encoding="utf-8")
        self.client = FakeClient()
        self.httpd = server.make_server(self.client, "127.0.0.1", 0, Path(self.tmp.name))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def request(self, method, path, body=None):
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.headers.get("Content-Type"), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type"), exc.read()

    def test_serves_index(self):
        code, ctype, body = self.request("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertEqual(body, b"<html>ui</html>")

    def test_health(self):
        code, _, body = self.request("GET", "/api/health")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["workers"]["idle"], 1)

    def test_run_forwards_input(self):
        code, _, body = self.request("POST", "/api/run", {"input": {"prompt": "cat"}})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["id"], "job-1")
        self.assertEqual(self.client.runs, [{"input": {"prompt": "cat"}}])

    def test_run_rejects_bad_json(self):
        code, _, _ = self.request("POST", "/api/run", b"{not json")
        self.assertEqual(code, 400)
        self.assertEqual(self.client.runs, [])

    def test_run_requires_input_object(self):
        code, _, _ = self.request("POST", "/api/run", {"prompt": "cat"})
        self.assertEqual(code, 400)

    def test_status_and_cancel(self):
        code, _, body = self.request("GET", "/api/status/job-1")
        self.assertEqual(json.loads(body)["status"], "COMPLETED")
        code, _, body = self.request("POST", "/api/cancel/job-1")
        self.assertEqual(code, 200)
        self.assertEqual(self.client.cancelled, ["job-1"])

    def test_invalid_job_id(self):
        code, _, _ = self.request("GET", "/api/status/bad!id")
        self.assertEqual(code, 400)

    def test_upstream_error_is_passed_through(self):
        code, _, body = self.request("GET", "/api/status/conflict")
        self.assertEqual(code, 409)
        self.assertEqual(json.loads(body)["detail"], "paused")

    def test_unknown_path(self):
        code, _, _ = self.request("GET", "/nope")
        self.assertEqual(code, 404)


if __name__ == "__main__":
    unittest.main()
