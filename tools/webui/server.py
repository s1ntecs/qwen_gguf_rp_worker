"""Local web UI for manual tests of the RunPod endpoint.

The RunPod API key stays in this process; the browser only talks to
127.0.0.1 and never sees it.

    python tools/webui/server.py [--port 8787]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent
DEFAULT_ENDPOINT_ID = "kpppmrdo844hmo"
MAX_BODY = 64 * 1024 * 1024
JOB_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,128}$")


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return env
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def valid_job_id(job_id: str) -> bool:
    return bool(JOB_ID_RE.match(job_id))


class RunPodError(Exception):
    def __init__(self, status: int, payload: Any):
        super().__init__(f"RunPod returned {status}")
        self.status = status
        self.payload = payload


class RunPodClient:
    def __init__(self, api_key: str, endpoint_id: str, timeout: int = 30):
        self.api_key = api_key
        self.base = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {"error": raw.decode("utf-8", "replace")[:2000]}
            raise RunPodError(exc.code, payload) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RunPodError(502, {"error": f"RunPod unreachable: {exc}"}) from exc

    def health(self) -> Any:
        return self._call("GET", "/health")

    def run(self, payload: dict) -> Any:
        return self._call("POST", "/run", payload)

    def status(self, job_id: str) -> Any:
        return self._call("GET", f"/status/{job_id}")

    def cancel(self, job_id: str) -> Any:
        return self._call("POST", f"/cancel/{job_id}")


def make_server(client: Any, host: str, port: int, static_dir: Path = STATIC_DIR) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            if not self.path.startswith(("/api/status", "/api/health")):
                super().log_message(fmt, *args)

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass  # the browser gave up on this poll; the next one retries

        def _proxy(self, fn, *args: Any) -> None:
            try:
                self._json(200, fn(*args))
            except RunPodError as exc:
                self._json(exc.status, exc.payload)

        def _job_id(self, prefix: str) -> str | None:
            job_id = self.path[len(prefix):]
            if not valid_job_id(job_id):
                self._json(400, {"error": "invalid job id"})
                return None
            return job_id

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = (Path(static_dir) / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._proxy(client.health)
            elif self.path.startswith("/api/status/"):
                job_id = self._job_id("/api/status/")
                if job_id:
                    self._proxy(client.status, job_id)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path == "/api/run":
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    self._json(413, {"error": "request too large"})
                    return
                try:
                    payload = json.loads(self.rfile.read(length))
                except ValueError:
                    self._json(400, {"error": "invalid JSON"})
                    return
                if not isinstance(payload, dict) or not isinstance(payload.get("input"), dict):
                    self._json(400, {"error": "body must be {\"input\": {...}}"})
                    return
                self._proxy(client.run, payload)
            elif self.path.startswith("/api/cancel/"):
                job_id = self._job_id("/api/cancel/")
                if job_id:
                    self._proxy(client.cancel, job_id)
            else:
                self._json(404, {"error": "not found"})

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--endpoint", help="RunPod endpoint id (default: RUNPOD_ENDPOINT_ID from .env)")
    args = parser.parse_args()

    env = {**load_env(ROOT / ".env"), **os.environ}
    api_key = env.get("RUNPOD_API_KEY") or env.get("Authorization_RUNPOD")
    if not api_key:
        raise SystemExit("RunPod API key not found: set Authorization_RUNPOD or RUNPOD_API_KEY in .env")
    endpoint_id = args.endpoint or env.get("RUNPOD_ENDPOINT_ID") or DEFAULT_ENDPOINT_ID

    httpd = make_server(RunPodClient(api_key, endpoint_id), "127.0.0.1", args.port)
    print(f"Endpoint {endpoint_id} -> http://127.0.0.1:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
