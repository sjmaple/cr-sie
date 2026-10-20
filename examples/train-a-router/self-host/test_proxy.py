"""Exercise the real HAProxy boundary against local HTTP fixtures, without models."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import socket
import subprocess
import threading
import time
import unittest
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHEAP = "Qwen/Qwen3.5-4B"
STRONG = "Qwen/Qwen3.8-27B-FP8"
ENCODER = "Qwen/Qwen3-Embedding-4B"
CALLS = []


class Fixture(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply()

    def do_POST(self):
        self.reply()

    def reply(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        record = {
            "worker": self.server.server_port,
            "path": self.path,
            "sha256": hashlib.sha256(body).hexdigest(),
            "headers": dict(self.headers),
        }
        CALLS.append(record)
        payload = json.dumps(record).encode()
        self.send_response(429 if self.headers.get("X-Fixture-Error") else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-SIE-Model-Revision", "fixture-revision")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


def request(method, path, body=None, headers=None):
    with closing(http.client.HTTPConnection("127.0.0.1", 8080, timeout=5)) as client:
        client.request(method, path, body, headers or {})
        reply = client.getresponse()
        return reply.status, dict(reply.getheaders()), reply.read()


def raw_request(data):
    with socket.create_connection(("127.0.0.1", 8080), timeout=5) as client:
        client.sendall(data)
        client.shutdown(socket.SHUT_WR)
        reply = client.recv(65536)
    return int(reply.split(b" ", 2)[1]) if reply else 0


class ProxyBoundary(unittest.TestCase):
    def chat(self, body):
        return request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})

    def rejected(self, method, path, body=None, headers=None):
        before = len(CALLS)
        status, _, _ = request(method, path, body, headers)
        self.assertGreaterEqual(status, 400)
        self.assertEqual(len(CALLS), before)

    def test_exact_routes_and_untouched_body_and_revision(self):
        for model, port in ((CHEAP, 18081), (STRONG, 18082)):
            body = json.dumps({"model": model, "messages": [{"role": "user", "content": "2+2?"}]}).encode()
            status, headers, payload = self.chat(body)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["worker"], port)
            self.assertEqual(json.loads(payload)["sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual({k.lower(): v for k, v in headers.items()}["x-sie-model-revision"], "fixture-revision")

    def test_escaped_model_and_model_like_message(self):
        body = ('{"m\\u006fdel":"Qwen\\/Qwen3.5-4B",'
                '"messages":[{"role":"user","content":"\\\"model\\\":\\\"Qwen/Qwen3.8-27B-FP8\\\""}]}')
        status, _, payload = self.chat(body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["worker"], 18081)

    def test_duplicate_top_level_model_and_escaped_duplicate(self):
        for key in ('"model"', '"m\\u006fdel"'):
            for value in (CHEAP, STRONG):
                body = '{"model":"' + CHEAP + '",' + key + ':"' + value + '","messages":[]}'
                self.rejected("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})

    def test_missing_wrong_model_and_malformed_json(self):
        for body in ('{}', '[]', '{"model":null}', '{"model":123}', '{"model":"unknown"}',
                     '{"model":"' + ENCODER + '"}', '{"model":"' + CHEAP + ':no-spec"}',
                     '{"model":"' + CHEAP + '","temperature":NaN}', '{"model":"' + CHEAP + '",}'):
            self.rejected("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})

    def test_metadata_and_msgpack_encode(self):
        for model, port in ((CHEAP, 18081), (STRONG, 18082), (ENCODER, 18083)):
            status, _, payload = request("GET", "/v1/models/" + model)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["worker"], port)
        payload = b"\x81\xa5items\x90"
        status, _, result = request("POST", "/v1/encode/" + ENCODER, payload,
                                    {"Content-Type": "application/msgpack"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(result)["sha256"], hashlib.sha256(payload).hexdigest())

    def test_wrong_endpoint_method_and_query(self):
        for method, path in (("GET", "/v1/models"), ("GET", "/healthz"),
                             ("POST", "/v1/encode/" + CHEAP), ("POST", "/v1/models/" + CHEAP),
                             ("GET", "/v1/chat/completions"), ("DELETE", "/v1/models/" + CHEAP),
                             ("GET", "/v1/models/Qwen%2FQwen3.5-4B"),
                             ("GET", "/v1/models/" + CHEAP + "?upstream=evil.invalid")):
            self.rejected(method, path)

    def test_credentials_and_forwarding_headers_removed(self):
        supplied = {"Authorization": "Bearer fake-fixture", "X-API-Key": "fake-fixture",
                    "Proxy-Authorization": "fake-fixture", "Cookie": "fixture=1",
                    "Forwarded": "host=evil.invalid", "X-Forwarded-Host": "evil.invalid"}
        status, _, payload = request("GET", "/v1/models/" + CHEAP, headers=supplied)
        self.assertEqual(status, 200)
        forwarded = {k.lower() for k in json.loads(payload)["headers"]}
        self.assertFalse(forwarded.intersection(k.lower() for k in supplied))

    def test_oversized_body_and_headers(self):
        self.rejected("POST", "/v1/chat/completions", b" " * 131073, {"Content-Type": "application/json"})
        self.rejected("GET", "/v1/models/" + CHEAP, headers={"X-Large": "x" * 9000})

    def test_upstream_error_status_body_and_revision_are_preserved(self):
        before = len(CALLS)
        status, headers, payload = request("GET", "/v1/models/" + CHEAP, headers={"X-Fixture-Error": "1"})
        self.assertEqual(status, 429)
        self.assertEqual(len(CALLS), before + 1)
        self.assertEqual(json.loads(payload)["worker"], 18081)
        self.assertEqual({k.lower(): v for k, v in headers.items()}["x-sie-model-revision"], "fixture-revision")

    def test_incomplete_chunked_conflicting_and_missing_lengths(self):
        start = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
        for suffix in (b"Content-Length: 100\r\n\r\n{}",
                       b"Transfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
                       b"Content-Length: 2\r\nContent-Length: 3\r\n\r\n{}",
                       b"\r\n{}"):
            before = len(CALLS)
            status = raw_request(start + suffix)
            self.assertTrue(status == 0 or status >= 400)
            self.assertEqual(len(CALLS), before)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("haproxy.cfg"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing test receipts; choose a new output")
    subprocess.run(["haproxy", "-c", "-f", str(args.config)], check=True)
    servers = [ThreadingHTTPServer(("127.0.0.1", port), Fixture) for port in (18081, 18082, 18083)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    started = time.monotonic()
    process = subprocess.Popen(["haproxy", "-db", "-f", str(args.config)])
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("HAProxy exited before validation")
            try:
                with socket.create_connection(("127.0.0.1", 8080), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProxyBoundary))
        receipt = {"format": "sie-self-host-proxy-fixtures-v1", "tests": result.testsRun,
                   "failures": len(result.failures), "errors": len(result.errors),
                   "elapsed_seconds": time.monotonic() - started, "model_inference": False,
                   "haproxy": subprocess.check_output(["haproxy", "-v"], text=True).strip(),
                   "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (Path(__file__), args.config, args.config.with_name("chat-model.lua"))}}
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        if not result.wasSuccessful():
            raise SystemExit(1)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        for server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
