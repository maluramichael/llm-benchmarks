"""Tiny threaded HTTP server for adapter tests.

Routes map a path to a reply:
  dict          sent as JSON
  str           sent as is (for example ndjson)
  bytes         sent as is (for example a zip archive)
  (302, url)    redirect to url
  list          streamed, items are text or (delay_seconds, text), flushed one by one
  callable      called with the decoded request body, returns one of the above
Every request is recorded in `requests` as (method, path, decoded JSON body or None).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeServer:
    def __init__(self, routes: dict | None = None):
        self.routes = dict(routes or {})
        self.requests: list[tuple[str, str, object]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                outer.requests.append((self.command, self.path, body))
                reply = outer.routes.get(self.path)
                if callable(reply):
                    reply = reply(body)
                if reply is None:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error":"not found"}')
                    return
                if isinstance(reply, tuple) and reply[0] == 302:
                    self.send_response(302)
                    self.send_header("Location", reply[1])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                if isinstance(reply, dict):
                    self.wfile.write(json.dumps(reply).encode())
                elif isinstance(reply, str):
                    self.wfile.write(reply.encode())
                elif isinstance(reply, bytes):
                    self.wfile.write(reply)
                else:
                    for item in reply:
                        delay, text = item if isinstance(item, tuple) else (0, item)
                        time.sleep(delay)
                        self.wfile.write(text.encode())
                        self.wfile.flush()

            do_GET = do_POST = _handle

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
