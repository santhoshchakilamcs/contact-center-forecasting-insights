#!/usr/bin/env python3
"""Serve the dashboard: stdlib http.server only, no extra dependencies.

  python3 app.py            -> http://localhost:8000/ (dashboard)
  http://localhost:8000/metrics  -> reports/metrics.json
"""
import http.server
import json
import os
import socketserver

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/dashboard.html"
            self.directory = REPORTS
            return super().do_GET()
        if self.path == "/metrics":
            with open(os.path.join(REPORTS, "metrics.json"), "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"serving dashboard at http://127.0.0.1:{port}/  (Ctrl-C to stop)")
        httpd.serve_forever()
