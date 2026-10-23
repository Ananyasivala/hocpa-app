import json, os, sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(__file__))
from _hocpa import run_point  # noqa: E402

MODES = {"central", "noncentral", "secure", "adaptive"}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        try:
            mode = q.get("mode", ["central"])[0]
            if mode not in MODES:
                raise ValueError("bad mode")
            traces = min(int(q.get("traces", [1000])[0]), 5000)   # caps keep us
            trials = min(int(q.get("trials", [20])[0]), 60)       # inside Vercel limits
            lam = float(q.get("lam", [0.5])[0])
            body, status = run_point(mode, traces, trials, lam), 200
        except Exception as e:
            body, status = {"error": str(e)}, 400
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)
