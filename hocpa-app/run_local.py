import os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "api"))
from simulate import handler as ApiHandler

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=os.path.join(ROOT, "public"), **k)

    def do_GET(self):
        if self.path.startswith("/api/simulate"):
            return ApiHandler.do_GET(self)
        return super().do_GET()

if __name__ == "__main__":
    print("Open http://localhost:8000")
    ThreadingHTTPServer(("", 8000), H).serve_forever()