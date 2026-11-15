"""Tiny local server for Meow Meow.  Run:  python serve.py   (then open http://localhost:8000)
 - "/" opens CLICK_HERE.html directly
 - /__models lists every model it finds in any folder (up to 2 levels down), whatever the folder is called
A model = manifest.json + weights.bin  (or NAME.json + NAME.bin)  and is named after its folder (or NAME)."""
import http.server, socketserver, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
SKIP = {"node_modules", "__pycache__"}

def ls(rel):
    d = os.path.join(ROOT, rel) if rel else ROOT
    try: names = sorted(os.listdir(d))
    except OSError: return [], []
    files = [n for n in names if os.path.isfile(os.path.join(d, n))]
    dirs = [n for n in names if os.path.isdir(os.path.join(d, n)) and not n.startswith(".") and n not in SKIP]
    return files, dirs

def pairs(files, prefix, fallback):
    s, out = set(files), []
    if "manifest.json" in s and "weights.bin" in s:
        out.append({"name": fallback, "manifest": prefix + "manifest.json", "weights": prefix + "weights.bin"})
    for f in files:
        if not f.lower().endswith(".json") or f == "manifest.json": continue
        base = f[:-5]
        stem = base[:-len("manifest")].rstrip("._-") if base.lower().endswith("manifest") else base
        for w in (base + ".bin", stem + ".weights.bin", stem + "_weights.bin", stem + "-weights.bin", stem + ".bin"):
            if w in s:
                out.append({"name": stem or base, "manifest": prefix + f, "weights": prefix + w}); break
    return out

def find_models():
    found = []
    def walk(path, depth):
        files, dirs = ls(path)
        found.extend(pairs(files, path + "/", os.path.basename(path)))
        if depth:
            for d in dirs: walk(path + "/" + d, depth - 1)
    files, dirs = ls("")
    found.extend(pairs(files, "", "model"))
    for d in dirs: walk(d, 1)
    seen, out = set(), []
    for m in found:
        if m["name"] in seen: m["name"] = os.path.dirname(m["manifest"]) or m["name"]
        seen.add(m["name"]); out.append(m)
    return sorted(out, key=lambda m: m["name"].lower())

class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            self.send_response(302); self.send_header("Location", "/CLICK_HERE.html"); self.end_headers(); return
        if p == "/__models":
            body = json.dumps(find_models()).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body); return
        super().do_GET()

socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("", PORT), H) as srv:
    print(f"Meow Meow running at http://localhost:{PORT}  (Ctrl+C to stop)")
    for m in find_models(): print("  model:", m["name"])
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
