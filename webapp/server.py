"""C-FRFS web app: results viewer + live demo. Standard library only.

    .venv/bin/python -m webapp.server --host 172.17.0.1 --port 8094

Every route needs HTTP Basic auth (user `cfrfs`, password from --password-file).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hmac
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from src import cfrfs, cici, data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
USER = "cfrfs"

# result set -> (folder, file prefix)
SETS = {
    "nslkdd": ("results_fixed", "nslkdd"),
    "cici_group": ("results_fixed_cici2023_group", "cici2023"),
    "cici_binary": ("results_fixed_cici2023_binary", "cici2023"),
}
FIG_RE = re.compile(r"^fig[0-9a-z_]+\.png$")

_demo_lock = threading.Lock()      # one demo at a time
_data_lock = threading.Lock()
_data_cache: dict = {}             # dataset -> (X, y, names); ~6 MB each after the 20k cap


def get_data(dataset: str):
    with _data_lock:
        if dataset not in _data_cache:
            if dataset == "nslkdd":
                _data_cache[dataset] = data.load_nslkdd(
                    os.path.join(ROOT, "data", "KDDTrain.txt"), cap=20000, seed=0)
            else:
                _data_cache[dataset] = cici.load_cici2023(
                    os.path.join(ROOT, "data", "cici2023"), cap=20000, seed=0, labels="group")
        return _data_cache[dataset]


def _intarg(q, name, lo, hi, default):
    v = int(q.get(name, [default])[0])
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be {lo}..{hi}")
    return v


def read_results(set_name: str) -> dict:
    folder, prefix = SETS[set_name]
    d = os.path.join(ROOT, folder)

    def js(n):
        with open(os.path.join(d, n), encoding="utf-8") as f:
            return json.load(f)

    def cs(n):
        with open(os.path.join(d, n), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    figs = sorted(n for n in os.listdir(os.path.join(d, "figures")) if FIG_RE.match(n))
    return {"rows": js(f"{prefix}_results.json"), "stability": js(f"{prefix}_stability.json"),
            "paired": cs(f"{prefix}_paired.csv"), "attack_map": cs(f"{prefix}_attack_map.csv"),
            "figures": figs}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):          # never log headers; keep the journal short
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    # ---- helpers
    def _send(self, code, body: bytes, ctype="application/json", extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode())

    def _authed(self) -> bool:
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(h[6:]).decode().partition(":")
            except Exception:
                return False
            ok_user = hmac.compare_digest(user.encode(), USER.encode())
            ok_pw = hmac.compare_digest(pw.encode(), self.server.password.encode())
            return ok_user and ok_pw
        return False

    def _sse(self, event, obj):
        self.wfile.write(f"event: {event}\ndata: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    # ---- routes
    def do_GET(self):
        if not self._authed():
            return self._send(401, b'{"error":"login required"}',
                              extra=[("WWW-Authenticate", 'Basic realm="C-FRFS"')])
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/favicon.ico":
                return self._send(204, b"")
            if u.path == "/logo.svg":
                with open(os.path.join(HERE, "logo.svg"), "rb") as f:
                    return self._send(200, f.read(), "image/svg+xml", [("Cache-Control", "max-age=3600")])
            if u.path == "/fonts/lexend-latin.woff2":
                with open(os.path.join(HERE, "fonts", "lexend-latin.woff2"), "rb") as f:
                    return self._send(200, f.read(), "font/woff2", [("Cache-Control", "max-age=2592000")])
            if u.path == "/api/results":
                s = q.get("set", [""])[0]
                if s not in SETS:
                    return self._json(400, {"error": "unknown set"})
                return self._json(200, read_results(s))
            m = re.fullmatch(r"/figures/([a-z_0-9]+)/([^/]+)", u.path)
            if m and m.group(1) in SETS and FIG_RE.match(m.group(2)):
                with open(os.path.join(ROOT, SETS[m.group(1)][0], "figures", m.group(2)), "rb") as f:
                    return self._send(200, f.read(), "image/png")
            if u.path == "/api/demo":
                return self._demo(q)
        except FileNotFoundError:
            return self._json(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):    # browser cancelled the download
            return
        self._json(404, {"error": "not found"})

    def _demo(self, q):
        try:
            dataset = q.get("dataset", [""])[0]
            if dataset not in ("nslkdd", "cici2023"):
                raise ValueError("unknown dataset")
            k = _intarg(q, "k", 4, 20, 12)
            runs = _intarg(q, "runs", 3, 30, 10)
            stratify = bool(_intarg(q, "stratify", 0, 1, 1))
            seed = _intarg(q, "seed", 0, 1_000_000, 0)
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        if not _demo_lock.acquire(blocking=False):
            return self._json(409, {"busy": True})
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            if dataset not in _data_cache:
                self._sse("status", {"msg": "loading data (first time only, up to 30 s)"})
            X, y, names = get_data(dataset)
            self._sse("start", {"names": list(names), "runs": runs})

            def progress(i, n, freq, masks):          # a failed write (browser left) aborts the run
                self._sse("run", {"i": i, "runs": n, "freq": np.round(freq, 4).tolist()})

            mask, freq, masks = cfrfs.cfrfs(X, y, k=k, n_runs=runs, n_boot=300, seed=seed,
                                            stratify=stratify, progress=progress)
            self._sse("done", {"selected": np.flatnonzero(mask).tolist(), "names": list(names),
                               "freq": np.round(freq, 4).tolist(),
                               "jaccard": round(cfrfs.mean_jaccard(masks), 4)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:                         # shown in the page, details stay in the journal
            sys.stderr.write(f"demo error: {e!r}\n")
            try:
                self._sse("error", {"msg": "the demo failed, please try again"})
            except Exception:
                pass
        finally:
            _demo_lock.release()


def make_server(host: str, port: int, password: str) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.password = password
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.17.0.1")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--password-file", default="/root/.cfrfs-app-password")
    a = ap.parse_args()
    with open(a.password_file, encoding="utf-8") as f:
        pw = f.read().strip()
    if len(pw) < 12:
        sys.exit("password file must hold at least 12 characters")
    httpd = make_server(a.host, a.port, pw)
    print(f"serving on http://{a.host}:{a.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
