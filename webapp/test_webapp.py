"""Self-test for the web app:  .venv/bin/python -m webapp.test_webapp"""
import base64
import http.client
import json
import threading
import time

import numpy as np

from src import cfrfs
from webapp import server

PW = "test-password-123456"
AUTH = {"Authorization": "Basic " + base64.b64encode(f"cfrfs:{PW}".encode()).decode()}


def get(port, path, headers=AUTH):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("GET", path, headers=headers)
    return c, c.getresponse()


def events(resp):
    """Yield (event, data) from an SSE response until it closes."""
    ev = None
    for raw in resp:
        line = raw.decode().rstrip("\n")
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            yield ev, json.loads(line[6:])


httpd = server.make_server("127.0.0.1", 0, PW)
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()

# 1. auth
assert get(port, "/", {})[1].status == 401
bad = {"Authorization": "Basic " + base64.b64encode(b"cfrfs:wrong").decode()}
assert get(port, "/", bad)[1].status == 401
c, r = get(port, "/")
assert r.status == 200 and b"C-FRFS" in r.read()

c, r = get(port, "/logo.svg")
assert r.status == 200 and r.getheader("Content-Type") == "image/svg+xml" and b"<svg" in r.read()
assert get(port, "/logo.svg", {})[1].status == 401

# 2. results + path traversal
c, r = get(port, "/api/results?set=nslkdd")
d = json.loads(r.read())
assert r.status == 200 and len(d["rows"]) == 14 and "cfrfs" in d["stability"], d.keys()
assert get(port, "/api/results?set=../etc")[1].status == 400
assert get(port, "/figures/nslkdd/..%2f..%2fCLAUDE.md")[1].status == 404
c, r = get(port, "/figures/nslkdd/" + d["figures"][0])
assert r.status == 200 and r.read()[:4] == b"\x89PNG"

# 6. bad parameters
assert get(port, "/api/demo?dataset=nslkdd&k=99")[1].status == 400
assert get(port, "/api/demo?dataset=nope")[1].status == 400

# 3 + 4. a 3-run demo streams 3 run events then done, and matches cfrfs() with the same seed
c, r = get(port, "/api/demo?dataset=nslkdd&k=12&runs=3&stratify=1&seed=0")
assert r.status == 200
evs = list(events(r))
names = [e for e, _ in evs]
assert names.count("run") == 3 and names[-1] == "done", names
done = evs[-1][1]
X, y, feat = server.get_data("nslkdd")
mask, freq, masks = cfrfs.cfrfs(X, y, k=12, n_runs=3, n_boot=300, seed=0, stratify=True)
assert done["selected"] == np.flatnonzero(mask).tolist(), (done["selected"], np.flatnonzero(mask))
assert done["names"] == list(feat)

# 5. a second demo while one is running gets 409; closing the first frees the lock
c1, r1 = get(port, "/api/demo?dataset=nslkdd&runs=30&seed=1")
assert r1.status == 200
assert get(port, "/api/demo?dataset=nslkdd&runs=3")[1].status == 409
c1.close()
r1.close()
for _ in range(60):
    if not server._demo_lock.locked():
        break
    time.sleep(1)
assert not server._demo_lock.locked(), "lock not released after the browser left"

print("webapp ACCEPTANCE: PASS")
