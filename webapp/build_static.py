"""Build a static copy of the web app (no Python server, no password, no live runs) into webapp/dist/.

    .venv\\Scripts\\python.exe -m webapp.build_static

dist/index.html             the page, with the API calls pointed at the files below
dist/results/<set>.json     what /api/results?set=<set> returns
dist/figures/<set>/*.png    the charts
dist/fonts/                 Lexend, served with the page (no Google Fonts request)
"""
from __future__ import annotations

import json
import os
import shutil

from webapp import server

DIST = os.path.join(server.HERE, "dist")


def main():
    if os.path.isdir(DIST):
        shutil.rmtree(DIST)
    os.makedirs(os.path.join(DIST, "results"))

    with open(os.path.join(server.HERE, "index.html"), encoding="utf-8") as f:
        html = f.read()
    swaps = [("fetch(location.origin + '/api/results?set=' + set)", "fetch('results/' + set + '.json')"),
             ('href="/figures/${set}/', 'href="figures/${set}/'),
             ('src="/figures/${set}/', 'src="figures/${set}/')]
    for old, new in swaps:
        if old not in html:
            raise SystemExit(f"index.html changed, cannot find: {old}")
        html = html.replace(old, new)
    with open(os.path.join(DIST, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(html)

    for name, (folder, _) in server.SETS.items():
        res = server.read_results(name)
        with open(os.path.join(DIST, "results", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, separators=(",", ":"))
        out = os.path.join(DIST, "figures", name)
        os.makedirs(out)
        for fig in res["figures"]:
            shutil.copy2(os.path.join(server.ROOT, folder, "figures", fig), out)

    shutil.copytree(os.path.join(server.HERE, "fonts"), os.path.join(DIST, "fonts"))

    n = sum(len(fs) for _, _, fs in os.walk(DIST))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(DIST) for x in fs)
    print(f"wrote {n} files, {size / 1e6:.1f} MB, to {DIST}")


if __name__ == "__main__":
    main()
