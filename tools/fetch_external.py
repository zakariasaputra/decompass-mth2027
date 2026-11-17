"""Download the external (other-game) training and calibration datasets into ./external/.

These matches are used ONLY for external model training and weight calibration (see tools/train_super_hybrid.py
and tools/sweep_ensemble_weights.py). Nothing from the provided BAR-ATH match is uploaded anywhere, and nothing
here is needed to regenerate the submission -- run_pipeline.py only needs ./data plus frozen weights/configs.

Datasets
--------
1. IDSSE -- Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event
   data in elite soccer", Scientific Data. figshare DOI 10.6084/m9.figshare.28196177. CC BY 4.0.
   7 Bundesliga / 2. Bundesliga matches, TRACAB optical tracking at 25 fps.
2. Metrica Sports sample data -- https://github.com/metrica-sports/sample-data. 3 anonymised matches,
   25 fps tracking synchronised with events. No formal licence; README asks users to "be
   responsible" and acknowledge the source.

Usage:
    python tools/fetch_external.py               # everything (~2.6 GB)
    python tools/fetch_external.py --only metrica
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXT = os.path.join(ROOT, "external")
UA = {"User-Agent": "mtah2027-external-fetch"}

IDSSE_ARTICLE = "https://api.figshare.com/v2/articles/28196177"
METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/"
METRICA_FILES = [
    "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "data/Sample_Game_3/Sample_Game_3_events.json",
    "data/Sample_Game_3/Sample_Game_3_metadata.xml",
    "data/Sample_Game_3/Sample_Game_3_tracking.txt",
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest, expected_size=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and (expected_size is None or os.path.getsize(dest) == expected_size):
        print(f"  [skip] {os.path.relpath(dest, ROOT)} already present")
        return
    tmp = dest + ".part"
    t0 = time.time()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    mb = os.path.getsize(dest) / 1e6
    print(f"  [ok]   {os.path.relpath(dest, ROOT)}  {mb:.1f} MB in {time.time() - t0:.0f}s", flush=True)


def fetch_idsse():
    req = urllib.request.Request(IDSSE_ARTICLE, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        meta = json.load(r)
    print(f"IDSSE: {len(meta['files'])} files, licence: {meta['license']['name']}")
    for f in sorted(meta["files"], key=lambda x: x["size"]):
        _download(f["download_url"], os.path.join(EXT, "idsse", f["name"]), expected_size=f["size"])


def fetch_metrica():
    print(f"Metrica sample data: {len(METRICA_FILES)} files")
    for rel in METRICA_FILES:
        _download(METRICA_RAW + rel, os.path.join(EXT, "metrica", rel.replace("data/", "")))


def write_manifest():
    manifest = {}
    for dirpath, _, files in os.walk(EXT):
        for name in files:
            if name.endswith((".part", ".npz")) or name in ("MANIFEST.json", "LICENSES.md"):
                continue
            p = os.path.join(dirpath, name)
            manifest[os.path.relpath(p, EXT).replace("\\", "/")] = {
                "bytes": os.path.getsize(p),
                "sha256": _sha256(p),
            }
    with open(os.path.join(EXT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sorted(manifest.items())), f, indent=2)
    print(f"Wrote external/MANIFEST.json ({len(manifest)} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["idsse", "metrica"])
    args = ap.parse_args()
    if args.only in (None, "metrica"):
        fetch_metrica()
    if args.only in (None, "idsse"):
        fetch_idsse()
    write_manifest()
    sys.exit(0)

