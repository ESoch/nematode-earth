"""Run with `uv run --no-sync python -m marine fetch|experiment` from pipeline/."""
import argparse
import json
from pathlib import Path

import requests

from .sources import discover_pangaea, discover_wei, download, verify_download


def fetch(out):
    raw = out / "raw"
    manifest_path = out / "sources.v1.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["sources"]:
            for key in ("landing", "data"):
                verify_download(out / item[key]["path"], item[key])
        print("Verified existing source cache; no downloads repeated.")
        return
    manifest = {"schema_version": 1, "sources": []}
    with requests.Session() as session:
        urls = [(i, f"https://doi.pangaea.de/10.1594/PANGAEA.{i}", discover_pangaea)
                for i in ("873272", "973147", "776716")]
        urls.append(("wei2010", "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0015323", discover_wei))
        for name, url, discover in urls:
            landing_path = raw / f"{name}.html"
            landing = download(session, url, landing_path, limit=5_000_000)
            source = discover(landing_path.read_text(), url)
            data_path = raw / (f"{name}.kml" if name == "wei2010" else f"{name}.tsv")
            data = download(session, source["download_url"], data_path)
            source.update(id=name, landing=dict(**landing, path=str(landing_path.relative_to(out))),
                          data=dict(**data, path=str(data_path.relative_to(out))))
            manifest["sources"].append(source)
            print(f"Fetched {name}: {data['bytes']} bytes, sha256 {data['sha256']}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "experiment"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "out/marine")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.out)
    else:
        from .experiment import run
        run(args.out)


if __name__ == "__main__":
    main()
