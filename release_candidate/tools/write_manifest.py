"""Write a SHA-256 inventory for the release-candidate package."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.json"


def main() -> None:
    files = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p != MANIFEST):
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        files.append({
            "path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    manifest = {
        "source_commit": (ROOT / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.name}: {len(files)} files")


if __name__ == "__main__":
    main()
