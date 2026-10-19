from __future__ import annotations

import hashlib
import json


def pipeline_fingerprint(pipeline_config: dict, data_hash: str, code_hashes: dict,
                         runtime_versions: dict) -> str:
    payload = {"pipeline_config": pipeline_config, "data_sha256": data_hash,
               "code_sha256": code_hashes, "runtime_versions": runtime_versions}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
