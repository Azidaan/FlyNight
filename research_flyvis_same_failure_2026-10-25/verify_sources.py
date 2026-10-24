"""Freeze the local evidence used by the targeted literature comparison.

This checks already saved artifacts only; it performs no tracking calculation.
"""
from datetime import datetime, timezone, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged


def main():
    folder = Path(__file__).resolve().parent
    output = folder / "local_evidence_audit.json"
    if output.exists():
        raise FileExistsError("Preserve the first audit; use a new version for another one")
    diagnosis = json.loads((ROOT / "runs/z13_refusal_causes_20261025/diagnosis.json").read_text(encoding="utf-8"))
    explanation = json.loads((ROOT / "runs/z13_refusal_causes_20261025/explanation.json").read_text(encoding="utf-8"))
    assert diagnosis["D_selected_known"] == 64
    assert diagnosis["D_selected_beneficial"] == 0
    assert diagnosis["D_selected_unknown"] == 20
    assert diagnosis["D_missed_beneficial_top1"] == 48
    assert explanation["selected_D_negative_reasons"]["target_not_annotated_in_frame"] == 62
    names = [
        "runs/z13_refusal_causes_20261025/diagnosis.json",
        "runs/z13_refusal_causes_20261025/explanation.json",
        "runs/z13_region_readout_20261025/recovery1/report.json",
        "scripts/z13_train.py", "tracking/z13_candidate_ranker.py",
        "tracking/z13_spatial_localizer.py", "tracking/z13_ridge_localizer.py",
        "research_flyvis_same_failure_2026-10-25/СВОДКА.md",
        "research_flyvis_same_failure_2026-10-25/verify_sources.py",
    ]
    hashes = {}
    for name in names:
        path = (ROOT / name).resolve()
        path.relative_to(ROOT.resolve())
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    record = {
        "at": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "scope": "Targeted primary-source literature comparison; no new experiment",
        "local_claims_verified": True,
        "sources_sha256": hashes,
        "protected_foreign_files": len(foreign_unchanged()),
        "external_sources": [
            {"url": "https://arxiv.org/pdf/2001.02593", "version": "v1", "locations": ["section 4", "figures 4-5", "section 5"], "status": "preprint; read primary text"},
            {"url": "https://arxiv.org/html/2105.11595v1", "version": "v1", "locations": ["section 3.4", "section 5.2, table 2", "appendix C, table 7"], "status": "primary text and ablation checked"},
            {"url": "https://arxiv.org/html/2103.16556v2", "version": "v2", "locations": ["section 3", "section 4.1, table 1", "appendix F"], "status": "primary method, ablation and remaining failures checked"},
        ],
        "external_content_sha256": None,
        "external_hash_note": "Remote pages inspected with the browser; no local full-text snapshot or hash is claimed.",
        "cause_in_our_current_readout_proven": False,
        "scientific_calculations_started": False,
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"local_claims_verified": True, "sources": len(hashes), "foreign_preserved": record["protected_foreign_files"]}))


if __name__ == "__main__":
    main()
