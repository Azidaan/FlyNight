"""Record technical repair evidence without changing Z13 research criteria."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "experiments/project_history.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    card = next(e for e in catalog["experiments"] if e["id"] == "fly_object_readout_z13_v1")
    source = "runs/z13_codex_repair_20261013/data_check.json"
    if not any(e.get("ref") == source for e in card["evidence"]):
        card["evidence"].append({"kind": "json", "ref": source,
            "locator": "passed, ownership_mismatches",
            "claim": "После исправления загрузчика все метки 130 train относятся к своим эпизодам; Т1 пройдена"})
    card["state"] = "running"
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
