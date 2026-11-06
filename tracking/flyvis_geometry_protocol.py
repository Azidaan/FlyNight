"""Explicit geometry for secondary FlyVis experiments and historical artifacts."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from tracking.flyvis_motion import GEOMETRY_VERSION, GEOMETRY_VERSIONS

SCHEMA = "flyvis_motion_geometry_v1"


def add_geometry_arguments(parser):
    parser.add_argument("--geometry-version", choices=GEOMETRY_VERSIONS, default=GEOMETRY_VERSION,
                        help="legacy_v1 preserves current behavior; pixel_centers_v2 and uncalibrated resize_v2 are explicit research choices")
    parser.add_argument("--out", type=Path, help="New result file/directory inside the project")


def resolve_output(value, default, root):
    root = Path(root).resolve()
    path = Path(default if value is None else value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    path.relative_to(root)
    return path


def _known(version):
    if version not in GEOMETRY_VERSIONS:
        raise ValueError(f"Unknown motion geometry: {version!r}")
    return version


def require_empirical_gain_geometry(geometry_version, context):
    """Do not silently carry an entry point's historical gain into axis rescaling.

    This permits the unchanged vector convention; it does not certify ROI
    selection, trained weights, or tracker quality after point coordinates move.
    """
    if _known(geometry_version) == 'resize_v2':
        raise ValueError(f'{context}: resize_v2 is uncalibrated research geometry; '
                         'this entry point uses a historical fixed gain. '
                         'Use a separately specified unit-gain diagnostic protocol.')


def ensure_geometry_output(path, geometry_version, *, file=False):
    """Validate before writing; unversioned existing results are legacy_v1.

    A same-version directory can resume. A different version requires a distinct
    output. File sidecars do not claim ownership of unrelated sibling results.
    """
    path = Path(path)
    requested = _known(geometry_version)
    marker = path.with_name(path.name + ".geometry.json") if file else path / "geometry_protocol.json"
    inherited = False
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(saved, Mapping) or saved.get("schema") != SCHEMA:
            raise ValueError(f"Invalid geometry protocol: {marker}")
        found = _known(saved.get("geometry_version"))
    else:
        if path.exists() and path.is_file() != file:
            raise ValueError(f"Output file/directory kind mismatch: {path}")
        existing = path.exists() if file else path.exists() and any(path.iterdir())
        found = "legacy_v1" if existing else requested
        if existing and file and path.suffix != ".json":
            metadata = path.with_suffix(".json")
            if metadata.exists():
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                if isinstance(meta, Mapping) and "geometry_version" in meta:
                    found = _known(meta["geometry_version"])
        inherited = bool(existing)
    if found != requested:
        raise ValueError(f"{path}: saved geometry {found}, requested {requested}; use a new --out or explicit --geometry-version {found}")
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents competing protocols from overwriting one another.
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(dict(schema=SCHEMA, geometry_version=requested,
                           inherited_existing_artifacts=inherited), stream, indent=2)
            stream.write("\n")
    return marker


def require_arm_geometry(arms, geometry_version):
    requested = _known(geometry_version)
    if not isinstance(arms, Mapping) or not arms:
        raise ValueError("Nonempty checkpoint arms mapping required")
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            raise ValueError(f"Invalid checkpoint arm: {name}")
        found = _known(arm.get("geometry_version", "legacy_v1"))
        if found != requested:
            raise ValueError(f"Checkpoint arm {name}: saved geometry {found}, requested {requested}; historical weights require explicit legacy_v1")


def require_cache_geometry(archive, geometry_version):
    found = str(archive["geometry_version"].item()) if "geometry_version" in archive else "legacy_v1"
    if _known(found) != _known(geometry_version):
        raise ValueError(f"Cache geometry {found} differs from requested {geometry_version}")


def require_reference_geometry(directory, geometry_version):
    """Read-only check of a result directory used as a numeric reference."""
    marker = Path(directory) / "geometry_protocol.json"
    if marker.exists():
        spec = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(spec, Mapping) or spec.get("schema") != SCHEMA:
            raise ValueError(f"Invalid reference geometry protocol: {marker}")
        found = _known(spec.get("geometry_version"))
    else:
        found = "legacy_v1"
    if found != _known(geometry_version):
        raise ValueError(f"Reference geometry {found} differs from requested {geometry_version}; select matching --reference-results")
