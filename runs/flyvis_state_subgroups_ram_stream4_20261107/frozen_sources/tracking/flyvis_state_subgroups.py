"""Prespecified disjoint partition; CT1 has two named model compartments."""
GROUPS = {
    "lamina": ["L1", "L2", "L3", "L4", "L5"],
    "wide_lamina": ["Lawf1", "Lawf2", "Am"],
    "centrifugal": ["C2", "C3"],
    "ct1": ["CT1(Lo1)", "CT1(M10)"],
    "mi1": ["Mi1"], "mi4": ["Mi4"], "mi9": ["Mi9"],
    "other_mi": ["Mi2", "Mi3", "Mi10", "Mi11", "Mi12", "Mi13", "Mi14", "Mi15"],
}
LABELS = dict(lamina="L1–L5", wide_lamina="Lawf1/Lawf2/Am", centrifugal="C2/C3",
              ct1="CT1", mi1="Mi1", mi4="Mi4", mi9="Mi9", other_mi="Остальные Mi",
              parent="Все 23 обозначения")

def validate_partition(parent):
    values = sum(GROUPS.values(), [])
    if len(values) != 23 or len(set(values)) != len(values):
        raise ValueError("Subgroups overlap or do not contain 23 entries")
    if set(values) != set(parent) or len(parent) != 23:
        raise ValueError("Subgroups differ from the previously studied parent")
    return {**{k: list(v) for k, v in GROUPS.items()}, "parent": list(parent)}
