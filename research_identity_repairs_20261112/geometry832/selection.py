"""Generic proposal guard; no IDs, labels, scores fitted, or ownership writes."""
import math


def choose_correspondence(original_score,repair_score,appearance_support,has_active_owner):
    """Preserve positive originals, admit bounded rescue only with evidence.

    appearance_support is True only after the CALLER has checked the complete
    competitor appearance set. False is an explicit contradiction; None means
    unknown and cannot authorize a repair. has_active_owner reports a competing
    active assignment that the caller must protect. This function proposes no
    reassignment and does not replace matrix-level ownership verification.
    """
    for value in (original_score,repair_score):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError('Finite scores or None required')
    if appearance_support is not None and type(appearance_support) is not bool:
        raise ValueError('Appearance support must be True, False, or None')
    if type(has_active_owner) is not bool:
        raise ValueError('Active-owner flag must be bool')
    original_positive=original_score is not None and float(original_score)>0
    repair_positive=repair_score is not None and float(repair_score)>0
    if original_positive:
        if appearance_support is False:
            return dict(coordinate=None,use_repair=False,accepted=False,action='defer',reason='appearance_contradicts_original',selected_score=None)
        return dict(coordinate='original',use_repair=False,accepted=True,action='continue_original',reason='preserve_original_positive',selected_score=float(original_score))
    if original_score is None:
        return dict(coordinate=None,use_repair=False,accepted=False,action='defer',reason='original_score_unknown',selected_score=None)
    if not repair_positive:
        return dict(coordinate=None,use_repair=False,accepted=False,action='defer',reason='no_positive_repair',selected_score=None)
    if appearance_support is not True:
        return dict(coordinate=None,use_repair=False,accepted=False,action='defer',reason='appearance_not_supported',selected_score=None)
    if has_active_owner:
        return dict(coordinate=None,use_repair=False,accepted=False,action='defer',reason='active_owner_protected',selected_score=None)
    return dict(coordinate='census',use_repair=True,accepted=True,action='propose_repair',reason='negative_original_positive_repair_supported_unowned',selected_score=float(repair_score))
