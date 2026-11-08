from research_singleton_verify_20261108.common import *
FULL=RUN/'full'


def configure_paths():
    import research_singleton_free_20261108.full_paths as paths
    paths.FULL=FULL
    import research_singleton_free_20261108.full_features as features
    features.FULL=FULL
    return paths,features
