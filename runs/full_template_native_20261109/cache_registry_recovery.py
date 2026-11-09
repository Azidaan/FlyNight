"""Account for a third CLOSED same-recipe cache producer, without relaxing hashes."""
from common import ROOT,OUT,LEGACY,read,sha


def install():
    from native_runtime import FeatureBank
    original=FeatureBank.__init__
    def init(self,video,arm,backbone=None):
        original(self,video,arm,backbone)
        path=LEGACY/'rollouts/G_PERM'/(video+'.receipt.json')
        phase=read(OUT/'cache_registry_protocol.json')
        name=path.relative_to(ROOT).as_posix();assert sha(path)==phase['sources'][name]
        record=read(path);assert record['passed'] and record['arm']=='G_PERM' and record['video']==video
        for source,digest in record['feature_files'].items():
            file=ROOT/source;file.resolve().relative_to((LEGACY/'features'/video).resolve())
            if source in self.legacy_sources:assert self.legacy_sources[source]==digest
            self.legacy_sources[source]=digest
        self.files[name]=sha(path)
    FeatureBank.__init__=init
    return FeatureBank
