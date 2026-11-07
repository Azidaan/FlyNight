"""Контрольные суммы входов отдельной завершённой проверки видео."""
import hashlib,json
from pathlib import Path

def input_manifest(root,source,video,variants):
    files=[source/'protocol.json',source/'geometry.npz',source/'completed'/(video+'.json')]
    for arm in variants:
        files.extend([source/arm/(video+'.json'),source/'events'/arm/(video+'.json'),source/'raw'/arm/(video+'.npz')])
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def validate_checkpoint(saved,manifest):
    assert saved['passed'] is True and saved['input_manifest']==manifest,'Проверенные входы изменились'
    assert set(saved['metrics'])=={'baseline','appearance','zero','C','D','R1','B'}
    assert saved['events_checked']>=0
    return saved
