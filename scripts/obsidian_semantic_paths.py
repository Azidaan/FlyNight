"""Названия файлов совпадают со смысловыми названиями карточек, коды — в aliases."""
import hashlib

def semantic_paths(cards,planned,fly_theme,safe):
    paths={}
    for ident,card in {**cards,**planned}.items():
        folder='09 Запланировано' if ident in planned else ('01 Муха/Эксперименты' if ident in fly_theme else '02 Эксперименты')
        paths[ident]=folder+'/'+safe(card['title'])
    groups={}
    for ident,path in paths.items():groups.setdefault(path.casefold(),[]).append(ident)
    for ids in groups.values():
        if len(ids)>1:
            for ident in ids:
                folder=paths[ident].rsplit('/',1)[0]
                paths[ident]=folder+'/'+safe(({**cards,**planned})[ident]['title'],100)+' — '+hashlib.sha256(ident.encode()).hexdigest()[:6]
    # Чужая ручная заметка сохраняется по существующему адресу до отдельного согласования.
    if 'z13_distribution_pilot_v1' in cards:
        paths['z13_distribution_pilot_v1']='01 Муха/Эксперименты/z13_distribution_pilot_v1'
    assert len({p.casefold() for p in paths.values()})==len(paths)
    return paths
