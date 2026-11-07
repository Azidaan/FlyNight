"""Зафиксировать короткий GPUзамер до полного извлечения/обучения; готовить после данных."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'appearance_gpu_profile';assert not folder.exists()
    source=['scripts/recovery_appearance_gpu_profile_20261027.py','tracking/busca_runtime.py','weights/busca_official_20261027/model_busca.pth',
        'runs/b123_20261027/recovery_source_snapshot/BUSCA/busca/network.py','tracking/gpu_lock.py']
    video=read(RUN/'admission_dataset/protocol.json')['videos']['train'][0]
    p=dict(id='recovery_appearance_profile_v1',created=datetime.now(timezone.utc).isoformat(),budget_seconds=300,
        author_stop=read(RUN/'authorization.json')['save_at'],device='cuda',video=video,
        action='После полного отчёта данных взять первые64пары первогоtrain-видео. Проверить официальные379ключей BUSCA, '
               'стоимость frozenReIDи20технических шагов новой головы допуска. Не сохранять веса головы и не оценивать качество.',
        hypothesis='Замороженный извлекатель областей может дать измеримо доступные причинные признаки для следующей головы; это техническая проверка, не польза.',
        counterargument='Авторский ReIDобучался на другом домене; причинность и вычислительная доступность не доказывают различимость или HOTA. '
                        'Штатная domain adaptation BNсмешивает образцы пакета, поэтому при пакетировании кадров её нельзя молча сохранять.',
        algorithm='Собственный извлекатель на авторских весах: use_domain_adaptation=False, BNв eval, буферы неизменны. '
                  'Авторская BGRнормировка mean[.406,.456,.485]/std[.225,.224,.299], затемRGBCHW;JPEG95кропы нашего экспортера. '
                  'Пакеты16, FP32. Проверить cosineпризнаков одной области отдельно и в пакете>.99999,без соседнего/будущего влияния. '
                  'Это намеренное отличие от авторской полной BUSCA,никакой MASAподстановки. Последующие модели должны использовать тот же режим.',
        acceptance=['Полный проверенный checkpoint, конечные признаки и градиенты головы, неизвестные метки исключены из ошибки',
                    'BNбуферы не меняются, признаки отдельно/в пакете согласованы, источник и режим явно сохранены',
                    'Стоимость чтения/подготовки/GPUи память измерены; разрешение следующей ступени только по удвоенной измеренной цене'],
        limitation='Технический профиль на данных обучения,не полное обучение,неHOTA,не результат статьиBUSCA и не вкладFlyVis. '
                   'Режим BNявно изменён относительно авторского применения и не может называться точным воспроизведением.',
        sha256={s:sha(ROOT/s) for s in source})
    write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Области объекта во времени — причинность и стоимость признаков внешнего вида',area='flyvis',state='ready',started='2026-10-27',
        question=p['action'],hypothesis=p['hypothesis'],expected=p['algorithm'],summary=p['counterargument'],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],
        acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],metrics=[],commits=[],files=source[:2],
        evidence=[dict(kind='file',ref='runs/recovery_research_20261027/appearance_gpu_profile/protocol.json',claim='Технические условия, явное отличиеBNи критерии до GPUпробы')]))
    assert path.read_bytes()==before;write(path,cat)
    job=dict(name='030_appearance_gpu_profile',folder='runs/recovery_research_20261027/appearance_gpu_profile',script=source[0],script_sha256=sha(ROOT/source[0]),protocol_sha256=sha(folder/'protocol.json'))
    write(RUN/'appearance_profile_job_prepared.json',job)
    journal(p['id'],'подготовлен короткий GPUзамер причинных признаков областей',
        '*Постановка.* '+p['action']+'\n\n*Режим.* '+p['algorithm']+'\n\n*Граница.* '+p['limitation']+' '
        'Профиль300снаодномGPUпосле полногоCPUотчёта,до негообязательныепроверкикаталога и сверкаконцаCUDAдеревьев. '
        'Внешний вид — новый проверяемый вход относительно числового контроля,не гарантияполезности. '
        'Источник runs/recovery_research_20261027/appearance_gpu_profile/protocol.json. '
        'Послеценыготовитьотдельныйпротоколполногоизвлечения/обучения,необъявлять20техническихшаговполнойтренировкой.')
    print('Prepared GPU profile, not published before catalogue checks')

if __name__=='__main__':main()
