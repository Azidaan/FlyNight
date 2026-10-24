"""Publish only this verified bounded diagnostic and its derived graph notes."""
from pathlib import Path
import inspect
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_activation_timing_20261025 import RUN, read


def main():
    from scripts import z13_publish_assignment_effect_20261024 as publisher
    assert read(RUN / 'finish_recorded.json')['all_owned_ended']
    publisher.RUN = RUN
    extra = ['scripts/z13_activation_timing_20261025.py',
             'scripts/z13_activation_timing_console.ps1',
             'scripts/z13_record_activation_timing_20261025.py',
             Path(__file__).relative_to(ROOT).as_posix(),
             'tests/test_z13_activation_timing.py']
    for name in ('protocol.json','report.json','controller.json','controller_result.json',
                 'launch.json','launch_error_preserved.json','price.json','finish_recorded.json','publication_attempt1.json'):
        if (RUN / name).exists():
            extra.append((RUN / name).relative_to(ROOT).as_posix())
    extra += [p.relative_to(ROOT).as_posix() for p in (RUN / 'videos').glob('*.json')]
    source = inspect.getsource(publisher.publish)
    source = source.replace("names=['experiments/project_history.json'", "names=EXTRA+['experiments/project_history.json'")
    source = source.replace("'tests/test_z13_assignment_effect.py','-q'", "'tests/test_z13_assignment_effect.py','tests/test_z13_activation_timing.py','-q'")
    source = source.replace("prefix='assignment_effect_'+mode", "prefix='activation_timing_final'")
    source = source.replace("title='Correct invalid crop-sampling premise and start exact assignment-effect supervision' if mode=='start' else 'Record complete assignment-effect supervision with exact baseline replay'", "title='Measure Lost-only activation blind spots with exact shadow baseline replay'")
    namespace = {**vars(publisher), 'EXTRA':extra}
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    namespace['publish']('activation_timing_final')


if __name__ == '__main__':
    main()
