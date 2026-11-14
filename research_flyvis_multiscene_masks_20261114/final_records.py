from common import *
import re
c=read(OUT/'checks.json');assert c['passed'] and read(OUT/'audit.json')['passed'] and read(OUT/'geometry_intervention_audit.json')['passed']
assert all(sha(ROOT/n)==h for n,h in c['inputs'].items());log=(OUT/'history_tests.log').read_text('utf8');count=int(re.search(r'(\d+) passed',log).group(1))
journal('Проверки расширенного считывателя масок завершены: каталог согласован с источниками; '+str(count)+'тестовистории пройдены. Независимый пересчёт подтвердил96таблиц ошибок и предсказания с максимальным расхождением5,96×10⁻¹⁴; '+rel(OUT/'audit.json')+'. Дополнительное вмешательство в фиксированный считыватель подтвердило воспроизведение рамки: нейтрализация FlyVis0изменённых ячеек, геометрии872; '+rel(OUT/'geometry_intervention_audit.json')+'. Основной научный исход отрицательный; допуск по разметке уверенных участков сам по себе не означает получение контура. Два прежних предупреждения валидатора о шаблонных путях сохранены. Новые веса предназначены для исследовательской копии.')
save('final_record_checks.json',dict(passed=True,created=now(),checks_sha256=sha(OUT/'checks.json'),audit_sha256=sha(OUT/'audit.json'),intervention_audit_sha256=sha(OUT/'geometry_intervention_audit.json'),history_tests=count))
print('FINAL_RECORDS_READY',count)
