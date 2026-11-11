"""Render explicitly registered deferred ideas without scheduling experiments."""
import json

from scripts.build_obsidian_vault import ROOT, Vault, fm


def build_source_notes(vault_path):
    registry = json.loads((ROOT / 'experiments/deferred_identity_hypotheses.json').read_text(encoding='utf8'))
    vault = Vault(vault_path)
    for item in registry['ideas']:
        assert item['state'] == 'deferred' and item['result'] is None
        path = item['note'] + '.md'
        title = item['title']
        text = fm({'tags': ['гипотезы-подход', 'проверка-гипотезы/не-завершена'],
                   'код': item['id'], 'состояние': 'отложено',
                   'дата-регистрации': item['registered'],
                   'роль': 'непроверенная гипотеза; не действующая задача',
                   'источник-записи': 'experiments/deferred_identity_hypotheses.json'})
        text += '# ' + title + '\n\n'
        text += '**Статус:** отложено по поручению автора. Польза не проверена; научные запуски этим поручением не разрешены.\n\n'
        for heading, key in [('Какую проблему решает', 'problem'), ('Что меняется', 'mechanism'),
                             ('Почему может помочь', 'rationale'), ('Связь с прежними проверками', 'prior_work'),
                             ('Сильнейшая альтернатива', 'alternative'), ('Условие возвращения и малая проверка', 'return_gate'),
                             ('Чего пока нельзя утверждать', 'cannot_claim')]:
            text += '## ' + heading + '\n\n' + item[key] + '\n\n'
        text += '## Первичные источники и граница переноса\n\n'
        for source in item['sources']:
            text += '[' + source['title'] + '](' + source['url'] + '). ' + source['supports'] + '\n\n'
        text += 'Полная запись: [Единый план, раздел 7.12](' + (ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md').as_uri() + ').\n'
        vault.write(path, text)
    return vault.written
