"""Append verified engineering evidence without changing earlier user records."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/flyvis_repairs_20261106"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main():
    suite_path = "runs/flyvis_repairs_20261106/integrated_1793996598125382200"
    suite = read(suite_path + "/result.json")
    assert suite["passed"] and suite["returncode"] == 0
    assert suite["concurrently_changed_sources"] == [] and suite["test_files"] == 48
    assert not suite["runtime"]["cuda_initialized_before"]
    assert not suite["runtime"]["cuda_initialized_after"]
    assert "747 passed, 2 skipped" in (ROOT / suite_path / "tests.log").read_text()
    publication = read("runs/flyvis_repairs_20261106/final_evidence_remote_verification.json")
    assert publication["verified"]
    for label in ("third", "fourth"):
        assert read(f"runs/flyvis_repairs_20261106/{label}_stable_remote_verification.json")["verified"]
    branches = read("runs/flyvis_repairs_20261106/gitlab_branch_decision.json")
    assert branches["all_three_branches_protected"] and not branches["main_deleted"]
    assert branches["final_user_decision"] == "Оставить main дополнительной копией"
    now = datetime.now(timezone(timedelta(hours=3)))
    evidence = [
        (suite_path + "/result.json", "Итоговая совместная проверка без изменения исходников и без CUDA"),
        (suite_path + "/tests.log", "747 успешных тестов и два явно пропущенных"),
        ("runs/flyvis_repairs_20261106/observation_independent/native_retry_report.json", "Сохранность состояния настоящей модели при отказе и повторе"),
        ("runs/flyvis_repairs_20261106/hdf5_transaction/actual_hdf5_links.json", "Неизменность существующих HDF5 и проверка файловых ссылок"),
        ("runs/flyvis_repairs_20261106/historical_residual_independent/report.json", "Независимая проверка исторического Б1"),
        ("runs/flyvis_timestamp_repair_20261106/independent_validation_v5.json", "Ограниченная проверка нового временного пути"),
        ("runs/flyvis_repairs_20261106/coordinate_only_roi_review/report.json", "Контрдоказательство автоматическому включению геометрии"),
        ("runs/flyvis_repairs_20261106/final_evidence_remote_verification.json", "Проверка точного опубликованного содержимого набора патчей и доказательств"),
        ("runs/flyvis_repairs_20261106/gitlab_branch_decision.json", "Выбор основной ветки и прямое решение сохранить main"),
        ("runs/flyvis_repairs_20261106/FINAL_AUDIT_REPORT.txt", "Сводка технических результатов и незакрытых границ"),
    ]
    assert all((ROOT / path).is_file() for path, _ in evidence)
    card = dict(
        id="flyvis_system_integrity_final_20261106",
        title="Авторская система — защита состояния от каскада ошибок и итоговая совместная проверка",
        area="infra", state="infrastructure", started="2026-11-06", done=now.strftime("%Y-%m-%d"),
        question="Сохраняются ли исправные состояния и данные при отказах и остаётся ли проверенный обычный путь воспроизводимым после совместного применения защит?",
        hypothesis="Техническая проверка воспроизведённых дефектов исполнения; не гипотеза о приросте качества трекинга.",
        expected="Отказы не публикуют неполное или нечисловое состояние; обычные сценарии проходят сопоставление с прежней реализацией. Непрошедшие проверку изменения не становятся режимом по умолчанию.",
        acceptance=[
            dict(criterion="Совместная адресная проверка завершилась без ошибок и без изменения исходников. Технический критерий записан после проверки, научной предварительной регистрацией не является.", met=True, evidence=suite_path + "/result.json"),
            dict(criterion="Отказ наблюдения FlyVis, исторического Б1 и записи HDF5 воспроизведён до исправления; сохранность и повтор проверены после него.", met=True, evidence="runs/flyvis_repairs_20261106/observation_transaction/final_cases_before.log; runs/flyvis_repairs_20261106/historical_residual_patch_bundle_v2/manifest.json; runs/flyvis_repairs_20261106/hdf5_transaction/REPORT.txt"),
            dict(criterion="Собственные изменения ранее неотслеживаемых исходников сохранены проверяемыми патчами без включения полного пользовательского дерева; опубликованные файлы совпадают с локальными объектами Git.", met=True, evidence="runs/flyvis_repairs_20261106/final_evidence_remote_verification.json"),
        ],
        obtained=("Совместно пройдено 747 тестов, два пропущены из-за отсутствия привилегии создания символических ссылок Windows; проверено 48 файлов тестов, CUDA не инициализирована до и после, исходники неизменны "
                  f"({suite_path}/tests.log, result.json, runtime.json). "
                  "В проверке существующих HDF5 совпали контрольные суммы 992 файлов общим объёмом 2164046975 байтов "
                  "(runs/flyvis_repairs_20261106/hdf5_transaction/actual_hdf5_links.json). "
                  "Третий и четвёртый пакеты исходников опубликованы и проверены по идентификаторам содержимого Git "
                  "(runs/flyvis_repairs_20261106/third_stable_remote_verification.json; fourth_stable_remote_verification.json). "
                  "Проверены прямое применение и отмена собственных наборов lifecycle, Б2, временных голов и исторического ResidualEngine; они сохранены в итоговом наборе доказательств."),
        interpretation=("Подтверждена защита проверенных путей от публикации NaN, частичного состояния и неполной записи данных. Исправлены также восстановление флага предупреждений детерминированного режима и проверки обученных выходных голов. "
                        "Обычные результаты настоящих моделей CPU сравнивались с замороженными прежними версиями. Повреждение текущих штатных весов и структуры связей ядра FlyVis исходным аудитом не обнаружено. "
                        "Новый временной путь Б1 технически проверен, но диагностические шаги оптимизации не являются обучением научной модели."),
        decision=("Оставить обычную геометрию legacy_v1: предложенные изменения не прошли проверку неухудшения. "
                  "Историческую схему З13 не переводить автоматически на другой смысл времени у уже обученных весов; это остаётся отдельным незакрытым местом. "
                  f"Итоговый набор доказательств сохранён локальным коммитом `{publication['local_commit']}` и опубликован коммитом `{publication['remote_commit']}`; точное содержимое проверено. "
                  "По прямому решению автора новая codex/system-integrity-20261106-e8f674ce назначена основной в GitLab; old-Pro хранит старую версию, main оставлена дополнительной защищённой копией. "
                  "Рабочее дерево с пользовательскими изменениями и параллельным исследовательским коммитом не переключалось и не переписывалось. "
                  "Красные задания GitLab не исполняли код: авторизованный интерфейс требует проверки личности владельца перед запуском; локальные результаты тестов учитываются отдельно."),
        cannot_claim=("Не доказаны отсутствие всех возможных ошибок и прирост HOTA. Не выполнены научное переобучение с новым временем, миграция исторического З13 и полный пересчёт прежних опытов. "
                      "Ранее зафиксированное расхождение Б2 между обучением и применением не объявлено устранённым проверками сброса. "
                      "Гарантия наблюдения относится к проверенным реализациям, не к произвольному внешнему модулю с изменением входного состояния на месте и не к транзакции всего трекера. "
                      "Полная работа настоящих символических ссылок Windows не проверена из-за прав среды; отказы и разрешение целевого пути проверены отдельно. "
                      "Патчи к прежним неотслеживаемым файлам требуют исходных версий с указанными контрольными суммами; чистый удалённый репозиторий сам по себе не содержит всего локального экспериментального окружения."),
        metrics=[], evidence=[dict(kind="json" if path.endswith(".json") else "file", ref=path, claim=claim) for path, claim in evidence],
    )
    catalogue = ROOT / "experiments/project_history.json"
    plan = ROOT / "dissertation/ЕДИНЫЙ_ПЛАН.md"
    before_cat, before_plan = catalogue.read_bytes(), plan.read_bytes()
    text = before_cat.decode("utf-8")
    data = json.loads(text)
    assert not any(c["id"] == card["id"] for c in data["experiments"])
    start = re.search(r'"experiments"\s*:\s*\[', text).end() - 1
    _, end = json.JSONDecoder().raw_decode(text, start)
    offset = len(text[:end - 1].rstrip())
    insertion = ",\n" + "\n".join("    " + line for line in json.dumps(card, ensure_ascii=False, indent=2).splitlines()) + "\n  "
    updated = text[:offset] + insertion + text[offset:]
    assert json.loads(updated) == dict(data, experiments=[*data["experiments"], card])
    entry = (f"\n\n**{now:%d.%m.%Y, %H:%M} MSK** — Совместная проверка защит состояния и сохранение основной ветки\n\n"
             "*Постановка.* Проверить совместное применение исправлений с приоритетом отсутствия новых ошибок; сохранить отрицательные проверки собственного предложения и все прежние пользовательские материалы.\n\n"
             f"*Получено (`{card['id']}`).* {card['obtained']}\n\n"
             f"*Вывод.* {card['interpretation']} {card['cannot_claim']}\n\n"
             f"*Решение.* {card['decision']}\n").encode("utf-8")
    marker = "## Журнал изменений".encode("utf-8")
    assert before_plan.count(marker) == 1
    updated_plan = before_plan.replace(marker, marker + entry, 1)
    assert updated_plan.replace(marker + entry, marker, 1) == before_plan
    assert catalogue.read_bytes() == before_cat and plan.read_bytes() == before_plan
    (OUT / "final_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_journal.txt").write_bytes(entry)
    catalogue.write_bytes(updated.encode("utf-8"))
    plan.write_bytes(updated_plan)
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    receipt = dict(at=now.isoformat(), other_cards_unchanged=True, previous_plan_bytes_preserved=True,
                   before=dict(catalogue=sha(before_cat), plan=sha(before_plan)),
                   after=dict(catalogue=sha(catalogue.read_bytes()), plan=sha(plan.read_bytes())))
    (OUT / "final_record_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    spec = dict(id=card["id"], card="runs/flyvis_repairs_20261106/final_card.json",
                journal="runs/flyvis_repairs_20261106/final_journal.txt", receipt="runs/flyvis_repairs_20261106/final_record_receipt.json")
    (OUT / "final_ledger_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
