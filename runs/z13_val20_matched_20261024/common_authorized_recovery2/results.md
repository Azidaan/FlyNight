Все20val,исходные кадры,общие детекции960/conf0,01,TrackEval. Показаны раздельно исходные трекеры и одна общая замороженная сшивка MASA. Набор не является независимым итоговым тестом.

| Система | HOTA без сшивки | HOTA с одинаковой MASA |
|---|---:|---:|
| XYWHByteTrack | 57,9008 | 58,4858 |
| ByteTrack | 55,7209 | 56,3011 |
| BoT-SORT | 58,4903 | 58,8737 |
| TrackTrack | 59,2648 | 59,6224 |
| XYWH + обученная D, прежнее считывание | 57,8535 | 58,4893 |
| XYWH + D, центрированное считывание | 57,8944 | 58,4807 |
| XYWH + штатное ядро C, прежнее считывание | 57,8397 | 58,3738 |
| XYWH + C, центрированное считывание | 57,8955 | 58,4815 |
| XYWH + малая обученная GRU B | 57,8853 | 58,4366 |
| XYWH + временной вход глаза R1 | 57,5053 | 58,3763 |

C is the pretrained fixed FlyVis core with task-trained frozen readout; D is one saved adapted core. Their readout/policy weights and training differ. B is small learnedGRU, R1 temporal receptors, parameter counts differ. D-C is functional system comparison, not an isolated backbone/anatomy effect. Same four observed frames followed by continuing recurrent search history, max4eyes, fixed per-arm tau/lambda. Geometry-only tracker without any correction is included.

All200val videos were used in older low-frame-rate tracker research; detector val monitored during weight selection. New20 are additional original-frame caches, not untouched independent confirmation or final test. No outcome-dependent exclusion; ambiguous alignment stops preparation, no replacement video. Additional20 were selected before outcomes, but are not untouched final test. Native algorithms not optimally tuned. MASA scorer trained on XYWH pairs may transfer unequally;show raw and MASA. One saved D is not a seed-independence result. Prior7 subset and these20 remain separate tables;do not compare their raw numbers as changes.

Первичный источник: runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json. Парные интервалы всех сравнений: paired. Проценты к базе: verified_summary.json/summary. Доля измеренного потолка неизвестна.
