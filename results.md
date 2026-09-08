# Repaired evaluation results

Generated from `artifacts/metrics.json` and `submission_metadata.json`.

Selected model: **ensemble_equal**, using highest development OOF accuracy, then macro F1, then candidate order.

| Model | CV accuracy ± SD | CV macro F1 ± SD | CV balanced accuracy ± SD | Holdout accuracy | Holdout macro F1 | Holdout balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| catboost | 84.43% ± 0.75 | 84.53% ± 0.76 | 84.47% ± 0.75 | 85.17% | 85.24% | 85.22% |
| xgboost | 83.81% ± 0.76 | 83.89% ± 0.78 | 83.87% ± 0.76 | 84.49% | 84.54% | 84.55% |
| lightgbm | 84.71% ± 0.85 | 84.80% ± 0.86 | 84.75% ± 0.85 | 85.51% | 85.56% | 85.57% |
| **ensemble_equal (selected)** | 85.08% ± 0.77 | 85.17% ± 0.78 | 85.12% ± 0.77 | 85.77% | 85.82% | 85.82% |
| ensemble_weighted | 85.06% ± 0.77 | 85.15% ± 0.78 | 85.10% ± 0.77 | 85.86% | 85.91% | 85.91% |
| dummy | 19.93% ± 0.72 | 19.90% ± 0.72 | 19.90% ± 0.72 | 19.86% | 19.85% | 19.85% |

CV values summarize five development folds; SD is the sample standard deviation (ddof=1), in percentage points, not a confidence interval. OOF selection scores use all development predictions together. Holdout values are separate, after selection was frozen.

Development OOF: strongest ensemble `ensemble_equal` minus strongest single `lightgbm` = +0.375 percentage points. Holdout: strongest ensemble `ensemble_weighted` minus strongest single `lightgbm` = +0.343 percentage points. The ensemble beats the strongest single model on holdout accuracy. These are descriptive comparisons, not significance tests. The holdout does not change selection.


## Per-class holdout metrics

### catboost

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.8564 | 0.8667 | 0.8615 | 1335 |
| 1 | 0.7731 | 0.7779 | 0.7755 | 1463 |
| 2 | 0.8037 | 0.8125 | 0.8081 | 1381 |
| 3 | 0.8725 | 0.8632 | 0.8678 | 1403 |
| 4 | 0.9576 | 0.9408 | 0.9491 | 1418 |

Confusion matrix: rows = true labels, columns = predicted labels, ordered 0–4.
```text
1157 173 5 0 0
189 1138 134 2 0
5 159 1122 94 1
0 2 132 1211 58
0 0 3 81 1334
```

### xgboost

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.8443 | 0.8854 | 0.8644 | 1335 |
| 1 | 0.7575 | 0.7580 | 0.7578 | 1463 |
| 2 | 0.7889 | 0.7712 | 0.7799 | 1381 |
| 3 | 0.8698 | 0.8667 | 0.8683 | 1403 |
| 4 | 0.9669 | 0.9464 | 0.9565 | 1418 |

Confusion matrix: rows = true labels, columns = predicted labels, ordered 0–4.
```text
1182 152 1 0 0
206 1109 146 2 0
12 197 1065 106 1
0 6 136 1216 45
0 0 2 74 1342
```

### lightgbm

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.8676 | 0.8839 | 0.8757 | 1335 |
| 1 | 0.7755 | 0.7792 | 0.7774 | 1463 |
| 2 | 0.8018 | 0.7849 | 0.7933 | 1381 |
| 3 | 0.8663 | 0.8824 | 0.8743 | 1403 |
| 4 | 0.9676 | 0.9478 | 0.9576 | 1418 |

Confusion matrix: rows = true labels, columns = predicted labels, ordered 0–4.
```text
1180 153 2 0 0
175 1140 146 2 0
5 174 1084 116 2
0 3 119 1238 43
0 0 1 73 1344
```

### ensemble_equal

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.8645 | 0.8839 | 0.8741 | 1335 |
| 1 | 0.7806 | 0.7854 | 0.7830 | 1463 |
| 2 | 0.8081 | 0.7987 | 0.8034 | 1381 |
| 3 | 0.8743 | 0.8724 | 0.8733 | 1403 |
| 4 | 0.9642 | 0.9506 | 0.9574 | 1418 |

Confusion matrix: rows = true labels, columns = predicted labels, ordered 0–4.
```text
1180 153 2 0 0
179 1149 131 4 0
6 167 1103 104 1
0 3 127 1224 49
0 0 2 68 1348
```

### ensemble_weighted

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.8638 | 0.8839 | 0.8738 | 1335 |
| 1 | 0.7823 | 0.7861 | 0.7842 | 1463 |
| 2 | 0.8091 | 0.8009 | 0.8049 | 1381 |
| 3 | 0.8751 | 0.8738 | 0.8745 | 1403 |
| 4 | 0.9656 | 0.9506 | 0.9581 | 1418 |

Confusion matrix: rows = true labels, columns = predicted labels, ordered 0–4.
```text
1180 153 2 0 0
180 1150 129 4 0
6 165 1106 103 1
0 2 128 1226 47
0 0 2 68 1348
```

### dummy

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.1939 | 0.1910 | 0.1925 | 1335 |
| 1 | 0.2004 | 0.2030 | 0.2017 | 1463 |
| 2 | 0.2069 | 0.1991 | 0.2030 | 1381 |
| 3 | 0.2028 | 0.2053 | 0.2040 | 1403 |
| 4 | 0.1891 | 0.1939 | 0.1915 | 1418 |

Confusion matrix: rows = true labels, columns = predicted labels, ordered 0–4.
```text
255 281 242 271 286
287 297 296 278 305
245 296 275 289 276
261 291 251 288 312
267 317 265 294 275
```

No labeled external-test score is available. The selected model was refitted on all labeled rows only after evaluation. The repaired holdout uses historically seen data; see README limitations.
