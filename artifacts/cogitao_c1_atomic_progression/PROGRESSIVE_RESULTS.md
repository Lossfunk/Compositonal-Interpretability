# C1 experiment-1 atomic checkpoint progression

Validation and test use the same atomic examples at every checkpoint. Exact-grid accuracy, object-pixel accuracy, and cross entropy are shown below; the CSV files also contain all-pixel accuracy and checkpoint deltas.

## rot90

| Checkpoint | Step | Val exact % | Test exact % | Val object-pixel % | Test object-pixel % | Val CE | Test CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| milestone_start | 0 | 0.00 | 0.00 | 0.91 | 0.73 | 2.2740 | 2.2661 |
| epoch=016 | 6,630 | 0.00 | 0.00 | 27.14 | 28.17 | 0.1653 | 0.1680 |
| epoch=018 | 7,410 | 0.00 | 0.00 | 28.51 | 28.26 | 0.1624 | 0.1637 |
| epoch=021 | 8,580 | 0.00 | 0.00 | 27.96 | 26.54 | 0.1536 | 0.1604 |
| milestone_025pct | 19,500 | 0.00 | 0.00 | 43.12 | 42.21 | 0.1148 | 0.1171 |
| milestone_050pct | 39,000 | 14.41 | 23.42 | 78.76 | 80.65 | 0.0477 | 0.0475 |
| milestone_075pct | 58,500 | 30.63 | 36.04 | 89.18 | 87.80 | 0.0257 | 0.0329 |
| last | 78,000 | 47.75 | 47.75 | 93.74 | 92.75 | 0.0155 | 0.0179 |

Coverage: 111 validation and 111 test examples.

## translate_up

| Checkpoint | Step | Val exact % | Test exact % | Val object-pixel % | Test object-pixel % | Val CE | Test CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| milestone_start | 0 | 0.00 | 0.00 | 0.82 | 0.96 | 2.2563 | 2.2654 |
| epoch=016 | 6,630 | 0.00 | 0.00 | 34.07 | 34.54 | 0.1332 | 0.1447 |
| epoch=018 | 7,410 | 0.00 | 0.00 | 38.18 | 38.35 | 0.1291 | 0.1401 |
| epoch=021 | 8,580 | 0.00 | 0.00 | 37.51 | 37.26 | 0.1254 | 0.1354 |
| milestone_025pct | 19,500 | 1.79 | 0.00 | 59.81 | 58.64 | 0.0755 | 0.0847 |
| milestone_050pct | 39,000 | 43.75 | 37.50 | 89.61 | 89.47 | 0.0256 | 0.0257 |
| milestone_075pct | 58,500 | 76.79 | 81.25 | 98.33 | 98.90 | 0.0042 | 0.0037 |
| last | 78,000 | 91.96 | 89.29 | 99.32 | 98.27 | 0.0024 | 0.0031 |

Coverage: 112 validation and 112 test examples.

The released C1 OOD splits have no atomic rows for these functions, so this report covers atomic ID validation and test only.
