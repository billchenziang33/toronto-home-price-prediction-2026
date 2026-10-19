# Development model-family screen

Fixed development months: 2025-06, 2025-09, and 2025-11. Scores are selection diagnostics, not independent test results.

| Track | Family | Months complete | MdAPE | MAPE | vs fold HGBR MdAPE | Worst month MdAPE | Fit seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| list_aware | lightgbm | 3/3 | 4.4479% | 5.5866% | -0.0879 pp | 4.6531% | 12.7 |
| list_aware | catboost | 3/3 | 4.5250% | 5.7042% | -0.0109 pp | 4.8150% | 101.9 |
| list_aware | hgbr | 3/3 | 4.5358% | 5.7212% | +0.0000 pp | 4.6973% | 7.6 |
| off_market | lightgbm | 3/3 | 10.0368% | 12.8786% | -0.8419 pp | 10.1708% | 12.5 |
| off_market | hgbr | 3/3 | 10.8787% | 13.8891% | +0.0000 pp | 11.0228% | 7.9 |
| off_market | catboost | 3/3 | 11.1356% | 14.1935% | +0.2569 pp | 11.3748% | 95.5 |

These are development-fold results used to pick the next small search. The six-month benchmark is separate and remains locked until the final review.
