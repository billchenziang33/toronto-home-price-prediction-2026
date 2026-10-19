# Target and blend comparison

Scores use the three fixed development folds. A weight of 0 is the same-fold HGBR reference; 1 is the selected challenger.

| Track | Raw log-price MdAPE | Alternative target MdAPE | Chosen target | Chosen blend weight | Development blend MdAPE | MAPE |
|---|---:|---:|---|---:|---:|---:|
| list_aware | 4.4055% | 4.3995% | log_sold_over_list | 0.75 | 4.3890% | 5.4708% |
| off_market | 9.5106% | 9.5321% | log_sold_price | 1.00 | 9.5107% | 12.4760% |

The blend weight and target are now locked; later benchmark results will not change them.
