# Validation report

**Scope:** retrospective standard validation; not an independent holdout.
Window: 2025-12 to 2026-05. Each month is predicted using sales strictly before its first day.
Scores use nearest-dollar predictions, as in the final CSV. All JSON metrics retain full precision.
MdAPE is the primary competition metric; MAPE is the tiebreak metric. Brackets show a home-level bootstrap 95% interval.

| Track | Route | Homes | MdAPE % | MAPE % | P90 APE % | Median signed % |
|---|---|---:|---:|---:|---:|---:|
| list_aware | R35 | 10,069 | 4.00575871 (3.91376416–4.10014371) | 5.07528631 (4.99551809–5.16053117) | 10.70023058 | +0.53943850 |
| list_aware | R97 | 10,069 | 3.93961679 (3.86503463–4.03875877) | 5.02935657 (4.94922280–5.11463332) | 10.57496780 | +0.22157272 |
| off_market | R151 | 10,069 | 7.23638935 (7.03572080–7.39708285) | 9.52504341 (9.34858508–9.72018110) | 20.19626298 | +0.11487409 |
| off_market | R22 | 10,069 | 7.19315110 (6.99763195–7.34163666) | 9.63650976 (9.44934457–9.82499442) | 20.64667929 | +0.17034377 |
| off_market | R52 | 10,069 | 7.06819266 (6.91116044–7.27273409) | 9.52229909 (9.34287253–9.71140951) | 20.22997032 | +0.18601684 |

## Paired route comparisons

A negative MdAPE delta favors the candidate. These home-level intervals do not capture common market movement within a month or repeated model selection.

| Track | Baseline | Candidate | Homes | MdAPE delta pp (95% interval) | MAPE delta pp |
|---|---|---|---:|---:|---:|
| off_market | R22 | R52 | 10,069 | -0.12495844 (-0.20721943 to +0.01289251) | -0.11421066 |
| off_market | R22 | R151 | 10,069 | +0.04323825 (-0.06545217 to +0.16420031) | -0.11146635 |
| off_market | R52 | R151 | 10,069 | +0.16819669 (+0.04668183 to +0.23495036) | +0.00274432 |
| list_aware | R35 | R97 | 10,069 | -0.06614192 (-0.10150819 to -0.02927266) | -0.04592974 |

## Error breakdown

`outputs/error_analysis.csv` includes overall, month, property type, sold-price decile, and community results. Community metrics are suppressed below n=30; those groups retain counts only.

## Interpretation limits

The six-month window has been used in prior work, so it is retrospective validation rather than an independent holdout. Home-level bootstrap intervals treat homes as independent and omit shared monthly shocks.
R97's earlier eight-month development result and promotion-gate outcome are reported separately in the technical audit; this report is a new fixed-window pipeline run.

![Error by price range](../figures/error_by_price_range.png)
