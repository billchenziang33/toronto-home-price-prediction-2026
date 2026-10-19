# Toronto Fourcast - five-minute video narration

**Target length:** about 4 minutes 40 seconds, including slide transitions and brief pauses. The spoken text is 477 words. The six sections match the six slides in `../slides/toronto_fourcast_final.pptx`; the figures are from the frozen walk-forward validation or its reproduced starter baseline.

## Slide 1 | Two valuation questions | 0:00-0:35

“Toronto Fourcast answers two different home-price questions. Before a home is listed, an estimate has no asking price to use. After listing, the model can also see the asking price. We kept these tracks separate so information from the list-aware task never enters the off-market estimate. Today we will show the model, its forward validation, and where its errors remain.”

## Slide 2 | A forward-looking test | 0:35-1:15

“We used six monthly walk-forward windows from December 2025 through May 2026. In every window, the model learned from sales that happened earlier and predicted homes sold in the next month. Across those windows, each track contains 10,069 validation homes. This follows the direction of the challenge: predict later sales from earlier information. We did not use hidden sale prices or real-world listing records. The supplied transaction data are synthetic, and their community labels and coordinates do not form a real Toronto map.”

## Slide 3 | What the model sees | 1:15-2:00

“The two models share property details, location fields, and listing time. They also use property type, house style, municipality, and community as categorical features. The list-aware model receives list price. The off-market model does not. Each model is a histogram gradient-boosting regressor trained on log sale price. We froze the structured-feature model at the last published version. The later remarks experiment stays in a separate archive.”

## Slide 4 | Validation performance | 2:00-2:55

“We report median absolute percentage error, or MdAPE, because it determines the competition ranking. Mean absolute percentage error, or MAPE, is the tie-breaker. On the same forward validation, off-market MdAPE went from the reproduced starter score of 20.8 percent to 10.60 percent. List-aware MdAPE went from 7.3 percent to 4.36 percent. The intervals show home-level bootstrap uncertainty. They do not capture market shocks shared by homes in the same month. These numbers describe validation only; we do not know the hidden-test score.”

## Slide 5 | Where errors remain | 2:55-3:50

“The error chart groups the same validation homes by sold-price decile. Off-market MdAPE is 13.10 percent in the lowest decile and 16.55 percent in the highest, compared with roughly 8 to 12 percent across the middle. List-aware MdAPE also rises at the top, reaching 5.79 percent. This points to price extremes as the clearest area for further validation. The chart describes patterns in this synthetic sample; it does not explain why any particular home was mispriced.”

## Slide 6 | What we would trust | 3:50-4:40

“The supplied remarks describe market dynamics, so we tested them separately. We found that the earlier uncertainty calculation did not match the MdAPE difference reported as its point estimate. After resampling the paired homes and calculating the two medians separately, the confirmation-period intervals include zero on both tracks. We therefore kept the previously published structured model and retained remarks only as an experiment. The validation also covers only six months, and its home-level intervals do not capture shared monthly shocks. This synthetic benchmark is not a real-market appraisal. Our next experiment would test price-tail calibration using the same forward-only validation. Thank you.”

## Recording checklist

- Keep the six slides in this order and use the matching validation table and chart.
- Aim for a clear pace and stay below five minutes; judges stop at the time limit.
- Replace the pending video line in the PDF after recording.
- Upload to YouTube as public or unlisted, then confirm the link opens without a login.
