# Toronto Fourcast: five-minute video storyboard and draft narration

**Working language:** English, matching the report and competition materials. Keep the final recording under five minutes. All numeric claims below come from our walk-forward validation or the starter model's reproduced validation; no current-market statistics or outside-data numbers belong in the video.

## Slide 1 | Two valuation questions | 0:00-0:35

**On screen:** “What will this home sell for?” Then two simple labels: “Before listing: off-market” and “After listing: list-aware.”

**Narration:** “A home estimate can answer two different questions. Before the owner lists, there is no asking price to guide the estimate. After listing, the model can also use the asking price. We kept those tasks separate because an estimate that sees list price cannot answer the off-market question. Off-market is also the harder scoring track, so that is where we focused most of our feature work.”

## Slide 2 | A forward-looking test | 0:35-1:15

**On screen:** A left-to-right timeline of past sales, six monthly validation windows, and later hidden sales. Label the validation as “train on earlier sales, predict the next month.” Add a small “synthetic data” note.

**Narration:** “The challenge provides synthetic Toronto-style transaction data. We tested the model with six walk-forward monthly windows. In each window, we trained on sales before the month and predicted sales in that month, for 10,069 homes per track overall. This matches the direction of the real task: predict later outcomes from earlier information. We did not use hidden sale prices or outside records.”

## Slide 3 | What the model sees | 1:15-2:00

**On screen:** A home icon with size, rooms, parking, time, coordinates, community, municipality, and house style; a separate list-price label only feeding the list-aware track.

**Narration:** “Both models use home size, rooms, parking, location coordinates, property type, and listing time. We added house style, municipality, and community as native categorical features. The model learns their category values from each training window, and treats an unseen value as missing. The list-aware model also receives list price. The off-market model never sees it. Both models predict log sale price with histogram gradient boosting.”

## Slide 4 | Validation performance | 2:00-3:05

**On screen:** One compact table with MdAPE and MAPE, including the reproduced starter baselines. Highlight MdAPE because it determines ranking.

| Track | Starter MdAPE | Our MdAPE | Our MAPE |
|---|---:|---:|---:|
| Off-market | 20.8% | 10.60% | 13.40% |
| List-aware | 7.3% | 4.36% | 5.57% |

**Narration:** “On the same walk-forward validation, off-market MdAPE fell from the starter's 20.8 percent to 10.60 percent. List-aware MdAPE fell from 7.3 percent to 4.36 percent. Adding the three categories to our numeric feature model reduced MdAPE by a further 3.80 percentage points off-market and 0.19 points list-aware in paired bootstrap comparisons. These are validation results, not a claim about the hidden test.”

## Slide 5 | Where errors remain | 3:05-4:05

**On screen:** The shared-price-decile chart from `figures/error_by_price_range.png`. Mark the cheapest and most expensive off-market deciles. Show the small Co-op Apt validation count beside its result.

**Narration:** “The error chart shows a remaining weakness. Off-market MdAPE is 13.10 percent in the cheapest price decile and 16.55 percent in the most expensive, versus roughly 8 to 12 percent through the middle. List-aware error is lower overall, but rises to 5.79 percent in the top decile. Off-market Co-op Apt has 14.33 percent MdAPE, but that estimate comes from only 48 validation homes, so we would not overinterpret it.”

## Slide 6 | What we would trust | 4:05-4:50

**On screen:** “Useful as a starting estimate” beside “Not a real-world appraisal.” Close with one proposed next step: time-aware community comparables.

**Narration:** “The strongest result is a much better off-market estimate without using asking price. The main limit is that this is a synthetic dataset, and its community names and coordinates do not match real Toronto maps. The hidden period may also behave differently from the validation months. We would present this model as a starting estimate, not an appraisal, and next test community comparables built only from sales available before each prediction date. Thank you.”

## Recording checklist

- Use Toronto Fourcast on the title slide.
- Show the report's final validation table and final chart. Make sure all text is readable on a phone-sized player.
- Speak clearly and record a practice take. Aim for about four minutes and fifty seconds so the final edit stays under the five-minute judging cutoff.
- Keep the video unlisted rather than private so judges can open it.
- After recording, replace the video URL in the report PDF and test the link in a private browser window.
