# Remarks Text Feature Experiment

## Context

The Discord `#case-questions` reply says the remarks reflect market dynamics and that the LLM uses a QLoRA adapter plus an embedded distribution for some latent variables. This supports testing the text but does not establish that the text adds price information beyond structured columns. It should not be described as random prose or as proven price leakage.

## Goal

Measure whether a compact text representation of `remarks` improves the existing off-market and list-aware price models under the same leakage-safe, expanding monthly validation.

## Proposed experiment

- Preserve all existing structured features and model settings.
- Add a small, predeclared set of transparent remarks features: text length and counts of renovation, negative condition/maintenance, building-system updates, location amenities, transit/schools, views, investment/rental, and luxury language.
- Define patterns before scoring validation outcomes and apply the same deterministic text transform to each fold; this transform has no learned vocabulary, IDF weights, target encoding, or threshold.
- Keep off-market and list-aware tracks separate and preserve the list-price restriction.
- Use December 2025–February 2026 as an initial selection/debug period; freeze the pipeline and assess March–May 2026 as the temporal confirmation period.
- Compare paired MdAPE and MAPE changes on identical homes, month-level results, signed error, and runtime against the current model.
- Retain text only if the later period shows a stable improvement without a material failure in either track; otherwise document the negative result and keep the baseline.

## Guardrails

- Do not use validation/test prices or validation text to tune the predeclared lexicon.
- Do not use internet-sourced or post-May-2026 data.
- Do not infer that rates are explicitly modeled; the organizer said they are not directly modeled and that MLS-calibrated market effects carry over.
- Report the result as a validation experiment, not hidden-test performance or real-world Toronto prediction.
- Keep the change isolated in an experiment script until it passes the temporal confirmation; only then consider changing production features.

## Success evidence

The experiment must produce a reproducible fold-level predictions file and a report with track/month metrics, paired home-level bootstrap intervals for MdAPE differences, and measured runtime. Both validation halves must be reported, even if the experiment fails the retention criterion.
