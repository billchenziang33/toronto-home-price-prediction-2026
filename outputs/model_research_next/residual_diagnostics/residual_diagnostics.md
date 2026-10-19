# Saved-prediction residual diagnostics

Paired on month and listing ID using only the four already-used development folds. Sale-price bands and property/geography splits are descriptive and were not used to fit a model.

## off_market/Gamma

n=7,066; MdAPE 9.1745% → 9.0636% (+1.21%); MAPE 11.7867% → 11.9006% (+0.97%); median signed error +0.069% → +0.771%.

- `month`: 2025-03 n=1931, MdAPE change +2.5%, MAPE change -0.1%; 2024-10 n=2451, MdAPE change +1.6%, MAPE change +1.1%; 2025-01 n=1339, MdAPE change -0.1%, MAPE change +2.1%; 2024-12 n=1345, MdAPE change -0.6%, MAPE change +1.3%
- `actual_price_band`: Q4 n=1767, MdAPE change +2.1%, MAPE change -0.9%; Q2 n=1766, MdAPE change +1.7%, MAPE change +1.8%; Q3 n=1766, MdAPE change +1.3%, MAPE change +1.5%; Q1 n=1767, MdAPE change +0.7%, MAPE change +2.3%
- `house_type_name`: Condo Apt n=3560, MdAPE change +2.2%, MAPE change +0.5%; Freehold Townhouse n=274, MdAPE change +0.3%, MAPE change +0.6%; Semi-Detached n=673, MdAPE change -2.1%, MAPE change +1.0%; Condo Townhouse n=427, MdAPE change -2.2%, MAPE change +1.8%; Detached n=2099, MdAPE change -2.6%, MAPE change +1.4%
- `house_style`: 2 1/2 Storey n=147, MdAPE change +8.9%, MAPE change +1.2%; Apartment n=3375, MdAPE change +2.3%, MAPE change +0.4%; Loft n=121, MdAPE change -0.9%, MAPE change +2.4%; Bungalow n=650, MdAPE change -1.3%, MAPE change +1.7%; 2-Storey n=1736, MdAPE change -2.0%, MAPE change +0.6%; Bungalow-Raised n=110, MdAPE change -5.5%, MAPE change +5.4%; 3-Storey n=452, MdAPE change -5.7%, MAPE change +2.8%
- `municipality`: Scarborough n=977, MdAPE change +3.9%, MAPE change +1.5%; Toronto n=4019, MdAPE change +0.4%, MAPE change +0.8%; Etobicoke n=678, MdAPE change -0.5%, MAPE change +0.8%; North York n=1318, MdAPE change -2.1%, MAPE change +1.2%

## list_aware/Q50

n=7,066; MdAPE 4.4726% → 4.4039% (+1.54%); MAPE 5.5769% → 5.5302% (-0.84%); median signed error +0.765% → +0.440%.

- `month`: 2024-12 n=1345, MdAPE change +1.5%, MAPE change -1.2%; 2024-10 n=2451, MdAPE change +1.2%, MAPE change -0.9%; 2025-03 n=1931, MdAPE change +1.1%, MAPE change -0.4%; 2025-01 n=1339, MdAPE change +0.5%, MAPE change -0.9%
- `actual_price_band`: Q2 n=1766, MdAPE change +2.3%, MAPE change -1.7%; Q1 n=1767, MdAPE change +1.2%, MAPE change -2.2%; Q3 n=1766, MdAPE change +0.6%, MAPE change -0.7%; Q4 n=1767, MdAPE change +0.2%, MAPE change +0.7%
- `house_type_name`: Detached n=2099, MdAPE change +2.8%, MAPE change -0.4%; Condo Townhouse n=427, MdAPE change +2.4%, MAPE change +0.0%; Condo Apt n=3560, MdAPE change +1.8%, MAPE change -1.7%; Freehold Townhouse n=274, MdAPE change +0.0%, MAPE change -1.5%; Semi-Detached n=673, MdAPE change -4.0%, MAPE change +0.8%
- `house_style`: Bungalow-Raised n=110, MdAPE change +7.4%, MAPE change -3.4%; 2 1/2 Storey n=147, MdAPE change +6.5%, MAPE change -0.6%; 2-Storey n=1736, MdAPE change +3.0%, MAPE change -0.3%; Apartment n=3375, MdAPE change +2.0%, MAPE change -1.7%; Bungalow n=650, MdAPE change -1.5%, MAPE change -0.0%; 3-Storey n=452, MdAPE change -1.6%, MAPE change -0.7%; Loft n=121, MdAPE change -10.9%, MAPE change -0.5%
- `municipality`: Scarborough n=977, MdAPE change +5.6%, MAPE change -1.1%; Toronto n=4019, MdAPE change +1.0%, MAPE change -0.8%; North York n=1318, MdAPE change +0.2%, MAPE change -0.7%; Etobicoke n=678, MdAPE change -1.3%, MAPE change -0.8%

## list_aware/CatBoost_MAE

n=7,066; MdAPE 4.4726% → 4.4096% (+1.41%); MAPE 5.5769% → 5.5623% (-0.26%); median signed error +0.765% → +0.310%.

- `month`: 2024-12 n=1345, MdAPE change +3.3%, MAPE change -0.1%; 2024-10 n=2451, MdAPE change +1.6%, MAPE change -0.8%; 2025-03 n=1931, MdAPE change +1.1%, MAPE change -0.1%; 2025-01 n=1339, MdAPE change -0.1%, MAPE change +0.4%
- `actual_price_band`: Q2 n=1766, MdAPE change +1.7%, MAPE change -1.0%; Q1 n=1767, MdAPE change +1.1%, MAPE change -2.5%; Q3 n=1766, MdAPE change +0.8%, MAPE change -0.0%; Q4 n=1767, MdAPE change -1.8%, MAPE change +1.7%
- `house_type_name`: Condo Townhouse n=427, MdAPE change +3.0%, MAPE change +2.4%; Condo Apt n=3560, MdAPE change +1.2%, MAPE change -1.6%; Detached n=2099, MdAPE change +0.3%, MAPE change +0.3%; Semi-Detached n=673, MdAPE change -0.5%, MAPE change +1.1%; Freehold Townhouse n=274, MdAPE change -8.9%, MAPE change +0.8%
- `house_style`: Bungalow-Raised n=110, MdAPE change +2.6%, MAPE change +1.1%; 2-Storey n=1736, MdAPE change +1.6%, MAPE change +0.5%; 2 1/2 Storey n=147, MdAPE change +1.5%, MAPE change -0.1%; Apartment n=3375, MdAPE change +0.6%, MAPE change -1.6%; Bungalow n=650, MdAPE change -4.0%, MAPE change +1.0%; 3-Storey n=452, MdAPE change -4.5%, MAPE change +0.3%; Loft n=121, MdAPE change -10.3%, MAPE change -0.5%
- `municipality`: Scarborough n=977, MdAPE change +4.1%, MAPE change +1.5%; Toronto n=4019, MdAPE change +2.3%, MAPE change -0.8%; Etobicoke n=678, MdAPE change -0.7%, MAPE change -0.1%; North York n=1318, MdAPE change -2.6%, MAPE change +0.3%
