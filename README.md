# nanjing-housing-regression
Final project for housing price prediction using regression
# Nanjing Housing Price Prediction

Final project using multiple linear regression to analyze and predict housing prices in Qixia District, Nanjing.

## Project Highlights

- Web-scraped data from “fangtianxia”
- Cleaned, structured with Moonshot AI
- Linear regression using sklearn and statsmodels
- R² ≈ 0.67, MAE ≈ 74w, p-value filtering
- Future plan: XGBoost, log(price), interactive map

## Files

- `multiple linear regression.py` — main regression model
- `Final Structured Dataset of Housing Listings.xlsx` — cleaned dataset
- `report_summary.pdf` — final analysis/PPT

## Result Summary

| Metric | Value     |
|--------|-----------|
| R²     | 0.669     |
| MAE    | ¥741,125  |
| MSE    | 1.38e+12  |

## To Run

```bash
python multiple linear regression.py
