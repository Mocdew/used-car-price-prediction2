# Used Car Price Prediction

Predicting resale prices of used cars using a combined Cars24 listings dataset, comparing Linear Regression, Ridge Regression, and XGBoost, tuned with [Optuna](https://optuna.org/).

## Problem

Given listing details for a used car (make, age, distance driven, fuel type, location, ownership history, etc.), predict its resale price.

## Data

This project expects a file named `cars_24_combined.csv` in the project root, with (at minimum) the following columns:

| Column      | Description                              |
|-------------|-------------------------------------------|
| Car Name    | Full listing name (brand is extracted from this) |
| Year        | Model year                               |
| Distance    | Distance driven (km)                     |
| Owner       | Number of previous owners                |
| Fuel        | Fuel type (Petrol, Diesel, CNG, etc.)     |
| Drive       | Transmission type                        |
| Type        | Body type (Hatchback, SUV, Sedan, etc.)  |
| Location    | Listing location                         |
| Price       | Target — resale price                    |

> The dataset isn't included in this repo. If you have a license/source to redistribute it, add it here and remove this note; otherwise, point users to where they can obtain it (e.g. Kaggle) and update the note above.

## Approach

1. **Clean** — drop rows missing an identifiable car or year, fill missing locations, convert model year into car age.
2. **Explore** — check distributions and outliers in price and distance, and see how price relates to age, distance, and fuel type.
3. **Encode** — frequency-encode `Location`; one-hot encode `Brand` (extracted from the car name), `Fuel`, `Drive`, and `Type`.
4. **Model** — train baseline Linear Regression, Ridge Regression, and XGBoost models.
5. **Tune** — run Optuna hyperparameter searches (5-fold CV on RMSE) for each model, then compare tuned test-set performance.

## Results

See the final cell of the notebook for the up-to-date comparison table (`Best CV RMSE` vs. `Test RMSE` for each model). XGBoost is expected to outperform the linear models since it can capture non-linear interactions between features that Linear/Ridge regression can't represent.

## Setup

```bash
pip install -r requirements.txt
```

Then open `used_car_price_prediction.ipynb` and run all cells (with `cars_24_combined.csv` present in the same directory).

## Notes

- Model year is converted into car age (`current_year - Year`) as of the notebook's write date; update `current_year` in the data cleaning section if you re-run this later.
- The Optuna tuning cells (50 trials each for Ridge and XGBoost) can take a few minutes to run.
