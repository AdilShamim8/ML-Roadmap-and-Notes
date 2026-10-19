# End-to-End Regression Project: California Housing Prices

The **second Level 5 capstone** of the [ML Roadmap and Notes](https://github.com/AdilShamim8/ML-Roadmap-and-Notes) — the regression companion to the [Titanic classification project](../End-to-End%20ML%20Project/README.md). Same professional lifecycle, different craft: a **log-transformed target**, measured feature engineering, dollar-metric reporting, and an error analysis that finds exactly where the model's money is lost.

## Results

| Aspect | Result |
|---|---|
| Holdout RMSE (20% of 20,640 blocks) | **$46,985** · MAE **$30,503** · R² **0.832** · MAPE **16.9%** |
| Naive baseline (predict the median) | $117,312 — the model cuts typical error to a quarter |
| Best model | HistGradientBoosting (CV RMSE 0.141 log vs linear's 0.218) |
| Feature engineering | `bedrooms_per_room` built and kept: 0.1395 → 0.1386 CV, rank-6 importance |
| Honest finding | The $500k price cap costs 2.6× the average error on luxury blocks; location (lat/long) drives the model more than income |
| Leakage check | Holdout log-RMSE 0.1395 vs CV 0.1386 — the CV estimate was an honest promise |

## The Skills It Exercises

Every stage maps back to a roadmap level: **Level 1** (pandas EDA, distribution thinking), **Level 2** (regression metrics, skew and transforms), **Level 3** (pipelines, cross-validation, model tuning, permutation importance, model serving, data-leakage discipline), and **Level 4** (spatial intuition — the map *is* the feature).

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib joblib
jupyter notebook "End-to-End Regression Project.ipynb"
```

The dataset (`california_housing.csv`, 20,640 California census block groups from Pace & Barry 1997, as bundled with scikit-learn) ships in this folder — the notebook runs fully offline and writes `housing_model.joblib` on execution.

## What's Next

- Add SHAP for per-block explanations, or quantile regression for uncertainty bands
- Try the harder, uncensored Ames (Iowa) housing dataset on Kaggle
- Reuse the lifecycle on a regression problem from the [100 AI/ML/DL Projects](https://github.com/AdilShamim8/100-AI-Machine-Learning-Deep-Learnin-Projects) collection
