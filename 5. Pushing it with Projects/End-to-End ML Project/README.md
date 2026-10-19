# End-to-End ML Project: Titanic Survival Prediction

The **Level 5 capstone** of the [ML Roadmap and Notes](https://github.com/AdilShamim8/ML-Roadmap-and-Notes): one complete machine-learning project, run the way a professional would run it — problem framing → EDA → feature engineering → leak-free pipelines → model selection → tuning → holdout evaluation → persistence and serving.

## Results

| Aspect | Result |
|---|---|
| Final holdout F1 (survivors) | **0.71** · accuracy **0.79** |
| Best model | RandomForest (500 trees, max depth 4) |
| Top signals | Title (Mr), Sex, Fare, Pclass, HasCabin, FamilySize |
| Validation | 5-fold stratified CV + untouched 20% holdout |

## The Skills It Exercises

Every stage maps back to a roadmap level: **Level 1** (pandas EDA), **Level 2** (metrics, bias-variance thinking), **Level 3** (imputation, encoding, pipelines, cross-validation, tuning, data leakage, model serving), and **Level 4** (domain intuition for feature engineering).

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib joblib
jupyter notebook "End-to-End ML Project.ipynb"
```

The dataset (`titanic.csv`, 891 passengers) ships in this folder — the notebook runs offline end-to-end and writes `titanic_pipeline.joblib` on execution.

## What's Next

- Reuse this lifecycle on a bigger dataset from the [100 AI/ML/DL Projects](https://github.com/AdilShamim8/100-AI-Machine-Learning-Deep-Learnin-Projects) collection
- Take the serving sketch to production with the Level 3 §17 Model Serving notes and the Level 6 MLOps stack
