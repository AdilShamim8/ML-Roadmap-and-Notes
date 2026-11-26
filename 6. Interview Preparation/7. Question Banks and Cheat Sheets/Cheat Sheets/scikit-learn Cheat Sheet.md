# scikit-learn Cheat Sheet

One page. The API grammar, the leakage-safe patterns, and the estimator table interviewers expect you to have memorized.

## The API grammar (everything follows it)
```python
model = Estimator(hyperparams)
model.fit(X_train, y_train)                 # (X, y) supervised; (X) unsupervised
y_pred = model.predict(X_new)               # labels / values
p = model.predict_proba(X_new)              # class probabilities (classifiers)
model.transform(X)                          # preprocessing / dim reduction
model.fit_transform(X)                      # shorthand (fit then transform)
model.score(X_test, y_test)                 # default metric (accuracy / R²)
```
`set_params` / `get_params` power GridSearchCV — every estimator speaks them.

## The leakage-safe pattern (the one thing to never get wrong)
```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                         OneHotEncoder(handle_unknown="ignore"))

pre = ColumnTransformer([("num", num_pipe, num_cols),
                         ("cat", cat_pipe, cat_cols)])
pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])
pipe.fit(X_train, y_train)      # transformers refit per fold inside cross_val_score
```
**Say it:** *"all preprocessing lives inside the pipeline, so cross-validation refits it per fold — fitting a scaler before the split leaks."*

## Model selection & evaluation
```python
from sklearn.model_selection import (train_test_split, cross_val_score,
    StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV)

train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

cv = StratifiedKFold(5, shuffle=True, random_state=0)   # classification
cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")   # -> array; report mean ± std

GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X, y)
RandomizedSearchCV(pipe, dist, n_iter=50, cv=cv)        # big spaces

# split scheme = how data was generated:
#   grouped rows (users/patients)  -> GroupKFold(groups=...)
#   time series                   -> TimeSeriesSplit()
```

## Metrics
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, classification_report)

roc_auc_score(y, p)                 # ranking (p = proba[:, 1])
average_precision_score(y, p)       # PR-AUC — the imbalanced-data choice
precision_score(y, yhat, zero_division=0)
classification_report(y, yhat)      # per-class everything
mean_squared_error(y, yhat) ** 0.5  # RMSE (no rmse in sklearn)
```

## Estimator table (pick by data shape)

| Data | First reach | Notes |
| --- | --- | --- |
| Tabular classification | `LogisticRegression` → `HistGradientBoostingClassifier` | linear baseline then boosted trees |
| Tabular regression | `Ridge` → `HistGradientBoostingRegressor` | same ladder |
| Interpretable tree | `DecisionTreeClassifier(max_depth=3)` | for explanation, not performance |
| Bagging | `RandomForestClassifier` | variance reduction; OOB score free |
| Boosting | `HistGradientBoosting*` | LightGBM-style, in-sklearn |
| Naive Bayes | `MultinomialNB` | text/sparse, one pass |
| KNN | `KNeighborsClassifier(n_neighbors=5)` | scale first; lazy |
| SVM | `SVC(kernel="rbf")` | scale first; n ≲ 10⁵ |
| Clustering | `KMeans` → `DBSCAN` | spheres vs arbitrary shape+noise |
| Dim reduction | `PCA(n_components=0.95)` | 0.95 = keep 95% variance |
| Anomaly | `IsolationForest` | contamination = expected rate |

## The 2026 API notes (updated APIs matter)
```python
LogisticRegression(multi_class=...)        # REMOVED — multiclass is automatic
StackingClassifier(estimators=[...])       # 'base_estimator' -> 'estimators/estimator'
OneHotEncoder(sparse_output=False)         # was 'sparse'
KMeans(n_init=10)                          # explicit n_init (default changed)
HistGradientBoostingClassifier(...)        # handles NaN natively, no scaling
```

## Preprocessing quick-map
| Problem | Tool |
| --- | --- |
| Numeric missing | `SimpleImputer(median)` / `KNNImputer` |
| Categorical missing | `SimpleImputer("most_frequent")` or constant `"missing"` |
| Scale | `StandardScaler` (default) · `RobustScaler` (outliers) · `MinMaxScaler` (bounds) |
| Skew | `PowerTransformer` / `FunctionTransformer(np.log1p)` |
| Ordinal category | `OrdinalEncoder(categories=[order])` |
| High-cardinality | `TargetEncoder(cv=5)` **inside pipeline** |
| Binning | `KBinsDiscretizer(strategy="quantile")` |

## Fit-time survival
- `n_jobs=-1` everywhere it exists.
- `HistGradientBoosting*` for >10⁴ rows (histogram algorithm).
- `SGDClassifier(loss="log_loss")` for streaming/huge-n linear.
- `partial_fit` for true online learning.

## The interview reflexes
- Baseline first (dummy/majority/median) — every number is relative to it.
- Report CV mean **± std**, never a single fold.
- Scaling: trees no, distance/gradient models yes.
- Imbalance: `class_weight="balanced"` + PR-AUC, not accuracy.
- Calibration: `CalibratedClassifierCV(method="isotonic")` when probabilities are consumed.
