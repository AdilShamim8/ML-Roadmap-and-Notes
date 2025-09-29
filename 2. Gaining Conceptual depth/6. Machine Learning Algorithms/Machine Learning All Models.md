# Machine Learning — Quick Reference Table (most important models)

| **Model / Method**                     |                      **Type / Role** | **When to use**                                        | **Intuition (one line)**                                                       |                                      **Key sklearn / lib** | **Tuning knobs (important)**                                                  |
| -------------------------------------- | -----------------------------------: | ------------------------------------------------------ | ------------------------------------------------------------------------------ | ---------------------------------------------------------: | ----------------------------------------------------------------------------- |
| **Linear Regression**                  |                  Regression (linear) | Baseline for continuous targets                        | Fit a line (or plane) that minimizes squared error.                            |                    `sklearn.linear_model.LinearRegression` | `fit_intercept`, regularize with `Ridge/Lasso`                                |
| **Gradient Descent**                   |               Optimization algorithm | Used to train many models (logistic, NN, linear w/SGD) | Iteratively update params in negative gradient direction.                      | `sklearn.linear_model.SGDClassifier/Regressor` (or custom) | `learning_rate` (α), `batch_size`, `epochs`, momentum                         |
| **Logistic Regression**                |              Classification (linear) | Baseline binary/multi-class classification             | Sigmoid on linear combo → probability of class.                                |                  `sklearn.linear_model.LogisticRegression` | `C` (inverse reg), `penalty` (l1/l2), solver                                  |
| **Support Vector Machines (SVM)**      |          Classification / Regression | Small–medium, high-dimensional data                    | Find max-margin hyperplane (can use kernels).                                  |                                  `sklearn.svm.SVC` / `SVR` | `C`, `kernel`, `gamma` (RBF), `degree`                                        |
| **Naive Bayes**                        |             Probabilistic classifier | Text, high-d sparse features, quick baseline           | Assume feature independence; compute class probabilities.                      |          `sklearn.naive_bayes.GaussianNB`, `MultinomialNB` | `alpha` (smoothing, for multinomial)                                          |
| **K-Nearest Neighbors (KNN)**          |                       Instance-based | Small datasets, simple baseline                        | Predict by majority (classification) or average (regression) of nearest neighbors. |                   `sklearn.neighbors.KNeighborsClassifier` | `n_neighbors`, `weights` (uniform/distance), `p` (distance metric)            |
| **Decision Trees**                     |             Interpretable tree model | Quick interpretable models, non-linearities            | Split features to make pure leaves (if unchecked, overfits).                   |            `sklearn.tree.DecisionTreeClassifier/Regressor` | `max_depth`, `min_samples_leaf`, `criterion`                                  |
| **Random Forest**                      |          Ensemble (bagging of trees) | Strong general-purpose baseline                        | Many trees trained on bootstraps → average/vote reduces variance.              |                  `sklearn.ensemble.RandomForestClassifier` | `n_estimators`, `max_depth`, `max_features`                                   |
| **Bagging**                            |                 Ensemble meta-method | Reduce variance for unstable learners                  | Train same base learner on bootstrap samples, aggregate.                       |                       `sklearn.ensemble.BaggingClassifier` | `n_estimators`, `base_estimator`, `max_samples`                               |
| **AdaBoost**                           |                Boosting (sequential) | When weak learners are available                       | Sequentially focus on misclassified samples (weights).                         |                      `sklearn.ensemble.AdaBoostClassifier` | `n_estimators`, `learning_rate`, base estimator depth                         |
| **Gradient Boosting**                  |                Boosting (sequential) | High-performance tabular predictions                   | Sequentially add models to correct previous residuals.                         |              `sklearn.ensemble.GradientBoostingClassifier` | `n_estimators`, `learning_rate`, `max_depth`                                  |
| **XGBoost**                            |                 Boosting (optimized) | Large/tabular datasets (industry standard)             | Fast, regularized gradient boosting with many optimizations.                   |                   `xgboost.XGBClassifier` / `XGBRegressor` | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` |
| **PCA (Principal Component Analysis)** |             Dimensionality reduction | Compress features, visualize, speed-up models          | Rotate to orthogonal axes that capture most variance.                          |                                `sklearn.decomposition.PCA` | `n_components`, `svd_solver`                                                  |
| **K-Means Clustering**                 |              Unsupervised clustering | Partitioning into K spherical-ish clusters             | Assign points to nearest centroid, update centroids iteratively.               |                                   `sklearn.cluster.KMeans` | `n_clusters`, `init`, `n_init`, `max_iter`                                    |
| **Hierarchical Clustering**            |              Unsupervised clustering | Small datasets, dendrogram insights                    | Merge/split clusters to form a tree (agglomerative/divisive).                    |                  `sklearn.cluster.AgglomerativeClustering` | `n_clusters`, `linkage`                                                       |
| **DBSCAN**                             |             Density-based clustering | Arbitrary-shaped clusters, noise detection             | Expand clusters from dense cores using eps & minPts.                           |                                   `sklearn.cluster.DBSCAN` | `eps`, `min_samples`, `metric`                                                |
| **t-SNE (T-SNE)**                      | Non-linear embedding / visualization | Visualize high-dim data in 2D/3D (exploration)         | Preserves local neighbor structure; good for plots (not features).             |                                    `sklearn.manifold.TSNE` | `perplexity`, `n_iter`, `learning_rate`                                       |

---
```python
# ML model demos — run in a Jupyter cell
# Requirements:
# pip install numpy pandas scikit-learn matplotlib seaborn xgboost --upgrade

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from time import time

# Helpers for consistent evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score
)

# Datasets
from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

diabetes = datasets.load_diabetes()
X_db, y_db = diabetes.data, diabetes.target

from sklearn.datasets import make_blobs
X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, n_features=4, random_state=42)

# Split datasets
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_db, y_db, test_size=0.2, random_state=42)
Xcl_train, Xcl_test, _y1, _y2 = train_test_split(X_blobs, y_blobs, test_size=0.2, random_state=42)


########################################################
# 1) Linear Regression (sklearn) — Regression demo
########################################################
print("\n# 1) Linear Regression (sklearn)")
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xr_train, yr_train)
y_pred = lr.predict(Xr_test)
print("MSE:", mean_squared_error(yr_test, y_pred))
print("R2 :", r2_score(yr_test, y_pred))

########################################################
# 2) Gradient Descent (manual) demonstrating optimization
#    — simple linear regression via GD on diabetes dataset
########################################################
print("\n# 2) Gradient Descent (manual) for linear regression")
# add intercept
X = np.hstack([np.ones((Xr_train.shape[0],1)), Xr_train])
Xt = np.hstack([np.ones((Xr_test.shape[0],1)), Xr_test])

def gradient_descent(X, y, lr_init=0.01, epochs=5000, verbose=False):
    n, m = X.shape
    theta = np.zeros(m)
    lr = lr_init
    for epoch in range(epochs):
        preds = X.dot(theta)
        error = preds - y
        grad = (1.0/n) * X.T.dot(error)
        theta -= lr * grad
        if verbose and epoch % 1000 == 0:
            print(f"epoch {epoch}, loss {np.mean(error**2):.4f}")
    return theta

theta = gradient_descent(X, yr_train, lr_init=0.01, epochs=5000, verbose=False)
preds_test = Xt.dot(theta)
print("Manual GD MSE:", mean_squared_error(yr_test, preds_test))
print("Manual GD R2 :", r2_score(yr_test, preds_test))

########################################################
# 3) Logistic Regression — Classification demo (Iris)
########################################################
print("\n# 3) Logistic Regression (sklearn)")
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto', random_state=42)
logreg.fit(Xc_train, yc_train)
y_pred = logreg.predict(Xc_test)
print("Accuracy:", accuracy_score(yc_test, y_pred))
print(classification_report(yc_test, y_pred))

########################################################
# 4) Support Vector Machine (SVM)
########################################################
print("\n# 4) Support Vector Machine (SVM)")
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=42)
svm.fit(Xc_train, yc_train)
y_pred = svm.predict(Xc_test)
print("Accuracy:", accuracy_score(yc_test, y_pred))

########################################################
# 5) Naive Bayes (Gaussian) — good baseline for small feature sets / numeric
########################################################
print("\n# 5) Naive Bayes (GaussianNB)")
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(Xc_train, yc_train)
y_pred = gnb.predict(Xc_test)
print("Accuracy:", accuracy_score(yc_test, y_pred))

########################################################
# 6) K-Nearest Neighbors (KNN)
########################################################
print("\n# 6) K-Nearest Neighbors (KNN)")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xc_train, yc_train)
y_pred = knn.predict(Xc_test)
print("Accuracy:", accuracy_score(yc_test, y_pred))

########################################################
# 7) Decision Tree
########################################################
print("\n# 7) Decision Tree")
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(Xc_train, yc_train)
y_pred = dt.predict(Xc_test)
print("Accuracy:", accuracy_score(yc_test, y_pred))
# (plotting omitted for script; use plot_tree in notebook to visualize)

########################################################
# 8) Random Forest
########################################################
print("\n# 8) Random Forest")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(Xc_train, yc_train)
y_pred = rf.predict(Xc_test)
print("Accuracy:", accuracy_score(yc_test, y_pred))

########################################################
# 9) Bagging (BaggingClassifier) demo using DecisionTree as base
########################################################
print("\n# 9) Bagging (BaggingClassifier)")
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, random_state=42)
bag.fit(Xc_train, yc_train)
print("Accuracy:", accuracy_score(yc_test, bag.predict(Xc_test)))

########################################################
# 10) AdaBoost
########################################################
print("\n# 10) AdaBoost")
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
adb.fit(Xc_train, yc_train)
print("Accuracy:", accuracy_score(yc_test, adb.predict(Xc_test)))

########################################################
# 11) Gradient Boosting (sklearn)
########################################################
print("\n# 11) GradientBoosting (sklearn)")
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(Xc_train, yc_train)
print("Accuracy:", accuracy_score(yc_test, gb.predict(Xc_test)))

########################################################
# 12) XGBoost (if available)
########################################################
print("\n# 12) XGBoost (if installed)")
try:
    import xgboost as xgb
    xgbc = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42)
    xgbc.fit(Xc_train, yc_train)
    print("Accuracy:", accuracy_score(yc_test, xgbc.predict(Xc_test)))
except Exception as e:
    print("xgboost not available. To install: pip install xgboost")
    # fallback: show earlier gb result
    print("Skipping XGBoost demo (using sklearn GradientBoosting instead).")

########################################################
# 13) PCA — dimensionality reduction (fit + transform)
########################################################
print("\n# 13) PCA (dimensionality reduction)")
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_iris)
print("Explained variance ratios:", pca.explained_variance_ratio_)

########################################################
# 14) K-Means clustering demo (with blobs)
########################################################
print("\n# 14) K-Means Clustering")
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_blobs)
labels = km.predict(X_blobs)
print("Silhouette (kmeans):", silhouette_score(X_blobs, labels))

########################################################
# 15) Hierarchical Clustering (Agglomerative)
########################################################
print("\n# 15) Hierarchical (Agglomerative) Clustering")
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg_labels = agg.fit_predict(X_blobs)
print("Silhouette (agg):", silhouette_score(X_blobs, agg_labels))

########################################################
# 16) DBSCAN — density-based clustering
########################################################
print("\n# 16) DBSCAN")
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=5, metric='euclidean')
db_labels = dbscan.fit_predict(X_blobs)
# DBSCAN labels -1 are noise; only compute silhouette on core points if >1 cluster
unique_clusters = set(db_labels)
if len([c for c in unique_clusters if c != -1]) > 1:
    s = silhouette_score(X_blobs, db_labels)
    print("Silhouette (DBSCAN):", s)
else:
    print("DBSCAN found <=1 cluster (likely noise or a single cluster). Labels:", unique_clusters)

########################################################
# 17) t-SNE — visualization (use small datasets)
########################################################
print("\n# 17) t-SNE (visualization)")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca')
X_tsne = tsne.fit_transform(X_iris)
print("t-SNE result shape:", X_tsne.shape)

########################################################
# End summary: quick printout of models ran and sample metrics
########################################################
print("\n# Summary metrics snapshot")
print("LinearRegression R2:", round(r2_score(yr_test, y_pred if 'y_pred' in locals() else preds_test), 4))
print("LogisticRegression Acc:", round(accuracy_score(yc_test, logreg.predict(Xc_test)), 4))
print("RandomForest Acc:", round(accuracy_score(yc_test, rf.predict(Xc_test)), 4))
```

# Machine Learning — All Models (comprehensive list table)

## Supervised — Regression

| Model                           |             Type | Use case                     | Notes                                 |
| ------------------------------- | ---------------: | ---------------------------- | ------------------------------------- |
| Linear Regression               |       Parametric | Continuous targets           | Baseline, interpretable               |
| Polynomial Regression           |       Parametric | Nonlinear relationships      | Linear model on expanded features     |
| Ridge / L2                      |      Regularized | Multicollinear data          | L2 penalty shrinks weights            |
| Lasso / L1                      |      Regularized | Sparsity / feature selection | L1 sets some weights to 0             |
| ElasticNet                      |      Regularized | Mix of L1 & L2               | Balance sparsity + stability          |
| Bayesian Ridge                  |    Probabilistic | Uncertainty estimates        | Bayesian version of linear regression |
| Huber Regression                |           Robust | Outliers present             | Combines L1 & L2 loss                 |
| SGD Regressor                   |           Online | Very large datasets          | Stochastic gradient updates           |
| KNN Regressor                   |   Instance-based | Local interpolation          | Simple, non-parametric                |
| Decision Tree Regressor         |             Tree | Nonlinear regression         | Easy to interpret, can overfit        |
| Random Forest Regressor         |         Ensemble | Strong baseline              | Bagged trees — robust                 |
| ExtraTrees Regressor            |         Ensemble | Fast tree ensemble           | Randomized splits                     |
| Gradient Boosting Regressor     |         Boosting | High-performance tabular     | Sequential boosting                   |
| XGBoost / LightGBM / CatBoost   |         Boosting | Large-scale tabular          | Optimized boosters                    |
| SVR (Support Vector Regression) |           Kernel | Small/med nonlinear data     | Kernel trick                          |
| Gaussian Process Regressor      |    Probabilistic | Small data, uncertainty      | Expensive, flexible                   |
| PLS Regression                  |  Latent variable | Multicollinearity            | Few components explain X→y            |
| MLP Regressor (NN)              |           Neural | Complex non-linear           | Needs tuning + data                   |
| ARIMA / SARIMA                  | Time-series stat | Univariate forecasting       | Statistical time-series model         |

---

## Supervised — Classification

| Model                                        |            Type | Use case                     | Notes                            |
| -------------------------------------------- | --------------: | ---------------------------- | -------------------------------- |
| Logistic Regression                          |          Linear | Binary / multiclass          | Baseline classifier              |
| Linear Discriminant Analysis (LDA)           |          Linear | Class separation             | Assumes Gaussian features        |
| Quadratic Discriminant Analysis (QDA)        |   Probabilistic | Nonlinear class boundaries   | Class-specific covariances       |
| KNN Classifier                               |  Instance-based | Small datasets               | Distance-based                   |
| Decision Tree Classifier                     |            Tree | Interpretable classification | Prone to overfitting             |
| Random Forest Classifier                     |        Ensemble | Tabular classification       | Strong baseline                  |
| ExtraTrees Classifier                        |        Ensemble | Fast ensemble                | Randomized trees                 |
| Gradient Boosting Classifier                 |        Boosting | High-performance             | Strong on tabular                |
| XGBoost / LightGBM / CatBoost                |        Boosting | Industry standard            | Handles categorical / large data |
| SVM (SVC / LinearSVC)                        |   Kernel/Linear | High-dim data                | Kernelized or linear             |
| Naive Bayes (Gaussian/Multinomial/Bernoulli) |   Probabilistic | Text, baseline               | Fast, simple                     |
| Perceptron / SGD Classifier                  | Linear / online | Large-scale linear tasks     | Online updates                   |
| MLP Classifier (Neural)                      |          Neural | Non-linear classification    | Needs tuning                     |
| Calibrated Classifiers                       |   Probabilistic | Calibrated probs             | Platt / isotonic                 |
| One-vs-Rest / One-vs-One                     |            Meta | Multi-class strategies       | Wraps binary models              |

---

## Unsupervised — Clustering

| Model                        |            Type | Use case                 | Notes                     |
| ---------------------------- | --------------: | ------------------------ | ------------------------- |
| K-Means / MiniBatchKMeans    |       Partition | Spherical clusters       | Fast, needs K             |
| Hierarchical / Agglomerative |    Hierarchical | Dendrograms, small data  | No K needed initially     |
| DBSCAN                       |   Density-based | Arbitrary shapes + noise | Detects outliers          |
| OPTICS                       |   Density-based | Varying density clusters | Like DBSCAN but flexible  |
| MeanShift                    |    Mode-seeking | Unknown cluster counts   | Kernel-based              |
| Birch                        |    Hierarchical | Large datasets           | Builds clustering tree    |
| Spectral Clustering          |     Graph-based | Non-convex clusters      | Uses eigenvectors         |
| Gaussian Mixture Model (GMM) |   Probabilistic | Soft clustering          | Probabilistic memberships |
| Affinity Propagation         | Message-passing | Detect exemplars         | No K but sensitive params |

---

## Unsupervised — Dimensionality reduction / Representation

| Model                                |                    Type | Use case                   | Notes                       |
| ------------------------------------ | ----------------------: | -------------------------- | --------------------------- |
| PCA (Principal Component Analysis)   |                  Linear | Compression, noise removal | Orthogonal components       |
| Truncated SVD                        |                  Linear | Sparse / TF-IDF data       | SVD for sparse matrices     |
| Kernel PCA                           |              Non-linear | Nonlinear features         | Uses kernel trick           |
| Incremental PCA                      |                  Online | Streaming / large data     | Batch updates               |
| NMF (Non-negative MF)                |           Factorization | Parts-based decomposition  | Non-negativity constraint   |
| ICA (Independent Component Analysis) | Blind source separation | Signal separation          | Assumes independence        |
| t-SNE                                |                Manifold | Visualization              | Nonlinear, slow             |
| UMAP                                 |                Manifold | Visualization + structure  | Faster than t-SNE often     |
| LLE / Isomap                         |                Manifold | Nonlinear embeddings       | Preserve manifold geometry  |
| Autoencoders (NN)                    |                  Neural | Learned compressed reps    | Undercomplete / variational |

---

## Ensemble & Meta methods

| Model                                 |          Type | Use case                 | Notes                        |
| ------------------------------------- | ------------: | ------------------------ | ---------------------------- |
| Bagging / BaggingClassifier           |      Ensemble | Reduce variance          | Parallel trees/models        |
| Random Forest / ExtraTrees            |      Ensemble | Robust baseline          | Bagged tree ensembles        |
| Boosting (AdaBoost, GradientBoosting) |      Ensemble | Improve weak learners    | Sequential learning          |
| XGBoost / LightGBM / CatBoost         |      Boosting | Fast & accurate          | Widely used in practice      |
| Stacking / Stacked Generalization     | Meta-ensemble | Combine diverse learners | Meta-learner on base outputs |
| Voting Classifier                     |      Ensemble | Simple model blend       | Hard/soft voting             |
| Snapshot Ensembles                    |      Ensemble | Deep learning ensembles  | Multiple epochs snapshots    |

---

## Probabilistic / Bayesian models

| Model                       |                     Type | Use case                          | Notes                     |
| --------------------------- | -----------------------: | --------------------------------- | ------------------------- |
| Gaussian Process            |            Probabilistic | Regression/classification with UQ | Expensive O(n³)           |
| Bayesian Linear / Ridge     |            Probabilistic | Uncertainty on params             | Priors + posteriors       |
| Hidden Markov Models (HMM)  | Sequential probabilistic | Speech, sequences                 | Discrete latent states    |
| Bayesian Networks           |         Graphical models | Causal / probabilistic reasoning  | Structure learning needed |
| Kalman Filter / Extended KF |              State-space | Tracking, time-series             | Online estimation         |

---

## Time Series & Forecasting

| Model                                |           Type | Use case                       | Notes                          |
| ------------------------------------ | -------------: | ------------------------------ | ------------------------------ |
| ARIMA / SARIMA                       |    Statistical | Univariate forecasting         | Stationarity required          |
| VAR (Vector AR)                      |   Multivariate | Interrelated series            | Multivariate time series       |
| Exponential Smoothing / Holt-Winters |    Statistical | Seasonality/trend              | Simple seasonal forecasting    |
| Prophet                              | Additive model | Business forecasting           | User-friendly                  |
| State Space Models                   |    Statistical | Time-series with latent states | Flexible frameworks            |
| LSTM / GRU (RNNs)                    |         Neural | Sequence forecasting           | Captures temporal dependencies |
| Temporal Fusion Transformer          |         Neural | Multi-horizon forecasting      | Attention + temporal features  |

---

## Deep Learning Architectures (core)

| Model                            |              Type | Use case                | Notes                              |
| -------------------------------- | ----------------: | ----------------------- | ---------------------------------- |
| MLP (Fully-connected)            |            Neural | Tabular / dense tasks   | Baseline NN                        |
| CNN (Convolutional)              |            Neural | Images, spatial data    | Local receptive fields             |
| RNN / LSTM / GRU                 |            Neural | Sequences, time-series  | Temporal dependencies              |
| Transformer                      |      Attention NN | NLP, sequences          | Scales well; BERT/GPT family       |
| BERT / RoBERTa / DistilBERT      |       Transformer | NLP tasks               | Pretrained encoders                |
| GPT / Decoder models             |       Transformer | Text generation         | Large language models              |
| Autoencoders / VAE               | Neural generative | Compression, generation | Latent representation              |
| GANs (DCGAN, StyleGAN)           |        Generative | Image synthesis         | Generator + discriminator          |
| U-Net / SegNet                   |       CNN variant | Image segmentation      | Encoder-decoder skip connections   |
| ResNet / DenseNet / EfficientNet |      CNN variants | Deep conv tasks         | Residual / efficient architectures |
| Graph Neural Networks (GCN, GAT) |            Neural | Graph-structured data   | Node/edge representation           |
| Capsule Networks                 |            Neural | Spatial hierarchy       | Less common in practice            |

---

## Recommender Systems

| Model                               |                  Type | Use case                          | Notes                    |
| ----------------------------------- | --------------------: | --------------------------------- | ------------------------ |
| User/Item KNN                       |          Neighborhood | Collaborative filtering           | Simple similarity-based  |
| Matrix Factorization (SVD, ALS)     |         Latent factor | Collaborative filtering           | Popular for large-scale  |
| Factorization Machines              | Linear + interactions | Sparse features / recommendations | Handles high-cardinality |
| BPR (Bayesian Personalised Ranking) |               Ranking | Implicit feedback                 | Pairwise ranking loss    |
| Content-based models                |         Feature match | Item content matching             | No cold-start for items  |
| Hybrid recommender                  |              Combined | Robust recommendations            | Mix CF + content         |
| Neural CF (NCF)                     |                Neural | Learned interactions              | Flexible deep models     |

---

## Anomaly / Outlier Detection

| Model                         |       Type | Use case              | Notes                      |
| ----------------------------- | ---------: | --------------------- | -------------------------- |
| One-Class SVM                 |     Kernel | Novelty detection     | Sensitive to kernel params |
| Isolation Forest              | Tree-based | Outlier detection     | Scales well                |
| Local Outlier Factor (LOF)    |    Density | Local anomaly scoring | Unsupervised               |
| EllipticEnvelope              | Parametric | Gaussian outliers     | Assumes normality          |
| Autoencoders (reconstruction) |     Neural | Anomaly if high error | For complex data           |

---

## Topic Modeling & NLP-specific

| Model                             |                 Type | Use case             | Notes                   |
| --------------------------------- | -------------------: | -------------------- | ----------------------- |
| LDA (Latent Dirichlet Allocation) |        Probabilistic | Topic discovery      | Bag-of-words assumption |
| NMF (for topics)                  | Matrix factorization | Topic decomposition  | Nonnegative parts       |
| Word2Vec / GloVe                  |           Embeddings | Word representations | Unsupervised embedding  |
| Transformer-based models          |           Pretrained | NLP fine-tuning      | BERT / GPT variants     |

---

## Reinforcement Learning (select algorithms)

| Model / Algo                 |            Type | Use case               | Notes                    |
| ---------------------------- | --------------: | ---------------------- | ------------------------ |
| Q-Learning / SARSA           |      Tabular RL | Small/finite MDPs      | Value-based              |
| DQN / Double DQN             | Deep Q-learning | High-dimensional state | Uses NN to approximate Q |
| Policy Gradients / REINFORCE |    Policy-based | Direct policy learning | High variance            |
| Actor-Critic / A2C / A3C     |          Hybrid | Continuous tasks       | Uses actor + critic      |
| PPO / TRPO / SAC / DDPG      |     Advanced RL | Continuous control     | State-of-the-art methods |
| Model-based RL               |  Model+planning | Sample efficient       | Learn transition model   |

---

## Other / Advanced / Miscellaneous

| Model                                    |              Type | Use case                     | Notes                            |
| ---------------------------------------- | ----------------: | ---------------------------- | -------------------------------- |
| AutoML (AutoSklearn, TPOT, H2O)          |            AutoML | Model search & tuning        | Automates pipeline search        |
| Meta-learning (MAML, Reptile)            | Few-shot learning | Fast adaptation              | Learns to learn                  |
| Transfer learning                        |       Pretraining | Use pretrained models        | Common in CV/NLP                 |
| Survival analysis (Cox PH, Kaplan-Meier) |     Time-to-event | Censoring problems           | Medical studies, reliability     |
| Factor analysis                          |     Latent factor | Dimensionality reduction     | Different assumptions than PCA   |
| Topic / semantic models                  |               NLP | Document semantics           | e.g., LDA, NMF                   |
| Sequence-to-sequence (seq2seq)           |            Neural | Translation, summarization   | Encoder-decoder models           |
| Bayesian Optimization                    |    Hyperparam opt | Expensive function opt       | Efficient hyperparameter search  |
| Sparse models (OMP, LARS)                | Sparse regression | High-dimensional sparse data | Feature selection oriented       |
| Streaming / online models                |   Online learning | Data streams                 | e.g., Hoeffding tree, SGD online |

---
