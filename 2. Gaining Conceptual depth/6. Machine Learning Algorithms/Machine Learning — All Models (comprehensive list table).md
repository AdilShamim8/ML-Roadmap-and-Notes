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
