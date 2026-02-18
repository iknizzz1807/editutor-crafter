# AUDIT & FIX: knn

## CRITIQUE
- **Feature Normalization Missing**: The audit correctly identifies this as a critical gap. KNN is a distance-based algorithm where feature scale directly affects results. If one feature ranges [0, 1000] and another [0, 1], the first feature dominates all distance calculations. Normalization must be required before distance computation, not just mentioned as a pitfall.
- **KD-Tree Appears Only in M3 Deliverables**: The KD-tree is listed as a deliverable in M3 (Improvements) but is not in any AC. For datasets beyond a few thousand points, brute-force O(N) per query is impractical. However, for a beginner project, expecting a KD-tree implementation is arguably too advanced. The audit's suggestion to include it is valid for production use, but the fix should make it optional or explicitly scoped.
- **Cosine Similarity in M1 Without Context**: Cosine similarity is listed as a deliverable but not in any AC. For KNN classification with normalized features, cosine similarity is less common than Euclidean/Manhattan. It should either be properly motivated or removed.
- **'Negative under square root' Pitfall is Wrong**: Euclidean distance uses the sum of SQUARED differences, which is always non-negative. There is no negative-under-square-root issue. This pitfall is technically incorrect.
- **Confusion Matrix AC in M3 is Wrong for Multi-Class**: The AC states 'true positives, false positives, true negatives, and false negatives per class' which is only meaningful for binary classification. For multi-class KNN, the confusion matrix is NxN. The AC should specify this.
- **Missing Data Splitting**: Like the linear regression project, there's no requirement for train/test split methodology. The AC mentions 'held out' data implicitly but doesn't require proper splitting.
- **Weighted Voting Distance Edge Case**: Inverse distance weighting fails when distance is exactly zero (division by zero). This pitfall is not mentioned.

## FIXED YAML
```yaml
id: knn
name: "KNN Classifier from Scratch"
description: "Implement K-Nearest Neighbors classification with distance metrics, feature normalization, and cross-validation evaluation."
difficulty: beginner
estimated_hours: "8-14"
essence: >
  Instance-based classification through distance metric computation over
  normalized feature spaces, majority voting among k-nearest training samples,
  and hyperparameter optimization via cross-validation—implementing lazy
  learning without explicit model training.
why_important: >
  Building KNN from scratch teaches fundamental ML concepts—distance metrics,
  feature normalization, the bias-variance tradeoff, and model evaluation—
  without the complexity of gradient descent or parameter optimization,
  making it an ideal first ML algorithm to implement.
learning_outcomes:
  - Implement distance metrics (Euclidean, Manhattan) with NumPy vectorization
  - Apply feature normalization to ensure equal contribution of all features to distance
  - Build a KNN classifier with majority and weighted voting
  - Evaluate classifier performance with train/test split and cross-validation
  - Optimize k through systematic hyperparameter search
  - Understand the bias-variance tradeoff through k selection
  - Compute multi-class evaluation metrics (precision, recall, F1, confusion matrix)
skills:
  - Distance Metrics
  - Feature Normalization
  - Non-parametric Classification
  - Cross-validation
  - NumPy Array Operations
  - Classification Evaluation
  - Hyperparameter Tuning
tags:
  - ai-ml
  - algorithms
  - beginner-friendly
  - classification
  - distance-metrics
  - neighbors
  - python
architecture_doc: architecture-docs/knn/index.md
languages:
  recommended:
    - Python
  also_possible:
    - JavaScript
    - Julia
resources:
  - name: KNN Explained (scikit-learn)""
    url: https://scikit-learn.org/stable/modules/neighbors.html
    type: documentation
  - name: KNN from Scratch Tutorial""
    url: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
    type: tutorial
  - name: Curse of Dimensionality Explained""
    url: https://en.wikipedia.org/wiki/Curse_of_dimensionality
    type: article
prerequisites:
  - type: skill
    name: "Python and NumPy basics"
  - type: skill
    name: "Basic statistics (mean, standard deviation)"
milestones:
  - id: knn-m1
    name: "Distance Computation & Feature Normalization"
    description: >
      Implement distance metrics and feature normalization to prepare data
      for distance-based classification.
    acceptance_criteria:
      - "Euclidean distance correctly computes sqrt(Σ(a_i - b_i)²) between two feature vectors"
      - "Manhattan distance correctly computes Σ|a_i - b_i| between two feature vectors"
      - "Z-score normalization scales each feature to zero mean and unit variance using training set statistics"
      - "Normalization is fit on training data only; test data is transformed using training statistics (no data leakage)"
      - "Distance computation is vectorized using NumPy—computing distances from one query point to all N training points runs without Python-level loops"
      - "Unit tests verify distance functions against hand-calculated examples (e.g., distance([0,0], [3,4]) == 5.0 for Euclidean)"
    pitfalls:
      - "Computing normalization statistics on the full dataset (including test) causes data leakage and inflated accuracy"
      - "Features on different scales (e.g., age [0-100] vs. income [0-1000000]) make KNN meaningless without normalization"
      - "Integer input types cause integer division in distance computation—cast to float64"
      - "Distance matrix for large datasets can exhaust memory—compute on-demand per query instead of precomputing all pairs"
    concepts:
      - Distance metrics (L1, L2)
      - Feature normalization (z-score)
      - Data leakage prevention
      - Vectorized computation
    skills:
      - NumPy array manipulation
      - Mathematical distance calculations
      - Feature preprocessing
      - Unit testing
    deliverables:
      - "Euclidean distance function computing L2 norm between feature vectors"
      - "Manhattan distance function computing L1 norm between feature vectors"
      - "Z-score normalizer fit on training data and applied to both train and test sets"
      - "Vectorized distance-to-all function computing distances from query to all training points"
    estimated_hours: "2-3"

  - id: knn-m2
    name: "K-Nearest Neighbors Classifier"
    description: >
      Implement the full KNN classification algorithm with majority voting,
      tie-breaking, and train/test evaluation.
    acceptance_criteria:
      - "Data is split into training (80%) and test (20%) sets with reproducible random seed"
      - "find_k_nearest returns exactly K neighbors sorted by ascending distance from the query point"
      - "Majority voting assigns the class with the highest count among K neighbors"
      - "Tie-breaking strategy is defined and implemented (e.g., choose the class of the nearest neighbor among tied classes)"
      - "Classification accuracy on the test set is computed as correct_predictions / total_predictions"
      - "Classifier output matches sklearn KNeighborsClassifier on the same dataset and K value (within 1% accuracy)"
    pitfalls:
      - "K larger than training set size causes an error—validate K < len(training_data)"
      - "Ties in majority voting produce non-deterministic results without explicit tie-breaking"
      - "Using even K values increases tie probability—prefer odd K for binary classification"
      - "Forgetting to exclude the query point when it exists in the training set (leave-one-out scenario)"
    concepts:
      - Lazy learning (no training phase)
      - Majority voting
      - Classification accuracy
      - Train/test evaluation
    skills:
      - Algorithm implementation
      - Sorting and selection algorithms
      - Evaluation metric computation
      - Comparison with reference implementation
    deliverables:
      - "Train/test splitter with configurable ratio and random seed"
      - "K-nearest neighbor finder returning K closest training samples to a query point"
      - "Majority voting classifier with deterministic tie-breaking"
      - "Accuracy evaluator comparing predictions to ground truth labels on test set"
    estimated_hours: "2-4"

  - id: knn-m3
    name: "Improvements & Rigorous Evaluation"
    description: >
      Add weighted voting, cross-validation for K selection, and comprehensive
      multi-class evaluation metrics.
    acceptance_criteria:
      - "Weighted voting assigns each neighbor a weight of 1/distance; closer neighbors have proportionally more influence"
      - "Weighted voting handles zero-distance edge case (exact match) without division by zero—assign weight 1.0 and skip other neighbors, or add small epsilon"
      - "K-fold cross-validation (default K=5 folds) partitions training data, trains on K-1 folds, evaluates on the held-out fold, and averages accuracy across all folds"
      - "Optimal K is found by evaluating K values from 1 to sqrt(N) and selecting the K with highest mean cross-validated accuracy"
      - "Confusion matrix for C classes is a CxC matrix where entry (i,j) is the count of samples with true class i predicted as class j"
      - "Per-class precision, recall, and F1-score are computed from the confusion matrix"
      - "Plot of accuracy vs. K value visualizes the bias-variance tradeoff"
    pitfalls:
      - "K=1 overfits (high variance, low bias); large K underfits (low variance, high bias)—must demonstrate this tradeoff"
      - "Cross-validation with too few folds on small datasets produces high-variance estimates"
      - "Division by zero in precision/recall when a class has no predictions or no true samples—handle with zero-division fallback"
      - "Brute-force KNN is O(N*D) per query—for large datasets, mention that KD-trees or Ball trees would be needed (optional extension)"
    concepts:
      - Weighted voting (inverse distance)
      - K-fold cross-validation
      - Bias-variance tradeoff
      - Multi-class evaluation metrics
    skills:
      - Cross-validation implementation
      - Hyperparameter search
      - Multi-class metric computation
      - Data visualization
    deliverables:
      - "Inverse-distance weighted voting classifier with zero-distance handling"
      - "K-fold cross-validation module computing mean accuracy across folds"
      - "K optimization routine testing multiple K values and reporting best by CV accuracy"
      - "Multi-class confusion matrix, precision, recall, and F1-score computation"
      - "Accuracy vs. K plot demonstrating bias-variance tradeoff"
    estimated_hours: "3-5"
```