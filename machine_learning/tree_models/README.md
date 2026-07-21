# Tree-Based Models

## 0-build.py

### Decision Tree Classifier

Write a function `build_decision_tree(min_samples_leaf, min_samples_split, random_state)` to create a decision tree classifier using Scikit-learn.

The decision tree uses the Gini impurity measure to evaluate the quality of splits.
No maximum depth is set, allowing the tree to grow until all leaves are pure or other stopping criteria are met.

Arguments:
- min_samples_leaf: Minimum number of samples required to be at a leaf node.
- min_samples_split: Minimum number of samples required to split an internal node.
- random_state: Seed used by the random number generator for reproducibility.

Returns:
- model: A Scikit-learn DecisionTreeClassifier instance.

_Required import: `from sklearn import tree`._