# Scikit-PTM-FS

[![PyPI version](https://badge.fury.io/py/scikit-ptm-fs.svg)](https://pypi.org/project/scikit-ptm-fs/)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Scikit-PTM-FS** is a Python library that enables seamless integration of `scikit-learn` feature selection methods with multi-label classification via problem transformation methods (PTMs). The framework enables transformation-aware feature selection without requiring classification stages, allowing researchers to analyze the behavior of FS methods across various PTMs.

---

## 📦 Installation

```bash
pip install scikit-ptm-fs
```

---

## 🚀 Usage Example

Here is a basic example of how to apply a feature selection method within a problem transformation method using `Scikit-PTM-FS`:

```python
from scikit_ptm_fs.problem_transformation import LabelPowerset
from sklearn.feature_selection import SelectKBest, f_classif

# Example feature and label matrices (replace with your own)
# X = ...  # shape (n_samples, n_features)
# y = ...  # shape (n_samples, n_labels)

# Step 1: Define a scikit-learn selector
selector = SelectKBest(score_func=f_classif, k=5)

# Step 2: Wrap it using the LabelPowerset PTM
lp_selector = LabelPowerset(selector=selector, require_dense=[False, True])

# Step 3: Apply feature selection
X_selected = lp_selector.fit_transform(X, y)

# Step 4: Get selected feature indices
selected_indices = lp_selector.get_support()
print("Selected features:", selected_indices)

indices = np.where(lp_selector.get_support())[0]
print("The Number of Selected Features:", len(indices))
```

> 💡 You can use other PTMs like `BinaryRelevance`, `PPT`, or `PairwiseComparison` similarly.

---

## 🧠 Supported Problem Transformation Methods (PTMs)

- **Binary Relevance (BR)** — One-vs-All decomposition.
- **Label Powerset (LP)** — Unique labelset encoding.
- **Pairwise Comparison (PW)** — One-vs-One label pair modeling.
- **Pruned Problem Transformation (PPT)** — Labelset pruning before transformation.
- **Entropy-based Label Assignment (ELA)** — Copy-based transformation for NLP.

---

## 🔗 Project Links

- 📦 PyPI: [https://pypi.org/project/scikit-ptm-fs](https://pypi.org/project/scikit-ptm-fs)
- 🛠️ Source Code: [https://github.com/Omaimah-AlHosni/scikit-ptm-fs](https://github.com/Omaimah-AlHosni/scikit-ptm-fs)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
