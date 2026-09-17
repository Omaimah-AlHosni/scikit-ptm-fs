import numpy as np

from scipy.sparse import hstack, issparse, lil_matrix

import copy
from ..Base.ProblemTransformation import ProblemTransformationBase
from ..Base.base import MLSelectorBase

class Entropy_Label_Assignment(ProblemTransformationBase):
    
    def __init__(self, selector=None, require_dense=None):
        super(Entropy_Label_Assignment, self).__init__(selector, require_dense)
        self.selector = selector
    
    def calculate_label_entropy(self, y):
        """
        Binary entropy of each label's marginal distribution across the
        WHOLE dataset: H(y_j) = -p_j*log(p_j) - (1-p_j)*log(1-p_j), where
        p_j is the fraction of samples with label j active. A label near a
        50/50 split carries the most information (highest entropy); a
        heavily skewed label carries the least.
        """
        p = np.clip(y.mean(axis=0), 1e-15, 1 - 1e-15)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return entropy
    
    def ELA_label_assignment(self, X, y):
        n_samples, n_classes = y.shape
        label_entropy = self.calculate_label_entropy(y)

        # samples with no active label get a dedicated "no-label" class
        # instead of silently defaulting to class 0
        no_label_class = n_classes
        transformed_labels = np.full(n_samples, no_label_class, dtype=int)

        for i in range(n_samples):
            active = np.where(y[i] == 1)[0]
            if len(active) == 0:
                continue
            # among this sample's active labels, assign the one with the
            # highest global entropy (most informative label)
            transformed_labels[i] = active[np.argmax(label_entropy[active])]

        return transformed_labels
    
    def transform(self, y):
        transformed_labels = self.ELA_label_assignment(None, y)  # Pass None for X as it's not used
        return transformed_labels
    
    def fit(self, X, y):
        X = self._ensure_input_format(X, sparse_format="csr", enforce_sparse=True)

        self.selector.fit(self._ensure_input_format(X), self.transform(y))

        return self
     
    
    def fit_transform(self, X, y):
        selection = self.selector.fit_transform(X, self.transform(y))
        
        return selection
    
    def get_support(self, indices=True):
        indices = self.selector.get_support()
        return indices
