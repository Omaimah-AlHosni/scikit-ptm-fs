
import numpy as np

from scipy.sparse import hstack, issparse, lil_matrix

import copy
from ..Base.ProblemTransformation import ProblemTransformationBase
from ..Base.base import MLSelectorBase

class Entropy_Label_Assignment(ProblemTransformationBase):
    
    def __init__(self, selector=None, require_dense=None):
        super(Entropy_Label_Assignment, self).__init__(selector, require_dense)
        self.selector = selector
    
    def calculate_entropy(self, probabilities):
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-15))
        return entropy
    
    def g(self, entropy):
        return np.exp(-entropy)
    
    def ELA_label_assignment(self, X, y):
        n_samples, n_classes = y.shape
        transformed_labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            class_probabilities = np.zeros(n_classes)
            for j in range(n_classes):
                entropy_for_class = self.calculate_entropy(y[i, j])
                class_probabilities[j] = self.g(entropy_for_class)
            class_probabilities /= np.sum(class_probabilities)
            transformed_labels[i] = np.argmax(class_probabilities)
        
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

