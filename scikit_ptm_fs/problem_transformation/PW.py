
import numpy as np
import itertools

from scipy.sparse import hstack, issparse, lil_matrix
import copy

from ..Base.ProblemTransformation import ProblemTransformationBase
from ..Base.base import MLSelectorBase


class PairwiseComparsion(ProblemTransformationBase):
    
    def __init__(self, selector=None, require_dense=None):
        super(PairwiseComparsion, self).__init__()
        
        
        self.selector = selector
        self.require_dense = require_dense
        self.copyable_attrs = [
            "selector",
            "require_dense",
    
            
        ]
    def transform(self, X, Y):
        label_count = Y.shape[1]
        X_pairwise_list = []
        y_trans_list = []
        
        # Iterate through all pairs of labels (i, j) where i < j
        for label1, label2 in itertools.combinations(range(label_count), 2):
            
            # Determine indices of samples with label1 and label2
            indices_label1 = (Y[:, label1] == 1)
            indices_label2 = (Y[:, label2] == 1)
            
            # Select samples where labels are either label1 or label2, but not both
            mask = (indices_label1 | indices_label2) & ~(indices_label1 & indices_label2)
            X_pairwise = X[mask]

            # Combine label1 and label2 into a single label array
            y_pairwise = np.where(indices_label1[mask], label1, label2)
            X_pairwise_list.append(X_pairwise)
            y_trans_list.append(y_pairwise)
            
            
            # Find the maximum number of samples among all pairs
        max_samples = max(len(pair) for pair in X_pairwise_list)
        # Fill arrays to have the same number of samples (if needed)
        
        for i in range(len(X_pairwise_list)):
            pair_samples = len(X_pairwise_list[i])
            if pair_samples < max_samples:
                missing_samples = max_samples - pair_samples
                X_pairwise_list[i] = np.vstack((X_pairwise_list[i], X_pairwise_list[i][:missing_samples]))
                y_trans_list[i] = np.hstack((y_trans_list[i], y_trans_list[i][:missing_samples]))
        # Convert lists to arrays
        self.x_trans = np.vstack(X_pairwise_list)
        self.y_trans = np.hstack(y_trans_list)
        
        return self.x_trans, self.y_trans
    
    def fit(self, X, y):
        """Fits selector to training data
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        Returns
        -------
        self
            fitted instance of self
        Notes
        -----
        .. note :: Input matrices are converted to sparse format internally if a numpy representation is passed
        """
        x_trans, y_trans=self.transform(X,y)
     
        self.selector.fit(self._ensure_input_format(x_trans), self._ensure_output_format(y_trans))
        

        return self
        
     

    def fit_transform(self, X, y):
        """
        Fit the selector and return the transformed features.
        """
        x_trans, y_trans = self.transform(X, y)  # Ensure x_trans and y_trans are created
        selected_features = self.selector.fit_transform(
        self._ensure_input_format(x_trans),
        self._ensure_output_format(y_trans))
        return selected_features

    
    def get_support(self, indices=True):
        indices = self.selector.get_support()
        return indices
    

