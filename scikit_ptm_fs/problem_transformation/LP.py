from scipy import sparse

from ..Base.ProblemTransformation import ProblemTransformationBase

from ..Base.base import MLSelectorBase

import copy
import numpy as np
from scipy import sparse


class LabelPowerset(ProblemTransformationBase):
    
    
    def __init__(self, selector=None, require_dense=None):
        super(LabelPowerset, self).__init__(selector, require_dense)
        self.selector=selector
        
        
   
        self._clean()

    def _clean(self):
        """Reset classifier internals before refitting"""
        self.unique_combinations_ = {}
        self.reverse_combinations_ = []
        self._label_count = None
       
    def transform(self, y):
        """Transform multi-label output space to multi-class
        Transforms a mutli-label problem into a single-label multi-class
        problem where each label combination is a separate class.
        Parameters
        -----------
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        Returns
        -------
        numpy.ndarray of `{0, ... , n_classes-1}`, shape=(n_samples,)
            a multi-class output space vector
        """

        y = self._ensure_output_format(y, sparse_format="lil", enforce_sparse=True)

        self._clean()
        self._label_count = y.shape[1]

        last_id = 0
        train_vector = []
        for labels_applied in y.rows:
            label_string = ",".join(map(str, labels_applied))

            if label_string not in self.unique_combinations_:
                self.unique_combinations_[label_string] = last_id
                self.reverse_combinations_.append(labels_applied)
                last_id += 1

            train_vector.append(self.unique_combinations_[label_string])

        return np.array(train_vector)


    def fit(self, X, y):
        """Fits classifier to training data
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
  
        X = self._ensure_input_format(X, sparse_format="csr", enforce_sparse=True)

        self.selector.fit(self._ensure_input_format(X), self.transform(y))

        return self

    def fit_transform(self, X,y):
        """Predict labels for X
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
            
        """
  
        self.selector.fit(self._ensure_input_format(X), self.transform(y))

        # this will be an np.array of integers representing classes
        lp_prediction = self.selector.fit_transform(self._ensure_input_format(X), self.transform(y))

        return lp_prediction



    def inverse_transform(self, y):
        """Transforms multi-class assignment to multi-label
        Transforms a mutli-label problem into a single-label multi-class
        problem where each label combination is a separate class.
        Parameters
        -----------
        y : numpy.ndarray of `{0, ... , n_classes-1}`, shape=(n_samples,)
            binary indicator matrix with label assignments
        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """
        n_samples = len(y)
        result = sparse.lil_matrix((n_samples, self._label_count), dtype="i8")
        for row in range(n_samples):
            assignment = y[row]
            result[row, self.reverse_combinations_[assignment]] = 1

        return result
    
    def get_support(self, indices=True):
        
        indices = self.selector.get_support()
        return indices