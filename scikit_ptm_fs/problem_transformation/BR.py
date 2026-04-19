import copy
import numpy as np
from ..Base.ProblemTransformation import ProblemTransformationBase
from scipy.sparse import hstack, issparse, lil_matrix

class BinaryRelevance(ProblemTransformationBase):
    
    def __init__(self, selector=None, require_dense=None):
        super(BinaryRelevance, self).__init__(selector=selector, require_dense=require_dense)
        self.selector = selector

    def _generate_partition(self, X, y):
        """
        Partitions the label space into individual labels (singletons).
        """
        self.partition_ = list(range(y.shape[1]))
        self.model_count_ = y.shape[1]

    def fit(self, X, y):
        """
        Fits the selector independently for each label.
        """
        X = self._ensure_input_format(X, sparse_format="csr", enforce_sparse=True)
        y = self._ensure_output_format(y, sparse_format="csc", enforce_sparse=True)

        self.selectors_ = []
        self._generate_partition(X, y)
        self._label_count = y.shape[1]

        for i in range(self.model_count_):
            selector = copy.deepcopy(self.selector)
            y_subset = self._generate_data_subset(y, self.partition_[i], axis=1)
            if issparse(y_subset) and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())
            selector.fit(
                self._ensure_input_format(X),
                self._ensure_output_format(y_subset)
            )
            self.selectors_.append(selector)

        return self

    def fit_transform(self, X, y):
        """
        Fits and transforms the data for each label.
        Returns a list of transformed feature subsets (one per label).
        """
        self.fit(X, y)

        selected_features_per_label = []

        for i in range(self.model_count_):
            transformed = self.selectors_[i].transform(self._ensure_input_format(X))
            selected_features_per_label.append(transformed)

        return selected_features_per_label

    def get_support(self, indices=True, merge=False):
        """
        Returns the selected feature indices or masks per label.
        
        Parameters
        ----------
        indices : bool
            If True, returns indices; otherwise, returns a boolean mask.
        merge : bool
            If True, merges the selected supports across all labels.

        Returns
        -------
        list of supports per label or a merged support (union).
        """
        support_all = []
        for label in range(self.model_count_):
            support = self.selectors_[label].get_support(indices=indices)
            support_all.append(support)

        if merge:
            if indices:
                # Merge indices by union and return sorted list
                return sorted(set(np.concatenate(support_all)))
            else:
                # Merge masks using logical OR
                return np.any(np.vstack(support_all), axis=0)

        return support_all

  