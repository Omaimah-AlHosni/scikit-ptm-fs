import pandas as pd
import numpy as np
from ..problem_transformation import LabelPowerset
from ..Base.ProblemTransformation import ProblemTransformationBase
from scipy.sparse import csr_matrix


class Pruned_Problem_Transformation(ProblemTransformationBase):
    def __init__(self, base_selector=None, require_dense=None, p=None, b=None, strategy="A"):
        super(Pruned_Problem_Transformation, self).__init__()
        self.base_selector = base_selector
        self.require_dense = require_dense
        self.p = p
        self.b = b
        self.strategy = strategy
        self.selector_ = None  # ✅ FIXED: removed the list assignment

        self.copyable_attrs = [
            "base_selector", "require_dense", "p", "strategy", "b",
            "X_Startegy_A", "y_Startegy_A", "X_Startegy_B", "y_Startegy_B",
            "selected_instances", "common_labelsets", "labelsets", "selected_instances_b"
        ]

    def pruned_labelsets(self, X, y, p=None, b=None):
        if self.strategy not in ['A', 'B']:
            raise ValueError("Invalid strategy. Choose either 'A' or 'B'.")
        if self.b < 0:
            raise ValueError("The parameter b must be >= 0")
        if self.p < 1:
            raise ValueError("The pruning value must be > 0")

        X = pd.DataFrame(X)
        X.columns = ["x" + str(col) for col in X.columns]

        y = y.astype('int')
        y = pd.DataFrame(y)
        y.columns = ["y" + str(col) for col in y.columns]

        df = pd.concat([X, y], axis=1)
        df = df.reset_index().set_index('index')

        label_col_names = list(y.columns.values)
        dfX = df.rename(columns={col: "labelset" if col in label_col_names else col for col in df.columns})

        labelset_col_index = dfX.columns.tolist().index("labelset")
        d = y.shape[1]
        
        
        labels = dfX.iloc[:, labelset_col_index:labelset_col_index + d].astype(str).agg(''.join, axis=1)

        dfX = dfX.drop(dfX.columns[-d:], axis=1)
        dfX['labelsets'] = labels

        common_label = dfX['labelsets'].value_counts()
        self.common_labelsets = common_label[common_label > self.p].index.tolist()

        if len(self.common_labelsets) == 0:
            raise ValueError(f"All labelsets appear less than {self.p} time(s) in the training data.")

        self.selected_instances = np.isin(dfX['labelsets'], self.common_labelsets)
        self.labelsets = [[int(c) for c in str(x) if c.isdigit()] for x in self.common_labelsets]
        self.labelsets.sort(key=lambda x: sum(x), reverse=True)

        labelset = []
        if self.strategy == "B":
            labelset = [x for x in self.labelsets if sum(x) > self.b]
            if not labelset:
                raise ValueError("There are no labelsets greater than the b value")
            labelsets_strings = [''.join(str(bit) for bit in row) for row in labelset]
            self.selected_instances_b = np.isin(dfX['labelsets'], labelsets_strings)
        else:
            self.selected_instances_b = []

        New_X = dfX.drop(columns=['labelsets'])
        self.X_Startegy_A = np.array(New_X[self.selected_instances])
        self.y_Startegy_A = np.array(y[self.selected_instances])
        self.X_Startegy_B = np.array(New_X[self.selected_instances_b])
        self.y_Startegy_B = np.array(y[self.selected_instances_b])

        return self.X_Startegy_A, self.y_Startegy_A, self.X_Startegy_B, self.y_Startegy_B

    def fit(self, X, y):
        self.X_Startegy_A, self.y_Startegy_A, self.X_Startegy_B, self.y_Startegy_B = self.pruned_labelsets(X, y, self.p, self.b)

        if self.strategy == "B":
            self.selector_ = LabelPowerset(selector=self.base_selector, require_dense=self.require_dense)
            return self.selector_.fit(self.X_Startegy_B, self.y_Startegy_B)
        elif self.strategy == "A":
            self.selector_ = LabelPowerset(selector=self.base_selector, require_dense=self.require_dense)
            return self.selector_.fit(self.X_Startegy_A, self.y_Startegy_A)

    def fit_transform(self, X, y):
        self.fit(X, y)  # ✅ ensures selector_ is initialized

        if self.strategy == "B":
            return self.selector_.fit_transform(self.X_Startegy_B, self.y_Startegy_B)
        elif self.strategy == "A":
            return self.selector_.fit_transform(self.X_Startegy_A, self.y_Startegy_A)

    def transform(self, X, y):
        if self.strategy == "B":
            return self.selector_.transform(self.y_Startegy_B), self.X_Startegy_B
        elif self.strategy == "A":
            return self.selector_.transform(self.y_Startegy_A), self.X_Startegy_A

    def get_support(self, indices=True):
        return self.selector_.get_support(indices=indices)
