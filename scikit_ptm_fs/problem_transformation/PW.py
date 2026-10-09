"""Checks for the corrected PairwiseComparsion.

Run from this folder:  python test_pw.py
  1. Two labels: same selected features as scikit-ptm-fs 0.12.
  2. Several labels: one selector per pair, equal to fitting each pair by hand.
  3. Sparse X, a skipped pair, cloning and the API.
"""
import importlib, itertools, sys
import numpy as np
from scipy import sparse
from sklearn.feature_selection import SelectFpr, SelectKBest, f_classif


def load(path):
    for m in [k for k in sys.modules if k.startswith("scikit_ptm_fs")]:
        del sys.modules[m]
    sys.path.insert(0, path)
    mod = importlib.import_module("scikit_ptm_fs.problem_transformation.PW")
    sys.path.pop(0)
    return mod.PairwiseComparsion, mod.__file__


def make(L, n=400, p=60, r=0.45, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, p))
    C = rng.uniform(-0.5, 0.5, size=(L, 2))
    Y = (np.linalg.norm(X[:, None, :2] - C[None], axis=2) <= r).astype(int)
    return X, Y


OLD = sys.argv[1] if len(sys.argv) > 1 else None      # site-packages of 0.12
NewPW, f_new = load("pkg")
print("new class from:", f_new)

# 1 ---------------------------------------------------------------- two labels
X, Y = make(2, seed=3)
new = NewPW(selector=SelectFpr(f_classif, alpha=0.01)).fit(X, Y)
print("two labels: pairs", new.pairs_, "rows used", new.n_samples_,
      "selected", int(new.get_support().sum()))
if OLD:
    OldPW, f_old = load(OLD)
    old = OldPW(selector=SelectFpr(f_classif, alpha=0.01),
                require_dense=[True, True]).fit(X, Y)   # 0.12 fails without it
    same = np.array_equal(np.asarray(old.get_support(), bool), new.get_support())
    print("  identical to version 0.12 on two labels:", same)
    assert same
    NewPW, _ = load("pkg")

# 2 ------------------------------------------------------------ several labels
for L in (3, 5, 8):
    X, Y = make(L, seed=L)
    pw = NewPW(selector=SelectFpr(f_classif, alpha=0.01)).fit(X, Y)
    n_pairs = L * (L - 1) // 2
    assert len(pw.pairs_) + len(pw.skipped_pairs_) == n_pairs
    assert len(pw.selectors_) == len(pw.pairs_)
    union = np.zeros(X.shape[1], bool)
    votes = np.zeros(X.shape[1], int)
    for (a, b), fitted, n_used in zip(pw.pairs_, pw.selectors_, pw.n_samples_):
        mask = Y[:, a] != Y[:, b]
        assert n_used == mask.sum()
        y = np.where(Y[mask, a] == 1, a, b)
        assert set(np.unique(y)) == {a, b}
        ref = SelectFpr(f_classif, alpha=0.01).fit(X[mask], y).get_support()
        assert np.array_equal(ref, fitted.get_support())
        union |= ref
        votes += ref
    assert np.array_equal(union, pw.get_support())
    assert np.array_equal(np.flatnonzero(union), pw.get_support(indices=True))
    assert np.array_equal(votes, pw.get_vote_counts())
    assert len(pw.get_support(merge=False)) == len(pw.pairs_)
    assert pw.fit_transform(X, Y).shape == (len(X), int(union.sum()))
    assert all(len(t[1]) == n for t, n in zip(pw.transform(X, Y), pw.n_samples_))
    print(f"{L} labels: {len(pw.pairs_)} of {n_pairs} pairs fitted, "
          f"{len(pw.skipped_pairs_)} skipped, union selects {int(union.sum())}, "
          f"relevant features 0 and 1 voted {votes[:2].tolist()} times")

# 3 --------------------------------------------- sparse X, skipped pair, clone
X, Y = make(4, seed=1)
Y[:, 3] = Y[:, 2]                       # labels 2 and 3 identical -> no task
dense = NewPW(selector=SelectKBest(f_classif, k=5)).fit(X, Y)
assert (2, 3) in dense.skipped_pairs_ and (2, 3) not in dense.pairs_
sp_ = NewPW(selector=SelectKBest(f_classif, k=5)).fit(sparse.csr_matrix(X),
                                                      sparse.csr_matrix(Y))
assert np.array_equal(dense.get_support(), sp_.get_support())
from sklearn.base import clone
c = clone(NewPW(selector=SelectKBest(f_classif, k=5), min_class_size=3))
assert c.min_class_size == 3 and c.selector.k == 5
print("sparse input, skipped pair and cloning: ok")
print("ALL CHECKS PASSED")
