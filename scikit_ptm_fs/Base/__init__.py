"""
The :mod:`Scikit-PTM-FS.base` module implements base for multi-label feature selection.

Two base classes are in use currently in Scikit-PTM-FS:

- :class:`MLSelectorBase` - a generic base class for multi-label feature selection
- :class:`ProblemTransformationBase` - the base class for problem transformation and ensemble approaches that handles a base selector
"""

from .base import MLSelectorBase
from .ProblemTransformation import ProblemTransformationBase

__all__ = ["MLSelectorBase", "ProblemTransformationBase"]