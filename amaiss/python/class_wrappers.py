# Class wrappers for amaiss Python bindings
# Provides Pythonic interface on top of SWIG-generated bindings

import numpy as np


def swig_ptr(arr):
    """Get SWIG-compatible pointer from numpy array."""
    return arr.__array_interface__["data"][0]


def handle_Index(the_class):
    """Add Pythonic methods to Index class."""

    def replacement_search(self, n, indptr, indices, values, k, params=None):
        """Search for k nearest neighbors.

        Parameters
        ----------
        n : int
            Number of query vectors
        indptr : array_like
            CSR indptr array (int32)
        indices : array_like
            CSR indices array (uint16)
        values : array_like
            CSR values array (float32)
        k : int
            Number of nearest neighbors to return
        params : SearchParameters, optional
            Search parameters (e.g., SeismicSearchParameters)

        Returns
        -------
        labels : ndarray
            Array of shape (n, k) with neighbor indices
        """
        # Ensure arrays are contiguous with correct dtypes for SWIG typemaps
        indptr = np.ascontiguousarray(indptr, dtype=np.int32)
        indices = np.ascontiguousarray(indices, dtype=np.uint16)
        values = np.ascontiguousarray(values, dtype=np.float32)

        if params is not None:
            return self.search_with_params(n, indptr, indices, values, k, params)
        else:
            return self.search_c(n, indptr, indices, values, k)

    the_class.search = replacement_search


def handle_all_classes(module):
    """Apply wrappers to all relevant classes in the module."""
    if hasattr(module, "Index"):
        handle_Index(module.Index)
