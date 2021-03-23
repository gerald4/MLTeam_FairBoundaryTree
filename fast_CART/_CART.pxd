# cython: language_level=3
# cython: cdivision=False
# cython: boundscheck=True
# cython: wraparound=True

# More info in _CART.pyx

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_intp SIZE_t

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t parent                        # id of the parent of the node
    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    SIZE_t value                         # Label of the node

cdef class Tree:

    # Input/Output layout
    cdef public SIZE_t min_samples
    cdef public SIZE_t max_leaves
    cdef public double _lambda
    cdef public str impurity_type

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes

    # Methods
    cdef void _add_node(self, SIZE_t parent, bint is_left, bint is_leaf, SIZE_t feature, 
                        double threshold, double impurity, SIZE_t value) nogil
    cdef void _update_node(self, SIZE_t node_id, SIZE_t feature, double threshold) nogil
    cdef double entropy(self, np.ndarray[SIZE_t, ndim=1] freq)
    cdef double gini(self, np.ndarray[SIZE_t, ndim=1] freq)
    cdef void _resize_c(self) nogil
    cpdef void print_struct(self)
    cpdef void fit(self, np.ndarray[DOUBLE_t, ndim=2] X, np.ndarray[SIZE_t, ndim=1] y)
    cdef np.ndarray _get_node_ndarray(self)