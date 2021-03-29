# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True


# Code inspired by the scikit-learn implementation
# https://github.com/scikit-learn/scikit-learn/tree/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/tree
# 
# Implementation of the CART algorithm
#
# Authors: Géraldin Nanfack
#          Valentin Delchevalerie
# Creation: 18-03-2021
# Last modification: 29-03-2021


from copy import deepcopy

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdio cimport printf
from libc.math cimport log2, pow, sqrt, abs
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy

import numpy as np
cimport numpy as np
np.import_array()

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

cdef Node dummy;
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)


cdef class Tree:
    """
    The tree class
    """


    def __cinit__(self, SIZE_t min_samples, SIZE_t max_leaves, double _lambda, 
                  str impurity_type='gini', SIZE_t axis=0):
        """Initialization."""
        self.min_samples = min_samples
        self.max_leaves = max_leaves
        self._lambda = _lambda
        self.impurity_type = impurity_type
        self.axis = axis

        self.capacity = 2*max_leaves - 1

        # Initialize self.nodes
        self.nodes = <Node*>malloc(sizeof(Node)*self.capacity)
        self.node_count = 0


    def __dealloc__(self):
        """Destructor."""
        free(self.nodes)


    # __reduce__, __getstate__ and __setstate__ are needed to perform deepcopy of the tree
    def __reduce__(self):
        """Reduce re-implementation, for pickling."""

        return (Tree, 
                (self.min_samples, self.max_leaves, self._lambda, self.impurity_type),
                self.__getstate__())


    def __getstate__(self):
        """Getstate re-implementation, for pickling."""

        d = {}
        d["capacity"] = self.capacity
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        return d


    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""

        self.capacity = d["capacity"]
        self.node_count = d["node_count"]

        node_ndarray = d["nodes"]
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))


    cdef void _resize_c(self) nogil:
        """Resize the nodes internal array."""

        # Resize
        if (self.node_count >= self.capacity):
            self.capacity = self.capacity * 2
            self.nodes = <Node*>realloc(&self.nodes[0], sizeof(self.nodes[0])*self.capacity)


    cpdef void print_struct(self):
        """Print the tree structure."""

        cdef:
            SIZE_t i
            Node* node

        for i in range(self.node_count):
            node = &self.nodes[i]
            print("\nNode", i)
            print("-------")
            print('\t * parent -->', node.parent)
            print('\t * left_child -->', node.left_child)
            print('\t * right_child -->', node.right_child)
            print('\t * feature -->', node.feature)
            print('\t * threshold -->', node.threshold)
            print('\t * impurity -->', node.impurity)
            print('\t * value -->', node.value)
            print()


    cdef void _add_node(self, SIZE_t parent, bint is_left, bint is_leaf, SIZE_t feature, 
                        double threshold, double impurity, SIZE_t value) nogil:
        """Add a node to the tree."""

        cdef:
            SIZE_t node_id = self.node_count
            Node* node = &self.nodes[node_id]

        # Resize the Tree if it needs more memory
        self._resize_c()

        # Complete the Node meta-data
        node.parent = parent
        node.impurity = impurity
        node.value = value
        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = INFINITY
        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        # Complete the parent Node meta-data
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id


    cdef void _update_node(self, SIZE_t node_id, SIZE_t feature, double threshold) nogil:
        """Update the node meta-data in the tree."""

        cdef:
            Node* node = &self.nodes[node_id]

        # Modify the Node meta-data
        node.feature = feature
        node.threshold = threshold
        

    cdef double entropy(self, double[:] freq, int n) nogil:
        """
        Compute the entropy 
        """

        cdef:
            int i
            double probs = 0.
            double sum = 0.

        for i in range(n):
            sum += freq[i]

        for i in range(n):
            if freq[i] != 0:
                probs -= freq[i] * log2(freq[i] / sum)

        return probs / sum


    cdef double gini(self, double[:] freq, int n) nogil:
        """
        Compute the gini
        """

        cdef:
            int i
            double probs = 0.
            double sum = 0.

        for i in range(n):
            sum += freq[i]

        for i in range(n):
            probs += 1.- pow(freq[i] / sum, 2)

        return probs


    cpdef void fit(self, np.ndarray[DOUBLE_t, ndim=2] X, np.ndarray[SIZE_t, ndim=1] y):
        """
        Build the tree.

        X should be of shape [n_samples, n_features] of type double
        y should be of shape [n_samples,] of type intp
        """

        cdef:
            # Input
            SIZE_t n_data = X.shape[0]
            SIZE_t n_features = X.shape[1]

            # Working
            SIZE_t node_id
            SIZE_t feature
            SIZE_t value, value_1, value_2
            SIZE_t end
            SIZE_t p
            double impurity, impurity_1, impurity_2
            double gain, weight, unfairness
            Node* node
            Tree _tree
            double[:] stats
            double[:] X_feat
            np.ndarray[SIZE_t, ndim=1] y_feat
            np.ndarray[SIZE_t, ndim=1] temp
            np.ndarray[double, ndim=2] X_in_node

            # Save
            list splittable_nodes = []
            list leaf_nodes = []
            SIZE_t best_node
            SIZE_t _best_feature, best_feature
            SIZE_t _best_value_1, _best_value_2, best_value_1, best_value_2
            double _best_impurity_1, _best_impurity_2, best_impurity_1, best_impurity_2
            double _best_threshold, best_threshold
            double _best_gain, best_gain
            np.ndarray[SIZE_t, ndim=1] in_node = np.zeros((n_data), dtype=np.intp)

        # Building the root node 
        stats = np.bincount(y).astype(np.double)
        value = np.argmax(stats)
        if self.impurity_type == 'entropy':
            impurity = self.entropy(stats, len(stats))
        else:
            impurity = self.gini(stats, len(stats))

        # Pattern to add a node
        # --> self_add_node(parent, is_left, is_leaf, feature, threshold, impurity, value)
        self._add_node(_TREE_UNDEFINED, _TREE_UNDEFINED, 1, _TREE_UNDEFINED, INFINITY, impurity, value)
        leaf_nodes.append(0)

        # If there are not enough data w.r.t. self.min_samples
        if n_data < self.min_samples:
            # root node will stay a leaf
            return
        # Else, node 0 (root node) is splittable
        else:
            if impurity != 0.:
                splittable_nodes.append(0)

        # Splitting until there is no other splitting possibilities or a limit is reached 
        while ( self.node_count < (2*self.max_leaves-1) ) and len(splittable_nodes) != 0:
            # Considering splitting the node 'node_id'
            best_gain = 0.
 
            for node_id in splittable_nodes:
                # Saving the best results while splitting node 'node_id'
                _best_gain = 0.

                # Loading the node
                node = &self.nodes[node_id]

                # Considering splitting w.r.t. each feature
                X_in_node = X[np.where(in_node[:] == node_id)[0],:]
                end = X_in_node.shape[0]

                for feature in range(n_features):
                    temp = np.argsort(X_in_node[:,feature])
                    X_feat = X_in_node[temp, feature]
                    y_feat = y[np.where(in_node[:] == node_id)[0]][temp]

                    # No variance on this feature
                    if X_feat[end-1] <= X_feat[0] + EPSILON:
                        continue

                    #Loop to search for threshold
                    p = 0
                    while p < end:
                        while (p + 1 < end) and (X_feat[p + 1] <= X_feat[p] + EPSILON):
                            p += 1
                        p += 1

                        if p < end: 
                            if (p < self.min_samples) or (end - p < self.min_samples):
                                continue

                            # Left child node
                            stats = np.bincount(y_feat[0:p]).astype(np.double)
                            value_1 = np.argmax(stats)
                            if self.impurity_type == 'entropy':
                                impurity_1 = self.entropy(stats, len(stats))
                            else:
                                impurity_1 = self.gini(stats, len(stats))

                            # Right child node
                            stats = np.bincount(y_feat[p::]).astype(np.double)
                            value_2 = np.argmax(stats)
                            if self.impurity_type == 'entropy':
                                impurity_2 = self.entropy(stats, len(stats))
                            else:
                                impurity_2 = self.gini(stats, len(stats))

                            # Computing how good this split is
                            weight = <double>p / <double>end
                            unfairness = self.get_DI_corr(X, [x for x in leaf_nodes if x != node_id], node_id, 
                                                     X_feat[p-1], feature, value_1, value_2)
                            #print(unfairness, self.get_DI_corr(X, [x for x in leaf_nodes if x != node_id], node_id, 
                            #                                   X_feat[p-1], feature, value_1, value_2))
                            gain = node.impurity - weight*impurity_1 - (1-weight)*impurity_2 - self._lambda*unfairness

                            if gain > _best_gain:
                                _best_gain = gain

                                _best_threshold = (X_feat[p-1] + X_feat[p])/2.0
                                _best_feature = feature

                                _best_impurity_1 = impurity_1
                                _best_impurity_2 = impurity_2
                                _best_value_1 = value_1
                                _best_value_2 = value_2

                if _best_gain > best_gain:
                    best_gain = _best_gain

                    best_node = node_id
                    best_threshold = _best_threshold
                    best_feature = _best_feature

                    best_impurity_1 = _best_impurity_1
                    best_impurity_2 = _best_impurity_2
                    best_value_1 = _best_value_1
                    best_value_2 = _best_value_2
                elif _best_gain <= 0:
                    # Splitting this node is useless
                    splittable_nodes.remove(node_id)

            # Update the tree by performing the best split, if it exists
            if not best_gain > 0:
                break

            node = &self.nodes[best_node]
            self._update_node(best_node, best_feature, best_threshold)
            leaf_nodes.remove(best_node)

            set1 = np.where((X[:,best_feature] <= best_threshold) & (in_node[:] == best_node))[0]
            set2 = np.where((X[:,best_feature] >  best_threshold) & (in_node[:] == best_node))[0]

            # Left child node
            self._add_node(best_node, 1, 1, _TREE_UNDEFINED, INFINITY, best_impurity_1, best_value_1)
            leaf_nodes.append(node.left_child)

            in_node[set1] = node.left_child
            if set1.shape[0] >= self.min_samples and best_impurity_1 != 0.:
                splittable_nodes.append(node.left_child)
            
            # Right child node
            self._add_node(best_node, 0, 1, _TREE_UNDEFINED, INFINITY, best_impurity_2, best_value_2)
            leaf_nodes.append(node.right_child)

            in_node[set2] = node.right_child
            if set2.shape[0] >= self.min_samples and best_impurity_2 != 0.:
                splittable_nodes.append(node.right_child)

            # node 'best_node' is not anymore splittable
            splittable_nodes.remove(best_node)


    cpdef SIZE_t[:] predict(self, double[:,:] X):
        """Return the prediction of the tree."""

        cdef:
            SIZE_t n_data = X.shape[0]
            int i
            Node* node
            SIZE_t[:] predictions = np.empty((n_data,), dtype=np.intp)

        for i in range(n_data):
            node = &self.nodes[0]
            while node.feature != -2:
                if X[i,node.feature] <= node.threshold:
                    node = &self.nodes[node.left_child]
                else:
                    node = &self.nodes[node.right_child]

            predictions[i] = node.value

        return predictions[:]


    cdef double get_DI(self, double[:,:] X, list leaves, SIZE_t split, double threshold, 
                       SIZE_t feature, SIZE_t value_1, SIZE_t value_2):

        cdef:
            SIZE_t n_data = X.shape[0]
            SIZE_t d = X.shape[1]
            SIZE_t node_id, curr_node
            SIZE_t pred
            int i, j
            double z_bar = 0.
            double DI = 0.
            double dist, curr_dist
            double[:] proj = np.empty((d,), dtype=np.double)

        for i in range(n_data):
            z_bar += X[i,self.axis]
        z_bar = z_bar/n_data

        for i in range(n_data):

            # Predict the label of X[i,:]
            node_id = 0
            node = &self.nodes[node_id]
            while node.feature != -2:
                if node_id != split:
                    if X[i,node.feature] <= node.threshold:
                        node_id = node.left_child
                        node = &self.nodes[node_id]
                    else:
                        node_id = node.right_child
                        node = &self.nodes[node.right_child]
                else:
                    if X[i,feature] <= threshold:
                        pred = value_1
                    else:
                        pred = value_2
                    break

            if node.feature == -2:
                pred = node.value

            # Go through each of the leaf nodes, except the new ones
            dist = 1000000.
            for node_id in leaves:
                node = &self.nodes[node_id]
                if node.value == pred:
                    continue
                
                for j in range(d):
                    proj[j] = X[i,j]

                curr_node = node_id
                while node.parent != _TREE_UNDEFINED:
                    node = &self.nodes[node.parent]
                    if node.left_child == curr_node:
                        if proj[node.feature] > node.threshold:
                            proj[node.feature] = node.threshold
                    elif proj[node.feature] <= node.threshold:
                        proj[node.feature] = node.threshold
                    curr_node = node.parent

                curr_dist = 0.0
                for j in range(d):
                    curr_dist += pow(proj[j] - X[i,j],2)
                curr_dist = sqrt(curr_dist)

                if curr_dist < dist:
                    dist = curr_dist

            # Go through the left new leaf if needed
            if value_1 != pred:
                for j in range(d):
                    proj[j] = X[i,j]

                if proj[feature] > threshold:
                    proj[feature] = threshold

                curr_node = split
                node = &self.nodes[curr_node]
                while node.parent != _TREE_UNDEFINED:
                    node = &self.nodes[node.parent]
                    if node.left_child == curr_node:
                        if proj[node.feature] > node.threshold:
                            proj[node.feature] = node.threshold
                    elif proj[node.feature] <= node.threshold:
                        proj[node.feature] = node.threshold
                    curr_node = node.parent

                curr_dist = 0.0
                for j in range(d):
                    curr_dist += pow(proj[j] - X[i,j],2)
                curr_dist = sqrt(curr_dist)

                if curr_dist < dist:
                    dist = curr_dist

            # Go through the right new leaf if needed
            if value_2 != pred:
                for j in range(d):
                    proj[j] = X[i,j]

                if proj[feature] <= threshold:
                    proj[feature] = threshold

                curr_node = split
                node = &self.nodes[curr_node]
                while node.parent != _TREE_UNDEFINED:
                    node = &self.nodes[node.parent]
                    if node.left_child == curr_node:
                        if proj[node.feature] > node.threshold:
                            proj[node.feature] = node.threshold
                    elif proj[node.feature] <= node.threshold:
                        proj[node.feature] = node.threshold
                    curr_node = node.parent
            
                curr_dist = 0.0
                for j in range(d):
                    curr_dist += pow(proj[j] - X[i,j],2)
                curr_dist = sqrt(curr_dist)

                if curr_dist < dist:
                    dist = curr_dist

            if pred == 0:
                dist = -1.0 * dist
            
            DI += (X[i,self.axis] - z_bar) * dist

        return abs(DI * pow(n_data, -1.))


    cdef double get_DI_corr(self, double[:,:] X, list leaves, SIZE_t split, double threshold, 
                            SIZE_t feature, SIZE_t value_1, SIZE_t value_2):

        cdef:
            SIZE_t n_data = X.shape[0]
            SIZE_t d = X.shape[1]
            SIZE_t node_id, curr_node
            SIZE_t pred
            int i, j
            double z_bar = 0.
            double cov_z = 0.
            double d_bar = 0.
            double cov_d = 0.
            double DI = 0.
            double curr_dist, dist
            double[:] all_dist = np.empty((n_data,), dtype=np.double)
            double[:] proj = np.empty((d,), dtype=np.double)

        for i in range(n_data):

            # Predict the label of X[i,:]
            node_id = 0
            node = &self.nodes[node_id]
            while node.feature != -2:
                if node_id != split:
                    if X[i,node.feature] <= node.threshold:
                        node_id = node.left_child
                        node = &self.nodes[node_id]
                    else:
                        node_id = node.right_child
                        node = &self.nodes[node.right_child]
                else:
                    if X[i,feature] <= threshold:
                        pred = value_1
                    else:
                        pred = value_2
                    break

            if node.feature == -2:
                pred = node.value

            # Go through each of the leaf nodes, except the new ones
            dist = 1000000.
            for node_id in leaves:
                node = &self.nodes[node_id]
                if node.value == pred:
                    continue
                
                for j in range(d):
                    proj[j] = X[i,j]

                curr_node = node_id
                while node.parent != _TREE_UNDEFINED:
                    node = &self.nodes[node.parent]
                    if node.left_child == curr_node:
                        if proj[node.feature] > node.threshold:
                            proj[node.feature] = node.threshold
                    elif proj[node.feature] <= node.threshold:
                        proj[node.feature] = node.threshold
                    curr_node = node.parent

                curr_dist = 0.0
                for j in range(d):
                    curr_dist += pow(proj[j] - X[i,j],2)
                curr_dist = sqrt(curr_dist)

                if curr_dist < dist:
                    dist = curr_dist

            # Go through the left new leaf if needed
            if value_1 != pred:
                for j in range(d):
                    proj[j] = X[i,j]

                if proj[feature] > threshold:
                    proj[feature] = threshold

                curr_node = split
                node = &self.nodes[curr_node]
                while node.parent != _TREE_UNDEFINED:
                    node = &self.nodes[node.parent]
                    if node.left_child == curr_node:
                        if proj[node.feature] > node.threshold:
                            proj[node.feature] = node.threshold
                    elif proj[node.feature] <= node.threshold:
                        proj[node.feature] = node.threshold
                    curr_node = node.parent

                curr_dist = 0.0
                for j in range(d):
                    curr_dist += pow(proj[j] - X[i,j],2)
                curr_dist = sqrt(curr_dist)

                if curr_dist < dist:
                    dist = curr_dist

            # Go through the right new leaf if needed
            if value_2 != pred:
                for j in range(d):
                    proj[j] = X[i,j]

                if proj[feature] <= threshold:
                    proj[feature] = threshold

                curr_node = split
                node = &self.nodes[curr_node]
                while node.parent != _TREE_UNDEFINED:
                    node = &self.nodes[node.parent]
                    if node.left_child == curr_node:
                        if proj[node.feature] > node.threshold:
                            proj[node.feature] = node.threshold
                    elif proj[node.feature] <= node.threshold:
                        proj[node.feature] = node.threshold
                    curr_node = node.parent
            
                curr_dist = 0.0
                for j in range(d):
                    curr_dist += pow(proj[j] - X[i,j],2)
                curr_dist = sqrt(curr_dist)

                if curr_dist < dist:
                    dist = curr_dist

            if pred == 0:
                dist = -1.0 * dist

            all_dist[i] = dist
            
        # Computing mean values
        for i in range(n_data):
            z_bar += X[i,self.axis]
            d_bar += all_dist[i]
        z_bar = z_bar/n_data
        d_bar = d_bar/n_data

        # Computing standard deviation
        for i in range(n_data):
            cov_z += pow(abs(X[i,self.axis] - z_bar),2)
            cov_d += pow(abs(all_dist[i] - d_bar),2)
        cov_z = sqrt(cov_z/n_data)
        cov_d = sqrt(cov_d/n_data)

        for i in range(n_data):
            DI += (X[i,self.axis] - z_bar) * all_dist[i]

        return abs(DI * pow(n_data*cov_d*cov_z, -1.))


    cdef np.ndarray _get_node_ndarray(self):
        """
        Wraps nodes as a NumPy struct array.
        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """

        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        return arr