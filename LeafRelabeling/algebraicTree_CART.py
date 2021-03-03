"""
This code implements the computation of the distance to the decision boundary induced
by a decision tree trained using the code of Géraldin Nanfack (CART).

This distance is introduced in :
    Alvarez Isabelle, Stephan Bernard, and Guillaume Deffuant. "Keep the Decision Tree
    and Estimate the Class Probabilities Using its Decision Boundary." IJCAI. 2007.
    
Last modification : 03 March 2021
           Author : Delchevalerie Valentin
            email : valentin.delchevalerie@unamur.be
"""


import numpy as np


class algebraicTree_CART:
    
    
    def __init__(self, tree, path_to_leaves=None, label_of_leaves=None):
        """
        tree : Trained decision tree provided by the code of Géraldin Nanfack (CART).
        
        <optional>
        paths_to_leaves : List of paths to each leaf node of the tree, if already computed
        label_of_leaves : List of label of each leaf node, if already computed
        """
        
        self.tree = tree
        
        if (path_to_leaves == None) or (label_of_leaves == None):
            self.path_to_leaves, self.label_of_leaves = self.getPathToLeaves()
        else:
            self.path_to_leaves = path_to_leaves
            self.label_of_leaves = label_of_leaves
            
        self.n_leaves = len(self.label_of_leaves)
        
    
    def getPathToLeaves(self):
        """
        If not already computed, compute paths_to_leaves and label_of_leaves
        based on the trained decision tree
        """
        
        path_to_leaves = []
        label_of_leaves = []
        for index in list(dt.nodes.keys()):
            if dt.nodes[index].feature == -2:
                path = [index]
                while dt.nodes[path[-1]].index_parent != -1:
                    path.append(dt.nodes[path[-1]].index_parent)
                path.reverse()
                path_to_leaves.append(path)
                label_of_leaves.append(np.argmax(dt.nodes[path[-1]].stats))
            
        return path_to_leaves, label_of_leaves
    
    
    def getDistance(self, x):
        """
        Compute the signed distance to the decision boundary of sample x.
        
        x : sample to compute the distance from the decision boundary
        """
        
        c_x = self.tree.predict(x.reshape(1, -1))[0]
        
        # Go to leaves such that their label != c_x
        dist = np.inf
        for k in [i for i in range(self.n_leaves) if self.label_of_leaves[i] != c_x]:
            # Compute the projection on the leaf k
            proj = x[:].copy()
            for i in range(len(self.path_to_leaves[k][:-1])):
                id = self.path_to_leaves[k][:-1][i]
                next_id = self.path_to_leaves[k][i+1]
                
                feature = self.tree.nodes[id].feature
                threshold = self.tree.nodes[id].threshold

                if self.tree.nodes[id].left_node == next_id:
                    if proj[feature] > threshold:
                        proj[feature] = threshold
                elif proj[feature] <= threshold:
                    proj[feature] = threshold
            
            # Only keep the smallest
            curr_d = np.sqrt(np.sum((proj - x)**2))
            if curr_d < dist:
                dist = curr_d
            
        return dist
    
    
    def signedDistance(self, x):
        """
        Compute the signed distance to the decision boundary of sample x
        If the label guess by the tree (c_x) is 1, then signedDistance > 0.
        Otherwise, the distance is negative.
        
        x : sample to compute the distance from the decision boundary
        """
        
        c_x = self.tree.predict(x.reshape(1, -1))[0]
        dist = self.getDistance(x)
                
        if c_x == 0 or c_x == -1:
            dist = -1.0 * np.abs(dist)
        else:
            dist = np.abs(dist)
            
        return dist
    
    
    def misclassifiedDistance(self, x, y):
        """
        Compute the misclassified distance to the decision boundary of sample x
        If the true label of x (y) is the same than the one guess
        by the decision tree (c_x), the distance is 0.
        Otherwise, the distance is negative.
        
        x : sample to compute the distance from the decision boundary
        y : true label of x
        """
        
        c_x = self.tree.predict(x.reshape(1, -1))[0]
        dist = self.getDistance(x)
                
        if c_x == y:
            dist = 0.0
        else:
            dist = -1.0 * np.abs(dist)
            
        return dist
    
    
    def trustfulnessDistance(self, x, y):
        """
        Compute the distance to the decision boundary of sample x
        If the true label of x (y) is the same than the one guess
        by the decision tree (c_x), the distance is negative.
        Otherwise, it is positive.
        
        x : sample to compute the distance from the decision boundary
        y : true label of x
        """
        
        c_x = self.tree.predict(x.reshape(1, -1))[0]
        dist = self.getDistance(x)
                
        if c_x == y:
            dist = -1.0 * dist
            
        return dist