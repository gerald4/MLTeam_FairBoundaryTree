"""
This code aims to perform leaf relabeling in order to reduce
the unfairness in trees provided by sklearn.

Metrics used to determine unfairness are introduced in :
    Zafar, Muhammad Bilal, et al. "Fairness Constraints: A Flexible Approach for Fair Classification."
    Journal of Machine Learning Research 20.75 (2019): 1-42.
    
Last modification : 16 November 2020
           Author : Delchevalerie Valentin
            email : valentin.delchevalerie@unamur.be
"""


# Librairies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib import rcParams
from matplotlib import gridspec
params = {'axes.labelsize': 28,
          'axes.grid': False,
          'axes.linewidth': 1.6,
          'axes.titlepad': 20,
          'axes.xmargin': 0.05,
          'axes.ymargin': 0.05,
          'grid.alpha': 0.4,
          'grid.color': '#666666',
          'grid.linestyle': '-.',
          'legend.fontsize': 42,
          'legend.loc': 'lower right',
          'xtick.labelsize': 28,
          'xtick.major.width': 1.6,
          'xtick.major.size': 10,
          'xtick.minor.width': 1.0,
          'xtick.minor.size': 4,
          'ytick.labelsize': 28,
          'ytick.major.width': 1.6,
          'ytick.major.size': 10,
          'ytick.minor.width': 1.0,
          'ytick.minor.size': 4,
          'text.usetex': True,
          'figure.figsize': [12, 12],
          'font.size': 42.0, 
          'lines.markersize': np.sqrt(20) * 2.5,
          'figure.autolayout': False,
          }
rcParams.update(params)
from algebraicTree import algebraicTree
from fairnessMetrics import getDisparateImpact, getDisparateMistreatment_overall, getDisparateMistreatment_fpfn


class leafRelabeling:
    
    
    def __init__(self, tree, X_train, y_train, X_test, y_test, c=None, 
                 controled_axis=-1, constraint_type='DI', verbose=1):
        """
        Blablabla
        """
        
        self.myAlgebraicTree = algebraicTree(tree)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.c = c
        self.controled_axis = -1
        self.constraint_type = constraint_type
        self.verbose = verbose
        
        self.ini_accu = self.myAlgebraicTree.tree.score(self.X_test, self.y_test)
        self.ini_fairness = self.getFairness()
        if self.c == None:
            self.c = self.ini_fairness * 0.75
            if self.verbose > 0:
                print('\nConstraint automatically set to actual fairness * 0.75\n') 
        
    
    def getFairness(self):
        """
        Blablabla
        """
        
        if self.constraint_type not in ['DI', 'DM_overall', 'DM_fpfn']:
            self.constraint_type = 'DI'
            if self.verbose > 0:
                print('Constraint type', self.constraint_type, 'not available...')
                print('Constraint type set to DI (Disparate Impact)')
        
        if self.constraint_type == 'DI':
            fairness = getDisparateImpact(self.X_test, self.myAlgebraicTree, 
                                          controled_axis=self.controled_axis)
        elif self.constraint_type == 'DM_overall':
            fairness = getDisparateMistreatment_overall(self.X_test, self.y_train, self.myAlgebraicTree, 
                                                        controled_axis=self.controled_axis)
        else:
            fairness = getDisparateMistreatment_fpfn(self.X_test, self.y_train, self.myAlgebraicTree, 
                                                     controled_axis=self.controled_axis)
        
        return fairness
        
    
    def fit_transform(self):
        """
        Blablabla
        """
        
        if self.ini_fairness <= self.c:
            if self.verbose > 0:
                print('Constraint already satisfied')
                print('\tc=', self.c, '\n\tfairness=', self.ini_fairness)
            return self.myAlgebraicTree.tree, self.ini_accu, self.ini_fairness
        else:
            if self.verbose > 0:
                print('\t# Actual fairness:', self.ini_fairness, '\n\t# Objective:', self.c, '\n\t# Actual accuracy:', self.ini_accu, '\n')
            
        if self.verbose > 1:
            print('\n>> Starting leaf relabeling...\n')
            
        # Actual accuracy and fairness
        fairness = self.ini_fairness
        accu = self.ini_accu
        
        # Get all the leaves in the tree
        leaves_id = [x[-1] for x in self.myAlgebraicTree.paths_to_leaves]
        
        # Main loop
        last_id = None
        last_last_id = None
        while fairness > self.c:
            new_accu = []
            new_fairness = []
            ratio = []
            
            # Test each possible leaf relabeling in order to keep the best one
            for id in range(len(leaves_id)):
                # Relabeling
                self.myAlgebraicTree.label_of_leaves[id] = int(np.abs(self.myAlgebraicTree.label_of_leaves[id] - 1))
                self.myAlgebraicTree.tree.tree_.value[leaves_id[id]][0] = np.flip(self.myAlgebraicTree.tree.tree_.value[leaves_id[id]][0])
                if self.verbose > 1:
                    print('Considering relabeling leaf', id, 'to', self.myAlgebraicTree.label_of_leaves[id])
                
                # Evaluate the new tree
                new_accu.append(self.myAlgebraicTree.tree.score(self.X_test, self.y_test))
                new_fairness.append(self.getFairness())
                if new_fairness[-1] <= fairness:
                    if not np.isclose(accu, new_accu[-1]):
                        ratio.append((fairness - new_fairness[-1]) / (accu - new_accu[-1]))
                    else:
                        ratio.append((fairness - new_fairness[-1]) / 1e-3)
                else:
                    ratio.append(-np.inf)
                if self.verbose > 1:
                    print('\t * Accuracy:', accu, '->', new_accu[-1])
                    print('\t * Fairness:', fairness, '->', new_fairness[-1], '\n')
                
                # Restore original tree
                self.myAlgebraicTree.label_of_leaves[id] = int(np.abs(self.myAlgebraicTree.label_of_leaves[id] - 1))
                self.myAlgebraicTree.tree.tree_.value[leaves_id[id]][0] = np.flip(self.myAlgebraicTree.tree.tree_.value[leaves_id[id]][0])
                
            # Keep the best relabeling
            id = np.argmax(ratio)
            self.myAlgebraicTree.label_of_leaves[id] = int(np.abs(self.myAlgebraicTree.label_of_leaves[id] - 1))
            self.myAlgebraicTree.tree.tree_.value[leaves_id[id]][0] = np.flip(self.myAlgebraicTree.tree.tree_.value[leaves_id[id]][0])
            
            if id == last_id or last_last_id == id:
                if self.verbose > 0:
                    print('Impossible to obtain a fairness <', self.c)
                break
            
            last_last_id = last_id
            last_id = id
            accu = new_accu[id]
            fairness = new_fairness[id]
            if self.verbose > 1:
                print('\n\t>> Relabeling leaf', id, '\n\n')
                
        if self.verbose > 0:
            print('\t# Final fairness:', fairness, '\n\t# Objective:', self.c, '\n\t# Final accuracy:', accu, '\n')      
                
        return self.myAlgebraicTree.tree, accu, fairness