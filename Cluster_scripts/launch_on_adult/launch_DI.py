# Libraries
import sys
import json
import os
import numpy as np

# Other libraries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats

# User defined libraries
from _CART import Tree


def get_unfairness(X, y, axis):
    y1 = np.where(y == 1)[0]
    A = len(np.where(X[:,axis] == 0)[0])
    if A == 0:
        A = 1
    B = len(np.where(X[:,axis] == 1)[0])
    if B == 0:
        B = 1
    return np.abs(len(np.where(X[y1,axis] == 0)[0])/A - len(np.where(X[y1,axis] == 1)[0])/B)


# Random Forest 
class randomForest:
    
    def __init__(self, n_trees=50, n_samples=100, impurity='entropy', max_leaves=10, n_features=1, _lambda=0.5, axis=0, balanced=True):
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.impurity = impurity
        self.max_leaves = max_leaves
        self.n_features = n_features
        self._lambda = _lambda
        self.axis = axis
        self.balanced = balanced
        
        self.forest = []
        
    def fit(self, X, y):
        for i in range(self.n_trees):
            
            if not self.balanced:
                randomIndices = np.random.randint(low = 0, high = X.shape[0], size = self.n_samples)
            else:
                labels = np.unique(y)
                n_labels = labels.shape[0]
                batch_size = self.n_samples // n_labels
                
                randomIndices = np.empty((n_labels * batch_size,), dtype=np.int)
                for j, label in enumerate(labels):
                    idx = np.where(y == label)[0]
                    np.random.shuffle(idx)
                    randomIndices[j*batch_size:(j+1)*batch_size] = idx[0:batch_size]
            
            randomFeatures = np.linspace(0, X.shape[1]-1, X.shape[1], dtype=np.int)
            np.random.shuffle(randomFeatures)
            randomFeatures = list(randomFeatures[0:self.n_features])
            
            tree = Tree(min_samples=1, impurity_type=self.impurity, _lambda=self._lambda, max_leaves=self.max_leaves, axis=self.axis)
            tree.fit(X[randomIndices,:], y[randomIndices], features=randomFeatures)
            
            self.forest.append(tree)
            
    def predict(self, X):
        votes = np.empty((self.n_trees, X.shape[0]), dtype=np.intp)
        
        for i in range(self.n_trees):
            votes[i,:] = np.array(self.forest[i].predict(X))
         
        return stats.mode(votes)[0][0]
    
    def get_DI(self, X):
        DI = np.zeros((self.n_trees,), dtype=np.double)
        
        for i in range(self.n_trees):
            DI[i] = self.forest[i].compute_DI_corr(X)
            
        return np.mean(DI)

    def get_RF_features_count(self):
        count = np.zeros((self.n_features,), dtype=np.double)
        for tree in self.forest:
            count += tree.get_DT_features_count(self.n_features)

        return count


idx = int(sys.argv[1])
config = [{'n_leaves':15, 'impurity':'entropy', 'n_trees':50, 'n_samples':5000}][idx-1]

n_leaves = config['n_leaves']
impurity = config['impurity']
n_trees = config['n_trees']
n_samples = config['n_samples']

n_iter = 10
_lambda = np.concatenate((np.linspace(0.,4.,30), np.linspace(4.,10.,11)[1:]))

path = os.path.join('./results_adult_DI', str(n_leaves) + '_' + impurity + '_' + str(n_trees))
if not os.path.exists(path):
    os.makedirs(path)

for i in range(10,31):

    X = np.load('X_adult.npy').astype(np.double)
    y = np.load('y_adult.npy').astype(np.intp)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=i)

    results = {'Tree':{'train_accuracy': [], 
                       'test_accuracy': [], 
                       'train_unfairness': [], 
                       'test_unfairness': [],
                       'train_DI': [],
                       'test_DI': [],
                       'count': [],
                       'structures': []},
               'RandomForest':{'train_accuracy': [], 
                               'test_accuracy': [], 
                               'train_unfairness': [], 
                               'test_unfairness': [],
                                'train_DI': [],
                                'test_DI': [],
                               'count': []},
               'lambda': []}

    for value in _lambda:
        results['lambda'].append(value)
        
        # Our tree
        tree = Tree(min_samples=1, impurity_type=impurity, _lambda=value, max_leaves=n_leaves, axis=0)
        tree.fit(X_train, y_train, features=list(range(X_train.shape[1])))
        y_train_pred = np.array(tree.predict(X_train))
        y_test_pred = np.array(tree.predict(X_test))
        results['Tree']['train_accuracy'].append(accuracy_score(y_train_pred, y_train))
        results['Tree']['train_unfairness'].append(get_unfairness(X_train, y_train_pred, 0))
        results['Tree']['train_DI'].append(tree.compute_DI_corr(X_train))
        results['Tree']['test_accuracy'].append(accuracy_score(y_test_pred, y_test))
        results['Tree']['test_unfairness'].append(get_unfairness(X_test, y_test_pred, 0))
        results['Tree']['test_DI'].append(tree.compute_DI_corr(X_test))
        results['Tree']['count'].append(list(tree.get_DT_features_count(X_train.shape[1])))
        results['Tree']['structures'].append(tree.print_struct())

        # Our randomForest
        rf = randomForest(n_trees=n_trees, n_samples=n_samples, impurity=impurity, max_leaves=n_leaves, 
                          n_features=X_train.shape[1], _lambda=value, axis=0, balanced=False)
        rf.fit(X_train, y_train)
        y_train_pred = np.array(rf.predict(X_train))
        y_test_pred = np.array(rf.predict(X_test))
        results['RandomForest']['train_accuracy'].append(accuracy_score(y_train_pred, y_train))
        results['RandomForest']['train_unfairness'].append(get_unfairness(X_train, y_train_pred, 0))
        results['RandomForest']['train_DI'].append(rf.get_DI(X_train))
        results['RandomForest']['test_accuracy'].append(accuracy_score(y_test_pred, y_test))
        results['RandomForest']['test_unfairness'].append(get_unfairness(X_test, y_test_pred, 0))
        results['RandomForest']['test_DI'].append(rf.get_DI(X_test))
        results['RandomForest']['count'].append(list(rf.get_RF_features_count()))

    with open(os.path.join(path, 'results_' + str(i) + '.json'), 'w') as fp:
        json.dump(results, fp, indent=4)
