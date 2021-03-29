import sys
import pstats, cProfile
import time
import numpy as np
import _CART
sys.path.append("D:\Github\MLTeam_FairBoundaryTree\CART")
from cart_tree import DecisionTree

np.random.seed(42)
X = np.random.rand(1000,5) ; y = np.random.randint(low=2, size=(1000,), dtype=np.intp)

fast_tree = _CART.Tree(min_samples=1, max_leaves=5, _lambda=0.0, impurity_type='entropy')
"""
cProfile.runctx("fast_tree.fit(X,y)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

"""
start = time.perf_counter()
fast_tree.fit(X,y)
print('fast tree building:', time.perf_counter() - start, 'sec')
start = time.perf_counter()
pred = fast_tree.predict(X)
print('fast tree predict:', time.perf_counter() - start, 'sec')
fast_tree.print_struct()
print(np.asarray(pred)[0:100])

tree = DecisionTree(min_samples=1, impurity_type='entropy', _lambda=0.0, max_leaves=5)
start = time.perf_counter()
tree.fit(X,y)
print('classic tree building:', time.perf_counter() - start, 'sec')
start = time.perf_counter()
pred = tree.predict(X)
print('classic tree predict:', time.perf_counter() - start, 'sec')
print(pred[0:100])
tree_graph = tree.builTree()
tree_graph.render("geraldin_tree")
