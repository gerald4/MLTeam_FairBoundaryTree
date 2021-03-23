import numpy as np
import _CART

tree = _CART.Tree(min_samples=1, max_leaves=10, _lambda=0.5, impurity_type='entropy')

X = np.random.rand(1000,5) ; y = np.random.randint(low=2, size=(1000,), dtype=np.intp)

print(np.unique(y, return_counts=True))

tree.fit(X,y)
tree.print_struct()