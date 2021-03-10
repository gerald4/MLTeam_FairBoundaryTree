
"""
gnanfack edit : 1 March 2021
"""
#This code is inspired from https://github.com/lucksd356/DecisionTrees/blob/master/dtree.py
# and from sklearn tree implementation

from copy import deepcopy

from collections import defaultdict
import pydotplus
import graphviz
import  numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import col1


FEATURE_THRESHOLD = 1e-7

class Node:
	""""This is a node class"""
	def __init__(self, index, index_parent, left_node, right_node, samples_index, y_unique, stats, feature = -2, threshold = - np.inf):
		self.index = index
		self.index_parent = index_parent
		self.left_node = left_node
		self.right_node = right_node
		self.feature = feature
		self.threshold = threshold
		self.stats = stats
		self.y_unique = y_unique
		self.samples_index = samples_index


	def update_split(self, feature, threshold, gain):
		self.feature = feature
		self.threshold = threshold
		self.gain = gain


def impurity(impurity_type):

	return {"entropy": entropy, "gini": gini}[impurity_type]

def entropy(freq):
	probs=freq/np.sum(freq)

	return np.sum(-probs*np.log2(probs))

def gini(freq):

	probs = freq/np.sum(freq)

	return np.sum(1-probs**2)



class DecisionTree:
	""" This a binary decision tree built from BestFirstSearch"""
	def __init__(self, min_samples, impurity_type, _lambda, max_leaves = 1000):
		self.min_samples = min_samples
		self.impurity_func = impurity(impurity_type)
		self._lambda = _lambda
		self.nodes = {}
		self.unvisited = {}
		self.stack_nodes = []
		self.counter = 0
		self.max_leaves = max_leaves

	def unfairness_term(self, X, y,):

		return 0.0

	def fit(self, X, y):
		y_unique, stats = np.unique(y, return_counts=True)
		self.y_unique = y_unique

		self.M = X.shape[1]
		#Tree initialisation
		root_node = Node(self.counter, -1, self.counter + 1, self.counter + 2, np.arange(X.shape[0]), y_unique, stats)
		self.counter += 3


		self.nodes[root_node.index] = root_node
		self.stack_nodes.append(root_node)
		self.unvisited[root_node.index] = root_node


		
		drap = True
		#BestFirstSearch
		while len(self.stack_nodes)!=0 and len(self.nodes)  < 2 * self.max_leaves - 1 and drap: 
			#print(len(self.nodes), len(self.unvisited), 2 * self.max_leaves - 1)
			#get the current node
			for node in self.unvisited.values(): 
				#Loop to search for feature
				best_feature = 0
				best_threshold = 0
				best_gain = 0

				current_score = self.impurity_func(node.stats)
				# if the current node is pure, it is a leaf node, continue
				if current_score == 0.0:
					node.update_split(feature = -2, threshold = -np.inf, gain = 0)

				else:
					
					for feat in range(X.shape[1]):

						X_feat = np.sort(X[node.samples_index,feat])

						end = X_feat.shape[0]

						if X_feat[-1] <= X_feat[0] + FEATURE_THRESHOLD:
							pass
						

						else:
							#Loop to search for threshold
							p = 0
							while p < end:
								while (p + 1 < end) and (X_feat[p + 1] <= X_feat[p] + FEATURE_THRESHOLD):
									p += 1

								p += 1
								
								if p < end:
									if (p < self.min_samples) or (end - p < self.min_samples):
										continue

									set1 = np.where(X[node.samples_index,feat]<= X_feat[p - 1])[0]
									set2 = np.where(X[node.samples_index,feat] > X_feat[p - 1])[0]

									#Copy the tree to compute unfairness
									tree_copy = deepcopy(self)
									tree_copy.stack_nodes.pop()
									#create virtual nodes
									tree_copy.nodes[node.index].feature = feat
									tree_copy.nodes[node.index].threshold = (X_feat[p-1] + X_feat[p])/2.0

									node1_y_unique, node1_stats = np.unique(y[node.samples_index[set1]], return_counts=True)
									node2_y_unique, node2_stats = np.unique(y[node.samples_index[set2]], return_counts=True)

									node1 = Node(node.left_node, node.index, tree_copy.counter + 1, tree_copy.counter + 2, 
													node.samples_index[set1], node1_y_unique, node1_stats)
									node2 = Node(node.right_node, node.index, tree_copy.counter + 3, tree_copy.counter + 4,
													node.samples_index[set2], node2_y_unique, node2_stats)

									tree_copy.nodes[node.left_node] = node1
									tree_copy.nodes[node.right_node] = node2

									unfairness = tree_copy.unfairness_term(X, y)

									del node1
									del node2
									del tree_copy

									weight = set1.shape[0]/(set1.shape[0] + set2.shape[0])

									gain = current_score - weight * self.impurity_func(node1_stats) - (1-weight)*self.impurity_func(node2_stats) - self._lambda * unfairness						
										
									if gain > best_gain:
										
										best_gain = gain
										best_feature = feat
										best_threshold = (X_feat[p-1] + X_feat[p])/2.0
										
											
				#if the best gain is greater than 0, we can split!
				if best_gain > 0:
					node.update_split(best_feature, best_threshold, best_gain)
				else:
					node.update_split(feature = -2, threshold = -np.inf, gain = 0)
					self.stack_nodes.pop(self.stack_nodes.index(node))

			
			# Now look for the best in the stack_nodes list to split
			best_i = 0
			best_gain_for_nodes = 0
			for i in range(len(self.stack_nodes)):
				if self.stack_nodes[i].gain > best_gain_for_nodes:
					best_i = i
					best_gain_for_nodes = self.stack_nodes[i].gain

			if best_gain_for_nodes > 0:
				#split the node and create child nodes
				node = self.stack_nodes.pop(best_i)
				self.unvisited = {}
				set1 = np.where(X[node.samples_index, node.feature]<= node.threshold )[0]
				set2 = np.where(X[node.samples_index, node.feature] > node.threshold )[0]

				node1_y_unique, node1_stats = np.unique(y[node.samples_index[set1]], return_counts=True)
				node2_y_unique, node2_stats = np.unique(y[node.samples_index[set2]], return_counts=True)

				#This place needs to be optimised: gnanfack edit, 10/03/2021
				# node1_y_unique = list(node1_y_unique)
				# node2_y_unique = list(node2_y_unique)

				# node1_stats = np.zeros(y_unique.shape[0])   
				# node2_stats = np.zeros(y_unique.shape[0])
			
				# for p in range(y_unique.shape[0]):
				# 	if y_unique[p] in node1_y_unique:
				# 		node1_stats[p] = node1_stats_[node1_y_unique.index(y_unique[p])]

				# 	if y_unique[p] in node2_y_unique:
				# 		node2_stats[p] = node2_stats_[node1_y_unique.index(y_unique[p])]


				# #end edit


				node1 = Node(node.left_node, node.index, self.counter, self.counter + 1, 
													node.samples_index[set1], node1_y_unique, node1_stats)
				node2 = Node(node.right_node, node.index, self.counter + 2, self.counter + 3,
													node.samples_index[set2], node2_y_unique, node2_stats)

				self.counter += 4

				self.nodes[node.left_node] = node1
				self.nodes[node.right_node] = node2

				if self.impurity_func(node1_stats) != 0:
					self.unvisited[node1.index] = node1
					self.stack_nodes.append(node1)

				if self.impurity_func(node2_stats) !=0:
					self.unvisited[node2.index] = node2
					self.stack_nodes.append(node2)
			else:
				drap = False

	def builTree(self, colnames = None, classnames = None):
		y_unique = self.nodes[0].y_unique
		
		if not(colnames):
			colnames = [f'X[{i}]' for i in range(self.M)]
		if not(classnames):
			classnames = [f"class{i}" for i in y_unique]
		
		Tree = graphviz.Digraph(format='png',graph_attr={"randir":"LR"},
									node_attr={'shape':"box"})
		
		for node in self.nodes.values():
			#print(node.index, node.y_unique, node.stats)
			stats_update = np.zeros(y_unique.shape[0])
			stats_update[node.y_unique] = node.stats
			impurity = np.round(self.impurity_func(node.stats))
			n_samples = len(node.samples_index)
			if node.feature != -2:
				Tree.node(str(node.index), 
						 label = f"{colnames[node.feature]} <= {node.threshold}\nimpurity={impurity}\nsamples={n_samples}\nvalues={list(stats_update)}",
						 fillcolor=col1.get_color(stats_update,len(y_unique)), style="rounded,filled")
			else:
				Tree.node(str(node.index), 
						 label = f"impurity={impurity}\nsamples={n_samples}\nvalues={list(stats_update)}",
						 fillcolor=col1.get_color(stats_update,len(y_unique)), style="rounded,filled")
			if node.index!= 0:
				if self.nodes[node.index_parent].left_node == node.index:
					Tree.edge(str(node.index_parent),str(node.index))#,label='\nTrue')
				else:
					Tree.edge(str(node.index_parent),str(node.index))#,label='\nFalse')
		return Tree
	
	def predict(self, X):

		def map_node_y(n_unique):
			return [list(self.y_unique).index(y) for y in n_unique]


		def predict_univalue(x):
			drap = False
			node = self.nodes[0]
			labels = np.zeros(self.y_unique.shape[0])
			while not(drap):
				if node.feature!= -2:
					if x[node.feature] <= node.threshold:
						node = self.nodes[node.left_node]
					else:
						node = self.nodes[node.right_node]
				else:
					drap = True
					labels[map_node_y(node.y_unique)] = node.stats
					return self.y_unique[np.argmax(labels)]
				
		y_predict = np.zeros(X.shape[0], dtype=self.nodes[0].y_unique.dtype)
		
		for i in range(X.shape[0]):
			y_predict[i] = predict_univalue(X[i])
		
		return y_predict
					

if __name__ == "__main__":

	iris = datasets.load_iris()
	
	X = iris.data  
	y = iris.target - 2 

	dt = DecisionTree(min_samples = 1, impurity_type = "gini", _lambda = 2., max_leaves = 15)


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 111)

	dt.fit(X_train, y_train)
	print(dt.predict(X_test))
	print("Predictive accuracy: ", accuracy_score(dt.predict(X_test), y_test))

	tree_graph = dt.builTree()
	tree_graph.render("tree_iris")








