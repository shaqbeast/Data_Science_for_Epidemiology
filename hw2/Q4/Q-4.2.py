import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import steiner_tree

K1 = [10, 14, 3]
K2 = [10, 14, 3, 4, 2]
K3 = [10, 14, 3, 4, 2, 31, 49]
K4 = [10, 14, 3, 4, 2, 31, 49, 21, 25, 36, 43]

G = nx.random_graphs.extended_barabasi_albert_graph(50, 1, 0.2, 0.1, seed=10)

# create steiner trees
steiner_tree1 = steiner_tree(G, K1, weight='weight')
steiner_tree2 = steiner_tree(G, K2, weight='weight')
steiner_tree3 = steiner_tree(G, K3, weight='weight')
steiner_tree4 = steiner_tree(G, K4, weight='weight')

# printing each steiner tree
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=steiner_tree1.edges(), edge_color='red', width=2)
plt.show()

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=steiner_tree2.edges(), edge_color='red', width=2)
plt.show()

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=steiner_tree3.edges(), edge_color='red', width=2)
plt.show()

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=steiner_tree4.edges(), edge_color='red', width=2)
plt.show()

# putting adjacency list 
n = max(steiner_tree1.nodes) + 1
adj_matrix_1 = np.zeros((n, n), dtype=int)
for u, v in steiner_tree1.edges():
    adj_matrix_1[u, v] = 1
    adj_matrix_1[v, u] = 1
np.save('steiner_tree_1_adjacency.npy', adj_matrix_1)
loaded_matrix = np.load('steiner_tree_1_adjacency.npy')
print("Matrix 1: ")
print(loaded_matrix)
print(" ")

n2 = max(steiner_tree2.nodes) + 1
adj_matrix_2 = np.zeros((n2, n2), dtype=int)
for u, v in steiner_tree2.edges():
    adj_matrix_2[u, v] = 1
    adj_matrix_2[v, u] = 1
np.save('steiner_tree_2_adjacency.npy', adj_matrix_2)
loaded_matrix_2 = np.load('steiner_tree_2_adjacency.npy')
print("Matrix 2: ")
print(loaded_matrix_2)
print(" ")

n3 = max(steiner_tree3.nodes) + 1
adj_matrix_3 = np.zeros((n3, n3), dtype=int)
for u, v in steiner_tree3.edges():
    adj_matrix_3[u, v] = 1
    adj_matrix_3[v, u] = 1
np.save('steiner_tree_3_adjacency.npy', adj_matrix_3)
loaded_matrix_3 = np.load('steiner_tree_3_adjacency.npy')
print("Matrix 3: ")
print(loaded_matrix_3)
print(" ")

n4 = max(steiner_tree4.nodes) + 1
adj_matrix_4 = np.zeros((n4, n4), dtype=int)
for u, v in steiner_tree4.edges():
    adj_matrix_4[u, v] = 1
    adj_matrix_4[v, u] = 1
np.save('steiner_tree_4_adjacency.npy', adj_matrix_4)
loaded_matrix_4 = np.load('steiner_tree_4_adjacency.npy')
print("Matrix 4: ")
print(loaded_matrix_4)
print(" ")
